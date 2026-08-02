"""Generate DNA via the full DiscDiff pipeline: latent DDPM sampling -> VAE decode ->
Absorb-Escape refinement (Sec 4.1-4.2, Fig 3).

IMPORTANT — untrained by default. With no --weights, the VAE/U-Net are RANDOMLY INITIALIZED,
so this exercises the whole generate->refine->evaluate PATH end-to-end but the sequences are
NOT paper-comparable (a random model cannot match S-FID 3.21). Pass trained weights (author
repo Zehui127/Latent-DNA-Diffusion) to get real DiscDiff samples. This is deliberately loud
about which regime you are in.

    python generate.py --config configs/config.yaml --n 100
"""
from __future__ import annotations

import argparse
import os
import sys

import torch
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from discdiff.models.ddpm import DDPM
from discdiff.models.unet import LatentUNet
from discdiff.models.vae import DiscDiffVAE
from discdiff.refine.absorb_escape import AbsorbEscapeConfig, absorb_escape

BASES = "ACGT"


class DecoderConfidenceAR:
    """A cheap autoregressive corrector for the demo pipeline: it re-samples a token from
    the VAE decoder's own per-position distribution with high confidence. Stands in for the
    fine-tuned Hyena (which needs weights). Model-agnostic interface expected by Absorb-Escape.
    """

    def __init__(self, probs):  # probs: [L, 4] decoder softmax
        self.probs = probs

    def generate(self, prefix, pos):
        p = self.probs[pos]
        tok = int(p.argmax())
        return tok, float(p[tok])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--weights", default=None, help="path to trained DiscDiff weights (optional)")
    ap.add_argument("--out", default="results/generated.fasta")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    L = cfg["data"]["seq_len"]
    v, d = cfg["vae"], cfg["denoiser"]
    ae = cfg["absorb_escape"]

    torch.manual_seed(0)
    vae = DiscDiffVAE(k_channels=v["k_channels"], c_channels=v["c_channels"], seq_len=L).eval()
    unet = LatentUNet(ch=d["ch"], emb_dim=d["emb_dim"], n_species=d["n_species"],
                      use_self_attention=d["use_self_attention"]).eval()
    ddpm = DDPM(n_steps=d["n_steps"])

    trained = False
    if args.weights and os.path.exists(args.weights):
        sd = torch.load(args.weights, map_location="cpu")
        vae.load_state_dict(sd["vae"], strict=False)
        unet.load_state_dict(sd["unet"], strict=False)
        trained = True

    print("=" * 72)
    if trained:
        print(f"[discdiff] loaded trained weights from {args.weights}")
    else:
        print("[discdiff] !! UNTRAINED (random init) — pipeline runs end-to-end but sequences")
        print("[discdiff] !! are NOT paper-comparable. Pass --weights for real DiscDiff samples.")
    print("=" * 72)

    # infer latent shape from a dry VAE encode
    with torch.no_grad():
        dummy = torch.randint(0, 4, (1, L))
        mu, _ = vae.encode(torch.nn.functional.one_hot(dummy, 4).permute(0, 2, 1).float())
    latent_shape = (args.n,) + tuple(mu.shape[1:])

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    print(f"[generate] n={args.n} L={L} | latent {latent_shape[1:]} | DDPM steps={d['n_steps']}")
    ae_cfg = AbsorbEscapeConfig(t_absorb=ae["t_absorb"], escape=ae["escape"],
                                t_escape=ae["t_escape"], t_random=ae["t_random"],
                                max_sub_length=ae["max_sub_length"])

    seqs = []
    with torch.no_grad():
        # sample all latents at once, decode, then refine per-sequence
        z0 = ddpm.sample(unet, latent_shape,
                         species=torch.zeros(args.n, dtype=torch.long))
        logits = vae.decode(z0)                          # [n, 4, L]
        probs = torch.softmax(logits, dim=1)             # [n, 4, L]
        conf, toks = probs.max(dim=1)                    # [n, L] each
        for i in range(args.n):
            raw = toks[i].tolist()
            l_d = conf[i].tolist()
            ar = DecoderConfidenceAR(probs[i].transpose(0, 1))  # [L,4]
            refined = absorb_escape(raw, l_d, ar, ae_cfg)
            seqs.append("".join(BASES[t] for t in refined))

    with open(args.out, "w") as f:
        for i, s in enumerate(seqs):
            f.write(f">gen_{i}\n{s}\n")
    print(f"[done] wrote {len(seqs)} sequences -> {args.out}")


if __name__ == "__main__":
    main()
