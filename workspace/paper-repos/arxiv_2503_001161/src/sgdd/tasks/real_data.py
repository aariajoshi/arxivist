"""Real-data task stubs (DNA enhancers / MNIST XOR-AND / music infilling).

These tasks reuse the SAME SGDD sampler (`sgdd.samplers.split_gibbs.sgdd`); only the
prior model and the likelihood f(z;y) change:

  * DNA enhancers (reward guidance, Sec 4.3):
        prior  = SEDD-small (~90M) trained on Gosai et al. ~700k MPRA DNA seqs, N=4, D=200
        f(z;y) = -beta * r(z), r = HepG2 activity reward oracle
        Table 7: K=50, MH_T=200, H=20, beta in {30,50}

  * MNIST XOR/AND (inverse problem, Sec 4.4):
        prior  = SEDD binary-MNIST, N=2, D=1024
        f(z;y) = ||G(z) - y|| / sigma_y, G = AND/XOR over random position pairs
        Table 7: K in {50,100}, MH_T in {2000,5000}, H=20

  * Monophonic music infilling (inverse problem, Sec 4.5):
        prior  = SEDD music, N=129, D=256
        f(z;y) = infill consistency on observed positions
        Table 7: K=100, MH_T=5000, H=20

They are GATED on the trained SEDD priors + task oracles, available from the official
repo (github.com/chuwd19/Split-Gibbs-Discrete-Diffusion-Posterior-Sampling). This module
documents the wiring so the sampler can be pointed at those weights without changes.
"""
from __future__ import annotations


def dna_reward_nll(reward_oracle, beta: float):
    """Return f(z;y) = -beta * r(z) for reward-guided DNA generation (Sec 4.3).

    `reward_oracle` is the trained HepG2 activity predictor (weight-gated).
    """
    def nll(z):
        return -beta * float(reward_oracle(z))
    return nll


def inverse_problem_nll(forward_G, y, sigma_y: float):
    """Return f(z;y) = ||G(z) - y|| / sigma_y for MNIST/music inverse problems (Sec 4.1)."""
    import numpy as np

    def nll(z):
        return float(np.linalg.norm(forward_G(z) - y)) / sigma_y
    return nll


# NOTE: loading the SEDD prior would look like:
#   from sedd import load_pretrained            # official repo
#   model = load_pretrained("dna-enhancer")     # concrete score s_theta
#   prior = SEDDPriorAdapter(model)             # expose .posterior_x0 like ClosedFormUniformPrior
# then call sgdd(prior=prior, neg_log_likelihood=dna_reward_nll(oracle, beta), ...).
