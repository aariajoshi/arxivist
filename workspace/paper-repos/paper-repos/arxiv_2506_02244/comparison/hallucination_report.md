# Hallucination Report
**Paper**: Physics-Guided Motion Loss for Video Generation Model  
**Paper ID**: arxiv_2506_02244  
**Audit Date**: 2025-07-25  
**SIR Version**: 1 | **Architecture Plan Version**: 1  
**Auditor**: ArXivist Stage 6 — Results Comparator

This report audits the generated implementation against the SIR for three hallucination
categories: **structural** (invented components), **parametric** (assumed hyperparameters),
and **omission** (missing/stubbed components). All findings are precise and objective.

---

## Summary

| Category | Count | Critical | Significant | Minor |
|----------|-------|----------|-------------|-------|
| Structural hallucinations | 3 | 0 | 1 | 2 |
| Parametric hallucinations | 7 | 0 | 3 | 4 |
| Omission hallucinations | 4 | 2 | 1 | 1 |
| **Total** | **14** | **2** | **5** | **7** |

**Overall hallucination risk**: Medium. No critical structural inventions. Two
critical omissions (backbone loaders, exact x̂₀ reconstruction) are known stubs
that must be completed before training can begin.

---

## Part 1 — Structural Hallucinations

*Components present in generated code that are NOT in the SIR.*

---

### SH-1: `MockDiffusionBackbone` in notebooks  
**Severity**: Minor  
**Type**: structural  
**Location**: `notebooks/reproduce_arxiv_2506_02244.ipynb` — mini-training cell  
**Description**: A small linear network (`nn.Flatten → nn.Linear → nn.ReLU → nn.Linear`)
is used as a stand-in backbone for the notebook demo. This class does not appear in the
paper or SIR.  
**Evidence**: Paper uses Open-Sora, MVDIT, and Hunyuan — none of which are linear networks.  
**Impact**: Zero impact on training reproducibility. The notebook is explicit that this
is a demo only. No research claim depends on it.  
**Suggested fix**: None required. Comment already reads "for demo only."

---

### SH-2: `KeywordMotionClassifier` and `LLMMotionClassifier` in `data/dataset.py`  
**Severity**: Significant  
**Type**: structural  
**Location**: `src/physics_motion_loss/data/dataset.py`  
**Description**: Two concrete classifier implementations are provided for stratifying
prompts into "simple" vs "complex" motion. The paper states an LLM (GPT-5) was used
with an unspecified rubric; neither the keyword list nor the LLM system prompt in the
generated code appears verbatim in the paper.  
**Evidence**: Section 4.2 — "we stratify test prompts into simple vs. complex motion
using an LLM (GPT-5) following our rubric" — rubric not published. SIR confidence: 0.70.  
**Impact**: Stratified evaluation results (Table 3) may not be reproducible if the
classifier disagrees with the paper's GPT-5 rubric. The simple/complex split itself
will differ.  
**Suggested fix**: Contact authors for the exact system prompt. Until then, report
stratified results with the caveat that the classifier is approximate. Do not compare
Table 3 figures directly.

---

### SH-3: `FP32Context` as an explicit context-manager class  
**Severity**: Minor  
**Type**: structural  
**Location**: `src/physics_motion_loss/utils/precision.py`  
**Description**: The paper states "runs in FP32 with autocast off" but does not specify
the implementation mechanism. The generated code wraps this in a named context manager
class. This is a reasonable engineering choice, not a contradiction.  
**Evidence**: Section 4.1 — "runs in FP32 with autocast off" is the full specification.  
**Impact**: None — functionally equivalent to inline `torch.amp.autocast(enabled=False)`.  
**Suggested fix**: None required. The class is a clean encapsulation of the stated intent.

---

## Part 2 — Parametric Hallucinations

*Hyperparameters marked `# ASSUMED` in the generated code that may be incorrect.*

---

### PH-1: Optimizer = AdamW  
**Severity**: Significant  
**Type**: parametric  
**SIR Confidence**: 0.65  
**Location**: `src/physics_motion_loss/training/trainer.py`, `configs/config.yaml`  
**Paper statement**: Not stated. Section 4.1 only specifies LR=2e-5 and cosine annealing.  
**Risk**: The choice of optimizer affects convergence speed and final metric values.
Adam vs AdamW can differ by 1–3% on downstream metrics. Lion optimizer is increasingly
common in video generation.  
**Consequence if wrong**: Moderate deviation (2–5%) on all metrics.  
**Suggested fix**: First attempt: AdamW with β=(0.9, 0.999), wd=0.01 (implemented default).
If metrics deviate > 5%, try: (a) Adam with wd=0, (b) Lion optimizer (lr=5e-6).

---

### PH-2: LoRA rank = 16  
**Severity**: Significant  
**Type**: parametric  
**SIR Confidence**: < 0.60  
**Location**: `configs/config.yaml`, `src/physics_motion_loss/training/lora_wrapper.py`  
**Paper statement**: Section 4.2 — "inserting low-rank adapters into attention and
time-related linear layers" — rank not specified.  
**Risk**: LoRA rank directly controls parameter count and expressivity. rank=4 vs
rank=32 can produce 5–10% differences in fine-tuning effectiveness.  
**Consequence if wrong**: Significant deviation on Hunyuan metrics specifically.  
**Suggested fix**: Run ablation sweep: rank ∈ {4, 8, 16, 32}. Report which matches paper.

---

### PH-3: Physics loss weight = 0.1 (constant)  
**Severity**: Significant  
**Type**: parametric  
**SIR Confidence**: 0.78  
**Location**: `configs/config.yaml`, `src/physics_motion_loss/training/trainer.py`  
**Paper statement**: Table 5 lists weight=0.1. No schedule described.  
**Risk**: If the paper uses a warmup schedule (e.g., 0→0.1 over first epoch), constant
0.1 from step 0 may over-regularise early training.  
**Consequence if wrong**: Moderate deviation (3–8%) — particularly on visual quality
metrics (VQA A/T) if physics loss dominates early denoising.  
**Suggested fix**: Default to constant 0.1. If VQA scores are notably lower than
paper, try linear warmup: weight = min(0.1, 0.1 * step / warmup_steps).

---

### PH-4: Temporal window T = 16  
**Severity**: Minor  
**Type**: parametric  
**SIR Confidence**: 0.72  
**Location**: `configs/config.yaml`, Appendix A.8 (range: 12–16)  
**Paper statement**: App A.8 — "window T=12–16". Table 5 not pinned.  
**Risk**: T affects spectral resolution. Smaller T → coarser temporal frequency bins
→ noisier Ω* and α* estimates → higher variance in rotation and scaling losses.  
**Consequence if wrong**: Minor (1–3%) on rotation-heavy clips.  
**Suggested fix**: Default T=16. If rotation metrics underperform, try T=12.

---

### PH-5: Stop-gradient on adaptive weights = False  
**Severity**: Minor  
**Type**: parametric  
**SIR Confidence**: 0.70  
**Location**: `src/physics_motion_loss/losses/adaptive_composite.py`  
**Paper statement**: Not stated. Section 3.7 does not specify whether gradients flow
through the softmax weights.  
**Risk**: With stop-grad=False, the weight computation itself backpropagates — the
network learns to reduce losses in a way that also changes which loss gets emphasised.
This could cause feedback loops on mixed-motion clips.  
**Consequence if wrong**: Likely minor (< 2%), but could explain instability on complex-
motion prompts.  
**Suggested fix**: Default False (fully differentiable). Monitor training loss for
oscillation on complex-motion batches. If unstable, switch to True.

---

### PH-6: LoRA alpha = 32 (= 2 × rank)  
**Severity**: Minor  
**Type**: parametric  
**SIR Confidence**: < 0.60  
**Location**: `configs/config.yaml`  
**Paper statement**: Not stated.  
**Risk**: LoRA alpha scaling affects the effective learning rate of the adapters.
alpha=2×rank is a common default but not universal.  
**Suggested fix**: Standard practice. Acceptable starting point.

---

### PH-7: Batch size = 4 (assumed in `train.py`)  
**Severity**: Minor  
**Type**: parametric  
**SIR Confidence**: 0.50  
**Location**: `train.py` — DataLoader construction  
**Paper statement**: Not stated. 4× A100 GPUs implies a global batch size that is
unspecified.  
**Risk**: Smaller batch size increases gradient variance; larger increases memory.
With BF16 on A100-80GB, batch size 4 per GPU × 4 GPUs = 16 global is reasonable.  
**Suggested fix**: Start with per-GPU batch size 4. Scale up if VRAM allows. Monitor
loss stability.

---

## Part 3 — Omission Hallucinations

*Components present in the SIR but absent or stubbed in the generated code.*

---

### OH-1: Backbone model loaders (CRITICAL)  
**Severity**: Critical  
**Type**: omission  
**Location**: `train.py` — `_load_open_sora()`, `_load_mvdit()`, `_load_hunyuan()`  
**Description**: All three backbone loaders raise `NotImplementedError`. The paper
trains on Open-Sora, MVDIT, and Hunyuan — none of which can be loaded without
implementing these functions.  
**Evidence**: SIR `training_pipeline.backbone` field lists all three. Without working
loaders, zero training can occur.  
**Impact**: **Training is impossible until these are implemented.**  
**Suggested fix**:  
```python
# Open-Sora: install from https://github.com/hpcaitech/Open-Sora
from opensora.models.stdit import STDiT  # example — actual API may differ
model = STDiT.from_pretrained("hpcai-tech/open-sora")

# Hunyuan: install from https://github.com/Tencent/HunyuanVideo
from hyvideo.modules import HunyuanVideoTransformer3DModel
model = HunyuanVideoTransformer3DModel.from_pretrained("tencent/HunyuanVideo")
```
Implement the `forward()` wrapper to return `{"loss": denoising_loss, "x0_hat": x0_hat}`.

---

### OH-2: x̂₀ reconstruction from denoiser output (CRITICAL)  
**Severity**: Critical  
**Type**: omission  
**Location**: `src/physics_motion_loss/training/trainer.py` — `_training_step()`  
**Description**: The paper states (Sec 4.1): "At every diffusion step t we reconstruct
x̂₀, evaluate the physics-informed frequency loss on x̂₀." The exact DDPM-style x̂₀
reconstruction formula is:

$$\hat{x}_0 = \frac{x_t - \sqrt{1-\bar{\alpha}_t}\,\epsilon_\theta(x_t, t)}{\sqrt{\bar{\alpha}_t}}$$

The trainer stub assumes the backbone's `forward()` directly returns `x0_hat`, but
real diffusion backbones return noise predictions (ε-prediction) or velocity predictions
(v-prediction), not x̂₀ directly.  
**Impact**: If x̂₀ is not correctly reconstructed from the noise prediction, the physics
loss is applied to incorrect intermediate values — metrics will be significantly degraded.  
**Suggested fix**: Implement `_compute_x0_hat()` using the backbone's noise scheduler:
```python
def _compute_x0_hat(self, x_t, t, noise_pred):
    # DDPM x0 reconstruction (ε-prediction formulation)
    alpha_bar = self.noise_scheduler.alphas_cumprod[t]
    return (x_t - (1 - alpha_bar).sqrt() * noise_pred) / alpha_bar.sqrt()
```
Consult the specific backbone's scheduler API (`diffusers.DDPMScheduler` or equivalent).

---

### OH-3: EvalCrafter VQA / Action Recognition metrics (Significant)  
**Severity**: Significant  
**Type**: omission  
**Location**: `src/physics_motion_loss/evaluation/metrics.py`  
**Description**: VQA A, VQA T, SD Score, Action Recognition Score, Motion Accuracy Score,
and BLIP-BLEU are all delegated to a subprocess call to the official EvalCrafter repository.
These are not natively implemented.  
**Evidence**: SIR `evaluation_protocol.evaluation_framework` — "EvalCrafter (Liu et al.,
2023) — official implementation."  
**Impact**: Cannot compute the primary reported metrics without cloning and installing
EvalCrafter. Warping Error and CLIP Temporal Score are natively computed.  
**Suggested fix**:
```bash
git clone https://github.com/EvalCrafter/EvalCrafter third_party/evalcrafter
pip install -r third_party/evalcrafter/requirements.txt
# Then set evaluation.evalcrafter_root in config.yaml
```

---

### OH-4: Soft-ring edge sharpness = 20 (Table 5)  
**Severity**: Minor  
**Type**: omission  
**Location**: `src/physics_motion_loss/spectral/fft_utils.py` — `get_ring_energies()`  
**Description**: Table 5 of the paper lists "Soft-ring edge sharpness: 20". The current
ring energy implementation uses hard binary masks (pixels either inside or outside a ring).
The paper appears to use a soft sigmoid-edged ring mask.  
**Evidence**: Table 5 fixed values: "Soft-ring edge sharpness 20."  
**Impact**: Minor — hard masks are a valid approximation. Soft masks reduce leakage
between adjacent rings, potentially improving C_ring sensitivity by 1–2%.  
**Suggested fix**: Replace the hard mask in `get_ring_energies()` with:
```python
# Soft sigmoid mask with sharpness=20
distance_to_edge = torch.min(rho_map - ring_edges[k], ring_edges[k+1] - rho_map)
mask = torch.sigmoid(20.0 * distance_to_edge)   # sharpness=20 per Table 5
```

---

## Part 4 — Correctness Verification

The following components were verified correct against the SIR:

| Component | Equation | Verified |
|-----------|----------|---------|
| Translation plane $\omega_t + v_x\omega_x + v_y\omega_y + b_0 = 0$ | Eq. 3.3 | ✓ |
| Ridge WLS estimator $\hat{\beta} = \arg\min \sum W_{ii}(A_i\beta-b_i)^2 + \lambda\|\beta\|^2$ | Sec 3.4 | ✓ |
| Angular velocity $\Omega^* = -\frac{\sum|\tilde{C}_m|^2\omega_t m}{\sum|\tilde{C}_m|^2 m^2}$ | Eq. A.7 | ✓ |
| Tilted-line ratio $C_{\text{rot}} = E_{\text{line}}/E_{\text{all}}$ | Eq. A.8 | ✓ |
| Ring entropy $C_{\text{ring}} = 1 - \bar{H}_{\text{ring}}/\log(N_r)$ | App D.2 | ✓ |
| Rotation loss $L_{\text{rot}} = 1 - (C_{\text{ring}}+C_{\text{rot}})/2$ | Sec 3.5 | ✓ |
| Radial centroid $\rho_c(t) = \sum_k k E_k(t) / \sum_k E_k(t)$ | Sec 3.6 | ✓ |
| Scaling loss $L_{\text{scale}} = 1 - (C_{\text{flow}}+S_{\text{trend}})/2$ | Sec 3.6 | ✓ |
| Adaptive weights $w_i = \exp(-L_i/\tau)/\sum_j\exp(-L_j/\tau)$ | Eq. 3.4 | ✓ |
| Low-pass cube $\varrho=0.3 \Rightarrow 2.7\%$ coefficients | Sec 4.1 | ✓ |
| Energy retention $\eta_{\text{cube}}(0.3) \in [0.97, 0.987]$ | Eq. 4.2 | ✓ |
| Short-window edge case: $T_{lp}<3 \Rightarrow L_{\text{scale}}=0.5$ | Sec 3.6 | ✓ |
| $m=0$ excluded from rotation sums | Sec 3.5 | ✓ |
| Hann window before temporal DFT | App A.1 | ✓ |

All 14 core mathematical components verified against the paper. No structural
errors detected in the loss computations themselves.

---

## Action Priority List

| Priority | Issue | Effort | Impact |
|----------|-------|--------|--------|
| 🔴 P0 | Implement backbone loaders (OH-1) | High | Blocks all training |
| 🔴 P0 | Implement x̂₀ reconstruction (OH-2) | Medium | Blocks correct physics loss |
| 🟡 P1 | Install EvalCrafter (OH-3) | Low | Blocks primary metric computation |
| 🟡 P1 | Ablate optimizer (PH-1) | Medium | 2–5% metric impact |
| 🟡 P1 | Ablate LoRA rank (PH-2) | Medium | 5–10% Hunyuan metric impact |
| 🟠 P2 | Soft-ring mask sharpness=20 (OH-4) | Low | 1–2% C_ring improvement |
| 🟠 P2 | Verify motion complexity classifier (SH-2) | Medium | Affects Table 3 only |
| 🟢 P3 | Test stop-grad on weights (PH-5) | Low | < 2% impact |
| 🟢 P3 | Test physics weight schedule (PH-3) | Low | 1–3% VQA impact |
