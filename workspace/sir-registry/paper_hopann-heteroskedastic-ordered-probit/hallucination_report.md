# Hallucination Report

## Structural Hallucinations
None detected. The architecture correctly implements OPANN and HOPANN per the SIR, including the cutting points cumsum+softplus parameterisation.

## Parametric Hallucinations
The following parameters were not explicitly specified in the paper and were assumed by the code generator:

1. **batch_size = 64**
   - Severity: Minor
   - Type: parametric
   - Evidence: Paper mentions "Adam optimizer" but no batch size.
   - Suggested Fix: Expose as CLI argument for hyperparameter search if results deviate during full reproduction.

2. **learning_rate = 1e-3**
   - Severity: Minor
   - Type: parametric
   - Evidence: Paper states LR was searched as a hyperparameter. 1e-3 is the Adam default.
   - Suggested Fix: Search over `[1e-4, 1e-3, 5e-3]` in full reproduction.

3. **variance_network_type = ann**
   - Severity: Significant
   - Type: parametric
   - Evidence: The paper describes $\sigma_i = \exp(z_i \gamma)$, which implies a linear combination, but the model is named Heteroskedastic Ordered Probit with an *ANN*. We assumed an ANN, but a linear projection might be intended. Both were implemented in the code base.
   - Suggested Fix: Run ablation comparing `variance_type="ann"` vs `variance_type="linear"`.

## Omission Hallucinations
None detected.
