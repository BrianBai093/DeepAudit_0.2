# Paper-Ready Result Framing

This section reports the cleaned analysis set.

这份摘要只基于现有 `artifacts/`，不重跑实验，不修改原始 verdict。它把 DeepAudit 的能力拆成两层：严格 claim-level exact support 与 experiment-level execution evidence。

## Key Numbers

| Metric | Value | How to phrase it |
| --- | ---: | --- |
| Papers audited | 18 | We evaluated 18 paper-code pairs after excluding incomplete/mismatched cases. |
| Claims extracted | 1804 | The system extracted 1804 paper claims. |
| Code-verifiable claims | 1637 (90.7%) | Most extracted claims were classified as code-verifiable. |
| Environment setup success | 13/18 (72.2%) | The environment agent established runnable environments for a majority of repos. |
| Standard Phase2 package complete | 17/18 (94.4%) | Most runs produced the standardized execution package consumed by Phase3. |
| Positive experiment-level evidence | 12/18 (66.7%) | Existing artifacts contain positive execution evidence for a large fraction of the cleaned corpus. |
| Positive evidence among packaged repos | 12/17 (70.6%) | Conditional on producing the standard Phase2 package, most repositories produced positive evidence. |
| Positive evidence among repos with non-skipped run rows | 12/14 (85.7%) | Conditional on non-skipped run records existing, positive evidence was recovered in most cases. |
| Full reproduction evidence | 3/18 (16.7%) | These repositories contain at least one full reproduction run. |
| Trend-level evidence | 6/18 (33.3%) | These repositories reproduce reduced-fidelity or trend-level evidence. |
| Executable/smoke evidence | 3/18 (16.7%) | These repositories reached executable/smoke-level validation. |
| Run rows | 90 total; 20 ok, 9 partial, 21 failed | The executor produced structured run records. |
| Claim-level exact support | 3 supported, 81 not supported, 1720 inconclusive | This is intentionally strict and should be described as a lower bound. |
| Reproduced comparison figures | 105 | Phase3 generated reproduced/comparison visual artifacts. |
| Median calibrated score | 65.0/100 | Calibrated scores summarize environment, data, execution, and claim-match evidence. |

## Paper-Ready Paragraph

Across 18 cleaned paper-code pairs, DeepAudit extracted 1804 paper claims, of which 1637 (90.7%) were classified as code-verifiable. The strict claim-level verifier produced 3 exactly supported claims, reflecting a deliberately conservative metric-to-claim alignment policy. However, this exact-support number is a lower bound on system performance: reanalyzing the same artifacts at the experiment-evidence level shows positive execution evidence for 12/18 repositories, including 3 repositories with full reproduction evidence, 6 with trend-level evidence, and 3 with executable/smoke evidence. In addition, 17/18 repositories produced a standardized Phase2 execution package and 13/18 completed environment setup. Conditional on producing a standard Phase2 package, 12/17 repositories produced positive experiment-level evidence; conditional on having non-skipped canonical run records, the rate was 12/14. These results suggest that the main bottleneck is not only repository execution, but the harder final step of aligning heterogeneous execution metrics back to fine-grained paper claims.

## Safer Claim Wording

- Use **"strict exact-support lower bound"** instead of only "supported claims".
- Use **"positive experiment-level evidence"** for FULLY_REPRODUCED / TREND_SUPPORTED / EXECUTABLE outcomes.
- Say **"DeepAudit separates execution success from exact claim support"**, which makes the low supported count a design choice rather than a simple failure.
- Say **"claim alignment is the bottleneck"** when full/trend runs exist but verdicts remain inconclusive.
- Avoid saying **"all positive-evidence repos were reproduced"**. Say **"positive experiment-level evidence"**.

## Evidence Tier Lists

### Full reproduction evidence

- `04_Learning_without_Feedback_DRTP`: score=80.0, runs ok/partial/failed=3/0/0, figures=5/1
- `13_Liquid_TimeConstant_Networks_LTC`: score=80.0, runs ok/partial/failed=3/0/3, figures=7/5
- `23_Next_Generation_Reservoir_Computing`: score=80.0, runs ok/partial/failed=3/2/0, figures=10/1

### Trend evidence

- `05_Lagrangian_Neural_Networks`: score=65.0, runs ok/partial/failed=0/1/0, figures=3/2
- `07_Symplectic_ODE_Net`: score=65.0, runs ok/partial/failed=0/1/6, figures=6/4
- `09_Fourier_Neural_Operator_FNO`: score=65.0, runs ok/partial/failed=3/0/0, figures=4/6
- `10_Your_Classifier_is_Secretly_an_EBM_JEM`: score=65.0, runs ok/partial/failed=2/4/0, figures=12/7
- `11_MACE_Higher_Order_Equivariant_MPNNs copy`: score=65.0, runs ok/partial/failed=1/0/0, figures=3/2
- `21_KAN_KolmogorovArnold_Networks`: score=65.0, runs ok/partial/failed=1/0/0, figures=11/18

### Executable/smoke evidence

- `01_Scaling_Equilibrium_Propagation_to_Deep_ConvNets`: score=50.0, runs ok/partial/failed=0/1/0, figures=0/12
- `03_PEPITA`: score=65.0, runs ok/partial/failed=3/0/1, figures=8/6
- `08_E_n_Equivariant_Graph_Neural_Networks_EGNN`: score=50.0, runs ok/partial/failed=1/0/2, figures=6/2
