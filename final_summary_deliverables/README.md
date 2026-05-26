# DeepAudit Final Summary Deliverables

This folder contains the cleaned final reporting artifacts for the DeepAudit experiment summary.

## Main documents

- `methods.md`: manuscript-ready Methods section for the DeepAudit pipeline.
- `result.md`: full per-run and aggregate metric report for the cleaned 18-paper set.
- `paper_ready_result_framing.md`: concise paper-ready framing text and headline numbers.

## Data

- `data/deepaudit_metrics_extracted.json`: machine-readable extracted metrics used to generate the report and figures.

## Figures

- `figures/deepaudit_pipeline.svg`: vector pipeline figure for the Methods section.
- `figures/posthoc_evidence_tiers.png`: recommended main figure for evidence-tier results.
- `figures/score_by_run.png`: calibrated DeepAudit score by run.
- `figures/execution_outcomes.png`: repository-level execution outcome summary.
- `figures/failure_modes.png`: inferred failure taxonomy summary.
- `figures/verdict_distribution.png`: strict claim-level verdict distribution; best suited for appendix/limitations.

## Reproduction script

- `scripts/extract_deepaudit_metrics.py`: script used to regenerate the summary from `artifacts/`.

Current cleaned analysis excludes `02`, `12`, `14`, and `20` from the summary output.
