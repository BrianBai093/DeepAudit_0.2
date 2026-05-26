# DeepAudit Experiment Result Summary

Generated at `2026-05-18T02:22:45Z` from `/home/yb2636_columbia_edu/DeepAudit_0.2/artifacts`.

Cleaned analysis set only.

The per-run sections below are written immediately after each run is parsed. Aggregate metrics and plots are appended after all run sections.

# Per-Run Metrics

## 01_Scaling_Equilibrium_Propagation_to_Deep_ConvNets

| field | value |
| --- | --- |
| paper_id | 01 |
| title | Scaling Equilibrium Propagation to Deep ConvNets by Drastically Reducing its Gradient Estimator Bias |
| arxiv_id | 1407.7906 |
| venue | NA |
| year | 2019 |
| model_family | Scaling Equilibrium Propagation to Deep ConvNets |
| repo_url | https://github.com/Laborieux-Axel/Equilibrium-Propagation |
| repo_available / cloned | True / True |
| expected_entry_point | readme-python:main.py |
| documented_entry_point_exists | True |
| compute_requirement | GPU/CUDA likely required or supported |
| dataset_requirement | CIFAR-10 |

### Claim Extraction Metrics

| metric | value |
| --- | --- |
| total_claims_extracted | 109 |
| code_verifiable_claims | 109 |
| non_code_verifiable_claims | 0 |
| pct_code_verifiable_claims | 100.00 |
| claims_from_table / figure / text | 75 / 0 / 34 |
| main_claim_source | table |
| metric_contracts_generated | 8 |
| claims_with_metric_contract | 81 |
| pct_claims_with_metric_contract | 74.30 |
| claims_without_metric_contract | 28 |
| reported_metric_names | best test error, best test error without collapsed runs, loss, mean best test error, test error, test error comparison, test error difference, train error |
| top_reported_metrics | loss:26, mean best test error:5, best test error:4, test error:3, train error:3, test error difference:2, best test error without collapsed runs:1, test error comparison:1 |
| reported_dataset | CIFAR-10 |
| reported_model | none |
| reported_method | none |
| reported_baseline | none |
| claim granularity | headline=12, secondary=97, comparison=0, ablation=0, efficiency=3, qualitative=0, theoretical=0 |

### Task, Entrypoint, Dataset, Command Metrics

| metric | value |
| --- | --- |
| tasks_generated | 10 |
| tasks_per_paper | 10 |
| tasks_per_claim | 0.092 |
| claims_with_candidate_task | 109 |
| claims_without_candidate_task | 0 |
| task_generation_coverage | 100.00 |
| candidate_entry_points_found | 10 |
| entry_points_per_repo | 10 |
| claim_to_entrypoint_mapped | 109 |
| claim_to_entrypoint_unmapped | 0 |
| entrypoint_mapping_coverage | 100.00 |
| entrypoint_ambiguity_count | 0 |
| datasets_identified | CIFAR-10 |
| dataset_available | 1 |
| dataset_missing | 0 |
| dataset_mapping_coverage | 20.00 |
| commands_generated | 10 |
| commands_with_required_args | 10 |
| commands_missing_required_args | 0 |
| commands_matching_readme | 1 |
| commands_inferred_by_agent | 9 |
| command_confidence_score | 0.745 |

### Environment, Repair, Execution Metrics

| metric | value |
| --- | --- |
| env_setup_success / failed / repaired | True / False / False |
| requirements/environment/setup.py/pyproject found | False / False / False / False |
| dependency_install_success / failed | True / False |
| dependency_conflict_count | 0 |
| missing_package_count | 0 |
| obsolete_package_count | 0 |
| repair_attempts | 1 |
| repos_requiring_repair | False |
| repair_success / failed | False / False |
| repair_types | compute_limit, dependency_failure, entrypoint_ambiguity, metric_mismatch, metric_not_found |
| env_setup_runtime_min | 0.350 |
| phase2_elapsed_min | 0.170 |
| runs_attempted / ok / partial / failed / timeout | 5 / 0 / 1 / 0 / 3 |
| run_success_rate | 0.000 |
| runtime_min | 40.53 |
| standard_execution_package_complete | True |
| artifact_count | 48 |
| result/log/checkpoint/figure files | 15 / 19 / 0 / 0 |

Dependency manifests detected: `none`

#### Run-Level / Command-Level Metrics

| run | exp | status | fidelity | exit | runtime_sec | command | cwd | stdout | stderr | reason_codes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| exp_01 | exp_01 | partial | smoke | 0 | 2432.00 | python main.py --model 'CNN' --task 'CIFAR10' --data-aug --channels 128 256 512 512 --kernels 3 3 3 3 --pools 'mmmm' --strides 1 1 1 1 --paddings 1 1 1 0 --fc 10 --optim 'sgd' --lrs 0.25 0.15 0.1 0.08 0.05 --wds 3e-4 ... | /home/yb2636_columbia_edu/DeepAudit_0.2/paper_with_code/01_Scaling_Equilibrium_Propagation_to_Deep_ConvNets/code | True | True | README_VERIFIED_COMMAND, SMOKE_TEST_PASSED, GPU_REQUIRED, BUDGET_INSUFFICIENT, ...(+1) |
| exp_02 | exp_02 | skipped | trend | 0 | 0.000 | python main.py --model 'VFCNN' --task 'CIFAR10' --data-aug --channels 128 256 512 512 --kernels 3 3 3 3 --pools 'mmmm' --strides 1 1 1 1 --paddings 1 1 1 0 --fc 10 --optim 'sgd' --lrs 0.25 0.15 0.1 0.08 0.05 --wds 3e-... | /home/yb2636_columbia_edu/DeepAudit_0.2/paper_with_code/01_Scaling_Equilibrium_Propagation_to_Deep_ConvNets/code | True | True | README_VERIFIED_COMMAND, BUDGET_INSUFFICIENT, LONG_HORIZON_TRAINING, COMMAND_NOT_OBSERVED |
| exp_03 | exp_03 | skipped | full | 0 | 0.000 |  | . | True | True | NON_PRIMARY_EXPERIMENT, SYNTHETIC_RUN_LOG_FROM_EXECUTOR_RESULTS |
| exp_04 | exp_04 | skipped | trend | 0 | 0.000 | python main.py --model 'CNN' --task 'CIFAR10' --data-aug --channels 128 256 512 512 --kernels 3 3 3 3 --pools 'mmmm' --strides 1 1 1 1 --paddings 1 1 1 0 --fc 10 --optim 'sgd' --lrs 0.25 0.15 0.1 0.08 0.05 --wds 3e-4 ... | /home/yb2636_columbia_edu/DeepAudit_0.2/paper_with_code/01_Scaling_Equilibrium_Propagation_to_Deep_ConvNets/code | True | True | BUDGET_INSUFFICIENT, MULTIPLE_RUNS_REQUIRED, LONG_HORIZON_TRAINING, COMMAND_NOT_OBSERVED |
| exp_05 | exp_05 | skipped | full | 0 | 0.000 |  | . | True | True | NON_PRIMARY_EXPERIMENT, SYNTHETIC_RUN_LOG_FROM_EXECUTOR_RESULTS |

### Evidence Alignment, Verdict, Failure Metrics

| metric | value |
| --- | --- |
| claims_evaluated / not_evaluated | 109 / 0 |
| evaluated_claim_rate | 100.00 |
| executable / non_executable claims | 109 / 0 |
| claims_with_observed_metric / without | 0 / 109 |
| metric_recovery_rate | 0.000 |
| metric_parser_success / failed | 2 / 109 |
| reported/observed comparisons | 0 |
| within_tolerance / outside_tolerance | 0 / 0 |
| claim_to_evidence_mapped / unmapped | 0 / 109 |
| supported / partial / not_supported / inconclusive | 0 / 0 / 0 / 109 |
| main_verdict_per_paper | INCONCLUSIVE |
| headline evaluated/supported/partial/not/inconclusive | 12 / 0 / 0 / 0 / 12 |
| headline_metric_recovery_rate | 100.00 |
| executed_but_no_headline_evidence | False |
| executed_but_metric_missing | False |
| executed_but_claim_inconclusive | False |
| failed_but_diagnostic_evidence_available | True |
| failure_modes | dependency_failure:1, entrypoint_ambiguity:1, metric_not_found:1, metric_mismatch:1, compute_limit:1 |
| main_inconclusive_reason | Configuration claim requires direct code/config evidence; execution metrics alone do not verify the paper setup. |
| score_total / raw | 50.0 / 39.5 |
| ECR | False |
| posthoc evidence_tier | EXECUTABLE_OR_SMOKE_EVIDENCE |
| execution_outcome_counts | EXECUTABLE:1, none:4 |
| reproduced/skipped figures | 0 / 12 |
| logs_scanned | 18 |

Top reason codes: LLM_REASON_PROVIDED:166, LLM_TABLE_EXTRACTED:112, VISUAL_ENRICHED:112, CONFIG_CLAIM:64, NO_DIRECT_CONFIG_EVIDENCE:64, VISUAL_TABLE_EXTRACTED:54, ALIGNMENT_AMBIGUOUS:45, README_VERIFIED_COMMAND:18, ENTRYPOINT_DERIVED_FROM_WRAPPER:12, PARSE_LOW_CONFIDENCE:12, BUDGET_INSUFFICIENT:11, COMMAND_NOT_OBSERVED:8


<!-- completed 01_Scaling_Equilibrium_Propagation_to_Deep_ConvNets -->

## 03_PEPITA

| field | value |
| --- | --- |
| paper_id | 03 |
| title | Error-driven Input Modulation: Solving the Credit Assignment Problem without a Backward Pass |
| arxiv_id | 1909.01311 |
| venue | NA |
| year | 2015 |
| model_family | PEPITA |
| repo_url | NA |
| repo_available / cloned | False / False |
| expected_entry_point | readme-python:main_pytorch.py |
| documented_entry_point_exists | True |
| compute_requirement | CPU feasible in observed run |
| dataset_requirement | CIFAR10, CIFAR100, MNIST |

### Claim Extraction Metrics

| metric | value |
| --- | --- |
| total_claims_extracted | 120 |
| code_verifiable_claims | 117 |
| non_code_verifiable_claims | 3 |
| pct_code_verifiable_claims | 97.50 |
| claims_from_table / figure / text | 67 / 0 / 53 |
| main_claim_source | table |
| metric_contracts_generated | 2 |
| claims_with_metric_contract | 117 |
| pct_claims_with_metric_contract | 97.50 |
| claims_without_metric_contract | 3 |
| reported_metric_names | accuracy, slowness |
| top_reported_metrics | accuracy:76, slowness:12 |
| reported_dataset | CIFAR10, CIFAR100, MNIST |
| reported_model | none |
| reported_method | none |
| reported_baseline | none |
| claim granularity | headline=26, secondary=94, comparison=0, ablation=4, efficiency=0, qualitative=0, theoretical=0 |

### Task, Entrypoint, Dataset, Command Metrics

| metric | value |
| --- | --- |
| tasks_generated | 12 |
| tasks_per_paper | 12 |
| tasks_per_claim | 0.100 |
| claims_with_candidate_task | 120 |
| claims_without_candidate_task | 0 |
| task_generation_coverage | 100.00 |
| candidate_entry_points_found | 12 |
| entry_points_per_repo | 12 |
| claim_to_entrypoint_mapped | 120 |
| claim_to_entrypoint_unmapped | 0 |
| entrypoint_mapping_coverage | 100.00 |
| entrypoint_ambiguity_count | 0 |
| datasets_identified | CIFAR10, CIFAR100, MNIST |
| dataset_available | 3 |
| dataset_missing | 0 |
| dataset_mapping_coverage | 60.00 |
| commands_generated | 12 |
| commands_with_required_args | 12 |
| commands_missing_required_args | 0 |
| commands_matching_readme | 2 |
| commands_inferred_by_agent | 10 |
| command_confidence_score | 0.927 |

### Environment, Repair, Execution Metrics

| metric | value |
| --- | --- |
| env_setup_success / failed / repaired | False / True / False |
| requirements/environment/setup.py/pyproject found | False / False / False / False |
| dependency_install_success / failed | False / False |
| dependency_conflict_count | 0 |
| missing_package_count | 0 |
| obsolete_package_count | 0 |
| repair_attempts | 1 |
| repos_requiring_repair | False |
| repair_success / failed | False / False |
| repair_types | compute_limit, dependency_failure, entrypoint_ambiguity, metric_mismatch, metric_not_found, missing_dataset, ...(+1) |
| env_setup_runtime_min | 0.230 |
| phase2_elapsed_min | 115.33 |
| runs_attempted / ok / partial / failed / timeout | 5 / 3 / 0 / 1 / 0 |
| run_success_rate | 60.00 |
| runtime_min | 50.20 |
| standard_execution_package_complete | True |
| artifact_count | 57 |
| result/log/checkpoint/figure files | 15 / 30 / 0 / 20 |

Dependency manifests detected: `none`

#### Run-Level / Command-Level Metrics

| run | exp | status | fidelity | exit | runtime_sec | command | cwd | stdout | stderr | reason_codes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| exp_01 | exp_01 | ok | smoke | 0 | 2059.14 | /home/yb2636_columbia_edu/miniconda3/condabin/mamba run -n PEPITA_new_executor python main.py --exp_name exp_01_smoke_cifar10_fc_erin --learn_type ERIN --n_runs 1 --train_epochs 3 --sample_passes 2 --n_samples all --e... | /home/yb2636_columbia_edu/DeepAudit_0.2/Target/code | True | True | SMOKE_TEST_ONLY, LOW_SNR_FOR_TREND, SYNTHETIC_RUN_LOG_FROM_EXECUTOR_RESULTS, COMMAND_NOT_OBSERVED |
| exp_02 | exp_02 | ok | smoke | 0 | 450.00 | /home/yb2636_columbia_edu/miniconda3/condabin/mamba run -n PEPITA_new_executor python main_pytorch.py --exp_name exp_02_smoke_cifar100_conv_bp --learn_type BP --n_runs 1 --train_epochs 3 --eta 0.01 --dropout 0.9 --Bst... | /home/yb2636_columbia_edu/DeepAudit_0.2/Target/code | True | True | SMOKE_TEST_ONLY, SYNTHETIC_RUN_LOG_FROM_EXECUTOR_RESULTS, COMMAND_NOT_OBSERVED |
| exp_03 | exp_03 | failed | smoke | 1 | NA |  | . | True | True | EXPERIMENT_RESULT_MISSING |
| exp_04 | exp_04 | ok | smoke | 0 | 503.00 | /home/yb2636_columbia_edu/miniconda3/condabin/mamba run -n PEPITA_new_executor python main.py --exp_name exp_04_smoke_mnist_fc_pepita_whitened --learn_type ERIN --n_runs 1 --train_epochs 3 --sample_passes 2 --n_sample... | /home/yb2636_columbia_edu/DeepAudit_0.2/Target/code | True | True | SMOKE_TEST_ONLY, ABLATION_EFFECT_CONFIRMED, SYNTHETIC_RUN_LOG_FROM_EXECUTOR_RESULTS, COMMAND_NOT_OBSERVED |
| exp_05 | exp_05 | skipped | full | 0 | 0.000 | skipped | /home/yb2636_columbia_edu/DeepAudit_0.2/Target/code | True | True | NON_PRIMARY_EXPERIMENT, SETUP_ONLY, SYNTHETIC_RUN_LOG_FROM_EXECUTOR_RESULTS, COMMAND_NOT_OBSERVED |

### Evidence Alignment, Verdict, Failure Metrics

| metric | value |
| --- | --- |
| claims_evaluated / not_evaluated | 120 / 0 |
| evaluated_claim_rate | 100.00 |
| executable / non_executable claims | 117 / 3 |
| claims_with_observed_metric / without | 10 / 110 |
| metric_recovery_rate | 8.30 |
| metric_parser_success / failed | 4 / 110 |
| reported/observed comparisons | 10 |
| within_tolerance / outside_tolerance | 0 / 10 |
| claim_to_evidence_mapped / unmapped | 10 / 110 |
| supported / partial / not_supported / inconclusive | 0 / 0 / 10 / 110 |
| main_verdict_per_paper | INCONCLUSIVE |
| headline evaluated/supported/partial/not/inconclusive | 26 / 0 / 0 / 4 / 22 |
| headline_metric_recovery_rate | 100.00 |
| executed_but_no_headline_evidence | False |
| executed_but_metric_missing | False |
| executed_but_claim_inconclusive | True |
| failed_but_diagnostic_evidence_available | False |
| failure_modes | missing_dataset:1, dependency_failure:1, entrypoint_ambiguity:1, metric_not_found:1, metric_mismatch:1, compute_limit:1, undocumented_hyperparameter:1 |
| main_inconclusive_reason | Experiment run exists but metric `accuracy` could not be aligned for this claim. |
| score_total / raw | 65.0 / 47.1 |
| ECR | False |
| posthoc evidence_tier | EXECUTABLE_OR_SMOKE_EVIDENCE |
| execution_outcome_counts | EXECUTABLE:3, none:2 |
| reproduced/skipped figures | 8 / 6 |
| logs_scanned | 29 |

Top reason codes: LLM_REASON_PROVIDED:158, VISUAL_ENRICHED:106, VISUAL_TABLE_EXTRACTED:104, ALIGNMENT_AMBIGUOUS:78, MEAN_STD_TARGET_NORMALIZED:76, TRAIN_ACCURACY:57, VALIDATION_ACCURACY:57, LLM_TABLE_EXTRACTED:54, TABLE_EXPANDED:52, CAPTION_METRIC_MATRIX:52, CONFIG_CLAIM:29, NO_DIRECT_CONFIG_EVIDENCE:29


<!-- completed 03_PEPITA -->

## 04_Learning_without_Feedback_DRTP

| field | value |
| --- | --- |
| paper_id | 04 |
| title | Learning Without Feedback: Fixed Random Learning Signals Allow for Feedforward Training of Deep Neural Networks |
| arxiv_id | 1901.08164 |
| venue | NA |
| year | 2020 |
| model_family | Learning without Feedback DRTP |
| repo_url | https://github.com/ChFrenkel/DirectRandomTargetProjection |
| repo_available / cloned | True / True |
| expected_entry_point | python-file:synth_dataset_gen.py |
| documented_entry_point_exists | False |
| compute_requirement | GPU/CUDA likely required or supported |
| dataset_requirement | CIFAR10aug, MNIST |

### Claim Extraction Metrics

| metric | value |
| --- | --- |
| total_claims_extracted | 120 |
| code_verifiable_claims | 66 |
| non_code_verifiable_claims | 54 |
| pct_code_verifiable_claims | 55.00 |
| claims_from_table / figure / text | 52 / 0 / 68 |
| main_claim_source | text |
| metric_contracts_generated | 1 |
| claims_with_metric_contract | 117 |
| pct_claims_with_metric_contract | 97.50 |
| claims_without_metric_contract | 3 |
| reported_metric_names | none |
| top_reported_metrics | none |
| reported_dataset | CIFAR10aug, MNIST |
| reported_model | none |
| reported_method | none |
| reported_baseline | none |
| claim granularity | headline=0, secondary=120, comparison=0, ablation=0, efficiency=0, qualitative=0, theoretical=0 |

### Task, Entrypoint, Dataset, Command Metrics

| metric | value |
| --- | --- |
| tasks_generated | 2 |
| tasks_per_paper | 2 |
| tasks_per_claim | 0.017 |
| claims_with_candidate_task | 120 |
| claims_without_candidate_task | 0 |
| task_generation_coverage | 100.00 |
| candidate_entry_points_found | 2 |
| entry_points_per_repo | 2 |
| claim_to_entrypoint_mapped | 120 |
| claim_to_entrypoint_unmapped | 0 |
| entrypoint_mapping_coverage | 100.00 |
| entrypoint_ambiguity_count | 0 |
| datasets_identified | CIFAR10aug, MNIST |
| dataset_available | 2 |
| dataset_missing | 0 |
| dataset_mapping_coverage | 50.00 |
| commands_generated | 2 |
| commands_with_required_args | 2 |
| commands_missing_required_args | 0 |
| commands_matching_readme | 0 |
| commands_inferred_by_agent | 2 |
| command_confidence_score | 0.930 |

### Environment, Repair, Execution Metrics

| metric | value |
| --- | --- |
| env_setup_success / failed / repaired | False / True / False |
| requirements/environment/setup.py/pyproject found | False / False / True / False |
| dependency_install_success / failed | False / False |
| dependency_conflict_count | 0 |
| missing_package_count | 0 |
| obsolete_package_count | 0 |
| repair_attempts | 1 |
| repos_requiring_repair | False |
| repair_success / failed | False / False |
| repair_types | compute_limit, dependency_failure, entrypoint_ambiguity, metric_mismatch, metric_not_found, missing_dataset |
| env_setup_runtime_min | 0.110 |
| phase2_elapsed_min | 0.110 |
| runs_attempted / ok / partial / failed / timeout | 4 / 3 / 0 / 0 / 0 |
| run_success_rate | 75.00 |
| runtime_min | 395.00 |
| standard_execution_package_complete | True |
| artifact_count | 52 |
| result/log/checkpoint/figure files | 15 / 22 / 0 / 10 |

Dependency manifests detected: `setup.py`

#### Run-Level / Command-Level Metrics

| run | exp | status | fidelity | exit | runtime_sec | command | cwd | stdout | stderr | reason_codes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| exp_01 | exp_01 | skipped | full | 0 | 0.000 | skipped | . | True | True | non_primary_experiment, undefined_configuration |
| exp_02 | exp_02 | ok | full | 0 | 7200.00 | python main.py --dataset MNIST --train-mode BP --epochs 100 --topology CONV_32_5_1_2_FC_1000_FC_10 --dropout 0.0 --loss CE --output-act none --cpu | /home/yb2636_columbia_edu/DeepAudit_0.2/paper_with_code/04_Learning_without_Feedback_DRTP/code | True | True | primary_experiment, full_run_completed, convergence_observed, SYNTHETIC_RUN_LOG_FROM_EXECUTOR_RESULTS |
| table_1 | table_1 | ok | full | 0 | 7500.00 | python main.py --dataset MNIST --train-mode BP --epochs 100 --topology FC_500_FC_500_FC_10 --dropout 0.25 --loss CE --output-act none --cpu | /home/yb2636_columbia_edu/DeepAudit_0.2/paper_with_code/04_Learning_without_Feedback_DRTP/code | True | True | primary_experiment, full_run_completed, convergence_observed, SYNTHETIC_RUN_LOG_FROM_EXECUTOR_RESULTS |
| table_2 | table_2 | ok | full | 0 | 9000.00 | python main.py --dataset CIFAR10aug --train-mode BP --epochs 100 --topology FC_500_FC_10 --loss CE --output-act none --cpu | /home/yb2636_columbia_edu/DeepAudit_0.2/paper_with_code/04_Learning_without_Feedback_DRTP/code | True | True | primary_experiment, full_run_completed, convergence_observed, SYNTHETIC_RUN_LOG_FROM_EXECUTOR_RESULTS |

### Evidence Alignment, Verdict, Failure Metrics

| metric | value |
| --- | --- |
| claims_evaluated / not_evaluated | 120 / 0 |
| evaluated_claim_rate | 100.00 |
| executable / non_executable claims | 66 / 54 |
| claims_with_observed_metric / without | 0 / 120 |
| metric_recovery_rate | 0.000 |
| metric_parser_success / failed | 15 / 120 |
| reported/observed comparisons | 0 |
| within_tolerance / outside_tolerance | 0 / 0 |
| claim_to_evidence_mapped / unmapped | 0 / 120 |
| supported / partial / not_supported / inconclusive | 0 / 0 / 0 / 120 |
| main_verdict_per_paper | INCONCLUSIVE |
| headline evaluated/supported/partial/not/inconclusive | 0 / 0 / 0 / 0 / 0 |
| headline_metric_recovery_rate | 0.000 |
| executed_but_no_headline_evidence | False |
| executed_but_metric_missing | True |
| executed_but_claim_inconclusive | True |
| failed_but_diagnostic_evidence_available | False |
| failure_modes | missing_dataset:1, dependency_failure:1, entrypoint_ambiguity:1, metric_not_found:1, metric_mismatch:1, compute_limit:1 |
| main_inconclusive_reason | Configuration claim requires direct code/config evidence; execution metrics alone do not verify the paper setup. |
| score_total / raw | 80.0 / 63.1 |
| ECR | False |
| posthoc evidence_tier | FULL_REPRODUCTION_EVIDENCE |
| execution_outcome_counts | none:1, FULLY_REPRODUCED:3 |
| reproduced/skipped figures | 5 / 1 |
| logs_scanned | 21 |

Top reason codes: LLM_REASON_PROVIDED:170, VISUAL_ENRICHED:105, LLM_TABLE_EXTRACTED:91, VISUAL_TABLE_EXTRACTED:79, CONFIG_CLAIM:66, NO_DIRECT_CONFIG_EVIDENCE:66, MISSING_RECORDS:54, MEAN_STD_TARGET_NORMALIZED:51, PHASE2_PACKAGE_METRIC:15, PARSE_LOW_CONFIDENCE:15, primary_experiment:9, full_run_completed:9


<!-- completed 04_Learning_without_Feedback_DRTP -->

## 05_Lagrangian_Neural_Networks

| field | value |
| --- | --- |
| paper_id | 05 |
| title | LAGRANGIAN NEURAL NETWORKS |
| arxiv_id | 1909.13334 |
| venue | NA |
| year | 2012 |
| model_family | Lagrangian Neural Networks |
| repo_url | https://github.com/MilesCranmer/lagrangian_nns |
| repo_available / cloned | True / True |
| expected_entry_point | python-file:examples/double_pendulum/train.py |
| documented_entry_point_exists | False |
| compute_requirement | CPU feasible in observed run |
| dataset_requirement | none |

### Claim Extraction Metrics

| metric | value |
| --- | --- |
| total_claims_extracted | 31 |
| code_verifiable_claims | 4 |
| non_code_verifiable_claims | 27 |
| pct_code_verifiable_claims | 12.90 |
| claims_from_table / figure / text | 25 / 0 / 6 |
| main_claim_source | table |
| metric_contracts_generated | 1 |
| claims_with_metric_contract | 2 |
| pct_claims_with_metric_contract | 6.50 |
| claims_without_metric_contract | 29 |
| reported_metric_names | loss |
| top_reported_metrics | loss:2 |
| reported_dataset | none |
| reported_model | none |
| reported_method | none |
| reported_baseline | none |
| claim granularity | headline=6, secondary=25, comparison=0, ablation=0, efficiency=0, qualitative=0, theoretical=0 |

### Task, Entrypoint, Dataset, Command Metrics

| metric | value |
| --- | --- |
| tasks_generated | 4 |
| tasks_per_paper | 4 |
| tasks_per_claim | 0.129 |
| claims_with_candidate_task | 31 |
| claims_without_candidate_task | 0 |
| task_generation_coverage | 100.00 |
| candidate_entry_points_found | 4 |
| entry_points_per_repo | 4 |
| claim_to_entrypoint_mapped | 31 |
| claim_to_entrypoint_unmapped | 0 |
| entrypoint_mapping_coverage | 100.00 |
| entrypoint_ambiguity_count | 0 |
| datasets_identified | none |
| dataset_available | 0 |
| dataset_missing | NA |
| dataset_mapping_coverage | 0.000 |
| commands_generated | 4 |
| commands_with_required_args | 4 |
| commands_missing_required_args | 0 |
| commands_matching_readme | 0 |
| commands_inferred_by_agent | 4 |
| command_confidence_score | 0.877 |

### Environment, Repair, Execution Metrics

| metric | value |
| --- | --- |
| env_setup_success / failed / repaired | True / False / False |
| requirements/environment/setup.py/pyproject found | False / False / False / True |
| dependency_install_success / failed | True / False |
| dependency_conflict_count | 0 |
| missing_package_count | 0 |
| obsolete_package_count | 0 |
| repair_attempts | 1 |
| repos_requiring_repair | False |
| repair_success / failed | False / False |
| repair_types | compute_limit, dependency_failure, entrypoint_ambiguity, metric_mismatch, metric_not_found |
| env_setup_runtime_min | 2.44 |
| phase2_elapsed_min | 47.03 |
| runs_attempted / ok / partial / failed / timeout | 2 / 0 / 1 / 0 / 1 |
| run_success_rate | 0.000 |
| runtime_min | 0.000 |
| standard_execution_package_complete | True |
| artifact_count | 45 |
| result/log/checkpoint/figure files | 15 / 13 / 0 / 6 |

Dependency manifests detected: `pyproject.toml`

#### Run-Level / Command-Level Metrics

| run | exp | status | fidelity | exit | runtime_sec | command | cwd | stdout | stderr | reason_codes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| exp_01 | exp_01 | partial | artifact | 0 | 0.000 | python examples/double_pendulum/train.py --model baseline_nn --num-batches 500 | /home/yb2636_columbia_edu/DeepAudit_0.2/paper_with_code/05_Lagrangian_Neural_Networks/code | True | True | ARTIFACT_AVAILABLE, BUDGET_INSUFFICIENT_FOR_CPU_TRAINING, CODE_VERIFIED, JAX_JIT_OVERHEAD |
| exp_02 | exp_02 | skipped | full | 0 | 0.000 | N/A - qualitative comparison table | . | True | True | QUALITATIVE_TABLE, NON_EXECUTABLE, TAXONOMY_COMPARISON |

### Evidence Alignment, Verdict, Failure Metrics

| metric | value |
| --- | --- |
| claims_evaluated / not_evaluated | 31 / 0 |
| evaluated_claim_rate | 100.00 |
| executable / non_executable claims | 4 / 27 |
| claims_with_observed_metric / without | 0 / 31 |
| metric_recovery_rate | 0.000 |
| metric_parser_success / failed | 1 / 31 |
| reported/observed comparisons | 0 |
| within_tolerance / outside_tolerance | 0 / 0 |
| claim_to_evidence_mapped / unmapped | 0 / 31 |
| supported / partial / not_supported / inconclusive | 0 / 0 / 0 / 31 |
| main_verdict_per_paper | INCONCLUSIVE |
| headline evaluated/supported/partial/not/inconclusive | 6 / 0 / 0 / 0 / 6 |
| headline_metric_recovery_rate | 100.00 |
| executed_but_no_headline_evidence | False |
| executed_but_metric_missing | False |
| executed_but_claim_inconclusive | False |
| failed_but_diagnostic_evidence_available | True |
| failure_modes | dependency_failure:1, entrypoint_ambiguity:1, metric_not_found:1, metric_mismatch:1, compute_limit:1 |
| main_inconclusive_reason | No aligned metric record found. |
| score_total / raw | 65.0 / 56.9 |
| ECR | False |
| posthoc evidence_tier | TREND_EVIDENCE |
| execution_outcome_counts | TREND_SUPPORTED:1, none:1 |
| reproduced/skipped figures | 3 / 2 |
| logs_scanned | 12 |

Top reason codes: LLM_REASON_PROVIDED:50, LLM_TABLE_EXTRACTED:34, VISUAL_ENRICHED:34, MISSING_RECORDS:27, VISUAL_TABLE_EXTRACTED:16, ENTRYPOINT_CWD_INFERRED:9, ERROR_LOG:7, LLM_SKIP_OVERRIDDEN_BY_DETERMINISTIC_FALLBACK:6, QUALITATIVE_TABLE:5, NON_EXECUTABLE:5, TAXONOMY_COMPARISON:5, PARSE_LOW_CONFIDENCE:5


<!-- completed 05_Lagrangian_Neural_Networks -->

## 06_Simplifying_HNN_LNN_via_Explicit_Constraints

| field | value |
| --- | --- |
| paper_id | 06 |
| title | Simplifying Hamiltonian and Lagrangian Neural Networks via Explicit Constraints |
| arxiv_id | 1909.13334 |
| venue | NA |
| year | NA |
| model_family | Simplifying HNN LNN via Explicit Constraints |
| repo_url | https://github.com/mfinzi/constrained-hamiltonian-neural-networks |
| repo_available / cloned | True / True |
| expected_entry_point | readme-python:pl_trainer.py |
| documented_entry_point_exists | True |
| compute_requirement | unknown |
| dataset_requirement | 3-pendulum, gyroscope system |

### Claim Extraction Metrics

| metric | value |
| --- | --- |
| total_claims_extracted | 34 |
| code_verifiable_claims | 34 |
| non_code_verifiable_claims | 0 |
| pct_code_verifiable_claims | 100.00 |
| claims_from_table / figure / text | 0 / 0 / 34 |
| main_claim_source | text |
| metric_contracts_generated | 1 |
| claims_with_metric_contract | 13 |
| pct_claims_with_metric_contract | 38.20 |
| claims_without_metric_contract | 21 |
| reported_metric_names | accuracy |
| top_reported_metrics | accuracy:1 |
| reported_dataset | 3-pendulum, gyroscope system |
| reported_model | none |
| reported_method | none |
| reported_baseline | none |
| claim granularity | headline=3, secondary=31, comparison=0, ablation=0, efficiency=2, qualitative=0, theoretical=0 |

### Task, Entrypoint, Dataset, Command Metrics

| metric | value |
| --- | --- |
| tasks_generated | 6 |
| tasks_per_paper | 6 |
| tasks_per_claim | 0.176 |
| claims_with_candidate_task | 34 |
| claims_without_candidate_task | 0 |
| task_generation_coverage | 100.00 |
| candidate_entry_points_found | 6 |
| entry_points_per_repo | 6 |
| claim_to_entrypoint_mapped | 34 |
| claim_to_entrypoint_unmapped | 0 |
| entrypoint_mapping_coverage | 100.00 |
| entrypoint_ambiguity_count | 0 |
| datasets_identified | 3-pendulum, gyroscope system |
| dataset_available | 0 |
| dataset_missing | 0 |
| dataset_mapping_coverage | 28.60 |
| commands_generated | 6 |
| commands_with_required_args | 6 |
| commands_missing_required_args | 0 |
| commands_matching_readme | 1 |
| commands_inferred_by_agent | 5 |
| command_confidence_score | 0.853 |

### Environment, Repair, Execution Metrics

| metric | value |
| --- | --- |
| env_setup_success / failed / repaired | True / False / False |
| requirements/environment/setup.py/pyproject found | False / True / True / False |
| dependency_install_success / failed | True / False |
| dependency_conflict_count | 0 |
| missing_package_count | 0 |
| obsolete_package_count | 0 |
| repair_attempts | 1 |
| repos_requiring_repair | False |
| repair_success / failed | False / False |
| repair_types | compute_limit, dependency_failure, entrypoint_ambiguity, metric_not_found |
| env_setup_runtime_min | 1.90 |
| phase2_elapsed_min | 28.60 |
| runs_attempted / ok / partial / failed / timeout | 7 / 0 / 0 / 7 / 0 |
| run_success_rate | 0.000 |
| runtime_min | 0 |
| standard_execution_package_complete | True |
| artifact_count | 23 |
| result/log/checkpoint/figure files | 15 / 4 / 0 / 16 |

Dependency manifests detected: `conda_env.yml, p2c_env.yml, setup.py`

#### Run-Level / Command-Level Metrics

| run | exp | status | fidelity | exit | runtime_sec | command | cwd | stdout | stderr | reason_codes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| exp_01 | exp_01 | failed | smoke | 1 | NA |  | . | True | True | EXPERIMENT_RESULT_MISSING |
| exp_02 | exp_02 | failed | smoke | 1 | NA |  | . | True | True | EXPERIMENT_RESULT_MISSING |
| exp_03 | exp_03 | failed | smoke | 1 | NA |  | . | True | True | EXPERIMENT_RESULT_MISSING |
| exp_04 | exp_04 | failed | smoke | 1 | NA |  | . | True | True | EXPERIMENT_RESULT_MISSING |
| exp_05 | exp_05 | failed | smoke | 1 | NA |  | . | True | True | EXPERIMENT_RESULT_MISSING |
| exp_06 | exp_06 | failed | smoke | 1 | NA |  | . | True | True | EXPERIMENT_RESULT_MISSING |
| exp_07 | exp_07 | failed | smoke | 1 | NA |  | . | True | True | EXPERIMENT_RESULT_MISSING |

### Evidence Alignment, Verdict, Failure Metrics

| metric | value |
| --- | --- |
| claims_evaluated / not_evaluated | 34 / 0 |
| evaluated_claim_rate | 100.00 |
| executable / non_executable claims | 34 / 0 |
| claims_with_observed_metric / without | 0 / 34 |
| metric_recovery_rate | 0.000 |
| metric_parser_success / failed | 1 / 34 |
| reported/observed comparisons | 0 |
| within_tolerance / outside_tolerance | 0 / 0 |
| claim_to_evidence_mapped / unmapped | 0 / 34 |
| supported / partial / not_supported / inconclusive | 0 / 0 / 0 / 34 |
| main_verdict_per_paper | INCONCLUSIVE |
| headline evaluated/supported/partial/not/inconclusive | 3 / 0 / 0 / 0 / 3 |
| headline_metric_recovery_rate | 100.00 |
| executed_but_no_headline_evidence | False |
| executed_but_metric_missing | False |
| executed_but_claim_inconclusive | False |
| failed_but_diagnostic_evidence_available | True |
| failure_modes | dependency_failure:1, entrypoint_ambiguity:1, metric_not_found:1, compute_limit:1 |
| main_inconclusive_reason | Configuration claim requires direct code/config evidence; execution metrics alone do not verify the paper setup. |
| score_total / raw | 43.7 / 43.7 |
| ECR | False |
| posthoc evidence_tier | ATTEMPTED_NO_POSITIVE_EVIDENCE |
| execution_outcome_counts | none:7 |
| reproduced/skipped figures | 8 / 2 |
| logs_scanned | 3 |

Top reason codes: CONFIG_CLAIM:33, NO_DIRECT_CONFIG_EVIDENCE:33, VISUAL_ENRICHED:30, ENTRYPOINT_CWD_INFERRED:25, LLM_SKIP_OVERRIDDEN_BY_DETERMINISTIC_FALLBACK:16, DETERMINISTIC_NO_EVIDENCE_TEXT_PANEL:16, ERROR_LOG:15, NO_NUMERIC_EVIDENCE:12, FAILED_EXECUTION:10, README_VERIFIED_COMMAND:7, EXPERIMENT_RESULT_MISSING:7, LLM_CODEGEN_RENDERED:7


<!-- completed 06_Simplifying_HNN_LNN_via_Explicit_Constraints -->

## 07_Symplectic_ODE_Net

| field | value |
| --- | --- |
| paper_id | 07 |
| title | SYMPLECTIC ODE-NET: LEARNING HAMILTONIAN DYNAMICS WITH CONTROL |
| arxiv_id | 1902.11136 |
| venue | NA |
| year | 2016 |
| model_family | Symplectic ODE Net |
| repo_url | https://github.com/d-biswa/Symplectic-ODENet |
| repo_available / cloned | True / True |
| expected_entry_point | python-file:experiment-single-force/train_hnn.py |
| documented_entry_point_exists | False |
| compute_requirement | unknown |
| dataset_requirement | Acrobot, CartPole, Pendulum, Pendulum (embed), Task 1 Pendulum, Task 1 SymODEN structured, ...(+6) |

### Claim Extraction Metrics

| metric | value |
| --- | --- |
| total_claims_extracted | 78 |
| code_verifiable_claims | 78 |
| non_code_verifiable_claims | 0 |
| pct_code_verifiable_claims | 100.00 |
| claims_from_table / figure / text | 72 / 0 / 6 |
| main_claim_source | table |
| metric_contracts_generated | 6 |
| claims_with_metric_contract | 62 |
| pct_claims_with_metric_contract | 79.50 |
| claims_without_metric_contract | 16 |
| reported_metric_names | mse, prediction error, prediction error per trajectory, test error, train error, train error per trajectory |
| top_reported_metrics | train error:14, test error:14, prediction error per trajectory:14, mse:12, prediction error:4, train error per trajectory:4 |
| reported_dataset | Acrobot, CartPole, Pendulum, Pendulum (embed), Task 1 Pendulum, Task 1 SymODEN structured, Task 2 (Pendulum embed), Task 2 / pendulum embed-like setting with annulus and rectangle training data, ...(+4) |
| reported_model | none |
| reported_method | none |
| reported_baseline | none |
| claim granularity | headline=12, secondary=66, comparison=0, ablation=0, efficiency=45, qualitative=0, theoretical=0 |

### Task, Entrypoint, Dataset, Command Metrics

| metric | value |
| --- | --- |
| tasks_generated | 7 |
| tasks_per_paper | 7 |
| tasks_per_claim | 0.090 |
| claims_with_candidate_task | 78 |
| claims_without_candidate_task | 0 |
| task_generation_coverage | 100.00 |
| candidate_entry_points_found | 7 |
| entry_points_per_repo | 7 |
| claim_to_entrypoint_mapped | 78 |
| claim_to_entrypoint_unmapped | 0 |
| entrypoint_mapping_coverage | 100.00 |
| entrypoint_ambiguity_count | 0 |
| datasets_identified | Acrobot, CartPole, Pendulum, Pendulum (embed), Task 1 Pendulum, Task 1 SymODEN structured, Task 2 (Pendulum embed), Task 2 / pendulum embed-like setting with annulus and rectangle training data, ...(+4) |
| dataset_available | 1 |
| dataset_missing | 0 |
| dataset_mapping_coverage | 171.40 |
| commands_generated | 7 |
| commands_with_required_args | 7 |
| commands_missing_required_args | 0 |
| commands_matching_readme | 0 |
| commands_inferred_by_agent | 7 |
| command_confidence_score | 0.930 |

### Environment, Repair, Execution Metrics

| metric | value |
| --- | --- |
| env_setup_success / failed / repaired | True / False / True |
| requirements/environment/setup.py/pyproject found | False / False / False / False |
| dependency_install_success / failed | True / False |
| dependency_conflict_count | 0 |
| missing_package_count | 0 |
| obsolete_package_count | 0 |
| repair_attempts | 1 |
| repos_requiring_repair | True |
| repair_success / failed | True / False |
| repair_types | compute_limit, dependency_failure, entrypoint_ambiguity, metric_mismatch, metric_not_found, missing_dataset |
| env_setup_runtime_min | 0.180 |
| phase2_elapsed_min | 101.55 |
| runs_attempted / ok / partial / failed / timeout | 7 / 0 / 1 / 6 / 0 |
| run_success_rate | 0.000 |
| runtime_min | 78.43 |
| standard_execution_package_complete | True |
| artifact_count | 43 |
| result/log/checkpoint/figure files | 15 / 18 / 0 / 12 |

Dependency manifests detected: `none`

#### Run-Level / Command-Level Metrics

| run | exp | status | fidelity | exit | runtime_sec | command | cwd | stdout | stderr | reason_codes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| exp_01 | exp_01 | failed | trend | 0 | 593.50 | /home/yb2636_columbia_edu/miniconda3/condabin/mamba run -n 07_Symplectic_ODE_Net_executor python experiment-single-force/train.py --structure --verbose --seed 0 | . | True | True | COMMAND_NOT_OBSERVED, UNTRACEABLE_METRICS, FULL_WITH_OVERRIDE_ARGS |
| exp_02 | exp_02 | failed | trend | 0 | 2975.21 | /home/yb2636_columbia_edu/miniconda3/condabin/mamba run -n 07_Symplectic_ODE_Net_executor python experiment-single-embed/train.py --structure --verbose --num_points 1 --seed 0 | . | True | True | COMMAND_NOT_OBSERVED, UNTRACEABLE_METRICS, FULL_WITH_OVERRIDE_ARGS |
| exp_03 | exp_03 | failed | trend | 0 | 230.85 | /home/yb2636_columbia_edu/miniconda3/condabin/mamba run -n 07_Symplectic_ODE_Net_executor python experiment-single-force/train.py --structure --verbose --seed 0 | . | True | True | COMMAND_NOT_OBSERVED, UNTRACEABLE_METRICS, FULL_WITH_OVERRIDE_ARGS |
| exp_04 | exp_04 | failed | trend | 0 | 311.17 | /home/yb2636_columbia_edu/miniconda3/condabin/mamba run -n 07_Symplectic_ODE_Net_executor python experiment-single-embed/train.py --structure --verbose --seed 0 | . | True | True | COMMAND_NOT_OBSERVED, UNTRACEABLE_METRICS, FULL_WITH_OVERRIDE_ARGS |
| exp_05 | exp_05 | failed | trend | 0 | 590.61 | /home/yb2636_columbia_edu/miniconda3/condabin/mamba run -n 07_Symplectic_ODE_Net_executor python experiment-cartpole-embed/train.py --structure --verbose --seed 0 | . | True | True | COMMAND_NOT_OBSERVED, UNTRACEABLE_METRICS, FULL_WITH_OVERRIDE_ARGS |
| exp_06 | exp_06 | partial | trend | 0 | 4.31 | /home/yb2636_columbia_edu/miniconda3/condabin/mamba run -n 07_Symplectic_ODE_Net_executor python experiment-double-embed/train.py --structure --verbose --seed 0 | . | True | True | COMMAND_NOT_OBSERVED, FULL_WITH_OVERRIDE_ARGS |
| exp_07 | exp_07 | failed | smoke | 1 | NA |  | . | True | True | EXPERIMENT_RESULT_MISSING |

### Evidence Alignment, Verdict, Failure Metrics

| metric | value |
| --- | --- |
| claims_evaluated / not_evaluated | 78 / 0 |
| evaluated_claim_rate | 100.00 |
| executable / non_executable claims | 78 / 0 |
| claims_with_observed_metric / without | 0 / 78 |
| metric_recovery_rate | 0.000 |
| metric_parser_success / failed | 1 / 78 |
| reported/observed comparisons | 0 |
| within_tolerance / outside_tolerance | 0 / 0 |
| claim_to_evidence_mapped / unmapped | 0 / 78 |
| supported / partial / not_supported / inconclusive | 0 / 0 / 0 / 78 |
| main_verdict_per_paper | INCONCLUSIVE |
| headline evaluated/supported/partial/not/inconclusive | 12 / 0 / 0 / 0 / 12 |
| headline_metric_recovery_rate | 100.00 |
| executed_but_no_headline_evidence | False |
| executed_but_metric_missing | False |
| executed_but_claim_inconclusive | False |
| failed_but_diagnostic_evidence_available | True |
| failure_modes | missing_dataset:1, dependency_failure:1, entrypoint_ambiguity:1, metric_not_found:1, metric_mismatch:1, compute_limit:1 |
| main_inconclusive_reason | Configuration claim requires direct code/config evidence; execution metrics alone do not verify the paper setup. |
| score_total / raw | 65.0 / 53.2 |
| ECR | False |
| posthoc evidence_tier | TREND_EVIDENCE |
| execution_outcome_counts | none:6, TREND_SUPPORTED:1 |
| reproduced/skipped figures | 6 / 4 |
| logs_scanned | 17 |

Top reason codes: LLM_REASON_PROVIDED:159, VISUAL_ENRICHED:123, LLM_TABLE_EXTRACTED:107, ALIGNMENT_AMBIGUOUS:59, VISUAL_TABLE_EXTRACTED:52, CONFIG_CLAIM:19, NO_DIRECT_CONFIG_EVIDENCE:19, COMMAND_NOT_OBSERVED:18, FULL_WITH_OVERRIDE_ARGS:18, TABLE_EXPANDED:16, CAPTION_METRIC_MATRIX:16, MEAN_STD_TARGET_NORMALIZED:12


<!-- completed 07_Symplectic_ODE_Net -->

## 08_E_n_Equivariant_Graph_Neural_Networks_EGNN

| field | value |
| --- | --- |
| paper_id | 08 |
| title | Victor Garcia Satorras 1 Emiel Hoogeboom 1 Max Welling 1 |
| arxiv_id | 1906.04015 |
| venue | NA |
| year | 2013 |
| model_family | E n Equivariant Graph Neural Networks EGNN |
| repo_url | https://github.com/vgsatorras/egnn |
| repo_available / cloned | True / True |
| expected_entry_point | python-file:qm9/dataset.py |
| documented_entry_point_exists | False |
| compute_requirement | GPU/CUDA likely required or supported |
| dataset_requirement | Community Small and Erdos&Renyi, Community graphs and Erdos-Renyi graphs, N-body system, N-body system (synthetic charged particles), QM9 |

### Claim Extraction Metrics

| metric | value |
| --- | --- |
| total_claims_extracted | 120 |
| code_verifiable_claims | 112 |
| non_code_verifiable_claims | 8 |
| pct_code_verifiable_claims | 93.30 |
| claims_from_table / figure / text | 120 / 0 / 0 |
| main_claim_source | table |
| metric_contracts_generated | 6 |
| claims_with_metric_contract | 118 |
| pct_claims_with_metric_contract | 98.30 |
| claims_without_metric_contract | 2 |
| reported_metric_names | % Error, Binary Cross Entropy, Forward time, Mean Absolute Error, f1, mse |
| top_reported_metrics | f1:75, mse:12, % Error:8, Forward time:6, Binary Cross Entropy:6, Mean Absolute Error:5 |
| reported_dataset | Community Small and Erdos&Renyi, Community graphs and Erdos-Renyi graphs, N-body system, N-body system (synthetic charged particles), QM9 |
| reported_model | none |
| reported_method | none |
| reported_baseline | none |
| claim granularity | headline=16, secondary=104, comparison=0, ablation=0, efficiency=6, qualitative=0, theoretical=0 |

### Task, Entrypoint, Dataset, Command Metrics

| metric | value |
| --- | --- |
| tasks_generated | 12 |
| tasks_per_paper | 12 |
| tasks_per_claim | 0.100 |
| claims_with_candidate_task | 120 |
| claims_without_candidate_task | 0 |
| task_generation_coverage | 100.00 |
| candidate_entry_points_found | 12 |
| entry_points_per_repo | 12 |
| claim_to_entrypoint_mapped | 120 |
| claim_to_entrypoint_unmapped | 0 |
| entrypoint_mapping_coverage | 100.00 |
| entrypoint_ambiguity_count | 0 |
| datasets_identified | Community Small and Erdos&Renyi, Community graphs and Erdos-Renyi graphs, N-body system, N-body system (synthetic charged particles), QM9 |
| dataset_available | 1 |
| dataset_missing | 0 |
| dataset_mapping_coverage | 125.00 |
| commands_generated | 12 |
| commands_with_required_args | 12 |
| commands_missing_required_args | 0 |
| commands_matching_readme | 0 |
| commands_inferred_by_agent | 12 |
| command_confidence_score | 0.930 |

### Environment, Repair, Execution Metrics

| metric | value |
| --- | --- |
| env_setup_success / failed / repaired | True / False / False |
| requirements/environment/setup.py/pyproject found | False / False / False / False |
| dependency_install_success / failed | True / False |
| dependency_conflict_count | 0 |
| missing_package_count | 0 |
| obsolete_package_count | 0 |
| repair_attempts | 1 |
| repos_requiring_repair | False |
| repair_success / failed | False / False |
| repair_types | compute_limit, data_download_failure, dependency_failure, entrypoint_ambiguity, metric_mismatch, metric_not_found |
| env_setup_runtime_min | 0.110 |
| phase2_elapsed_min | 65.47 |
| runs_attempted / ok / partial / failed / timeout | 4 / 1 / 0 / 2 / 0 |
| run_success_rate | 25.00 |
| runtime_min | 32.08 |
| standard_execution_package_complete | True |
| artifact_count | 53 |
| result/log/checkpoint/figure files | 15 / 17 / 0 / 12 |

Dependency manifests detected: `none`

#### Run-Level / Command-Level Metrics

| run | exp | status | fidelity | exit | runtime_sec | command | cwd | stdout | stderr | reason_codes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| exp_01 | exp_01 | skipped | artifact | 0 | 0.000 | N/A | . | True | True | not_reproducible_literature_table, SYNTHETIC_RUN_LOG_FROM_EXECUTOR_RESULTS |
| exp_02 | exp_02 | ok | smoke | 0 | 1850.00 | python n_body_system/dataset/generate_dataset.py --num-train 10000 --seed 43 --sufix small && python main_nbody.py --epochs 10 --model egnn_vel --max_training_samples 3000 --lr 5e-4 | . | True | True | smoke_run_success, loss_improvement_observed, metrics_confirmed, SYNTHETIC_RUN_LOG_FROM_EXECUTOR_RESULTS |
| exp_03 | exp_03 | failed | trend | 1 | 15.00 | python main_qm9.py --epochs 10 --property homo | . | True | True | data_download_failure, external_data_unavailable, SYNTHETIC_RUN_LOG_FROM_EXECUTOR_RESULTS |
| exp_04 | exp_04 | failed | trend | 1 | 60.00 | python main_ae.py --epochs 10 --model ae_egnn --dataset community_ours | . | True | True | code_version_incompatible, cannot_fix_without_modifying_code, SYNTHETIC_RUN_LOG_FROM_EXECUTOR_RESULTS |

### Evidence Alignment, Verdict, Failure Metrics

| metric | value |
| --- | --- |
| claims_evaluated / not_evaluated | 120 / 0 |
| evaluated_claim_rate | 100.00 |
| executable / non_executable claims | 112 / 8 |
| claims_with_observed_metric / without | 12 / 108 |
| metric_recovery_rate | 10.00 |
| metric_parser_success / failed | 3 / 108 |
| reported/observed comparisons | 12 |
| within_tolerance / outside_tolerance | 2 / 10 |
| claim_to_evidence_mapped / unmapped | 12 / 108 |
| supported / partial / not_supported / inconclusive | 2 / 0 / 10 / 108 |
| main_verdict_per_paper | INCONCLUSIVE |
| headline evaluated/supported/partial/not/inconclusive | 16 / 0 / 0 / 2 / 14 |
| headline_metric_recovery_rate | 100.00 |
| executed_but_no_headline_evidence | False |
| executed_but_metric_missing | False |
| executed_but_claim_inconclusive | True |
| failed_but_diagnostic_evidence_available | False |
| failure_modes | dependency_failure:1, entrypoint_ambiguity:1, metric_not_found:1, metric_mismatch:1, compute_limit:1, data_download_failure:1 |
| main_inconclusive_reason | Experiment run exists but metric `f1` could not be aligned for this claim. |
| score_total / raw | 50.0 / 41.4 |
| ECR | False |
| posthoc evidence_tier | EXECUTABLE_OR_SMOKE_EVIDENCE |
| execution_outcome_counts | none:3, EXECUTABLE:1 |
| reproduced/skipped figures | 6 / 2 |
| logs_scanned | 16 |

Top reason codes: LLM_REASON_PROVIDED:187, TABLE_EXPANDED:139, VISUAL_TABLE_EXTRACTED:137, CAPTION_METRIC_MATRIX:127, VISUAL_ENRICHED:125, ALIGNMENT_AMBIGUOUS:100, LLM_TABLE_EXTRACTED:50, ERROR_LOG:16, PHASE2_PACKAGE_METRIC:15, MATCHED_METRIC:12, REDUCED_FIDELITY_EVIDENCE:12, OUTSIDE_TOLERANCE:10


<!-- completed 08_E_n_Equivariant_Graph_Neural_Networks_EGNN -->

## 09_Fourier_Neural_Operator_FNO

| field | value |
| --- | --- |
| paper_id | 09 |
| title | FOURIER NEURAL OPERATOR FORPARAMETRIC PARTIAL DIFFERENTIAL EQUATIONS |
| arxiv_id | 1904.05417 |
| venue | NA |
| year | 2019 |
| model_family | Fourier Neural Operator FNO |
| repo_url | https://github.com/neuraloperator/neuraloperator |
| repo_available / cloned | True / True |
| expected_entry_point | python-file:scripts/train_burgers_rno.py |
| documented_entry_point_exists | False |
| compute_requirement | GPU/CUDA likely required or supported |
| dataset_requirement | 1-d Burgers’ equation, 2-d Darcy Flow, Navier-Stokes |

### Claim Extraction Metrics

| metric | value |
| --- | --- |
| total_claims_extracted | 102 |
| code_verifiable_claims | 95 |
| non_code_verifiable_claims | 7 |
| pct_code_verifiable_claims | 93.10 |
| claims_from_table / figure / text | 79 / 0 / 23 |
| main_claim_source | table |
| metric_contracts_generated | 5 |
| claims_with_metric_contract | 58 |
| pct_claims_with_metric_contract | 56.90 |
| claims_without_metric_contract | 44 |
| reported_metric_names | best relative error, error, error rate, relative error, time per epoch |
| top_reported_metrics | relative error:20, error rate:15, error:13, time per epoch:5, best relative error:4 |
| reported_dataset | 1-d Burgers’ equation, 2-d Darcy Flow, Navier-Stokes |
| reported_model | none |
| reported_method | none |
| reported_baseline | none |
| claim granularity | headline=8, secondary=94, comparison=0, ablation=0, efficiency=18, qualitative=0, theoretical=0 |

### Task, Entrypoint, Dataset, Command Metrics

| metric | value |
| --- | --- |
| tasks_generated | 2 |
| tasks_per_paper | 2 |
| tasks_per_claim | 0.020 |
| claims_with_candidate_task | 102 |
| claims_without_candidate_task | 0 |
| task_generation_coverage | 100.00 |
| candidate_entry_points_found | 2 |
| entry_points_per_repo | 2 |
| claim_to_entrypoint_mapped | 102 |
| claim_to_entrypoint_unmapped | 0 |
| entrypoint_mapping_coverage | 100.00 |
| entrypoint_ambiguity_count | 0 |
| datasets_identified | 1-d Burgers’ equation, 2-d Darcy Flow, Navier-Stokes |
| dataset_available | 3 |
| dataset_missing | 0 |
| dataset_mapping_coverage | 50.00 |
| commands_generated | 2 |
| commands_with_required_args | 2 |
| commands_missing_required_args | 0 |
| commands_matching_readme | 0 |
| commands_inferred_by_agent | 2 |
| command_confidence_score | 0.930 |

### Environment, Repair, Execution Metrics

| metric | value |
| --- | --- |
| env_setup_success / failed / repaired | True / False / False |
| requirements/environment/setup.py/pyproject found | True / False / False / True |
| dependency_install_success / failed | True / False |
| dependency_conflict_count | 0 |
| missing_package_count | 0 |
| obsolete_package_count | 0 |
| repair_attempts | 1 |
| repos_requiring_repair | False |
| repair_success / failed | False / False |
| repair_types | compute_limit, dependency_failure, entrypoint_ambiguity, metric_not_found |
| env_setup_runtime_min | 3.01 |
| phase2_elapsed_min | 60.25 |
| runs_attempted / ok / partial / failed / timeout | 6 / 3 / 0 / 0 / 0 |
| run_success_rate | 50.00 |
| runtime_min | 28.75 |
| standard_execution_package_complete | True |
| artifact_count | 70 |
| result/log/checkpoint/figure files | 15 / 33 / 0 / 6 |

Dependency manifests detected: `doc/Makefile, doc/requirements_doc.txt, pyproject.toml, requirements.txt, requirements_dev.txt`

#### Run-Level / Command-Level Metrics

| run | exp | status | fidelity | exit | runtime_sec | command | cwd | stdout | stderr | reason_codes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| exp_01 | exp_01 | ok | trend | 0 | 1581.50 |  | . | True | True | none |
| exp_02 | exp_02 | ok | trend | 0 | 58.33 |  | . | True | True | none |
| exp_03 | exp_03 | ok | trend | 0 | 85.43 |  | . | True | True | none |
| exp_04 | exp_04 | skipped | smoke | 0 | 0.000 |  | . | True | True | SYNTHETIC_RUN_LOG_FROM_EXECUTOR_RESULTS |
| exp_05 | exp_05 | skipped | smoke | 0 | 0.000 |  | . | True | True | SYNTHETIC_RUN_LOG_FROM_EXECUTOR_RESULTS |
| table_2 | table_2 | skipped | smoke | 0 | 0.000 |  | . | True | True | SYNTHETIC_RUN_LOG_FROM_EXECUTOR_RESULTS |

### Evidence Alignment, Verdict, Failure Metrics

| metric | value |
| --- | --- |
| claims_evaluated / not_evaluated | 102 / 0 |
| evaluated_claim_rate | 100.00 |
| executable / non_executable claims | 95 / 7 |
| claims_with_observed_metric / without | 0 / 102 |
| metric_recovery_rate | 0.000 |
| metric_parser_success / failed | 1 / 102 |
| reported/observed comparisons | 0 |
| within_tolerance / outside_tolerance | 0 / 0 |
| claim_to_evidence_mapped / unmapped | 0 / 102 |
| supported / partial / not_supported / inconclusive | 0 / 0 / 0 / 102 |
| main_verdict_per_paper | INCONCLUSIVE |
| headline evaluated/supported/partial/not/inconclusive | 8 / 0 / 0 / 0 / 8 |
| headline_metric_recovery_rate | 100.00 |
| executed_but_no_headline_evidence | False |
| executed_but_metric_missing | True |
| executed_but_claim_inconclusive | True |
| failed_but_diagnostic_evidence_available | False |
| failure_modes | dependency_failure:1, entrypoint_ambiguity:1, metric_not_found:1, compute_limit:1 |
| main_inconclusive_reason | Configuration claim requires direct code/config evidence; execution metrics alone do not verify the paper setup. |
| score_total / raw | 65.0 / 52.3 |
| ECR | False |
| posthoc evidence_tier | TREND_EVIDENCE |
| execution_outcome_counts | TREND_SUPPORTED:3, none:3 |
| reproduced/skipped figures | 4 / 6 |
| logs_scanned | 32 |

Top reason codes: LLM_REASON_PROVIDED:179, VISUAL_ENRICHED:113, LLM_TABLE_EXTRACTED:97, VISUAL_TABLE_EXTRACTED:82, ALIGNMENT_AMBIGUOUS:57, CONFIG_CLAIM:38, NO_DIRECT_CONFIG_EVIDENCE:38, PARSE_LOW_CONFIDENCE:19, MISSING_RECORDS:7, SYNTHETIC_RUN_LOG_FROM_EXECUTOR_RESULTS:6, LLM_SKIP_OVERRIDDEN_BY_DETERMINISTIC_FALLBACK:6, DETERMINISTIC_METRIC_TABLE_FALLBACK:6


<!-- completed 09_Fourier_Neural_Operator_FNO -->

## 10_Your_Classifier_is_Secretly_an_EBM_JEM

| field | value |
| --- | --- |
| paper_id | 10 |
| title | YOUR CLASSIFIER IS SECRETLY AN ENERGY BASEDMODEL AND YOU SHOULD TREAT IT LIKE ONE |
| arxiv_id | 1707.07397 |
| venue | NA |
| year | 2006 |
| model_family | Your Classifier is Secretly an EBM JEM |
| repo_url | https://github.com/wgrathwohl/JEM |
| repo_available / cloned | True / True |
| expected_entry_point | readme-python:train_wrn_ebm.py |
| documented_entry_point_exists | True |
| compute_requirement | GPU/CUDA likely required or supported |
| dataset_requirement | CIFAR10, CIFAR100 |

### Claim Extraction Metrics

| metric | value |
| --- | --- |
| total_claims_extracted | 120 |
| code_verifiable_claims | 110 |
| non_code_verifiable_claims | 10 |
| pct_code_verifiable_claims | 91.70 |
| claims_from_table / figure / text | 106 / 0 / 14 |
| main_claim_source | table |
| metric_contracts_generated | 10 |
| claims_with_metric_contract | 110 |
| pct_claims_with_metric_contract | 91.70 |
| claims_without_metric_contract | 10 |
| reported_metric_names | AUROC, FID, FID (D&M), FID (H), FID (from paper), Inception Score, Inception Score (B&S), Inception Score (D&M), Inception Score (from paper), accuracy |
| top_reported_metrics | AUROC:58, accuracy:22, Inception Score:9, Inception Score (D&M):5, FID (H):4, FID (from paper):4, FID:3, Inception Score (B&S):2 |
| reported_dataset | CIFAR10, CIFAR100 |
| reported_model | none |
| reported_method | none |
| reported_baseline | none |
| claim granularity | headline=17, secondary=103, comparison=0, ablation=2, efficiency=0, qualitative=0, theoretical=0 |

### Task, Entrypoint, Dataset, Command Metrics

| metric | value |
| --- | --- |
| tasks_generated | 6 |
| tasks_per_paper | 6 |
| tasks_per_claim | 0.050 |
| claims_with_candidate_task | 120 |
| claims_without_candidate_task | 0 |
| task_generation_coverage | 100.00 |
| candidate_entry_points_found | 6 |
| entry_points_per_repo | 6 |
| claim_to_entrypoint_mapped | 120 |
| claim_to_entrypoint_unmapped | 0 |
| entrypoint_mapping_coverage | 100.00 |
| entrypoint_ambiguity_count | 0 |
| datasets_identified | CIFAR10, CIFAR100 |
| dataset_available | 2 |
| dataset_missing | 0 |
| dataset_mapping_coverage | 25.00 |
| commands_generated | 6 |
| commands_with_required_args | 6 |
| commands_missing_required_args | 0 |
| commands_matching_readme | 3 |
| commands_inferred_by_agent | 3 |
| command_confidence_score | 0.908 |

### Environment, Repair, Execution Metrics

| metric | value |
| --- | --- |
| env_setup_success / failed / repaired | True / False / False |
| requirements/environment/setup.py/pyproject found | False / False / False / False |
| dependency_install_success / failed | True / False |
| dependency_conflict_count | 0 |
| missing_package_count | 0 |
| obsolete_package_count | 0 |
| repair_attempts | 1 |
| repos_requiring_repair | False |
| repair_success / failed | False / False |
| repair_types | compute_limit, dependency_failure, entrypoint_ambiguity, metric_mismatch, metric_not_found, missing_dataset, ...(+2) |
| env_setup_runtime_min | 0.090 |
| phase2_elapsed_min | 28.80 |
| runs_attempted / ok / partial / failed / timeout | 8 / 2 / 4 / 0 / 1 |
| run_success_rate | 25.00 |
| runtime_min | 11.47 |
| standard_execution_package_complete | True |
| artifact_count | 92 |
| result/log/checkpoint/figure files | 15 / 40 / 1 / 24 |

Dependency manifests detected: `none`

#### Run-Level / Command-Level Metrics

| run | exp | status | fidelity | exit | runtime_sec | command | cwd | stdout | stderr | reason_codes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| exp_01 | exp_01 | ok | artifact | 0 | 18.00 | eval_wrn_ebm.py --eval test_clf --dataset cifar_test | . | True | True | ARTIFACT_AVAILABLE, SYNTHETIC_RUN_LOG_FROM_EXECUTOR_RESULTS |
| exp_02 | exp_02 | ok | smoke | 0 | 180.00 | eval_wrn_ebm.py --eval OOD --ood_dataset <various> --score_fn <various> | . | True | True | OOD_EVAL_COMPLETE, SYNTHETIC_RUN_LOG_FROM_EXECUTOR_RESULTS |
| exp_03 | exp_03 | partial | smoke | 1 | 120.00 | eval_wrn_ebm.py --eval uncond_samples / cond_samples | . | True | True | REPO_MISSING_EVALUATION_CODE, PARTIAL_SUCCESS, SYNTHETIC_RUN_LOG_FROM_EXECUTOR_RESULTS |
| exp_04 | exp_04 | partial | smoke | 0 | 60.00 | eval_wrn_ebm.py --eval OOD --ood_dataset <synthetic> | . | True | True | REPO_LIMITED_OOD, PROXY_EVALUATION, SYNTHETIC_RUN_LOG_FROM_EXECUTOR_RESULTS |
| exp_05 | exp_05 | skipped | full | 0 | 0.000 | Comparison of FID/Inception Score implementations | . | True | True | REPO_MISSING_EVALUATION_CODE, SYNTHETIC_RUN_LOG_FROM_EXECUTOR_RESULTS |
| exp_06 | exp_06 | partial | smoke | 0 | 250.00 | train_wrn_ebm.py --dataset cifar100 | . | True | True | TRAINING_INSTABILITY, REDUCED_FIDELITY, SYNTHETIC_RUN_LOG_FROM_EXECUTOR_RESULTS |
| exp_07 | exp_07 | skipped | full | 0 | 0.000 | N/A | . | True | True | UNCLEAR_REQUIREMENTS, SYNTHETIC_RUN_LOG_FROM_EXECUTOR_RESULTS |
| exp_08 | exp_08 | partial | smoke | 0 | 60.00 | eval_wrn_ebm.py --eval logp_hist | . | True | True | QUALITATIVE_ONLY, SYNTHETIC_RUN_LOG_FROM_EXECUTOR_RESULTS |

### Evidence Alignment, Verdict, Failure Metrics

| metric | value |
| --- | --- |
| claims_evaluated / not_evaluated | 120 / 0 |
| evaluated_claim_rate | 100.00 |
| executable / non_executable claims | 110 / 10 |
| claims_with_observed_metric / without | 7 / 113 |
| metric_recovery_rate | 5.80 |
| metric_parser_success / failed | 1 / 113 |
| reported/observed comparisons | 7 |
| within_tolerance / outside_tolerance | 0 / 7 |
| claim_to_evidence_mapped / unmapped | 7 / 113 |
| supported / partial / not_supported / inconclusive | 0 / 0 / 7 / 113 |
| main_verdict_per_paper | INCONCLUSIVE |
| headline evaluated/supported/partial/not/inconclusive | 17 / 0 / 0 / 0 / 17 |
| headline_metric_recovery_rate | 100.00 |
| executed_but_no_headline_evidence | False |
| executed_but_metric_missing | False |
| executed_but_claim_inconclusive | True |
| failed_but_diagnostic_evidence_available | False |
| failure_modes | missing_dataset:1, dependency_failure:1, entrypoint_ambiguity:1, metric_not_found:1, metric_mismatch:1, compute_limit:1, repo_clone_failure:1, undocumented_hyperparameter:1 |
| main_inconclusive_reason | Experiment run exists but metric `AUROC` could not be aligned for this claim. |
| score_total / raw | 65.0 / 48.5 |
| ECR | False |
| posthoc evidence_tier | TREND_EVIDENCE |
| execution_outcome_counts | TREND_SUPPORTED:1, EXECUTABLE:5, none:2 |
| reproduced/skipped figures | 12 / 7 |
| logs_scanned | 39 |

Top reason codes: LLM_REASON_PROVIDED:254, VISUAL_ENRICHED:174, VISUAL_TABLE_EXTRACTED:128, LLM_TABLE_EXTRACTED:126, ALIGNMENT_AMBIGUOUS:103, PARSE_LOW_CONFIDENCE:31, LLM_SKIP_OVERRIDDEN_BY_DETERMINISTIC_FALLBACK:20, DETERMINISTIC_METRIC_TABLE_FALLBACK:20, SYNTHETIC_RUN_LOG_FROM_EXECUTOR_RESULTS:16, README_VERIFIED_COMMAND:15, TABLE_EXPANDED:14, LLM_CODEGEN_RENDERED:11


<!-- completed 10_Your_Classifier_is_Secretly_an_EBM_JEM -->

## 11_MACE_Higher_Order_Equivariant_MPNNs copy

| field | value |
| --- | --- |
| paper_id | 11 |
| title | MACE: Higher Order Equivariant Message Passing Neural Networks for Fast and Accurate Force Fields |
| arxiv_id | 1803.01588 |
| venue | NA |
| year | NA |
| model_family | MACE Higher Order Equivariant MPNNs |
| repo_url | https://github.com/ACEsuit/mace |
| repo_available / cloned | True / True |
| expected_entry_point | python-file:scripts/run_train.py |
| documented_entry_point_exists | False |
| compute_requirement | GPU/CUDA likely required or supported |
| dataset_requirement | 3BPA, Acetylacetone, rMD17 |

### Claim Extraction Metrics

| metric | value |
| --- | --- |
| total_claims_extracted | 120 |
| code_verifiable_claims | 117 |
| non_code_verifiable_claims | 3 |
| pct_code_verifiable_claims | 97.50 |
| claims_from_table / figure / text | 89 / 0 / 31 |
| main_claim_source | table |
| metric_contracts_generated | 6 |
| claims_with_metric_contract | 111 |
| pct_claims_with_metric_contract | 92.50 |
| claims_without_metric_contract | 9 |
| reported_metric_names | energy MAE, force MAE, latency, mae, mse, training time |
| top_reported_metrics | mae:51, mse:25, energy MAE:10, force MAE:10, latency:2, training time:1 |
| reported_dataset | 3BPA, Acetylacetone, rMD17 |
| reported_model | none |
| reported_method | none |
| reported_baseline | none |
| claim granularity | headline=43, secondary=77, comparison=0, ablation=0, efficiency=3, qualitative=0, theoretical=0 |

### Task, Entrypoint, Dataset, Command Metrics

| metric | value |
| --- | --- |
| tasks_generated | 12 |
| tasks_per_paper | 12 |
| tasks_per_claim | 0.100 |
| claims_with_candidate_task | 120 |
| claims_without_candidate_task | 0 |
| task_generation_coverage | 100.00 |
| candidate_entry_points_found | 12 |
| entry_points_per_repo | 12 |
| claim_to_entrypoint_mapped | 120 |
| claim_to_entrypoint_unmapped | 0 |
| entrypoint_mapping_coverage | 100.00 |
| entrypoint_ambiguity_count | 0 |
| datasets_identified | 3BPA, Acetylacetone, rMD17 |
| dataset_available | 0 |
| dataset_missing | 0 |
| dataset_mapping_coverage | 37.50 |
| commands_generated | 12 |
| commands_with_required_args | 12 |
| commands_missing_required_args | 0 |
| commands_matching_readme | 0 |
| commands_inferred_by_agent | 12 |
| command_confidence_score | 0.930 |

### Environment, Repair, Execution Metrics

| metric | value |
| --- | --- |
| env_setup_success / failed / repaired | True / False / False |
| requirements/environment/setup.py/pyproject found | False / False / False / True |
| dependency_install_success / failed | True / False |
| dependency_conflict_count | 0 |
| missing_package_count | 0 |
| obsolete_package_count | 0 |
| repair_attempts | 1 |
| repos_requiring_repair | False |
| repair_success / failed | False / False |
| repair_types | compute_limit, data_download_failure, dependency_failure, entrypoint_ambiguity, metric_mismatch, metric_not_found, ...(+1) |
| env_setup_runtime_min | 1.66 |
| phase2_elapsed_min | 14.45 |
| runs_attempted / ok / partial / failed / timeout | 8 / 1 / 0 / 0 / 0 |
| run_success_rate | 12.50 |
| runtime_min | 5.94 |
| standard_execution_package_complete | True |
| artifact_count | 62 |
| result/log/checkpoint/figure files | 15 / 51 / 0 / 6 |

Dependency manifests detected: `pyproject.toml, setup.cfg`

#### Run-Level / Command-Level Metrics

| run | exp | status | fidelity | exit | runtime_sec | command | cwd | stdout | stderr | reason_codes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| exp_01 | exp_01 | ok | trend | 0 | 356.20 | python run_exp01_learning_curve.py | . | True | True | SYNTHETIC_DATA_USED, SMOKE_TEST_PASSED, TREND_TEST_PASSED, CODE_VERIFIED, ...(+2) |
| exp_02 | exp_02 | skipped | full | 0 | 0.000 | Documentation only - configuration study | . | True | True | NOT_A_TRAINING_EXPERIMENT, CONFIGURATION_ONLY, NO_METRICS_DEFINED, SYNTHETIC_RUN_LOG_FROM_EXECUTOR_RESULTS, ...(+1) |
| exp_03 | exp_03 | skipped | full | 0 | 0.000 | mace_run_train | . | True | True | DATASET_UNAVAILABLE, NO_DOWNLOAD_INSTRUCTIONS, EXTERNAL_DATA_REQUIRED, CODE_VERIFIED_ON_SYNTHETIC, ...(+2) |
| exp_04 | exp_04 | skipped | full | 0 | 0.000 | mace_run_train | . | True | True | INCOMPLETE_SPECIFICATION, MISSING_DATASET, MISSING_TARGET_METRIC, MISSING_MODEL_CONFIG, ...(+2) |
| exp_05 | exp_05 | skipped | full | 0 | 0.000 | mace_run_train | . | True | True | DATASET_UNAVAILABLE, NO_DOWNLOAD_INSTRUCTIONS, PRIMARY_BENCHMARK_TABLE, EXTERNAL_DATA_REQUIRED, ...(+2) |
| exp_06 | exp_06 | skipped | full | 0 | 0.000 | mace_run_train | . | True | True | DATASET_UNAVAILABLE, LOW_DATA_REGIME, SAME_DATASET_AS_EXP_05, CODE_VERIFIED_ON_SMALL_DATA, ...(+2) |
| exp_07 | exp_07 | skipped | full | 0 | 0.000 | mace_run_train | . | True | True | DATASET_UNAVAILABLE, GENERALIZATION_BENCHMARK, TEMPERATURE_EXTRAPOLATION, NO_DOWNLOAD_INSTRUCTIONS, ...(+2) |
| exp_08 | exp_08 | skipped | full | 0 | 0.000 | mace_run_train | . | True | True | DATASET_UNAVAILABLE, COMPARATIVE_BENCHMARK, MISSING_BASELINE_MODELS, NO_DOWNLOAD_INSTRUCTIONS, ...(+2) |

### Evidence Alignment, Verdict, Failure Metrics

| metric | value |
| --- | --- |
| claims_evaluated / not_evaluated | 120 / 0 |
| evaluated_claim_rate | 100.00 |
| executable / non_executable claims | 117 / 3 |
| claims_with_observed_metric / without | 0 / 120 |
| metric_recovery_rate | 0.000 |
| metric_parser_success / failed | 1 / 120 |
| reported/observed comparisons | 0 |
| within_tolerance / outside_tolerance | 0 / 0 |
| claim_to_evidence_mapped / unmapped | 0 / 120 |
| supported / partial / not_supported / inconclusive | 0 / 0 / 0 / 120 |
| main_verdict_per_paper | INCONCLUSIVE |
| headline evaluated/supported/partial/not/inconclusive | 43 / 0 / 0 / 0 / 43 |
| headline_metric_recovery_rate | 100.00 |
| executed_but_no_headline_evidence | False |
| executed_but_metric_missing | True |
| executed_but_claim_inconclusive | True |
| failed_but_diagnostic_evidence_available | False |
| failure_modes | missing_dataset:1, dependency_failure:1, entrypoint_ambiguity:1, metric_not_found:1, metric_mismatch:1, compute_limit:1, data_download_failure:1 |
| main_inconclusive_reason | Experiment run exists but metric `mae` could not be aligned for this claim. |
| score_total / raw | 65.0 / 47.3 |
| ECR | False |
| posthoc evidence_tier | TREND_EVIDENCE |
| execution_outcome_counts | TREND_SUPPORTED:1, none:7 |
| reproduced/skipped figures | 3 / 2 |
| logs_scanned | 28 |

Top reason codes: LLM_TABLE_EXTRACTED:177, LLM_REASON_PROVIDED:102, ALIGNMENT_AMBIGUOUS:99, VISUAL_TABLE_EXTRACTED:56, COMMAND_NOT_OBSERVED:31, PARSE_LOW_CONFIDENCE:23, DATASET_UNAVAILABLE:20, CONFIG_CLAIM:18, NO_DIRECT_CONFIG_EVIDENCE:18, NO_DOWNLOAD_INSTRUCTIONS:16, SYNTHETIC_RUN_LOG_FROM_EXECUTOR_RESULTS:14, EXTERNAL_DATA_REQUIRED:8


<!-- completed 11_MACE_Higher_Order_Equivariant_MPNNs copy -->

## 13_Liquid_TimeConstant_Networks_LTC

| field | value |
| --- | --- |
| paper_id | 13 |
| title | Liquid Time-constant Networks |
| arxiv_id | 1606.01540 |
| venue | NA |
| year | 2018 |
| model_family | Liquid TimeConstant Networks LTC |
| repo_url | https://github.com/raminmh/CfC |
| repo_available / cloned | True / True |
| expected_entry_point | readme-python:train_physio.py |
| documented_entry_point_exists | True |
| compute_requirement | CPU feasible in observed run |
| dataset_requirement | Activity recognition, Gesture, Half-Cheetah, Occupancy, Ozone, Person activity, ...(+3) |

### Claim Extraction Metrics

| metric | value |
| --- | --- |
| total_claims_extracted | 120 |
| code_verifiable_claims | 103 |
| non_code_verifiable_claims | 17 |
| pct_code_verifiable_claims | 85.80 |
| claims_from_table / figure / text | 119 / 0 / 1 |
| main_claim_source | table |
| metric_contracts_generated | 5 |
| claims_with_metric_contract | 111 |
| pct_claims_with_metric_contract | 92.50 |
| claims_without_metric_contract | 9 |
| reported_metric_names | accuracy, computational depth, f1, mse, squared error |
| top_reported_metrics | accuracy:74, computational depth:22, mse:11, squared error:2, f1:1 |
| reported_dataset | Activity recognition, Gesture, Half-Cheetah, Occupancy, Ozone, Person activity, Power, Sequential MNIST, ...(+1) |
| reported_model | none |
| reported_method | none |
| reported_baseline | none |
| claim granularity | headline=22, secondary=98, comparison=0, ablation=0, efficiency=5, qualitative=0, theoretical=3 |

### Task, Entrypoint, Dataset, Command Metrics

| metric | value |
| --- | --- |
| tasks_generated | 11 |
| tasks_per_paper | 11 |
| tasks_per_claim | 0.092 |
| claims_with_candidate_task | 120 |
| claims_without_candidate_task | 0 |
| task_generation_coverage | 100.00 |
| candidate_entry_points_found | 11 |
| entry_points_per_repo | 11 |
| claim_to_entrypoint_mapped | 120 |
| claim_to_entrypoint_unmapped | 0 |
| entrypoint_mapping_coverage | 100.00 |
| entrypoint_ambiguity_count | 0 |
| datasets_identified | Activity recognition, Gesture, Half-Cheetah, Occupancy, Ozone, Person activity, Power, Sequential MNIST, ...(+1) |
| dataset_available | 2 |
| dataset_missing | 0 |
| dataset_mapping_coverage | 75.00 |
| commands_generated | 11 |
| commands_with_required_args | 11 |
| commands_missing_required_args | 0 |
| commands_matching_readme | 1 |
| commands_inferred_by_agent | 10 |
| command_confidence_score | 0.890 |

### Environment, Repair, Execution Metrics

| metric | value |
| --- | --- |
| env_setup_success / failed / repaired | True / False / False |
| requirements/environment/setup.py/pyproject found | False / False / False / False |
| dependency_install_success / failed | True / False |
| dependency_conflict_count | 0 |
| missing_package_count | 0 |
| obsolete_package_count | 0 |
| repair_attempts | 1 |
| repos_requiring_repair | False |
| repair_success / failed | False / False |
| repair_types | compute_limit, dependency_failure, entrypoint_ambiguity, metric_mismatch, metric_not_found, missing_dataset, ...(+1) |
| env_setup_runtime_min | 0.090 |
| phase2_elapsed_min | 41.30 |
| runs_attempted / ok / partial / failed / timeout | 12 / 3 / 0 / 3 / 0 |
| run_success_rate | 25.00 |
| runtime_min | 45.50 |
| standard_execution_package_complete | True |
| artifact_count | 66 |
| result/log/checkpoint/figure files | 15 / 40 / 0 / 14 |

Dependency manifests detected: `none`

#### Run-Level / Command-Level Metrics

| run | exp | status | fidelity | exit | runtime_sec | command | cwd | stdout | stderr | reason_codes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| exp_01 | exp_01 | skipped | artifact | 0 | 0.000 | theoretical_comparison | . | True | True | THEORETICAL_EXPERIMENT, SYNTHETIC_RUN_LOG_FROM_EXECUTOR_RESULTS, COMMAND_NOT_OBSERVED |
| exp_02 | exp_02 | skipped | artifact | 0 | 0.000 | theoretical_comparison | . | True | True | THEORETICAL_EXPERIMENT, SYNTHETIC_RUN_LOG_FROM_EXECUTOR_RESULTS, COMMAND_NOT_OBSERVED |
| exp_03 | exp_03 | skipped | full | 0 | 0.000 | unknown_script | . | True | True | CODE_NOT_IN_REPO, SYNTHETIC_RUN_LOG_FROM_EXECUTOR_RESULTS, COMMAND_NOT_OBSERVED |
| exp_04 | exp_04 | skipped | full | 0 | 0.000 | unknown_script | . | True | True | CODE_NOT_IN_REPO, SYNTHETIC_RUN_LOG_FROM_EXECUTOR_RESULTS, COMMAND_NOT_OBSERVED |
| exp_05 | exp_05 | ok | full | 0 | 900.00 | python train_person_acitivity.py --model cfc | . | True | True | SYNTHETIC_RUN_LOG_FROM_EXECUTOR_RESULTS |
| exp_06 | exp_06 | failed | smoke | 1 | 30.00 | python train_et_smnist.py --epochs 200 | . | True | True | INCOMPATIBLE_NUMPY_API, INCOMPATIBLE_TENSORFLOW_API, NO_CODE_MODIFICATION_ALLOWED, SYNTHETIC_RUN_LOG_FROM_EXECUTOR_RESULTS, ...(+1) |
| exp_07 | exp_07 | failed | full | 1 | 0.000 | python traffic_with_cfc.py | . | True | True | INCOMPATIBLE_NUMPY_API, INCOMPATIBLE_TENSORFLOW_API, NO_CODE_MODIFICATION_ALLOWED, SYNTHETIC_RUN_LOG_FROM_EXECUTOR_RESULTS |
| exp_08 | exp_08 | skipped | full | 0 | 0.000 | unknown_script | . | True | True | CODE_NOT_IN_REPO, SYNTHETIC_RUN_LOG_FROM_EXECUTOR_RESULTS, COMMAND_NOT_OBSERVED |
| exp_09 | exp_09 | skipped | full | 0 | 0.000 | unknown_script | . | True | True | CODE_NOT_IN_REPO, SYNTHETIC_RUN_LOG_FROM_EXECUTOR_RESULTS, COMMAND_NOT_OBSERVED |
| exp_10 | exp_10 | ok | full | 0 | 900.00 | python train_person_acitivity.py --model cfc | . | True | True | SYNTHETIC_RUN_LOG_FROM_EXECUTOR_RESULTS |
| exp_11 | exp_11 | ok | full | 0 | 900.00 | python train_person_acitivity.py --model cfc | . | True | True | SYNTHETIC_RUN_LOG_FROM_EXECUTOR_RESULTS |
| exp_12 | exp_12 | failed | full | 1 | 0.000 | python train_walker.py | . | True | True | INCOMPATIBLE_NUMPY_API, INCOMPATIBLE_TENSORFLOW_API, NO_CODE_MODIFICATION_ALLOWED, SYNTHETIC_RUN_LOG_FROM_EXECUTOR_RESULTS, ...(+1) |

### Evidence Alignment, Verdict, Failure Metrics

| metric | value |
| --- | --- |
| claims_evaluated / not_evaluated | 120 / 0 |
| evaluated_claim_rate | 100.00 |
| executable / non_executable claims | 103 / 17 |
| claims_with_observed_metric / without | 0 / 120 |
| metric_recovery_rate | 0.000 |
| metric_parser_success / failed | 1 / 120 |
| reported/observed comparisons | 0 |
| within_tolerance / outside_tolerance | 0 / 0 |
| claim_to_evidence_mapped / unmapped | 0 / 120 |
| supported / partial / not_supported / inconclusive | 0 / 0 / 0 / 120 |
| main_verdict_per_paper | INCONCLUSIVE |
| headline evaluated/supported/partial/not/inconclusive | 22 / 0 / 0 / 0 / 22 |
| headline_metric_recovery_rate | 100.00 |
| executed_but_no_headline_evidence | False |
| executed_but_metric_missing | True |
| executed_but_claim_inconclusive | True |
| failed_but_diagnostic_evidence_available | False |
| failure_modes | missing_dataset:1, dependency_failure:1, entrypoint_ambiguity:1, metric_not_found:1, metric_mismatch:1, compute_limit:1, undocumented_hyperparameter:1 |
| main_inconclusive_reason | Experiment run exists but metric `accuracy` could not be aligned for this claim. |
| score_total / raw | 80.0 / 44.7 |
| ECR | False |
| posthoc evidence_tier | FULL_REPRODUCTION_EVIDENCE |
| execution_outcome_counts | none:9, FULLY_REPRODUCED:3 |
| reproduced/skipped figures | 7 / 5 |
| logs_scanned | 39 |

Top reason codes: VISUAL_ENRICHED:209, LLM_REASON_PROVIDED:195, ALIGNMENT_AMBIGUOUS:110, TABLE_EXPANDED:108, VISUAL_TABLE_EXTRACTED:108, LLM_TABLE_EXTRACTED:101, CAPTION_METRIC_MATRIX:72, MEAN_STD_TARGET_NORMALIZED:41, COMMAND_NOT_OBSERVED:32, PARSE_LOW_CONFIDENCE:27, SYNTHETIC_RUN_LOG_FROM_EXECUTOR_RESULTS:24, CODE_NOT_IN_REPO:16


<!-- completed 13_Liquid_TimeConstant_Networks_LTC -->

## 16_Neural_Controlled_Differential_Equations

| field | value |
| --- | --- |
| paper_id | 16 |
| title | Neural Controlled Differential Equations for Irregular Time Series |
| arxiv_id | 1710.04110 |
| venue | NA |
| year | NA |
| model_family | Neural Controlled Differential Equations |
| repo_url | https://github.com/patrick-kidger/NeuralCDE |
| repo_available / cloned | True / True |
| expected_entry_point | python-file:experiments/parse_results.py |
| documented_entry_point_exists | False |
| compute_requirement | GPU/CUDA likely required or supported |
| dataset_requirement | CharacterTrajectories, PhysioNet sepsis prediction, Speech Commands, Speech Commands v0.02 |

### Claim Extraction Metrics

| metric | value |
| --- | --- |
| total_claims_extracted | 120 |
| code_verifiable_claims | 120 |
| non_code_verifiable_claims | 0 |
| pct_code_verifiable_claims | 100.00 |
| claims_from_table / figure / text | 88 / 0 / 32 |
| main_claim_source | table |
| metric_contracts_generated | 3 |
| claims_with_metric_contract | 111 |
| pct_claims_with_metric_contract | 92.50 |
| claims_without_metric_contract | 9 |
| reported_metric_names | accuracy, auc, memory usage |
| top_reported_metrics | accuracy:52, auc:36, memory usage:21 |
| reported_dataset | CharacterTrajectories, PhysioNet sepsis prediction, Speech Commands, Speech Commands v0.02 |
| reported_model | none |
| reported_method | none |
| reported_baseline | none |
| claim granularity | headline=14, secondary=106, comparison=0, ablation=0, efficiency=26, qualitative=0, theoretical=0 |

### Task, Entrypoint, Dataset, Command Metrics

| metric | value |
| --- | --- |
| tasks_generated | 6 |
| tasks_per_paper | 6 |
| tasks_per_claim | 0.050 |
| claims_with_candidate_task | 120 |
| claims_without_candidate_task | 0 |
| task_generation_coverage | 100.00 |
| candidate_entry_points_found | 6 |
| entry_points_per_repo | 6 |
| claim_to_entrypoint_mapped | 120 |
| claim_to_entrypoint_unmapped | 0 |
| entrypoint_mapping_coverage | 100.00 |
| entrypoint_ambiguity_count | 0 |
| datasets_identified | CharacterTrajectories, PhysioNet sepsis prediction, Speech Commands, Speech Commands v0.02 |
| dataset_available | 0 |
| dataset_missing | 0 |
| dataset_mapping_coverage | 133.30 |
| commands_generated | 6 |
| commands_with_required_args | 6 |
| commands_missing_required_args | 0 |
| commands_matching_readme | 0 |
| commands_inferred_by_agent | 6 |
| command_confidence_score | 0.883 |

### Environment, Repair, Execution Metrics

| metric | value |
| --- | --- |
| env_setup_success / failed / repaired | True / False / False |
| requirements/environment/setup.py/pyproject found | False / False / True / False |
| dependency_install_success / failed | True / False |
| dependency_conflict_count | 0 |
| missing_package_count | 0 |
| obsolete_package_count | 0 |
| repair_attempts | 1 |
| repos_requiring_repair | False |
| repair_success / failed | False / False |
| repair_types | compute_limit, dependency_failure, entrypoint_ambiguity, metric_not_found, missing_dataset |
| env_setup_runtime_min | 1.31 |
| phase2_elapsed_min | 9.41 |
| runs_attempted / ok / partial / failed / timeout | 3 / 0 / 0 / 0 / 0 |
| run_success_rate | 0.000 |
| runtime_min | 0.010 |
| standard_execution_package_complete | True |
| artifact_count | 41 |
| result/log/checkpoint/figure files | 15 / 17 / 0 / 6 |

Dependency manifests detected: `setup.py`

#### Run-Level / Command-Level Metrics

| run | exp | status | fidelity | exit | runtime_sec | command | cwd | stdout | stderr | reason_codes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| exp_01 | exp_01 | skipped | full | 0 | 0.600 | import uea; uea.main(dataset_name='CharacterTrajectories', missing_rate=0.3, max_epochs=3, model_name='ncde', hidden_channels=32, hidden_hidden_channels=32, num_hidden_layers=3) | /home/yb2636_columbia_edu/DeepAudit_0.2/paper_with_code/16_Neural_Controlled_Differential_Equations/code/experiments | True | True | external_data_unavailable, url_404, no_smoke_evidence, SYNTHETIC_RUN_LOG_FROM_EXECUTOR_RESULTS, ...(+1) |
| exp_02 | exp_02 | skipped | full | 0 | 0.100 | import sepsis; sepsis.main(intensity=True, max_epochs=3, model_name='ncde', hidden_channels=49, hidden_hidden_channels=49, num_hidden_layers=4) | /home/yb2636_columbia_edu/DeepAudit_0.2/paper_with_code/16_Neural_Controlled_Differential_Equations/code/experiments | True | True | external_data_unavailable, url_404, no_smoke_evidence, SYNTHETIC_RUN_LOG_FROM_EXECUTOR_RESULTS, ...(+1) |
| exp_03 | exp_03 | skipped | full | 0 | 0.000 | import speech_commands; speech_commands.main(max_epochs=3, model_name='ncde', hidden_channels=90, hidden_hidden_channels=40, num_hidden_layers=4) | /home/yb2636_columbia_edu/DeepAudit_0.2/paper_with_code/16_Neural_Controlled_Differential_Equations/code/experiments | True | True | system_dependency_missing, ffmpeg_not_available, guardrail_enforced, SYNTHETIC_RUN_LOG_FROM_EXECUTOR_RESULTS, ...(+1) |

### Evidence Alignment, Verdict, Failure Metrics

| metric | value |
| --- | --- |
| claims_evaluated / not_evaluated | 120 / 0 |
| evaluated_claim_rate | 100.00 |
| executable / non_executable claims | 120 / 0 |
| claims_with_observed_metric / without | 0 / 120 |
| metric_recovery_rate | 0.000 |
| metric_parser_success / failed | 1 / 120 |
| reported/observed comparisons | 0 |
| within_tolerance / outside_tolerance | 0 / 0 |
| claim_to_evidence_mapped / unmapped | 0 / 120 |
| supported / partial / not_supported / inconclusive | 0 / 0 / 0 / 120 |
| main_verdict_per_paper | INCONCLUSIVE |
| headline evaluated/supported/partial/not/inconclusive | 14 / 0 / 0 / 0 / 14 |
| headline_metric_recovery_rate | 100.00 |
| executed_but_no_headline_evidence | False |
| executed_but_metric_missing | False |
| executed_but_claim_inconclusive | False |
| failed_but_diagnostic_evidence_available | True |
| failure_modes | missing_dataset:1, dependency_failure:1, entrypoint_ambiguity:1, metric_not_found:1, compute_limit:1 |
| main_inconclusive_reason | Experiment run exists but metric `accuracy` could not be aligned for this claim. |
| score_total / raw | 45.7 / 45.7 |
| ECR | False |
| posthoc evidence_tier | ATTEMPTED_NO_POSITIVE_EVIDENCE |
| execution_outcome_counts | none:3 |
| reproduced/skipped figures | 3 / 1 |
| logs_scanned | 16 |

Top reason codes: VISUAL_TABLE_EXTRACTED:117, LLM_REASON_PROVIDED:117, ALIGNMENT_AMBIGUOUS:106, VISUAL_ENRICHED:74, TABLE_EXPANDED:64, COMMAND_NOT_OBSERVED:18, ERROR_LOG:16, CONFIG_CLAIM:14, NO_DIRECT_CONFIG_EVIDENCE:14, MEAN_STD_TARGET_NORMALIZED:10, external_data_unavailable:8, url_404:8


<!-- completed 16_Neural_Controlled_Differential_Equations -->

## 17_Learning_to_Simulate_Complex_Physics_GNS

| field | value |
| --- | --- |
| paper_id | 17 |
| title | Learning to Simulate Complex Physics with Graph Networks |
| arxiv_id | 1607.06450 |
| venue | NA |
| year | 2020 |
| model_family | Learning to Simulate Complex Physics GNS |
| repo_url | https://github.com/deepmind/graph_nets |
| repo_available / cloned | True / True |
| expected_entry_point | notebook:graph_nets/demos_tf2/sort.ipynb |
| documented_entry_point_exists | False |
| compute_requirement | GPU/CUDA likely required or supported |
| dataset_requirement | CONTINUOUS, Water-3D / Sand-3D / Goop-3D, multiple datasets |

### Claim Extraction Metrics

| metric | value |
| --- | --- |
| total_claims_extracted | 120 |
| code_verifiable_claims | 100 |
| non_code_verifiable_claims | 20 |
| pct_code_verifiable_claims | 83.30 |
| claims_from_table / figure / text | 120 / 0 / 0 |
| main_claim_source | table |
| metric_contracts_generated | 13 |
| claims_with_metric_contract | 117 |
| pct_claims_with_metric_contract | 97.50 |
| claims_without_metric_contract | 3 |
| reported_metric_names | Learned GNS time per step including neighborhood computation, Learned GNS time relative to simulator, Maximum Mean Discrepancy one-step, Maximum Mean Discrepancy rollout, Maximum learned GNS time per step without neig... |
| top_reported_metrics | mse:21, Learned GNS time per step including neighborhood computation:17, Simulator time per step:16, Learned GNS time relative to simulator:16, Mean Squared Error rollout:5, Mean Squared Error one-step:5, Maximum Mean... |
| reported_dataset | CONTINUOUS, Water-3D / Sand-3D / Goop-3D, multiple datasets |
| reported_model | none |
| reported_method | none |
| reported_baseline | none |
| claim granularity | headline=70, secondary=50, comparison=0, ablation=0, efficiency=70, qualitative=0, theoretical=0 |

### Task, Entrypoint, Dataset, Command Metrics

| metric | value |
| --- | --- |
| tasks_generated | 0 |
| tasks_per_paper | 0 |
| tasks_per_claim | 0.000 |
| claims_with_candidate_task | 120 |
| claims_without_candidate_task | 0 |
| task_generation_coverage | 100.00 |
| candidate_entry_points_found | 6 |
| entry_points_per_repo | 6 |
| claim_to_entrypoint_mapped | 120 |
| claim_to_entrypoint_unmapped | 0 |
| entrypoint_mapping_coverage | 100.00 |
| entrypoint_ambiguity_count | 6 |
| datasets_identified | CONTINUOUS, Water-3D / Sand-3D / Goop-3D, multiple datasets |
| dataset_available | 0 |
| dataset_missing | 0 |
| dataset_mapping_coverage | 50.00 |
| commands_generated | 0 |
| commands_with_required_args | 0 |
| commands_missing_required_args | 0 |
| commands_matching_readme | 0 |
| commands_inferred_by_agent | 0 |
| command_confidence_score | 0.780 |

### Environment, Repair, Execution Metrics

| metric | value |
| --- | --- |
| env_setup_success / failed / repaired | True / False / False |
| requirements/environment/setup.py/pyproject found | False / False / True / False |
| dependency_install_success / failed | True / False |
| dependency_conflict_count | 0 |
| missing_package_count | 0 |
| obsolete_package_count | 0 |
| repair_attempts | 1 |
| repos_requiring_repair | False |
| repair_success / failed | False / False |
| repair_types | compute_limit, dependency_failure, entrypoint_ambiguity, metric_not_found, missing_dataset |
| env_setup_runtime_min | 0.210 |
| phase2_elapsed_min | 4.68 |
| runs_attempted / ok / partial / failed / timeout | 6 / 0 / 0 / 0 / 0 |
| run_success_rate | 0.000 |
| runtime_min | 0.000 |
| standard_execution_package_complete | True |
| artifact_count | 45 |
| result/log/checkpoint/figure files | 15 / 22 / 0 / 18 |

Dependency manifests detected: `setup.py`

#### Run-Level / Command-Level Metrics

| run | exp | status | fidelity | exit | runtime_sec | command | cwd | stdout | stderr | reason_codes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| exp_01 | exp_01 | skipped | full | 0 | 0.000 | N/A | /home/yb2636_columbia_edu/DeepAudit_0.2/paper_with_code/17_Learning_to_Simulate_Complex_Physics_GNS/code | True | True | REPO_LIBRARY_NOT_IMPLEMENTATION, MISSING_TRAINING_SCRIPTS, MISSING_DATASETS, MISSING_PRETRAINED_MODELS, ...(+2) |
| exp_02 | exp_02 | skipped | full | 0 | 0.000 | N/A | /home/yb2636_columbia_edu/DeepAudit_0.2/paper_with_code/17_Learning_to_Simulate_Complex_Physics_GNS/code | True | True | REPO_LIBRARY_NOT_IMPLEMENTATION, MISSING_TRAINING_SCRIPTS, MISSING_DATASETS, MISSING_PRETRAINED_MODELS, ...(+2) |
| exp_03 | exp_03 | skipped | full | 0 | 0.000 | N/A | /home/yb2636_columbia_edu/DeepAudit_0.2/paper_with_code/17_Learning_to_Simulate_Complex_Physics_GNS/code | True | True | REPO_LIBRARY_NOT_IMPLEMENTATION, MISSING_TRAINING_SCRIPTS, MISSING_DATASETS, MISSING_PRETRAINED_MODELS, ...(+2) |
| exp_04 | exp_04 | skipped | full | 0 | 0.000 | N/A | /home/yb2636_columbia_edu/DeepAudit_0.2/paper_with_code/17_Learning_to_Simulate_Complex_Physics_GNS/code | True | True | REPO_LIBRARY_NOT_IMPLEMENTATION, MISSING_TRAINING_SCRIPTS, MISSING_DATASETS, MISSING_PRETRAINED_MODELS, ...(+3) |
| exp_05 | exp_05 | skipped | full | 0 | 0.000 | N/A | /home/yb2636_columbia_edu/DeepAudit_0.2/paper_with_code/17_Learning_to_Simulate_Complex_Physics_GNS/code | True | True | REPO_LIBRARY_NOT_IMPLEMENTATION, MISSING_TRAINING_SCRIPTS, MISSING_DATASETS, MISSING_PRETRAINED_MODELS, ...(+2) |
| exp_1 | exp_1 | skipped | full | 0 | 0.000 | N/A | /home/yb2636_columbia_edu/DeepAudit_0.2/paper_with_code/17_Learning_to_Simulate_Complex_Physics_GNS/code | True | True | REPO_LIBRARY_NOT_IMPLEMENTATION, MISSING_TRAINING_SCRIPTS, MISSING_DATASETS, MISSING_PRETRAINED_MODELS, ...(+2) |

### Evidence Alignment, Verdict, Failure Metrics

| metric | value |
| --- | --- |
| claims_evaluated / not_evaluated | 120 / 0 |
| evaluated_claim_rate | 100.00 |
| executable / non_executable claims | 100 / 20 |
| claims_with_observed_metric / without | 0 / 120 |
| metric_recovery_rate | 0.000 |
| metric_parser_success / failed | 1 / 120 |
| reported/observed comparisons | 0 |
| within_tolerance / outside_tolerance | 0 / 0 |
| claim_to_evidence_mapped / unmapped | 0 / 120 |
| supported / partial / not_supported / inconclusive | 0 / 0 / 0 / 120 |
| main_verdict_per_paper | INCONCLUSIVE |
| headline evaluated/supported/partial/not/inconclusive | 70 / 0 / 0 / 0 / 70 |
| headline_metric_recovery_rate | 100.00 |
| executed_but_no_headline_evidence | False |
| executed_but_metric_missing | False |
| executed_but_claim_inconclusive | False |
| failed_but_diagnostic_evidence_available | True |
| failure_modes | missing_dataset:1, dependency_failure:1, entrypoint_ambiguity:1, metric_not_found:1, compute_limit:1 |
| main_inconclusive_reason | Experiment run exists but metric `mse` could not be aligned for this claim. |
| score_total / raw | 45.7 / 45.7 |
| ECR | False |
| posthoc evidence_tier | ATTEMPTED_NO_POSITIVE_EVIDENCE |
| execution_outcome_counts | none:6 |
| reproduced/skipped figures | 9 / 5 |
| logs_scanned | 21 |

Top reason codes: LLM_TABLE_EXTRACTED:293, LLM_REASON_PROVIDED:184, VISUAL_TABLE_EXTRACTED:143, ALIGNMENT_AMBIGUOUS:100, VISUAL_ENRICHED:41, REPO_LIBRARY_NOT_IMPLEMENTATION:24, MISSING_TRAINING_SCRIPTS:24, MISSING_DATASETS:24, MISSING_PRETRAINED_MODELS:24, MISSING_EVALUATION_SCRIPTS:24, MISSING_RECORDS:20, LLM_SKIP_OVERRIDDEN_BY_DETERMINISTIC_FALLBACK:18


<!-- completed 17_Learning_to_Simulate_Complex_Physics_GNS -->

## 18_ScoreBased_Generative_Modeling_via_SDEs

| field | value |
| --- | --- |
| paper_id | 18 |
| title | SCORE-BASED GENERATIVE MODELING THROUGH STOCHASTIC DIFFERENTIAL EQUATIONS |
| arxiv_id | 1703.06975 |
| venue | NA |
| year | 2019 |
| model_family | ScoreBased Generative Modeling via SDEs |
| repo_url | https://github.com/yang-song/score_sde |
| repo_available / cloned | True / True |
| expected_entry_point | python-file:main.py |
| documented_entry_point_exists | False |
| compute_requirement | CPU feasible in observed run |
| dataset_requirement | CIFAR-10 |

### Claim Extraction Metrics

| metric | value |
| --- | --- |
| total_claims_extracted | 120 |
| code_verifiable_claims | 102 |
| non_code_verifiable_claims | 18 |
| pct_code_verifiable_claims | 85.00 |
| claims_from_table / figure / text | 120 / 0 / 0 |
| main_claim_source | table |
| metric_contracts_generated | 7 |
| claims_with_metric_contract | 109 |
| pct_claims_with_metric_contract | 90.80 |
| claims_without_metric_contract | 11 |
| reported_metric_names | FID, FID (ODE), IS, NLL Test, best FID, lowest FID, test NLL |
| top_reported_metrics | FID:77, IS:7, FID (ODE):6, test NLL:5, best FID:4, NLL Test:2, lowest FID:1 |
| reported_dataset | CIFAR-10 |
| reported_model | none |
| reported_method | none |
| reported_baseline | none |
| claim granularity | headline=16, secondary=104, comparison=0, ablation=0, efficiency=0, qualitative=0, theoretical=0 |

### Task, Entrypoint, Dataset, Command Metrics

| metric | value |
| --- | --- |
| tasks_generated | 1 |
| tasks_per_paper | 1 |
| tasks_per_claim | 0.008 |
| claims_with_candidate_task | 120 |
| claims_without_candidate_task | 0 |
| task_generation_coverage | 100.00 |
| candidate_entry_points_found | 1 |
| entry_points_per_repo | 1 |
| claim_to_entrypoint_mapped | 120 |
| claim_to_entrypoint_unmapped | 0 |
| entrypoint_mapping_coverage | 100.00 |
| entrypoint_ambiguity_count | 0 |
| datasets_identified | CIFAR-10 |
| dataset_available | 0 |
| dataset_missing | 0 |
| dataset_mapping_coverage | 25.00 |
| commands_generated | 1 |
| commands_with_required_args | 1 |
| commands_missing_required_args | 0 |
| commands_matching_readme | 0 |
| commands_inferred_by_agent | 1 |
| command_confidence_score | 0.930 |

### Environment, Repair, Execution Metrics

| metric | value |
| --- | --- |
| env_setup_success / failed / repaired | False / True / True |
| requirements/environment/setup.py/pyproject found | True / False / False / False |
| dependency_install_success / failed | False / True |
| dependency_conflict_count | 2 |
| missing_package_count | 2 |
| obsolete_package_count | 0 |
| repair_attempts | 1 |
| repos_requiring_repair | False |
| repair_success / failed | False / False |
| repair_types | checkpoint_missing, compute_limit, dependency_failure, entrypoint_ambiguity, metric_not_found |
| env_setup_runtime_min | 0.450 |
| phase2_elapsed_min | 8.77 |
| runs_attempted / ok / partial / failed / timeout | 4 / 0 / 0 / 0 / 0 |
| run_success_rate | 0.000 |
| runtime_min | 0.000 |
| standard_execution_package_complete | True |
| artifact_count | 47 |
| result/log/checkpoint/figure files | 15 / 16 / 0 / 10 |

Dependency manifests detected: `requirements.txt`

#### Run-Level / Command-Level Metrics

| run | exp | status | fidelity | exit | runtime_sec | command | cwd | stdout | stderr | reason_codes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| exp_01 | exp_01 | skipped | full | 0 | 0.000 | python main.py --config=configs/ve/cifar10_ncsnpp_deep_continuous.py --mode=eval --workdir=eval_output --eval_folder=samples | /home/yb2636_columbia_edu/DeepAudit_0.2/paper_with_code/18_ScoreBased_Generative_Modeling_via_SDEs/code | True | True | PYTHON_VERSION_INCOMPATIBLE, MISSING_PRETRAINED_CHECKPOINTS, MISSING_CALIBRATION_FILES, DEPENDENCY_RESOLUTION_FAILURE, ...(+2) |
| exp_02 | exp_02 | skipped | full | 0 | 0.000 | python main.py --config=configs/vp/cifar10_ddpmpp_deep_continuous.py --mode=eval --workdir=eval_output --eval_folder=likelihoods | /home/yb2636_columbia_edu/DeepAudit_0.2/paper_with_code/18_ScoreBased_Generative_Modeling_via_SDEs/code | True | True | PYTHON_VERSION_INCOMPATIBLE, MISSING_PRETRAINED_CHECKPOINTS, DEPENDENCY_RESOLUTION_FAILURE, SYNTHETIC_RUN_LOG_FROM_EXECUTOR_RESULTS, ...(+1) |
| exp_03 | exp_03 | skipped | full | 0 | 0.000 | python main.py --config=configs/ve/cifar10_ncsnpp_deep_continuous.py --mode=eval --workdir=eval_output --eval_folder=samples | /home/yb2636_columbia_edu/DeepAudit_0.2/paper_with_code/18_ScoreBased_Generative_Modeling_via_SDEs/code | True | True | PYTHON_VERSION_INCOMPATIBLE, MISSING_PRETRAINED_CHECKPOINTS, DEPENDENCY_RESOLUTION_FAILURE, SYNTHETIC_RUN_LOG_FROM_EXECUTOR_RESULTS, ...(+1) |
| exp_04 | exp_04 | skipped | full | 0 | 0.000 | python main.py --config=configs/ve/cifar10_ncsnpp_deep_continuous.py --mode=eval --workdir=eval_output --eval_folder=snr_search | /home/yb2636_columbia_edu/DeepAudit_0.2/paper_with_code/18_ScoreBased_Generative_Modeling_via_SDEs/code | True | True | PYTHON_VERSION_INCOMPATIBLE, MISSING_PRETRAINED_CHECKPOINTS, NONPRIMARY_EXPERIMENT, DEPENDENCY_RESOLUTION_FAILURE, ...(+2) |

### Evidence Alignment, Verdict, Failure Metrics

| metric | value |
| --- | --- |
| claims_evaluated / not_evaluated | 120 / 0 |
| evaluated_claim_rate | 100.00 |
| executable / non_executable claims | 102 / 18 |
| claims_with_observed_metric / without | 0 / 120 |
| metric_recovery_rate | 0.000 |
| metric_parser_success / failed | 1 / 120 |
| reported/observed comparisons | 0 |
| within_tolerance / outside_tolerance | 0 / 0 |
| claim_to_evidence_mapped / unmapped | 0 / 120 |
| supported / partial / not_supported / inconclusive | 0 / 0 / 0 / 120 |
| main_verdict_per_paper | INCONCLUSIVE |
| headline evaluated/supported/partial/not/inconclusive | 16 / 0 / 0 / 0 / 16 |
| headline_metric_recovery_rate | 100.00 |
| executed_but_no_headline_evidence | False |
| executed_but_metric_missing | False |
| executed_but_claim_inconclusive | False |
| failed_but_diagnostic_evidence_available | True |
| failure_modes | dependency_failure:1, entrypoint_ambiguity:1, metric_not_found:1, compute_limit:1, checkpoint_missing:1 |
| main_inconclusive_reason | Experiment run exists but metric `FID` could not be aligned for this claim. |
| score_total / raw | 40.7 / 40.7 |
| ECR | False |
| posthoc evidence_tier | ATTEMPTED_NO_POSITIVE_EVIDENCE |
| execution_outcome_counts | none:4 |
| reproduced/skipped figures | 5 / 17 |
| logs_scanned | 15 |

Top reason codes: LLM_REASON_PROVIDED:340, LLM_TABLE_EXTRACTED:173, VISUAL_ENRICHED:173, VISUAL_TABLE_EXTRACTED:167, ALIGNMENT_AMBIGUOUS:102, MEAN_STD_TARGET_NORMALIZED:71, MISSING_RECORDS:18, PYTHON_VERSION_INCOMPATIBLE:16, MISSING_PRETRAINED_CHECKPOINTS:16, DEPENDENCY_RESOLUTION_FAILURE:16, COMMAND_NOT_OBSERVED:16, PARSE_LOW_CONFIDENCE:12


<!-- completed 18_ScoreBased_Generative_Modeling_via_SDEs -->

## 19_SINDy_Autoencoder

| field | value |
| --- | --- |
| paper_id | 19 |
| title | Data-driven discovery of coordinates and governing equations |
| arxiv_id | 1804.00183 |
| venue | NA |
| year | NA |
| model_family | SINDy Autoencoder |
| repo_url | https://github.com/kpchamp/SindyAutoencoders |
| repo_available / cloned | True / True |
| expected_entry_point | notebook:examples/rd/train_reactiondiffusion.ipynb |
| documented_entry_point_exists | False |
| compute_requirement | unknown |
| dataset_requirement | Lorenz system, Nonlinear pendulum, Reaction-diffusion |

### Claim Extraction Metrics

| metric | value |
| --- | --- |
| total_claims_extracted | 84 |
| code_verifiable_claims | 84 |
| non_code_verifiable_claims | 0 |
| pct_code_verifiable_claims | 100.00 |
| claims_from_table / figure / text | 70 / 0 / 14 |
| main_claim_source | table |
| metric_contracts_generated | 5 |
| claims_with_metric_contract | 41 |
| pct_claims_with_metric_contract | 48.80 |
| claims_without_metric_contract | 43 |
| reported_metric_names | correct identifications, error, fraction of unexplained variance, loss, mean square error |
| top_reported_metrics | loss:37, mean square error:1, error:1, fraction of unexplained variance:1, correct identifications:1 |
| reported_dataset | Lorenz system, Nonlinear pendulum, Reaction-diffusion |
| reported_model | none |
| reported_method | none |
| reported_baseline | none |
| claim granularity | headline=4, secondary=80, comparison=0, ablation=0, efficiency=0, qualitative=0, theoretical=0 |

### Task, Entrypoint, Dataset, Command Metrics

| metric | value |
| --- | --- |
| tasks_generated | 0 |
| tasks_per_paper | 0 |
| tasks_per_claim | 0.000 |
| claims_with_candidate_task | 84 |
| claims_without_candidate_task | 0 |
| task_generation_coverage | 100.00 |
| candidate_entry_points_found | 9 |
| entry_points_per_repo | 9 |
| claim_to_entrypoint_mapped | 84 |
| claim_to_entrypoint_unmapped | 0 |
| entrypoint_mapping_coverage | 100.00 |
| entrypoint_ambiguity_count | 9 |
| datasets_identified | Lorenz system, Nonlinear pendulum, Reaction-diffusion |
| dataset_available | 0 |
| dataset_missing | 0 |
| dataset_mapping_coverage | 100.00 |
| commands_generated | 0 |
| commands_with_required_args | 0 |
| commands_missing_required_args | 0 |
| commands_matching_readme | 0 |
| commands_inferred_by_agent | 0 |
| command_confidence_score | 0.780 |

### Environment, Repair, Execution Metrics

| metric | value |
| --- | --- |
| env_setup_success / failed / repaired | True / False / False |
| requirements/environment/setup.py/pyproject found | False / False / False / False |
| dependency_install_success / failed | True / False |
| dependency_conflict_count | 0 |
| missing_package_count | 0 |
| obsolete_package_count | 0 |
| repair_attempts | 1 |
| repos_requiring_repair | False |
| repair_success / failed | False / False |
| repair_types | compute_limit, dependency_failure, entrypoint_ambiguity, metric_mismatch, metric_not_found, missing_dataset, ...(+1) |
| env_setup_runtime_min | 0.090 |
| phase2_elapsed_min | 8.68 |
| runs_attempted / ok / partial / failed / timeout | 3 / 0 / 0 / 2 / 0 |
| run_success_rate | 0.000 |
| runtime_min | 0 |
| standard_execution_package_complete | True |
| artifact_count | 42 |
| result/log/checkpoint/figure files | 15 / 13 / 0 / 6 |

Dependency manifests detected: `none`

#### Run-Level / Command-Level Metrics

| run | exp | status | fidelity | exit | runtime_sec | command | cwd | stdout | stderr | reason_codes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| exp_01 | exp_01 | failed | artifact | 0 | NA |  | . | True | True | UNTRACEABLE_METRICS |
| exp_02 | exp_02 | skipped | artifact | 0 | NA |  | . | True | True | MISSING_DATA_FILE |
| exp_03 | exp_03 | failed | artifact | 0 | NA |  | . | True | True | UNTRACEABLE_METRICS |

### Evidence Alignment, Verdict, Failure Metrics

| metric | value |
| --- | --- |
| claims_evaluated / not_evaluated | 84 / 0 |
| evaluated_claim_rate | 100.00 |
| executable / non_executable claims | 84 / 0 |
| claims_with_observed_metric / without | 26 / 58 |
| metric_recovery_rate | 31.00 |
| metric_parser_success / failed | 6 / 58 |
| reported/observed comparisons | 19 |
| within_tolerance / outside_tolerance | 1 / 18 |
| claim_to_evidence_mapped / unmapped | 26 / 58 |
| supported / partial / not_supported / inconclusive | 1 / 0 / 18 / 65 |
| main_verdict_per_paper | INCONCLUSIVE |
| headline evaluated/supported/partial/not/inconclusive | 4 / 0 / 0 / 0 / 4 |
| headline_metric_recovery_rate | 100.00 |
| executed_but_no_headline_evidence | False |
| executed_but_metric_missing | False |
| executed_but_claim_inconclusive | False |
| failed_but_diagnostic_evidence_available | True |
| failure_modes | missing_dataset:1, dependency_failure:1, entrypoint_ambiguity:1, metric_not_found:1, metric_mismatch:1, compute_limit:1, undocumented_hyperparameter:1 |
| main_inconclusive_reason | Configuration claim requires direct code/config evidence; execution metrics alone do not verify the paper setup. |
| score_total / raw | 65.0 / 51.6 |
| ECR | False |
| posthoc evidence_tier | ATTEMPTED_NO_POSITIVE_EVIDENCE |
| execution_outcome_counts | none:3 |
| reproduced/skipped figures | 3 / 6 |
| logs_scanned | 12 |

Top reason codes: VISUAL_TABLE_EXTRACTED:84, LLM_REASON_PROVIDED:84, TABLE_EXPANDED:64, CAPTION_METRIC_MATRIX:64, NO_DIRECT_CONFIG_EVIDENCE:58, CONFIG_CLAIM:56, PHASE2_PACKAGE_METRIC:33, ARTIFACT_BASED_EVIDENCE:26, MATCHED_METRIC:19, OUTSIDE_TOLERANCE:18, VERDICT_NOT_SUPPORTED:18, ERROR_LOG:10


<!-- completed 19_SINDy_Autoencoder -->

## 21_KAN_KolmogorovArnold_Networks

| field | value |
| --- | --- |
| paper_id | 21 |
| title | KAN: Kolmogorov–Arnold Networks |
| arxiv_id | 2309.08600 |
| venue | NA |
| year | NA |
| model_family | KAN KolmogorovArnold Networks |
| repo_url | https://github.com/KindXiaoming/pykan |
| repo_available / cloned | True / True |
| expected_entry_point | notebook:tutorials/Example/Example_1_function_fitting.ipynb |
| documented_entry_point_exists | False |
| compute_requirement | unknown |
| dataset_requirement | special functions task |

### Claim Extraction Metrics

| metric | value |
| --- | --- |
| total_claims_extracted | 120 |
| code_verifiable_claims | 120 |
| non_code_verifiable_claims | 0 |
| pct_code_verifiable_claims | 100.00 |
| claims_from_table / figure / text | 120 / 0 / 0 |
| main_claim_source | table |
| metric_contracts_generated | 2 |
| claims_with_metric_contract | 120 |
| pct_claims_with_metric_contract | 100.00 |
| claims_without_metric_contract | 0 |
| reported_metric_names | loss, mse |
| top_reported_metrics | mse:84, loss:36 |
| reported_dataset | special functions task |
| reported_model | none |
| reported_method | none |
| reported_baseline | none |
| claim granularity | headline=21, secondary=99, comparison=0, ablation=0, efficiency=0, qualitative=0, theoretical=0 |

### Task, Entrypoint, Dataset, Command Metrics

| metric | value |
| --- | --- |
| tasks_generated | 0 |
| tasks_per_paper | 0 |
| tasks_per_claim | 0.000 |
| claims_with_candidate_task | 120 |
| claims_without_candidate_task | 0 |
| task_generation_coverage | 100.00 |
| candidate_entry_points_found | 97 |
| entry_points_per_repo | 97 |
| claim_to_entrypoint_mapped | 120 |
| claim_to_entrypoint_unmapped | 0 |
| entrypoint_mapping_coverage | 100.00 |
| entrypoint_ambiguity_count | 97 |
| datasets_identified | special functions task |
| dataset_available | 1 |
| dataset_missing | 0 |
| dataset_mapping_coverage | 100.00 |
| commands_generated | 0 |
| commands_with_required_args | 0 |
| commands_missing_required_args | 0 |
| commands_matching_readme | 0 |
| commands_inferred_by_agent | 0 |
| command_confidence_score | 0.780 |

### Environment, Repair, Execution Metrics

| metric | value |
| --- | --- |
| env_setup_success / failed / repaired | False / True / False |
| requirements/environment/setup.py/pyproject found | True / False / True / False |
| dependency_install_success / failed | False / False |
| dependency_conflict_count | 0 |
| missing_package_count | 0 |
| obsolete_package_count | 0 |
| repair_attempts | 1 |
| repos_requiring_repair | False |
| repair_success / failed | False / False |
| repair_types | compute_limit, dependency_failure, entrypoint_ambiguity, metric_mismatch, metric_not_found, missing_dataset |
| env_setup_runtime_min | 3.01 |
| phase2_elapsed_min | 9.94 |
| runs_attempted / ok / partial / failed / timeout | 1 / 1 / 0 / 0 / 0 |
| run_success_rate | 100.00 |
| runtime_min | 0.360 |
| standard_execution_package_complete | True |
| artifact_count | 36 |
| result/log/checkpoint/figure files | 15 / 7 / 0 / 22 |

Dependency manifests detected: `docs/Makefile, requirements.txt, setup.py`

#### Run-Level / Command-Level Metrics

| run | exp | status | fidelity | exit | runtime_sec | command | cwd | stdout | stderr | reason_codes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| exp_01 | exp_01 | ok | trend | 0 | 21.77 | /home/yb2636_columbia_edu/miniconda3/condabin/mamba run -n 21_KAN_KolmogorovArnold_Networks_executor python /home/yb2636_columbia_edu/DeepAudit_0.2/artifacts/21_KAN_KolmogorovArnold_Networks/execution/executor_outputs... | /home/yb2636_columbia_edu/DeepAudit_0.2/paper_with_code/21_KAN_KolmogorovArnold_Networks/code | True | True | TREND_RUN_FIDELITY, INCREASED_DATASET_SIZE, INCREASED_TRAINING_STEPS, CONVERGENCE_VALIDATED, ...(+1) |

### Evidence Alignment, Verdict, Failure Metrics

| metric | value |
| --- | --- |
| claims_evaluated / not_evaluated | 120 / 0 |
| evaluated_claim_rate | 100.00 |
| executable / non_executable claims | 120 / 0 |
| claims_with_observed_metric / without | 36 / 84 |
| metric_recovery_rate | 30.00 |
| metric_parser_success / failed | 6 / 84 |
| reported/observed comparisons | 36 |
| within_tolerance / outside_tolerance | 0 / 36 |
| claim_to_evidence_mapped / unmapped | 36 / 84 |
| supported / partial / not_supported / inconclusive | 0 / 0 / 36 / 84 |
| main_verdict_per_paper | INCONCLUSIVE |
| headline evaluated/supported/partial/not/inconclusive | 21 / 0 / 0 / 0 / 21 |
| headline_metric_recovery_rate | 100.00 |
| executed_but_no_headline_evidence | False |
| executed_but_metric_missing | False |
| executed_but_claim_inconclusive | True |
| failed_but_diagnostic_evidence_available | False |
| failure_modes | missing_dataset:1, dependency_failure:1, entrypoint_ambiguity:1, metric_not_found:1, metric_mismatch:1, compute_limit:1 |
| main_inconclusive_reason | Experiment run exists but metric `mse` could not be aligned for this claim. |
| score_total / raw | 65.0 / 61.6 |
| ECR | False |
| posthoc evidence_tier | TREND_EVIDENCE |
| execution_outcome_counts | TREND_SUPPORTED:1 |
| reproduced/skipped figures | 11 / 18 |
| logs_scanned | 6 |

Top reason codes: VISUAL_ENRICHED:413, TABLE_EXPANDED:356, LLM_REASON_PROVIDED:161, VISUAL_TABLE_EXTRACTED:123, ENTRYPOINT_CWD_INFERRED:96, ALIGNMENT_AMBIGUOUS:84, LLM_TABLE_EXTRACTED:61, PHASE2_PACKAGE_METRIC:42, REDUCED_FIDELITY_EVIDENCE:38, MATCHED_METRIC:36, OUTSIDE_TOLERANCE:36, VERDICT_NOT_SUPPORTED:36


<!-- completed 21_KAN_KolmogorovArnold_Networks -->

## 22_Spikformer_When_Spiking_Neural_Network_Meets_Transformer

| field | value |
| --- | --- |
| paper_id | 22 |
| title | SPIKFORMER: WHEN SPIKING NEURAL NETWORK MEETS TRANSFORMER |
| arxiv_id | 2009.14794 |
| venue | NA |
| year | 1997 |
| model_family | Spikformer When Spiking Neural Network Meets Transformer |
| repo_url | https://github.com/ZK-Zhou/spikformer |
| repo_available / cloned | True / True |
| expected_entry_point | python-file:imagenet/train.py |
| documented_entry_point_exists | False |
| compute_requirement | unknown |
| dataset_requirement | CIFAR10-DVS/DVS128, CIFAR10/CIFAR100 |

### Claim Extraction Metrics

| metric | value |
| --- | --- |
| total_claims_extracted | 120 |
| code_verifiable_claims | 120 |
| non_code_verifiable_claims | 0 |
| pct_code_verifiable_claims | 100.00 |
| claims_from_table / figure / text | 120 / 0 / 0 |
| main_claim_source | table |
| metric_contracts_generated | 3 |
| claims_with_metric_contract | 120 |
| pct_claims_with_metric_contract | 100.00 |
| claims_without_metric_contract | 0 |
| reported_metric_names | acc, accuracy, accuracy_difference |
| top_reported_metrics | accuracy:112, acc:6, accuracy_difference:2 |
| reported_dataset | CIFAR10-DVS/DVS128, CIFAR10/CIFAR100 |
| reported_model | none |
| reported_method | none |
| reported_baseline | none |
| claim granularity | headline=10, secondary=110, comparison=0, ablation=6, efficiency=8, qualitative=0, theoretical=0 |

### Task, Entrypoint, Dataset, Command Metrics

| metric | value |
| --- | --- |
| tasks_generated | 4 |
| tasks_per_paper | 4 |
| tasks_per_claim | 0.033 |
| claims_with_candidate_task | 120 |
| claims_without_candidate_task | 0 |
| task_generation_coverage | 100.00 |
| candidate_entry_points_found | 4 |
| entry_points_per_repo | 4 |
| claim_to_entrypoint_mapped | 120 |
| claim_to_entrypoint_unmapped | 0 |
| entrypoint_mapping_coverage | 100.00 |
| entrypoint_ambiguity_count | 0 |
| datasets_identified | CIFAR10-DVS/DVS128, CIFAR10/CIFAR100 |
| dataset_available | 0 |
| dataset_missing | 0 |
| dataset_mapping_coverage | 40.00 |
| commands_generated | 4 |
| commands_with_required_args | 4 |
| commands_missing_required_args | 0 |
| commands_matching_readme | 0 |
| commands_inferred_by_agent | 4 |
| command_confidence_score | 0.930 |

### Environment, Repair, Execution Metrics

| metric | value |
| --- | --- |
| env_setup_success / failed / repaired | True / False / False |
| requirements/environment/setup.py/pyproject found | False / False / False / False |
| dependency_install_success / failed | True / False |
| dependency_conflict_count | 0 |
| missing_package_count | 0 |
| obsolete_package_count | 2 |
| repair_attempts | 1 |
| repos_requiring_repair | False |
| repair_success / failed | False / False |
| repair_types | compute_limit, dependency_failure, entrypoint_ambiguity, metric_mismatch, metric_not_found, missing_dataset |
| env_setup_runtime_min | 0.090 |
| phase2_elapsed_min | 0.090 |
| runs_attempted / ok / partial / failed / timeout | 0 / 0 / 0 / 0 / 0 |
| run_success_rate | 0.000 |
| runtime_min | 0 |
| standard_execution_package_complete | False |
| artifact_count | 34 |
| result/log/checkpoint/figure files | 15 / 8 / 0 / 4 |

Dependency manifests detected: `none`

No run-level execution rows were available.


### Evidence Alignment, Verdict, Failure Metrics

| metric | value |
| --- | --- |
| claims_evaluated / not_evaluated | 120 / 0 |
| evaluated_claim_rate | 100.00 |
| executable / non_executable claims | 120 / 0 |
| claims_with_observed_metric / without | 0 / 120 |
| metric_recovery_rate | 0.000 |
| metric_parser_success / failed | 1 / 120 |
| reported/observed comparisons | 0 |
| within_tolerance / outside_tolerance | 0 / 0 |
| claim_to_evidence_mapped / unmapped | 0 / 120 |
| supported / partial / not_supported / inconclusive | 0 / 0 / 0 / 120 |
| main_verdict_per_paper | INCONCLUSIVE |
| headline evaluated/supported/partial/not/inconclusive | 10 / 0 / 0 / 0 / 10 |
| headline_metric_recovery_rate | 100.00 |
| executed_but_no_headline_evidence | False |
| executed_but_metric_missing | False |
| executed_but_claim_inconclusive | False |
| failed_but_diagnostic_evidence_available | True |
| failure_modes | missing_dataset:1, dependency_failure:1, entrypoint_ambiguity:1, metric_not_found:1, metric_mismatch:1, compute_limit:1 |
| main_inconclusive_reason | Experiment run exists but metric `accuracy` could not be aligned for this claim. |
| score_total / raw | 65.0 / 49.0 |
| ECR | False |
| posthoc evidence_tier | NO_PHASE2_RUNS |
| execution_outcome_counts | none |
| reproduced/skipped figures | 2 / 9 |
| logs_scanned | 7 |

Top reason codes: VISUAL_ENRICHED:290, LLM_REASON_PROVIDED:240, TABLE_EXPANDED:184, CAPTION_METRIC_MATRIX:184, VISUAL_TABLE_EXTRACTED:134, ALIGNMENT_AMBIGUOUS:120, LLM_TABLE_EXTRACTED:106, SKIP_NO_PHASE2_EVIDENCE:8, ERROR_LOG:5, LLM_SKIP_OVERRIDDEN_BY_DETERMINISTIC_FALLBACK:4, DETERMINISTIC_NO_EVIDENCE_TEXT_PANEL:4, NO_NUMERIC_EVIDENCE:4


<!-- completed 22_Spikformer_When_Spiking_Neural_Network_Meets_Transformer -->

## 23_Next_Generation_Reservoir_Computing

| field | value |
| --- | --- |
| paper_id | 23 |
| title | Next Generation Reservoir Computing |
| arxiv_id | 2103.0036 |
| venue | NA |
| year | 2021 |
| model_family | Next Generation Reservoir Computing |
| repo_url | https://github.com/quantinfo/ng-rc-paper-code |
| repo_available / cloned | True / True |
| expected_entry_point | NA |
| documented_entry_point_exists | False |
| compute_requirement | CPU feasible in observed run |
| dataset_requirement | Lorenz63, double-scroll system |

### Claim Extraction Metrics

| metric | value |
| --- | --- |
| total_claims_extracted | 46 |
| code_verifiable_claims | 46 |
| non_code_verifiable_claims | 0 |
| pct_code_verifiable_claims | 100.00 |
| claims_from_table / figure / text | 36 / 0 / 10 |
| main_claim_source | table |
| metric_contracts_generated | 3 |
| claims_with_metric_contract | 17 |
| pct_claims_with_metric_contract | 37.00 |
| claims_without_metric_contract | 29 |
| reported_metric_names | computational cost reduction, estimated speed up, mse |
| top_reported_metrics | estimated speed up:8, mse:3, computational cost reduction:1 |
| reported_dataset | Lorenz63, double-scroll system |
| reported_model | none |
| reported_method | none |
| reported_baseline | none |
| claim granularity | headline=5, secondary=41, comparison=0, ablation=0, efficiency=16, qualitative=0, theoretical=0 |

### Task, Entrypoint, Dataset, Command Metrics

| metric | value |
| --- | --- |
| tasks_generated | 0 |
| tasks_per_paper | 0 |
| tasks_per_claim | 0.000 |
| claims_with_candidate_task | 46 |
| claims_without_candidate_task | 0 |
| task_generation_coverage | 100.00 |
| candidate_entry_points_found | 0 |
| entry_points_per_repo | 0 |
| claim_to_entrypoint_mapped | 0 |
| claim_to_entrypoint_unmapped | 46 |
| entrypoint_mapping_coverage | 0.000 |
| entrypoint_ambiguity_count | 0 |
| datasets_identified | Lorenz63, double-scroll system |
| dataset_available | 2 |
| dataset_missing | 0 |
| dataset_mapping_coverage | 40.00 |
| commands_generated | 0 |
| commands_with_required_args | 0 |
| commands_missing_required_args | 0 |
| commands_matching_readme | 0 |
| commands_inferred_by_agent | 0 |
| command_confidence_score | NA |

### Environment, Repair, Execution Metrics

| metric | value |
| --- | --- |
| env_setup_success / failed / repaired | False / True / True |
| requirements/environment/setup.py/pyproject found | True / False / False / False |
| dependency_install_success / failed | False / True |
| dependency_conflict_count | 2 |
| missing_package_count | 2 |
| obsolete_package_count | 0 |
| repair_attempts | 1 |
| repos_requiring_repair | False |
| repair_success / failed | False / False |
| repair_types | compute_limit, dependency_failure, entrypoint_ambiguity, metric_mismatch, metric_not_found, missing_dataset |
| env_setup_runtime_min | 2.09 |
| phase2_elapsed_min | 7.10 |
| runs_attempted / ok / partial / failed / timeout | 5 / 3 / 2 / 0 / 0 |
| run_success_rate | 60.00 |
| runtime_min | 0.640 |
| standard_execution_package_complete | True |
| artifact_count | 41 |
| result/log/checkpoint/figure files | 15 / 19 / 0 / 20 |

Dependency manifests detected: `requirements-exact.txt, requirements.txt`

#### Run-Level / Command-Level Metrics

| run | exp | status | fidelity | exit | runtime_sec | command | cwd | stdout | stderr | reason_codes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| exp_01 | exp_01 | ok | full | 0 | 2.68 | python LorenzConstLinQuadraticNVARtimedelayNRMSE-RK23.py | /home/yb2636_columbia_edu/DeepAudit_0.2/paper_with_code/23_Next_Generation_Reservoir_Computing/code | True | True | none |
| exp_02 | exp_02 | partial | full | 0 | 0.000 | N/A (comparison experiment) | /home/yb2636_columbia_edu/DeepAudit_0.2/paper_with_code/23_Next_Generation_Reservoir_Computing/code | True | True | COMPARISON_EXPERIMENT_NO_REFERENCE_CODE, SYNTHETIC_RUN_LOG_FROM_EXECUTOR_RESULTS, COMMAND_NOT_OBSERVED |
| exp_03 | exp_03 | ok | full | 0 | 13.86 | python DoubleScrollNVAR-RK23.py | /home/yb2636_columbia_edu/DeepAudit_0.2/paper_with_code/23_Next_Generation_Reservoir_Computing/code | True | True | none |
| exp_04 | exp_04 | partial | full | 0 | 0.000 | N/A (comparison experiment) | /home/yb2636_columbia_edu/DeepAudit_0.2/paper_with_code/23_Next_Generation_Reservoir_Computing/code | True | True | COMPARISON_EXPERIMENT_NO_REFERENCE_CODE, SYNTHETIC_RUN_LOG_FROM_EXECUTOR_RESULTS, COMMAND_NOT_OBSERVED |
| exp_05 | exp_05 | ok | full | 0 | 21.86 | Multiple variants executed for timing analysis | /home/yb2636_columbia_edu/DeepAudit_0.2/paper_with_code/23_Next_Generation_Reservoir_Computing/code | True | True | SYNTHETIC_RUN_LOG_FROM_EXECUTOR_RESULTS |

### Evidence Alignment, Verdict, Failure Metrics

| metric | value |
| --- | --- |
| claims_evaluated / not_evaluated | 46 / 0 |
| evaluated_claim_rate | 100.00 |
| executable / non_executable claims | 46 / 0 |
| claims_with_observed_metric / without | 0 / 46 |
| metric_recovery_rate | 0.000 |
| metric_parser_success / failed | 1 / 46 |
| reported/observed comparisons | 0 |
| within_tolerance / outside_tolerance | 0 / 0 |
| claim_to_evidence_mapped / unmapped | 0 / 46 |
| supported / partial / not_supported / inconclusive | 0 / 0 / 0 / 46 |
| main_verdict_per_paper | INCONCLUSIVE |
| headline evaluated/supported/partial/not/inconclusive | 5 / 0 / 0 / 0 / 5 |
| headline_metric_recovery_rate | 100.00 |
| executed_but_no_headline_evidence | False |
| executed_but_metric_missing | True |
| executed_but_claim_inconclusive | True |
| failed_but_diagnostic_evidence_available | False |
| failure_modes | missing_dataset:1, dependency_failure:1, entrypoint_ambiguity:1, metric_not_found:1, metric_mismatch:1, compute_limit:1 |
| main_inconclusive_reason | Configuration claim requires direct code/config evidence; execution metrics alone do not verify the paper setup. |
| score_total / raw | 80.0 / 67.1 |
| ECR | False |
| posthoc evidence_tier | FULL_REPRODUCTION_EVIDENCE |
| execution_outcome_counts | FULLY_REPRODUCED:3, none:2 |
| reproduced/skipped figures | 10 / 1 |
| logs_scanned | 18 |

Top reason codes: LLM_REASON_PROVIDED:94, LLM_TABLE_EXTRACTED:62, VISUAL_ENRICHED:62, CONFIG_CLAIM:34, NO_DIRECT_CONFIG_EVIDENCE:34, VISUAL_TABLE_EXTRACTED:32, ALIGNMENT_AMBIGUOUS:12, LLM_SKIP_OVERRIDDEN_BY_DETERMINISTIC_FALLBACK:12, DETERMINISTIC_METRIC_TABLE_FALLBACK:12, LLM_CODEGEN_RENDERED:9, PARSE_LOW_CONFIDENCE:9, COMPARISON_EXPERIMENT_NO_REFERENCE_CODE:8


<!-- completed 23_Next_Generation_Reservoir_Computing -->

# Aggregate Metrics

| metric | value |
| --- | --- |
| N_papers | 18 |
| papers_by_model_family | Scaling Equilibrium Propagation to Deep ConvNets:1, PEPITA:1, Learning without Feedback DRTP:1, Lagrangian Neural Networks:1, Simplifying HNN LNN via Explicit Constraints:1, Symplectic ODE Net:1, E n Equivariant Graph... |
| papers_by_year | NA:5, 2019:3, 2020:2, 2015:1, 2012:1, 2016:1, 2013:1, 2006:1, 2018:1, 1997:1, 2021:1 |
| papers_by_venue | NA:18 |
| papers_by_compute_requirement | GPU/CUDA likely required or supported:8, CPU feasible in observed run:5, unknown:5 |
| TOTAL_CLAIMS | 1804 |
| CODE_VERIFIABLE | 1637 |
| PCT_CODE_VERIFIABLE | 90.70 |
| METRIC_CONTRACT_COUNT | 87 |
| PCT_METRIC_CONTRACT_COVERAGE | 85.10 |
| ENV_SUCCESS | 13 |
| EXEC_SUCCESS | 9 |
| PACKAGE_SUCCESS | 17 |
| TOTAL_VERIFIED | 1804 |
| SUPPORTED | 3 |
| PARTIALLY_SUPPORTED | 0 |
| NOT_SUPPORTED | 81 |
| INCONCLUSIVE | 1720 |
| METRIC_RECOVERY_RATE | 5.00 |
| median_claims_per_paper | 120.00 |
| mean_claims_per_paper | 100.22 |
| median_runtime_min | 3.29 |
| mean_runtime_min | 38.27 |
| median_env_setup_runtime_min | 0.290 |
| mean_env_setup_runtime_min | 0.970 |
| repair_success_rate | 100.00 |
| standard_package_success_rate | 94.40 |
| posthoc_full_reproduction_evidence_repos | 3 |
| posthoc_trend_evidence_repos | 6 |
| posthoc_executable_or_smoke_evidence_repos | 3 |
| posthoc_positive_execution_evidence_repos | 12 |
| posthoc_run_outcomes | none:63, EXECUTABLE:10, FULLY_REPRODUCED:9, TREND_SUPPORTED:8 |

## No-Rerun Evidence Tier Summary

This section is a post-hoc reporting view over the existing artifacts. It does not change the strict claim-level verdicts above. It separates exact claim support from experiment-level execution evidence so completed runs are not hidden by strict metric-alignment failures.

| evidence_tier | repo_count | interpretation |
| --- | --- | --- |
| FULL_REPRODUCTION_EVIDENCE | 3 | At least one Phase2 run was marked FULLY_REPRODUCED. |
| TREND_EVIDENCE | 6 | At least one run supported a trend or reduced-fidelity result, but no full run was marked FULLY_REPRODUCED. |
| EXECUTABLE_OR_SMOKE_EVIDENCE | 3 | The repo executed or smoke-tested successfully, but did not recover a full/trend result. |
| ATTEMPTED_NO_POSITIVE_EVIDENCE | 5 | Phase2 produced run rows, but no positive execution outcome. |
| NO_PHASE2_RUNS | 1 | No canonical Phase2 run rows were available. |

Paper-ready phrasing: under strict claim-level matching, only 3 claims were exactly supported; however, existing execution artifacts contain positive experiment-level evidence for 12 / 18 repositories (3 full, 6 trend, 3 executable/smoke). The gap is attributable primarily to conservative metric-to-claim alignment rather than absence of execution evidence.


## Cross-Run Summary Table

| run_id | score | claims | code_verifiable | tasks | env_ok | runs ok/partial/failed | metric_recovery | verdict S/P/N/I | package | evidence_tier | figures |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 01_Scaling_Equilibrium_Propagation_to_Deep_ConvNets | 50.00 | 109 | 109 | 10 | True | 0/1/0 | 0.000 | 0/0/0/109 | True | EXECUTABLE_OR_SMOKE_EVIDENCE | 0/12 |
| 03_PEPITA | 65.00 | 120 | 117 | 12 | False | 3/0/1 | 8.30 | 0/0/10/110 | True | EXECUTABLE_OR_SMOKE_EVIDENCE | 8/6 |
| 04_Learning_without_Feedback_DRTP | 80.00 | 120 | 66 | 2 | False | 3/0/0 | 0.000 | 0/0/0/120 | True | FULL_REPRODUCTION_EVIDENCE | 5/1 |
| 05_Lagrangian_Neural_Networks | 65.00 | 31 | 4 | 4 | True | 0/1/0 | 0.000 | 0/0/0/31 | True | TREND_EVIDENCE | 3/2 |
| 06_Simplifying_HNN_LNN_via_Explicit_Constraints | 43.70 | 34 | 34 | 6 | True | 0/0/7 | 0.000 | 0/0/0/34 | True | ATTEMPTED_NO_POSITIVE_EVIDENCE | 8/2 |
| 07_Symplectic_ODE_Net | 65.00 | 78 | 78 | 7 | True | 0/1/6 | 0.000 | 0/0/0/78 | True | TREND_EVIDENCE | 6/4 |
| 08_E_n_Equivariant_Graph_Neural_Networks_EGNN | 50.00 | 120 | 112 | 12 | True | 1/0/2 | 10.00 | 2/0/10/108 | True | EXECUTABLE_OR_SMOKE_EVIDENCE | 6/2 |
| 09_Fourier_Neural_Operator_FNO | 65.00 | 102 | 95 | 2 | True | 3/0/0 | 0.000 | 0/0/0/102 | True | TREND_EVIDENCE | 4/6 |
| 10_Your_Classifier_is_Secretly_an_EBM_JEM | 65.00 | 120 | 110 | 6 | True | 2/4/0 | 5.80 | 0/0/7/113 | True | TREND_EVIDENCE | 12/7 |
| 11_MACE_Higher_Order_Equivariant_MPNNs copy | 65.00 | 120 | 117 | 12 | True | 1/0/0 | 0.000 | 0/0/0/120 | True | TREND_EVIDENCE | 3/2 |
| 13_Liquid_TimeConstant_Networks_LTC | 80.00 | 120 | 103 | 11 | True | 3/0/3 | 0.000 | 0/0/0/120 | True | FULL_REPRODUCTION_EVIDENCE | 7/5 |
| 16_Neural_Controlled_Differential_Equations | 45.70 | 120 | 120 | 6 | True | 0/0/0 | 0.000 | 0/0/0/120 | True | ATTEMPTED_NO_POSITIVE_EVIDENCE | 3/1 |
| 17_Learning_to_Simulate_Complex_Physics_GNS | 45.70 | 120 | 100 | 0 | True | 0/0/0 | 0.000 | 0/0/0/120 | True | ATTEMPTED_NO_POSITIVE_EVIDENCE | 9/5 |
| 18_ScoreBased_Generative_Modeling_via_SDEs | 40.70 | 120 | 102 | 1 | False | 0/0/0 | 0.000 | 0/0/0/120 | True | ATTEMPTED_NO_POSITIVE_EVIDENCE | 5/17 |
| 19_SINDy_Autoencoder | 65.00 | 84 | 84 | 0 | True | 0/0/2 | 31.00 | 1/0/18/65 | True | ATTEMPTED_NO_POSITIVE_EVIDENCE | 3/6 |
| 21_KAN_KolmogorovArnold_Networks | 65.00 | 120 | 120 | 0 | False | 1/0/0 | 30.00 | 0/0/36/84 | True | TREND_EVIDENCE | 11/18 |
| 22_Spikformer_When_Spiking_Neural_Network_Meets_Transformer | 65.00 | 120 | 120 | 4 | True | 0/0/0 | 0.000 | 0/0/0/120 | False | NO_PHASE2_RUNS | 2/9 |
| 23_Next_Generation_Reservoir_Computing | 80.00 | 46 | 46 | 0 | False | 3/2/0 | 0.000 | 0/0/0/46 | True | FULL_REPRODUCTION_EVIDENCE | 10/1 |

## Failure Taxonomy Aggregate

| failure_mode | count | rate_over_affected_repos_pct | example_paper |
| --- | --- | --- | --- |
| dependency_failure | 18 | 100.00 | 01_Scaling_Equilibrium_Propagation_to_Deep_ConvNets |
| entrypoint_ambiguity | 18 | 100.00 | 01_Scaling_Equilibrium_Propagation_to_Deep_ConvNets |
| metric_not_found | 18 | 100.00 | 01_Scaling_Equilibrium_Propagation_to_Deep_ConvNets |
| compute_limit | 18 | 100.00 | 01_Scaling_Equilibrium_Propagation_to_Deep_ConvNets |
| metric_mismatch | 13 | 72.20 | 01_Scaling_Equilibrium_Propagation_to_Deep_ConvNets |
| missing_dataset | 12 | 66.70 | 03_PEPITA |
| undocumented_hyperparameter | 4 | 22.20 | 03_PEPITA |
| data_download_failure | 2 | 11.10 | 08_E_n_Equivariant_Graph_Neural_Networks_EGNN |
| repo_clone_failure | 1 | 5.60 | 10_Your_Classifier_is_Secretly_an_EBM_JEM |
| checkpoint_missing | 1 | 5.60 | 18_ScoreBased_Generative_Modeling_via_SDEs |

## Top Reported Metrics

| metric | frequency |
| --- | --- |
| accuracy | 337 |
| mse | 168 |
| loss | 101 |
| FID | 80 |
| f1 | 76 |
| AUROC | 58 |
| mae | 51 |
| auc | 36 |
| computational depth | 22 |
| memory usage | 21 |
| relative error | 20 |
| test error | 17 |

## Case Study Metrics

| case_type | case_paper_id | case_claim_id | reported_or_successful_component | observed_or_missing_component | case_command/evidence | runtime_or_exit | case_verdict/reason |
| --- | --- | --- | --- | --- | --- | --- | --- |
| successful_reproduction | 08_E_n_Equivariant_Graph_Neural_Networks_EGNN | claim_03 | 0.024 | 0.029 | see run-level command table | 32.08 | SUPPORTED |
| partial_reproduction | 01_Scaling_Equilibrium_Propagation_to_Deep_ConvNets | NA | successful components: 0 | missing/partial components: 1 | see run-level command table | 40.53 | INCONCLUSIVE |
| diagnostic_failure | 01_Scaling_Equilibrium_Propagation_to_Deep_ConvNets | NA | readme-python:main.py | NA | LLM_TABLE_EXTRACTED, LLM_REASON_PROVIDED, VISUAL_ENRICHED, LLM_TABLE_EXTRACTED, LLM_REASON_PROVIDED, VISUAL_ENRICHED, ...(+760) | NA | Configuration claim requires direct code/config evidence; execution metrics alone do not verify the paper setup. |

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


# Checklist Coverage

Coverage legend: `direct` means the value is read from pipeline artifacts; `derived` means it is computed from one or more artifacts; `not captured` means the current artifact schema does not contain enough information and the report marks the field as `NA` or a diagnostic proxy.

| checklist_metric | coverage_in_result_md |
| --- | --- |
| paper_id | derived |
| run_id | direct |
| title | derived |
| arxiv_id | derived |
| venue | not captured or proxy only |
| year | derived |
| model_family | derived |
| repo_url | derived |
| N_papers | derived |
| papers_by_model_family | derived |
| papers_by_year | derived |
| papers_by_venue | derived |
| papers_by_compute_requirement | derived |
| repo_available | derived |
| repo_url_valid | not captured or proxy only |
| repo_cloned_successfully | derived |
| expected_entry_point | derived |
| documented_entry_point_exists | derived |
| compute_requirement | derived |
| dataset_requirement | derived |
| total_claims_extracted | direct |
| code_verifiable_claims | derived |
| non_code_verifiable_claims | derived |
| pct_code_verifiable_claims | derived |
| median_claims_per_paper | derived |
| mean_claims_per_paper | derived |
| claims_from_table | derived |
| claims_from_figure | derived |
| claims_from_text | derived |
| main_claim_source | derived |
| claim_source_distribution | derived |
| metric_contracts_generated | direct |
| claims_with_metric_contract | derived |
| pct_claims_with_metric_contract | derived |
| metric_contract_coverage | derived |
| claims_without_metric_contract | derived |
| reported_metric_names | derived |
| reported_metric_frequency | derived |
| top_reported_metrics | derived |
| reported_metric_value | derived |
| reported_metric_unit | not captured or proxy only |
| reported_metric_direction | not captured or proxy only |
| reported_dataset | derived |
| reported_split | not captured or proxy only |
| reported_model | derived |
| reported_method | derived |
| reported_baseline | derived |
| headline_claims | derived |
| secondary_claims | derived |
| comparison_claims | derived |
| ablation_claims | derived |
| efficiency_claims | derived |
| qualitative_claims | derived |
| theoretical_claims | derived |
| tasks_generated | derived |
| tasks_per_paper | derived |
| tasks_per_claim | derived |
| claims_with_candidate_task | derived |
| claims_without_candidate_task | derived |
| task_generation_coverage | derived |
| candidate_entry_points_found | direct |
| entry_points_per_repo | derived |
| claim_to_entrypoint_mapped | derived |
| claim_to_entrypoint_unmapped | derived |
| entrypoint_mapping_coverage | derived |
| entrypoint_ambiguity_count | derived |
| datasets_identified | derived |
| dataset_download_scripts_found | not captured or proxy only |
| dataset_paths_identified | not captured or proxy only |
| dataset_available | derived |
| dataset_missing | derived |
| dataset_mapping_coverage | derived |
| commands_generated | derived |
| commands_with_required_args | derived |
| commands_missing_required_args | derived |
| commands_matching_readme | derived |
| commands_inferred_by_agent | derived |
| command_confidence_score | derived |
| env_setup_success | derived |
| env_setup_failed | derived |
| env_setup_repaired | derived |
| env_setup_success_rate | derived |
| env_setup_failure_rate | derived |
| requirements_file_found | derived |
| environment_file_found | derived |
| setup_py_found | derived |
| pyproject_toml_found | derived |
| dependency_install_success | derived |
| dependency_install_failed | derived |
| dependency_conflict_count | derived |
| missing_package_count | derived |
| obsolete_package_count | derived |
| repair_attempts | derived |
| repos_requiring_repair | derived |
| repair_success | derived |
| repair_failed | derived |
| repair_success_rate | derived |
| repair_types | derived |
| dependency_repair_count | derived |
| command_repair_count | derived |
| data_path_repair_count | derived |
| env_setup_runtime_sec | derived |
| env_setup_runtime_min | derived |
| median_env_setup_runtime | derived |
| mean_env_setup_runtime | derived |
| repos_attempted | derived |
| repos_with_at_least_one_successful_run | derived |
| repos_with_all_runs_failed | derived |
| repos_with_partial_execution | derived |
| execution_success_rate | derived |
| execution_partial_rate | derived |
| execution_failure_rate | derived |
| runs_attempted | derived |
| runs_successful | derived |
| runs_failed | derived |
| runs_timeout | derived |
| runs_partial | derived |
| run_success_rate | derived |
| run_failure_rate | derived |
| command | direct |
| cwd | direct |
| exit_code | direct |
| status | direct |
| stdout_available | derived |
| stderr_available | derived |
| stdout_tail | direct |
| stderr_tail | direct |
| runtime_sec | direct |
| runtime_min | derived |
| total_runtime_per_repo | derived |
| median_runtime_per_repo | derived |
| mean_runtime_per_repo | derived |
| max_runtime_per_repo | derived |
| timeout_count | derived |
| artifacts_generated | derived |
| artifact_count | derived |
| result_files_generated | derived |
| log_files_generated | derived |
| checkpoint_files_generated | derived |
| figure_files_generated | derived |
| standard_execution_package_complete | derived |
| package_success_rate | derived |
| claims_evaluated | derived |
| claims_not_evaluated | derived |
| evaluated_claim_rate | derived |
| executable_claims | derived |
| non_executable_claims | derived |
| claims_with_observed_metric | derived |
| claims_without_observed_metric | derived |
| metric_recovery_rate | derived |
| observed_metric_name | derived |
| observed_metric_value | derived |
| observed_metric_unit | derived |
| metric_parser_success | derived |
| metric_parser_failed | derived |
| reported_value | derived |
| observed_value | derived |
| absolute_delta | derived |
| relative_delta | derived |
| tolerance | not captured or proxy only |
| within_tolerance | derived |
| outside_tolerance | derived |
| metric_direction_match | derived |
| metric_direction_mismatch | derived |
| claim_id | direct |
| evidence_run_id | derived |
| evidence_command | derived |
| evidence_artifact | derived |
| evidence_log_path | direct |
| evidence_metric_source | derived |
| claim_to_evidence_mapped | derived |
| claim_to_evidence_unmapped | derived |
| supported_count | direct |
| partially_supported_count | direct |
| not_supported_count | direct |
| inconclusive_count | direct |
| total_verified_claims | derived |
| supported_rate | derived |
| partially_supported_rate | derived |
| not_supported_rate | derived |
| inconclusive_rate | derived |
| supported_per_paper | derived |
| partially_supported_per_paper | derived |
| not_supported_per_paper | derived |
| inconclusive_per_paper | derived |
| main_verdict_per_paper | derived |
| headline_claims_evaluated | derived |
| headline_claim_supported | derived |
| headline_claim_partially_supported | derived |
| headline_claim_not_supported | derived |
| headline_claim_inconclusive | derived |
| headline_metric_recovery_rate | derived |
| repos_executed_but_no_headline_evidence | derived |
| repos_executed_but_metric_missing | derived |
| repos_executed_but_claim_inconclusive | derived |
| repos_failed_but_diagnostic_evidence_available | derived |
| missing_dataset_count | derived |
| dependency_failure_count | derived |
| entrypoint_ambiguity_count | derived |
| execution_timeout_count | derived |
| metric_not_found_count | derived |
| metric_mismatch_count | derived |
| compute_limit_count | derived |
| repo_clone_failure_count | derived |
| data_download_failure_count | derived |
| preprocessing_missing_count | derived |
| checkpoint_missing_count | derived |
| undocumented_hyperparameter_count | derived |
| missing_dataset_rate | derived |
| dependency_failure_rate | derived |
| entrypoint_ambiguity_rate | derived |
| execution_timeout_rate | derived |
| metric_not_found_rate | derived |
| metric_mismatch_rate | derived |
| compute_limit_rate | derived |
| failure_mode | derived |
| failure_reason_code | derived |
| failure_stage | derived |
| affected_claim_count | derived |
| affected_repo_count | derived |
| example_paper | derived |
| evidence_snippet | derived |
| recommended_remediation | not captured or proxy only |
| main_inconclusive_reason | derived |
| inconclusive_due_to_missing_dataset | derived |
| inconclusive_due_to_dependency_failure | derived |
| inconclusive_due_to_timeout | derived |
| inconclusive_due_to_metric_not_found | derived |
| inconclusive_due_to_compute_limit | derived |
| inconclusive_due_to_entrypoint_ambiguity | derived |
| total_repair_attempts | derived |
| repair_attempts_per_repo | derived |
| repair_attempts_per_failed_run | derived |
| repos_with_repair_attempts | derived |
| successful_repairs | derived |
| failed_repairs | derived |
| repair_success_rate | derived |
| post_repair_execution_success | derived |
| post_repair_metric_recovery | derived |
| dependency_repairs | derived |
| version_pin_repairs | derived |
| missing_package_repairs | derived |
| path_repairs | derived |
| command_argument_repairs | derived |
| dataset_path_repairs | derived |
| case_paper_id | derived |
| case_claim_id | derived |
| case_claim_text | derived |
| case_reported_metric | derived |
| case_reported_value | derived |
| case_observed_metric | derived |
| case_observed_value | derived |
| case_tolerance | derived |
| case_command | derived |
| case_runtime_min | derived |
| case_evidence_artifact | derived |
| case_verdict | derived |
| case_paper_id | derived |
| case_claim_id | derived |
| case_successful_component | derived |
| case_missing_component | derived |
| case_recovered_metric | derived |
| case_missing_metric | derived |
| case_partial_reason | derived |
| case_evidence_artifact | derived |
| case_verdict | derived |
| case_paper_id | derived |
| case_claim_id | derived |
| case_expected_entrypoint | derived |
| case_attempted_command | derived |
| case_failure_reason_code | derived |
| case_exit_code | derived |
| case_stderr_snippet | derived |
| case_repair_attempted | derived |
| case_repair_success | derived |
| case_recommended_remediation | derived |
| N_papers | derived |
| TOTAL_CLAIMS | derived |
| CODE_VERIFIABLE | derived |
| PCT_CODE_VERIFIABLE | derived |
| METRIC_CONTRACT_COUNT | derived |
| PCT_METRIC_CONTRACT_COVERAGE | derived |
| ENV_SUCCESS | derived |
| EXEC_SUCCESS | derived |
| PACKAGE_SUCCESS | derived |
| TOTAL_VERIFIED | derived |
| SUPPORTED | derived |
| PARTIALLY_SUPPORTED | derived |
| NOT_SUPPORTED | derived |
| INCONCLUSIVE | derived |
| METRIC_RECOVERY_RATE | derived |
| top_failure_mode_1 | derived |
| top_failure_mode_1_count | derived |
| top_failure_mode_2 | derived |
| top_failure_mode_2_count | derived |
| top_failure_mode_3 | derived |
| top_failure_mode_3_count | derived |
| main_inconclusive_reason | derived |
| repos_executed_but_no_headline_evidence | derived |
| median_runtime_min | derived |
| mean_runtime_min | derived |
| median_env_setup_runtime_min | derived |
| median_execution_runtime_min | derived |
| repair_success_rate | derived |
| standard_package_success_rate | derived |

# Figures

![score by run](artifacts/summary_figures/score_by_run.png)

![verdict distribution](artifacts/summary_figures/verdict_distribution.png)

![execution outcomes](artifacts/summary_figures/execution_outcomes.png)

![posthoc evidence tiers](artifacts/summary_figures/posthoc_evidence_tiers.png)

![failure modes](artifacts/summary_figures/failure_modes.png)

