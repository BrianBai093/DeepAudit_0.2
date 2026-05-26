# DeepAudit 指标提取清单

> Source: paper framework :contentReference[oaicite:0]{index=0}

## 1. Corpus-Level Metrics

### 1.1 Paper Identity
- `paper_id`
- `run_id`
- `title`
- `arxiv_id`
- `venue`
- `year`
- `model_family`
- `repo_url`

### 1.2 Corpus Composition
- `N_papers`
- `papers_by_model_family`
- `papers_by_year`
- `papers_by_venue`
- `papers_by_compute_requirement`

### 1.3 Repository Metadata
- `repo_available`
- `repo_url_valid`
- `repo_cloned_successfully`
- `expected_entry_point`
- `documented_entry_point_exists`
- `compute_requirement`
- `dataset_requirement`

---

## 2. Claim Extraction Metrics

### 2.1 Claim Counts
- `total_claims_extracted`
- `code_verifiable_claims`
- `non_code_verifiable_claims`
- `pct_code_verifiable_claims`
- `median_claims_per_paper`
- `mean_claims_per_paper`

### 2.2 Claim Source Metrics
- `claims_from_table`
- `claims_from_figure`
- `claims_from_text`
- `main_claim_source`
- `claim_source_distribution`

### 2.3 Metric Contract Metrics
- `metric_contracts_generated`
- `claims_with_metric_contract`
- `pct_claims_with_metric_contract`
- `metric_contract_coverage`
- `claims_without_metric_contract`

### 2.4 Reported Metric Metrics
- `reported_metric_names`
- `reported_metric_frequency`
- `top_reported_metrics`
- `reported_metric_value`
- `reported_metric_unit`
- `reported_metric_direction`
- `reported_dataset`
- `reported_split`
- `reported_model`
- `reported_method`
- `reported_baseline`

### 2.5 Claim Granularity Metrics
- `headline_claims`
- `secondary_claims`
- `comparison_claims`
- `ablation_claims`
- `efficiency_claims`
- `qualitative_claims`
- `theoretical_claims`

---

## 3. Task Specification Metrics

### 3.1 Task Generation
- `tasks_generated`
- `tasks_per_paper`
- `tasks_per_claim`
- `claims_with_candidate_task`
- `claims_without_candidate_task`
- `task_generation_coverage`

### 3.2 Entrypoint Mapping
- `candidate_entry_points_found`
- `entry_points_per_repo`
- `claim_to_entrypoint_mapped`
- `claim_to_entrypoint_unmapped`
- `entrypoint_mapping_coverage`
- `entrypoint_ambiguity_count`

### 3.3 Dataset Mapping
- `datasets_identified`
- `dataset_download_scripts_found`
- `dataset_paths_identified`
- `dataset_available`
- `dataset_missing`
- `dataset_mapping_coverage`

### 3.4 Command Mapping
- `commands_generated`
- `commands_with_required_args`
- `commands_missing_required_args`
- `commands_matching_readme`
- `commands_inferred_by_agent`
- `command_confidence_score`

---

## 4. Environment Setup Metrics

### 4.1 Setup Outcome
- `env_setup_success`
- `env_setup_failed`
- `env_setup_repaired`
- `env_setup_success_rate`
- `env_setup_failure_rate`

### 4.2 Dependency Metrics
- `requirements_file_found`
- `environment_file_found`
- `setup_py_found`
- `pyproject_toml_found`
- `dependency_install_success`
- `dependency_install_failed`
- `dependency_conflict_count`
- `missing_package_count`
- `obsolete_package_count`

### 4.3 Repair Metrics
- `repair_attempts`
- `repos_requiring_repair`
- `repair_success`
- `repair_failed`
- `repair_success_rate`
- `repair_types`
- `dependency_repair_count`
- `command_repair_count`
- `data_path_repair_count`

### 4.4 Setup Runtime
- `env_setup_runtime_sec`
- `env_setup_runtime_min`
- `median_env_setup_runtime`
- `mean_env_setup_runtime`

---

## 5. Execution Metrics

### 5.1 Repository-Level Execution
- `repos_attempted`
- `repos_with_at_least_one_successful_run`
- `repos_with_all_runs_failed`
- `repos_with_partial_execution`
- `execution_success_rate`
- `execution_partial_rate`
- `execution_failure_rate`

### 5.2 Run-Level Execution
- `runs_attempted`
- `runs_successful`
- `runs_failed`
- `runs_timeout`
- `runs_partial`
- `run_success_rate`
- `run_failure_rate`

### 5.3 Command-Level Metrics
- `command`
- `cwd`
- `exit_code`
- `status`
- `stdout_available`
- `stderr_available`
- `stdout_tail`
- `stderr_tail`

### 5.4 Runtime Metrics
- `runtime_sec`
- `runtime_min`
- `total_runtime_per_repo`
- `median_runtime_per_repo`
- `mean_runtime_per_repo`
- `max_runtime_per_repo`
- `timeout_count`

### 5.5 Artifact Metrics
- `artifacts_generated`
- `artifact_count`
- `result_files_generated`
- `log_files_generated`
- `checkpoint_files_generated`
- `figure_files_generated`
- `standard_execution_package_complete`
- `package_success_rate`

---

## 6. Evidence Alignment Metrics

### 6.1 Claim Evaluation Coverage
- `claims_evaluated`
- `claims_not_evaluated`
- `evaluated_claim_rate`
- `executable_claims`
- `non_executable_claims`

### 6.2 Metric Recovery
- `claims_with_observed_metric`
- `claims_without_observed_metric`
- `metric_recovery_rate`
- `observed_metric_name`
- `observed_metric_value`
- `observed_metric_unit`
- `metric_parser_success`
- `metric_parser_failed`

### 6.3 Metric Comparison
- `reported_value`
- `observed_value`
- `absolute_delta`
- `relative_delta`
- `tolerance`
- `within_tolerance`
- `outside_tolerance`
- `metric_direction_match`
- `metric_direction_mismatch`

### 6.4 Evidence Linking
- `claim_id`
- `evidence_run_id`
- `evidence_command`
- `evidence_artifact`
- `evidence_log_path`
- `evidence_metric_source`
- `claim_to_evidence_mapped`
- `claim_to_evidence_unmapped`

---

## 7. Verdict Metrics

### 7.1 Verdict Counts
- `supported_count`
- `partially_supported_count`
- `not_supported_count`
- `inconclusive_count`
- `total_verified_claims`

### 7.2 Verdict Rates
- `supported_rate`
- `partially_supported_rate`
- `not_supported_rate`
- `inconclusive_rate`

### 7.3 Paper-Level Verdict Distribution
- `supported_per_paper`
- `partially_supported_per_paper`
- `not_supported_per_paper`
- `inconclusive_per_paper`
- `main_verdict_per_paper`

### 7.4 Headline Claim Metrics
- `headline_claims_evaluated`
- `headline_claim_supported`
- `headline_claim_partially_supported`
- `headline_claim_not_supported`
- `headline_claim_inconclusive`
- `headline_metric_recovery_rate`

### 7.5 Execution-Evidence Gap Metrics
- `repos_executed_but_no_headline_evidence`
- `repos_executed_but_metric_missing`
- `repos_executed_but_claim_inconclusive`
- `repos_failed_but_diagnostic_evidence_available`

---

## 8. Failure Taxonomy Metrics

### 8.1 Failure Mode Counts
- `missing_dataset_count`
- `dependency_failure_count`
- `entrypoint_ambiguity_count`
- `execution_timeout_count`
- `metric_not_found_count`
- `metric_mismatch_count`
- `compute_limit_count`
- `repo_clone_failure_count`
- `data_download_failure_count`
- `preprocessing_missing_count`
- `checkpoint_missing_count`
- `undocumented_hyperparameter_count`

### 8.2 Failure Mode Rates
- `missing_dataset_rate`
- `dependency_failure_rate`
- `entrypoint_ambiguity_rate`
- `execution_timeout_rate`
- `metric_not_found_rate`
- `metric_mismatch_rate`
- `compute_limit_rate`

### 8.3 Failure Attribution
- `failure_mode`
- `failure_reason_code`
- `failure_stage`
- `affected_claim_count`
- `affected_repo_count`
- `example_paper`
- `evidence_snippet`
- `recommended_remediation`

### 8.4 Inconclusive Reason Metrics
- `main_inconclusive_reason`
- `inconclusive_due_to_missing_dataset`
- `inconclusive_due_to_dependency_failure`
- `inconclusive_due_to_timeout`
- `inconclusive_due_to_metric_not_found`
- `inconclusive_due_to_compute_limit`
- `inconclusive_due_to_entrypoint_ambiguity`

---

## 9. Repair and Self-Healing Metrics

### 9.1 Repair Attempt Metrics
- `total_repair_attempts`
- `repair_attempts_per_repo`
- `repair_attempts_per_failed_run`
- `repos_with_repair_attempts`

### 9.2 Repair Outcome Metrics
- `successful_repairs`
- `failed_repairs`
- `repair_success_rate`
- `post_repair_execution_success`
- `post_repair_metric_recovery`

### 9.3 Repair Type Metrics
- `dependency_repairs`
- `version_pin_repairs`
- `missing_package_repairs`
- `path_repairs`
- `command_argument_repairs`
- `dataset_path_repairs`

---

## 10. Case Study Metrics

### 10.1 Successful Reproduction Case
- `case_paper_id`
- `case_claim_id`
- `case_claim_text`
- `case_reported_metric`
- `case_reported_value`
- `case_observed_metric`
- `case_observed_value`
- `case_tolerance`
- `case_command`
- `case_runtime_min`
- `case_evidence_artifact`
- `case_verdict`

### 10.2 Partial Reproduction Case
- `case_paper_id`
- `case_claim_id`
- `case_successful_component`
- `case_missing_component`
- `case_recovered_metric`
- `case_missing_metric`
- `case_partial_reason`
- `case_evidence_artifact`
- `case_verdict`

### 10.3 Diagnostic Failure Case
- `case_paper_id`
- `case_claim_id`
- `case_expected_entrypoint`
- `case_attempted_command`
- `case_failure_reason_code`
- `case_exit_code`
- `case_stderr_snippet`
- `case_repair_attempted`
- `case_repair_success`
- `case_recommended_remediation`

---

## 11. Aggregate Headline Metrics

### 11.1 Main Paper Numbers
- `N_papers`
- `TOTAL_CLAIMS`
- `CODE_VERIFIABLE`
- `PCT_CODE_VERIFIABLE`
- `METRIC_CONTRACT_COUNT`
- `PCT_METRIC_CONTRACT_COVERAGE`
- `ENV_SUCCESS`
- `EXEC_SUCCESS`
- `PACKAGE_SUCCESS`
- `TOTAL_VERIFIED`
- `SUPPORTED`
- `PARTIALLY_SUPPORTED`
- `NOT_SUPPORTED`
- `INCONCLUSIVE`
- `METRIC_RECOVERY_RATE`

### 11.2 Main Failure Numbers
- `top_failure_mode_1`
- `top_failure_mode_1_count`
- `top_failure_mode_2`
- `top_failure_mode_2_count`
- `top_failure_mode_3`
- `top_failure_mode_3_count`
- `main_inconclusive_reason`
- `repos_executed_but_no_headline_evidence`

### 11.3 Efficiency Numbers
- `median_runtime_min`
- `mean_runtime_min`
- `median_env_setup_runtime_min`
- `median_execution_runtime_min`
- `repair_success_rate`
- `standard_package_success_rate`