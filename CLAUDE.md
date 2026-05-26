# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Project Does

DeepAudit is an automated research paper reproducibility audit system. It takes a paper (Markdown + optional PDF) and a companion code repository, then runs a three-phase pipeline to determine whether the paper's executable claims are reproducible:

- **Phase 1**: Extracts verifiable claims, visual elements, and repo analysis → produces `claims_ir.json`, `task_spec.json`, `metric_contract.json`
- **Phase 2**: Builds an isolated conda/venv environment and uses the Claude Code Agent SDK to autonomously execute experiments → produces `phase2_execution_package.json`
- **Phase 3**: Aligns execution evidence against claims and produces `verdict.json`, `reproducibility_score.json`, `report.md`

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install pytest
```

Required env vars before any run:
- `OPENAI_API_KEY` — Phase 1/3 LLM calls
- `ANTHROPIC_API_KEY` — Phase 2 Claude Code Agent SDK

## Running the Pipeline

Single phase:
```bash
python -m p2c.main --phase 1 --paper_md Target/paper/full.md --paper_md_out output/paper.md --paper_pdf Target/paper.pdf --repo_dir Target/code --run_id audit_001 --artifacts_dir artifacts --budget_minutes 60
python -m p2c.main --phase 2 --paper_md output/paper.md --paper_md_out output/paper.md --repo_dir Target/code --run_id audit_001 --artifacts_dir artifacts --budget_minutes 180
python -m p2c.main --phase 3 --paper_md output/paper.md --paper_md_out output/paper.md --repo_dir Target/code --run_id audit_001 --artifacts_dir artifacts --budget_minutes 60
```

Via script (edit `PROJECT_ROOT`, `PAPER_MD`, `PAPER_PDF`, `REPO_DIR` in the DEFAULTS block first):
```bash
./scripts/run_audit.sh audit_001          # all 3 phases
./scripts/run_audit.sh audit_001 1,2      # phases 1 and 2 only
./scripts/run_audit.sh audit_001 3        # phase 3 only
```

View results:
```bash
cat artifacts/audit_001/results/verdict.json
cat artifacts/audit_001/results/report.md
cat artifacts/audit_001/execution/executor_outputs/PHASE2_RESULTS.md
```

## Running Tests

```bash
python -m pytest tests/ -v

# Target specific modules:
python -m pytest tests/test_main_context.py -v
python -m pytest tests/test_phase2_fixes.py -v
python -m pytest tests/test_phase3_report.py -v
python -m pytest tests/test_visual_enrichment.py -v
```

## Architecture

### Control Flow

`p2c/main.py` → parses CLI args, creates `ArtifactManager`, builds all agents via `p2c/graph.py::build_agents()`, then calls `run_phase_1/2/3()`.

`p2c/graph.py` defines the exact agent call sequence for each phase. It is the authoritative source of step order.

### Agent Pattern

All agents inherit from `p2c/agents/base.py::BaseAgent`. Each agent:
- Takes `(llm, artifacts, step_index, step_total)` at construction
- Implements `execute(ctx: dict) -> dict`
- Uses `self.safe_chat_json/text/vision()` for LLM calls with built-in error handling
- Writes results to `artifacts/<run_id>/` via `self.artifacts`

The `ctx` dict is the shared mutable pipeline context — phases pass data between agents through it.

### LLM Client

`p2c/llm/client.py::LLMClient` is an OpenAI-compatible HTTP client (uses `urllib`, no SDK dependency). Phase 1/3 agents use this. Controlled by `OPENAI_MODEL`, `OPENAI_BASE_URL`, `OPENAI_TIMEOUT_SEC`.

### Phase 2 Execution Model

Phase 2 is a state machine in `p2c/agents/phase2/orchestrator.py::Phase2Orchestrator`:
1. `ToolAgent.build_env_spec()` — reads `task/repo_analysis.json`, infers `ExecutorEnvSpec`
2. `ToolAgent.run()` — creates isolated conda/venv, installs deps, writes `env_setup_result.json`
3. `ExecutorAgent.run()` — opens a Claude Code Agent SDK session in the target repo; the executor writes results to `execution/executor_outputs/`; the host scans for new artifacts and copies them to `execution/phase2_artifacts/files/`
4. On dependency failure, the orchestrator patches the env and retries (up to `P2C_MAX_ENV_PATCH` times)

**Important**: `ExecutorAgent` has a repo mutation guard. If Claude modifies tracked source files in the target repo, the run is rejected with `SOURCE_MUTATION_DETECTED`.

### Artifacts

`p2c/io_artifacts.py::ArtifactManager` manages all file I/O under `artifacts/<run_id>/`. `ensure_tree()` pre-creates placeholder JSON files — a file existing does **not** mean that phase completed. To check Phase 2 validity, inspect `phase2_execution_package.json` or `run_manifest.json` for non-empty `experiments`/`runs` arrays.

### Data Models

All Pydantic schemas are in `p2c/schemas.py`. Key cross-phase models:
- `ClaimsIR` / `Experiment` / `ClaimItem` — Phase 1 → 2/3 hand-off
- `ExecutorEnvSpec` / `EnvSetupResult` — Phase 2 environment lifecycle
- `phase2_execution_package.json` — the primary evidence source Phase 3 consumes (prefer over `run_manifest.json`)
- `VerdictDoc` / `ReproducibilityScore` — Phase 3 final output

### RAG (Optional)

`p2c/rag/` provides a code embedding index built during Phase 1 repo analysis. It degrades gracefully if embedding is unavailable. The index is passed via `ctx["_code_index"]`.

### Env Detection

`p2c/env_detection.py` defines the priority-ordered list of conda environment filenames the system searches for (e.g., `p2c_env.yml` > `environment.yml` > `env.yml`). Native env files are preferred over synthesized ones.

## Key Environment Variables

| Variable | Purpose | Default |
|---|---|---|
| `OPENAI_MODEL` | Phase 1/3 model | `gpt-5.4` |
| `OPENAI_BASE_URL` | OpenAI-compatible endpoint | `https://api.openai.com/v1` |
| `MINERU_API_TOKEN` | MinerU standard PDF parsing fallback for large/long PDFs | unset |
| `P2C_MINERU_MODE` | PDF→Markdown mode: `auto`, `agent`, `standard`, or `off` | `auto` |
| `P2C_CLAUDE_MODEL` | Phase 2 executor model | `claude-haiku-4-5-20251001` |
| `P2C_MAX_ENV_PATCH` | Max env repair attempts | `2` |
| `P2C_KEEP_CONDA_ENV` | Preserve Phase 2 env for debugging | unset |
| `P2C_PHASE2_ARTIFACT_MAX_MB` | Max single artifact collected | `50` |
| `BUDGET_MINUTES` | Override in `run_audit.sh` | `180` |

## Target Materials

Place audit targets under `Target/`:
- `Target/paper.pdf` — source PDF; Phase 1 can generate `Target/paper/full.md` automatically with MinerU
- `Target/paper/full.md` — optional cached paper Markdown
- `Target/code/` — the companion code repository to audit

`paper_with_code/` contains reference paper+code pairs used for development and batch testing.
