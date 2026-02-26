from __future__ import annotations

import os
import re
from pathlib import Path

from p2c.agents.base import BaseAgent
from p2c.agents.phase1.fingerprint_prompt_templates import FILTER_SYSTEM_PROMPT, FILTER_USER_PROMPT_TEMPLATE
from p2c.schemas import (
    Fingerprint,
    FingerprintClaim,
    FingerprintConfigurations,
    FingerprintEvidenceAnchors,
    FingerprintMetadata,
    FingerprintTolerance,
)

DOI_RE = re.compile(r"10\.\d{4,9}/[-._;()/:A-Z0-9]+", flags=re.I)
ARXIV_RE = re.compile(r"arXiv:\s*\d{4}\.\d{4,5}", flags=re.I)
YEAR_RE = re.compile(r"\b(19\d{2}|20\d{2})\b")
URL_RE = re.compile(r"https?://[^\s)]+|github\.com/[^\s)]+", flags=re.I)
NUMBER_RE = re.compile(r"\d+(?:\.\d+)?")
DATASET_ACTIONABLE_HINTS = [
    "split",
    "train",
    "test",
    "validation",
    "preprocess",
    "normalize",
    "augmentation",
    "tokenization",
    "partition",
    "fold",
    "version",
    "class",
]
DATASET_NOISE_HINTS = [
    "future work",
    "leading cause",
    "mortality",
    "importance",
    "literature",
    "benchmark for further development",
    "addresses the need",
]


class ExtractFingerprintFilterAgent(BaseAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(name="extract_fingerprint_filter", *args, **kwargs)

    @staticmethod
    def _norm(text: str | None) -> str:
        return re.sub(r"\s+", " ", (text or "").strip().lower())

    @staticmethod
    def _fact_core(text: str) -> str:
        lower = re.sub(r"\s+", " ", text.lower()).strip()
        return re.sub(r"\d+(?:\.\d+)?", "#", lower)

    def _cluster_key(self, row: dict) -> str:
        facet = self._norm(str(row.get("facet") or "other"))
        metric = self._norm(str(row.get("metric_name") or ""))
        entity = self._norm(str(row.get("entity") or ""))
        dataset_scope = self._norm(str(row.get("dataset_scope") or row.get("scope") or ""))
        fact_core = self._fact_core(str(row.get("fact") or row.get("criterion") or ""))
        numeric_value = row.get("metric_value")
        if numeric_value is None:
            num_match = NUMBER_RE.search(str(row.get("fact") or ""))
            numeric_value = num_match.group(0) if num_match else ""
        comparator = self._norm(str(row.get("comparator") or ""))

        # For metric rows, prioritize semantic signature over raw wording so paraphrases dedup.
        if facet == "metric":
            fact_core = metric or "metric"
        return "|".join(
            [
                facet,
                metric,
                entity,
                dataset_scope,
                fact_core,
                str(numeric_value),
                comparator,
            ]
        )

    @staticmethod
    def _select_rep_heuristic(cluster_rows: list[dict]) -> int:
        best_idx = 0
        best_score = -10_000.0
        for i, row in enumerate(cluster_rows):
            score = 0.0
            scope = str(row.get("scope") or "")
            fact = str(row.get("fact") or "")
            if row.get("metric_value") is not None:
                score += 3.0
            if scope and "from paper context" not in scope.lower():
                score += 2.0
            if row.get("facet") in {"hyperparameter", "architecture", "algorithm", "metric"}:
                score += 2.0
            score += min(len(fact), 200) / 200.0
            if score > best_score:
                best_score = score
                best_idx = i
        return best_idx

    @staticmethod
    def _claim_type(row: dict) -> str:
        facet = str(row.get("facet") or "other")
        comparator = row.get("comparator")
        fact = str(row.get("fact") or "")
        has_metric_value = row.get("metric_value") is not None or bool(re.search(r"\d+(?:\.\d+)?\s*%", fact))
        has_comparative_text = bool(
            re.search(r"\b(vs\\.?|versus|better than|outperform|highest|lower than|improves over)\b", fact, flags=re.I)
        )
        if facet == "metric" and (comparator or has_comparative_text):
            return "Comparative"
        if facet == "metric" and has_metric_value:
            return "Empirical"
        return "Methodological"

    @staticmethod
    def _verification_logic(claim_type: str) -> str:
        if claim_type == "Comparative":
            return "greater_than_margin"
        return "exact_match"

    @staticmethod
    def _extract_hparams(text: str) -> dict[str, object]:
        lower = text.lower()
        out: dict[str, object] = {}

        lr = re.search(r"(?:learning rate|\blr\b)\s*(?:=|of|:)?\s*(\d+(?:\.\d+)?(?:e-?\d+)?)", lower)
        if lr:
            try:
                out["learning_rate"] = float(lr.group(1))
            except ValueError:
                out["learning_rate"] = lr.group(1)

        bs = re.search(r"batch(?:\s+size)?\s*(?:=|of|:)?\s*(\d+)", lower)
        if bs:
            out["batch_size"] = int(bs.group(1))

        ep = re.search(r"epochs?\s*(?:=|of|:)?\s*(\d+)", lower)
        if ep:
            out["epochs"] = int(ep.group(1))

        dr = re.search(r"dropout\s*(?:=|of|:)?\s*(\d+(?:\.\d+)?)", lower)
        if dr:
            out["dropout"] = float(dr.group(1))

        seed = re.search(r"seed\s*(?:=|of|:)?\s*(\d+)", lower)
        if seed:
            out["seed"] = int(seed.group(1))

        opt = re.search(r"optimizer\s*(?:=|is|:)?\s*([a-z0-9_-]+)", lower)
        if opt:
            out["optimizer"] = opt.group(1)

        return out

    @staticmethod
    def _append_unique_dict(rows: list[dict], item: dict) -> None:
        norm = (item.get("detail", "").strip().lower(), item.get("scope", "").strip().lower())
        seen = {(x.get("detail", "").strip().lower(), x.get("scope", "").strip().lower()) for x in rows}
        if norm not in seen:
            rows.append(item)

    def _build_configurations(self, selected_rows: list[dict]) -> FingerprintConfigurations:
        dataset_specs: list[dict] = []
        hyperparameters: dict[str, object] = {}
        model_arch: list[str] = []
        environment: dict[str, object] = {}
        evaluation_metrics: list[str] = []

        for row in selected_rows:
            fact = str(row.get("fact") or "")
            scope = str(row.get("scope") or "")
            facet = str(row.get("facet") or "other")
            source = f"{fact} {scope}".strip()

            if facet in {"dataset_task", "preprocess"}:
                self._append_unique_dict(
                    dataset_specs,
                    {
                        "facet": facet,
                        "detail": fact,
                        "scope": scope,
                    },
                )

            for k, v in self._extract_hparams(source).items():
                hyperparameters[k] = v

            if facet == "architecture":
                item = fact.strip()
                if item and item not in model_arch:
                    model_arch.append(item)

            lower = source.lower()
            if facet == "environment":
                if "pytorch" in lower:
                    environment["framework"] = "pytorch"
                if "tensorflow" in lower:
                    environment["framework"] = "tensorflow"
                cuda = re.search(r"cuda\s*(\d+(?:\.\d+)?)", lower)
                if cuda:
                    environment["cuda"] = cuda.group(1)
                if "a100" in lower:
                    environment["hardware"] = "A100"
                if "v100" in lower:
                    environment["hardware"] = "V100"

            metric_name = row.get("metric_name")
            if isinstance(metric_name, str) and metric_name and metric_name not in evaluation_metrics:
                evaluation_metrics.append(metric_name)

        return FingerprintConfigurations(
            dataset_specs=dataset_specs,
            hyperparameters=hyperparameters,
            model_arch=model_arch,
            environment=environment,
            evaluation_metrics=evaluation_metrics,
        )

    @staticmethod
    def _is_actionable_row(row: dict) -> bool:
        source_type = str(row.get("source_type") or "")
        if source_type == "table_metric":
            return True

        facet = str(row.get("facet") or "other")
        fact = str(row.get("fact") or "")
        scope = str(row.get("scope") or "")
        text = f"{fact} {scope}".lower()
        has_number = bool(re.search(r"\d", text))

        if facet == "metric":
            if row.get("metric_value") is not None:
                return True
            return bool(
                re.search(
                    r"\b(vs\\.?|versus|better than|outperform|highest|lower than|improves over)\b",
                    text,
                    flags=re.I,
                )
            )
        if facet == "hyperparameter":
            return has_number
        if facet == "architecture":
            return has_number or any(k in text for k in ["resnet", "transformer", "cnn", "lstm", "svm"])
        if facet == "preprocess":
            return any(k in text for k in ["normalize", "augmentation", "tokenization", "preprocess"])
        if facet == "environment":
            return has_number or any(k in text for k in ["pytorch", "tensorflow", "cuda", "gpu", "python"])
        if facet == "algorithm":
            has_formula = any(k in text for k in ["=", "\\sum", "loss", "objective", "fedavg", "fedadam", "fedyogi"])
            return has_formula
        if facet in {"dataset_task"}:
            if any(n in text for n in DATASET_NOISE_HINTS):
                return False
            return any(h in text for h in DATASET_ACTIONABLE_HINTS)
        if facet in {"other"}:
            return False
        return False

    def execute(self, ctx: dict) -> dict:
        atomic = self.artifacts.read_json("fingerprint/atomic_criteria.json")
        criteria = [c for c in atomic.get("criteria", []) if isinstance(c, dict)]
        filtered_criteria: list[dict] = []
        dropped_non_actionable = 0
        for row in criteria:
            if self._is_actionable_row(row):
                filtered_criteria.append(row)
            else:
                dropped_non_actionable += 1
        criteria = filtered_criteria

        if not criteria:
            reason = "NO_ATOMIC_CRITERIA_AFTER_FILTERING" if dropped_non_actionable else "NO_ATOMIC_CRITERIA"
            fingerprint = Fingerprint(reason_codes=[reason])
            self.artifacts.write_json("fingerprint/filter_clusters.json", {"clusters": [], "reason_codes": ["NO_ATOMIC_CRITERIA"]})
            self.artifacts.write_json("fingerprint/filter_selected.json", {"selected_indices": [], "reason_codes": [reason]})
            self.artifacts.write_json("fingerprint/fingerprint.json", fingerprint.model_dump())
            return {"fingerprint": fingerprint.model_dump()}

        reason_codes: list[str] = []
        if dropped_non_actionable:
            reason_codes.append("FILTER_DROPPED_NON_ACTIONABLE")
        clusters: dict[str, list[int]] = {}
        table_selected: list[int] = []
        table_seen: set[tuple[str, str, str, str]] = set()

        for idx, row in enumerate(criteria):
            if row.get("source_type") == "table_metric":
                table_key = (
                    str(row.get("table_anchor") or ""),
                    str(row.get("entity") or row.get("model") or ""),
                    str(row.get("metric_name") or ""),
                    str(row.get("metric_value") or row.get("value_raw") or ""),
                )
                if table_key not in table_seen:
                    table_seen.add(table_key)
                    table_selected.append(idx)
                continue
            key = self._cluster_key(row)
            clusters.setdefault(key, []).append(idx)

        llm_budget = int(os.getenv("P2C_FILTER_LLM_CLUSTER_BUDGET", "20"))
        llm_calls = 0

        selected_indices: list[int] = []
        cluster_debug: list[dict] = []

        for key, indices in clusters.items():
            cluster_rows = [criteria[i] for i in indices]
            selected_local = 0
            selection_reason = "heuristic"

            if len(indices) > 1 and llm_calls < llm_budget:
                llm_calls += 1
                group = "\n".join(f"{i}. {cluster_rows[i].get('criterion') or cluster_rows[i].get('fact') or ''}" for i in range(len(cluster_rows)))
                llm_schema = {
                    "type": "object",
                    "properties": {
                        "selected_index": {"type": "integer"},
                        "reason": {"type": "string"},
                    },
                    "required": ["selected_index", "reason"],
                }
                llm_user = FILTER_USER_PROMPT_TEMPLATE.format(cluster_of_similar_criteria=group)
                llm_data, llm_err = self.safe_chat_json(schema=llm_schema, system=FILTER_SYSTEM_PROMPT, user=llm_user)
                if llm_data and isinstance(llm_data.get("selected_index"), int):
                    picked = int(llm_data["selected_index"])
                    if 0 <= picked < len(cluster_rows):
                        selected_local = picked
                        selection_reason = str(llm_data.get("reason") or "llm")
                    else:
                        selected_local = self._select_rep_heuristic(cluster_rows)
                        selection_reason = "llm_out_of_range_fallback"
                else:
                    selected_local = self._select_rep_heuristic(cluster_rows)
                    selection_reason = "llm_empty_fallback"

                if llm_err:
                    reason_codes.append("LLM_UNAVAILABLE")
            else:
                selected_local = self._select_rep_heuristic(cluster_rows)

            selected_global = indices[selected_local]
            selected_indices.append(selected_global)
            cluster_debug.append(
                {
                    "cluster_key": key,
                    "candidate_indices": indices,
                    "selected_index": selected_global,
                    "selection_reason": selection_reason,
                }
            )

        selected_indices = sorted(set(selected_indices).union(table_selected))

        selected_rows = [criteria[i] for i in selected_indices]

        paper_md = Path(ctx["paper_md_out"])
        raw_text = paper_md.read_text(encoding="utf-8", errors="ignore") if paper_md.exists() else ""
        doi = DOI_RE.search(raw_text)
        arxiv = ARXIV_RE.search(raw_text)
        urls = URL_RE.findall(raw_text)
        year_candidates = [int(m.group(1)) for m in YEAR_RE.finditer(raw_text)]
        year = None
        if year_candidates:
            modern = [y for y in year_candidates if 1990 <= y <= 2035]
            year = min(modern) if modern else min(year_candidates)

        metadata = FingerprintMetadata(
            paper_id=(doi.group(0) if doi else (arxiv.group(0) if arxiv else None)),
            repository_url=(urls[0] if urls else None),
            venue=None,
            year=year,
        )

        claims: list[FingerprintClaim] = []
        for out_idx, crit_idx in enumerate(selected_indices, start=1):
            row = criteria[crit_idx]
            fact = str(row.get("fact") or row.get("criterion") or "").strip() or "unknown fact"
            scope = str(row.get("scope") or "from paper context").strip() or "from paper context"
            claim_type = self._claim_type(row)
            verification_logic = self._verification_logic(claim_type)

            is_metric = claim_type in {"Empirical", "Comparative"}
            tolerance = FingerprintTolerance(
                abs=0.005 if is_metric else None,
                rel=0.02 if is_metric else None,
                text="default metric tolerance" if is_metric else "exact config/value match",
            )

            claims.append(
                FingerprintClaim(
                    id=f"claim_{out_idx:02d}",
                    claim_type=claim_type,
                    fact=fact,
                    scope=scope,
                    comparator=row.get("comparator"),
                    verification_logic=verification_logic,
                    tolerance=tolerance,
                    evidence_anchors=FingerprintEvidenceAnchors(
                        text_anchor=f"atomic_criteria[{crit_idx}]",
                        visual_anchor=row.get("table_anchor") if row.get("source_type") == "table_metric" else None,
                        visual_data={},
                    ),
                    reason_codes=[str(x) for x in row.get("reason_codes", [])],
                )
            )

        configurations = self._build_configurations(selected_rows)

        fingerprint = Fingerprint(
            fingerprint_id=metadata.paper_id,
            metadata=metadata,
            configurations=configurations,
            claims=claims,
            reason_codes=sorted(set(reason_codes)),
            notes="Generated via strict guide -> atomic -> filter pipeline aligned with paper_fingerprint_schema.md",
        )

        self.artifacts.write_json("fingerprint/filter_clusters.json", {"clusters": cluster_debug, "reason_codes": []})
        self.artifacts.write_json("fingerprint/filter_selected.json", {"selected_indices": selected_indices, "reason_codes": []})
        self.artifacts.write_json("fingerprint/fingerprint.json", fingerprint.model_dump())
        return {"fingerprint": fingerprint.model_dump()}
