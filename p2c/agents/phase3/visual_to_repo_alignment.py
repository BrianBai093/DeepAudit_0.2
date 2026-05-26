"""Align paper visual elements to repo-produced figure/data artifacts."""

from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Any

from p2c.agents.base import BaseAgent
from p2c.schemas import VisualRepoAlignmentDoc, VisualRepoAlignmentItem

MODEL_ALIASES = {
    "autoencoder": ("auto encoder", "auto-encoder", "autoencoder", "ae"),
    "xgboost": ("xgboost", "xgb"),
    "random_forest": ("random forest", "random_forest", "randomforest", "rf"),
    "logistic_regression": ("logistic regression", "logistic_regression", "lr", "lor"),
    "knn": ("knn", "k-nearest", "nearest neighbor"),
    "svc": ("svc", "svm"),
    "linear_svc": ("lsvc", "linear svc", "linear_svc"),
    "mlp": ("mlp", "multi layer", "multilayer"),
    "decision_tree": ("decision tree", "dt", "gini", "entropy"),
    "gbm": ("gbm", "gradient boosting"),
    "adaboost": ("adaboost", "ada boost"),
}

SAMPLING_ALIASES = {
    "under-sampling": ("under-sampling", "under sampling", "undersampling", "under-sampled"),
    "over-sampling": ("over-sampling", "over sampling", "oversampling", "over-sampled"),
}


class VisualToRepoAlignmentAgent(BaseAgent):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(name="visual_to_repo_alignment", *args, **kwargs)

    def execute(self, ctx: dict[str, Any]) -> dict[str, Any]:
        visual_doc = self.artifacts.read_json("fingerprint/visual_elements.json")
        elements = [row for row in visual_doc.get("elements", []) if isinstance(row, dict)]
        repo_dir = Path(str(ctx.get("repo_dir") or "")).expanduser()
        if repo_dir and not repo_dir.is_absolute():
            repo_dir = repo_dir.resolve()
        inventory = _repo_visual_inventory(repo_dir) if repo_dir else []

        alignments = [
            _align_element_to_inventory(element, inventory)
            for element in elements
        ]
        doc = VisualRepoAlignmentDoc(
            alignments=alignments,
            reason_codes=["VISUAL_REPO_ALIGNMENT_COMPLETE"],
        )
        self.artifacts.write_json("results/visual_to_repo_alignment.json", doc.model_dump())
        matched = sum(1 for row in alignments if row.status == "MATCH")
        self.log("DONE", f"aligned {matched}/{len(alignments)} visual elements to repo artifacts")
        return {"visual_to_repo_alignment": doc.model_dump()}


def _repo_visual_inventory(repo_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    figures_dir = repo_dir / "figures"
    if figures_dir.exists():
        for path in sorted(figures_dir.glob("*.png")):
            semantics = _semantics_from_text(path.stem)
            rows.append({
                "path": str(path.resolve()),
                "artifact_type": "image",
                "chart_family": _chart_family_from_text(path.name),
                **semantics,
            })

    metrics_dir = repo_dir / "metrics"
    if metrics_dir.exists():
        for path in sorted(metrics_dir.glob("*.csv")):
            columns: list[str] = []
            try:
                with path.open("r", encoding="utf-8", newline="") as handle:
                    reader = csv.reader(handle)
                    columns = next(reader, [])
            except Exception:  # noqa: BLE001
                columns = []
            semantics = _semantics_from_text(f"{path.stem} {' '.join(columns)}")
            rows.append({
                "path": str(path.resolve()),
                "artifact_type": "csv",
                "chart_family": _chart_family_from_text(f"{path.name} {' '.join(columns)}"),
                "columns": columns,
                **semantics,
            })
    return rows


def _align_element_to_inventory(element: dict[str, Any], inventory: list[dict[str, Any]]) -> VisualRepoAlignmentItem:
    required = _element_semantics(element)
    candidates: list[tuple[float, dict[str, Any], list[str]]] = []
    for artifact in inventory:
        score, reasons = _score_candidate(required, artifact)
        if score >= 0.80 and not reasons:
            candidates.append((score, artifact, reasons))

    if candidates:
        score, artifact, _reasons = sorted(candidates, key=lambda row: row[0], reverse=True)[0]
        return VisualRepoAlignmentItem(
            element_id=str(element.get("element_id") or ""),
            status="MATCH",
            repo_artifact_path=artifact.get("path"),
            artifact_type=artifact.get("artifact_type"),
            confidence=round(score, 3),
            matched_model_names=sorted(required["models"] & set(artifact.get("models", []))),
            matched_sampling_strategy=required["sampling_strategy"] or artifact.get("sampling_strategy"),
            matched_metric_names=sorted(required["metrics"] & set(artifact.get("metrics", []))),
            mismatch_reasons=[],
            reason_codes=["STRICT_VISUAL_MATCH"],
        )

    mismatch_reasons = _best_mismatch_reasons(required, inventory)
    return VisualRepoAlignmentItem(
        element_id=str(element.get("element_id") or ""),
        status="NO_MATCH",
        confidence=0.0,
        mismatch_reasons=mismatch_reasons,
        reason_codes=["STRICT_NO_MATCH"],
    )


def _score_candidate(required: dict[str, Any], artifact: dict[str, Any]) -> tuple[float, list[str]]:
    reasons: list[str] = []
    score = 0.0

    if required["chart_family"] and artifact.get("chart_family") == required["chart_family"]:
        score += 0.30
    elif required["chart_family"]:
        reasons.append(f"chart_family mismatch: paper={required['chart_family']} repo={artifact.get('chart_family')}")

    artifact_metrics = set(artifact.get("metrics", []))
    if required["metrics"] and required["metrics"] <= artifact_metrics:
        score += 0.25
    elif required["metrics"]:
        reasons.append(f"metric mismatch: paper={sorted(required['metrics'])} repo={sorted(artifact_metrics)}")

    artifact_models = set(artifact.get("models", []))
    if required["models"]:
        if required["models"] <= artifact_models:
            score += 0.30
        else:
            reasons.append(f"model mismatch: paper={sorted(required['models'])} repo={sorted(artifact_models)}")
    else:
        score += 0.10

    paper_sampling = required["sampling_strategy"]
    repo_sampling = artifact.get("sampling_strategy")
    if paper_sampling:
        if paper_sampling == repo_sampling:
            score += 0.15
        else:
            reasons.append(f"sampling mismatch: paper={paper_sampling} repo={repo_sampling or 'none'}")
    else:
        score += 0.05

    return score, reasons


def _best_mismatch_reasons(required: dict[str, Any], inventory: list[dict[str, Any]]) -> list[str]:
    if not inventory:
        return ["repo has no visual artifacts to align"]
    scored = sorted(
        (_score_candidate(required, artifact)[0], _score_candidate(required, artifact)[1])
        for artifact in inventory
    )
    reasons = scored[-1][1] if scored else []
    if reasons:
        return reasons
    return ["no repo artifact satisfied strict visual matching"]


def _element_semantics(element: dict[str, Any]) -> dict[str, Any]:
    text = " ".join([
        str(element.get("caption") or ""),
        str(element.get("visual_anchor") or ""),
        " ".join(str(x) for x in element.get("legend_entries", []) if x),
        " ".join(str(x) for x in element.get("model_names", []) if x),
        str(element.get("sampling_strategy") or ""),
        str(element.get("axis_labels") or ""),
    ])
    semantics = _semantics_from_text(text)
    if isinstance(element.get("model_names"), list):
        semantics["models"].update(_normalize_model_name(str(x)) for x in element["model_names"] if str(x).strip())
        semantics["models"].discard("")
    sampling = _normalize_sampling(str(element.get("sampling_strategy") or "")) or semantics["sampling_strategy"]
    semantics["sampling_strategy"] = sampling
    chart_family = _chart_family_from_text(text) or _chart_family_from_chart_type(str(element.get("chart_type") or ""))
    semantics["chart_family"] = chart_family
    return semantics


def _semantics_from_text(text: str) -> dict[str, Any]:
    normalized = _norm(text)
    models = {
        canonical
        for canonical, aliases in MODEL_ALIASES.items()
        if any(_has_tokenish(normalized, alias) for alias in aliases)
    }
    metrics: set[str] = set()
    if (
        any(token in normalized for token in ("roc", "aucroc", "auroc"))
        or ("false positive" in normalized and "true positive" in normalized)
    ):
        metrics.add("roc_auc")
    if "pr_auc" in normalized or "pr-auc" in normalized or "precision recall" in normalized:
        metrics.add("pr_auc")
    if "mse" in normalized:
        metrics.add("mse")
    if "confusion" in normalized or {"tp", "fp", "tn", "fn"} <= set(normalized.split()):
        metrics.add("confusion_matrix")
    if "classification report" in normalized or "precision" in normalized or "recall" in normalized or "f1" in normalized:
        metrics.update({"precision", "recall", "f1"})
    return {
        "models": models,
        "metrics": metrics,
        "sampling_strategy": _normalize_sampling(text),
        "chart_family": _chart_family_from_text(text),
    }


def _chart_family_from_text(text: str) -> str | None:
    normalized = _norm(text)
    if (
        "roc" in normalized
        or "aucroc" in normalized
        or "auroc" in normalized
        or ("false positive" in normalized and "true positive" in normalized)
    ):
        return "roc_curve"
    if "pr_" in normalized or "pr-" in normalized or "precision recall" in normalized:
        return "pr_curve"
    if "confusion" in normalized:
        return "confusion_matrix"
    if "classification report" in normalized:
        return "classification_report"
    if "mse" in normalized:
        return "mse_distribution"
    if "class_distribution" in normalized or "class distribution" in normalized:
        return "class_distribution"
    return None


def _chart_family_from_chart_type(chart_type: str) -> str | None:
    lowered = chart_type.lower()
    if lowered == "heatmap":
        return "confusion_matrix"
    if lowered == "table":
        return "classification_report"
    return lowered or None


def _normalize_model_name(raw: str) -> str:
    normalized = _norm(raw)
    for canonical, aliases in MODEL_ALIASES.items():
        if any(_has_tokenish(normalized, alias) for alias in aliases):
            return canonical
    return ""


def _normalize_sampling(raw: str) -> str | None:
    normalized = _norm(raw)
    for canonical, aliases in SAMPLING_ALIASES.items():
        if any(alias in normalized for alias in aliases):
            return canonical
    return None


def _norm(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(text or "").lower()).strip()


def _has_tokenish(text: str, alias: str) -> bool:
    alias_norm = _norm(alias)
    if not alias_norm:
        return False
    return re.search(rf"(^| )({re.escape(alias_norm)})( |$)", text) is not None
