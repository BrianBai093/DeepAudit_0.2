from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from p2c.agents.base import BaseAgent
from p2c.schemas import ClaimItem, ClaimsIR

SYSTEM_PROMPT = (
    "You extract verifiable claims from a paper."
    "Do not invent numbers."
    "Output one JSON object with claims array and reason_codes."
)

USER_PROMPT_TEMPLATE = (
    "Input file: fingerprint/fingerprint.json (preferred) or output/paper.md when fingerprint is unavailable\n"
    "Output file: fingerprint/claims_ir.json\n"
    "Fields required per claim: claim_id,type,predicate,metric,target,baseline,conditions,aggregation,evidence_set,tolerance_policy."
)


class BuildClaimsIRAgent(BaseAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(name="build_claims_ir", *args, **kwargs)

    @staticmethod
    def _infer_code_verifiable(
        *,
        mapped_type: str,
        metric: str | None,
        target: float | None,
        **_,
    ) -> bool:
        if mapped_type == "result" and metric and target is not None:
            return True
        if mapped_type == "config":
            return True
        return False

    @staticmethod
    def _extract_metric_and_target(text: str) -> tuple[str | None, float | None]:
        lower = text.lower()
        metric = None
        for name in ["accuracy", "acc", "f1", "bleu", "auc", "loss", "precision", "recall", "mse", "mae", "perplexity", "rmse", "rouge"]:
            if name in lower:
                metric = name
                break

        m_pct = re.search(r"(\d+(?:\.\d+)?)\s*%", text)
        if m_pct:
            val = float(m_pct.group(1)) / 100.0
            # Sanity check: bounded metrics (accuracy, f1, etc.) should be in [0, 1] after conversion
            if metric in {"accuracy", "acc", "f1", "auc", "precision", "recall", "bleu", "rouge"} and val > 1.0:
                return metric, None  # Implausible — likely a sample count, not a metric
            return metric, val
        m_dec = re.search(r"\b(0\.\d+|1\.0+)\b", text)
        if m_dec:
            return metric, float(m_dec.group(1))

        # Fallback: try raw integer (e.g. "accuracy = 97")
        m_int = re.search(r"\b(\d+)\b", text)
        if m_int and metric:
            val = float(m_int.group(1))
            # Only accept if plausible for the metric type
            BOUNDED = {"accuracy", "acc", "f1", "auc", "precision", "recall", "bleu", "rouge"}
            if metric in BOUNDED and val > 100.0:
                return metric, None  # Sample count, not a metric value
            if metric in BOUNDED and 1.0 < val <= 100.0:
                return metric, val / 100.0  # Likely a percentage without % sign

        return metric, None

    def _claims_from_fingerprint(self, fingerprint: dict) -> tuple[list[ClaimItem], list[str]]:
        rows = [r for r in fingerprint.get("claims", []) if isinstance(r, dict)]
        if not rows:
            return [], []

        reason_codes = ["SOURCE_FINGERPRINT_CLAIMS"]
        out: list[ClaimItem] = []

        for idx, row in enumerate(rows, start=1):
            claim_id = str(row.get("id") or f"C{idx}")
            fact = str(row.get("fact") or "").strip()
            scope = str(row.get("scope") or "").strip()
            claim_type = str(row.get("claim_type") or "config")
            tol = row.get("tolerance") or {}

            metric, target = self._extract_metric_and_target(fact)
            mapped_type = "result" if claim_type == "result" else "config"

            # Preserve experiment context from fingerprint for Phase 3 alignment
            conditions: dict[str, Any] = {}
            if scope:
                conditions["scope"] = scope
            evidence_anchors = row.get("evidence_anchors") or {}
            if evidence_anchors.get("visual_anchor"):
                conditions["table_anchor"] = str(evidence_anchors["visual_anchor"])
            # Carry forward visual_data if present (e.g. table caption context)
            if evidence_anchors.get("visual_data"):
                conditions["visual_data"] = evidence_anchors["visual_data"]

            out.append(
                ClaimItem(
                    claim_id=claim_id,
                    type=mapped_type,
                    predicate=fact or "fingerprint claim",
                    metric=metric,
                    target=target,
                    baseline=None,
                    conditions=conditions,
                    aggregation="best" if mapped_type == "result" else None,
                    evidence_set=[str(evidence_anchors.get("text_anchor") or "fingerprint")],
                    tolerance_policy={
                        "abs_eps": float(tol.get("abs", 0.02) or 0.02),
                        "rel_eps": float(tol.get("rel", 0.03) or 0.03),
                    },
                    unverifiable_from_paper=(target is None and mapped_type == "result"),
                    code_verifiable=self._infer_code_verifiable(
                        mapped_type=mapped_type,
                        metric=metric,
                        target=target,
                    ),
                    reason_codes=[str(x) for x in row.get("reason_codes", [])],
                    notes=str(fingerprint.get("notes") or "") or None,
                )
            )

        return out, reason_codes

    def execute(self, ctx: dict) -> dict:
        fingerprint = self.artifacts.read_json("fingerprint/fingerprint.json")
        claims_from_fp, reason_codes = self._claims_from_fingerprint(fingerprint)
        if claims_from_fp:
            claims_ir = ClaimsIR(claims=claims_from_fp, reason_codes=reason_codes)
            self.artifacts.write_json("fingerprint/claims_ir.json", claims_ir.model_dump())
            return {"claims_ir": claims_ir.model_dump()}

        paper_md_out = Path(ctx.get("paper_md_out", ""))
        raw_text = ""
        if paper_md_out.exists():
            raw_text = paper_md_out.read_text(encoding="utf-8", errors="ignore")

        llm_schema = {
            "type": "object",
            "properties": {
                "claims": {"type": "array", "items": {"type": "object"}},
                "reason_codes": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["claims", "reason_codes"],
        }
        llm_user = USER_PROMPT_TEMPLATE + "\n\n" + raw_text[:12000]
        llm_data, llm_err = self.safe_chat_json(llm_schema, SYSTEM_PROMPT, llm_user)

        if llm_data:
            claims = []
            for i, obj in enumerate(llm_data.get("claims", []), start=1):
                if "claim_id" not in obj:
                    obj["claim_id"] = f"C{i}"
                try:
                    claims.append(ClaimItem(**obj))
                except Exception:  # noqa: BLE001
                    continue
            if not claims:
                rc = ["LLM_OUTPUT_EMPTY"]
            else:
                rc = list(llm_data.get("reason_codes", []))
                if llm_err:
                    rc.append("LLM_UNAVAILABLE")
            claims_ir = ClaimsIR(claims=claims, reason_codes=rc)
        else:
            claims_ir = ClaimsIR(
                claims=[],
                reason_codes=["LLM_UNAVAILABLE"],
            )

        self.artifacts.write_json("fingerprint/claims_ir.json", claims_ir.model_dump())
        return {"claims_ir": claims_ir.model_dump()}
