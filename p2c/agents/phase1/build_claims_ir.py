from __future__ import annotations

import re
from pathlib import Path

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

EXECUTABLE_METHOD_KEYS = [
    "learning rate",
    "lr",
    "batch",
    "epoch",
    "optimizer",
    "dropout",
    "seed",
    "layer",
    "backbone",
    "transformer",
    "resnet",
]


class BuildClaimsIRAgent(BaseAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(name="build_claims_ir", *args, **kwargs)

    @staticmethod
    def _infer_code_verifiable(
        *,
        mapped_type: str,
        metric: str | None,
        target: float | None,
        scope: str,
        predicate: str,
    ) -> bool:
        if mapped_type in {"absolute", "relative"} and metric and target is not None:
            return True
        text = f"{scope} {predicate}".lower()
        if any(k in text for k in EXECUTABLE_METHOD_KEYS):
            return True
        return False

    @staticmethod
    def _extract_metric_and_target(text: str) -> tuple[str | None, float | None]:
        lower = text.lower()
        metric = None
        if "accuracy" in lower or "acc" in lower:
            metric = "accuracy"
        elif "f1" in lower:
            metric = "f1"
        elif "bleu" in lower:
            metric = "bleu"
        elif "auc" in lower:
            metric = "auc"

        m_pct = re.search(r"(\d+(?:\.\d+)?)\s*%", text)
        if m_pct:
            return metric, float(m_pct.group(1)) / 100.0
        m_dec = re.search(r"\b(0\.\d+|1\.0+)\b", text)
        if m_dec:
            return metric, float(m_dec.group(1))
        return metric, None

    @staticmethod
    def _is_executable_methodological(fact: str, scope: str) -> bool:
        text = f"{fact} {scope}".lower()
        return any(k in text for k in EXECUTABLE_METHOD_KEYS)

    def _claims_from_fingerprint(self, fingerprint: dict) -> tuple[list[ClaimItem], list[str]]:
        rows = [r for r in fingerprint.get("claims", []) if isinstance(r, dict)]
        if not rows:
            return [], []

        prioritized: list[dict] = []
        deprioritized: list[dict] = []
        for row in rows:
            claim_type = str(row.get("claim_type") or "Unknown")
            fact = str(row.get("fact") or "")
            scope = str(row.get("scope") or "")
            if claim_type in {"Empirical", "Comparative"}:
                prioritized.append(row)
            elif claim_type == "Methodological" and self._is_executable_methodological(fact, scope):
                prioritized.append(row)
            else:
                deprioritized.append(row)

        source_rows = prioritized
        reason_codes = ["SOURCE_FINGERPRINT_CLAIMS"]
        if deprioritized:
            reason_codes.append("NON_EXECUTABLE_METHOD_CLAIMS_SKIPPED")

        out: list[ClaimItem] = []
        for idx, row in enumerate(source_rows, start=1):
            claim_id = str(row.get("id") or f"C{idx}")
            fact = str(row.get("fact") or "").strip()
            scope = str(row.get("scope") or "").strip()
            claim_type = str(row.get("claim_type") or "Unknown")
            comparator = row.get("comparator")
            verification_logic = str(row.get("verification_logic") or "unknown")
            tol = row.get("tolerance") or {}

            metric, target = self._extract_metric_and_target(fact)
            mapped_type = "absolute"
            if claim_type == "Comparative":
                mapped_type = "relative"
            elif verification_logic == "trend_match":
                mapped_type = "ranking"
            elif claim_type == "Methodological":
                mapped_type = "other"

            baseline = None
            if comparator:
                _, baseline = self._extract_metric_and_target(str(comparator))

            out.append(
                ClaimItem(
                    claim_id=claim_id,
                    type=mapped_type,
                    predicate=fact or "fingerprint claim",
                    metric=metric,
                    target=target,
                    baseline=baseline,
                    conditions={"scope": scope} if scope else {},
                    aggregation="best" if mapped_type == "absolute" else "average",
                    evidence_set=[str((row.get("evidence_anchors") or {}).get("text_anchor") or "fingerprint")],
                    tolerance_policy={
                        "abs_eps": float(tol.get("abs", 0.02) or 0.02),
                        "rel_eps": float(tol.get("rel", 0.03) or 0.03),
                    },
                    unverifiable_from_paper=(target is None and mapped_type != "other"),
                    code_verifiable=self._infer_code_verifiable(
                        mapped_type=mapped_type,
                        metric=metric,
                        target=target,
                        scope=scope,
                        predicate=fact,
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
