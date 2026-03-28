from __future__ import annotations

import json
import os
import re

from p2c.agents.base import BaseAgent
from p2c.agents.phase1.fingerprint_prompt_templates import ATOMIC_SYSTEM_PROMPT, ATOMIC_USER_PROMPT_TEMPLATE

FACT_SCOPE_RE = re.compile(r"<fact>(.*?)</fact>.*?<scope>(.*?)</scope>", flags=re.I | re.S)
PERCENT_RE = re.compile(r"(\d+(?:\.\d+)?)\s*%")
DECIMAL_RE = re.compile(r"\b(0\.\d+|1\.0+)\b")
NUMBER_RE = re.compile(r"\d+(?:\.\d+)?")
TR_RE = re.compile(r"<tr>(.*?)</tr>", flags=re.I | re.S)
TD_RE = re.compile(r"<t[dh]>(.*?)</t[dh]>", flags=re.I | re.S)
TAG_RE = re.compile(r"<[^>]+>")
TABLE_ID_RE = re.compile(r"\btable\s+([ivxlcdm\d]+)", flags=re.I)

FACET_KEYWORDS = {
    "metric_result": [
        "accuracy", "acc", "f1", "auc", "bleu", "loss", "precision", "recall",
        "mse", "mae", "perplexity", "rmse", "map", "ndcg", "rouge", "error rate",
    ],
    "execution_param": [
        "learning rate", "lr", "batch", "epoch", "dropout", "weight decay", "seed",
        "optimizer", "dataset", "split", "train", "test", "validation",
        "pytorch", "tensorflow", "cuda", "gpu", "python",
    ],
}


class ExtractFingerprintAtomicAgent(BaseAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(name="extract_fingerprint_atomic", *args, **kwargs)

    @staticmethod
    def _clean_cell(text: str) -> str:
        text = TAG_RE.sub("", text)
        text = text.replace("&nbsp;", " ")
        return re.sub(r"\s+", " ", text).strip()

    @staticmethod
    def _extract_table_anchor(text: str) -> str | None:
        m = TABLE_ID_RE.search(text)
        if not m:
            return None
        return f"Table {m.group(1)}"

    @staticmethod
    def _parse_llm_list(text: str) -> list[dict]:
        if not text:
            return []
        candidate = text.strip()

        def to_rows(obj: object) -> list[dict]:
            if isinstance(obj, list):
                return [x for x in obj if isinstance(x, dict)]
            if isinstance(obj, dict):
                rows = obj.get("criteria")
                if isinstance(rows, list):
                    return [x for x in rows if isinstance(x, dict)]
            return []

        try:
            parsed = json.loads(candidate)
            rows = to_rows(parsed)
            if rows:
                return rows
        except Exception:  # noqa: BLE001
            pass

        match = re.search(r"\[[\s\S]*\]", candidate)
        if not match:
            return []
        try:
            parsed = json.loads(match.group(0))
            return to_rows(parsed)
        except Exception:  # noqa: BLE001
            return []

    @staticmethod
    def _facet_of(text: str) -> str:
        lower = text.lower()
        for facet, words in FACET_KEYWORDS.items():
            if any(w in lower for w in words):
                return facet
        return "other"

    @staticmethod
    def _extract_metric_name(text: str) -> str | None:
        lower = text.lower()
        for name in [
            "accuracy", "f1", "auc", "bleu", "loss", "precision", "recall",
            "mse", "mae", "perplexity", "rmse", "rouge", "error rate",
        ]:
            if name in lower:
                return name
        return None

    @staticmethod
    def _extract_metric_value(text: str) -> tuple[float | None, str | None]:
        m = PERCENT_RE.search(text)
        if m:
            return float(m.group(1)), "%"
        d = DECIMAL_RE.search(text)
        if d:
            return float(d.group(1)), "ratio"
        return None, None

    @staticmethod
    def _contains_malformed_numeric(text: str) -> bool:
        lower = text.lower()
        if "%$" in lower or "$%" in lower:
            return True
        if re.search(r"\$\s*\d", lower) and "%" in lower:
            return True
        if re.search(r"\d\s*\\%\$", lower):
            return True
        return False

    @staticmethod
    def _fact_scope_from_row(row: dict) -> tuple[str | None, str | None, str | None]:
        criterion = str(row.get("criterion") or "").strip()
        fact = row.get("fact")
        scope = row.get("scope")

        if criterion:
            m = FACT_SCOPE_RE.search(criterion)
            if m:
                f = re.sub(r"\s+", " ", m.group(1)).strip()
                s = re.sub(r"\s+", " ", m.group(2)).strip()
                return criterion, f, s

        if isinstance(fact, str) and fact.strip() and isinstance(scope, str) and scope.strip():
            f = re.sub(r"\s+", " ", fact).strip()
            s = re.sub(r"\s+", " ", scope).strip()
            c = f"<fact>{f}</fact> <scope>{s}</scope>"
            return c, f, s

        return None, None, None

    def _expand_table_unit(self, unit: dict) -> tuple[list[dict], list[dict]]:
        text = str(unit.get("text") or "")
        rows = TR_RE.findall(text)
        parsed_rows: list[list[str]] = []
        for row in rows:
            cells = [self._clean_cell(c) for c in TD_RE.findall(row)]
            if cells:
                parsed_rows.append(cells)

        if not parsed_rows:
            return [], [
                {
                    "unit_id": unit.get("unit_id"),
                    "raw": text[:2000],
                    "reason_codes": ["TABLE_PARSE_EMPTY"],
                }
            ]

        out: list[dict] = []
        rej: list[dict] = []
        table_anchor = self._extract_table_anchor(text)

        for i, row in enumerate(parsed_rows):
            metric_cell = row[0] if row else ""
            metric_name = self._extract_metric_name(metric_cell or " ".join(row))
            if not metric_name:
                continue

            header = parsed_rows[i - 1] if i - 1 >= 0 else []
            for col in range(1, len(row)):
                model = header[col] if header and col < len(header) else f"col_{col}"
                value_raw = row[col]
                if not value_raw:
                    continue
                value, unit_name = self._extract_metric_value(value_raw)
                if value is None:
                    raw_num = NUMBER_RE.search(value_raw)
                    if raw_num:
                        value = float(raw_num.group(0))
                        unit_name = "%" if metric_name in {"accuracy", "f1", "auc", "precision", "recall"} else "value"
                if value is None:
                    rej.append(
                        {
                            "unit_id": unit.get("unit_id"),
                            "raw": value_raw,
                            "reason_codes": ["TABLE_VALUE_PARSE_FAILED"],
                        }
                    )
                    continue

                fact = f"{model} {metric_name} = {value_raw if unit_name == '%' else value}"
                scope = "from table in paper"
                out.append(
                    {
                        "criterion": f"<fact>{fact}</fact> <scope>{scope}</scope>",
                        "fact": fact,
                        "scope": scope,
                        "facet": "metric_result",
                        "source_type": "table_metric",
                        "metric_name": metric_name,
                        "metric_value": value,
                        "metric_unit": unit_name,
                        "entity": model,
                        "comparator": None,
                        "dataset_scope": None,
                        "table_anchor": table_anchor,
                        "input_unit_id": unit.get("unit_id"),
                        "reason_codes": ["TABLE_EXPANDED"],
                    }
                )

        return out, rej

    def execute(self, ctx: dict) -> dict:
        guide = self.artifacts.read_json("fingerprint/guide_sentences.json")
        units = [u for u in guide.get("units", []) if isinstance(u, dict)]
        selected_unit_ids = [x for x in guide.get("selected_unit_ids", []) if isinstance(x, str)]

        # Backward compatibility with older guide format.
        if not units:
            sentences = guide.get("sentences", [])
            units = [
                {
                    "unit_id": f"legacy_s_{i}",
                    "type": "sentence",
                    "text": s,
                    "origin_indices": [i],
                }
                for i, s in enumerate(sentences)
                if isinstance(s, str)
            ]
            selected_unit_ids = [
                f"legacy_s_{i}"
                for i in guide.get("selected_sentence_indices", [])
                if isinstance(i, int) and 0 <= i < len(units)
            ]

        unit_map = {str(u.get("unit_id")): u for u in units if u.get("unit_id")}
        selected_units = [unit_map[uid] for uid in selected_unit_ids if uid in unit_map]

        if not selected_units:
            payload = {"criteria": [], "reason_codes": ["NO_INPUT_GUIDE_UNITS"], "selected_unit_ids": []}
            self.artifacts.write_json("fingerprint/atomic_criteria.json", payload)
            self.artifacts.write_json(
                "fingerprint/atomic_rejected.json",
                {"rejected": [], "reason_codes": ["NO_INPUT_GUIDE_UNITS"]},
            )
            return {"atomic_criteria": payload}

        llm_budget = int(os.getenv("P2C_ATOMIC_LLM_SENTENCE_BUDGET", "8"))
        llm_calls = 0
        llm_enabled = True

        accepted: list[dict] = []
        rejected: list[dict] = []
        reason_codes: list[str] = []

        self.log("PROGRESS", f"atomic stage processing {len(selected_units)} units (llm_budget={llm_budget})")

        for idx, unit in enumerate(selected_units, start=1):
            if idx == 1 or idx % 25 == 0 or idx == len(selected_units):
                self.log("PROGRESS", f"atomic progress {idx}/{len(selected_units)}")

            unit_id = str(unit.get("unit_id"))
            text = str(unit.get("text") or "")
            unit_type = str(unit.get("type") or "sentence")

            if unit_type == "table_block":
                rows, rej = self._expand_table_unit(unit)
                accepted.extend(rows)
                rejected.extend(rej)
                continue

            candidates: list[dict] = []
            if llm_enabled and llm_calls < llm_budget:
                llm_calls += 1
                prompt = ATOMIC_USER_PROMPT_TEMPLATE.format(extracted_sentences=f"- {text}")
                llm_text, llm_err = self.safe_chat_text(system=ATOMIC_SYSTEM_PROMPT, user=prompt)
                candidates = self._parse_llm_list(llm_text or "")
                if llm_err:
                    reason_codes.append("LLM_UNAVAILABLE")
                    llm_enabled = False
                    reason_codes.append("LLM_CIRCUIT_BREAKER")
                if not candidates:
                    rejected.append(
                        {
                            "unit_id": unit_id,
                            "raw": text,
                            "reason_codes": ["LLM_EXTRACTION_EMPTY"],
                        }
                    )
            else:
                if llm_calls >= llm_budget:
                    reason_codes.append("LLM_BUDGET_EXCEEDED")
                    rejected.append(
                        {
                            "unit_id": unit_id,
                            "raw": text,
                            "reason_codes": ["LLM_BUDGET_EXCEEDED"],
                        }
                    )
                else:
                    rejected.append(
                        {
                            "unit_id": unit_id,
                            "raw": text,
                            "reason_codes": ["LLM_UNAVAILABLE"],
                        }
                    )

            for row in candidates:
                criterion, fact, scope = self._fact_scope_from_row(row)
                if not criterion or not fact or not scope:
                    rejected.append(
                        {
                            "unit_id": unit_id,
                            "raw": row,
                            "reason_codes": ["MISSING_FACT_SCOPE_TAGS"],
                        }
                    )
                    continue

                if self._contains_malformed_numeric(fact):
                    rejected.append(
                        {
                            "unit_id": unit_id,
                            "raw": row,
                            "reason_codes": ["MALFORMED_NUMERIC"],
                        }
                    )
                    continue

                facet = self._facet_of(f"{fact} {scope}")
                metric_name = self._extract_metric_name(f"{fact} {scope}")
                metric_value, metric_unit = self._extract_metric_value(fact)

                if facet == "other":
                    rejected.append(
                        {
                            "unit_id": unit_id,
                            "raw": row,
                            "reason_codes": ["NON_ACTIONABLE_FACET"],
                        }
                    )
                    continue

                source_type = "text_metric" if facet == "metric_result" else "text_statement"

                accepted.append(
                    {
                        "criterion": criterion,
                        "fact": fact,
                        "scope": scope,
                        "facet": facet,
                        "source_type": source_type,
                        "metric_name": metric_name,
                        "metric_value": metric_value,
                        "metric_unit": metric_unit,
                        "entity": None,
                        "comparator": None,
                        "dataset_scope": None,
                        "table_anchor": None,
                        "input_unit_id": unit_id,
                        "reason_codes": [str(x) for x in row.get("reason_codes", [])],
                    }
                )

        payload = {
            "criteria": accepted,
            "selected_unit_ids": selected_unit_ids,
            "reason_codes": sorted(set(reason_codes)),
        }
        rejected_payload = {
            "rejected": rejected,
            "reason_codes": sorted({code for item in rejected for code in item.get("reason_codes", [])}),
        }

        self.artifacts.write_json("fingerprint/atomic_criteria.json", payload)
        self.artifacts.write_json("fingerprint/atomic_rejected.json", rejected_payload)
        return {"atomic_criteria": payload}
