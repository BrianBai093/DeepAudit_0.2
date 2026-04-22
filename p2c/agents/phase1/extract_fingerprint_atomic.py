from __future__ import annotations

import json
import os
import re

from p2c.agents.base import BaseAgent
from p2c.agents.phase1.fingerprint_prompt_templates import ATOMIC_SYSTEM_PROMPT, ATOMIC_USER_PROMPT_TEMPLATE
from p2c.agents.phase1.table_llm import (
    TABLE_CRITERIA_SCHEMA,
    TABLE_CRITERIA_SYSTEM_PROMPT,
    build_table_criteria_user_prompt,
    normalize_table_criteria,
    rows_from_llm_table_response,
)

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
    def _infer_table_anchor(unit: dict) -> str | None:
        anchor = ExtractFingerprintAtomicAgent._extract_table_anchor(str(unit.get("text") or ""))
        if anchor:
            return anchor
        unit_id = str(unit.get("unit_id") or "")
        m = re.fullmatch(r"t_(\d+)", unit_id)
        if m:
            return f"Table {int(m.group(1)) + 1}"
        return None

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
    def _is_plausible_metric_value(value: float, metric_name: str) -> bool:
        """Check if a numeric value is plausible for the given metric type.

        Rejects obviously wrong values like accuracy=10000 (sample counts).
        """
        # Metrics that should be in [0, 100] (percentage) or [0, 1] (ratio)
        BOUNDED_METRICS = {"accuracy", "f1", "auc", "precision", "recall", "bleu", "rouge"}
        # Metrics that can be any positive number
        UNBOUNDED_METRICS = {"loss", "mse", "mae", "rmse", "perplexity"}

        if metric_name in BOUNDED_METRICS:
            # Accept 0-100 (percentage) or 0-1 (ratio)
            return value <= 100.0
        if metric_name in UNBOUNDED_METRICS:
            # Loss/error metrics can be large but not absurdly so
            return value < 1e6
        return True

    @staticmethod
    def _is_classification_report_table(parsed_rows: list[list[str]]) -> bool:
        """Detect sklearn-style classification report tables.

        These tables have rows like "accuracy", "macro avg", "weighted avg"
        with a "support" column containing sample counts — NOT metric values.
        """
        summary_labels = {"accuracy", "macro avg", "macro_avg", "weighted avg", "weighted_avg", "micro avg", "micro_avg"}
        header_keywords = {"precision", "recall", "f1-score", "f1", "support"}

        # Check if any row looks like a header with classification report columns
        has_report_header = False
        has_summary_row = False
        for row in parsed_rows:
            row_lower = [c.lower().strip() for c in row]
            if len(set(row_lower) & header_keywords) >= 2:
                has_report_header = True
            if row_lower and row_lower[0] in summary_labels:
                has_summary_row = True

        return has_report_header and has_summary_row

    @staticmethod
    def _detect_header_row(parsed_rows: list[list[str]]) -> list[str] | None:
        """Find the actual header row (first row that is mostly non-numeric text)."""
        for row in parsed_rows:
            non_numeric = 0
            for cell in row:
                cell_stripped = cell.strip()
                if not cell_stripped:
                    continue
                if NUMBER_RE.fullmatch(cell_stripped):
                    continue
                non_numeric += 1
            # If most cells are non-numeric text, it's likely a header
            if non_numeric >= max(1, len(row) // 2):
                return row
        return None

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

        # Detect classification report tables — these need special handling
        is_clf_report = self._is_classification_report_table(parsed_rows)

        # Find the actual header row (mostly non-numeric text) instead of
        # blindly using the previous row, which may be a data row.
        global_header = self._detect_header_row(parsed_rows)

        # Column names that contain counts, not metric values
        COUNT_COLUMNS = {"support", "count", "samples", "n", "size", "total", "#"}

        # Check if header has metric names (column-oriented table):
        # e.g. "Method | Accuracy | F1" → metrics in header, entities in row[0]
        header_metric_cols: dict[int, str] = {}
        if global_header and not is_clf_report:
            for col_idx, cell in enumerate(global_header):
                m = self._extract_metric_name(cell)
                if m and cell.lower().strip() not in COUNT_COLUMNS:
                    header_metric_cols[col_idx] = m

        for i, row in enumerate(parsed_rows):
            # Skip the header row itself
            if row is global_header:
                continue

            metric_cell = row[0] if row else ""
            metric_name = self._extract_metric_name(metric_cell or " ".join(row))

            # Column-oriented table: metrics in header, entities in row[0]
            # e.g. "Method | Accuracy | F1" with rows "Baseline | 92.3% | 0.91"
            if not metric_name and header_metric_cols:
                entity = row[0].strip() if row else ""
                # Skip rows that look like headers themselves
                if not entity or entity.lower() in {"method", "model", "approach", ""}:
                    continue
                # Skip rows that are entirely non-numeric (likely sub-headers)
                has_any_number = any(NUMBER_RE.search(row[c]) for c in header_metric_cols if c < len(row))
                if not has_any_number:
                    continue

                for col_idx, col_metric in header_metric_cols.items():
                    if col_idx >= len(row):
                        continue
                    value_raw = row[col_idx].strip()
                    if not value_raw:
                        continue
                    value, unit_name = self._extract_metric_value(value_raw)
                    if value is None:
                        raw_num = NUMBER_RE.search(value_raw)
                        if raw_num:
                            value = float(raw_num.group(0))
                            unit_name = "%" if col_metric in {"accuracy", "f1", "auc", "precision", "recall"} else "value"
                    if value is None:
                        continue
                    if not self._is_plausible_metric_value(value, col_metric):
                        rej.append({
                            "unit_id": unit.get("unit_id"),
                            "raw": f"{entity} {col_metric} = {value}",
                            "reason_codes": ["IMPLAUSIBLE_METRIC_VALUE"],
                        })
                        continue

                    fact = f"{entity} {col_metric} = {value_raw if unit_name == '%' else value}"
                    scope = f"from {table_anchor} in paper" if table_anchor else "from table in paper"
                    out.append({
                        "criterion": f"<fact>{fact}</fact> <scope>{scope}</scope>",
                        "fact": fact,
                        "scope": scope,
                        "facet": "metric_result",
                        "source_type": "table_metric",
                        "metric_name": col_metric,
                        "metric_value": value,
                        "metric_unit": unit_name,
                        "entity": entity,
                        "comparator": None,
                        "dataset_scope": None,
                        "table_anchor": table_anchor,
                        "input_unit_id": unit.get("unit_id"),
                        "reason_codes": ["TABLE_EXPANDED"],
                    })
                continue

            if not metric_name:
                continue

            # For classification reports, only extract the actual metric value
            # from the correct column — skip the "support" column entirely.
            if is_clf_report:
                # In a classification report, the "accuracy" row has the metric
                # value in a known column (usually f1-score or the 3rd numeric col).
                # We extract from the row directly using the global header.
                self._extract_clf_report_row(
                    row=row,
                    header=global_header,
                    metric_name=metric_name,
                    table_anchor=table_anchor,
                    unit=unit,
                    out=out,
                    rej=rej,
                )
                continue

            # Standard table: use global header or fall back to previous row
            header = global_header
            if not header:
                header = parsed_rows[i - 1] if i - 1 >= 0 else []

            for col in range(1, len(row)):
                col_name = header[col] if header and col < len(header) else f"col_{col}"

                # Skip columns that are sample counts, not metric values
                if col_name.lower().strip() in COUNT_COLUMNS:
                    continue

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

                # Reject implausible values (e.g. accuracy = 10000)
                if not self._is_plausible_metric_value(value, metric_name):
                    rej.append(
                        {
                            "unit_id": unit.get("unit_id"),
                            "raw": f"{col_name} {metric_name} = {value}",
                            "reason_codes": ["IMPLAUSIBLE_METRIC_VALUE"],
                        }
                    )
                    continue

                # Don't use numeric strings as entity/model names — they're
                # likely data values from another row, not column headers.
                if NUMBER_RE.fullmatch(col_name.strip()):
                    col_name = f"col_{col}"

                fact = f"{col_name} {metric_name} = {value_raw if unit_name == '%' else value}"
                scope = f"from {table_anchor} in paper" if table_anchor else "from table in paper"
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
                        "entity": col_name,
                        "comparator": None,
                        "dataset_scope": None,
                        "table_anchor": table_anchor,
                        "input_unit_id": unit.get("unit_id"),
                        "reason_codes": ["TABLE_EXPANDED"],
                    }
                )

        return out, rej

    def _extract_clf_report_row(
        self,
        *,
        row: list[str],
        header: list[str] | None,
        metric_name: str,
        table_anchor: str | None,
        unit: dict,
        out: list[dict],
        rej: list[dict],
    ) -> None:
        """Extract metrics from a classification report summary row (accuracy, macro avg, etc.).

        In sklearn classification reports:
        - "accuracy" row: has ONE value (the overall accuracy) — metric_name stays "accuracy"
        - "macro avg" / "weighted avg" rows: each column IS a different metric (precision, recall, f1)
        - The 'support' column is always a sample count and must be skipped.
        """
        COUNT_COLUMNS = {"support", "count", "samples", "n", "size", "total", "#"}
        METRIC_COLUMNS = {"precision", "recall", "f1-score", "f1", "accuracy", "auc"}

        # Determine if this is an "accuracy" row (single-value) vs avg row (multi-value)
        row_label = row[0].lower().strip() if row else ""
        is_accuracy_row = row_label in {"accuracy", "acc"}

        for col in range(1, len(row)):
            value_raw = row[col].strip()
            if not value_raw or value_raw == "-":
                continue

            col_name = (header[col] if header and col < len(header) else "").lower().strip()

            # Skip count/support columns
            if col_name in COUNT_COLUMNS:
                continue

            value, unit_name = self._extract_metric_value(value_raw)
            if value is None:
                raw_num = NUMBER_RE.search(value_raw)
                if raw_num:
                    candidate = float(raw_num.group(0))
                    # In clf reports, only accept plausible metric values
                    if self._is_plausible_metric_value(candidate, "accuracy"):
                        value = candidate
                        unit_name = "ratio"

            if value is None:
                continue

            # Reject implausible values
            effective_metric = "accuracy" if is_accuracy_row else (col_name if col_name in METRIC_COLUMNS else metric_name)
            if not self._is_plausible_metric_value(value, effective_metric):
                rej.append({
                    "unit_id": unit.get("unit_id"),
                    "raw": f"{effective_metric} = {value}",
                    "reason_codes": ["IMPLAUSIBLE_METRIC_VALUE"],
                })
                continue

            # For "accuracy" row: the value IS the overall accuracy regardless of column
            # For avg rows: the column name determines the metric
            if is_accuracy_row:
                fact_metric = "accuracy"
            elif col_name in METRIC_COLUMNS:
                fact_metric = col_name
            else:
                fact_metric = metric_name

            # Build entity prefix for avg rows
            entity = None
            if not is_accuracy_row and row_label:
                entity = row_label  # "macro avg", "weighted avg"

            if entity:
                fact = f"{entity} {fact_metric} = {value_raw if unit_name == '%' else value}"
            else:
                fact = f"{fact_metric} = {value_raw if unit_name == '%' else value}"
            scope = f"from classification report in {table_anchor}" if table_anchor else "from classification report in paper"
            out.append({
                "criterion": f"<fact>{fact}</fact> <scope>{scope}</scope>",
                "fact": fact,
                "scope": scope,
                "facet": "metric_result",
                "source_type": "table_metric",
                "metric_name": fact_metric,
                "metric_value": value,
                "metric_unit": unit_name,
                "entity": entity,
                "comparator": None,
                "dataset_scope": None,
                "table_anchor": table_anchor,
                "input_unit_id": unit.get("unit_id"),
                "reason_codes": ["TABLE_EXPANDED", "CLF_REPORT"],
            })

    def _visual_table_index(self) -> dict[str, dict]:
        try:
            ve_doc = self.artifacts.read_json("fingerprint/visual_elements.json")
        except Exception:  # noqa: BLE001
            return {}
        out: dict[str, dict] = {}
        for elem in ve_doc.get("elements", []):
            if not isinstance(elem, dict) or elem.get("element_type") != "table":
                continue
            for key in (elem.get("visual_anchor"), elem.get("element_id")):
                normalized = str(key or "").strip().lower()
                if normalized:
                    out[normalized] = elem
        return out

    def _visual_for_table_unit(self, unit: dict, visual_index: dict[str, dict]) -> dict | None:
        anchor = self._infer_table_anchor(unit)
        if anchor:
            elem = visual_index.get(anchor.lower())
            if elem:
                return elem
        unit_id = str(unit.get("unit_id") or "")
        m = re.fullmatch(r"t_(\d+)", unit_id)
        if not m:
            return None
        candidates = [
            f"table_{int(m.group(1)) + 1}",
            f"Table {int(m.group(1)) + 1}",
        ]
        for candidate in candidates:
            elem = visual_index.get(candidate.lower())
            if elem:
                return elem
        return None

    def _expand_table_unit_with_llm(
        self,
        unit: dict,
        *,
        visual_index: dict[str, dict],
    ) -> tuple[list[dict], list[dict], str | None]:
        table_anchor = self._infer_table_anchor(unit)
        visual_elem = self._visual_for_table_unit(unit, visual_index)
        if visual_elem and not table_anchor:
            table_anchor = str(visual_elem.get("visual_anchor") or visual_elem.get("element_id") or "").strip() or None

        user = build_table_criteria_user_prompt(
            table_anchor=table_anchor,
            caption=str(unit.get("caption") or ""),
            table_html=str(unit.get("text") or ""),
            context_before=str(unit.get("context_before") or ""),
            context_after=str(unit.get("context_after") or ""),
            visual_element=visual_elem,
        )
        data, err = self.safe_chat_json(
            schema=TABLE_CRITERIA_SCHEMA,
            system=TABLE_CRITERIA_SYSTEM_PROMPT,
            user=user,
        )
        rows = rows_from_llm_table_response(data)
        criteria = normalize_table_criteria(
            rows,
            default_anchor=table_anchor,
            input_unit_id=str(unit.get("unit_id") or ""),
            source_prefix="llm_table",
            default_reason_code="LLM_TABLE_EXTRACTED",
            visual_data=_visual_data_from_element(visual_elem) if visual_elem else None,
        )
        rejected: list[dict] = []
        if not criteria:
            rejected.append(
                {
                    "unit_id": unit.get("unit_id"),
                    "raw": str(unit.get("text") or "")[:2000],
                    "reason_codes": ["LLM_TABLE_EXTRACTION_EMPTY" if not err else "LLM_TABLE_UNAVAILABLE"],
                }
            )
        return criteria, rejected, err

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
        table_llm_budget = int(os.getenv("P2C_ATOMIC_LLM_TABLE_BUDGET", "20"))
        llm_calls = 0
        table_llm_calls = 0
        llm_enabled = True
        table_llm_enabled = True
        visual_index = self._visual_table_index()

        accepted: list[dict] = []
        rejected: list[dict] = []
        reason_codes: list[str] = []

        self.log(
            "PROGRESS",
            f"atomic stage processing {len(selected_units)} units "
            f"(llm_budget={llm_budget}, table_llm_budget={table_llm_budget})",
        )

        for idx, unit in enumerate(selected_units, start=1):
            if idx == 1 or idx % 25 == 0 or idx == len(selected_units):
                self.log("PROGRESS", f"atomic progress {idx}/{len(selected_units)}")

            unit_id = str(unit.get("unit_id"))
            text = str(unit.get("text") or "")
            unit_type = str(unit.get("type") or "sentence")

            if unit_type == "table_block":
                rows, rej = self._expand_table_unit(unit)
                if not rows and table_llm_enabled and table_llm_calls < table_llm_budget:
                    table_llm_calls += 1
                    reason_codes.append("LLM_TABLE_FALLBACK_USED")
                    llm_rows, llm_rej, llm_err = self._expand_table_unit_with_llm(
                        unit,
                        visual_index=visual_index,
                    )
                    if llm_err:
                        reason_codes.append("LLM_TABLE_UNAVAILABLE")
                        table_llm_enabled = False
                    rows = llm_rows
                    rej.extend(llm_rej)
                elif not rows and table_llm_calls >= table_llm_budget:
                    reason_codes.append("LLM_TABLE_BUDGET_EXCEEDED")
                    rej.append(
                        {
                            "unit_id": unit_id,
                            "raw": text[:2000],
                            "reason_codes": ["LLM_TABLE_BUDGET_EXCEEDED"],
                        }
                    )
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


def _visual_data_from_element(elem: dict | None) -> dict:
    if not isinstance(elem, dict):
        return {}
    out = {
        "element_id": elem.get("element_id"),
        "chart_type": elem.get("chart_type"),
        "axis_labels": elem.get("axis_labels", {}),
        "legend_entries": elem.get("legend_entries", []),
        "data_series": elem.get("data_series", []),
    }
    for key in (
        "bbox",
        "raw_page_image",
        "crop_path",
        "x_axis_range",
        "y_axis_range",
        "series_semantics",
        "model_names",
        "sampling_strategy",
        "numeric_confidence",
    ):
        value = elem.get(key)
        if value not in (None, [], {}):
            out[key] = value
    return out
