from __future__ import annotations

import json
import re
from pathlib import Path

from p2c.agents.base import BaseAgent
from p2c.agents.phase1.fingerprint_prompt_templates import GUIDE_SYSTEM_PROMPT, GUIDE_USER_PROMPT_TEMPLATE

TABLE_BLOCK_RE = re.compile(r"<table\b.*?</table>", flags=re.I | re.S)
TABLE_CAPTION_RE = re.compile(r"table\s+[ivxlcdm\d]+[^\n]*", flags=re.I)


def split_sentences(text: str) -> list[str]:
    chunks = re.split(r"(?<=[.!?])\s+", text.replace("\n", " "))
    return [c.strip() for c in chunks if c.strip()]


def _normalize_ws(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


class ExtractFingerprintGuideAgent(BaseAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(name="extract_fingerprint_guide", *args, **kwargs)

    @staticmethod
    def _parse_index_array(text: str) -> list[int]:
        if not text:
            return []
        candidate = text.strip()
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, list):
                return [int(x) for x in parsed if isinstance(x, int)]
            if isinstance(parsed, dict) and isinstance(parsed.get("selected_indices"), list):
                return [int(x) for x in parsed["selected_indices"] if isinstance(x, int)]
        except Exception:  # noqa: BLE001
            pass

        match = re.search(r"\[[^\]]*\]", candidate, flags=re.S)
        if not match:
            return []
        try:
            parsed = json.loads(match.group(0))
            if isinstance(parsed, list):
                return [int(x) for x in parsed if isinstance(x, int)]
        except Exception:  # noqa: BLE001
            return []
        return []

    @staticmethod
    def _build_units(text: str, sentences: list[str]) -> list[dict]:
        units: list[dict] = []

        for idx, sentence in enumerate(sentences):
            units.append(
                {
                    "unit_id": f"s_{idx}",
                    "type": "sentence",
                    "text": sentence,
                    "origin_indices": [idx],
                }
            )

        table_sentence_indices = [
            i
            for i, s in enumerate(sentences)
            if "<table" in s.lower() or "</table>" in s.lower() or s.lower().strip().startswith("table ")
        ]

        for i, m in enumerate(TABLE_BLOCK_RE.finditer(text)):
            block = m.group(0)
            context_window = text[max(0, m.start() - 220) : m.start()]
            caption_m = list(TABLE_CAPTION_RE.finditer(context_window))
            caption = caption_m[-1].group(0).strip() if caption_m else ""
            unit_text = _normalize_ws((caption + "\n" + block).strip()) if caption else _normalize_ws(block)
            units.append(
                {
                    "unit_id": f"t_{i}",
                    "type": "table_block",
                    "text": unit_text,
                    "origin_indices": table_sentence_indices,
                }
            )

        return units

    def execute(self, ctx: dict) -> dict:
        paper_md_out = Path(ctx["paper_md_out"])
        text = paper_md_out.read_text(encoding="utf-8", errors="ignore")
        sentences = split_sentences(text)
        units = self._build_units(text, sentences)

        selected_unit_ids: set[str] = set()
        reason_codes: list[str] = []

        chunk_size = 180
        for start in range(0, len(units), chunk_size):
            end = min(len(units), start + chunk_size)
            chunk = units[start:end]
            numbered = "\n".join(f"{i}. [{u['type']}] {u['text']}" for i, u in enumerate(chunk))
            user_prompt = GUIDE_USER_PROMPT_TEMPLATE.format(paper_segment=numbered)

            llm_text, llm_err = self.safe_chat_text(system=GUIDE_SYSTEM_PROMPT, user=user_prompt)
            picked_local = self._parse_index_array(llm_text or "")
            if picked_local:
                for local_idx in picked_local:
                    if 0 <= local_idx < len(chunk):
                        selected_unit_ids.add(str(chunk[local_idx]["unit_id"]))
            else:
                reason_codes.append("GUIDE_SELECTION_EMPTY")

            if llm_err:
                reason_codes.append("LLM_UNAVAILABLE")

        # Always retain table units because they can be parsed deterministically downstream.
        for u in units:
            if u.get("type") == "table_block":
                selected_unit_ids.add(str(u["unit_id"]))
        if any(u.get("type") == "table_block" for u in units):
            reason_codes.append("TABLE_FORCE_RECALL")

        unit_map = {str(u["unit_id"]): u for u in units}
        selected = [uid for uid in sorted(selected_unit_ids) if uid in unit_map]
        if not selected:
            reason_codes.append("NO_GUIDE_UNITS_FOUND")

        selected_sentence_indices: set[int] = set()
        for uid in selected:
            for idx in unit_map[uid].get("origin_indices", []):
                if isinstance(idx, int):
                    selected_sentence_indices.add(idx)

        payload = {
            "sentence_count": len(sentences),
            "sentences": sentences,
            "unit_count": len(units),
            "units": units,
            "selected_unit_ids": selected,
            "selected_sentence_indices": sorted(selected_sentence_indices),
            "reason_codes": sorted(set(reason_codes)),
        }
        self.artifacts.write_json("fingerprint/guide_sentences.json", payload)
        return {"guide_sentences": payload}
