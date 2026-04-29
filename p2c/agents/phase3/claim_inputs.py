from __future__ import annotations

from copy import deepcopy
from typing import Any

from p2c.agents.phase1.build_claims_ir import BuildClaimsIRAgent


def load_effective_claims_ir(artifacts) -> dict[str, Any]:
    """Load claims_ir with Phase 3-safe deterministic target normalization."""
    raw = artifacts.read_json("fingerprint/claims_ir.json")
    effective, changed = normalize_claims_ir_payload(raw)
    if changed:
        artifacts.write_json("results/effective_claims_ir.json", effective)
    return effective


def normalize_claims_ir_payload(raw: dict[str, Any]) -> tuple[dict[str, Any], bool]:
    payload = deepcopy(raw) if isinstance(raw, dict) else {}
    changed = False
    claims = payload.get("claims", [])
    if not isinstance(claims, list):
        return payload, False

    for claim in claims:
        if not isinstance(claim, dict) or claim.get("type") != "result":
            continue
        predicate = str(claim.get("predicate") or "")
        parsed = BuildClaimsIRAgent._extract_metric_target_stats(predicate)
        parsed_target = parsed.get("target")
        if parsed_target is None:
            continue

        reason_codes = [str(code) for code in claim.get("reason_codes", []) if str(code).strip()]
        notes = claim.get("notes") if claim.get("notes") is None or isinstance(claim.get("notes"), str) else str(claim.get("notes"))
        tolerance_policy = dict(claim.get("tolerance_policy") or {})
        tolerance_policy, reason_codes, notes = BuildClaimsIRAgent._merge_mean_std_metadata(
            tolerance_policy=tolerance_policy,
            reason_codes=reason_codes,
            notes=notes,
            parsed=parsed,
        )

        should_replace_target = (
            "MEAN_STD_TARGET_NORMALIZED" in parsed.get("reason_codes", [])
            or claim.get("target") is None
            or BuildClaimsIRAgent._looks_like_std_target(claim.get("target"), parsed)
        )
        if should_replace_target and claim.get("target") != parsed_target:
            claim["target"] = parsed_target
            changed = True
        if claim.get("metric") is None and parsed.get("metric"):
            claim["metric"] = parsed["metric"]
            changed = True
        if claim.get("tolerance_policy") != tolerance_policy:
            claim["tolerance_policy"] = tolerance_policy
            changed = True
        if claim.get("reason_codes") != reason_codes:
            claim["reason_codes"] = reason_codes
            changed = True
        if notes and claim.get("notes") != notes:
            claim["notes"] = notes
            changed = True

    reason_codes = [str(code) for code in payload.get("reason_codes", []) if str(code).strip()]
    if changed and "PHASE3_EFFECTIVE_CLAIMS_NORMALIZED" not in reason_codes:
        reason_codes.append("PHASE3_EFFECTIVE_CLAIMS_NORMALIZED")
        payload["reason_codes"] = reason_codes
    return payload, changed

