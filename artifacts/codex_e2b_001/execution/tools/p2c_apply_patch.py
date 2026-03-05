#!/usr/bin/env python3
from __future__ import annotations

import subprocess
import sys


def _to_unified(payload: str) -> str:
    if "*** Begin Patch" not in payload:
        return payload

    lines = payload.splitlines()
    out: list[str] = []
    idx = 0
    while idx < len(lines):
        line = lines[idx]
        if line.startswith("*** Begin Patch") or line.startswith("*** End Patch"):
            idx += 1
            continue
        if line.startswith("*** Update File: "):
            path = line[len("*** Update File: ") :].strip()
            out.append(f"--- {path}")
            out.append(f"+++ {path}")
            idx += 1
            if idx < len(lines) and lines[idx].startswith("*** Move to: "):
                idx += 1
            while idx < len(lines):
                cur = lines[idx]
                if cur.startswith("*** "):
                    break
                if cur.startswith("@@") or cur.startswith("+") or cur.startswith("-") or cur.startswith(" "):
                    out.append(cur)
                idx += 1
            continue
        if line.startswith("*** Add File: "):
            path = line[len("*** Add File: ") :].strip()
            added: list[str] = []
            idx += 1
            while idx < len(lines):
                cur = lines[idx]
                if cur.startswith("*** "):
                    break
                if cur.startswith("+"):
                    added.append(cur[1:])
                idx += 1
            out.append("--- /dev/null")
            out.append(f"+++ {path}")
            out.append(f"@@ -0,0 +1,{len(added)} @@")
            out.extend([f"+{x}" for x in added])
            continue
        idx += 1
    if not out:
        return ""
    return "\n".join(out) + "\n"


def main() -> int:
    raw = sys.stdin.read()
    if not raw.strip():
        return 0
    patch_text = _to_unified(raw)
    if not patch_text.strip():
        sys.stderr.write("p2c_apply_patch: empty converted patch\n")
        return 2
    proc = subprocess.run(["patch", "-p0"], input=patch_text, text=True)
    return int(proc.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
