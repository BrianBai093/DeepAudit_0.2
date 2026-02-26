from __future__ import annotations

import argparse
import json
from typing import Any


def _split_csv(raw: str) -> list[str]:
    return [x.strip() for x in (raw or "").split(",") if x.strip()]


def _call_template_build(template_obj: Any, template_name: str) -> Any:
    for fn_name in ("build", "create", "publish"):
        fn = getattr(template_obj, fn_name, None)
        if not callable(fn):
            continue
        for kwargs in ({"name": template_name}, {}):
            try:
                return fn(**kwargs)
            except TypeError:
                continue
    raise RuntimeError("No compatible template build method found on E2B Template object")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build an E2B Codex template with pip capability preinstalled.",
    )
    parser.add_argument("--template-name", required=True, help="Target template name/tag.")
    parser.add_argument("--base-template", default="codex", help="Base E2B template name.")
    parser.add_argument("--apt", default="python3,python3-pip", help="Comma-separated apt packages.")
    parser.add_argument(
        "--pip",
        default="pip,setuptools,wheel,numpy,scikit-learn,pandas",
        help="Comma-separated pip packages.",
    )
    args = parser.parse_args()

    try:
        from e2b import Template
    except Exception as e:  # noqa: BLE001
        raise RuntimeError("E2B SDK with Template support is required") from e

    apt_packages = _split_csv(args.apt)
    pip_packages = _split_csv(args.pip)

    template = Template().from_template(args.base_template)
    if apt_packages:
        template = template.apt_install(apt_packages)
    if pip_packages:
        template = template.pip_install(pip_packages)

    result = _call_template_build(template, args.template_name)
    payload = {
        "template_name": args.template_name,
        "base_template": args.base_template,
        "apt_packages": apt_packages,
        "pip_packages": pip_packages,
        "build_result": str(result),
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
