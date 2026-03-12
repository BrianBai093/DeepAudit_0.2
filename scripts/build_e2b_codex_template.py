from __future__ import annotations

import argparse
import json
from typing import Any


def _split_csv(raw: str) -> list[str]:
    return [x.strip() for x in (raw or "").split(",") if x.strip()]


def _call_method(obj: Any, names: list[str], *args, **kwargs) -> Any:
    last_err: Exception | None = None
    for name in names:
        fn = getattr(obj, name, None)
        if not callable(fn):
            continue
        try:
            return fn(*args, **kwargs)
        except TypeError as e:
            last_err = e
            continue
    if last_err is not None:
        raise RuntimeError(f"Compatible method found for {names}, but signature did not match: {last_err}") from last_err
    raise RuntimeError(f"No compatible method found: {names}")


def _call_template_build(
    template_cls: Any,
    template_obj: Any,
    template_name: str,
    *,
    cpu_count: int | None = None,
    memory_mb: int | None = None,
    on_build_logs: Any = None,
) -> Any:
    build_fn = getattr(template_cls, "build", None)
    if not callable(build_fn):
        raise RuntimeError("Template.build is not available on the current E2B SDK")
    preferred_kwargs = {}
    if cpu_count is not None:
        preferred_kwargs["cpu_count"] = cpu_count
    if memory_mb is not None:
        preferred_kwargs["memory_mb"] = memory_mb
    if on_build_logs is not None:
        preferred_kwargs["on_build_logs"] = on_build_logs
    for args, kwargs in (
        ((template_obj, template_name), preferred_kwargs),
        ((template_obj, template_name), {}),
        ((template_obj,), {"alias": template_name, **preferred_kwargs}),
        ((template_obj,), {"name": template_name, **preferred_kwargs}),
        ((template_obj,), {"alias": template_name}),
        ((template_obj,), {"name": template_name}),
    ):
        try:
            return build_fn(*args, **kwargs)
        except TypeError:
            continue
    raise RuntimeError("No compatible Template.build signature found")


def _template_from_ubuntu(template_obj: Any, ubuntu_version: str) -> Any:
    try:
        return _call_method(
            template_obj,
            ["fromUbuntuImage", "from_ubuntu_image", "fromUbuntu", "from_ubuntu"],
            ubuntu_version,
        )
    except RuntimeError:
        return _call_method(
            template_obj,
            ["from_docker_image", "fromDockerImage", "from_image", "fromImage"],
            f"ubuntu:{ubuntu_version}",
        )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build an E2B template with a full Python toolchain and Codex CLI preinstalled.",
    )
    parser.add_argument("--template-name", default="p2c-codex-toolchain", help="Target template name/tag.")
    parser.add_argument("--ubuntu-version", default="22.04", help="Ubuntu base image version.")
    parser.add_argument("--cpu-count", type=int, default=2, help="CPU count to request during template build.")
    parser.add_argument("--memory-mb", type=int, default=2048, help="Memory in MB to request during template build.")
    parser.add_argument(
        "--apt",
        default="git,curl,jq,ca-certificates,build-essential,python3,python3-pip,python3-venv,python3-dev,pkg-config,libssl-dev,nodejs,npm,r-base",
        help="Comma-separated apt packages.",
    )
    parser.add_argument(
        "--pip",
        default="pip,setuptools,wheel,poetry,uv",
        help="Comma-separated pip packages to install globally in the template.",
    )
    parser.add_argument(
        "--npm",
        default="@openai/codex",
        help="Comma-separated npm packages to install globally in the template.",
    )
    args = parser.parse_args()

    try:
        from e2b import Template, default_build_logger
    except Exception as e:  # noqa: BLE001
        raise RuntimeError("E2B SDK with Template support is required") from e

    apt_packages = _split_csv(args.apt)
    pip_packages = _split_csv(args.pip)
    npm_packages = _split_csv(args.npm)

    template = _template_from_ubuntu(Template(), args.ubuntu_version)
    if apt_packages:
        template = _call_method(template, ["aptInstall", "apt_install"], apt_packages)
    if pip_packages:
        template = _call_method(template, ["pipInstall", "pip_install"], pip_packages)
    if npm_packages:
        try:
            template = _call_method(
                template,
                ["npmInstall", "npm_install"],
                npm_packages,
                {"g": True},
            )
        except RuntimeError:
            template = _call_method(
                template,
                ["runCmd", "run_cmd", "runCommand", "run_command"],
                "npm install -g " + " ".join(npm_packages),
            )

    for cmd in [
        "python3 --version",
        "python3 -m pip --version",
        "poetry --version",
        "uv --version",
        "node --version",
        "npm --version",
        "codex --version",
        "Rscript --version",
    ]:
        template = _call_method(template, ["runCmd", "run_cmd", "runCommand", "run_command"], cmd)

    build_logger = default_build_logger()
    result = _call_template_build(
        Template,
        template,
        args.template_name,
        cpu_count=args.cpu_count,
        memory_mb=args.memory_mb,
        on_build_logs=build_logger,
    )
    payload = {
        "template_name": args.template_name,
        "ubuntu_version": args.ubuntu_version,
        "cpu_count": args.cpu_count,
        "memory_mb": args.memory_mb,
        "apt_packages": apt_packages,
        "pip_packages": pip_packages,
        "npm_packages": npm_packages,
        "build_result": str(result),
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
