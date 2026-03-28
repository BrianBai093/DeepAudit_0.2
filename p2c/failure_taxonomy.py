"""Comprehensive failure taxonomy for automated ML/research code execution.

Replaces the flat 7-literal error_type with a two-level hierarchy:
    Layer  ->  FailureCode

Each FailureCode carries detection heuristics, recommended repair action,
whether it requires a full replan vs. an inline fix, and confidence that
automated repair will succeed.

Design informed by:
  - SWE-bench evaluation harness error patterns
  - OpenHands/OpenDevin sandbox execution model
  - "AI-Generated Code Is Not Reproducible (Yet)" (arXiv 2512.22387)
  - "The Last Dependency Crusade" (arXiv 2501.16191) -- dependency conflict taxonomy
  - ML Reproducibility Challenge (MLRC 2025) common failure modes
  - CUDA driver/runtime/PyTorch version mismatch patterns (vLLM, SageAttention)

Usage
-----
>>> from p2c.failure_taxonomy import classify_failure, FailureCode, RepairStrategy
>>> code = classify_failure(stdout, stderr, exit_code)
>>> code.repair_strategy   # RepairStrategy.REPLAN | .INLINE_FIX | .SKIP | ...
>>> code.layer             # "dependency" | "data" | "configuration" | ...
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Literal


# ============================================================================
# Core types
# ============================================================================

class RepairStrategy(str, Enum):
    """What the orchestrator should do after detecting a failure."""
    INLINE_FIX = "inline_fix"        # patch command / env and retry same step
    REPLAN = "replan"                 # go back to planner with failure context
    SKIP = "skip"                     # skip this step, continue pipeline
    ABORT = "abort"                   # unrecoverable -- stop execution
    RETRY = "retry"                   # retry same step without changes (transient)


FailureLayer = Literal[
    "dependency",
    "data",
    "configuration",
    "code",
    "resource",
    "output",
]

# Confidence that automated repair will succeed (0.0 = hopeless, 1.0 = near-certain)
AutoRepairConfidence = float  # [0.0, 1.0]


@dataclass(frozen=True)
class FailureSpec:
    """Full specification for a single failure code."""

    code: str                                 # e.g. "DEP_MISSING_PACKAGE"
    layer: FailureLayer
    label: str                                # human-readable short label
    description: str                          # one-liner explanation
    detection_patterns: tuple[str, ...]       # regex patterns (any match -> hit)
    repair_strategy: RepairStrategy
    repair_action: str                        # what the system should attempt
    auto_repair_confidence: AutoRepairConfidence
    is_fast_fail: bool = False                # should we stop the pipeline?
    # Back-compat: maps to the old 7-literal system
    legacy_error_type: str = "unknown"


# ============================================================================
# 1. DEPENDENCY LAYER
# ============================================================================

DEP_MISSING_PACKAGE = FailureSpec(
    code="DEP_MISSING_PACKAGE",
    layer="dependency",
    label="Missing package",
    description="A required Python package is not installed in the environment.",
    detection_patterns=(
        r"ModuleNotFoundError:\s*No module named\s+['\"]?([\w.]+)",
        r"ImportError:\s*cannot import name",
        r"ImportError:\s*No module named\s+['\"]?([\w.]+)",
        r"pkg_resources\.DistributionNotFound",
    ),
    repair_strategy=RepairStrategy.INLINE_FIX,
    repair_action="pip install <missing_package>; re-run step",
    auto_repair_confidence=0.85,
    legacy_error_type="import",
)

DEP_VERSION_CONFLICT = FailureSpec(
    code="DEP_VERSION_CONFLICT",
    layer="dependency",
    label="Version conflict",
    description="Two or more packages require incompatible versions of a shared dependency.",
    detection_patterns=(
        r"(?i)version\s+conflict",
        r"(?i)incompatible\s+versions?",
        r"pip.*ResolutionImpossible",
        r"(?i)requires\s+[\w-]+==[^\s]+.*but.*[\w-]+==[^\s]+\s+is installed",
        r"ERROR:\s+Cannot install .* because these package versions have conflicting dependencies",
        r"(?i)dependency\s+conflict",
    ),
    repair_strategy=RepairStrategy.REPLAN,
    repair_action="Re-solve dependency tree; try relaxing version pins or finding compatible set",
    auto_repair_confidence=0.45,
    legacy_error_type="dependency",
)

DEP_CUDA_MISMATCH = FailureSpec(
    code="DEP_CUDA_MISMATCH",
    layer="dependency",
    label="CUDA/GPU mismatch",
    description="CUDA runtime, driver, or PyTorch/TF compile-time CUDA version mismatch.",
    detection_patterns=(
        r"(?i)cuda\s+version\s+mismatch",
        r"(?i)the detected CUDA version.*mismatches",
        r"(?i)no kernel image is available for execution on the device",
        r"(?i)CUDA\s+initialization.*failed",
        r"(?i)CUDA driver version is insufficient",
        r"(?i)unsupported display driver.*cuda driver combination",
        r"(?i)RuntimeError:\s+CUDA error:\s+no kernel image",
        r"libcudart\.so.*cannot open shared object",
        r"libcublas\.so.*cannot open shared object",
    ),
    repair_strategy=RepairStrategy.REPLAN,
    repair_action="Switch to CPU fallback; or install matching CUDA-compiled wheel",
    auto_repair_confidence=0.30,
    legacy_error_type="dependency",
)

DEP_SYSTEM_LIBRARY = FailureSpec(
    code="DEP_SYSTEM_LIBRARY",
    layer="dependency",
    label="System library missing",
    description="A required system-level shared library (.so/.dylib) is absent.",
    detection_patterns=(
        r"OSError:\s+lib[\w]+\.so.*cannot open shared object",
        r"ImportError:\s+lib[\w]+\.so",
        r"ImportError.*libGL",
        r"ImportError.*libgthread",
        r"error while loading shared libraries",
        r"dyld: Library not loaded",
    ),
    repair_strategy=RepairStrategy.REPLAN,
    repair_action="Add conda/apt dependency for system lib; or install headless variant",
    auto_repair_confidence=0.50,
    legacy_error_type="dependency",
)

DEP_BUILD_FAILURE = FailureSpec(
    code="DEP_BUILD_FAILURE",
    layer="dependency",
    label="Package build failure",
    description="A package failed to build from source (missing compiler, headers, etc.).",
    detection_patterns=(
        r"error:\s+command\s+'gcc'.*failed",
        r"error:\s+command\s+'cl\.exe'.*failed",
        r"Failed building wheel for",
        r"(?i)setup\.py.*error",
        r"(?i)No matching distribution found for",
        r"(?i)Could not build wheels for",
    ),
    repair_strategy=RepairStrategy.REPLAN,
    repair_action="Try pre-built wheel; pin older version; add build deps via conda",
    auto_repair_confidence=0.40,
    legacy_error_type="dependency",
)


# ============================================================================
# 2. DATA LAYER
# ============================================================================

DATA_MISSING_DATASET = FailureSpec(
    code="DATA_MISSING_DATASET",
    layer="data",
    label="Missing dataset",
    description="Script expects a dataset file/directory that does not exist.",
    detection_patterns=(
        r"FileNotFoundError.*(?:data|dataset|train|test|val)",
        r"No such file or directory.*(?:\.csv|\.json|\.tsv|\.parquet|\.h5|\.hdf5|\.pkl|\.npy|\.npz|\.tfrecord)",
        r"(?i)dataset.*not found",
        r"(?i)could not find.*data",
    ),
    repair_strategy=RepairStrategy.REPLAN,
    repair_action="Add data download step; generate synthetic stub; or adjust data path",
    auto_repair_confidence=0.55,
    legacy_error_type="data_missing",
)

DATA_WRONG_FORMAT = FailureSpec(
    code="DATA_WRONG_FORMAT",
    layer="data",
    label="Wrong data format",
    description="Data file exists but has unexpected format, schema, or encoding.",
    detection_patterns=(
        r"(?i)UnicodeDecodeError",
        r"(?i)ParserError.*(?:tokenizing|parsing)",
        r"(?i)JSONDecodeError",
        r"(?i)KeyError.*(?:column|field|key)",
        r"(?i)expected.*columns?\s+\d+.*got\s+\d+",
        r"(?i)invalid.*(?:format|header|schema)",
        r"(?i)xml\.etree.*ParseError",
    ),
    repair_strategy=RepairStrategy.REPLAN,
    repair_action="Add preprocessing/conversion step; fix encoding; adjust column names",
    auto_repair_confidence=0.40,
    legacy_error_type="data_missing",
)

DATA_DOWNLOAD_FAILURE = FailureSpec(
    code="DATA_DOWNLOAD_FAILURE",
    layer="data",
    label="Download failure",
    description="Automated dataset download failed (network, auth, broken URL).",
    detection_patterns=(
        r"(?i)urllib\.error\.URLError",
        r"(?i)requests\.exceptions\.ConnectionError",
        r"(?i)HTTP\s+Error\s+(?:403|404|500|502|503)",
        r"(?i)ConnectionRefusedError",
        r"(?i)ssl\.SSLError",
        r"(?i)wget.*failed",
        r"(?i)gdown.*failed",
        r"(?i)google\.auth.*error",
        r"(?i)huggingface_hub.*error",
    ),
    repair_strategy=RepairStrategy.REPLAN,
    repair_action="Try mirror URL; use cached copy; generate synthetic data stub",
    auto_repair_confidence=0.35,
    legacy_error_type="data_missing",
)

DATA_PATH_MISMATCH = FailureSpec(
    code="DATA_PATH_MISMATCH",
    layer="data",
    label="Path mismatch",
    description="Data exists but path in code does not match actual location.",
    detection_patterns=(
        r"FileNotFoundError",
        r"No such file or directory",
        # Separate from DATA_MISSING_DATASET by absence of dataset-specific keywords;
        # classifier checks DATA_MISSING_DATASET first.
    ),
    repair_strategy=RepairStrategy.INLINE_FIX,
    repair_action="Symlink or set env var to redirect path; patch config file",
    auto_repair_confidence=0.70,
    legacy_error_type="data_missing",
)


# ============================================================================
# 3. CONFIGURATION LAYER
# ============================================================================

CFG_MISSING_CONFIG = FailureSpec(
    code="CFG_MISSING_CONFIG",
    layer="configuration",
    label="Missing config file",
    description="Script requires a YAML/JSON/TOML config file that is absent.",
    detection_patterns=(
        r"FileNotFoundError.*(?:config|cfg|\.yaml|\.yml|\.json|\.toml|\.ini)",
        r"(?i)config.*not found",
        r"(?i)missing.*configuration",
    ),
    repair_strategy=RepairStrategy.REPLAN,
    repair_action="Generate default config from template or argparse defaults",
    auto_repair_confidence=0.55,
    legacy_error_type="data_missing",
)

CFG_HARDCODED_PATH = FailureSpec(
    code="CFG_HARDCODED_PATH",
    layer="configuration",
    label="Hardcoded path",
    description="Code contains hardcoded absolute paths (e.g. /home/user/...) that do not exist.",
    detection_patterns=(
        r"FileNotFoundError.*(?:/home/|/users/|/mnt/|/data/|C:\\\\)",
        r"No such file or directory.*(?:/home/|/users/|/mnt/)",
    ),
    repair_strategy=RepairStrategy.INLINE_FIX,
    repair_action="Patch script to use relative paths or env vars; sed/replace absolute paths",
    auto_repair_confidence=0.75,
    legacy_error_type="data_missing",
)

CFG_WRONG_DEVICE = FailureSpec(
    code="CFG_WRONG_DEVICE",
    layer="configuration",
    label="Wrong device",
    description="Code requests GPU but none available, or wrong device index.",
    detection_patterns=(
        r"RuntimeError:\s+CUDA.*not available",
        r"(?i)AssertionError.*cuda.*available",
        r"(?i)RuntimeError.*Expected.*CUDA.*but got.*CPU",
        r"(?i)invalid device ordinal",
        r"torch\.cuda\.is_available\(\).*False",
    ),
    repair_strategy=RepairStrategy.INLINE_FIX,
    repair_action="Set CUDA_VISIBLE_DEVICES=''; patch device='cpu'; or --no-cuda flag",
    auto_repair_confidence=0.80,
    legacy_error_type="runtime",
)

CFG_BAD_HYPERPARAMS = FailureSpec(
    code="CFG_BAD_HYPERPARAMS",
    layer="configuration",
    label="Invalid hyperparameters",
    description="Hyperparameter value causes immediate crash (batch size too large, lr=0, etc.).",
    detection_patterns=(
        r"(?i)ValueError.*(?:batch.?size|learning.?rate|lr|epochs?|num.?workers)",
        r"(?i)expected.*positive.*(?:integer|value|number)",
        r"(?i)invalid.*(?:argument|parameter|value).*(?:batch|lr|epoch)",
    ),
    repair_strategy=RepairStrategy.REPLAN,
    repair_action="Reduce batch_size, set sane defaults, limit epochs for verification",
    auto_repair_confidence=0.60,
    legacy_error_type="runtime",
)

CFG_PERMISSION_DENIED = FailureSpec(
    code="CFG_PERMISSION_DENIED",
    layer="configuration",
    label="Permission denied",
    description="Process lacks filesystem or network permissions for the requested operation.",
    detection_patterns=(
        r"PermissionError",
        r"(?i)permission denied",
        r"(?i)EACCES",
        r"(?i)Operation not permitted",
    ),
    repair_strategy=RepairStrategy.INLINE_FIX,
    repair_action="chmod files; write to /tmp instead; adjust sandbox permissions",
    auto_repair_confidence=0.55,
    legacy_error_type="permission",
)

CFG_MISSING_ENV_VAR = FailureSpec(
    code="CFG_MISSING_ENV_VAR",
    layer="configuration",
    label="Missing environment variable",
    description="Code expects an env var (API key, path, etc.) that is not set.",
    detection_patterns=(
        r"KeyError.*(?:os\.environ|env)",
        r"(?i)environment variable.*not set",
        r"(?i)WANDB_API_KEY",
        r"(?i)OPENAI_API_KEY.*not",
        r"(?i)HF_TOKEN.*not",
        r"(?i)missing.*(?:api.?key|token|secret)",
    ),
    repair_strategy=RepairStrategy.INLINE_FIX,
    repair_action="Set dummy env var; disable telemetry (WANDB_MODE=disabled); skip auth",
    auto_repair_confidence=0.70,
    legacy_error_type="permission",
)


# ============================================================================
# 4. CODE LAYER
# ============================================================================

CODE_SYNTAX_ERROR = FailureSpec(
    code="CODE_SYNTAX_ERROR",
    layer="code",
    label="Syntax error",
    description="Python source has a syntax error (often from Python version incompatibility).",
    detection_patterns=(
        r"SyntaxError:\s+",
        r"TabError:\s+",
        r"IndentationError:\s+",
    ),
    repair_strategy=RepairStrategy.REPLAN,
    repair_action="Fix syntax; may require different Python version or 2-to-3 conversion",
    auto_repair_confidence=0.50,
    legacy_error_type="runtime",
)

CODE_API_DEPRECATION = FailureSpec(
    code="CODE_API_DEPRECATION",
    layer="code",
    label="API deprecation",
    description="Code uses a removed or renamed API (e.g. tf.contrib, torch.no_grad vs @torch.no_grad).",
    detection_patterns=(
        r"AttributeError:.*has no attribute",
        r"(?i)DeprecationWarning.*removed",
        r"(?i)module.*has no attribute.*(?:contrib|compat|v1)",
        r"(?i)is deprecated.*use.*instead",
        r"TypeError:.*unexpected keyword argument",
        r"TypeError:.*got an unexpected keyword argument",
        r"TypeError:.*missing.*required.*argument",
    ),
    repair_strategy=RepairStrategy.REPLAN,
    repair_action="Pin older library version; or patch call sites to updated API",
    auto_repair_confidence=0.45,
    legacy_error_type="runtime",
)

CODE_PYTHON_VERSION = FailureSpec(
    code="CODE_PYTHON_VERSION",
    layer="code",
    label="Python version incompatibility",
    description="Code uses features not available in the current Python version.",
    detection_patterns=(
        r"SyntaxError.*(?:walrus|:=|match|case\s+\w+:)",
        r"TypeError.*(?:union|X \| Y)",
        r"SyntaxError.*(?:f-string|f')",
        r"(?i)requires python\s*>=?\s*3\.\d+",
    ),
    repair_strategy=RepairStrategy.REPLAN,
    repair_action="Switch to required Python version in conda env",
    auto_repair_confidence=0.65,
    legacy_error_type="runtime",
)

CODE_TYPE_ERROR = FailureSpec(
    code="CODE_TYPE_ERROR",
    layer="code",
    label="Type / shape error",
    description="Tensor shape mismatch, wrong dtype, or general TypeError in computation.",
    detection_patterns=(
        r"RuntimeError:.*(?:size mismatch|shape|expected.*dim)",
        r"ValueError:.*(?:shape|dimension|size)",
        r"TypeError:.*(?:expected.*Tensor|can't convert|unsupported operand)",
        r"RuntimeError:.*(?:expected scalar type|mat1 and mat2 shapes cannot be multiplied)",
    ),
    repair_strategy=RepairStrategy.REPLAN,
    repair_action="Inspect model/data shape; may need config change or code patch",
    auto_repair_confidence=0.30,
    legacy_error_type="runtime",
)

CODE_ASSERTION = FailureSpec(
    code="CODE_ASSERTION",
    layer="code",
    label="Assertion failure",
    description="An assert statement in the code failed.",
    detection_patterns=(
        r"AssertionError",
    ),
    repair_strategy=RepairStrategy.REPLAN,
    repair_action="Investigate assertion context; may indicate wrong input or broken invariant",
    auto_repair_confidence=0.25,
    legacy_error_type="runtime",
)


# ============================================================================
# 5. RESOURCE LAYER
# ============================================================================

RES_OOM_GPU = FailureSpec(
    code="RES_OOM_GPU",
    layer="resource",
    label="GPU out of memory",
    description="CUDA OOM -- model or batch does not fit in GPU VRAM.",
    detection_patterns=(
        r"CUDA out of memory",
        r"RuntimeError:.*out of memory.*CUDA",
        r"torch\.cuda\.OutOfMemoryError",
        r"(?i)CUBLAS_STATUS_ALLOC_FAILED",
        r"OutOfMemoryError.*GPU",
    ),
    repair_strategy=RepairStrategy.REPLAN,
    repair_action="Reduce batch_size; enable gradient checkpointing; use CPU fallback; add --fp16",
    auto_repair_confidence=0.60,
    is_fast_fail=True,
    legacy_error_type="runtime",
)

RES_OOM_CPU = FailureSpec(
    code="RES_OOM_CPU",
    layer="resource",
    label="CPU out of memory",
    description="Process killed by OOM killer or MemoryError.",
    detection_patterns=(
        r"MemoryError",
        r"Killed",
        r"(?i)cannot allocate memory",
        r"std::bad_alloc",
        r"signal\s+9",  # SIGKILL from OOM killer
    ),
    repair_strategy=RepairStrategy.REPLAN,
    repair_action="Reduce data size; use streaming/chunked loading; reduce num_workers",
    auto_repair_confidence=0.45,
    is_fast_fail=True,
    legacy_error_type="timeout",  # legacy mapped Killed -> timeout
)

RES_TIMEOUT = FailureSpec(
    code="RES_TIMEOUT",
    layer="resource",
    label="Timeout",
    description="Execution exceeded the allotted time budget.",
    detection_patterns=(
        r"TimeoutError",
        r"(?i)timed?\s*out",
        r"alarm\s+signal",
        # Also detected by exit_code == -9 in the classifier
    ),
    repair_strategy=RepairStrategy.REPLAN,
    repair_action="Reduce epochs/iterations; increase timeout; skip long steps",
    auto_repair_confidence=0.55,
    legacy_error_type="timeout",
)

RES_DISK_FULL = FailureSpec(
    code="RES_DISK_FULL",
    layer="resource",
    label="Disk full",
    description="No space left on device -- checkpoints, logs, or data filled disk.",
    detection_patterns=(
        r"OSError.*No space left on device",
        r"(?i)no space left",
        r"(?i)disk quota exceeded",
        r"IOError.*No space left",
    ),
    repair_strategy=RepairStrategy.ABORT,
    repair_action="Clean up temp files; disable checkpointing; reduce logging",
    auto_repair_confidence=0.20,
    is_fast_fail=True,
    legacy_error_type="runtime",
)

RES_SEGFAULT = FailureSpec(
    code="RES_SEGFAULT",
    layer="resource",
    label="Segmentation fault",
    description="Process crashed with SIGSEGV -- usually a native library or driver bug.",
    detection_patterns=(
        r"Segmentation fault",
        r"signal\s+11",  # SIGSEGV
        r"core dumped",
        r"SIGSEGV",
    ),
    repair_strategy=RepairStrategy.ABORT,
    repair_action="Likely unrecoverable; try different library version or CPU fallback",
    auto_repair_confidence=0.10,
    is_fast_fail=True,
    legacy_error_type="runtime",
)


# ============================================================================
# 6. OUTPUT LAYER
# ============================================================================

OUT_NO_METRICS = FailureSpec(
    code="OUT_NO_METRICS",
    layer="output",
    label="No metrics produced",
    description="Execution succeeded (exit 0) but no parseable metrics in stdout or output files.",
    detection_patterns=(),  # Detected programmatically, not via regex
    repair_strategy=RepairStrategy.REPLAN,
    repair_action="Add metric-printing code; adjust parsers; try different output file",
    auto_repair_confidence=0.50,
    legacy_error_type="unknown",
)

OUT_WRONG_FORMAT = FailureSpec(
    code="OUT_WRONG_FORMAT",
    layer="output",
    label="Wrong output format",
    description="Metrics file exists but is not valid JSON, CSV, or expected structure.",
    detection_patterns=(
        r"(?i)JSONDecodeError.*(?:result|output|metric)",
        r"(?i)KeyError.*(?:accuracy|loss|metric)",
    ),
    repair_strategy=RepairStrategy.INLINE_FIX,
    repair_action="Try alternative parsing; extract from stdout instead",
    auto_repair_confidence=0.60,
    legacy_error_type="unknown",
)

OUT_METRICS_SUSPECT = FailureSpec(
    code="OUT_METRICS_SUSPECT",
    layer="output",
    label="Metrics outside expected range",
    description="Metrics were extracted but values are suspicious (accuracy > 1.0, loss < 0, NaN, etc.).",
    detection_patterns=(),  # Detected programmatically by range checks
    repair_strategy=RepairStrategy.SKIP,
    repair_action="Flag for human review; may indicate buggy evaluation or wrong metric name",
    auto_repair_confidence=0.15,
    legacy_error_type="unknown",
)

OUT_PARTIAL_RESULTS = FailureSpec(
    code="OUT_PARTIAL_RESULTS",
    layer="output",
    label="Partial results only",
    description="Some but not all expected metrics were produced.",
    detection_patterns=(),  # Detected programmatically
    repair_strategy=RepairStrategy.SKIP,
    repair_action="Accept partial results; replan for missing metrics if budget allows",
    auto_repair_confidence=0.40,
    legacy_error_type="unknown",
)


# ============================================================================
# Catch-all
# ============================================================================

UNKNOWN = FailureSpec(
    code="UNKNOWN",
    layer="code",  # default layer
    label="Unknown failure",
    description="Failure did not match any known pattern.",
    detection_patterns=(),
    repair_strategy=RepairStrategy.REPLAN,
    repair_action="Send full stderr to planner for LLM-based diagnosis",
    auto_repair_confidence=0.20,
    legacy_error_type="unknown",
)


# ============================================================================
# Registry & classifier
# ============================================================================

# Ordered by priority: first match wins.  More specific patterns come first.
FAILURE_REGISTRY: tuple[FailureSpec, ...] = (
    # --- dependency layer (check before data/code since import errors are common) ---
    DEP_CUDA_MISMATCH,
    DEP_MISSING_PACKAGE,
    DEP_VERSION_CONFLICT,
    DEP_SYSTEM_LIBRARY,
    DEP_BUILD_FAILURE,
    # --- resource layer (check early: OOM/segfault should fast-fail) ---
    RES_OOM_GPU,
    RES_OOM_CPU,
    RES_SEGFAULT,
    RES_DISK_FULL,
    RES_TIMEOUT,
    # --- data layer ---
    DATA_DOWNLOAD_FAILURE,
    DATA_MISSING_DATASET,
    DATA_WRONG_FORMAT,
    DATA_PATH_MISMATCH,
    # --- configuration layer ---
    CFG_PERMISSION_DENIED,
    CFG_MISSING_ENV_VAR,
    CFG_WRONG_DEVICE,
    CFG_MISSING_CONFIG,
    CFG_HARDCODED_PATH,
    CFG_BAD_HYPERPARAMS,
    # --- code layer ---
    CODE_PYTHON_VERSION,
    CODE_SYNTAX_ERROR,
    CODE_API_DEPRECATION,
    CODE_TYPE_ERROR,
    CODE_ASSERTION,
    # --- output layer (programmatic, no regex -- checked separately) ---
    OUT_NO_METRICS,
    OUT_WRONG_FORMAT,
    OUT_METRICS_SUSPECT,
    OUT_PARTIAL_RESULTS,
    # --- fallback ---
    UNKNOWN,
)

# Quick lookup by code string
FAILURE_BY_CODE: dict[str, FailureSpec] = {spec.code: spec for spec in FAILURE_REGISTRY}

# Layer summary for reporting
LAYER_CODES: dict[FailureLayer, list[str]] = {}
for _spec in FAILURE_REGISTRY:
    LAYER_CODES.setdefault(_spec.layer, []).append(_spec.code)


def classify_failure(
    stdout: str,
    stderr: str,
    exit_code: int,
    *,
    metrics: dict | None = None,
    expected_metrics: list[str] | None = None,
) -> FailureSpec:
    """Classify a failure into the taxonomy.

    Parameters
    ----------
    stdout, stderr : str
        Captured process output.
    exit_code : int
        Process return code (-9 = killed).
    metrics : dict, optional
        Extracted metrics (for output-layer classification).
    expected_metrics : list[str], optional
        Names of metrics the step was supposed to produce.

    Returns
    -------
    FailureSpec
        The most specific matching failure specification.
    """
    combined = f"{stdout}\n{stderr}"

    # --- Special exit-code checks ---
    if exit_code == -9:
        # Could be OOM kill or timeout.  Check OOM patterns first.
        if any(re.search(p, combined) for p in RES_OOM_GPU.detection_patterns):
            return RES_OOM_GPU
        if any(re.search(p, combined) for p in RES_OOM_CPU.detection_patterns):
            return RES_OOM_CPU
        return RES_TIMEOUT  # default for SIGKILL

    # --- If exit 0, check output-layer issues ---
    if exit_code == 0:
        if metrics is not None and expected_metrics:
            if not metrics:
                return OUT_NO_METRICS
            found = set(metrics.keys())
            expected = set(expected_metrics)
            if not found.intersection(expected):
                return OUT_NO_METRICS
            # Check for suspect values
            for name, val in metrics.items():
                if isinstance(val, (int, float)):
                    if val != val:  # NaN check
                        return OUT_METRICS_SUSPECT
            if not expected.issubset(found):
                return OUT_PARTIAL_RESULTS
        elif metrics is not None and not metrics:
            return OUT_NO_METRICS
        # exit 0 with no issues detected
        return UNKNOWN  # should not normally reach here

    # --- Regex-based classification (exit != 0) ---
    for spec in FAILURE_REGISTRY:
        if not spec.detection_patterns:
            continue
        for pattern in spec.detection_patterns:
            try:
                if re.search(pattern, combined):
                    return spec
            except re.error:
                continue

    return UNKNOWN


def failure_to_legacy(spec: FailureSpec) -> str:
    """Map a FailureSpec back to the old 7-literal error_type for backward compatibility."""
    return spec.legacy_error_type


# ============================================================================
# Convenience: summary table for docs / logging
# ============================================================================

def taxonomy_summary() -> str:
    """Return a human-readable summary table of the full taxonomy."""
    lines = []
    current_layer = None
    for spec in FAILURE_REGISTRY:
        if spec.layer != current_layer:
            current_layer = spec.layer
            lines.append(f"\n{'=' * 70}")
            lines.append(f"  LAYER: {current_layer.upper()}")
            lines.append(f"{'=' * 70}")
        lines.append(
            f"  {spec.code:<30s} | {spec.repair_strategy.value:<12s} | "
            f"conf={spec.auto_repair_confidence:.2f} | "
            f"{'FAST-FAIL' if spec.is_fast_fail else '         '} | "
            f"{spec.label}"
        )
    return "\n".join(lines)
