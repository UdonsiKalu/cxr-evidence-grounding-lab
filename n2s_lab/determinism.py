"""HF / CUDA determinism for N2S Ph9–14 and Workbench.

Greedy decode alone is not enough on GPU bf16 — without these settings,
commit-token margins can jitter across the decision boundary (±0.13).

Call ``configure_determinism()`` once at process start and again before each
generate / intervene. Safe to call repeatedly.
"""

from __future__ import annotations

import os
import random
from typing import Any

# Must be set before the first cuBLAS workspace alloc when using
# deterministic algorithms on CUDA >= 10.2.
_DEFAULT_CUBLAS = ":4096:8"
_DEFAULT_SEED = 0


def default_seed() -> int:
    raw = os.environ.get("N2S_HF_SEED", str(_DEFAULT_SEED))
    try:
        return int(raw)
    except ValueError:
        return _DEFAULT_SEED


def configure_determinism(seed: int | None = None) -> dict[str, Any]:
    """Lock RNG + CUDA/cuDNN for bit-stable greedy HF runs.

    Returns a small status dict (logged by callers / written into artifacts).
    """
    import torch

    if seed is None:
        seed = default_seed()

    # cuBLAS workspace for deterministic matmul (no-op if already set).
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", _DEFAULT_CUBLAS)
    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Disable TF32 — can change bf16 matmul results across runs.
    if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
        torch.backends.cuda.matmul.allow_tf32 = False
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.allow_tf32 = False

    det_mode = "off"
    try:
        # warn_only: some attention kernels still lack deterministic impls;
        # we still get cudnn/TF32 locks + fixed seeds for greedy paths.
        torch.use_deterministic_algorithms(True, warn_only=True)
        det_mode = "warn_only"
    except TypeError:
        try:
            torch.use_deterministic_algorithms(True)
            det_mode = "strict"
        except Exception as exc:  # noqa: BLE001 — report, don't crash science path
            det_mode = f"failed:{exc}"

    return {
        "seed": seed,
        "cublas_workspace": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "cudnn_deterministic": True,
        "tf32": False,
        "deterministic_algorithms": det_mode,
        "do_sample": False,
    }


def reseeds_before_generate(seed: int | None = None) -> int:
    """Re-apply seed immediately before a generate loop (extra safety)."""
    import torch

    if seed is None:
        seed = default_seed()
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return seed
