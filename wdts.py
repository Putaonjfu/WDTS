"""Public Python API and module launcher for WDTS."""

from interwoven_optimization import (
    InterwovenOptimizationConfig,
    InterwovenOptimizationResult,
    run_interwoven_optimization,
)
from pipeline import PipelineResult, run_pipeline
from skeletonization import (
    SkeletonizationConfig,
    SkeletonizationResult,
    run_skeletonization,
)

__version__ = "1.0.0"

__all__ = [
    "__version__",
    "SkeletonizationConfig",
    "SkeletonizationResult",
    "InterwovenOptimizationConfig",
    "InterwovenOptimizationResult",
    "PipelineResult",
    "run_skeletonization",
    "run_interwoven_optimization",
    "run_pipeline",
]


if __name__ == "__main__":
    from cli import main

    raise SystemExit(main())
