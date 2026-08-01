"""High-level orchestration for the complete two-stage WDTS workflow."""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from interwoven_optimization import (
    InterwovenOptimizationConfig,
    InterwovenOptimizationResult,
    run_interwoven_optimization,
)
from skeletonization import (
    PathLike,
    SkeletonizationConfig,
    SkeletonizationResult,
    run_skeletonization,
)


@dataclass
class PipelineResult:
    """Results from initial skeletonization and interwoven optimization."""

    skeletonization: SkeletonizationResult
    optimization: InterwovenOptimizationResult

    @property
    def initial_skeleton_path(self) -> Path:
        return self.skeletonization.skeleton_path

    @property
    def optimized_skeleton_path(self) -> Path:
        return self.optimization.output_path


def run_pipeline(
    input_path: PathLike,
    output_dir: PathLike = "results",
    gamma: Optional[float] = None,
    tree_id: Optional[str] = None,
    skeletonization_config: Optional[SkeletonizationConfig] = None,
    interwoven_config: Optional[InterwovenOptimizationConfig] = None,
) -> PipelineResult:
    """Run skeletonization followed by geometric-topological optimization."""
    input_path = Path(input_path)
    tree_id = tree_id or input_path.stem
    initial = run_skeletonization(
        input_path=input_path,
        output_dir=output_dir,
        gamma=gamma,
        tree_id=tree_id,
        config=skeletonization_config,
    )

    optimized_path = Path(output_dir) / tree_id / "optimized_skeleton.ply"
    optimized = run_interwoven_optimization(
        tls_path=input_path,
        skeleton_path=initial.skeleton_path,
        output_path=optimized_path,
        config=interwoven_config,
    )
    return PipelineResult(skeletonization=initial, optimization=optimized)
