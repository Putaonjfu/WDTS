"""Command-line interface for WDTS."""

import argparse
from pathlib import Path
from typing import Optional, Sequence

from . import __version__
from .interwoven_optimization import InterwovenOptimizationConfig, run_interwoven_optimization
from .pipeline import run_pipeline
from .skeletonization import SkeletonizationConfig, run_skeletonization


def _positive_integer(value):
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be at least 1")
    return parsed


def _nonzero_integer(value):
    parsed = int(value)
    if parsed == 0:
        raise argparse.ArgumentTypeError("value cannot be zero; use -1 for all cores")
    return parsed


def _add_common_skeletonization_arguments(parser):
    parser.add_argument("input", type=Path, help="TLS point cloud (.txt, .csv, .xyz, .ply or .pcd).")
    parser.add_argument("-o", "--output", type=Path, default=Path("results"), help="Output root directory.")
    parser.add_argument("--tree-id", help="Output tree identifier; defaults to the input file stem.")
    parser.add_argument("--gamma", type=float, default=0.1, help="Droplet contraction/merging weight (default: 0.1).")
    parser.add_argument(
        "--skeleton-iterations",
        type=_positive_integer,
        default=5,
        metavar="N",
        help="Maximum water-droplet iterations (default: 5).",
    )
    parser.add_argument(
        "--save-intermediate",
        action="store_true",
        help="Keep per-iteration diagnostic files.",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python run_wdts.py",
        description="Water-droplet tree skeletonization with interwoven optimization.",
    )
    parser.add_argument("--version", action="version", version=f"WDTS {__version__}")
    commands = parser.add_subparsers(dest="command", required=True)

    pipeline_parser = commands.add_parser("pipeline", help="Run both WDTS stages.")
    _add_common_skeletonization_arguments(pipeline_parser)
    pipeline_parser.add_argument(
        "--major-iterations",
        type=_positive_integer,
        default=2,
        metavar="N",
        help="Maximum interwoven major iterations (default: 2).",
    )
    pipeline_parser.add_argument(
        "--refinement-iterations",
        type=_positive_integer,
        default=10,
        metavar="N",
        help="Geometry refinements per major iteration (default: 10).",
    )
    pipeline_parser.add_argument("--jobs", type=_nonzero_integer, default=-1, help="Parallel jobs for edge fitting (default: all cores).")

    skeleton_parser = commands.add_parser("skeletonize", help="Run only initial skeletonization.")
    _add_common_skeletonization_arguments(skeleton_parser)

    optimize_parser = commands.add_parser("optimize", help="Run only interwoven optimization.")
    optimize_parser.add_argument("--tls", type=Path, required=True, help="Original TLS point cloud.")
    optimize_parser.add_argument("--skeleton", type=Path, required=True, help="Initial skeleton (.ply, .txt or .csv).")
    optimize_parser.add_argument("-o", "--output", type=Path, required=True, help="Final skeleton .ply path.")
    optimize_parser.add_argument("--major-iterations", type=_positive_integer, default=2, metavar="N")
    optimize_parser.add_argument("--refinement-iterations", type=_positive_integer, default=10, metavar="N")
    optimize_parser.add_argument("--jobs", type=_nonzero_integer, default=-1)
    optimize_parser.add_argument("--save-intermediate", action="store_true")
    return parser


def _skeletonization_config(args) -> SkeletonizationConfig:
    config = SkeletonizationConfig()
    config.max_iterations = args.skeleton_iterations
    config.save_intermediate_results = args.save_intermediate
    return config


def _interwoven_config(args) -> InterwovenOptimizationConfig:
    config = InterwovenOptimizationConfig()
    config.max_major_iterations = args.major_iterations
    config.refinement_iterations = args.refinement_iterations
    config.n_jobs = args.jobs
    config.save_intermediate_results = args.save_intermediate
    return config


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Parse CLI arguments and execute the selected WDTS stage."""
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "skeletonize":
            result = run_skeletonization(
                input_path=args.input,
                output_dir=args.output,
                gamma=args.gamma,
                tree_id=args.tree_id,
                config=_skeletonization_config(args),
            )
            print(f"Initial skeleton: {result.skeleton_path}")
        elif args.command == "optimize":
            result = run_interwoven_optimization(
                tls_path=args.tls,
                skeleton_path=args.skeleton,
                output_path=args.output,
                config=_interwoven_config(args),
            )
            print(f"Optimized skeleton: {result.output_path}")
        else:
            result = run_pipeline(
                input_path=args.input,
                output_dir=args.output,
                gamma=args.gamma,
                tree_id=args.tree_id,
                skeletonization_config=_skeletonization_config(args),
                interwoven_config=_interwoven_config(args),
            )
            print(f"Initial skeleton: {result.initial_skeleton_path}")
            print(f"Optimized skeleton: {result.optimized_skeleton_path}")
    except (OSError, ValueError, RuntimeError) as error:
        parser.exit(1, f"WDTS error: {error}\n")
    return 0
