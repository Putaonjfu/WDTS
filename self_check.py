"""Small, reproducible checks for the refactored WDTS interfaces."""

from pathlib import Path
import os
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import open3d as o3d

import skeletonization as skeletonization_module
from cli import build_parser
from interwoven_optimization import (
    InterwovenOptimizationConfig,
    InterwovenOptimizationResult,
    load_skeleton,
    load_tls_point_cloud,
    run_interwoven_optimization,
)
from pipeline import run_pipeline
from skeletonization import (
    SkeletonizationConfig,
    SkeletonizationResult,
    contract_water_droplets,
)
from skeletonization import load_point_cloud as load_stage1_point_cloud


REPOSITORY_ROOT = Path(__file__).resolve().parent
SAMPLE_DIRECTORY = REPOSITORY_ROOT / "example_data"


class SampleDataContractTests(unittest.TestCase):
    def test_headerless_tls_keeps_the_first_point(self):
        points = load_tls_point_cloud(SAMPLE_DIRECTORY / "Tree_1.txt")
        self.assertEqual(points.shape, (84556, 3))
        np.testing.assert_allclose(points[0], [9.59225270, 30.63890080, 3.70904450])

    def test_reference_skeleton_ply_contract(self):
        path = SAMPLE_DIRECTORY / "Tree_1_ske_without_interwoven_op.ply"
        points = load_skeleton(path)
        line_set = o3d.io.read_line_set(str(path))
        edges = np.asarray(line_set.lines)
        self.assertEqual(points.shape, (2780, 3))
        self.assertEqual(edges.shape, (2779, 2))
        self.assertGreaterEqual(int(edges.min()), 0)
        self.assertLess(int(edges.max()), len(points))

    def test_text_loaders_accept_headered_comma_text(self):
        with tempfile.TemporaryDirectory(prefix="wdts loader test ") as temporary_directory:
            path = Path(temporary_directory) / "points with commas.txt"
            path.write_text(
                "\n# example input\nx,y,z,intensity\n"
                "1.0,2.0,3.0,10\n4.0,5.0,6.0,20\n7.0,8.0,9.0,30\n",
                encoding="utf-8",
            )
            expected = np.array(
                [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]
            )
            np.testing.assert_allclose(load_stage1_point_cloud(path), expected)
            np.testing.assert_allclose(load_tls_point_cloud(path), expected)

    def test_text_loaders_reject_corrupt_or_nonfinite_coordinates(self):
        with tempfile.TemporaryDirectory(prefix="wdts invalid input test ") as temporary_directory:
            directory = Path(temporary_directory)
            corrupt_path = directory / "corrupt.txt"
            corrupt_path.write_text("oops 2 3\n1 2 3\n4 5 6\n7 8 9\n", encoding="utf-8")
            infinite_path = directory / "infinite.txt"
            infinite_path.write_text("1 2 3\n4 5 inf\n7 8 9\n", encoding="utf-8")
            for loader in (load_stage1_point_cloud, load_tls_point_cloud):
                with self.assertRaises(ValueError):
                    loader(corrupt_path)
                with self.assertRaises(ValueError):
                    loader(infinite_path)


class AlgorithmSmokeTests(unittest.TestCase):
    @staticmethod
    def _synthetic_tree(number_of_points=180):
        rng = np.random.RandomState(7)
        z = np.linspace(0.02, 2.0, number_of_points)
        theta = np.linspace(0.0, 12.0 * np.pi, number_of_points)
        radius = 0.08 + 0.005 * rng.randn(number_of_points)
        return np.column_stack((radius * np.cos(theta), radius * np.sin(theta), z))

    def test_water_droplet_array_api_builds_a_tree(self):
        points = self._synthetic_tree()
        config = SkeletonizationConfig(
            tree_root=np.array([0.0, 0.0, 0.0]),
            base_radius=0.1,
            max_radius=0.12,
        )
        config.gamma = 0.1
        config.max_iterations = 3
        config.tree_growth_start_iteration = 3

        real_minimize = skeletonization_module.minimize
        failed_once = {"value": False}

        def fail_the_first_local_optimization(*args, **kwargs):
            if not failed_once["value"]:
                failed_once["value"] = True
                return SimpleNamespace(success=False)
            return real_minimize(*args, **kwargs)

        with patch.object(
            skeletonization_module,
            "minimize",
            side_effect=fail_the_first_local_optimization,
        ):
            skeleton_points, skeleton_edges = contract_water_droplets(
                points,
                config,
                savepath="unused",
                tree_name="synthetic",
            )
        self.assertTrue(failed_once["value"])
        self.assertEqual(skeleton_points.shape[1], 10)
        self.assertGreater(len(skeleton_points), 1)
        self.assertEqual(skeleton_edges.shape, (len(skeleton_points) - 1, 2))

    def test_interwoven_file_api_writes_a_valid_ply(self):
        z = np.linspace(0.0, 1.1, 12)
        center_x = 0.01 * np.sin(4.0 * z)
        center_y = 0.008 * np.cos(3.0 * z) + 0.002 * np.sin(7.0 * z)
        skeleton_points = np.column_stack((center_x, center_y, z))
        skeleton_edges = np.column_stack((np.arange(11), np.arange(1, 12)))

        ring_z = np.repeat(z, 24)
        theta = np.tile(np.linspace(0.0, 2.0 * np.pi, 24, endpoint=False), len(z))
        tls_points = np.column_stack(
            (
                np.repeat(center_x, 24) + 0.06 * np.cos(theta),
                np.repeat(center_y, 24) + 0.06 * np.sin(theta),
                ring_z,
            )
        )

        with tempfile.TemporaryDirectory(prefix="wdts test ") as temporary_directory:
            directory = Path(temporary_directory)
            tls_path = directory / "tree tls.txt"
            skeleton_path = directory / "initial skeleton.ply"
            output_path = directory / "optimized skeleton.ply"
            np.savetxt(str(tls_path), tls_points, fmt="%.8f")
            line_set = o3d.geometry.LineSet(
                points=o3d.utility.Vector3dVector(skeleton_points),
                lines=o3d.utility.Vector2iVector(skeleton_edges),
            )
            self.assertTrue(o3d.io.write_line_set(str(skeleton_path), line_set, write_ascii=True))

            config = InterwovenOptimizationConfig(
                max_major_iterations=1,
                refinement_iterations=1,
                n_jobs=1,
            )
            config.edge_ref_dir[(99, 100)] = np.array([1.0, 0.0, 0.0])
            result = run_interwoven_optimization(
                tls_path=tls_path,
                skeleton_path=skeleton_path,
                output_path=output_path,
                config=config,
            )
            self.assertTrue(output_path.is_file())
            self.assertEqual(result.points.shape, (12, 3))
            self.assertEqual(result.edges.shape, (11, 2))
            self.assertTrue(np.all(np.isfinite(result.points)))
            self.assertEqual(config.global_tree_radius, 0.0)
            self.assertIn((99, 100), config.edge_ref_dir)


class CommandLineContractTests(unittest.TestCase):
    def test_import_has_no_filesystem_side_effects(self):
        with tempfile.TemporaryDirectory(prefix="wdts import test ") as temporary_directory:
            environment = os.environ.copy()
            environment["PYTHONPATH"] = str(REPOSITORY_ROOT)
            completed = subprocess.run(
                [sys.executable, "-c", "import wdts"],
                cwd=temporary_directory,
                env=environment,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertEqual(list(Path(temporary_directory).iterdir()), [])

    def test_pipeline_arguments_are_discoverable(self):
        arguments = build_parser().parse_args(
            ["pipeline", "tree.txt", "--output", "results", "--gamma", "0.1"]
        )
        self.assertEqual(arguments.command, "pipeline")
        self.assertEqual(arguments.input, Path("tree.txt"))
        self.assertEqual(arguments.output, Path("results"))
        self.assertEqual(arguments.gamma, 0.1)

    def test_pipeline_routes_canonical_paths(self):
        initial = SkeletonizationResult(
            points=np.zeros((2, 3)),
            edges=np.array([[0, 1]]),
            input_path=Path("input/tree.txt"),
            output_directory=Path("out/oak"),
            points_path=Path("out/oak/initial_skeleton_points.txt"),
            skeleton_path=Path("out/oak/initial_skeleton.ply"),
        )
        optimized = InterwovenOptimizationResult(
            points=np.zeros((2, 3)),
            edges=np.array([[0, 1]]),
            output_path=Path("out/oak/optimized_skeleton.ply"),
        )
        with patch("pipeline.run_skeletonization", return_value=initial) as stage_one:
            with patch("pipeline.run_interwoven_optimization", return_value=optimized) as stage_two:
                result = run_pipeline(
                    "input/tree.txt",
                    output_dir="out",
                    tree_id="oak",
                    gamma=0.3,
                )
        self.assertEqual(result.initial_skeleton_path, initial.skeleton_path)
        self.assertEqual(result.optimized_skeleton_path, optimized.output_path)
        self.assertEqual(stage_one.call_args[1]["gamma"], 0.3)
        self.assertEqual(stage_two.call_args[1]["skeleton_path"], initial.skeleton_path)
        self.assertEqual(
            stage_two.call_args[1]["output_path"],
            Path("out/oak/optimized_skeleton.ply"),
        )

    def test_invalid_interwoven_config_and_output_collision_are_rejected(self):
        with self.assertRaises(ValueError):
            InterwovenOptimizationConfig(topology_k=0).validate()
        with self.assertRaises(ValueError):
            run_interwoven_optimization(
                tls_path="same.ply",
                skeleton_path="initial.ply",
                output_path="same.ply",
                config=InterwovenOptimizationConfig(),
            )


if __name__ == "__main__":
    unittest.main()
