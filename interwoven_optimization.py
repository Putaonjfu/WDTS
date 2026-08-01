"""Geometric and topological interwoven optimization for WDTS skeletons.

The public entry point is :func:`run_interwoven_optimization`.  It accepts one
TLS point-cloud file, one skeleton file, and an output path, so callers do not
need to edit source-level directory constants before running the algorithm.

The numerical procedure is intentionally kept separate from file orchestration:
geometry refinement alternates with topology reconstruction, while optional
intermediate files are controlled by :class:`InterwovenOptimizationConfig`.
"""

from __future__ import annotations

from collections import defaultdict, deque
from copy import deepcopy
from dataclasses import dataclass, field
import heapq
from pathlib import Path

from joblib import Parallel, delayed
import numpy as np
import open3d as o3d
import pandas as pd
from scipy.optimize import minimize
from scipy.spatial import Delaunay, cKDTree as KDTree

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False


@dataclass
class InterwovenOptimizationConfig:
    """Configuration for geometric/topological interwoven optimization.

    ``global_tree_radius``, ``max_rotation_angle``, and the edge-reference maps
    are runtime state used by the original algorithm.  A fresh configuration is
    created automatically for each call when ``config`` is omitted.
    """

    debug_mode: bool = False
    n_jobs: int = -1
    save_intermediate_results: bool = False

    max_major_iterations: int = 2
    convergence_threshold: float = 0.0001

    global_tree_radius: float = 0.0
    association_radius_factor: float = 2.0

    refinement_iterations: int = 10
    step_rate: float = 0.1

    optimizer_method: str = "SLSQP"

    enable_edge_noise_filtering: bool = True
    noise_filtering_n_std: float = 1.0

    enable_length_preservation: bool = True
    use_bounding_sphere_constraint: bool = True

    enable_rotation_constraint: bool = True
    rotation_decay_start: float = 20.0
    rotation_decay_end: float = 5.0
    max_rotation_angle: float = 20.0

    use_fixed_reference_for_rotation: bool = True
    use_fixed_reference_for_length: bool = True

    edge_ref_dir: dict[tuple[int, int], np.ndarray] = field(default_factory=dict)
    edge_ref_len: dict[tuple[int, int], float] = field(default_factory=dict)

    enable_soft_assignment: bool = False
    soft_assignment_topk: int = 3
    soft_assignment_sigma: float = 0.02
    soft_assignment_temperature: float = 1.0
    force_hard_winner: bool = True

    enable_global_smoothing: bool = True
    laplacian_lambda: float = 0.05
    twohop_lambda: float = 0.05
    smoothing_iterations: int = 1

    enable_post_projection: bool = True
    length_projection_weight: float = 0.8
    angle_projection_weight: float = 0.6
    projection_iterations: int = 2

    enable_topology_update: bool = True
    topology_update_every_n_major: int = 1

    enable_min_edge_length: bool = True
    min_edge_length_factor: float = 0.9
    min_edge_projection_weight: float = 0.9
    min_edge_projection_iterations: int = 1

    topology_prune_percentile: float = 70.0
    topology_k: int = 30
    topology_alpha: float = 0.5
    topology_beta: float = 0.5

    @property
    def DEBUG_MODE(self) -> bool:
        """Backward-compatible alias for :attr:`debug_mode`."""
        return self.debug_mode

    @DEBUG_MODE.setter
    def DEBUG_MODE(self, value: bool) -> None:
        self.debug_mode = bool(value)

    @property
    def N_JOBS(self) -> int:
        """Backward-compatible alias for :attr:`n_jobs`."""
        return self.n_jobs

    @N_JOBS.setter
    def N_JOBS(self, value: int) -> None:
        self.n_jobs = int(value)

    @property
    def topology_K(self) -> int:
        """Backward-compatible alias for :attr:`topology_k`."""
        return self.topology_k

    @topology_K.setter
    def topology_K(self, value: int) -> None:
        self.topology_k = int(value)

    def validate(self) -> None:
        """Reject configuration values that cannot produce a valid run."""
        if self.max_major_iterations < 1:
            raise ValueError("max_major_iterations must be at least 1.")
        if self.refinement_iterations < 1:
            raise ValueError("refinement_iterations must be at least 1.")
        if self.n_jobs == 0:
            raise ValueError("n_jobs cannot be zero; use -1 for all available cores.")
        if self.association_radius_factor <= 0:
            raise ValueError("association_radius_factor must be positive.")
        if self.convergence_threshold < 0:
            raise ValueError("convergence_threshold cannot be negative.")
        if self.enable_topology_update and self.topology_update_every_n_major < 1:
            raise ValueError("topology_update_every_n_major must be at least 1.")
        if self.topology_k < 1:
            raise ValueError("topology_k must be at least 1.")
        if not 0.0 <= self.topology_prune_percentile <= 100.0:
            raise ValueError("topology_prune_percentile must be between 0 and 100.")
        if self.enable_soft_assignment and not self.force_hard_winner and self.soft_assignment_topk < 1:
            raise ValueError("soft_assignment_topk must be at least 1 when soft assignment is active.")


@dataclass(frozen=True)
class InterwovenOptimizationResult:
    """Result returned by :func:`run_interwoven_optimization`."""

    points: np.ndarray
    edges: np.ndarray
    output_path: Path


# The historical class name remains importable for existing scripts.
Parameters = InterwovenOptimizationConfig


__all__ = [
    "InterwovenOptimizationConfig",
    "InterwovenOptimizationResult",
    "Parameters",
    "estimate_trunk_base_radius",
    "optimize_skeleton_geometry",
    "reconstruct_skeleton_topology",
    "load_tls_point_cloud",
    "load_skeleton",
    "save_skeleton",
    "run_interwoven_optimization",
    "Interwoven_optimization",
]


def _canonical_edge(u: int, v: int) -> tuple[int, int]:
    return (u, v) if u < v else (v, u)


def _unit_vector_or_none(vector: np.ndarray):
    n = np.linalg.norm(vector)
    if n < 1e-12:
        return None
    return vector / n


def initialize_edge_references(parameters, sk_coords, edges):
    for (u, v) in edges:
        k = _canonical_edge(int(u), int(v))
        if k in parameters.edge_ref_dir and k in parameters.edge_ref_len:
            continue
        vec = sk_coords[int(v)] - sk_coords[int(u)]
        L = float(np.linalg.norm(vec))
        if L < 1e-9:
            continue
        d = vec / L
        parameters.edge_ref_dir[k] = d
        parameters.edge_ref_len[k] = L


def _optimize_single_edge(
    edge_idx,
    edge,
    assigned_items,
    tls_coords,
    sk_coords_current,
    parameters,
):
    u, v = int(edge[0]), int(edge[1])

    if assigned_items is None:
        return None

    if len(assigned_items) == 0:
        return None

    if isinstance(assigned_items[0], (tuple, list)) and len(assigned_items[0]) == 2:
        tls_indices = np.array([int(a) for a, _ in assigned_items], dtype=int)
        tls_weights = np.array([float(w) for _, w in assigned_items], dtype=float)
    else:
        tls_indices = np.array([int(i) for i in assigned_items], dtype=int)
        tls_weights = np.ones(len(tls_indices), dtype=float)

    if len(tls_indices) < 5:
        return None

    assigned_tls_points = tls_coords[tls_indices]
    w_sum = float(np.sum(tls_weights))
    if w_sum < 1e-12:
        return None
    tls_weights = tls_weights / w_sum

    u_orig, v_orig = sk_coords_current[u], sk_coords_current[v]
    original_length_sq = float(np.sum((u_orig - v_orig) ** 2))
    if original_length_sq < 1e-9:
        return None

    optimizer_method = getattr(parameters, 'optimizer_method', 'SLSQP')
    use_bounding_sphere = parameters.use_bounding_sphere_constraint
    enable_length_preservation = parameters.enable_length_preservation
    enable_rotation_constraint = parameters.enable_rotation_constraint
    max_rotation_angle = float(parameters.max_rotation_angle)

    def objective_to_minimize(xyz_params):
        p1, p2 = xyz_params[0:3], xyz_params[3:6]
        line_vec = p2 - p1
        l2 = float(np.dot(line_vec, line_vec))
        if l2 < 1e-12:
            d = np.linalg.norm(assigned_tls_points - p1, axis=1)
        else:
            t = np.dot(assigned_tls_points - p1, line_vec) / l2
            t = np.clip(t, 0.0, 1.0)
            proj = p1 + t[:, None] * line_vec
            d = np.linalg.norm(assigned_tls_points - proj, axis=1)

        mean_d = float(np.sum(tls_weights * d))
        var_metric = float(np.sum(tls_weights * (d - mean_d) ** 2))

        return var_metric / original_length_sq

    constraints = []

    if use_bounding_sphere:
        centroid = np.sum(assigned_tls_points * tls_weights[:, None], axis=0)
        radius = float(np.max(np.linalg.norm(assigned_tls_points - centroid, axis=1)))
        if radius > 1e-9:
            def bounding_sphere_constraint(xyz):
                p1, p2 = xyz[0:3], xyz[3:6]
                dist1 = float(np.linalg.norm(p1 - centroid))
                dist2 = float(np.linalg.norm(p2 - centroid))
                return np.array([1.0 - dist1 / radius, 1.0 - dist2 / radius], dtype=float)

            constraints.append({'type': 'ineq', 'fun': bounding_sphere_constraint})

    if enable_length_preservation:
        def length_constraint_func(xyz_params):
            p1, p2 = xyz_params[0:3], xyz_params[3:6]
            new_length_sq = float(np.sum((p1 - p2) ** 2))
            return (new_length_sq / original_length_sq) - 1.0

        if optimizer_method == 'COBYLA':
            constraints.append({'type': 'ineq', 'fun': length_constraint_func})
            constraints.append({'type': 'ineq', 'fun': lambda x: -length_constraint_func(x)})
        else:
            constraints.append({'type': 'eq', 'fun': length_constraint_func})

    if enable_rotation_constraint and max_rotation_angle >= 0.0:
        if parameters.use_fixed_reference_for_rotation:
            k = _canonical_edge(u, v)
            ref_dir = parameters.edge_ref_dir.get(k, None)
        else:
            ref_dir = _unit_vector_or_none(v_orig - u_orig)

        if ref_dir is not None:
            cos_theta_max = float(np.cos(np.deg2rad(max_rotation_angle)))

            def rotation_constraint_func(xyz_params):
                p1, p2 = xyz_params[0:3], xyz_params[3:6]
                v_new = p2 - p1
                n = float(np.linalg.norm(v_new))
                if n < 1e-12:
                    return 1.0
                v_new = v_new / n
                dotp = float(np.dot(v_new, ref_dir))
                return dotp - cos_theta_max

            constraints.append({'type': 'ineq', 'fun': rotation_constraint_func})

    result = minimize(
        fun=objective_to_minimize,
        x0=np.concatenate([u_orig, v_orig]),
        method=optimizer_method,
        constraints=constraints
    )

    if result.success:
        has_assignment_weights = (
            isinstance(assigned_items[0], (tuple, list))
            and len(assigned_items[0]) == 2
        )
        edge_w = (
            float(np.sum([w for _, w in assigned_items]))
            if has_assignment_weights
            else float(len(tls_indices))
        )
        if edge_w < 1e-12:
            edge_w = 1.0
        return (u, v, result.x[0:3], result.x[3:6], edge_w)

    return None


def visualize_optimization_state(
    tls_points,
    skeleton_points,
    edges,
    edge_to_tls_items,
    title="Debug",
):
    print(f"  [Debug] Rendering: {title} ... (close the window to continue)")
    geometries = []

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(tls_points[:, :3])
    pcd.paint_uniform_color([0.7, 0.7, 0.7])
    geometries.append(pcd)

    sk_coords = skeleton_points[:, :3]

    if len(edges) > 0:
        sk_lines = o3d.geometry.LineSet()
        sk_lines.points = o3d.utility.Vector3dVector(sk_coords)
        sk_lines.lines = o3d.utility.Vector2iVector(edges)
        sk_lines.paint_uniform_color([0, 0, 1])
        geometries.append(sk_lines)

        u_indices = edges[:, 0]
        v_indices = edges[:, 1]
        edge_lengths = np.linalg.norm(sk_coords[u_indices] - sk_coords[v_indices], axis=1)
        avg_len = np.mean(edge_lengths) if len(edge_lengths) > 0 else 0.1
        node_radius = avg_len * 0.15
    else:
        node_radius = 0.05

    for point in sk_coords:
        mesh_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=node_radius, resolution=10)
        mesh_sphere.compute_vertex_normals()
        mesh_sphere.paint_uniform_color([1, 0, 0])
        mesh_sphere.translate(point)
        geometries.append(mesh_sphere)

    all_points = []
    if len(edges) > 0:
        for edge_idx, items in edge_to_tls_items.items():
            if not items:
                continue
            u, v = edges[edge_idx]
            p_start = sk_coords[int(u)]
            p_end = sk_coords[int(v)]
            edge_vec = p_end - p_start
            edge_len_sq = float(np.dot(edge_vec, edge_vec))

            if isinstance(items[0], (tuple, list)) and len(items[0]) == 2:
                tls_idxs = [int(a) for a, _ in items]
            else:
                tls_idxs = [int(a) for a in items]

            assigned_tls = tls_points[tls_idxs, :3]
            if edge_len_sq < 1e-12:
                projections = np.tile(p_start, (len(assigned_tls), 1))
            else:
                t = np.dot(assigned_tls - p_start, edge_vec) / edge_len_sq
                t = np.clip(t, 0.0, 1.0)
                projections = p_start + t[:, None] * edge_vec

            for i in range(len(assigned_tls)):
                all_points.append(assigned_tls[i])
                all_points.append(projections[i])

        if len(all_points) > 0:
            conn_lines = o3d.geometry.LineSet()
            conn_lines.points = o3d.utility.Vector3dVector(np.array(all_points))
            N = len(all_points)
            indices = np.arange(N).reshape(-1, 2)
            conn_lines.lines = o3d.utility.Vector2iVector(indices)
            conn_lines.paint_uniform_color([0, 1, 0])
            geometries.append(conn_lines)

    if len(geometries) > 0:
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name=title, width=1024, height=768)
        for g in geometries:
            vis.add_geometry(g)
        opt = vis.get_render_option()
        opt.point_size = 2.0
        opt.background_color = np.asarray([1, 1, 1])
        vis.run()
        vis.destroy_window()
    else:
        print("  [Debug Warning] No geometry available for rendering.")


def estimate_trunk_base_radius(skeleton_points, tls_points, slice_height=0.2):
    if skeleton_points is None or len(skeleton_points) == 0:
        print("[Warning] Skeleton points are empty. Cannot calculate radius.")
        return 0.0
    if tls_points is None or len(tls_points) == 0:
        print("[Warning] TLS point cloud is empty. Cannot calculate radius.")
        return 0.0

    min_z_idx = np.argmin(skeleton_points[:, 2])
    root_point = skeleton_points[min_z_idx]
    root_z = root_point[2]
    root_xy = root_point[:2]

    mask = (tls_points[:, 2] >= root_z) & (tls_points[:, 2] <= (root_z + slice_height))
    points_in_slice = tls_points[mask]
    if len(points_in_slice) == 0:
        print(
            f"[Warning] No TLS points found within the bottom {slice_height} m "
            "range. Radius is set to 0."
        )
        return 0.0

    slice_xy = points_in_slice[:, :2]
    distances = np.linalg.norm(slice_xy - root_xy, axis=1)
    radius = float(np.mean(distances))
    return radius


def smooth_skeleton_globally(sk_coords, edges, parameters):
    if (not parameters.enable_global_smoothing) or edges is None or len(edges) == 0:
        return sk_coords

    n = len(sk_coords)
    adj = [[] for _ in range(n)]
    for u, v in edges:
        u, v = int(u), int(v)
        adj[u].append(v)
        adj[v].append(u)

    coords = sk_coords.copy()
    lam1 = float(parameters.laplacian_lambda)
    lam2 = float(parameters.twohop_lambda)
    iters = int(parameters.smoothing_iterations)

    for _ in range(iters):
        new_coords = coords.copy()

        if lam1 > 0:
            for i in range(n):
                nb = adj[i]
                if not nb:
                    continue
                avg = np.mean(coords[nb], axis=0)
                new_coords[i] = (1 - lam1) * new_coords[i] + lam1 * avg

        if lam2 > 0:
            coords_mid = new_coords.copy()

            for i in range(n):
                nb1 = adj[i]
                if not nb1:
                    continue

                nb2 = set()
                for j in nb1:
                    for k in adj[j]:
                        if k != i:
                            nb2.add(k)

                if not nb2:
                    continue

                avg2 = np.mean(coords_mid[list(nb2)], axis=0)
                new_coords[i] = (1 - lam2) * new_coords[i] + lam2 * avg2

        coords = new_coords

    return coords


def project_edges_onto_constraints(sk_coords, edges, parameters, current_angle_limit_deg):
    if (not parameters.enable_post_projection) or edges is None or len(edges) == 0:
        return sk_coords

    coords = sk_coords.copy()
    wL = float(parameters.length_projection_weight)
    wA = float(parameters.angle_projection_weight)
    proj_iters = int(parameters.projection_iterations)
    cos_limit = float(np.cos(np.deg2rad(current_angle_limit_deg)))

    for _ in range(proj_iters):
        for (u, v) in edges:
            u, v = int(u), int(v)

            p1 = coords[u].copy()
            p2 = coords[v].copy()
            mid = 0.5 * (p1 + p2)

            vec = p2 - p1
            L = float(np.linalg.norm(vec))
            if L < 1e-12:
                continue

            cur_dir = vec / L
            k = _canonical_edge(u, v)

            ref_dir = (
                parameters.edge_ref_dir.get(k, None)
                if parameters.use_fixed_reference_for_rotation
                else None
            )
            ref_len = (
                parameters.edge_ref_len.get(k, None)
                if parameters.use_fixed_reference_for_length
                else None
            )

            if ref_dir is not None and wA > 0:
                dotp = float(np.dot(cur_dir, ref_dir))

                if dotp < cos_limit:
                    uvec = cur_dir - dotp * ref_dir
                    nu = float(np.linalg.norm(uvec))

                    if nu > 1e-12:
                        uvec = uvec / nu
                        target_dir = (
                            cos_limit * ref_dir
                            + float(np.sin(np.deg2rad(current_angle_limit_deg)))
                            * uvec
                        )
                        target_dir = _unit_vector_or_none(target_dir)

                        if target_dir is not None:
                            use_len = (
                                ref_len
                                if (
                                    ref_len is not None
                                    and parameters.use_fixed_reference_for_length
                                )
                                else L
                            )
                            p1_t = mid - 0.5 * use_len * target_dir
                            p2_t = mid + 0.5 * use_len * target_dir

                            p1 = (1 - wA) * p1 + wA * p1_t
                            p2 = (1 - wA) * p2 + wA * p2_t

                            vec = p2 - p1
                            L = float(np.linalg.norm(vec))
                            if L > 1e-12:
                                cur_dir = vec / L

            if ref_len is not None and wL > 0:
                if L > 1e-12:
                    scale = ref_len / L
                    vec_scaled = vec * scale
                    p1_t = mid - 0.5 * vec_scaled
                    p2_t = mid + 0.5 * vec_scaled

                    p1 = (1 - wL) * p1 + wL * p1_t
                    p2 = (1 - wL) * p2 + wL * p2_t

            coords[u] = p1
            coords[v] = p2

    return coords


def enforce_minimum_edge_length(sk_coords, edges, parameters):
    if (
        not getattr(parameters, "enable_min_edge_length", False)
        or edges is None
        or len(edges) == 0
    ):
        return sk_coords

    coords = sk_coords.copy()
    w = float(getattr(parameters, "min_edge_projection_weight", 0.8))
    iters = int(getattr(parameters, "min_edge_projection_iterations", 1))
    factor = float(getattr(parameters, "min_edge_length_factor", 0.7))

    for _ in range(iters):
        for (u, v) in edges:
            u = int(u)
            v = int(v)

            k = _canonical_edge(u, v)
            ref_len = parameters.edge_ref_len.get(k, None)
            if ref_len is None:
                continue

            min_len = factor * float(ref_len)
            p1 = coords[u]
            p2 = coords[v]

            vec = p2 - p1
            L = float(np.linalg.norm(vec))

            if L < 1e-12:
                ref_dir = parameters.edge_ref_dir.get(k, None)
                if ref_dir is None:
                    continue

                mid = 0.5 * (p1 + p2)
                p1_t = mid - 0.5 * min_len * ref_dir
                p2_t = mid + 0.5 * min_len * ref_dir
            else:
                if L >= min_len:
                    continue

                mid = 0.5 * (p1 + p2)
                dirv = vec / L
                p1_t = mid - 0.5 * min_len * dirv
                p2_t = mid + 0.5 * min_len * dirv

            coords[u] = (1 - w) * coords[u] + w * p1_t
            coords[v] = (1 - w) * coords[v] + w * p2_t

    return coords


def optimize_skeleton_geometry(
    tls_points,
    skeleton_points,
    edges,
    parameters,
    *,
    major_iteration_number=1,
    intermediate_output_stem=None,
):
    """Refine skeleton coordinates for one major interwoven iteration."""
    if edges is None or len(edges) == 0 or tls_points is None or len(tls_points) == 0:
        return skeleton_points

    optimized_skeleton_points = np.copy(skeleton_points)
    tls_coords = tls_points[:, :3]

    output_stem = "" if intermediate_output_stem is None else str(intermediate_output_stem)
    can_save_inner_results = bool(
        parameters.save_intermediate_results and output_stem and major_iteration_number > 0
    )

    iterable_inner = tqdm(
        range(parameters.refinement_iterations),
        desc="Inner refinement (SoftAssign+Smooth+Project)",
        leave=False,
        ncols=120
    ) if TQDM_AVAILABLE else range(parameters.refinement_iterations)

    for i in iterable_inner:
        total_steps = parameters.max_major_iterations * parameters.refinement_iterations
        current_global_step = (major_iteration_number - 1) * parameters.refinement_iterations + i
        progress = min(1.0, current_global_step / max(1, total_steps - 1))

        current_angle_limit = parameters.rotation_decay_start - (
            parameters.rotation_decay_start - parameters.rotation_decay_end
        ) * progress
        parameters.max_rotation_angle = float(current_angle_limit)

        if TQDM_AVAILABLE:
            iterable_inner.set_postfix(angle=f"{current_angle_limit:.1f}°")

        sk_coords_current = optimized_skeleton_points[:, :3]
        edge_to_tls_items = defaultdict(list)

        sk_tree = KDTree(sk_coords_current)

        point_to_edge_map = [[] for _ in range(len(sk_coords_current))]
        for edge_idx, (u, v) in enumerate(edges):
            u, v = int(u), int(v)
            point_to_edge_map[u].append(edge_idx)
            point_to_edge_map[v].append(edge_idx)

        K_filter = 10
        k_val = min(K_filter, len(sk_coords_current))

        if k_val > 0:
            _, nearest_sk_indices_group = sk_tree.query(tls_coords, k=k_val)
            if nearest_sk_indices_group.ndim == 1:
                nearest_sk_indices_group = nearest_sk_indices_group.reshape(-1, 1)

            sigma = float(parameters.soft_assignment_sigma)
            temp = float(parameters.soft_assignment_temperature)
            topk = int(parameters.soft_assignment_topk)

            for tls_idx, tls_point in enumerate(tls_coords):
                candidate_edge_indices_set = set()
                for sk_idx in nearest_sk_indices_group[tls_idx]:
                    sk_idx = int(sk_idx)
                    if sk_idx < len(point_to_edge_map):
                        candidate_edge_indices_set.update(point_to_edge_map[sk_idx])

                if not candidate_edge_indices_set:
                    continue

                candidate_edge_indices = np.array(list(candidate_edge_indices_set), dtype=int)

                p = tls_point[np.newaxis, :]
                u_indices = edges[candidate_edge_indices][:, 0].astype(int)
                v_indices = edges[candidate_edge_indices][:, 1].astype(int)

                v_pts = sk_coords_current[u_indices]
                w_pts = sk_coords_current[v_indices]

                line_vec = w_pts - v_pts
                l2 = np.sum(line_vec ** 2, axis=1)
                l2[l2 < 1e-12] = 1.0

                t = np.sum((p - v_pts) * line_vec, axis=1) / l2
                valid_mask = (t >= 0.0) & (t <= 1.0)
                if not np.any(valid_mask):
                    continue

                t_valid = t[valid_mask]
                v_pts_valid = v_pts[valid_mask]
                line_vec_valid = line_vec[valid_mask]
                candidate_valid = candidate_edge_indices[valid_mask]

                proj = v_pts_valid + t_valid[:, None] * line_vec_valid
                distances = np.linalg.norm(p - proj, axis=1)

                if parameters.global_tree_radius > 0:
                    dist_threshold = (
                        parameters.global_tree_radius
                        * parameters.association_radius_factor
                    )
                    dist_mask = distances <= dist_threshold
                    if not np.any(dist_mask):
                        continue
                    distances = distances[dist_mask]
                    candidate_valid = candidate_valid[dist_mask]

                if len(candidate_valid) == 0:
                    continue

                if (not parameters.enable_soft_assignment) or parameters.force_hard_winner:
                    best = int(candidate_valid[np.argmin(distances)])
                    edge_to_tls_items[best].append(int(tls_idx))
                    continue

                k_take = min(topk, len(candidate_valid))
                order = np.argsort(distances)[:k_take]
                chosen_edges = candidate_valid[order]
                chosen_d = distances[order]

                if sigma <= 1e-12:
                    best = int(chosen_edges[0])
                    edge_to_tls_items[best].append(int(tls_idx))
                    continue

                base_w = np.exp(-(chosen_d ** 2) / (2.0 * sigma * sigma))
                if temp > 1e-12:
                    base_w = base_w ** (1.0 / temp)

                s = float(np.sum(base_w))
                if s < 1e-12:
                    best = int(chosen_edges[0])
                    edge_to_tls_items[best].append(int(tls_idx))
                    continue

                base_w = base_w / s

                for e_id, wgt in zip(chosen_edges, base_w):
                    edge_to_tls_items[int(e_id)].append((int(tls_idx), float(wgt)))

        if parameters.enable_edge_noise_filtering:
            filtered = defaultdict(list)
            for edge_idx, items in edge_to_tls_items.items():
                if len(items) < 3:
                    filtered[edge_idx] = items
                    continue

                u, v = edges[edge_idx]
                u, v = int(u), int(v)
                p1, p2 = sk_coords_current[u], sk_coords_current[v]

                if isinstance(items[0], (tuple, list)) and len(items[0]) == 2:
                    tls_indices = np.array([int(a) for a, _ in items], dtype=int)
                    tls_weights = np.array([float(w) for _, w in items], dtype=float)
                else:
                    tls_indices = np.array([int(a) for a in items], dtype=int)
                    tls_weights = np.ones(len(tls_indices), dtype=float)

                assigned_points = tls_coords[tls_indices]

                line_vec = p2 - p1
                l2 = float(np.dot(line_vec, line_vec))
                if l2 < 1e-12:
                    distances = np.linalg.norm(assigned_points - p1, axis=1)
                else:
                    t = np.dot(assigned_points - p1, line_vec) / l2
                    t = np.clip(t, 0, 1)
                    proj = p1 + t[:, None] * line_vec
                    distances = np.linalg.norm(assigned_points - proj, axis=1)

                mean_dist = float(np.mean(distances))
                std_dist = float(np.std(distances))
                thr = mean_dist + parameters.noise_filtering_n_std * std_dist

                keep = distances < thr
                if not np.any(keep):
                    continue

                if isinstance(items[0], (tuple, list)) and len(items[0]) == 2:
                    new_items = [
                        (int(tls_indices[i]), float(tls_weights[i]))
                        for i in range(len(keep))
                        if keep[i]
                    ]
                    filtered[edge_idx] = new_items
                else:
                    filtered[edge_idx] = [int(tls_indices[i]) for i in range(len(keep)) if keep[i]]

            edge_to_tls_items = filtered

        if parameters.debug_mode:
            step_name = (
                f"Iter {major_iteration_number}-{i + 1} "
                f"(Angle: {parameters.max_rotation_angle:.1f})"
            )
            visualize_optimization_state(
                tls_points,
                optimized_skeleton_points,
                edges,
                edge_to_tls_items,
                title=step_name
            )

        tasks = [
            delayed(_optimize_single_edge)(
                idx,
                edge,
                edge_to_tls_items.get(idx),
                tls_coords,
                sk_coords_current,
                parameters
            )
            for idx, edge in enumerate(edges)
        ]

        results = Parallel(n_jobs=parameters.n_jobs, backend="threading")(tasks)

        sum_of_positions = np.zeros_like(sk_coords_current)
        update_weights = np.zeros(len(sk_coords_current), dtype=float)
        step_rate = float(parameters.step_rate)

        for res in results:
            if res is None:
                continue
            u, v, u_target, v_target, edge_w = res
            u, v = int(u), int(v)
            edge_w = float(edge_w) if edge_w is not None else 1.0
            if edge_w < 1e-12:
                edge_w = 1.0

            u_orig, v_orig = sk_coords_current[u], sk_coords_current[v]
            u_new = u_orig + step_rate * (u_target - u_orig)
            v_new = v_orig + step_rate * (v_target - v_orig)

            sum_of_positions[u] += edge_w * u_new
            sum_of_positions[v] += edge_w * v_new
            update_weights[u] += edge_w
            update_weights[v] += edge_w

        updated_indices = np.where(update_weights > 0)[0]
        if len(updated_indices) > 0:
            avg_pos = sum_of_positions[updated_indices] / update_weights[updated_indices, None]
            optimized_skeleton_points[updated_indices, :3] = avg_pos

        optimized_skeleton_points[:, :3] = smooth_skeleton_globally(
            optimized_skeleton_points[:, :3], edges, parameters
        )

        if can_save_inner_results:
            out_smooth = (
                f"{output_stem}_major_{major_iteration_number}_inner_{i + 1}"
                "_global_smoothing.ply"
            )
            save_skeleton(out_smooth, optimized_skeleton_points, edges)

        optimized_skeleton_points[:, :3] = project_edges_onto_constraints(
            optimized_skeleton_points[:, :3],
            edges,
            parameters,
            current_angle_limit_deg=parameters.max_rotation_angle,
        )

        if can_save_inner_results:
            out_proj = (
                f"{output_stem}_major_{major_iteration_number}_inner_{i + 1}"
                "_project_edges_onto_constraints.ply"
            )
            save_skeleton(out_proj, optimized_skeleton_points, edges)

        optimized_skeleton_points[:, :3] = enforce_minimum_edge_length(
            optimized_skeleton_points[:, :3], edges, parameters
        )

        if can_save_inner_results:
            out_minedge = (
                f"{output_stem}_major_{major_iteration_number}_inner_{i + 1}"
                "_enforce_minimum_edge_length.ply"
            )
            save_skeleton(out_minedge, optimized_skeleton_points, edges)

        if can_save_inner_results:
            inner_loop_output_path = (
                f"{output_stem}_major_{major_iteration_number}"
                f"_inner_{i + 1}.ply"
            )
            save_skeleton(inner_loop_output_path, optimized_skeleton_points, edges)

    return optimized_skeleton_points


def _validate_xyz_points(points, *, path, data_name, minimum_points):
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"{data_name} must provide exactly three selected xyz columns: {path}")
    if len(points) < minimum_points:
        raise ValueError(
            f"{data_name} must contain at least {minimum_points} points; found {len(points)}: {path}"
        )
    if not np.all(np.isfinite(points)):
        raise ValueError(f"{data_name} contains NaN or infinite xyz coordinates: {path}")
    return points


def _load_xyz_text(filepath: str | Path, *, data_name: str, minimum_points: int) -> np.ndarray:
    """Read the first three numeric columns from a delimited text file.

    The first non-comment row may either be data or a column header.  This
    avoids the previous ``header=0`` behavior, which discarded the first point
    from headerless files.
    """
    path = Path(filepath)
    if not path.is_file():
        raise FileNotFoundError(f"{data_name} file does not exist: {path}")

    try:
        table = pd.read_csv(
            path,
            sep=r"[\s,;]+",
            comment="#",
            header=None,
            engine="python",
        )
    except Exception as exc:
        raise ValueError(f"Failed to parse {data_name} file '{path}': {exc}") from exc

    if table.shape[1] < 3:
        raise ValueError(
            f"{data_name} file must contain at least three columns (x, y, z): {path}"
        )

    xyz = table.iloc[:, :3].apply(pd.to_numeric, errors="coerce")
    valid_rows = xyz.notna().all(axis=1).to_numpy()
    invalid_rows = np.flatnonzero(~valid_rows)
    if len(invalid_rows) > 0:
        header = [str(value).strip().lower() for value in table.iloc[0, :3]]
        has_xyz_header = int(invalid_rows[0]) == 0 and header == ["x", "y", "z"]
        invalid_data_rows = invalid_rows[invalid_rows != 0] if has_xyz_header else invalid_rows
        if not has_xyz_header or len(invalid_data_rows) > 0:
            bad_row = int(invalid_data_rows[0]) + 1
            raise ValueError(
                f"{data_name} contains invalid xyz data at parsed row {bad_row}; "
                f"only one optional x/y/z header is allowed: {path}"
            )

    points = xyz.loc[valid_rows].to_numpy(dtype=np.float64)
    return _validate_xyz_points(
        points,
        path=path,
        data_name=data_name,
        minimum_points=minimum_points,
    )


def load_tls_point_cloud(filepath: str | Path) -> np.ndarray:
    """Load TLS xyz coordinates from TXT, CSV, XYZ, PLY, or PCD input."""
    path = Path(filepath)
    if path.suffix.lower() in {".ply", ".pcd"}:
        if not path.is_file():
            raise FileNotFoundError(f"TLS point-cloud file does not exist: {path}")
        point_cloud = o3d.io.read_point_cloud(str(path))
        points = np.asarray(point_cloud.points, dtype=np.float64)
        return _validate_xyz_points(
            points,
            path=path,
            data_name="TLS point cloud",
            minimum_points=3,
        )
    return _load_xyz_text(path, data_name="TLS point cloud", minimum_points=3)


def load_skeleton(filepath: str | Path) -> np.ndarray:
    """Load skeleton vertices from a TXT, CSV, or PLY file."""
    path = Path(filepath)
    suffix = path.suffix.lower()
    if suffix in {".txt", ".csv"}:
        return _load_xyz_text(path, data_name="Skeleton", minimum_points=2)
    if suffix != ".ply":
        raise ValueError(
            f"Unsupported skeleton format '{path.suffix}'. Use .txt, .csv, or .ply: {path}"
        )
    if not path.is_file():
        raise FileNotFoundError(f"Skeleton file does not exist: {path}")

    line_set = o3d.io.read_line_set(str(path))
    points = np.asarray(line_set.points, dtype=np.float64)
    if len(points) == 0:
        point_cloud = o3d.io.read_point_cloud(str(path))
        points = np.asarray(point_cloud.points, dtype=np.float64)
    return _validate_xyz_points(
        points,
        path=path,
        data_name="Skeleton",
        minimum_points=2,
    )


def save_skeleton(
    filepath: str | Path,
    points: np.ndarray,
    edges: np.ndarray,
) -> None:
    """Write skeleton vertices and edges as an ASCII PLY line set."""
    if points is None or len(points) == 0:
        raise ValueError("Cannot save an empty skeleton point array.")
    if edges is None:
        raise ValueError("Cannot save a skeleton without an edge array.")

    path = Path(filepath)
    if path.suffix.lower() != ".ply":
        raise ValueError(f"Skeleton output path must use the .ply extension: {path}")

    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(points[:, :3]),
        lines=o3d.utility.Vector2iVector(edges)
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    if not o3d.io.write_line_set(str(path), line_set, write_ascii=True):
        raise OSError(f"Failed to write optimized skeleton: {path}")


def reconstruct_skeleton_topology(parameters, skeleton_points):
    """Reconstruct skeleton edges with the original hybrid Delaunay/DMST method."""
    if skeleton_points is None or len(skeleton_points) < 2:
        return None

    points_3d = np.asarray(skeleton_points[:, :3], dtype=np.float64)

    def _ensure_connectivity(points, adj):
        components = []
        visited = set()
        for i in range(len(points)):
            if i not in visited:
                component = []
                q = deque([i])
                visited.add(i)
                while q:
                    u = q.popleft()
                    component.append(u)
                    for v_neighbor in adj.get(u, set()):
                        if v_neighbor not in visited:
                            visited.add(v_neighbor)
                            q.append(v_neighbor)
                components.append(component)

        if len(components) <= 1:
            return adj

        for i in range(len(components) - 1):
            comp1_indices, comp2_indices = components[i], components[i + 1]
            if not comp1_indices or not comp2_indices:
                continue
            tree_comp2 = KDTree(points[comp2_indices])
            dists, idx_in_comp2 = tree_comp2.query(points[comp1_indices], k=1)
            min_dist_idx_in_comp1 = np.argmin(dists)
            u = comp1_indices[min_dist_idx_in_comp1]
            v = comp2_indices[idx_in_comp2[min_dist_idx_in_comp1]]
            adj[u].add(v)
            adj[v].add(u)
        return adj

    def _build_hybrid_delaunay_graph(points, params):
        num_points = len(points)
        adj = {i: set() for i in range(num_points)}

        k_val = getattr(params, "topology_k", 16)
        k_val = min(k_val, num_points - 1)
        kdtree = KDTree(points)
        _, indices = kdtree.query(points, k=k_val + 1, workers=-1)

        knn_fallback_k = int(getattr(params, "topology_knn_fallback_k", k_val))
        knn_fallback_k = max(1, min(knn_fallback_k, k_val))

        knn_prune_n_std = float(getattr(params, "topology_knn_prune_n_std", 1.0))

        def _add_knn_fallback_edges_with_prune(i, nbrs):
            if len(nbrs) == 0:
                return
            nbrs = np.array(nbrs[:knn_fallback_k], dtype=int)

            vec = points[nbrs] - points[i]
            dists = np.linalg.norm(vec, axis=1)

            if len(dists) == 1:
                keep_mask = np.array([True])
            else:
                mu = float(np.mean(dists))
                sd = float(np.std(dists))
                thr = mu + knn_prune_n_std * sd
                keep_mask = dists <= thr

                if not np.any(keep_mask):
                    keep_mask[np.argmin(dists)] = True

            kept = nbrs[keep_mask]
            for nb in kept:
                nb = int(nb)
                adj[i].add(nb)
                adj[nb].add(i)

        for i in range(num_points):
            valid_indices = indices[i, 1:]
            if len(valid_indices) < 1:
                continue

            local_subset = np.hstack(([i], valid_indices))
            local_points = points[local_subset]

            try:
                tri = Delaunay(local_points)
                for simplex in tri.simplices:
                    if 0 in simplex:
                        for local_idx in simplex:
                            if local_idx != 0:
                                global_neighbor = int(local_subset[local_idx])
                                adj[i].add(global_neighbor)
                                adj[global_neighbor].add(i)
            except Exception:
                _add_knn_fallback_edges_with_prune(i, valid_indices)

        try:
            tri_global = Delaunay(points)
            all_g_edges = set()
            for s in tri_global.simplices:
                for a in range(4):
                    for b in range(a + 1, 4):
                        u, v = sorted((int(s[a]), int(s[b])))
                        all_g_edges.add((u, v))

            if all_g_edges:
                edge_list = np.array(list(all_g_edges), dtype=int)
                edge_vecs = points[edge_list[:, 0]] - points[edge_list[:, 1]]
                edge_lens = np.linalg.norm(edge_vecs, axis=1)
                prune_percentile = getattr(params, "topology_prune_percentile", 85)
                threshold = np.percentile(edge_lens, prune_percentile)
                valid_mask = edge_lens <= threshold
                valid_edges = edge_list[valid_mask]
                for u, v in valid_edges:
                    adj[int(u)].add(int(v))
                    adj[int(v)].add(int(u))
        except Exception as exc:
            print(
                f"  [Warning] Global Delaunay construction failed: {exc}. "
                "Using only local connections."
            )

        return adj

    def _build_weighted_tree(points, adj, params):
        num_points = len(points)
        start_node_idx = int(np.argmin(points[:, 2]))

        parent = np.full(num_points, -1, dtype=int)
        dist_to_root = np.full(num_points, np.inf)
        dist_to_root[start_node_idx] = 0.0

        in_tree = {start_node_idx}
        edge_heap = []

        for neighbor in adj.get(start_node_idx, []):
            neighbor = int(neighbor)
            mst_dist = float(np.linalg.norm(points[start_node_idx] - points[neighbor]))
            dijkstra_dist = float(dist_to_root[start_node_idx] + mst_dist)
            total_weight = float(
                params.topology_alpha * mst_dist
                + params.topology_beta * dijkstra_dist
            )
            heapq.heappush(edge_heap, (total_weight, neighbor, start_node_idx))

        while edge_heap and len(in_tree) < num_points:
            _, new_node, existing_node = heapq.heappop(edge_heap)
            if new_node in in_tree:
                continue
            in_tree.add(new_node)
            parent[new_node] = existing_node

            mst_dist = float(np.linalg.norm(points[new_node] - points[existing_node]))
            dist_to_root[new_node] = float(dist_to_root[existing_node] + mst_dist)

            for neighbor in adj.get(new_node, []):
                neighbor = int(neighbor)
                if neighbor not in in_tree:
                    mst_dist = float(np.linalg.norm(points[new_node] - points[neighbor]))
                    dijkstra_dist = float(dist_to_root[new_node] + mst_dist)
                    total_weight = float(
                        params.topology_alpha * mst_dist
                        + params.topology_beta * dijkstra_dist
                    )
                    heapq.heappush(edge_heap, (total_weight, neighbor, new_node))

        edges = [[int(p_idx), int(i)] for i, p_idx in enumerate(parent) if p_idx != -1]
        return np.array(edges, dtype=int) if edges else np.empty((0, 2), dtype=int)

    adj = _build_hybrid_delaunay_graph(points_3d, parameters)
    adj = _ensure_connectivity(points_3d, adj)
    final_edges = _build_weighted_tree(points_3d, adj, parameters)
    return final_edges


def run_interwoven_optimization(
    tls_path: str | Path,
    skeleton_path: str | Path,
    output_path: str | Path,
    config: InterwovenOptimizationConfig | None = None,
) -> InterwovenOptimizationResult:
    """Run interwoven optimization for one TLS/skeleton pair.

    Parameters
    ----------
    tls_path:
        Headered/headerless text, PLY, or PCD point cloud.
    skeleton_path:
        Initial skeleton vertices in TXT, CSV, or PLY format.
    output_path:
        Destination PLY file for the optimized line-set skeleton.
    config:
        Optional :class:`InterwovenOptimizationConfig`.  The default preserves
        the numerical settings of the original implementation.

    Returns
    -------
    InterwovenOptimizationResult
        Optimized points, reconstructed edges, and the final output path.

    Notes
    -----
    Input, topology, optimization, and output errors intentionally propagate to
    the caller.  This makes failed runs visible to scripts and batch pipelines.
    """
    # The algorithm updates radius, angle, and edge-reference state while it
    # runs.  Work on a private copy so callers can safely reuse their config.
    parameters = deepcopy(config) if config is not None else InterwovenOptimizationConfig()
    final_output_path = Path(output_path)
    parameters.validate()
    if final_output_path.suffix.lower() != ".ply":
        raise ValueError(
            f"Interwoven optimization output must be a .ply file: {final_output_path}"
        )
    resolved_output = final_output_path.expanduser().resolve()
    for input_label, input_path in (("TLS input", tls_path), ("skeleton input", skeleton_path)):
        if resolved_output == Path(input_path).expanduser().resolve():
            raise ValueError(f"Output path must not overwrite the {input_label}: {final_output_path}")

    # Edge references are specific to one input skeleton.  They remain fixed
    # across all major iterations in this run, as in the original algorithm.
    parameters.edge_ref_dir.clear()
    parameters.edge_ref_len.clear()

    tls_points = load_tls_point_cloud(tls_path)
    current_skeleton_points = load_skeleton(skeleton_path)

    tree_radius = estimate_trunk_base_radius(
        current_skeleton_points,
        tls_points,
        slice_height=0.2,
    )
    parameters.global_tree_radius = float(tree_radius)
    print(
        f"  [Info] Automatically computed tree radius: {tree_radius:.4f} m, "
        f"distance threshold: "
        f"{tree_radius * parameters.association_radius_factor:.4f} m"
    )

    current_edges = reconstruct_skeleton_topology(parameters, current_skeleton_points)
    if current_edges is None or len(current_edges) == 0:
        raise RuntimeError(
            f"Initial topology reconstruction produced no edges for '{skeleton_path}'."
        )

    initialize_edge_references(
        parameters,
        current_skeleton_points[:, :3],
        current_edges,
    )

    output_stem = str(final_output_path.with_suffix(""))
    output_extension = final_output_path.suffix

    for major_iteration_index in range(parameters.max_major_iterations):
        major_iteration_number = major_iteration_index + 1
        points_before_major_iteration = np.copy(current_skeleton_points)

        current_skeleton_points = optimize_skeleton_geometry(
            tls_points,
            current_skeleton_points,
            current_edges,
            parameters,
            major_iteration_number=major_iteration_number,
            intermediate_output_stem=output_stem,
        )

        should_update_topology = (
            parameters.enable_topology_update
            and major_iteration_number % parameters.topology_update_every_n_major == 0
        )
        if should_update_topology:
            new_edges = reconstruct_skeleton_topology(parameters, current_skeleton_points)
            if new_edges is None or len(new_edges) == 0:
                raise RuntimeError(
                    "Topology reconstruction produced no edges at major iteration "
                    f"{major_iteration_number}."
                )
            current_edges = new_edges
            initialize_edge_references(
                parameters,
                current_skeleton_points[:, :3],
                current_edges,
            )

        if parameters.save_intermediate_results:
            iteration_output_path = (
                f"{output_stem}_major_iter_{major_iteration_number}"
                f"{output_extension}"
            )
            save_skeleton(iteration_output_path, current_skeleton_points, current_edges)

        if len(current_skeleton_points) == len(points_before_major_iteration):
            displacement = np.linalg.norm(
                current_skeleton_points[:, :3]
                - points_before_major_iteration[:, :3],
                axis=1,
            )
            if float(np.mean(displacement)) < float(parameters.convergence_threshold):
                break

    save_skeleton(final_output_path, current_skeleton_points, current_edges)
    return InterwovenOptimizationResult(
        points=current_skeleton_points,
        edges=np.asarray(current_edges, dtype=int),
        output_path=final_output_path,
    )


def Interwoven_optimization(params, tls_path, sk_path, output_path):
    """Signature-compatible wrapper with isolated runtime state."""
    return run_interwoven_optimization(
        tls_path=tls_path,
        skeleton_path=sk_path,
        output_path=output_path,
        config=params,
    )


# Compatibility aliases for callers that imported the historical helper names.
edge_key = _canonical_edge
safe_unit = _unit_vector_or_none
init_edge_references = initialize_edge_references
visualize_debug_status = visualize_optimization_state
calculate_trunk_radius_at_base = estimate_trunk_base_radius
global_smoothing = smooth_skeleton_globally
project_edges_to_constraints = project_edges_onto_constraints
enforce_min_edge_length = enforce_minimum_edge_length


def Geometric_optimization(tlsPoints, skPoints, edges, parameters, **kwargs):
    return optimize_skeleton_geometry(
        tlsPoints,
        skPoints,
        edges,
        parameters,
        major_iteration_number=kwargs.get("major_iteration_num", 1),
        intermediate_output_stem=kwargs.get("base_output_path"),
    )


load_tls_points = load_tls_point_cloud
load_skeleton_points = load_skeleton
Topological_optimization_using_hrbrid_Delaunay_and_dmst = reconstruct_skeleton_topology
