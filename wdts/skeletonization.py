"""Water-droplet skeletonization stage used by WDTS.

The numerical routines in this module follow the original research code.  The
public entry point is :func:`run_skeletonization`; it owns file I/O and runtime
configuration so importing this module never starts a processing job.
"""

from dataclasses import dataclass
from copy import deepcopy
import numpy as np
from scipy.spatial import cKDTree, Delaunay
from scipy.optimize import minimize
from sklearn.decomposition import PCA
from tqdm import tqdm
import os
from scipy.spatial import KDTree
from scipy.sparse import coo_matrix
import heapq
from concurrent.futures import ThreadPoolExecutor
import warnings
import open3d as o3d
import itertools
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from sklearn.neighbors import KernelDensity
from collections import deque
import time
from pathlib import Path
from typing import Optional, Union


PathLike = Union[str, os.PathLike]
DROPLET_COLUMNS = ("x", "y", "z", "mass", "radius", "ldi", "pdi", "sdi")
SKELETON_COLUMNS = (
    "x", "y", "z", "path_length", "location", "branch_id", "ldi", "pdi", "sdi", "mass"
)


class SkeletonizationConfig:
    """Configuration for water-droplet contraction and topology recovery.

    Tree geometry is normally estimated by :func:`run_skeletonization`; the
    constructor arguments remain available for callers that use the array API.
    """

    def __init__(self, tree_root=None, base_radius=None, max_radius=None):
        base_radius = 0.0 if base_radius is None else float(base_radius)
        self.gamma = 0.2
        self.epsilon = base_radius * 0.05
        self.base_radius = base_radius
        self.max_iterations = 5
        self.min_neighbors = 3
        self.density_factor = 2.0
        self.tree_root = tree_root
        self.debug_ldi_filter = False
        self.tree_growth_start_iteration = 3
        self.max_radius = 0.0 if max_radius is None else float(max_radius)
        self.convergence_tolerance = 0.0001
        self.entropy_linearity_weight = 0.99
        self.entropy_planarity_weight = 0.01
        self.topology_save_path = None
        self.topology_neighbor_count = 20
        self.topology_alpha = 0.5
        self.topology_beta = 0.5
        self.w_mass = 0.1
        self.w_ldi = 0.1
        self.w_pdi = -0.1
        self.w_sdi = -0.1
        self.save_intermediate_results = False

    def set_tree_geometry(self, tree_root, base_radius, max_radius):
        """Update geometry-dependent values after estimating the tree base."""
        self.tree_root = np.asarray(tree_root, dtype=float)
        self.base_radius = float(base_radius)
        self.max_radius = float(max_radius)
        self.epsilon = self.base_radius * 0.05

    # Compatibility properties for code written against the debug release.
    # New code should use the descriptive names above.
    @property
    def r_base(self):
        return self.base_radius

    @r_base.setter
    def r_base(self, value):
        self.base_radius = float(value)
        self.epsilon = self.base_radius * 0.05

    @property
    def max_iteration(self):
        return self.max_iterations

    @max_iteration.setter
    def max_iteration(self, value):
        self.max_iterations = int(value)

    @property
    def tree_grow_thresh(self):
        return self.tree_growth_start_iteration

    @tree_grow_thresh.setter
    def tree_grow_thresh(self, value):
        self.tree_growth_start_iteration = int(value)

    @property
    def delta(self):
        return self.convergence_tolerance

    @delta.setter
    def delta(self, value):
        self.convergence_tolerance = float(value)

    @property
    def alpha(self):
        return self.entropy_linearity_weight

    @alpha.setter
    def alpha(self, value):
        self.entropy_linearity_weight = float(value)

    @property
    def beta(self):
        return self.entropy_planarity_weight

    @beta.setter
    def beta(self, value):
        self.entropy_planarity_weight = float(value)

    @property
    def topology_K(self):
        return self.topology_neighbor_count

    @topology_K.setter
    def topology_K(self, value):
        self.topology_neighbor_count = int(value)


@dataclass
class SkeletonizationResult:
    """Outputs produced by :func:`run_skeletonization`."""

    points: np.ndarray
    edges: np.ndarray
    input_path: Path
    output_directory: Path
    points_path: Path
    skeleton_path: Path


__all__ = [
    "DROPLET_COLUMNS",
    "SKELETON_COLUMNS",
    "SkeletonizationConfig",
    "SkeletonizationResult",
    "downsample_if_needed",
    "estimate_tree_base_geometry",
    "compute_shape_descriptors",
    "contract_water_droplets",
    "reconstruct_skeleton_topology",
    "load_point_cloud",
    "save_skeleton",
    "run_skeletonization",
]


def downsample_if_needed(point_cloud: np.ndarray, base_radius: float, threshold: int = 15000,
                         voxel_size_divisor: float = 40.0) -> np.ndarray:
    """Voxel-downsample dense inputs while leaving smaller trees unchanged."""
    if len(point_cloud) > threshold:
        print(f"Point count is {len(point_cloud)}, exceeding {threshold}. Starting voxel downsampling...")
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(point_cloud)
        voxel_size = base_radius / voxel_size_divisor
        print(f"Voxel size used: {voxel_size:.4f}")
        downsampled_pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
        processed_point_cloud = np.asarray(downsampled_pcd.points)
        print(f"Downsampling completed. New point count: {len(processed_point_cloud)}")
        return processed_point_cloud
    else:
        print(f"Point count is {len(point_cloud)}, not exceeding {threshold}. Skipping downsampling.")
        return point_cloud


def estimate_tree_base_geometry(points, gamma=0.9):
    """Estimate mean/max base radius and the root from the lowest tree slice."""
    print('Fitting the point cloud within 0.2 m above the minimum Z to estimate the root point and radius.')
    points = points[points[:, 2].argsort()]
    min_z = np.min(points[:, 2])
    z_threshold = min_z + 0.2
    bottom_points = points[points[:, 2] <= z_threshold, :]
    print(
        f"Total points: {points.shape[0]}, bottom points within Z range [{min_z:.2f}, {z_threshold:.2f}]: {bottom_points.shape[0]}")

    if bottom_points.shape[0] > 0:
        minX, maxX = np.quantile(bottom_points[:, 0], [0.01, 0.99])
        minY, maxY = np.quantile(bottom_points[:, 1], [0.01, 0.99])
        minZ, maxZ = np.quantile(bottom_points[:, 2], [0.01, 0.99])
        select_mask = (
                (bottom_points[:, 0] >= minX) & (bottom_points[:, 0] <= maxX) &
                (bottom_points[:, 1] >= minY) & (bottom_points[:, 1] <= maxY) &
                (bottom_points[:, 2] >= minZ) & (bottom_points[:, 2] <= maxZ)
        )
        bottom_points = bottom_points[select_mask]
        print(f"Remaining bottom points after outlier filtering: {bottom_points.shape[0]}")
    else:
        print("Warning: No points found within the selected Z range. Try relaxing the range.")
        return None, None, None

    if bottom_points.shape[0] < 3:
        print("Error: Fewer than 3 valid bottom points. Cannot determine the center point.")
        return None, None, None

    def create_normalized_objective(points_2d, initial_center, gamma_val):
        initial_distances = np.linalg.norm(points_2d - initial_center, axis=1)
        scale_sum = np.sum(initial_distances)
        scale_variance = np.sum((initial_distances - np.mean(initial_distances)) ** 2)
        if scale_sum == 0:
            scale_sum = 1
        if scale_variance == 0:
            scale_variance = 1

        def objective_func(center):
            distances = np.linalg.norm(points_2d - center, axis=1)
            d_mean = np.mean(distances)
            distance_sum = np.sum(distances)
            distance_variance_sum = np.sum((d_mean - distances) ** 2)
            normalized_sum = distance_sum / scale_sum
            normalized_variance = distance_variance_sum / scale_variance
            return (1 - gamma_val + 0.01) * normalized_sum + gamma_val * normalized_variance

        return objective_func

    initial_center = np.array([np.mean(bottom_points[:, 0]), np.mean(bottom_points[:, 1])])
    objective_for_optimizer = create_normalized_objective(bottom_points[:, :2], initial_center, gamma)
    result = minimize(objective_for_optimizer, initial_center, method='Nelder-Mead')

    opt_center = result.x
    min_z = np.min(bottom_points[:, 2])
    tree_root = [opt_center[0], opt_center[1], min_z]
    final_distances = np.linalg.norm(bottom_points[:, :2] - opt_center, axis=1)
    radius = np.mean(final_distances)
    max_radius = np.max(final_distances)

    if radius < 0.00001 or radius > 1000 or max_radius < 0.00001 or max_radius > 1000:
        return 0.0001, max_radius, tree_root

    return radius, max_radius, tree_root


def build_tree_growth_graph(point_cloud, parameters, tree_root, droplet_matrix):
    """Build the root-guided hybrid Delaunay graph used for tree growth."""
    def update_path_attributes(path, path_attributes):
        path_length = len(path)
        if path_length == 0:
            return
        path_indices = np.array(path)
        path_positions = np.linspace(1, path_length, path_length)
        np.add.at(path_attributes, (path_indices, 0), 1)
        selected_offsets = np.where(path_attributes[path_indices, 1] < path_length)[0]
        if selected_offsets.size > 0:
            selected_indices = path_indices[selected_offsets]
            path_attributes[selected_indices, 1] = path_length
            path_attributes[selected_indices, 2] = path_positions[selected_offsets]

    def extract_paths(predecessors, root_idx):
        paths = []
        for target in range(len(predecessors)):
            if target == root_idx:
                paths.append([root_idx])
                continue
            path = []
            current = target
            while current != -1:
                path.append(current)
                current = predecessors[current]
            paths.append(path)
        return paths

    def compute_mst(sparse_matrix, point_cloud, neighbor_indices, root_position, alpha=0, beta=1):
        num_points = len(point_cloud)
        kdtree = cKDTree(point_cloud)
        _, root_idx = kdtree.query(root_position, k=1)
        in_tree = {root_idx}
        parent = np.full(num_points, -1, dtype=int)
        dist_to_root = np.full(num_points, np.inf)
        dist_to_root[root_idx] = 0
        edge_heap = []
        if root_idx < len(neighbor_indices) and neighbor_indices[root_idx]:
            for neighbor in neighbor_indices[root_idx]:
                mst_dist = sparse_matrix[root_idx, neighbor]
                if mst_dist > 0:
                    dijkstra_dist = dist_to_root[root_idx] + mst_dist
                    total_weight = alpha * mst_dist + beta * dijkstra_dist
                    heapq.heappush(edge_heap, (total_weight, neighbor, root_idx))
        while len(in_tree) < num_points and edge_heap:
            total_weight, new_node, existing_node = heapq.heappop(edge_heap)
            if new_node in in_tree:
                continue
            mst_dist = sparse_matrix[new_node, existing_node]
            parent[new_node] = existing_node
            in_tree.add(new_node)
            dist_to_root[new_node] = dist_to_root[existing_node] + mst_dist
            if new_node < len(neighbor_indices) and neighbor_indices[new_node]:
                for neighbor in neighbor_indices[new_node]:
                    if neighbor in in_tree:
                        continue
                    mst_dist = sparse_matrix[new_node, neighbor]
                    if mst_dist > 0:
                        dijkstra_dist = dist_to_root[new_node] + mst_dist
                        total_weight = alpha * mst_dist + beta * dijkstra_dist
                        heapq.heappush(edge_heap, (total_weight, neighbor, new_node))
        return parent, root_idx

    def compute_sparse_distance_matrix(point_cloud, droplet_matrix, k, r=None):
        num_points = len(point_cloud)
        kdtree = cKDTree(point_cloud)
        distances, indices = kdtree.query(point_cloud, k=k, workers=-1)
        local_delaunay_edges = set()
        for i in tqdm(range(num_points), desc="  - Building local Delaunay edges", leave=False):
            valid_indices = indices[i, 1:][indices[i, 1:] < num_points]
            if len(valid_indices) < 3:
                continue
            local_points = np.vstack([point_cloud[i], point_cloud[valid_indices]])
            local_map = {0: i, **{j + 1: idx for j, idx in enumerate(valid_indices)}}
            try:
                tri = Delaunay(local_points)
                for simplex in tri.simplices:
                    if 0 in simplex:
                        for idx in simplex:
                            if idx != 0:
                                local_delaunay_edges.add(tuple(sorted((i, local_map[idx]))))
            except Exception:
                continue

        global_edges_pruned = set()
        try:
            tri = Delaunay(point_cloud)
            all_g_edges = {tuple(sorted((s[i], s[j]))) for s in tri.simplices for i in range(4) for j in
                           range(i + 1, 4)}
            edge_lengths = [np.linalg.norm(point_cloud[u] - point_cloud[v]) for u, v in all_g_edges]
            if edge_lengths:
                threshold_global = np.percentile(edge_lengths, 85)
                global_edges_pruned = {e for e, l in zip(all_g_edges, edge_lengths) if l <= threshold_global}
        except Exception:
            pass

        final_edges = local_delaunay_edges.union(global_edges_pruned)
        rows, cols, data = [], [], []
        for u, v in final_edges:
            dist = np.linalg.norm(point_cloud[u] - point_cloud[v]) / 2
            mean_LDI = (droplet_matrix[u, 5] + droplet_matrix[v, 5]) / 2
            mean_PDI = (droplet_matrix[u, 6] + droplet_matrix[v, 6]) / 2
            mean_SDI = (droplet_matrix[u, 7] + droplet_matrix[v, 7]) / 2
            mean_mass = (droplet_matrix[u, 3] + droplet_matrix[v, 3]) / 2
            w_pdi = 0.1
            w_ldi = 0.1
            w_sdi = 0.1
            w_mass = 0.1
            alpha = 0.5
            feature_score = (w_pdi * mean_PDI) + (w_sdi * mean_SDI) - (w_ldi * mean_LDI) - (w_mass * mean_mass)
            weight_modifier = np.exp(alpha * feature_score)
            weight = dist * weight_modifier
            weight = max(weight, 1e-6)
            rows.extend([u, v])
            cols.extend([v, u])
            data.extend([weight, weight])

        sparse_matrix = coo_matrix((data, (rows, cols)), shape=(num_points, num_points)).tocsr()
        neighbor_indices = [[] for _ in range(num_points)]
        for u, v in final_edges:
            neighbor_indices[u].append(v)
            neighbor_indices[v].append(u)
        return sparse_matrix, np.array(neighbor_indices, dtype=object)

    kdtree = cKDTree(point_cloud)
    dist, idx = kdtree.query(tree_root, k=1)
    if dist > 1e-6:
        point_cloud_with_root = np.vstack([point_cloud, tree_root])
        droplet_state_with_root = np.vstack(
            [droplet_matrix, np.array([tree_root[0], tree_root[1], tree_root[2], 1.0, parameters.base_radius, 1.0, 0.0, 0.0])])
        added_root = True
    else:
        point_cloud_with_root = point_cloud
        droplet_state_with_root = droplet_matrix
        added_root = False

    graph_matrix, neighbor_indices = compute_sparse_distance_matrix(
        point_cloud_with_root, droplet_state_with_root, k=50
    )
    predecessors, root_idx = compute_mst(
        graph_matrix, point_cloud_with_root, neighbor_indices, tree_root
    )
    paths = extract_paths(predecessors, root_idx)
    if added_root:
        # The synthetic root was appended as the final row and is already
        # marked as considered by the growth stage, so it needs no path entry.
        paths = paths[:-1]

    path_attributes = np.zeros((len(point_cloud_with_root), 3))
    with ThreadPoolExecutor() as executor:
        valid_paths = [p for p in paths if p]
        list(tqdm(executor.map(lambda path: update_path_attributes(path, path_attributes), valid_paths),
                  total=len(valid_paths), desc='Computing node attributes along paths'))

    ldi_pdi_sdi_mass = droplet_state_with_root[:, [5, 6, 7, 3]]
    points_with_attributes = np.hstack(
        [point_cloud_with_root, path_attributes, ldi_pdi_sdi_mass]
    )
    return paths, points_with_attributes, root_idx, added_root


def select_skeleton_points(points_with_attributes, parameters, paths, _iteration, root_index):
    """Select ordered skeleton candidates from the root-guided growth paths."""
    neighbor_tree = KDTree(points_with_attributes[:, :3])
    neighbor_indices_by_point = [None] * points_with_attributes.shape[0]
    for point_index in tqdm(range(points_with_attributes.shape[0]), desc='Building neighbor matrix'):
        point_coordinates = points_with_attributes[point_index, :3]
        path_length = (
            points_with_attributes[point_index, 4]
            if points_with_attributes[point_index, 4] != 0
            else 1e-6
        )
        neighbor_radius = max(
            np.sqrt(points_with_attributes[point_index, 5] / path_length)
            * parameters.max_radius
            * 2,
            parameters.max_radius * 0.1 * 2,
        )
        neighbor_indices = neighbor_tree.query_ball_point(point_coordinates, neighbor_radius)
        neighbor_indices_by_point[point_index] = neighbor_indices

    skeleton_point_indices = []
    considered_points = np.zeros(points_with_attributes.shape[0], dtype=bool)
    considered_points[root_index] = True
    branches = []

    while not np.all(considered_points):
        unconsidered_indices = np.where(~considered_points)[0]
        path_locations = points_with_attributes[unconsidered_indices, 5]
        path_lengths = points_with_attributes[unconsidered_indices, 4]
        path_lengths = np.where(path_lengths == 0, 1e-6, path_lengths)
        selected_offset = np.argmin(np.sqrt(path_locations / path_lengths))
        current_point_index = unconsidered_indices[selected_offset]

        path_point_indices = paths[current_point_index]
        if not path_point_indices:
            considered_points[current_point_index] = True
        else:
            branch = [
                point_index
                for point_index in path_point_indices
                if point_index not in skeleton_point_indices
            ]
            if branch:
                branches.append(branch)
                skeleton_point_indices.extend(branch)
                for added_point_index in branch:
                    considered_points[neighbor_indices_by_point[added_point_index]] = True

    ordered_skeleton_indices = []
    branch_ids = []
    for branch_idx, branch in enumerate(branches):
        ordered_skeleton_indices.extend(branch)
        branch_ids.extend([branch_idx] * len(branch))

    unique_indices = list(dict.fromkeys(ordered_skeleton_indices))
    skeleton_points_without_branch_ids = points_with_attributes[
        unique_indices, :
    ][:, [0, 1, 2, 4, 5, 6, 7, 8, 9]]
    branch_ids = np.array([
        branch_ids[ordered_skeleton_indices.index(point_index)]
        for point_index in unique_indices
    ])
    skeleton_points = np.column_stack(
        (
            skeleton_points_without_branch_ids[:, :5],
            branch_ids,
            skeleton_points_without_branch_ids[:, 5:],
        )
    )

    skeleton_tree = KDTree(skeleton_points[:, :3])
    nearby_skeleton_indices = [
        skeleton_tree.query_ball_point(point, parameters.base_radius * 0.1)
        for point in skeleton_points[:, :3]
    ]
    duplicate_indices = []
    for point_index in range(len(skeleton_points)):
        if point_index in duplicate_indices:
            continue
        duplicate_indices.extend(
            list(set(nearby_skeleton_indices[point_index]) - {point_index})
        )

    duplicate_indices = sorted(set(duplicate_indices))
    skeleton_points = np.delete(skeleton_points, duplicate_indices, axis=0)
    if parameters.debug_ldi_filter:
        ldi_mask = skeleton_points[:, 6] >= 0.9
        skeleton_points = skeleton_points[ldi_mask]
        print(
            "Points with LDI < 0.9 have been removed. Remaining points:",
            skeleton_points.shape[0],
        )
    return skeleton_points


def compute_entropy_metric(ldi, pdi, sdi, alpha, beta, gamma_sdi=0.3, C_constant=1.0,
                           clip_eps=1e-6, subsample_size=7500):
    """Compute the KDE-transformed entropy metric for droplet convergence."""
    if not isinstance(C_constant, (int, float)) or C_constant <= 0:
        raise ValueError("C_constant must be a positive number.")

    ldi_arr = np.clip(np.asarray(ldi), clip_eps, 1.0 - clip_eps)
    pdi_arr = np.clip(np.asarray(pdi), clip_eps, 1.0 - clip_eps)
    sdi_arr = np.clip(np.asarray(sdi), clip_eps, 1.0 - clip_eps)
    num_points = len(ldi_arr)

    if num_points == 0:
        return 1.0

    if subsample_size is not None and num_points > subsample_size:
        indices = np.random.choice(num_points, subsample_size, replace=False)
        ldi_sample = ldi_arr[indices]
        pdi_sample = pdi_arr[indices]
        sdi_sample = sdi_arr[indices]
    else:
        ldi_sample = ldi_arr
        pdi_sample = pdi_arr
        sdi_sample = sdi_arr

    def _calculate_kde_densities(data_sample):
        n_pts = len(data_sample)
        if n_pts == 0:
            return None, "Empty data sample"
        data_skl = data_sample[:, np.newaxis]
        std_dev = np.std(data_sample)
        if n_pts > 1 and std_dev > 1e-9:
            bandwidth = 1.06 * std_dev * (n_pts ** (-0.2))
            bandwidth = max(bandwidth, 1e-7)
        else:
            bandwidth = 0.1 if n_pts == 1 else 1e-5
        try:
            kde = KernelDensity(kernel='gaussian', bandwidth=bandwidth)
            kde.fit(data_skl)
            log_p_values = kde.score_samples(data_skl)
            return np.exp(log_p_values), None
        except Exception as e:
            return None, f"KDE computation failed: {e}"

    p_ldi_values, _ = _calculate_kde_densities(ldi_sample)
    p_pdi_values, _ = _calculate_kde_densities(pdi_sample)
    p_sdi_values, _ = _calculate_kde_densities(sdi_sample)

    if p_ldi_values is None or p_pdi_values is None or p_sdi_values is None:
        return 1.0

    p_ldi_values = np.clip(p_ldi_values, clip_eps, np.inf)
    p_pdi_values = np.clip(p_pdi_values, clip_eps, np.inf)
    p_sdi_values = np.clip(p_sdi_values, clip_eps, np.inf)
    s_terms_ldi = alpha * p_ldi_values * np.log(p_ldi_values)
    s_terms_pdi = beta * p_pdi_values * np.log(p_pdi_values)
    s_terms_sdi = gamma_sdi * p_sdi_values * np.log(p_sdi_values)
    S_avg = np.mean(s_terms_ldi + s_terms_pdi + s_terms_sdi)
    denominator = C_constant + S_avg

    if denominator <= 1e-9:
        return 1.0

    new_metric = C_constant / denominator
    return new_metric

def has_converged(prev_entropy, curr_entropy, iteration, delta, max_iterations, tree_grow_thresh):
    """Return whether the entropy history satisfies the stopping criterion."""
    if iteration <= tree_grow_thresh:
        return False
    if iteration >= max_iterations:
        return True
    if prev_entropy is not None:
        if prev_entropy != 0:
            relative_change = abs(curr_entropy - prev_entropy) / abs(prev_entropy)
            if relative_change < delta:
                return True
    return False


def save_entropy_plot(entropy_history, iterations, savepath, name):
    """Save an entropy-versus-iteration diagnostic plot."""
    times_new_roman = fm.FontProperties(family='Times New Roman')
    plt.figure(figsize=(8, 6))
    plt.plot(iterations, entropy_history, marker='o', linestyle='-', color='#1f77b4', linewidth=2, markersize=8)
    plt.xlabel('Iteration', fontsize=14, fontproperties=times_new_roman)
    plt.ylabel('Entropy', fontsize=14, fontproperties=times_new_roman)
    plt.title('Entropy Change Over Iterations', fontsize=16, fontproperties=times_new_roman)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(fontsize=12, fontproperties=times_new_roman)
    plt.yticks(fontsize=12, fontproperties=times_new_roman)
    plot_filename = os.path.join(savepath, name, f'entropy_history_{name}.png')
    os.makedirs(os.path.dirname(plot_filename), exist_ok=True)
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    plt.close()


def initialize_droplets(point_cloud, parameters):
    """Initialize droplet mass, radius and local shape descriptors."""
    kdt = cKDTree(point_cloud)
    num_points = len(point_cloud)
    k_neighbors = min(3, num_points)
    if k_neighbors > 1:
        distances, _ = kdt.query(point_cloud, k=k_neighbors)
        mean_distances = np.mean(distances[:, 1:], axis=1)
        r_search = np.sqrt(mean_distances)
    else:
        r_search = np.zeros(num_points)

    masses = np.zeros(len(point_cloud))
    radii = np.zeros(len(point_cloud))
    neighbor_counts = np.zeros(len(point_cloud))
    ldi = np.zeros(len(point_cloud))
    pdi = np.zeros(len(point_cloud))
    sdi = np.zeros(len(point_cloud))

    for i in range(len(point_cloud)):
        neighbors = kdt.query_ball_point(point_cloud[i], r_search[i])
        neighbors = [n for n in neighbors if n != i]
        if not neighbors:
            radii[i] = r_search[i]
            masses[i] = 1.0
            neighbor_counts[i] = 0
            ldi[i] = 1.0
            pdi[i] = 0
            sdi[i] = 0
            continue
        neighbor_points = point_cloud[neighbors]
        neighbor_counts[i] = len(neighbors)
        h_i = np.sqrt(np.sum((neighbor_points.max(axis=0) - neighbor_points.min(axis=0)) ** 2))
        radii[i] = h_i / 2
        ldi[i], pdi[i], sdi[i] = compute_shape_descriptors(neighbor_points)

    min_neighbors = np.min(neighbor_counts)
    max_neighbors = np.max(neighbor_counts)
    if max_neighbors == min_neighbors:
        masses[:] = 1 + parameters.density_factor / 2
    else:
        normalized_counts = (neighbor_counts - min_neighbors) / (max_neighbors - min_neighbors)
        masses = 1 + parameters.density_factor * normalized_counts

    return masses, radii, ldi, pdi, sdi



def merge_droplets(droplet_state, k=0.0001, new_search_radius_method='average', kdt=None):
    """Merge overlapping droplets while preserving their combined mass."""
    if droplet_state is None or droplet_state.shape[0] < 2:
        return droplet_state
    if droplet_state.shape[1] <= 3:
        warnings.warn("Droplet state is missing mass data; returning it unchanged.")
        return droplet_state

    masses = np.maximum(droplet_state[:, 3], 0)
    points = droplet_state[:, :3]
    if kdt is None:
        try:
            points_for_kdtree = points.astype(float)
            kdt = cKDTree(points_for_kdtree)
        except Exception:
            return droplet_state

    max_mass = np.max(masses)
    if max_mass <= 0:
        return droplet_state

    proxy_radius = np.power(masses, 1 / 3)
    max_proxy_radius = np.max(proxy_radius)
    search_epsilon = 2.0 * k * max_proxy_radius * 1.1
    if search_epsilon <= 1e-9:
        search_epsilon = 1e-9

    try:
        potential_pairs = kdt.query_pairs(r=search_epsilon, output_type='set')
    except Exception:
        return droplet_state

    merged_away_indices = set()
    merge_map = {i: i for i in range(len(droplet_state))}
    merged_state = droplet_state.copy()
    current_proxy_radius = proxy_radius.copy()
    current_mass = masses.copy()
    sorted_pairs = sorted(list(potential_pairs))

    for i, j in sorted_pairs:
        target_i = merge_map[i]
        target_j = merge_map[j]
        if target_i == target_j or target_i in merged_away_indices or target_j in merged_away_indices:
            continue
        if target_i > target_j:
            target_i, target_j = target_j, target_i

        pos_i, pos_j = merged_state[target_i, :3], merged_state[target_j, :3]
        m_i, m_j = current_mass[target_i], current_mass[target_j]
        pr_i, pr_j = current_proxy_radius[target_i], current_proxy_radius[target_j]
        dist_sq = np.sum((pos_i - pos_j) ** 2)
        fuse_thresh_dist = k * (pr_i + pr_j)
        fuse_thresh_dist_sq = fuse_thresh_dist ** 2

        if dist_sq < fuse_thresh_dist_sq:
            total_mass = m_i + m_j
            if total_mass > 1e-12:
                merged_state[target_i, :3] = (m_i * pos_i + m_j * pos_j) / total_mass
            elif m_i > 1e-12:
                merged_state[target_i, :3] = pos_i
            elif m_j > 1e-12:
                merged_state[target_i, :3] = pos_j
            else:
                merged_state[target_i, :3] = (pos_i + pos_j) / 2.0

            merged_state[target_i, 3] = total_mass
            current_mass[target_i] = total_mass
            current_proxy_radius[target_i] = np.power(max(0, total_mass), 1 / 3)
            s_i_orig = merged_state[target_i, 4]
            s_j_orig = merged_state[target_j, 4]
            if new_search_radius_method == 'max':
                merged_state[target_i, 4] = max(s_i_orig, s_j_orig)
            elif new_search_radius_method == 'min':
                merged_state[target_i, 4] = min(s_i_orig, s_j_orig)
            elif new_search_radius_method == 'average':
                merged_state[target_i, 4] = (s_i_orig + s_j_orig) / 2.0
            elif new_search_radius_method == 'weighted_average':
                if total_mass > 1e-12:
                    merged_state[target_i, 4] = (m_i * s_i_orig + m_j * s_j_orig) / total_mass
                else:
                    merged_state[target_i, 4] = max(s_i_orig, s_j_orig)
            else:
                merged_state[target_i, 4] = max(s_i_orig, s_j_orig)

            has_ldi_pdi = merged_state.shape[1] >= 8
            if has_ldi_pdi:
                ldi_i, pdi_i, sdi_i = merged_state[target_i, 5], merged_state[target_i, 6], merged_state[target_i, 7]
                ldi_j, pdi_j, sdi_j = merged_state[target_j, 5], merged_state[target_j, 6], merged_state[target_j, 7]
                merged_state[target_i, 5] = (ldi_i + ldi_j) / 2.0
                merged_state[target_i, 6] = (pdi_i + pdi_j) / 2.0
                merged_state[target_i, 7] = (sdi_i + sdi_j) / 2.0

            merged_away_indices.add(target_j)
            keys_to_update = [k_map for k_map, v_map in merge_map.items() if v_map == target_j]
            for k_map in keys_to_update:
                merge_map[k_map] = target_i
            merge_map[j] = target_i
            current_mass[target_j] = 0
            current_proxy_radius[target_j] = 0

    if merged_away_indices:
        keep_mask = np.ones(len(merged_state), dtype=bool)
        keep_mask[list(merged_away_indices)] = False
        final_state = merged_state[keep_mask]
    else:
        final_state = merged_state
    return final_state


def update_droplet_masses(droplet_state, parameters):
    """Update droplet masses from local density and linearity."""
    num_particles = len(droplet_state)
    if num_particles == 0:
        return droplet_state
    positions = droplet_state[:, :3]
    radii = droplet_state[:, 4]
    kdtree = KDTree(positions)
    neighbors_indices_with_self = kdtree.query_ball_point(positions, radii, return_sorted=True)
    vlen = np.frompyfunc(len, 1, 1)
    neighbor_counts = vlen(neighbors_indices_with_self).astype(int) - 1
    ldi = droplet_state[:, 5]
    mask_isolated = neighbor_counts < 3
    mask_normal = ~mask_isolated
    droplet_state[mask_isolated, 3] = 1.0
    if np.any(mask_normal):
        normal_counts = neighbor_counts[mask_normal]
        min_neighbors = np.min(normal_counts)
        max_neighbors = np.max(normal_counts)
        normal_ldi = ldi[mask_normal]
        if max_neighbors == min_neighbors:
            new_mass = 1 + parameters.density_factor / 2 * normal_ldi
        else:
            normalized_counts = (normal_counts - min_neighbors) / (max_neighbors - min_neighbors)
            new_mass = 1 + parameters.density_factor * normalized_counts * normal_ldi
        droplet_state[mask_normal, 3] = new_mass
    return droplet_state

def create_contraction_objective(neighbor_points, masses, gamma, delta=1.5):
    """Create the mass-weighted robust surface contraction objective."""
    def _huber_loss(error_vectors, delta_val):
        abs_error = np.abs(error_vectors)
        return np.where(abs_error <= delta_val, 0.5 * error_vectors ** 2,
                        delta_val * abs_error - 0.5 * delta_val ** 2)

    def compute_mad(data):
        flat_data = np.asarray(data).flatten()
        median = np.median(flat_data)
        abs_dev = np.abs(flat_data - median)
        mad = np.median(abs_dev)
        return mad

    if neighbor_points.shape[0] == 0:
        return lambda p_i: 0, np.zeros(neighbor_points.shape[1])

    mass_sum = np.sum(masses)
    if mass_sum == 0:
        mass_sum = 1e-10
    initial_p_i = np.sum(neighbor_points * masses[:, np.newaxis], axis=0) / mass_sum
    initial_errors = neighbor_points - initial_p_i
    mad = compute_mad(initial_errors)
    delta = 1.345 * mad
    if delta == 0:
        delta = 1e-4

    initial_huber = _huber_loss(initial_errors, delta)
    masses_reshaped = np.maximum(masses[:, np.newaxis], 1e-10)
    scale_huber = np.sum(initial_huber / masses_reshaped)
    initial_distances_l2 = np.linalg.norm(neighbor_points - initial_p_i, axis=1)
    weighted_distances = initial_distances_l2 * masses
    mean_weighted_distance = np.sum(weighted_distances) / mass_sum
    scale_variance = np.sum(masses * (initial_distances_l2 - mean_weighted_distance) ** 2)

    if scale_huber == 0:
        scale_huber = 1
    if scale_variance == 0:
        scale_variance = 1

    def objective_function(p_i):
        errors = neighbor_points - p_i
        huber_loss = _huber_loss(errors, delta)
        weighted_huber_sum = np.sum(huber_loss / masses_reshaped)
        l2_distances = np.linalg.norm(neighbor_points - p_i, axis=1)
        weighted_l2_distances = l2_distances * masses
        d_mean = np.sum(weighted_l2_distances) / mass_sum
        distance_variance_sum = np.sum(masses * (l2_distances - d_mean) ** 2)
        normalized_huber = weighted_huber_sum / scale_huber
        normalized_variance = distance_variance_sum / scale_variance
        weight_sum = 1 - gamma
        return weight_sum * normalized_huber + gamma * normalized_variance

    return objective_function, initial_p_i


def evaporate_droplets(
    droplet_state,
    skeleton_points,
    skeleton_edges,
    parameters,
    savepath,
    tree_name,
    iteration,
):
    """Remove droplets outside the adaptive radius of the current skeleton."""
    def point_to_segment_distance(point, segment_start, segment_end):
        if np.all(segment_start == segment_end):
            return np.linalg.norm(point - segment_start)
        segment_vector = segment_end - segment_start
        point_vector = point - segment_start
        t = np.dot(point_vector, segment_vector) / np.dot(segment_vector, segment_vector)
        if t < 0.0:
            closest_point = segment_start
        elif t > 1.0:
            closest_point = segment_end
        else:
            closest_point = segment_start + t * segment_vector
        return np.linalg.norm(point - closest_point)

    if not droplet_state.size or not skeleton_points.size or not skeleton_edges.size:
        return droplet_state, None
    if droplet_state.shape[1] < 8 or skeleton_points.shape[1] < 10:
        return droplet_state, None

    num_droplets = droplet_state.shape[0]
    num_skeleton_points = skeleton_points.shape[0]
    num_skeleton_edges = skeleton_edges.shape[0]
    droplet_tree = cKDTree(droplet_state[:, :3])
    base_radius = getattr(parameters, 'base_radius', 0.1)
    min_neighbors = getattr(parameters, 'min_neighbors', 5)
    neighbor_counts = np.array([
        len(droplet_tree.query_ball_point(point, base_radius))
        for point in droplet_state[:, :3]
    ])
    density_mask = neighbor_counts >= min_neighbors
    droplet_state = droplet_state[density_mask]
    num_droplets = droplet_state.shape[0]
    if num_droplets == 0:
        return droplet_state, None

    skeleton_tree = cKDTree(skeleton_points[:, :3])
    adjacency = {i: [] for i in range(num_skeleton_points)}
    for u, v in skeleton_edges:
        adjacency[u].append(v)
        adjacency[v].append(u)

    distances = np.full(num_droplets, np.inf)
    nearest_edge_indices = np.full(num_droplets, -1, dtype=int)
    k_search = getattr(parameters, 'topology_K', 10)
    _, nearest_skeleton_point_indices = skeleton_tree.query(
        droplet_state[:, :3], k=min(k_search, num_skeleton_points)
    )
    if nearest_skeleton_point_indices.ndim == 1:
        nearest_skeleton_point_indices = nearest_skeleton_point_indices[:, np.newaxis]
    edge_map = {tuple(sorted(edge)): i for i, edge in enumerate(skeleton_edges)}

    for droplet_index in range(num_droplets):
        droplet_point = droplet_state[droplet_index, :3]
        minimum_distance_for_point = np.inf
        best_edge_index = -1
        candidate_edges = set()
        for skeleton_point_index in nearest_skeleton_point_indices[droplet_index]:
            for neighbor_idx in adjacency[skeleton_point_index]:
                edge_tuple = tuple(sorted((skeleton_point_index, neighbor_idx)))
                candidate_edges.add(edge_tuple)
        if not candidate_edges:
            continue
        for u, v in candidate_edges:
            dist = point_to_segment_distance(
                droplet_point, skeleton_points[u, :3], skeleton_points[v, :3]
            )
            if dist < minimum_distance_for_point:
                minimum_distance_for_point = dist
                best_edge_index = edge_map[tuple(sorted((u, v)))]
        distances[droplet_index] = minimum_distance_for_point
        nearest_edge_indices[droplet_index] = best_edge_index

    valid_mask = nearest_edge_indices != -1
    droplet_state = droplet_state[valid_mask]
    distances = distances[valid_mask]
    nearest_edge_indices = nearest_edge_indices[valid_mask]

    edge_radii = np.zeros(num_skeleton_edges)
    edge_radii_std = np.zeros(num_skeleton_edges)
    edge_to_distances = [[] for _ in range(num_skeleton_edges)]
    for droplet_index, edge_index in enumerate(nearest_edge_indices):
        edge_to_distances[edge_index].append(distances[droplet_index])
    for edge_index in range(num_skeleton_edges):
        dists = edge_to_distances[edge_index]
        if dists:
            edge_radii[edge_index] = np.mean(dists)
            edge_radii_std[edge_index] = np.std(dists)

    skeleton_ldi = skeleton_points[:, 6]
    skeleton_pdi = skeleton_points[:, 7]
    skeleton_sdi = skeleton_points[:, 8]
    skeleton_mass = skeleton_points[:, 9]
    edge_endpoints = skeleton_edges[nearest_edge_indices]
    ref_mass = (skeleton_mass[edge_endpoints[:, 0]] + skeleton_mass[edge_endpoints[:, 1]]) / 2
    ref_ldi = (skeleton_ldi[edge_endpoints[:, 0]] + skeleton_ldi[edge_endpoints[:, 1]]) / 2
    ref_pdi = (skeleton_pdi[edge_endpoints[:, 0]] + skeleton_pdi[edge_endpoints[:, 1]]) / 2
    ref_sdi = (skeleton_sdi[edge_endpoints[:, 0]] + skeleton_sdi[edge_endpoints[:, 1]]) / 2

    droplet_mass = droplet_state[:, 3]
    droplet_ldi = droplet_state[:, 5]
    droplet_pdi = droplet_state[:, 6]
    droplet_sdi = droplet_state[:, 7]

    def norm(x):
        min_val, max_val = np.min(x), np.max(x)
        range_val = max_val - min_val
        return (x - min_val) / (range_val + 1e-6)

    mass_norm = norm(ref_mass)
    ldi_norm = norm(ref_ldi)
    pdi_norm = norm(ref_pdi)
    sdi_norm = norm(ref_sdi)
    droplet_mass_norm = norm(droplet_mass)
    droplet_ldi_norm = norm(droplet_ldi)
    droplet_pdi_norm = norm(droplet_pdi)
    droplet_sdi_norm = norm(droplet_sdi)

    w_mass = getattr(parameters, 'w_mass', 0.1)
    w_ldi = getattr(parameters, 'w_ldi', 0.1)
    w_pdi = getattr(parameters, 'w_pdi', -0.1)
    w_sdi = getattr(parameters, 'w_sdi', -0.1)
    factor = 1.0 + w_mass * (mass_norm + droplet_mass_norm) + w_ldi * (ldi_norm + droplet_ldi_norm) + w_pdi * (
                pdi_norm + droplet_pdi_norm) + w_sdi * (sdi_norm + droplet_sdi_norm)
    base_thresh = edge_radii[nearest_edge_indices] + edge_radii_std[nearest_edge_indices]
    thresh = base_thresh * factor
    evap_mask = distances <= thresh
    retained_droplets = droplet_state[evap_mask]

    if parameters.save_intermediate_results:
        output_dir = os.path.join(savepath, tree_name)
        os.makedirs(output_dir, exist_ok=True)
        pcd_before = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(droplet_state[:, :3]))
        o3d.io.write_point_cloud(os.path.join(output_dir, f"{tree_name}_iter_{iteration}_before_evaporation.ply"), pcd_before)
        pcd_after = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(retained_droplets[:, :3]))
        o3d.io.write_point_cloud(os.path.join(output_dir, f"{tree_name}_iter_{iteration}_after_evaporation.ply"), pcd_after)
        evap_filename = os.path.join(output_dir, f"{tree_name}_iter_{iteration}_droplets.txt")
        header = "X Y Z mass radius LDI PDI SDI"
        np.savetxt(evap_filename, retained_droplets, fmt='%.6f', header=header, delimiter=' ', comments='')
    return retained_droplets, edge_radii


def compute_shape_descriptors(neighbor_points):
    """Return the linearity, planarity and scattering descriptors (LDI/PDI/SDI)."""
    if len(neighbor_points) < 3:
        return 0.0, 0.0, 1.0
    neighbor_points_arr = np.asanyarray(neighbor_points)
    if np.all(np.allclose(neighbor_points_arr, neighbor_points_arr[0, :], rtol=1e-5, atol=1e-8)):
        return 0.0, 0.0, 0.0
    pca = PCA(n_components=3)
    try:
        pca.fit(neighbor_points_arr)
    except ValueError:
        return 0.0, 0.0, 0.0
    eigenvalues = pca.explained_variance_
    total = np.sum(eigenvalues)
    l1, l2, l3 = eigenvalues[0], eigenvalues[1], eigenvalues[2]
    if total < 1e-9:
        return 0.0, 0.0, 0.0
    ldi = (l1 - l2) / l1
    pdi = (l2 - l3) / l1
    sdi = l3 / l1
    return ldi, pdi, sdi


def reconstruct_skeleton_topology(parameters, skeleton_points, _skeleton_tree=None):
    """Connect skeleton points with the original root-guided weighted tree."""
    def __triangulate_and_prune_region(points_subset, original_indices, prune_percentile):
        if len(points_subset) < 4:
            return []
        try:
            tri = Delaunay(points_subset)
            edges_local = {tuple(sorted((i, j))) for s in tri.simplices for i, j in itertools.combinations(s, 2)}
            if not edges_local:
                return []
            edges_local_arr = np.array(list(edges_local))
            edge_lengths = np.linalg.norm(points_subset[edges_local_arr[:, 0]] - points_subset[edges_local_arr[:, 1]],
                                          axis=1)
            length_threshold = np.percentile(edge_lengths, prune_percentile)
            pruned_edges_local = edges_local_arr[edge_lengths < length_threshold]
            return [(original_indices[u], original_indices[v]) for u, v in pruned_edges_local]
        except Exception:
            return []

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
            comp1_indices = components[i]
            comp2_indices = components[i + 1]
            if not comp1_indices or not comp2_indices:
                continue
            tree_comp2 = cKDTree(points[comp2_indices])
            dists, idx_in_comp2 = tree_comp2.query(points[comp1_indices], k=1)
            min_dist_idx_in_comp1 = np.argmin(dists)
            u = comp1_indices[min_dist_idx_in_comp1]
            v = comp2_indices[idx_in_comp2[min_dist_idx_in_comp1]]
            adj[u].add(v)
            adj[v].add(u)
        return adj

    def _build_graph_knn(points, params):
        adj = {i: set() for i in range(len(points))}
        tree = cKDTree(points)
        _, indices = tree.query(points, k=min(params.topology_neighbor_count + 1, len(points)))
        for i in range(len(points)):
            for neighbor_idx in indices[i][1:]:
                adj[i].add(neighbor_idx)
                adj[neighbor_idx].add(i)
        return adj

    def _build_weighted_tree(points, adj, params):
        num_points = len(points)
        start_node_idx = np.argmin(points[:, 2])
        parent = np.full(num_points, -1, dtype=int)
        dist_to_root = np.full(num_points, np.inf)
        dist_to_root[start_node_idx] = 0
        in_tree = {start_node_idx}
        edge_heap = []
        for neighbor in adj.get(start_node_idx, []):
            mst_dist = np.linalg.norm(points[start_node_idx] - points[neighbor])
            dijkstra_dist = dist_to_root[start_node_idx] + mst_dist
            total_weight = params.topology_alpha * mst_dist + params.topology_beta * dijkstra_dist
            heapq.heappush(edge_heap, (total_weight, neighbor, start_node_idx))
        while edge_heap and len(in_tree) < num_points:
            _, new_node, existing_node = heapq.heappop(edge_heap)
            if new_node in in_tree:
                continue
            in_tree.add(new_node)
            parent[new_node] = existing_node
            mst_dist = np.linalg.norm(points[new_node] - points[existing_node])
            dist_to_root[new_node] = dist_to_root[existing_node] + mst_dist
            for neighbor in adj.get(new_node, []):
                if neighbor not in in_tree:
                    mst_dist = np.linalg.norm(points[new_node] - points[neighbor])
                    dijkstra_dist = dist_to_root[new_node] + mst_dist
                    total_weight = params.topology_alpha * mst_dist + params.topology_beta * dijkstra_dist
                    heapq.heappush(edge_heap, (total_weight, neighbor, new_node))
        edges = [[p_idx, i] for i, p_idx in enumerate(parent) if p_idx != -1]
        return np.array(edges) if edges else np.empty((0, 2), dtype=int)

    if skeleton_points is None or len(skeleton_points) < 2:
        return None
    points_3d = np.asarray(skeleton_points[:, :3], dtype=np.float64)

    adj = _build_graph_knn(points_3d, parameters)
    adj = _ensure_connectivity(points_3d, adj)
    final_edges = _build_weighted_tree(points_3d, adj, parameters)
    save_path = parameters.topology_save_path
    if save_path:
        try:
            line_set = o3d.geometry.LineSet(points=o3d.utility.Vector3dVector(points_3d),
                                            lines=o3d.utility.Vector2iVector(final_edges))
            output_dir = os.path.dirname(save_path)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
            o3d.io.write_line_set(save_path, line_set, write_ascii=True)
        except Exception:
            pass
    return final_edges

def contract_water_droplets(point_cloud, parameters, savepath, tree_name):
    """Run contraction, merging, tree growth and evaporation on an XYZ array.

    The order of in-place point updates is intentionally preserved from the
    research implementation because later droplets in an iteration observe
    coordinates already updated for earlier droplets.
    """
    if parameters.tree_root is None or parameters.base_radius <= 0 or parameters.max_radius <= 0:
        raise ValueError("Tree geometry must be set before droplet contraction.")
    if not 0.0 <= parameters.gamma <= 1.0:
        raise ValueError("gamma must be between 0 and 1.")
    if parameters.tree_growth_start_iteration < 1:
        raise ValueError("tree_growth_start_iteration must be at least 1.")
    if parameters.max_iterations < parameters.tree_growth_start_iteration:
        raise ValueError("max_iterations must reach tree_growth_start_iteration.")
    if parameters.topology_neighbor_count < 1:
        raise ValueError("topology_neighbor_count must be at least 1.")

    masses, initial_radii, ldi, pdi, sdi = initialize_droplets(point_cloud, parameters)
    droplet_state = np.column_stack((point_cloud, masses, initial_radii, ldi, pdi, sdi))
    previous_entropy = None
    entropy_history = []
    completed_iterations = []
    skeleton_points = None
    skeleton_edges = None

    for iteration_index in range(parameters.max_iterations):
        iteration_number = iteration_index + 1
        print(f"Water-droplet contraction, iteration {iteration_number}")
        droplet_tree = cKDTree(droplet_state[:, :3])
        neighbors_list = [
            droplet_tree.query_ball_point(droplet_state[i, :3], droplet_state[i, 4])
            for i in range(len(droplet_state))
        ]

        for droplet_index in tqdm(range(len(droplet_state)), desc="Droplet index"):
            neighbors = [index for index in neighbors_list[droplet_index] if index != droplet_index]
            if len(neighbors) < 3:
                droplet_state[droplet_index, 5] = 0.5
                droplet_state[droplet_index, 6] = 0.0
                droplet_state[droplet_index, 4] = 0.0
                droplet_state[droplet_index, 7] = 0.0
                continue

            neighbor_points = droplet_state[neighbors, :3]
            neighbor_masses = droplet_state[neighbors, 3]
            objective, fallback_position = create_contraction_objective(
                neighbor_points=neighbor_points,
                gamma=parameters.gamma,
                masses=neighbor_masses,
            )
            optimization = minimize(
                objective,
                droplet_state[droplet_index, :3],
                method="L-BFGS-B",
            )

            if optimization.success:
                droplet_state[droplet_index, :3] = optimization.x
                ldi_value, pdi_value, sdi_value = compute_shape_descriptors(neighbor_points)
            else:
                # The debug release referenced stale local variables here.  The
                # assigned fallback descriptors make the intended update explicit.
                droplet_state[droplet_index, :3] = fallback_position
                ldi_value, pdi_value, sdi_value = 0.0, 0.0, 1.0

            droplet_state[droplet_index, 5:8] = (ldi_value, pdi_value, sdi_value)
            droplet_state[droplet_index, 4] *= 2 * (1 - ldi_value + pdi_value + sdi_value)

        droplet_state = update_droplet_masses(droplet_state, parameters)
        droplet_state = merge_droplets(droplet_state)

        if parameters.save_intermediate_results:
            iteration_path = Path(savepath) / tree_name / f"droplets_iteration_{iteration_number}.txt"
            iteration_path.parent.mkdir(parents=True, exist_ok=True)
            np.savetxt(
                str(iteration_path),
                droplet_state,
                fmt="%.6f",
                header="X Y Z mass radius LDI PDI SDI",
                delimiter=" ",
                comments="",
            )

        if iteration_number >= parameters.tree_growth_start_iteration:
            paths, points_with_attributes, root_index, _ = build_tree_growth_graph(
                droplet_state[:, :3], parameters, parameters.tree_root, droplet_state
            )
            skeleton_points = select_skeleton_points(
                points_with_attributes,
                parameters,
                paths,
                iteration_number,
                root_index,
            )
            skeleton_edges = reconstruct_skeleton_topology(parameters, skeleton_points)

            if parameters.save_intermediate_results:
                points_path = (
                    Path(savepath) / tree_name /
                    f"skeleton_points_iteration_{iteration_number}.txt"
                )
                points_path.parent.mkdir(parents=True, exist_ok=True)
                np.savetxt(
                    str(points_path),
                    skeleton_points,
                    fmt="%.6f",
                    header="X Y Z path_length location branch_id LDI PDI SDI mass",
                    delimiter=" ",
                    comments="",
                )

            if skeleton_edges is not None and len(skeleton_edges) > 0:
                droplet_state, _ = evaporate_droplets(
                    droplet_state,
                    skeleton_points,
                    skeleton_edges,
                    parameters,
                    savepath,
                    tree_name,
                    iteration_number,
                )

        current_entropy = compute_entropy_metric(
            droplet_state[:, 5],
            droplet_state[:, 6],
            droplet_state[:, 7],
            alpha=parameters.entropy_linearity_weight,
            beta=parameters.entropy_planarity_weight,
        )
        entropy_history.append(current_entropy)
        completed_iterations.append(iteration_number)

        converged = has_converged(
            prev_entropy=previous_entropy,
            curr_entropy=current_entropy,
            iteration=iteration_number,
            tree_grow_thresh=parameters.tree_growth_start_iteration,
            delta=parameters.convergence_tolerance,
            max_iterations=parameters.max_iterations,
        )
        if parameters.save_intermediate_results:
            save_entropy_plot(entropy_history, completed_iterations, savepath, tree_name)
        if converged:
            break
        previous_entropy = current_entropy

    if skeleton_points is None or skeleton_edges is None:
        raise RuntimeError("Skeleton topology was not produced by the configured iterations.")
    return skeleton_points, skeleton_edges


def load_point_cloud(input_path):
    """Load XYZ points from a headerless/headered text, CSV, XYZ, PLY or PCD file."""
    path = Path(input_path)
    if not path.is_file():
        raise FileNotFoundError(f"Point-cloud file does not exist: {path}")

    if path.suffix.lower() in {".ply", ".pcd"}:
        point_cloud = o3d.io.read_point_cloud(str(path))
        points = np.asarray(point_cloud.points, dtype=float)
    else:
        first_data_line = ""
        first_content_row = 0
        with path.open("r", encoding="utf-8-sig") as file:
            for row_number, line in enumerate(file):
                stripped = line.strip()
                if stripped and not stripped.startswith("#"):
                    first_data_line = stripped
                    first_content_row = row_number
                    break
        delimiter = "," if "," in first_data_line else None
        tokens = first_data_line.split(",") if delimiter == "," else first_data_line.split()
        try:
            [float(value) for value in tokens[:3]]
            skip_rows = first_content_row
        except (TypeError, ValueError):
            header = [value.strip().lower() for value in tokens[:3]]
            if header != ["x", "y", "z"]:
                raise ValueError(
                    f"The optional header must start with x, y, z; found {tokens[:3]} in {path}."
                )
            skip_rows = first_content_row + 1
        points = np.loadtxt(
            str(path), delimiter=delimiter, skiprows=skip_rows, usecols=(0, 1, 2), dtype=float
        )

    points = np.atleast_2d(points)
    if points.shape[1] != 3 or len(points) < 3 or not np.all(np.isfinite(points)):
        raise ValueError(f"Expected at least three finite XYZ points in {path}.")
    return points


def save_skeleton(skeleton_path, points, edges):
    """Write skeleton vertices and indexed edges as an ASCII PLY line set."""
    path = Path(skeleton_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(np.asarray(points)[:, :3]),
        lines=o3d.utility.Vector2iVector(np.asarray(edges, dtype=int)),
    )
    if not o3d.io.write_line_set(str(path), line_set, write_ascii=True):
        raise OSError(f"Failed to write skeleton: {path}")
    return path


def run_skeletonization(
    input_path: PathLike,
    output_dir: PathLike = "results",
    gamma: Optional[float] = None,
    tree_id: Optional[str] = None,
    config: Optional[SkeletonizationConfig] = None,
    downsample_threshold: int = 15000,
    voxel_size_divisor: float = 70.0,
) -> SkeletonizationResult:
    """Skeletonize one tree point cloud and save canonical result files."""
    started_at = time.perf_counter()
    input_path = Path(input_path)
    tree_id = tree_id or input_path.stem
    output_directory = Path(output_dir) / tree_id

    point_cloud = load_point_cloud(input_path)
    base_radius, max_radius, tree_root = estimate_tree_base_geometry(point_cloud)
    if tree_root is None or base_radius is None or max_radius is None or base_radius <= 0 or max_radius <= 0:
        raise ValueError(f"Could not estimate valid base geometry for {input_path}.")

    point_cloud = downsample_if_needed(
        point_cloud,
        base_radius=base_radius,
        threshold=downsample_threshold,
        voxel_size_divisor=voxel_size_divisor,
    )
    if config is None:
        config = SkeletonizationConfig()
        if gamma is None:
            gamma = 0.1
    else:
        config = deepcopy(config)
    config.set_tree_geometry(tree_root, base_radius, max_radius)
    if gamma is not None:
        config.gamma = float(gamma)
    if not 0.0 <= config.gamma <= 1.0:
        raise ValueError("gamma must be between 0 and 1.")

    skeleton_path = output_directory / "initial_skeleton.ply"
    points_path = output_directory / "initial_skeleton_points.txt"
    # Only the final successful result receives the canonical filename.  This
    # prevents an interrupted iteration from leaving a partial result that
    # looks complete.
    config.topology_save_path = None
    skeleton_points, skeleton_edges = contract_water_droplets(
        point_cloud,
        config,
        Path(output_dir),
        tree_id,
    )

    output_directory.mkdir(parents=True, exist_ok=True)
    np.savetxt(
        str(points_path),
        skeleton_points,
        fmt="%.6f",
        header="X Y Z path_length location branch_id LDI PDI SDI mass",
        delimiter=" ",
        comments="",
    )
    save_skeleton(skeleton_path, skeleton_points, skeleton_edges)
    elapsed = time.perf_counter() - started_at
    print(
        f"Skeletonization completed for {tree_id}: {len(skeleton_points)} points "
        f"in {elapsed:.2f} seconds."
    )
    return SkeletonizationResult(
        points=skeleton_points,
        edges=skeleton_edges,
        input_path=input_path,
        output_directory=output_directory,
        points_path=points_path,
        skeleton_path=skeleton_path,
    )


# Selected compatibility wrappers cover the original core call sites.  Dead
# debug-only helpers and unused configuration fields were intentionally pruned;
# new code should use the descriptive API above.
class Parameters(SkeletonizationConfig):
    def __init__(self, tree_root, r_base, max_radius):
        super().__init__(tree_root=tree_root, base_radius=r_base, max_radius=max_radius)


get_root_and_radii_lubang = estimate_tree_base_geometry
conditional_voxel_downsample = downsample_if_needed
compute_transformed_kde_metric_sklearn_optimized = compute_entropy_metric
check_convergence = has_converged
plot_entropy_history = save_entropy_plot
assign_initial_mass_lubang = initialize_droplets
create_surface_objective_huber_massweighted = create_contraction_objective
compute_ldi_pdi = compute_shape_descriptors


def fuse_droplets_mass_radius(TLSM, k=0.0001, new_search_radius_method="average", kdt=None):
    """Compatibility wrapper for the original droplet-merging signature."""
    return merge_droplets(
        TLSM,
        k=k,
        new_search_radius_method=new_search_radius_method,
        kdt=kdt,
    )


def update_mass_vectorized(TLSM, parameters):
    """Compatibility wrapper for the original mass-update signature."""
    return update_droplet_masses(TLSM, parameters)


def restore_skeleton_topology(parameters, skPoints, skTree=None):
    """Compatibility wrapper for the original topology-restoration signature."""
    return reconstruct_skeleton_topology(parameters, skPoints, skTree)


def global_delaunay_construct_graph(point_cloud, parameters, treeroot, TLSM):
    return build_tree_growth_graph(point_cloud, parameters, treeroot, TLSM)


def tree_grow_optimization(tlsPoints, parameters, paths, T, root_idx):
    return select_skeleton_points(tlsPoints, parameters, paths, T, root_idx)


def water_droplet_evaporate(TLSM, skPoints, skEdges, parameters, savepath, NAME, iter):
    return evaporate_droplets(TLSM, skPoints, skEdges, parameters, savepath, NAME, iter)


def water_droplet_contract(point_cloud, parameters, savepath, name):
    return contract_water_droplets(point_cloud, parameters, savepath, name)
