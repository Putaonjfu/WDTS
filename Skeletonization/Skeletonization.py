import numpy as np
from scipy.spatial import cKDTree, Delaunay
from scipy.optimize import minimize
from sklearn.decomposition import PCA
from tqdm import tqdm
import os
from datetime import datetime
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


class Parameters:
    def __init__(self, tree_root, r_base, max_radius):
        self.gamma = 0.2
        self.epsilon = r_base * 0.05
        self.r_base = r_base
        self.max_iter = 5
        self.max_iteration = 5
        self.linearity_threshold = 0.9
        self.surface_threshold = 0.001
        self.evap_factor = 5.0
        self.min_neighbors = 3
        self.density_factor = 2.0
        self.mass_threshold = 1.5
        self.step_factor = 0.5
        self.contraction_phase = 10
        self.optimize_threshold = 0.1
        self.tree_root = tree_root
        self.debug_ldi_filter = False
        self.tree_grow_thresh = 3
        self.max_radius = max_radius
        self.eta = 0.001
        self.delta = 0.0001
        self.alpha = 0.99
        self.beta = 0.01
        self.topology_method = 'knn'
        self.topology_visualize_steps = False
        self.topology_save_path = None
        self.topology_K = 20
        self.topology_z_split_ratio = 0.4
        self.topology_prune_percentile = 85
        self.topology_inter_connect_k = 20
        self.topology_alpha = 0.5
        self.topology_beta = 0.5
        self.w_mass = 0.1
        self.w_ldi = 0.1
        self.w_pdi = -0.1
        self.w_sdi = -0.1


def conditional_voxel_downsample(point_cloud: np.ndarray, base_radius: float, threshold: int = 15000,
                                 voxel_size_divisor: float = 40.0) -> np.ndarray:
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


def get_filename_with_time(NAME, base_filename, savepath):
    output_dir = os.path.join(savepath, NAME)
    os.makedirs(output_dir, exist_ok=True)
    current_time = datetime.now().strftime("%Y%m%d%H%M%S")
    file_name_parts = base_filename.split('.')
    new_file_name = f"{file_name_parts[0]}_{current_time}.{file_name_parts[1]}"
    return os.path.join(output_dir, new_file_name)


def get_root_and_radii_lubang(points, gamma=0.9):
    print('Fitting the point cloud within 0.5 m above the minimum Z to estimate the root point and radius.')
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


def global_delaunay_construct_graph(point_cloud, parameters, treeroot, TLSM):
    def update_attr_matrix(path, attrMatrix):
        path_length = len(path)
        if path_length == 0:
            return
        tempIDs = np.array(path)
        tempLocations = np.linspace(1, path_length, path_length)
        np.add.at(attrMatrix, (tempIDs, 0), 1)
        selectIDs = np.where(attrMatrix[tempIDs, 1] < path_length)[0]
        if selectIDs.size > 0:
            attrMatrix[tempIDs[selectIDs], 1] = path_length
            attrMatrix[tempIDs[selectIDs], 2] = tempLocations[selectIDs]

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

    def compute_mst(sparse_matrix, point_cloud, neighbor_indices, treeroot, alpha=0, beta=1):
        num_points = len(point_cloud)
        kdtree = cKDTree(point_cloud)
        _, root_idx = kdtree.query(treeroot, k=1)
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

    def compute_sparse_distance_matrix(point_cloud, TLSM, k, r=None):
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
            mean_LDI = (TLSM[u, 5] + TLSM[v, 5]) / 2
            mean_PDI = (TLSM[u, 6] + TLSM[v, 6]) / 2
            mean_SDI = (TLSM[u, 7] + TLSM[v, 7]) / 2
            mean_mass = (TLSM[u, 3] + TLSM[v, 3]) / 2
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
    dist, idx = kdtree.query(treeroot, k=1)
    if dist > 1e-6:
        point_cloud_with_root = np.vstack([point_cloud, treeroot])
        TLSM_with_root = np.vstack(
            [TLSM, np.array([treeroot[0], treeroot[1], treeroot[2], 1.0, parameters.r_base, 1.0, 0.0, 0.0])])
        added_root = True
    else:
        point_cloud_with_root = point_cloud
        TLSM_with_root = TLSM
        added_root = False

    G, neighbor_indices = compute_sparse_distance_matrix(point_cloud_with_root, TLSM_with_root, k=50)
    predecessors, root_idx = compute_mst(G, point_cloud_with_root, neighbor_indices, treeroot)
    paths = extract_paths(predecessors, root_idx)
    paths = paths[:-1]

    attrMatrix = np.zeros((len(point_cloud_with_root), 3))
    with ThreadPoolExecutor() as executor:
        valid_paths = [p for p in paths if p]
        list(tqdm(executor.map(lambda path: update_attr_matrix(path, attrMatrix), valid_paths),
                  total=len(valid_paths), desc='Computing node attributes along paths'))

    ldi_pdi_sdi_mass = TLSM_with_root[:, [5, 6, 7, 3]] # ldi pdi sdi mass
    tlsPoints_with_attrs = np.hstack([point_cloud_with_root, attrMatrix, ldi_pdi_sdi_mass])
    return paths, tlsPoints_with_attrs, root_idx, added_root


def tree_grow_optimization(tlsPoints, parameters, paths, T, root_idx):
    tree = KDTree(tlsPoints[:, :3])
    tlsNeighborMatrix = [None] * tlsPoints.shape[0]
    for i in tqdm(range(tlsPoints.shape[0]), desc='Building neighbor matrix'):
        tempPoint = tlsPoints[i, :3]
        path_length = tlsPoints[i, 4] if tlsPoints[i, 4] != 0 else 1e-6
        tempDistanceThreshold = max(np.sqrt(tlsPoints[i, 5] / path_length) * parameters.max_radius * 2,
                                    parameters.max_radius * 0.1 * 2)
        selectIDs = tree.query_ball_point(tempPoint, tempDistanceThreshold)
        tlsNeighborMatrix[i] = selectIDs

    skeletonPointIDs = []
    isConsider = np.zeros(tlsPoints.shape[0], dtype=bool)
    isConsider[root_idx] = True
    branches = []

    while not np.all(isConsider):
        selectIDs = np.where(~isConsider)[0]
        location = tlsPoints[selectIDs, 5]
        length = tlsPoints[selectIDs, 4]
        length = np.where(length == 0, 1e-6, length)
        idx = np.argmin(np.sqrt(location / length))
        currentPointID = selectIDs[idx]

        addSkeletonPointIDs = paths[currentPointID]
        if not addSkeletonPointIDs:
            isConsider[currentPointID] = True
        else:
            branch = [id for id in addSkeletonPointIDs if id not in skeletonPointIDs]
            if branch:
                branches.append(branch)
                skeletonPointIDs.extend(branch)
                for addID in branch:
                    isConsider[tlsNeighborMatrix[addID]] = True

    ordered_skeletonPointIDs = []
    branch_ids = []
    for branch_idx, branch in enumerate(branches):
        ordered_skeletonPointIDs.extend(branch)
        branch_ids.extend([branch_idx] * len(branch))

    unique_indices = list(dict.fromkeys(ordered_skeletonPointIDs))
    skPoints_base = tlsPoints[unique_indices, :][:, [0, 1, 2, 4, 5, 6, 7, 8, 9]]
    branch_ids = np.array([branch_ids[ordered_skeletonPointIDs.index(idx)] for idx in unique_indices])
    skPoints = np.column_stack((skPoints_base[:, :5], branch_ids, skPoints_base[:, 5:]))

    tree = KDTree(skPoints[:, :3])
    skNeighborIDs = [tree.query_ball_point(point, parameters.r_base * 0.1) for point in skPoints[:, :3]]
    delIDs = []
    for i in range(len(skPoints)):
        if i in delIDs:
            continue
        delIDs.extend(list(set(skNeighborIDs[i]) - {i}))

    delIDs = sorted(set(delIDs))
    skPoints = np.delete(skPoints, delIDs, axis=0)
    if parameters.debug_ldi_filter:
        ldi_mask = skPoints[:, 6] >= 0.9
        skPoints = skPoints[ldi_mask]
        print("Points with LDI < 0.9 have been removed. Remaining points:", skPoints.shape[0])
    return skPoints


def compute_transformed_kde_metric_sklearn_optimized(ldi, pdi, sdi, alpha, beta, gamma_sdi=0.3, C_constant=1.0,
                                                     clip_eps=1e-6,
                                                     subsample_size=7500):
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

def check_convergence(prev_entropy, curr_entropy, iteration, delta, max_iterations, tree_grow_thresh):
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


def plot_entropy_history(entropy_history, iterations, savepath, name):
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


def assign_initial_mass_lubang(point_cloud, parameters):
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
        ldi[i], pdi[i], sdi[i] = compute_ldi_pdi(neighbor_points)

    min_neighbors = np.min(neighbor_counts)
    max_neighbors = np.max(neighbor_counts)
    if max_neighbors == min_neighbors:
        masses[:] = 1 + parameters.density_factor / 2
    else:
        normalized_counts = (neighbor_counts - min_neighbors) / (max_neighbors - min_neighbors)
        masses = 1 + parameters.density_factor * normalized_counts

    return masses, radii, ldi, pdi, sdi



def fuse_droplets_mass_radius(TLSM, k=0.0001, new_search_radius_method='average', kdt=None):
    if TLSM is None or TLSM.shape[0] < 2:
        return TLSM
    if TLSM.shape[1] <= 3:
        warnings.warn("TLSM is missing mass data (column index 3). Fusion cannot be executed. Returning original data.")
        return TLSM

    mass = np.maximum(TLSM[:, 3], 0)
    points = TLSM[:, :3]
    if kdt is None:
        try:
            points_for_kdtree = points.astype(float)
            kdt = cKDTree(points_for_kdtree)
        except Exception:
            return TLSM

    max_mass = np.max(mass)
    if max_mass <= 0:
        return TLSM

    proxy_radius = np.power(mass, 1 / 3)
    max_proxy_radius = np.max(proxy_radius)
    search_epsilon = 2.0 * k * max_proxy_radius * 1.1
    if search_epsilon <= 1e-9:
        search_epsilon = 1e-9

    try:
        potential_pairs = kdt.query_pairs(r=search_epsilon, output_type='set')
    except Exception:
        return TLSM

    fused_indices = set()
    merge_map = {i: i for i in range(len(TLSM))}
    TLSM_fused = TLSM.copy()
    current_proxy_radius = proxy_radius.copy()
    current_mass = mass.copy()
    sorted_pairs = sorted(list(potential_pairs))

    for i, j in sorted_pairs:
        target_i = merge_map[i]
        target_j = merge_map[j]
        if target_i == target_j or target_i in fused_indices or target_j in fused_indices:
            continue
        if target_i > target_j:
            target_i, target_j = target_j, target_i

        pos_i, pos_j = TLSM_fused[target_i, :3], TLSM_fused[target_j, :3]
        m_i, m_j = current_mass[target_i], current_mass[target_j]
        pr_i, pr_j = current_proxy_radius[target_i], current_proxy_radius[target_j]
        dist_sq = np.sum((pos_i - pos_j) ** 2)
        fuse_thresh_dist = k * (pr_i + pr_j)
        fuse_thresh_dist_sq = fuse_thresh_dist ** 2

        if dist_sq < fuse_thresh_dist_sq:
            total_mass = m_i + m_j
            if total_mass > 1e-12:
                TLSM_fused[target_i, :3] = (m_i * pos_i + m_j * pos_j) / total_mass
            elif m_i > 1e-12:
                TLSM_fused[target_i, :3] = pos_i
            elif m_j > 1e-12:
                TLSM_fused[target_i, :3] = pos_j
            else:
                TLSM_fused[target_i, :3] = (pos_i + pos_j) / 2.0

            TLSM_fused[target_i, 3] = total_mass
            current_mass[target_i] = total_mass
            current_proxy_radius[target_i] = np.power(max(0, total_mass), 1 / 3)
            s_i_orig = TLSM_fused[target_i, 4]
            s_j_orig = TLSM_fused[target_j, 4]
            if new_search_radius_method == 'max':
                TLSM_fused[target_i, 4] = max(s_i_orig, s_j_orig)
            elif new_search_radius_method == 'min':
                TLSM_fused[target_i, 4] = min(s_i_orig, s_j_orig)
            elif new_search_radius_method == 'average':
                TLSM_fused[target_i, 4] = (s_i_orig + s_j_orig) / 2.0
            elif new_search_radius_method == 'weighted_average':
                if total_mass > 1e-12:
                    TLSM_fused[target_i, 4] = (m_i * s_i_orig + m_j * s_j_orig) / total_mass
                else:
                    TLSM_fused[target_i, 4] = max(s_i_orig, s_j_orig)
            else:
                TLSM_fused[target_i, 4] = max(s_i_orig, s_j_orig)

            has_ldi_pdi = TLSM_fused.shape[1] >= 8
            if has_ldi_pdi:
                ldi_i, pdi_i, sdi_i = TLSM_fused[target_i, 5], TLSM_fused[target_i, 6], TLSM_fused[target_i, 7]
                ldi_j, pdi_j, sdi_j = TLSM_fused[target_j, 5], TLSM_fused[target_j, 6], TLSM_fused[target_j, 7]
                TLSM_fused[target_i, 5] = (ldi_i + ldi_j) / 2.0
                TLSM_fused[target_i, 6] = (pdi_i + pdi_j) / 2.0
                TLSM_fused[target_i, 7] = (sdi_i + sdi_j) / 2.0

            fused_indices.add(target_j)
            keys_to_update = [k_map for k_map, v_map in merge_map.items() if v_map == target_j]
            for k_map in keys_to_update:
                merge_map[k_map] = target_i
            merge_map[j] = target_i
            current_mass[target_j] = 0
            current_proxy_radius[target_j] = 0

    if fused_indices:
        keep_mask = np.ones(len(TLSM_fused), dtype=bool)
        keep_mask[list(fused_indices)] = False
        TLSM_final = TLSM_fused[keep_mask]
    else:
        TLSM_final = TLSM_fused
    return TLSM_final


def update_mass_vectorized(TLSM, parameters):
    num_particles = len(TLSM)
    if num_particles == 0:
        return TLSM
    positions = TLSM[:, :3]
    radii = TLSM[:, 4]
    kdtree = KDTree(positions)
    neighbors_indices_with_self = kdtree.query_ball_point(positions, radii, return_sorted=True)
    vlen = np.frompyfunc(len, 1, 1)
    neighbor_counts = vlen(neighbors_indices_with_self).astype(int) - 1
    ldi = TLSM[:, 5]
    mask_isolated = neighbor_counts < 3
    mask_normal = ~mask_isolated
    TLSM[mask_isolated, 3] = 1.0
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
        TLSM[mask_normal, 3] = new_mass
    return TLSM

def create_surface_objective_huber_massweighted(neighbor_points, masses, gamma, delta=1.5):
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


def water_droplet_evaporate(TLSM, skPoints, skEdges, parameters, savepath, NAME, iter):
    def point_to_segment_distance(p, a, b):
        if np.all(a == b):
            return np.linalg.norm(p - a)
        ab = b - a
        ap = p - a
        t = np.dot(ap, ab) / np.dot(ab, ab)
        if t < 0.0:
            closest_point = a
        elif t > 1.0:
            closest_point = b
        else:
            closest_point = a + t * ab
        return np.linalg.norm(p - closest_point)

    if not TLSM.size or not skPoints.size or not skEdges.size:
        return TLSM, None
    if TLSM.shape[1] < 8 or skPoints.shape[1] < 10:
        return TLSM, None

    n_tlsm = TLSM.shape[0]
    n_sk_pts = skPoints.shape[0]
    n_sk_edges = skEdges.shape[0]
    tls_tree = cKDTree(TLSM[:, :3])
    r_base = getattr(parameters, 'r_base', 0.1)
    min_neighbors = getattr(parameters, 'min_neighbors', 5)
    neighbor_counts = np.array([len(tls_tree.query_ball_point(p, r_base)) for p in TLSM[:, :3]])
    density_mask = neighbor_counts >= min_neighbors
    TLSM = TLSM[density_mask]
    n_tlsm = TLSM.shape[0]
    if n_tlsm == 0:
        return TLSM, None

    sk_tree = cKDTree(skPoints[:, :3])
    adj = {i: [] for i in range(n_sk_pts)}
    for u, v in skEdges:
        adj[u].append(v)
        adj[v].append(u)

    distances = np.full(n_tlsm, np.inf)
    nearest_edge_indices = np.full(n_tlsm, -1, dtype=int)
    k_search = getattr(parameters, 'topology_K', 10)
    _, nearest_sk_point_indices = sk_tree.query(TLSM[:, :3], k=min(k_search, n_sk_pts))
    if nearest_sk_point_indices.ndim == 1:
        nearest_sk_point_indices = nearest_sk_point_indices[:, np.newaxis]
    edge_map = {tuple(sorted(edge)): i for i, edge in enumerate(skEdges)}

    for i in range(n_tlsm):
        p = TLSM[i, :3]
        min_dist_for_p = np.inf
        best_edge_idx = -1
        candidate_edges = set()
        for sk_pt_idx in nearest_sk_point_indices[i]:
            for neighbor_idx in adj[sk_pt_idx]:
                edge_tuple = tuple(sorted((sk_pt_idx, neighbor_idx)))
                candidate_edges.add(edge_tuple)
        if not candidate_edges:
            continue
        for u, v in candidate_edges:
            dist = point_to_segment_distance(p, skPoints[u, :3], skPoints[v, :3])
            if dist < min_dist_for_p:
                min_dist_for_p = dist
                best_edge_idx = edge_map[tuple(sorted((u, v)))]
        distances[i] = min_dist_for_p
        nearest_edge_indices[i] = best_edge_idx

    valid_mask = nearest_edge_indices != -1
    TLSM = TLSM[valid_mask]
    distances = distances[valid_mask]
    nearest_edge_indices = nearest_edge_indices[valid_mask]

    edge_radii = np.zeros(n_sk_edges)
    edge_radii_std = np.zeros(n_sk_edges)
    edge_to_distances = [[] for _ in range(n_sk_edges)]
    for tls_idx, edge_idx in enumerate(nearest_edge_indices):
        edge_to_distances[edge_idx].append(distances[tls_idx])
    for i in range(n_sk_edges):
        dists = edge_to_distances[i]
        if dists:
            edge_radii[i] = np.mean(dists)
            edge_radii_std[i] = np.std(dists)

    sk_ldi = skPoints[:, 6]
    sk_pdi = skPoints[:, 7]
    sk_sdi = skPoints[:, 8]
    sk_mass = skPoints[:, 9]
    edge_endpoints = skEdges[nearest_edge_indices]
    ref_mass = (sk_mass[edge_endpoints[:, 0]] + sk_mass[edge_endpoints[:, 1]]) / 2
    ref_ldi = (sk_ldi[edge_endpoints[:, 0]] + sk_ldi[edge_endpoints[:, 1]]) / 2
    ref_pdi = (sk_pdi[edge_endpoints[:, 0]] + sk_pdi[edge_endpoints[:, 1]]) / 2
    ref_sdi = (sk_sdi[edge_endpoints[:, 0]] + sk_sdi[edge_endpoints[:, 1]]) / 2

    tls_mass = TLSM[:, 3]
    tls_ldi = TLSM[:, 5]
    tls_pdi = TLSM[:, 6]
    tls_sdi = TLSM[:, 7]

    def norm(x):
        min_val, max_val = np.min(x), np.max(x)
        range_val = max_val - min_val
        return (x - min_val) / (range_val + 1e-6)

    mass_norm = norm(ref_mass)
    ldi_norm = norm(ref_ldi)
    pdi_norm = norm(ref_pdi)
    sdi_norm = norm(ref_sdi)
    tls_mass_norm = norm(tls_mass)
    tls_ldi_norm = norm(tls_ldi)
    tls_pdi_norm = norm(tls_pdi)
    tls_sdi_norm = norm(tls_sdi)

    w_mass = getattr(parameters, 'w_mass', 0.1)
    w_ldi = getattr(parameters, 'w_ldi', 0.1)
    w_pdi = getattr(parameters, 'w_pdi', -0.1)
    w_sdi = getattr(parameters, 'w_sdi', -0.1)
    factor = 1.0 + w_mass * (mass_norm + tls_mass_norm) + w_ldi * (ldi_norm + tls_ldi_norm) + w_pdi * (
                pdi_norm + tls_pdi_norm) + w_sdi * (sdi_norm + tls_sdi_norm)
    base_thresh = edge_radii[nearest_edge_indices] + edge_radii_std[nearest_edge_indices]
    thresh = base_thresh * factor
    evap_mask = distances <= thresh
    TLSM_new = TLSM[evap_mask]

    output_dir = os.path.join(savepath, NAME)
    os.makedirs(output_dir, exist_ok=True)
    pcd_before = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(TLSM[:, :3]))
    o3d.io.write_point_cloud(os.path.join(output_dir, f"{NAME}_iter_{iter}_before_eva.ply"), pcd_before)
    pcd_after = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(TLSM_new[:, :3]))
    o3d.io.write_point_cloud(os.path.join(output_dir, f"{NAME}_iter_{iter}_after_eva.ply"), pcd_after)
    evap_filename = os.path.join(output_dir, f"{NAME}_iter_{iter}_evap.txt")
    header = "X Y Z mass radius LDI PDI SDI"
    np.savetxt(evap_filename, TLSM_new, fmt='%.6f', header=header, delimiter=' ', comments='')
    return TLSM_new, edge_radii


def compute_ldi_pdi(neighbor_points):
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


def restore_skeleton_topology(parameters, skPoints, skTree=None):
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
        _, indices = tree.query(points, k=min(params.topology_K + 1, len(points)))
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

    if skPoints is None or len(skPoints) < 2:
        return None
    points_3d = np.asarray(skPoints[:, :3], dtype=np.float64)

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

def save_optimized_skeleton(optimized_skPoints, savepath, name, iterations):
    output_dir = os.path.join(savepath, name)
    os.makedirs(output_dir, exist_ok=True)
    filename = os.path.join(output_dir, f'optimized_xy_skeleton_{iterations}_iters.txt')
    header = "X Y Z"
    try:
        with open(filename, 'w') as f:
            f.write(header + '\n')
            np.savetxt(f, optimized_skPoints, fmt='%.6f', delimiter=' ')
    except Exception:
        pass


def water_droplet_contract(point_cloud, parameters, savepath, name):
    masses, initial_radii, LDI, PDI, SDI = assign_initial_mass_lubang(point_cloud, parameters)
    TLSM = np.column_stack((point_cloud, masses, initial_radii, LDI, PDI, SDI))
    prev_entropy = None
    entropy_history = []
    iterations = []
    for iter in range(parameters.max_iteration):
        print(f"Water-droplet contraction, iteration {iter + 1}")
        kdt = cKDTree(TLSM[:, :3])
        neighbors_list = [kdt.query_ball_point(TLSM[i, :3], TLSM[i, 4]) for i in range(len(TLSM))]
        for i in tqdm(range(len(TLSM)), desc='Droplet index'):
            neighbors = [n for n in neighbors_list[i] if n != i]
            if len(neighbors) < 3:
                TLSM[i, 5] = 0.5
                TLSM[i, 6] = 0
                TLSM[i, 4] = 0
                TLSM[i, 7] = 0
            else:
                neighbor_points = TLSM[neighbors, :3]
                masses = TLSM[neighbors, 3]
                surface_objective, start_point_suggestion = create_surface_objective_huber_massweighted(
                    neighbor_points=neighbor_points, gamma=parameters.gamma, masses=masses)
                result = minimize(surface_objective, TLSM[i, :3], method='L-BFGS-B')
                if result.success:
                    TLSM[i, :3] = result.x
                    ldi, pdi, sdi = compute_ldi_pdi(neighbor_points)
                    TLSM[i, 5] = ldi
                    TLSM[i, 6] = pdi
                    TLSM[i, 7] = sdi
                    TLSM[i, 4] *= 2 * (1 - ldi + pdi + sdi)
                else:
                    TLSM[i, :3] = start_point_suggestion
                    TLSM[i, 5] = 0.0
                    TLSM[i, 6] = 0.0
                    TLSM[i, 7] = 1.0
                    TLSM[i, 4] *= 2 * (1 - 0.0 + 0.0)

        TLSM = update_mass_vectorized(TLSM, parameters)
        TLSM = fuse_droplets_mass_radius(TLSM)
        iter_filename = os.path.join(savepath, name, f"{name}_waterball_shrink_{iter + 1}.txt")
        os.makedirs(os.path.dirname(iter_filename), exist_ok=True)
        header = "X Y Z mass radius LDI PDI SDI"
        np.savetxt(iter_filename, TLSM, fmt='%.6f', header=header, delimiter=' ', comments='')

        if iter + 1 >= parameters.tree_grow_thresh:
            paths, tlsPoints_with_attrs, root_idx, added_root = global_delaunay_construct_graph(
                TLSM[:, :3], parameters, treeroot, TLSM)
            skPoints = tree_grow_optimization(tlsPoints_with_attrs, parameters, paths, iter + 1, root_idx)
            output_dir = os.path.join(savepath, name)
            os.makedirs(output_dir, exist_ok=True)
            filename = os.path.join(output_dir, f'Entroy_waterbal_Treegrowth_optimization_skeleton_points{iter + 1}.txt')
            header1 = "X Y Z PathLength Location BranchID LDI PDI SDI Mass"
            with open(filename, 'w') as f:
                f.write(header1 + '\n')
                np.savetxt(f, skPoints, fmt='%.6f', delimiter=' ')
            tuopu = restore_skeleton_topology(parameters, skPoints)
            TLSM, sk_radii = water_droplet_evaporate(TLSM, skPoints, tuopu, parameters, savepath,
                                                                       name, iter + 1)

        curr_entropy = compute_transformed_kde_metric_sklearn_optimized(
            TLSM[:, 5], TLSM[:, 6], TLSM[:, 7], alpha=parameters.alpha, beta=parameters.beta)
        entropy_history.append(curr_entropy)
        iterations.append(iter + 1)
        if check_convergence(prev_entropy=prev_entropy, curr_entropy=curr_entropy, iteration=iter + 1,
                             tree_grow_thresh=parameters.tree_grow_thresh, delta=parameters.delta,
                             max_iterations=parameters.max_iteration):
            plot_entropy_history(entropy_history, iterations, savepath, name)
            break
        prev_entropy = curr_entropy
        if iter <= parameters.max_iteration:
            plot_entropy_history(entropy_history, iterations, savepath, name)

    return skPoints, tuopu


datapath = '/home/graper/WDTS_test/data/'
base_savepath = '/home/graper/WDTS_test/result_test/'
Name = ['Tree_16']
gamma_values = [0.1]
gamma_times = []

for NAME in Name:
    ORDATA = np.loadtxt(datapath + str(NAME) + '.txt', delimiter=' ', usecols=(0, 1, 2))
    point_cloud = ORDATA
    tree_radius, max_rad, treeroot = get_root_and_radii_lubang(point_cloud)
    point_cloud = conditional_voxel_downsample(point_cloud, base_radius=tree_radius, voxel_size_divisor=70)
    for gamma in gamma_values:
        print(f"\n--- Processing {NAME}, current gamma = {gamma} ---")
        start_time = time.perf_counter()
        current_savepath = os.path.join(base_savepath, f'k=50_local_global_dealaunay{gamma}')
        params = Parameters(r_base=tree_radius, tree_root=treeroot, max_radius=max_rad)
        params.gamma = gamma
        topology_dir = os.path.join(current_savepath, NAME)
        os.makedirs(topology_dir, exist_ok=True)
        params.topology_save_path = os.path.join(topology_dir, 'Entroy_waterbal_Treegrowth_optimization_skeleton_tuopu.ply')
        skPoints, tuopu = water_droplet_contract(point_cloud, params, current_savepath, NAME)
        print(f"Finished processing {NAME} (gamma={gamma}) | Number of droplets after contraction: {len(skPoints)}")
        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
        print(f"--- Runtime for gamma = {gamma}: {elapsed_time:.2f} seconds ---")
        gamma_times.append({'gamma': gamma, 'time': elapsed_time})

print("\nAll datasets have been processed.")
for record in gamma_times:
    print(f"Gamma: {record['gamma']}, time: {record['time']:.2f} seconds")