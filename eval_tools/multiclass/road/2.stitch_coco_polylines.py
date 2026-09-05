import argparse
import json
import os
import pickle
import re
import sys
import time
from collections import defaultdict

import cv2
import numpy as np
import scipy
from scipy.spatial import KDTree

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


PATCH_NAME_PATTERN = re.compile(r"^(?:region_)?(?P<region>\d+)_patch_(?P<patch>\d+)_(?P<x0>-?\d+)_(?P<y0>-?\d+)\.png$")


def parse_patch_name(file_name):
    match = PATCH_NAME_PATTERN.match(file_name)
    if match is None:
        return None
    return {
        "region_id": int(match.group("region")),
        "patch_id": int(match.group("patch")),
        "x0": int(match.group("x0")),
        "y0": int(match.group("y0")),
    }


def polyline_from_segmentation(segmentation):
    if not isinstance(segmentation, list):
        return []
    polylines = []
    for item in segmentation:
        if not isinstance(item, list):
            continue
        if len(item) < 4 or len(item) % 2 != 0:
            continue
        xy = np.asarray(item, dtype=np.float32).reshape(-1, 2)
        if xy.shape[0] >= 2:
            polylines.append(xy)
    return polylines


def build_region_graph(coco, score_threshold, category_id=None,
                       crop_size_orig=128, patch_size_model=256, stride=64,
                       core_keep_mode="midpoint",
                       core_margin_scale=1.0,
                       core_buffer_px=0.0):
    """
    从 COCO 预测结果构建每个 region 的全局图。

    关键处理：
    1. 坐标缩放：模型坐标(patch_size_model) -> 原始裁剪坐标(crop_size_orig)
    2. 重叠区过滤：支持中点判定或“端点/中点任一满足”，减少边界断边
    """
    coord_scale = float(crop_size_orig) / float(patch_size_model)
    overlap = crop_size_orig - stride  # 相邻 patch 的重叠像素
    margin = overlap / 2.0 * float(core_margin_scale)  # 核心区每侧收缩量

    image_meta = {}
    region_patch_offsets = defaultdict(list)  # 用于判断边界 patch
    for image in coco.get("images", []):
        image_id = image.get("id")
        file_name = image.get("file_name", "")
        parsed = parse_patch_name(file_name)
        if parsed is None:
            raise ValueError(f"Invalid road patch filename: {file_name}")
        image_meta[image_id] = parsed
        region_patch_offsets[parsed["region_id"]].append((parsed["x0"], parsed["y0"]))

    # 预计算每个 region 的全局范围，用于判断边界 patch
    region_bounds = {}
    for rid, offsets in region_patch_offsets.items():
        xs = [o[0] for o in offsets]
        ys = [o[1] for o in offsets]
        region_bounds[rid] = {
            "x_min": min(xs), "x_max": max(xs),
            "y_min": min(ys), "y_max": max(ys),
        }

    region_nodes = defaultdict(list)
    region_edges = defaultdict(list)
    for meta in image_meta.values():
        region_nodes[meta["region_id"]] = []
        region_edges[meta["region_id"]] = []

    for ann in coco.get("annotations", []):
        if ann.get("image_id") not in image_meta:
            continue
        if category_id is not None and ann.get("category_id") != category_id:
            continue
        if ann.get("score", 1.0) < score_threshold:
            continue

        meta = image_meta[ann["image_id"]]
        region_id = meta["region_id"]
        ox, oy = meta["x0"], meta["y0"]
        bounds = region_bounds[region_id]

        # 核心区范围（在原始裁剪局部坐标系中）
        # 边界 patch 在对应方向不收缩
        core_x0 = 0.0 if ox == bounds["x_min"] else margin
        core_y0 = 0.0 if oy == bounds["y_min"] else margin
        core_x1 = float(crop_size_orig) if ox == bounds["x_max"] else float(crop_size_orig) - margin
        core_y1 = float(crop_size_orig) if oy == bounds["y_max"] else float(crop_size_orig) - margin
        if core_buffer_px > 0:
            core_x0 = max(0.0, core_x0 - core_buffer_px)
            core_y0 = max(0.0, core_y0 - core_buffer_px)
            core_x1 = min(float(crop_size_orig), core_x1 + core_buffer_px)
            core_y1 = min(float(crop_size_orig), core_y1 + core_buffer_px)

        polylines = polyline_from_segmentation(ann.get("segmentation", []))
        for polyline in polylines:
            # polyline: [K, 2]，模型坐标系 (x, y)
            # 缩放到原始裁剪坐标
            local_xy = polyline * coord_scale  # [K, 2] in crop_size_orig space

            # 逐段判断是否位于核心区：
            # midpoint: 中点在核心区
            # endpoint_or_midpoint: 中点在核心区，或任一端点在核心区
            kept_segments = []
            for i in range(local_xy.shape[0] - 1):
                x0, y0 = local_xy[i, 0], local_xy[i, 1]
                x1, y1 = local_xy[i + 1, 0], local_xy[i + 1, 1]
                mid_x = (local_xy[i, 0] + local_xy[i + 1, 0]) / 2.0
                mid_y = (local_xy[i, 1] + local_xy[i + 1, 1]) / 2.0
                mid_in = (core_x0 <= mid_x <= core_x1 and core_y0 <= mid_y <= core_y1)
                ep0_in = (core_x0 <= x0 <= core_x1 and core_y0 <= y0 <= core_y1)
                ep1_in = (core_x0 <= x1 <= core_x1 and core_y0 <= y1 <= core_y1)
                if core_keep_mode == "endpoint_or_midpoint":
                    keep = mid_in or ep0_in or ep1_in
                else:
                    keep = mid_in
                if keep:
                    kept_segments.append(i)

            if not kept_segments:
                continue

            # 收集保留线段涉及的点，并转换到全局坐标 (row, col)
            kept_point_local_indices = set()
            for seg_i in kept_segments:
                kept_point_local_indices.add(seg_i)
                kept_point_local_indices.add(seg_i + 1)

            local_to_graph = {}  # polyline 局部索引 -> region 全局点索引
            for li in sorted(kept_point_local_indices):
                lx, ly = local_xy[li]
                global_r = ly + oy  # row = y + offset_y
                global_c = lx + ox  # col = x + offset_x
                gi = len(region_nodes[region_id])
                region_nodes[region_id].append([global_r, global_c])
                local_to_graph[li] = gi

            for seg_i in kept_segments:
                u = local_to_graph[seg_i]
                v = local_to_graph[seg_i + 1]
                region_edges[region_id].append((u, v))

    result = {}
    for region_id in region_nodes.keys():
        nodes = np.asarray(region_nodes[region_id], dtype=np.float32)
        edges = np.asarray(region_edges[region_id], dtype=np.int32)
        if edges.size == 0:
            edges = np.zeros((0, 2), dtype=np.int32)
        result[region_id] = (nodes, edges)
    return result


def deduplicate_undirected_edges(edges):
    if edges.shape[0] == 0:
        return edges
    undirected = np.sort(edges, axis=1)
    keep = undirected[:, 0] != undirected[:, 1]
    undirected = undirected[keep]
    if undirected.shape[0] == 0:
        return np.zeros((0, 2), dtype=np.int32)
    undirected = np.unique(undirected, axis=0)
    return undirected.astype(np.int32)


def union_find(num_items):
    parent = np.arange(num_items, dtype=np.int32)

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    return parent, find, union


def merge_close_nodes(nodes, edges, merge_node_dist):
    if nodes.shape[0] == 0:
        return nodes, np.zeros((0, 2), dtype=np.int32)
    if edges.shape[0] == 0:
        return nodes, edges

    kdtree = KDTree(nodes)
    pairs = kdtree.query_pairs(r=merge_node_dist)
    parent, find, union = union_find(nodes.shape[0])

    for i, j in pairs:
        union(int(i), int(j))

    groups = defaultdict(list)
    for idx in range(nodes.shape[0]):
        groups[find(idx)].append(idx)

    root_to_new = {}
    new_nodes = []
    old_to_new = np.zeros((nodes.shape[0],), dtype=np.int32)

    for root, members in groups.items():
        new_idx = len(new_nodes)
        root_to_new[root] = new_idx
        member_points = nodes[np.asarray(members, dtype=np.int32)]
        centroid = member_points.mean(axis=0)
        new_nodes.append(centroid)
        for m in members:
            old_to_new[m] = new_idx

    new_nodes = np.asarray(new_nodes, dtype=np.float32)
    remapped_edges = np.stack([old_to_new[edges[:, 0]], old_to_new[edges[:, 1]]], axis=1)
    remapped_edges = deduplicate_undirected_edges(remapped_edges)
    return new_nodes, remapped_edges


def convert_to_sat2graph_format(nodes, edges):
    neighbors = defaultdict(set)
    for u, v in edges:
        u, v = int(u), int(v)
        if u == v:
            continue
        neighbors[u].add(v)
        neighbors[v].add(u)

    int_nodes = [(int(round(r)), int(round(c))) for r, c in nodes]
    result = {}
    for idx in range(len(int_nodes)):
        key = int_nodes[idx]
        nbs = [int_nodes[n] for n in sorted(neighbors.get(idx, set()))]
        result[key] = nbs
    return result


def component_labels(num_nodes, edges):
    parent, find, union = union_find(num_nodes)

    for u, v in edges:
        union(int(u), int(v))

    labels = np.array([find(i) for i in range(num_nodes)], dtype=np.int32)
    return labels


def compute_node_degree(num_nodes, edges):
    degree = np.zeros((num_nodes,), dtype=np.int32)
    for u, v in edges:
        degree[int(u)] += 1
        degree[int(v)] += 1
    return degree


def snap_dangling_endpoints(nodes, edges, snap_dist):
    if nodes.shape[0] == 0 or edges.shape[0] == 0:
        return edges

    degree = np.zeros((nodes.shape[0],), dtype=np.int32)
    for u, v in edges:
        degree[u] += 1
        degree[v] += 1

    endpoint_idx = np.where(degree == 1)[0]
    if endpoint_idx.shape[0] < 2:
        return edges

    labels = component_labels(nodes.shape[0], edges)
    endpoint_points = nodes[endpoint_idx]
    kdtree = KDTree(endpoint_points)

    edge_set = {tuple(sorted((int(u), int(v)))) for u, v in edges}
    added = []

    for i, node_i in enumerate(endpoint_idx):
        near_local = kdtree.query_ball_point(endpoint_points[i], r=snap_dist)
        for j in near_local:
            if j <= i:
                continue
            node_j = endpoint_idx[j]
            if labels[node_i] == labels[node_j]:
                continue
            key = tuple(sorted((int(node_i), int(node_j))))
            if key in edge_set:
                continue
            edge_set.add(key)
            added.append(key)

    if not added:
        return edges

    add_edges = np.asarray(added, dtype=np.int32)
    out = np.concatenate([edges, add_edges], axis=0)
    return out


def _normalize_vec(vec):
    norm = float(np.linalg.norm(vec))
    if norm <= 1e-6:
        return None
    return vec / norm


def bridge_endpoints_by_extension(nodes, edges, extension_dist=12.0, min_cos=0.45):
    """
    端点延长线桥接：
    - 仅考虑度为 1 的端点；
    - 两端点必须距离接近、且朝向彼此（基于端点切向）；
    - 仅连接不同连通分量，避免在同一子图内制造短路。
    """
    if nodes.shape[0] == 0 or edges.shape[0] == 0 or extension_dist <= 0:
        return edges

    num_nodes = nodes.shape[0]
    degree = compute_node_degree(num_nodes, edges)
    endpoint_idx = np.where(degree == 1)[0]
    if endpoint_idx.shape[0] < 2:
        return edges

    neighbors = defaultdict(list)
    for u, v in edges:
        u = int(u)
        v = int(v)
        neighbors[u].append(v)
        neighbors[v].append(u)

    endpoint_dirs = {}
    for idx in endpoint_idx:
        nb = neighbors[int(idx)][0]
        # 外延方向：从邻点指向端点
        direction = _normalize_vec(nodes[int(idx)] - nodes[int(nb)])
        if direction is not None:
            endpoint_dirs[int(idx)] = direction

    labels = component_labels(num_nodes, edges)
    endpoint_points = nodes[endpoint_idx]
    kdtree = KDTree(endpoint_points)
    edge_set = {tuple(sorted((int(u), int(v)))) for u, v in edges}
    added = []

    for local_i, node_i in enumerate(endpoint_idx):
        node_i = int(node_i)
        if node_i not in endpoint_dirs:
            continue
        near_local = kdtree.query_ball_point(endpoint_points[local_i], r=extension_dist)
        for local_j in near_local:
            if local_j <= local_i:
                continue
            node_j = int(endpoint_idx[local_j])
            if node_j not in endpoint_dirs:
                continue
            if labels[node_i] == labels[node_j]:
                continue

            vec_ij = nodes[node_j] - nodes[node_i]
            dist = float(np.linalg.norm(vec_ij))
            if dist <= 1e-6:
                continue
            dir_ij = vec_ij / dist
            dir_ji = -dir_ij

            cos_i = float(np.dot(endpoint_dirs[node_i], dir_ij))
            cos_j = float(np.dot(endpoint_dirs[node_j], dir_ji))
            if cos_i < min_cos or cos_j < min_cos:
                continue

            key = tuple(sorted((node_i, node_j)))
            if key in edge_set:
                continue
            edge_set.add(key)
            added.append(key)

    if not added:
        return edges
    add_edges = np.asarray(added, dtype=np.int32)
    out = np.concatenate([edges, add_edges], axis=0)
    return deduplicate_undirected_edges(out)


def bridge_endpoints_to_near_perpendicular_segments(
    nodes,
    edges,
    bridge_dist=6.0,
    max_abs_cos=0.35,
    min_proj_t=0.15,
    min_dir_cos=0.10,
):
    """
    端点 -> 近垂直线段 的 T 型桥接：
    - 端点（度=1）到某条线段距离较近；
    - 端点方向与该线段近似垂直（|cos| 小）；
    - 投影点位于线段内部（避免贴近端点误连接）；
    - 端点外延方向大致指向投影点。

    命中后会在目标线段上插入一个新节点（投影点），并将端点连到该新节点。
    """
    if nodes.shape[0] == 0 or edges.shape[0] == 0 or bridge_dist <= 0:
        return nodes, edges

    nodes_work = np.asarray(nodes, dtype=np.float32)
    edge_list = [tuple(sorted((int(u), int(v)))) for u, v in edges]
    edge_list = list(dict.fromkeys(edge_list))  # 保序去重
    edge_set = set(edge_list)

    neighbors = defaultdict(list)
    for u, v in edge_list:
        neighbors[u].append(v)
        neighbors[v].append(u)

    degree = compute_node_degree(nodes_work.shape[0], np.asarray(edge_list, dtype=np.int32))
    endpoint_idx = np.where(degree == 1)[0]
    if endpoint_idx.shape[0] == 0:
        return nodes_work, np.asarray(edge_list, dtype=np.int32)

    endpoint_dirs = {}
    for ep in endpoint_idx:
        ep = int(ep)
        nb = neighbors[ep][0]
        d = _normalize_vec(nodes_work[ep] - nodes_work[nb])
        if d is not None:
            endpoint_dirs[ep] = d

    used_endpoints = set()
    # 复制一份，避免迭代时长度变化影响当前轮候选
    base_edges = list(edge_list)

    for ep in endpoint_idx:
        ep = int(ep)
        if ep in used_endpoints or ep not in endpoint_dirs:
            continue

        ep_pt = nodes_work[ep]
        ep_dir = endpoint_dirs[ep]
        best = None

        for ei, (u, v) in enumerate(base_edges):
            if ep == u or ep == v:
                continue
            a = nodes_work[u]
            b = nodes_work[v]
            ab = b - a
            ab_len2 = float(np.dot(ab, ab))
            if ab_len2 <= 1e-6:
                continue
            ab_len = float(np.sqrt(ab_len2))
            seg_dir = ab / ab_len

            # 近垂直约束
            abs_cos = abs(float(np.dot(ep_dir, seg_dir)))
            if abs_cos > max_abs_cos:
                continue

            # 投影与距离
            t = float(np.dot(ep_pt - a, ab) / ab_len2)
            if t < min_proj_t or t > (1.0 - min_proj_t):
                continue
            proj = a + t * ab
            d = float(np.linalg.norm(ep_pt - proj))
            if d > bridge_dist:
                continue

            to_proj = _normalize_vec(proj - ep_pt)
            if to_proj is None:
                continue
            dir_cos = float(np.dot(ep_dir, to_proj))
            if dir_cos < min_dir_cos:
                continue

            # 优先近距离，再偏好更垂直（abs_cos 更小）
            rank = (d, abs_cos)
            if best is None or rank < best[0]:
                best = (rank, ei, u, v, proj)

        if best is None:
            continue

        _, _, u, v, proj = best
        key_uv = tuple(sorted((u, v)))
        if key_uv not in edge_set:
            continue

        # 在线段上插入投影点，替换原边，并连接端点
        new_idx = int(nodes_work.shape[0])
        nodes_work = np.vstack([nodes_work, proj.astype(np.float32)])

        edge_set.remove(key_uv)
        edge_set.add(tuple(sorted((u, new_idx))))
        edge_set.add(tuple(sorted((new_idx, v))))
        edge_set.add(tuple(sorted((ep, new_idx))))
        used_endpoints.add(ep)

    out_edges = np.asarray(sorted(edge_set), dtype=np.int32) if edge_set else np.zeros((0, 2), dtype=np.int32)
    out_edges = deduplicate_undirected_edges(out_edges)
    return nodes_work.astype(np.float32), out_edges


def _point_to_segment_distance(pt, a, b):
    ab = b - a
    denom = float(np.dot(ab, ab))
    if denom <= 1e-6:
        return float(np.linalg.norm(pt - a))
    t = float(np.dot(pt - a, ab) / denom)
    t = max(0.0, min(1.0, t))
    proj = a + t * ab
    return float(np.linalg.norm(pt - proj))


def prune_redundant_parallel_edges(nodes, edges,
                                   parallel_dist=2.5,
                                   angle_deg=12.0,
                                   overlap_ratio=0.55):
    """
    删除重叠/平行冗余边（保守策略）：
    - 候选边中点接近；
    - 方向近似平行；
    - 两边彼此的端点到对方线段的最大距离较小；
    - 投影重叠比例足够高；
    - 只在双方端点度都 <=2 时执行（尽量避免破坏路口结构）。
    """
    if nodes.shape[0] == 0 or edges.shape[0] == 0 or parallel_dist <= 0:
        return edges
    if edges.shape[0] < 2:
        return edges

    edges = deduplicate_undirected_edges(edges)
    num_nodes = nodes.shape[0]
    degree = compute_node_degree(num_nodes, edges)

    seg_a = nodes[edges[:, 0]]
    seg_b = nodes[edges[:, 1]]
    vec = seg_b - seg_a
    length = np.linalg.norm(vec, axis=1)
    valid = length > 1e-6
    if not np.any(valid):
        return edges

    mid = (seg_a + seg_b) * 0.5
    kdtree = KDTree(mid)
    near_pairs = kdtree.query_pairs(r=max(parallel_dist * 2.0, 1.0))
    if not near_pairs:
        return edges

    cos_th = float(np.cos(np.deg2rad(angle_deg)))
    remove_idx = set()

    for i, j in near_pairs:
        if i in remove_idx or j in remove_idx:
            continue

        ui, vi = int(edges[i, 0]), int(edges[i, 1])
        uj, vj = int(edges[j, 0]), int(edges[j, 1])
        if len({ui, vi, uj, vj}) < 4:
            continue
        if (degree[ui] > 2 or degree[vi] > 2 or degree[uj] > 2 or degree[vj] > 2):
            continue
        if not valid[i] or not valid[j]:
            continue

        dir_i = vec[i] / length[i]
        dir_j = vec[j] / length[j]
        if abs(float(np.dot(dir_i, dir_j))) < cos_th:
            continue

        # 对称端点到线段距离，控制“重叠或非常近的平行”
        d_candidates = [
            _point_to_segment_distance(seg_a[i], seg_a[j], seg_b[j]),
            _point_to_segment_distance(seg_b[i], seg_a[j], seg_b[j]),
            _point_to_segment_distance(seg_a[j], seg_a[i], seg_b[i]),
            _point_to_segment_distance(seg_b[j], seg_a[i], seg_b[i]),
        ]
        if max(d_candidates) > parallel_dist:
            continue

        # 投影重叠比例（沿较长边方向）
        if length[i] >= length[j]:
            base_idx, other_idx = i, j
        else:
            base_idx, other_idx = j, i
        base_dir = vec[base_idx] / length[base_idx]
        o0 = float(np.dot(seg_a[other_idx] - seg_a[base_idx], base_dir))
        o1 = float(np.dot(seg_b[other_idx] - seg_a[base_idx], base_dir))
        lo = max(0.0, min(o0, o1))
        hi = min(length[base_idx], max(o0, o1))
        overlap_len = max(0.0, hi - lo)
        min_len = min(length[i], length[j])
        if min_len <= 1e-6 or (overlap_len / min_len) < overlap_ratio:
            continue

        # 删除更短边；若长度接近，删平均偏离更大的那条
        if abs(length[i] - length[j]) > 1e-4:
            drop = i if length[i] < length[j] else j
        else:
            d_i = _point_to_segment_distance(mid[i], seg_a[j], seg_b[j])
            d_j = _point_to_segment_distance(mid[j], seg_a[i], seg_b[i])
            drop = i if d_i >= d_j else j
        remove_idx.add(drop)

    if not remove_idx:
        return edges

    keep_mask = np.ones((edges.shape[0],), dtype=bool)
    keep_mask[list(remove_idx)] = False
    out = edges[keep_mask]
    return deduplicate_undirected_edges(out.astype(np.int32))


def merge_region_graph(nodes, edges, merge_node_dist, split_edge_dist, endpoint_snap_dist):
    if nodes.shape[0] == 0 or edges.shape[0] == 0:
        return nodes, np.zeros((0, 2), dtype=np.int32)

    nodes_m, edges_m = merge_close_nodes(nodes, edges, merge_node_dist)
    edges_m = deduplicate_undirected_edges(np.asarray(edges_m, dtype=np.int32))

    if endpoint_snap_dist > 0:
        edges_m = snap_dangling_endpoints(nodes_m, edges_m, endpoint_snap_dist)
        nodes_m, edges_m = merge_close_nodes(nodes_m, edges_m, merge_node_dist)
        edges_m = deduplicate_undirected_edges(np.asarray(edges_m, dtype=np.int32))

    return nodes_m, edges_m


def bridge_endpoints_near_patch_boundaries(nodes, edges, patch_metas, crop_size_orig,
                                           near_boundary_px=6.0, bridge_dist=8.0):
    """
    在 patch 边界附近做一次专门桥接：
    - 仅考虑度为 1 的端点
    - 端点需靠近某条 patch 边界线（x=x0/x1 或 y=y0/y1）
    - 在距离阈值内、且不在同一连通分量时补边
    """
    if nodes.shape[0] == 0 or edges.shape[0] == 0 or bridge_dist <= 0:
        return edges
    if not patch_metas:
        return edges

    x_lines = set()
    y_lines = set()
    for m in patch_metas:
        x0 = float(m["x0"])
        y0 = float(m["y0"])
        x_lines.add(x0)
        x_lines.add(x0 + float(crop_size_orig))
        y_lines.add(y0)
        y_lines.add(y0 + float(crop_size_orig))
    x_lines = np.asarray(sorted(x_lines), dtype=np.float32)
    y_lines = np.asarray(sorted(y_lines), dtype=np.float32)

    degree = np.zeros((nodes.shape[0],), dtype=np.int32)
    for u, v in edges:
        degree[int(u)] += 1
        degree[int(v)] += 1
    endpoint_idx = np.where(degree == 1)[0]
    if endpoint_idx.shape[0] < 2:
        return edges

    endpoint_pts = nodes[endpoint_idx]  # (row, col)
    endpoint_x = endpoint_pts[:, 1]
    endpoint_y = endpoint_pts[:, 0]

    near_x = np.zeros((endpoint_idx.shape[0],), dtype=bool)
    near_y = np.zeros((endpoint_idx.shape[0],), dtype=bool)
    if x_lines.size > 0:
        near_x = np.min(np.abs(endpoint_x[:, None] - x_lines[None, :]), axis=1) <= near_boundary_px
    if y_lines.size > 0:
        near_y = np.min(np.abs(endpoint_y[:, None] - y_lines[None, :]), axis=1) <= near_boundary_px
    boundary_mask = near_x | near_y
    boundary_local_idx = np.where(boundary_mask)[0]
    if boundary_local_idx.shape[0] < 2:
        return edges

    labels = component_labels(nodes.shape[0], edges)
    selected_endpoint_ids = endpoint_idx[boundary_local_idx]
    selected_pts = endpoint_pts[boundary_local_idx]
    kdtree = KDTree(selected_pts)

    edge_set = {tuple(sorted((int(u), int(v)))) for u, v in edges}
    added = []
    for i, node_i in enumerate(selected_endpoint_ids):
        nbrs = kdtree.query_ball_point(selected_pts[i], r=bridge_dist)
        for j in nbrs:
            if j <= i:
                continue
            node_j = selected_endpoint_ids[j]
            if labels[int(node_i)] == labels[int(node_j)]:
                continue
            key = tuple(sorted((int(node_i), int(node_j))))
            if key in edge_set:
                continue
            edge_set.add(key)
            added.append(key)

    if not added:
        return edges
    add_edges = np.asarray(added, dtype=np.int32)
    out = np.concatenate([edges, add_edges], axis=0)
    return deduplicate_undirected_edges(out)


def build_region_patch_meta(coco):
    """Extract per-region patch metadata from COCO images."""
    region_metas = defaultdict(list)
    for image in coco.get("images", []):
        file_name = image.get("file_name", "")
        parsed = parse_patch_name(file_name)
        if parsed is None:
            continue
        region_metas[parsed["region_id"]].append({
            "file_name": file_name,
            "x0": parsed["x0"],
            "y0": parsed["y0"],
        })
    return region_metas


def stitch_patch_canvas(region_id, region_metas, patch_image_dir, crop_size_orig):
    """Stitch all patches of a region into one BGR canvas."""
    metas = region_metas.get(region_id, [])
    if not metas:
        return None

    t0 = time.perf_counter()
    max_x = max(m["x0"] + crop_size_orig for m in metas)
    max_y = max(m["y0"] + crop_size_orig for m in metas)
    canvas = np.zeros((max_y, max_x, 3), dtype=np.uint8)
    loaded = 0

    for m in metas:
        img_path = os.path.join(patch_image_dir, m["file_name"])
        patch = cv2.imread(img_path)
        if patch is None:
            continue
        loaded += 1
        # patch 可能是模型尺寸(256)，缩放回原始裁剪尺寸(128)
        if patch.shape[0] != crop_size_orig or patch.shape[1] != crop_size_orig:
            patch = cv2.resize(patch, (crop_size_orig, crop_size_orig),
                               interpolation=cv2.INTER_LINEAR)
        h, w = patch.shape[:2]
        x0, y0 = m["x0"], m["y0"]
        x1, y1 = x0 + w, y0 + h
        if y1 <= canvas.shape[0] and x1 <= canvas.shape[1]:
            canvas[y0:y1, x0:x1] = patch
    dt = time.perf_counter() - t0
    print(
        f"[viz-bg] region={region_id} patch_read={loaded}/{len(metas)} "
        f"canvas=({max_y},{max_x}) "
        f"time={dt:.2f}s"
    )
    return canvas


def load_region_background(region_id, nodes, region_metas,
                           patch_image_dir, crop_size_orig,
                           image_root, image_pattern):
    bgr_canvas = None

    if patch_image_dir and os.path.isdir(patch_image_dir):
        bgr_canvas = stitch_patch_canvas(
            region_id, region_metas, patch_image_dir, crop_size_orig)

    if bgr_canvas is None:
        big_path = os.path.join(image_root, image_pattern.format(region_id))
        if os.path.exists(big_path):
            bgr_canvas = cv2.imread(big_path)

    if bgr_canvas is None:
        if nodes.shape[0] == 0:
            img_size = 512
        else:
            max_rc = int(np.ceil(np.max(nodes))) + 1
            img_size = max(512, max_rc)
        bgr_canvas = np.zeros((img_size, img_size, 3), dtype=np.uint8)

    return bgr_canvas


def create_viz_image(nodes, edges, bgr_canvas):
    bgr_canvas = bgr_canvas.copy()
    if nodes.shape[0] == 0:
        return bgr_canvas

    # nodes 是 (row, col), 转 (x, y) 画线画点
    xy_nodes = nodes[:, ::-1]  # (r,c) -> (x,y)

    # 画边
    for u, v in edges:
        p0 = xy_nodes[int(u)]
        p1 = xy_nodes[int(v)]
        cv2.line(bgr_canvas,
                 (int(round(p0[0])), int(round(p0[1]))),
                 (int(round(p1[0])), int(round(p1[1]))),
                 (15, 160, 253), 2, cv2.LINE_AA)

    # 画点
    for pt in xy_nodes:
        cv2.circle(bgr_canvas,
                   (int(round(pt[0])), int(round(pt[1]))),
                   3, (0, 255, 255), -1, cv2.LINE_AA)

    return bgr_canvas


def build_edge_key_set(nodes, edges):
    xy_nodes = nodes[:, ::-1] if nodes.shape[0] > 0 else np.zeros((0, 2), dtype=np.float32)
    key_set = set()
    for u, v in edges:
        p0 = xy_nodes[int(u)]
        p1 = xy_nodes[int(v)]
        a = (int(round(p0[0])), int(round(p0[1])))
        b = (int(round(p1[0])), int(round(p1[1])))
        if a <= b:
            key = (a, b)
        else:
            key = (b, a)
        key_set.add(key)
    return key_set


def create_diff_image(raw_nodes, raw_edges, nodes, edges, bgr_canvas):
    diff_img = bgr_canvas.copy()
    before_set = build_edge_key_set(raw_nodes, raw_edges)
    after_set = build_edge_key_set(nodes, edges)
    removed = before_set - after_set
    added = after_set - before_set

    # 新增边：绿色，删除边：红色
    for (p0, p1) in added:
        cv2.line(diff_img, p0, p1, (0, 220, 0), 2, cv2.LINE_AA)
    for (p0, p1) in removed:
        cv2.line(diff_img, p0, p1, (0, 0, 255), 2, cv2.LINE_AA)

    return diff_img, len(added), len(removed)


def save_image_unicode_safe(save_path, bgr_img):
    ok, encoded = cv2.imencode(".png", bgr_img)
    if not ok:
        raise RuntimeError(f"Failed to encode image: {save_path}")
    encoded.tofile(save_path)


def print_progress_bar(done, total, start_time, width=28):
    if total <= 0:
        return
    ratio = min(max(done / float(total), 0.0), 1.0)
    filled = int(round(width * ratio))
    bar = "#" * filled + "-" * (width - filled)
    elapsed = time.perf_counter() - start_time
    avg = elapsed / done if done > 0 else 0.0
    eta = avg * (total - done)
    msg = (
        f"\r[progress] |{bar}| {done}/{total} "
        f"({ratio * 100:5.1f}%) elapsed={elapsed:6.1f}s eta={eta:6.1f}s"
    )
    print(msg, end="", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_json",
        required=True,
        help="Full COCO road prediction JSON produced by step 1.",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Directory for stitched graph outputs.",
    )
    parser.add_argument("--score_threshold", type=float, default=0.0)
    parser.add_argument("--category_id", type=int, default=None)
    parser.add_argument("--merge_node_dist", type=float, default=5.0)
    parser.add_argument("--split_edge_dist", type=float, default=3.0)
    parser.add_argument("--endpoint_snap_dist", type=float, default=10.0)
    parser.add_argument("--single_region", type=int, default=None,
                        help="仅处理指定 region；None 表示处理全部。")
    parser.add_argument("--crop_size_orig", type=int, default=128,
                        help="原始裁剪尺寸 (px)")
    parser.add_argument("--patch_size_model", type=int, default=256,
                        help="模型输入 patch 尺寸 (px)")
    parser.add_argument("--stride", type=int, default=64,
                        help="裁剪步长 (px)")
    parser.add_argument("--core_keep_mode", type=str, default="endpoint_or_midpoint",
                        choices=["midpoint", "endpoint_or_midpoint"],
                        help="核心区过滤模式：midpoint 或 endpoint_or_midpoint")
    parser.add_argument("--core_margin_scale", type=float, default=0.5,
                        help="核心区收缩比例，1.0 为原策略，0.5 为 margin 减半")
    parser.add_argument("--core_buffer_px", type=float, default=2.0,
                        help="核心区额外缓冲像素（在 core_keep_mode 下扩展边界）")
    parser.add_argument("--patch_boundary_bridge_dist", type=float, default=4.0,
                        help="patch 边界附近端点桥接距离，<=0 表示关闭")
    parser.add_argument("--patch_boundary_near_px", type=float, default=4.0,
                        help="判定“靠近 patch 边界”的像素阈值")
    parser.add_argument("--endpoint_extension_dist", type=float, default=12.0,
                        help="端点延长线连接的候选距离，<=0 表示关闭")
    parser.add_argument("--endpoint_extension_cos", type=float, default=0.45,
                        help="端点延长线方向一致性阈值(cos)，越大越严格")
    parser.add_argument("--parallel_prune_dist", type=float, default=2.5,
                        help="平行/重叠冗余边去除距离阈值，<=0 表示关闭")
    parser.add_argument("--parallel_prune_angle_deg", type=float, default=12.0,
                        help="判定平行的最大夹角（度）")
    parser.add_argument("--parallel_overlap_ratio", type=float, default=0.55,
                        help="判定重叠冗余的最小投影重叠比例")
    parser.add_argument("--perp_bridge_dist", type=float, default=6.0,
                        help="端点到近垂直线段的桥接距离，<=0 表示关闭")
    parser.add_argument("--perp_bridge_max_abs_cos", type=float, default=0.35,
                        help="近垂直判定阈值 |cos|，越小越严格")
    parser.add_argument("--perp_bridge_min_t", type=float, default=0.15,
                        help="投影点在线段内部的最小相对位置阈值")
    parser.add_argument("--enable_viz", type=int, choices=[0, 1], default=0)
    parser.add_argument(
        "--patch_image_dir",
        default=None,
        help="Patch image directory used to stitch background.",
    )
    parser.add_argument(
        "--image_root",
        default=None,
        help="Fallback directory for region-level satellite images.",
    )
    parser.add_argument("--image_pattern", default="region_{}_sat.png")
    args = parser.parse_args()

    if bool(args.enable_viz) and not (args.patch_image_dir or args.image_root):
        parser.error("--enable_viz 1 requires --patch_image_dir and/or --image_root")

    with open(args.input_json, "r", encoding="utf-8") as f:
        coco = json.load(f)

    region_graphs = build_region_graph(
        coco,
        score_threshold=args.score_threshold,
        category_id=args.category_id,
        crop_size_orig=args.crop_size_orig,
        patch_size_model=args.patch_size_model,
        stride=args.stride,
        core_keep_mode=args.core_keep_mode,
        core_margin_scale=args.core_margin_scale,
        core_buffer_px=args.core_buffer_px,
    )

    # 构建每个 region 的 patch 元信息（用于拼接背景）
    region_metas = build_region_patch_meta(coco)

    os.makedirs(args.output_dir, exist_ok=True)
    graph_save_dir = os.path.join(args.output_dir, "graph")
    os.makedirs(graph_save_dir, exist_ok=True)
    viz_save_dir = os.path.join(args.output_dir, "viz")
    viz_before_dir = os.path.join(args.output_dir, "viz_before")
    viz_diff_dir = os.path.join(args.output_dir, "viz_diff")
    viz_compare_dir = os.path.join(args.output_dir, "viz_compare")
    if bool(args.enable_viz):
        os.makedirs(viz_save_dir, exist_ok=True)
        os.makedirs(viz_before_dir, exist_ok=True)
        os.makedirs(viz_diff_dir, exist_ok=True)
        os.makedirs(viz_compare_dir, exist_ok=True)

    summary = []
    selected_region_ids = [
        rid for rid in sorted(region_graphs.keys())
        if args.single_region is None or rid == args.single_region
    ]
    total_regions = len(selected_region_ids)
    processed_count = 0
    run_t0 = time.perf_counter()
    print_progress_bar(0, total_regions, run_t0)

    for region_id in selected_region_ids:
        processed_count += 1
        region_t0 = time.perf_counter()
        print(f"[region] start {processed_count}/{total_regions} region={region_id}")

        raw_nodes, raw_edges = region_graphs[region_id]

        # 合并后的图
        nodes, edges = merge_region_graph(
            raw_nodes,
            raw_edges,
            merge_node_dist=args.merge_node_dist,
            split_edge_dist=args.split_edge_dist,
            endpoint_snap_dist=args.endpoint_snap_dist,
        )
        if args.patch_boundary_bridge_dist > 0:
            edges = bridge_endpoints_near_patch_boundaries(
                nodes,
                edges,
                region_metas.get(region_id, []),
                crop_size_orig=args.crop_size_orig,
                near_boundary_px=args.patch_boundary_near_px,
                bridge_dist=args.patch_boundary_bridge_dist,
            )
            edges = deduplicate_undirected_edges(np.asarray(edges, dtype=np.int32))

        if args.endpoint_extension_dist > 0:
            edges = bridge_endpoints_by_extension(
                nodes,
                edges,
                extension_dist=args.endpoint_extension_dist,
                min_cos=args.endpoint_extension_cos,
            )
            nodes, edges = merge_close_nodes(nodes, edges, args.merge_node_dist)
            edges = deduplicate_undirected_edges(np.asarray(edges, dtype=np.int32))

        if args.perp_bridge_dist > 0:
            nodes, edges = bridge_endpoints_to_near_perpendicular_segments(
                nodes,
                edges,
                bridge_dist=args.perp_bridge_dist,
                max_abs_cos=args.perp_bridge_max_abs_cos,
                min_proj_t=args.perp_bridge_min_t,
            )
            nodes, edges = merge_close_nodes(nodes, edges, args.merge_node_dist)
            edges = deduplicate_undirected_edges(np.asarray(edges, dtype=np.int32))

        if args.parallel_prune_dist > 0:
            edges = prune_redundant_parallel_edges(
                nodes,
                edges,
                parallel_dist=args.parallel_prune_dist,
                angle_deg=args.parallel_prune_angle_deg,
                overlap_ratio=args.parallel_overlap_ratio,
            )
            edges = deduplicate_undirected_edges(np.asarray(edges, dtype=np.int32))

        sat2graph = convert_to_sat2graph_format(nodes, edges)
        output_path = os.path.join(graph_save_dir, f"{region_id}.p")
        with open(output_path, "wb") as f:
            pickle.dump(sat2graph, f)
        save_dt = time.perf_counter() - region_t0
        print(f"[region] graph_saved region={region_id} elapsed={save_dt:.2f}s")

        if bool(args.enable_viz):
            viz_t0 = time.perf_counter()
            viz_kwargs = dict(
                region_metas=region_metas,
                patch_image_dir=args.patch_image_dir,
                crop_size_orig=args.crop_size_orig,
                image_root=args.image_root,
                image_pattern=args.image_pattern,
            )
            background_canvas = load_region_background(
                region_id, nodes, **viz_kwargs)

            # 合并前
            viz_before = create_viz_image(
                raw_nodes, raw_edges, background_canvas)
            before_path = os.path.join(viz_before_dir, f"region_{region_id}.png")
            save_image_unicode_safe(before_path, viz_before)

            # 合并后
            viz_after = create_viz_image(
                nodes, edges, background_canvas)
            after_path = os.path.join(viz_save_dir, f"region_{region_id}.png")
            save_image_unicode_safe(after_path, viz_after)

            # 差异图
            viz_diff, add_n, del_n = create_diff_image(
                raw_nodes, raw_edges, nodes, edges, background_canvas)
            diff_path = os.path.join(viz_diff_dir, f"region_{region_id}.png")
            save_image_unicode_safe(diff_path, viz_diff)

            # 并排对比 (before | after | diff)
            h1, w1 = viz_before.shape[:2]
            h2, w2 = viz_after.shape[:2]
            h3, w3 = viz_diff.shape[:2]
            max_h = max(h1, h2, h3)
            # 统一高度
            if h1 != max_h:
                viz_before = cv2.resize(viz_before, (int(w1 * max_h / h1), max_h))
            if h2 != max_h:
                viz_after = cv2.resize(viz_after, (int(w2 * max_h / h2), max_h))
            if h3 != max_h:
                viz_diff = cv2.resize(viz_diff, (int(w3 * max_h / h3), max_h))
            # 添加标题
            cv2.putText(viz_before, "Before Merge", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.putText(viz_after, "After Merge", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(viz_diff, f"Diff +{add_n} -{del_n}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
            gap = np.full((max_h, 16, 3), 255, dtype=np.uint8)
            compare_img = np.concatenate([viz_before, gap, viz_after, gap, viz_diff], axis=1)
            compare_path = os.path.join(viz_compare_dir, f"region_{region_id}.png")
            save_image_unicode_safe(compare_path, compare_img)
            viz_dt = time.perf_counter() - viz_t0
            print(f"[region] viz_done region={region_id} elapsed={viz_dt:.2f}s")

        summary.append((
            region_id,
            int(raw_nodes.shape[0]), int(raw_edges.shape[0]),
            int(nodes.shape[0]), int(edges.shape[0]),
            output_path,
        ))
        total_dt = time.perf_counter() - region_t0
        print(f"[region] done region={region_id} total_elapsed={total_dt:.2f}s")
        print_progress_bar(processed_count, total_regions, run_t0)

    if total_regions > 0:
        print()
    print(f"Processed regions: {len(summary)}")
    for region_id, raw_n, raw_e, merge_n, merge_e, output_path in summary:
        print(
            f"region={region_id}  "
            f"before: nodes={raw_n}, edges={raw_e}  |  "
            f"after: nodes={merge_n}, edges={merge_e}  "
            f"saved={output_path}"
        )


if __name__ == "__main__":
    main()


