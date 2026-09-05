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

PARAM_PRESETS = {
    "balanced": {
        "merge_node_dist": 9.0,
        "endpoint_snap_dist": 7.0,
        "core_keep_mode": "endpoint_or_midpoint",
        "core_margin_scale": 0.50,
        "core_buffer_px": 2.0,
        "patch_boundary_bridge_dist": 3.5,
        "patch_boundary_near_px": 4.0,
        "consensus_filter": 1,
        "consensus_support_dist": 5.0,
        "consensus_angle_deg": 25.0,
        "consensus_sample_spacing": 8.0,
        "consensus_min_supported_fraction": 0.25,
        "consensus_long_keep_length": 55.0,
        "support_spur_filter": 0,
        "support_spur_length": 16.0,
        "support_spur_max_edges": 5,
        "support_spur_min_patches": 2,
        "support_spur_min_supported_fraction": 0.20,
        "spur_length": 7.0,
        "spur_max_edges": 3,
        "spur_rounds": 1,
        "min_component_edges": 1,
        "min_component_length": 8.0,
        "parallel_prune_dist": 1.0,
        "parallel_prune_angle_deg": 8.0,
        "parallel_overlap_ratio": 0.80,
        "junction_enable": 1,
        "junction_snap_node_dist": 5.0,
        "junction_split_edge_dist": 4.0,
        "junction_endpoint_bridge_dist": 0.0,
        "junction_min_patch_support": 2,
        "junction_cluster_dist": 5.0,
    },
    "precision": {
        "merge_node_dist": 7.0,
        "endpoint_snap_dist": 3.0,
        "core_keep_mode": "endpoint_or_midpoint",
        "core_margin_scale": 0.65,
        "core_buffer_px": 0.0,
        "patch_boundary_bridge_dist": 2.0,
        "patch_boundary_near_px": 3.5,
        "consensus_filter": 1,
        "consensus_support_dist": 4.5,
        "consensus_angle_deg": 22.0,
        "consensus_sample_spacing": 8.0,
        "consensus_min_supported_fraction": 0.35,
        "consensus_long_keep_length": 75.0,
        "support_spur_filter": 1,
        "support_spur_length": 22.0,
        "support_spur_max_edges": 7,
        "support_spur_min_patches": 2,
        "support_spur_min_supported_fraction": 0.30,
        "spur_length": 13.0,
        "spur_max_edges": 5,
        "spur_rounds": 2,
        "min_component_edges": 2,
        "min_component_length": 16.0,
        "parallel_prune_dist": 1.6,
        "parallel_prune_angle_deg": 10.0,
        "parallel_overlap_ratio": 0.72,
        "junction_enable": 1,
        "junction_snap_node_dist": 4.0,
        "junction_split_edge_dist": 3.5,
        "junction_endpoint_bridge_dist": 6.0,
        "junction_min_patch_support": 2,
        "junction_cluster_dist": 4.5,
    },
    "recall": {
        "merge_node_dist": 8.0,
        "endpoint_snap_dist": 8.0,
        "core_keep_mode": "endpoint_or_midpoint",
        "core_margin_scale": 0.50,
        "core_buffer_px": 2.0,
        "patch_boundary_bridge_dist": 4.0,
        "patch_boundary_near_px": 4.0,
        "consensus_filter": 0,
        "consensus_support_dist": 5.0,
        "consensus_angle_deg": 25.0,
        "consensus_sample_spacing": 8.0,
        "consensus_min_supported_fraction": 0.25,
        "consensus_long_keep_length": 55.0,
        "support_spur_filter": 0,
        "support_spur_length": 10.0,
        "support_spur_max_edges": 3,
        "support_spur_min_patches": 2,
        "support_spur_min_supported_fraction": 0.20,
        "spur_length": 8.0,
        "spur_max_edges": 3,
        "spur_rounds": 1,
        "min_component_edges": 1,
        "min_component_length": 8.0,
        "parallel_prune_dist": 1.0,
        "parallel_prune_angle_deg": 8.0,
        "parallel_overlap_ratio": 0.80,
        "junction_enable": 1,
        "junction_snap_node_dist": 6.0,
        "junction_split_edge_dist": 4.5,
        "junction_endpoint_bridge_dist": 10.0,
        "junction_min_patch_support": 1,
        "junction_cluster_dist": 5.5,
    },
}


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


def sample_segment_points(p0, p1, spacing):
    length = float(np.linalg.norm(p1 - p0))
    if length <= 1e-6:
        return [p0]
    steps = max(1, int(np.ceil(length / max(float(spacing), 1.0))))
    return [p0 + (p1 - p0) * (i / float(steps)) for i in range(steps + 1)]


def build_segment_support_index(ann_items, coord_scale, support_dist, sample_spacing):
    """
    Build a local spatial index of all raw overlapping-patch line evidence.

    The index is intentionally simple: sampled points are bucketed by grid cell,
    and each sample stores its patch id and segment direction. It lets us ask:
    "Does another patch predict a similarly oriented segment here?"
    """
    cell_size = max(float(support_dist), 2.0)
    index = defaultdict(list)

    for item in ann_items:
        ann = item["ann"]
        region_id = int(item["region_id"])
        patch_id = int(item["patch_id"])
        ox, oy = float(item["ox"]), float(item["oy"])

        for polyline in polyline_from_segmentation(ann.get("segmentation", [])):
            local_xy = polyline * coord_scale
            for i in range(local_xy.shape[0] - 1):
                lx0, ly0 = local_xy[i]
                lx1, ly1 = local_xy[i + 1]
                p0 = np.asarray([ly0 + oy, lx0 + ox], dtype=np.float32)
                p1 = np.asarray([ly1 + oy, lx1 + ox], dtype=np.float32)
                vec = p1 - p0
                norm = float(np.linalg.norm(vec))
                if norm <= 1e-6:
                    continue
                direction = vec / norm
                for pt in sample_segment_points(p0, p1, sample_spacing):
                    cr = int(np.floor(float(pt[0]) / cell_size))
                    cc = int(np.floor(float(pt[1]) / cell_size))
                    index[(region_id, cr, cc)].append((
                        float(pt[0]),
                        float(pt[1]),
                        float(direction[0]),
                        float(direction[1]),
                        patch_id,
                    ))

    return {
        "cell_size": cell_size,
        "index": index,
    }


def segment_has_overlap_support(item, local_xy, seg_i, support_index,
                                support_dist=5.0,
                                angle_deg=25.0,
                                sample_spacing=8.0,
                                min_supported_fraction=0.25,
                                long_keep_length=55.0):
    lx0, ly0 = local_xy[seg_i]
    lx1, ly1 = local_xy[seg_i + 1]
    p0 = np.asarray([ly0 + float(item["oy"]), lx0 + float(item["ox"])], dtype=np.float32)
    p1 = np.asarray([ly1 + float(item["oy"]), lx1 + float(item["ox"])], dtype=np.float32)
    vec = p1 - p0
    length = float(np.linalg.norm(vec))
    if length <= 1e-6:
        return False
    if long_keep_length > 0 and length >= long_keep_length:
        return True

    mid_x = (float(lx0) + float(lx1)) / 2.0
    mid_y = (float(ly0) + float(ly1)) / 2.0
    crop_size = float(item.get("crop_size_orig", 128))
    border_keep = min(8.0, crop_size * 0.08)
    on_outer_boundary = (
        (item["core_x0"] <= 0.0 and mid_x <= border_keep) or
        (item["core_y0"] <= 0.0 and mid_y <= border_keep) or
        (item["core_x1"] >= crop_size and mid_x >= crop_size - border_keep) or
        (item["core_y1"] >= crop_size and mid_y >= crop_size - border_keep)
    )
    if on_outer_boundary:
        return True

    direction = vec / length
    cos_th = float(np.cos(np.deg2rad(angle_deg)))
    support_dist2 = float(support_dist) * float(support_dist)
    cell_size = support_index["cell_size"]
    index = support_index["index"]
    region_id = int(item["region_id"])
    patch_id = int(item["patch_id"])

    points = sample_segment_points(p0, p1, sample_spacing)
    supported = 0
    for pt in points:
        cr = int(np.floor(float(pt[0]) / cell_size))
        cc = int(np.floor(float(pt[1]) / cell_size))
        found = False
        for rr in range(cr - 1, cr + 2):
            for cc2 in range(cc - 1, cc + 2):
                for sr, sc, dr, dc, other_patch_id in index.get((region_id, rr, cc2), []):
                    if int(other_patch_id) == patch_id:
                        continue
                    d2 = (float(pt[0]) - sr) ** 2 + (float(pt[1]) - sc) ** 2
                    if d2 > support_dist2:
                        continue
                    dot = abs(float(direction[0]) * dr + float(direction[1]) * dc)
                    if dot >= cos_th:
                        found = True
                        break
                if found:
                    break
            if found:
                break
        if found:
            supported += 1

    frac = supported / float(max(1, len(points)))
    return frac >= float(min_supported_fraction)


def build_region_support_index(coco, score_threshold, category_id=None,
                               crop_size_orig=128, patch_size_model=256,
                               support_dist=6.0, sample_spacing=8.0):
    coord_scale = float(crop_size_orig) / float(patch_size_model)

    image_meta = {}
    for image in coco.get("images", []):
        image_id = image.get("id")
        parsed = parse_patch_name(image.get("file_name", ""))
        if parsed is not None:
            image_meta[image_id] = parsed

    ann_items = []
    for ann in coco.get("annotations", []):
        if ann.get("image_id") not in image_meta:
            continue
        if category_id is not None and ann.get("category_id") != category_id:
            continue
        if ann.get("score", 1.0) < score_threshold:
            continue
        meta = image_meta[ann["image_id"]]
        ann_items.append({
            "ann": ann,
            "region_id": meta["region_id"],
            "patch_id": meta["patch_id"],
            "ox": meta["x0"],
            "oy": meta["y0"],
        })

    return build_segment_support_index(
        ann_items,
        coord_scale=coord_scale,
        support_dist=support_dist,
        sample_spacing=sample_spacing,
    )


def graph_segment_support_fraction(region_id, patch_id, p0, p1, support_index,
                                   support_dist=6.0,
                                   angle_deg=30.0,
                                   sample_spacing=8.0):
    vec = p1 - p0
    length = float(np.linalg.norm(vec))
    if length <= 1e-6:
        return 0.0, set()

    direction = vec / length
    cos_th = float(np.cos(np.deg2rad(angle_deg)))
    support_dist2 = float(support_dist) * float(support_dist)
    cell_size = support_index["cell_size"]
    index = support_index["index"]
    points = sample_segment_points(p0, p1, sample_spacing)
    supported = 0
    support_patches = set()

    for pt in points:
        cr = int(np.floor(float(pt[0]) / cell_size))
        cc = int(np.floor(float(pt[1]) / cell_size))
        found = False
        for rr in range(cr - 1, cr + 2):
            for cc2 in range(cc - 1, cc + 2):
                for sr, sc, dr, dc, other_patch_id in index.get((int(region_id), rr, cc2), []):
                    if patch_id is not None and int(other_patch_id) == int(patch_id):
                        continue
                    d2 = (float(pt[0]) - sr) ** 2 + (float(pt[1]) - sc) ** 2
                    if d2 > support_dist2:
                        continue
                    dot = abs(float(direction[0]) * dr + float(direction[1]) * dc)
                    if dot >= cos_th:
                        found = True
                        support_patches.add(int(other_patch_id))
                        break
                if found:
                    break
            if found:
                break
        if found:
            supported += 1

    return supported / float(max(1, len(points))), support_patches


def load_junction_points(junction_json, crop_size_orig=128, patch_size_model=256):
    """
    Load patch-local junction predictions and convert them to global (row, col)
    coordinates grouped by region.
    """
    if not junction_json:
        return defaultdict(list)
    with open(junction_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Junction json must be a list: {junction_json}")

    coord_scale = float(crop_size_orig) / float(patch_size_model)
    region_points = defaultdict(list)

    for item in data:
        if not isinstance(item, dict):
            continue
        parsed = parse_patch_name(item.get("file_name", ""))
        if parsed is None:
            continue
        patch_id = int(parsed["patch_id"])
        ox = float(parsed["x0"])
        oy = float(parsed["y0"])
        region_id = int(parsed["region_id"])
        for pt in item.get("junction", []) or []:
            if not isinstance(pt, (list, tuple)) or len(pt) < 2:
                continue
            x = float(pt[0]) * coord_scale + ox
            y = float(pt[1]) * coord_scale + oy
            region_points[region_id].append((y, x, patch_id))

    return region_points


def cluster_region_junctions(region_junctions, cluster_dist=5.0, min_patch_support=2):
    """
    Merge duplicated junctions from overlapping patches.

    Returns region -> ndarray[N, 2] in (row, col). A cluster is kept when it is
    observed by enough distinct patches; this filters many single-patch false
    junctions while preserving repeated overlap evidence.
    """
    clustered = {}
    for region_id, items in region_junctions.items():
        if not items:
            clustered[region_id] = np.zeros((0, 2), dtype=np.float32)
            continue
        pts = np.asarray([[r, c] for r, c, _ in items], dtype=np.float32)
        patch_ids = [int(p) for _, _, p in items]
        if pts.shape[0] == 1:
            if min_patch_support <= 1:
                clustered[region_id] = pts.astype(np.float32)
            else:
                clustered[region_id] = np.zeros((0, 2), dtype=np.float32)
            continue

        parent, find, union = union_find(pts.shape[0])
        tree = KDTree(pts)
        for i, j in tree.query_pairs(r=float(cluster_dist)):
            union(int(i), int(j))

        groups = defaultdict(list)
        for idx in range(pts.shape[0]):
            groups[find(idx)].append(idx)

        centers = []
        for members in groups.values():
            support = {patch_ids[i] for i in members}
            if len(support) < int(min_patch_support):
                continue
            centers.append(pts[np.asarray(members, dtype=np.int32)].mean(axis=0))

        if centers:
            clustered[region_id] = np.asarray(centers, dtype=np.float32)
        else:
            clustered[region_id] = np.zeros((0, 2), dtype=np.float32)
    return clustered


def _project_point_to_segment(pt, a, b):
    ab = b - a
    denom = float(np.dot(ab, ab))
    if denom <= 1e-6:
        return 0.0, a.copy(), float(np.linalg.norm(pt - a))
    t = float(np.dot(pt - a, ab) / denom)
    t_clamped = max(0.0, min(1.0, t))
    proj = a + t_clamped * ab
    return t_clamped, proj, float(np.linalg.norm(pt - proj))


def apply_junction_guidance(nodes, edges, junctions,
                            snap_node_dist=5.0,
                            split_edge_dist=4.0,
                            endpoint_bridge_dist=8.0):
    """
    Use junction predictions as graph anchors:
    1. Snap nearby graph nodes to the clustered junction position.
    2. Split edges that pass through a junction but do not have a node there.
    3. Bridge dangling endpoints to nearby junction nodes.
    """
    if nodes.shape[0] == 0 or edges.shape[0] == 0 or junctions is None or junctions.shape[0] == 0:
        return nodes, edges

    nodes_work = np.asarray(nodes, dtype=np.float32).copy()
    edges_work = deduplicate_undirected_edges(np.asarray(edges, dtype=np.int32))
    junctions = np.asarray(junctions, dtype=np.float32)

    # Step 1: snap existing graph nodes to reliable junction anchors.
    if snap_node_dist > 0 and nodes_work.shape[0] > 0:
        node_tree = KDTree(nodes_work)
        for jpt in junctions:
            idxs = node_tree.query_ball_point(jpt, r=float(snap_node_dist))
            if not idxs:
                continue
            idxs = [int(i) for i in idxs]
            # Move all very nearby nodes to one shared anchor. The later merge
            # pass will collapse duplicates safely.
            nodes_work[np.asarray(idxs, dtype=np.int32)] = jpt

    nodes_work, edges_work = merge_close_nodes(nodes_work, edges_work, max(1.0, float(snap_node_dist) * 0.6))
    edges_work = deduplicate_undirected_edges(edges_work)

    # Step 2: split edges at junctions that lie on them.
    if split_edge_dist > 0 and edges_work.shape[0] > 0:
        edge_list = [tuple(sorted((int(u), int(v)))) for u, v in edges_work]
        edge_set = set(edge_list)

        for jpt in junctions:
            best = None
            for u, v in list(edge_set):
                a = nodes_work[int(u)]
                b = nodes_work[int(v)]
                length = float(np.linalg.norm(b - a))
                if length <= max(2.0, split_edge_dist * 2.0):
                    continue
                t, proj, dist = _project_point_to_segment(jpt, a, b)
                if t <= 0.08 or t >= 0.92 or dist > split_edge_dist:
                    continue
                rank = (dist, abs(t - 0.5))
                if best is None or rank < best[0]:
                    best = (rank, u, v, proj)

            if best is None:
                continue
            _, u, v, proj = best
            key = tuple(sorted((int(u), int(v))))
            if key not in edge_set:
                continue
            new_idx = int(nodes_work.shape[0])
            nodes_work = np.vstack([nodes_work, proj.astype(np.float32)])
            edge_set.remove(key)
            edge_set.add(tuple(sorted((int(u), new_idx))))
            edge_set.add(tuple(sorted((new_idx, int(v)))))

        edges_work = np.asarray(sorted(edge_set), dtype=np.int32) if edge_set else np.zeros((0, 2), dtype=np.int32)
        nodes_work, edges_work = merge_close_nodes(nodes_work, edges_work, max(1.0, float(snap_node_dist) * 0.6))
        edges_work = deduplicate_undirected_edges(edges_work)

    # Step 3: bridge dangling endpoints into nearby junction nodes.
    if endpoint_bridge_dist > 0 and edges_work.shape[0] > 0:
        degree = compute_node_degree(nodes_work.shape[0], edges_work)
        endpoint_idx = np.where(degree == 1)[0]
        if endpoint_idx.shape[0] > 0:
            junction_tree = KDTree(junctions)
            nodes_list = nodes_work.tolist()
            edge_set = {tuple(sorted((int(u), int(v)))) for u, v in edges_work}
            labels = component_labels(nodes_work.shape[0], edges_work)

            for ep in endpoint_idx:
                ep = int(ep)
                d, j_local = junction_tree.query(nodes_work[ep])
                if not np.isfinite(d) or float(d) > float(endpoint_bridge_dist):
                    continue
                jpt = junctions[int(j_local)]

                # Prefer an existing node already snapped near this junction.
                near_nodes = np.where(np.linalg.norm(nodes_work - jpt, axis=1) <= max(1.5, snap_node_dist))[0]
                target = None
                for nid in near_nodes:
                    nid = int(nid)
                    if nid != ep:
                        target = nid
                        break
                if target is None:
                    target = len(nodes_list)
                    nodes_list.append([float(jpt[0]), float(jpt[1])])
                    labels = np.append(labels, -1)

                if labels[ep] == labels[target] and target < nodes_work.shape[0]:
                    continue
                key = tuple(sorted((ep, int(target))))
                if key not in edge_set:
                    edge_set.add(key)

            nodes_work = np.asarray(nodes_list, dtype=np.float32)
            edges_work = np.asarray(sorted(edge_set), dtype=np.int32) if edge_set else np.zeros((0, 2), dtype=np.int32)
            nodes_work, edges_work = merge_close_nodes(nodes_work, edges_work, max(1.0, float(snap_node_dist) * 0.6))
            edges_work = deduplicate_undirected_edges(edges_work)

    return compact_graph(nodes_work, edges_work)


def build_region_graph(coco, score_threshold, category_id=None,
                       crop_size_orig=128, patch_size_model=256, stride=64,
                       core_keep_mode="midpoint",
                       core_margin_scale=1.0,
                       core_buffer_px=0.0,
                       consensus_filter=False,
                       consensus_support_dist=5.0,
                       consensus_angle_deg=25.0,
                       consensus_sample_spacing=8.0,
                       consensus_min_supported_fraction=0.25,
                       consensus_long_keep_length=55.0):
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

    ann_items = []

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

        ann_items.append({
            "ann": ann,
            "region_id": region_id,
            "patch_id": meta["patch_id"],
            "ox": ox,
            "oy": oy,
            "crop_size_orig": crop_size_orig,
            "core_x0": core_x0,
            "core_y0": core_y0,
            "core_x1": core_x1,
            "core_y1": core_y1,
        })

    support_index = None
    if consensus_filter:
        support_index = build_segment_support_index(
            ann_items,
            coord_scale=coord_scale,
            support_dist=consensus_support_dist,
            sample_spacing=consensus_sample_spacing,
        )

    region_nodes = defaultdict(list)
    region_edges = defaultdict(list)
    for meta in image_meta.values():
        region_nodes[meta["region_id"]] = []
        region_edges[meta["region_id"]] = []

    for item in ann_items:
        ann = item["ann"]
        region_id = item["region_id"]
        ox, oy = item["ox"], item["oy"]
        core_x0 = item["core_x0"]
        core_y0 = item["core_y0"]
        core_x1 = item["core_x1"]
        core_y1 = item["core_y1"]

        polylines = polyline_from_segmentation(ann.get("segmentation", []))
        for polyline in polylines:
            local_xy = polyline * coord_scale

            kept_segments = []
            for i in range(local_xy.shape[0] - 1):
                x0, y0 = local_xy[i, 0], local_xy[i, 1]
                x1, y1 = local_xy[i + 1, 0], local_xy[i + 1, 1]
                mid_x = (x0 + x1) / 2.0
                mid_y = (y0 + y1) / 2.0
                mid_in = (core_x0 <= mid_x <= core_x1 and core_y0 <= mid_y <= core_y1)
                ep0_in = (core_x0 <= x0 <= core_x1 and core_y0 <= y0 <= core_y1)
                ep1_in = (core_x0 <= x1 <= core_x1 and core_y0 <= y1 <= core_y1)
                if core_keep_mode == "endpoint_or_midpoint":
                    core_keep = mid_in or ep0_in or ep1_in
                else:
                    core_keep = mid_in
                if not core_keep:
                    continue

                if consensus_filter:
                    supported = segment_has_overlap_support(
                        item=item,
                        local_xy=local_xy,
                        seg_i=i,
                        support_index=support_index,
                        support_dist=consensus_support_dist,
                        angle_deg=consensus_angle_deg,
                        sample_spacing=consensus_sample_spacing,
                        min_supported_fraction=consensus_min_supported_fraction,
                        long_keep_length=consensus_long_keep_length,
                    )
                    if not supported:
                        continue

                kept_segments.append(i)

            if not kept_segments:
                continue

            kept_point_local_indices = set()
            for seg_i in kept_segments:
                kept_point_local_indices.add(seg_i)
                kept_point_local_indices.add(seg_i + 1)

            local_to_graph = {}
            for li in sorted(kept_point_local_indices):
                lx, ly = local_xy[li]
                global_r = ly + oy
                global_c = lx + ox
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


def _edge_length(nodes, u, v):
    return float(np.linalg.norm(nodes[int(u)] - nodes[int(v)]))


def _build_adjacency(num_nodes, edges):
    adjacency = [set() for _ in range(num_nodes)]
    for u, v in edges:
        u = int(u)
        v = int(v)
        if u == v:
            continue
        adjacency[u].add(v)
        adjacency[v].add(u)
    return adjacency


def compact_graph(nodes, edges):
    """Remove isolated nodes and remap edges to a dense node array."""
    if nodes.shape[0] == 0:
        return nodes, np.zeros((0, 2), dtype=np.int32)

    edges = deduplicate_undirected_edges(np.asarray(edges, dtype=np.int32))
    if edges.shape[0] == 0:
        return np.zeros((0, 2), dtype=np.float32), np.zeros((0, 2), dtype=np.int32)

    used = np.unique(edges.reshape(-1))
    old_to_new = np.full((nodes.shape[0],), -1, dtype=np.int32)
    old_to_new[used] = np.arange(used.shape[0], dtype=np.int32)
    new_nodes = nodes[used].astype(np.float32)
    new_edges = np.stack([old_to_new[edges[:, 0]], old_to_new[edges[:, 1]]], axis=1)
    new_edges = deduplicate_undirected_edges(new_edges)
    return new_nodes, new_edges


def remove_small_components(nodes, edges, min_component_edges=2, min_component_length=18.0):
    """
    Drop tiny detached components. These are usually isolated false positives and
    hurt precision much more than recall.
    """
    if nodes.shape[0] == 0 or edges.shape[0] == 0:
        return nodes, np.zeros((0, 2), dtype=np.int32)
    if min_component_edges <= 0 and min_component_length <= 0:
        return compact_graph(nodes, edges)

    edges = deduplicate_undirected_edges(np.asarray(edges, dtype=np.int32))
    labels = component_labels(nodes.shape[0], edges)
    comp_edges = defaultdict(list)
    comp_nodes = defaultdict(set)
    comp_length = defaultdict(float)

    for ei, (u, v) in enumerate(edges):
        u = int(u)
        v = int(v)
        root = int(labels[u])
        comp_edges[root].append(ei)
        comp_nodes[root].add(u)
        comp_nodes[root].add(v)
        comp_length[root] += _edge_length(nodes, u, v)

    keep_edge_mask = np.ones((edges.shape[0],), dtype=bool)
    for root, edge_ids in comp_edges.items():
        edge_count = len(edge_ids)
        length = comp_length[root]
        should_drop = False
        if min_component_edges > 0 and edge_count < min_component_edges:
            should_drop = True
        if min_component_length > 0 and length < min_component_length:
            should_drop = True
        # Keep non-tiny connected pieces even when they have few long edges.
        if edge_count >= max(3, min_component_edges + 1) or length >= max(30.0, min_component_length * 1.8):
            should_drop = False
        if should_drop:
            keep_edge_mask[np.asarray(edge_ids, dtype=np.int32)] = False

    return compact_graph(nodes, edges[keep_edge_mask])


def prune_short_spurs(nodes, edges, spur_length=16.0, max_spur_edges=5, rounds=2):
    """
    Remove short endpoint branches connected to a junction.

    The walk starts from degree-1 endpoints and follows degree-2 nodes until a
    junction/endpoint is reached. If the branch is short enough and ends at a
    junction, the branch edges are removed while preserving the junction.
    """
    if nodes.shape[0] == 0 or edges.shape[0] == 0:
        return nodes, np.zeros((0, 2), dtype=np.int32)
    if spur_length <= 0 or max_spur_edges <= 0 or rounds <= 0:
        return compact_graph(nodes, edges)

    nodes_work = np.asarray(nodes, dtype=np.float32)
    edge_set = {tuple(sorted((int(u), int(v)))) for u, v in deduplicate_undirected_edges(edges)}

    for _ in range(int(rounds)):
        if not edge_set:
            break
        edge_arr = np.asarray(sorted(edge_set), dtype=np.int32)
        adjacency = _build_adjacency(nodes_work.shape[0], edge_arr)
        degree = np.asarray([len(nbs) for nbs in adjacency], dtype=np.int32)
        endpoints = np.where(degree == 1)[0]
        to_remove = set()

        for endpoint in endpoints:
            endpoint = int(endpoint)
            if degree[endpoint] != 1:
                continue

            prev = -1
            cur = endpoint
            path_edges = []
            path_len = 0.0
            stop_node = None

            while True:
                nbs = [n for n in adjacency[cur] if n != prev]
                if not nbs:
                    stop_node = cur
                    break
                nxt = int(nbs[0])
                edge_key = tuple(sorted((cur, nxt)))
                path_edges.append(edge_key)
                path_len += _edge_length(nodes_work, cur, nxt)

                prev, cur = cur, nxt
                if degree[cur] != 2:
                    stop_node = cur
                    break
                if path_len > spur_length or len(path_edges) > max_spur_edges:
                    break

            if stop_node is None:
                continue
            if path_len <= spur_length and len(path_edges) <= max_spur_edges and degree[stop_node] >= 3:
                to_remove.update(path_edges)

        if not to_remove:
            break
        edge_set.difference_update(to_remove)

    if not edge_set:
        return np.zeros((0, 2), dtype=np.float32), np.zeros((0, 2), dtype=np.int32)

    out_edges = np.asarray(sorted(edge_set), dtype=np.int32)
    return compact_graph(nodes_work, out_edges)


def prune_unsupported_spurs(nodes, edges, region_id, support_index,
                            spur_length=16.0,
                            max_spur_edges=5,
                            support_dist=6.0,
                            angle_deg=30.0,
                            sample_spacing=8.0,
                            min_supported_fraction=0.20,
                            min_support_patches=2):
    """
    Remove endpoint branches only when the branch has weak evidence from the
    overlapping-patch support index.
    """
    if nodes.shape[0] == 0 or edges.shape[0] == 0:
        return nodes, np.zeros((0, 2), dtype=np.int32)
    if support_index is None or spur_length <= 0 or max_spur_edges <= 0:
        return compact_graph(nodes, edges)

    nodes_work = np.asarray(nodes, dtype=np.float32)
    edge_set = {tuple(sorted((int(u), int(v)))) for u, v in deduplicate_undirected_edges(edges)}
    if not edge_set:
        return np.zeros((0, 2), dtype=np.float32), np.zeros((0, 2), dtype=np.int32)

    edge_arr = np.asarray(sorted(edge_set), dtype=np.int32)
    adjacency = _build_adjacency(nodes_work.shape[0], edge_arr)
    degree = np.asarray([len(nbs) for nbs in adjacency], dtype=np.int32)
    endpoints = np.where(degree == 1)[0]
    to_remove = set()

    for endpoint in endpoints:
        endpoint = int(endpoint)
        prev = -1
        cur = endpoint
        path_edges = []
        path_len = 0.0
        stop_node = None

        while True:
            nbs = [n for n in adjacency[cur] if n != prev]
            if not nbs:
                stop_node = cur
                break
            nxt = int(nbs[0])
            edge_key = tuple(sorted((cur, nxt)))
            path_edges.append(edge_key)
            path_len += _edge_length(nodes_work, cur, nxt)

            prev, cur = cur, nxt
            if degree[cur] != 2:
                stop_node = cur
                break
            if path_len > spur_length or len(path_edges) > max_spur_edges:
                break

        if stop_node is None:
            continue
        if degree[stop_node] < 3:
            continue
        if path_len > spur_length or len(path_edges) > max_spur_edges:
            continue

        path_support = []
        path_patches = set()
        for u, v in path_edges:
            frac, patches = graph_segment_support_fraction(
                region_id=region_id,
                patch_id=None,
                p0=nodes_work[int(u)],
                p1=nodes_work[int(v)],
                support_index=support_index,
                support_dist=support_dist,
                angle_deg=angle_deg,
                sample_spacing=sample_spacing,
            )
            path_support.append(frac)
            path_patches.update(patches)

        mean_support = float(np.mean(path_support)) if path_support else 0.0
        if mean_support < min_supported_fraction or len(path_patches) < min_support_patches:
            to_remove.update(path_edges)

    if not to_remove:
        return compact_graph(nodes_work, edge_arr)

    edge_set.difference_update(to_remove)
    if not edge_set:
        return np.zeros((0, 2), dtype=np.float32), np.zeros((0, 2), dtype=np.int32)
    out_edges = np.asarray(sorted(edge_set), dtype=np.int32)
    return compact_graph(nodes_work, out_edges)


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
                                   parallel_dist=1.6,
                                   angle_deg=10.0,
                                   overlap_ratio=0.70):
    """
    Remove near-duplicate parallel edges caused by overlapping patches.

    This intentionally avoids high-degree junctions, because deleting a road
    through an intersection is more harmful than leaving a duplicate.
    """
    if nodes.shape[0] == 0 or edges.shape[0] == 0 or parallel_dist <= 0:
        return edges
    if edges.shape[0] < 2:
        return edges

    edges = deduplicate_undirected_edges(edges)
    degree = compute_node_degree(nodes.shape[0], edges)

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
        if not valid[i] or not valid[j]:
            continue

        ui, vi = int(edges[i, 0]), int(edges[i, 1])
        uj, vj = int(edges[j, 0]), int(edges[j, 1])
        if len({ui, vi, uj, vj}) < 4:
            continue
        if degree[ui] > 2 or degree[vi] > 2 or degree[uj] > 2 or degree[vj] > 2:
            continue

        dir_i = vec[i] / length[i]
        dir_j = vec[j] / length[j]
        if abs(float(np.dot(dir_i, dir_j))) < cos_th:
            continue

        d_candidates = [
            _point_to_segment_distance(seg_a[i], seg_a[j], seg_b[j]),
            _point_to_segment_distance(seg_b[i], seg_a[j], seg_b[j]),
            _point_to_segment_distance(seg_a[j], seg_a[i], seg_b[i]),
            _point_to_segment_distance(seg_b[j], seg_a[i], seg_b[i]),
        ]
        if max(d_candidates) > parallel_dist:
            continue

        if length[i] >= length[j]:
            base_idx, other_idx = i, j
        else:
            base_idx, other_idx = j, i
        base_dir = vec[base_idx] / length[base_idx]
        o0 = float(np.dot(seg_a[other_idx] - seg_a[base_idx], base_dir))
        o1 = float(np.dot(seg_b[other_idx] - seg_a[base_idx], base_dir))
        lo = max(0.0, min(o0, o1))
        hi = min(float(length[base_idx]), max(o0, o1))
        overlap_len = max(0.0, hi - lo)
        min_len = min(float(length[i]), float(length[j]))
        if min_len <= 1e-6 or (overlap_len / min_len) < overlap_ratio:
            continue

        drop = i if length[i] < length[j] else j
        remove_idx.add(drop)

    if not remove_idx:
        return edges

    keep_mask = np.ones((edges.shape[0],), dtype=bool)
    keep_mask[list(remove_idx)] = False
    return deduplicate_undirected_edges(edges[keep_mask].astype(np.int32))


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


def draw_junction_points(bgr_canvas, junctions, radius=4):
    bgr_canvas = bgr_canvas.copy()
    if junctions is None or len(junctions) == 0:
        return bgr_canvas
    for row, col in np.asarray(junctions, dtype=np.float32):
        center = (int(round(col)), int(round(row)))
        cv2.circle(bgr_canvas, center, int(radius), (255, 0, 255), -1, cv2.LINE_AA)
        cv2.circle(bgr_canvas, center, int(radius) + 2, (255, 255, 255), 1, cv2.LINE_AA)
    return bgr_canvas


def sat2graph_to_nodes_edges(sat2graph):
    node_keys = set()
    edge_keys = set()
    for node, neighbors in sat2graph.items():
        node_t = (int(node[0]), int(node[1]))
        node_keys.add(node_t)
        for nb in neighbors:
            nb_t = (int(nb[0]), int(nb[1]))
            node_keys.add(nb_t)
            edge_keys.add(tuple(sorted((node_t, nb_t))))

    sorted_nodes = sorted(node_keys)
    node_to_idx = {node: idx for idx, node in enumerate(sorted_nodes)}
    nodes = np.asarray(sorted_nodes, dtype=np.float32)
    if edge_keys:
        edges = np.asarray(
            [[node_to_idx[u], node_to_idx[v]] for u, v in sorted(edge_keys)],
            dtype=np.int32,
        )
    else:
        edges = np.zeros((0, 2), dtype=np.int32)
    return nodes, edges


def save_region_viz(region_id, nodes, edges, region_metas, patch_image_dir,
                    crop_size_orig, image_root, image_pattern, viz_save_dir,
                    junctions=None, draw_junctions=False):
    background_canvas = load_region_background(
        region_id,
        nodes,
        region_metas=region_metas,
        patch_image_dir=patch_image_dir,
        crop_size_orig=crop_size_orig,
        image_root=image_root,
        image_pattern=image_pattern,
    )
    viz_img = create_viz_image(nodes, edges, background_canvas)
    if draw_junctions:
        viz_img = draw_junction_points(viz_img, junctions)
    save_path = os.path.join(viz_save_dir, f"region_{region_id}.png")
    save_image_unicode_safe(save_path, viz_img)
    return save_path


def run_viz_only(args, coco, region_metas, region_junctions):
    graph_dir = args.viz_graph_dir or os.path.join(args.output_dir, "graph")
    viz_save_dir = args.viz_output_dir or os.path.join(args.output_dir, "viz")
    os.makedirs(viz_save_dir, exist_ok=True)

    selected_region_set = parse_region_list(args.regions)
    if args.single_region is not None:
        selected_region_ids = [int(args.single_region)]
    elif selected_region_set is not None:
        selected_region_ids = sorted(selected_region_set)
    else:
        selected_region_ids = []
        if os.path.isdir(graph_dir):
            for name in os.listdir(graph_dir):
                match = re.match(r"^(\d+)\.p$", name)
                if match:
                    selected_region_ids.append(int(match.group(1)))
            selected_region_ids.sort()

    print(f"[viz-only] graph_dir={graph_dir}")
    print(f"[viz-only] viz_dir={viz_save_dir}")
    print(f"[viz-only] regions={selected_region_ids}")

    run_t0 = time.perf_counter()
    total_regions = len(selected_region_ids)
    done = 0
    print_progress_bar(0, total_regions, run_t0)
    for region_id in selected_region_ids:
        done += 1
        graph_path = os.path.join(graph_dir, f"{region_id}.p")
        if not os.path.exists(graph_path):
            print(f"[viz-only] skip missing graph: {graph_path}")
            print_progress_bar(done, total_regions, run_t0)
            continue
        t0 = time.perf_counter()
        with open(graph_path, "rb") as f:
            sat2graph = pickle.load(f)
        nodes, edges = sat2graph_to_nodes_edges(sat2graph)
        save_path = save_region_viz(
            region_id,
            nodes,
            edges,
            region_metas=region_metas,
            patch_image_dir=args.patch_image_dir,
            crop_size_orig=args.crop_size_orig,
            image_root=args.image_root,
            image_pattern=args.image_pattern,
            viz_save_dir=viz_save_dir,
            junctions=region_junctions.get(region_id),
            draw_junctions=bool(args.viz_draw_junctions),
        )
        dt = time.perf_counter() - t0
        print(
            f"[viz-only] saved region={region_id} "
            f"nodes={nodes.shape[0]} edges={edges.shape[0]} "
            f"path={save_path} elapsed={dt:.2f}s"
        )
        print_progress_bar(done, total_regions, run_t0)
    if total_regions > 0:
        print()


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


def apply_param_preset(args):
    preset_name = getattr(args, "preset", "balanced")
    if preset_name == "custom":
        return
    preset = PARAM_PRESETS[preset_name]
    for key, value in preset.items():
        setattr(args, key, value)


def summarize_args(args):
    keys = [
        "preset",
        "score_threshold",
        "merge_node_dist",
        "endpoint_snap_dist",
        "core_keep_mode",
        "core_margin_scale",
        "core_buffer_px",
        "patch_boundary_bridge_dist",
        "patch_boundary_near_px",
        "consensus_filter",
        "consensus_support_dist",
        "consensus_angle_deg",
        "consensus_sample_spacing",
        "consensus_min_supported_fraction",
        "consensus_long_keep_length",
        "support_spur_filter",
        "support_spur_length",
        "support_spur_max_edges",
        "support_spur_min_patches",
        "support_spur_min_supported_fraction",
        "spur_length",
        "spur_max_edges",
        "spur_rounds",
        "min_component_edges",
        "min_component_length",
        "parallel_prune_dist",
        "parallel_prune_angle_deg",
        "parallel_overlap_ratio",
        "junction_enable",
        "junction_snap_node_dist",
        "junction_split_edge_dist",
        "junction_endpoint_bridge_dist",
        "junction_min_patch_support",
        "junction_cluster_dist",
    ]
    return " ".join(f"{key}={getattr(args, key)}" for key in keys)


def parse_region_list(regions_arg):
    if not regions_arg or not regions_arg.strip():
        return None
    return {int(x.strip()) for x in regions_arg.split(",") if x.strip()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_json",
        required=True,
    )
    parser.add_argument(
        "--junction_json",
        default=None,
        help="Patch-local junction prediction json. Empty string disables junction guidance.",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
    )
    parser.add_argument("--preset", choices=["balanced", "precision", "recall", "custom"],
                        default="balanced",
                        help="后处理参数预设。custom 表示完全使用命令行传入参数。")
    parser.add_argument("--score_threshold", type=float, default=0.0)
    parser.add_argument("--category_id", type=int, default=None)
    parser.add_argument("--merge_node_dist", type=float, default=6.0)
    parser.add_argument("--split_edge_dist", type=float, default=3.0)
    parser.add_argument("--endpoint_snap_dist", type=float, default=3.0)
    parser.add_argument("--single_region", type=int, default=None,
                        help="仅处理指定 region；None 表示处理全部。")
    parser.add_argument("--regions", type=str, default="",
                        help="逗号分隔的 region 列表，例如 8,48,68；优先级低于 --single_region。")
    parser.add_argument("--crop_size_orig", type=int, default=128,
                        help="原始裁剪尺寸 (px)")
    parser.add_argument("--patch_size_model", type=int, default=256,
                        help="模型输入 patch 尺寸 (px)")
    parser.add_argument("--stride", type=int, default=64,
                        help="裁剪步长 (px)")
    parser.add_argument("--core_keep_mode", type=str, default="midpoint",
                        choices=["midpoint", "endpoint_or_midpoint"],
                        help="核心区过滤模式：midpoint 或 endpoint_or_midpoint")
    parser.add_argument("--core_margin_scale", type=float, default=0.75,
                        help="核心区收缩比例，1.0 为原策略，0.5 为 margin 减半")
    parser.add_argument("--core_buffer_px", type=float, default=0.0,
                        help="核心区额外缓冲像素（在 core_keep_mode 下扩展边界）")
    parser.add_argument("--patch_boundary_bridge_dist", type=float, default=2.5,
                        help="patch 边界附近端点桥接距离，<=0 表示关闭")
    parser.add_argument("--patch_boundary_near_px", type=float, default=3.0,
                        help="判定“靠近 patch 边界”的像素阈值")
    parser.add_argument("--consensus_filter", type=int, default=1,
                        help="是否启用重叠 patch 共识过滤：1 启用，0 关闭")
    parser.add_argument("--consensus_support_dist", type=float, default=5.0,
                        help="共识过滤：其他 patch 支持点的最大距离")
    parser.add_argument("--consensus_angle_deg", type=float, default=25.0,
                        help="共识过滤：其他 patch 支持线段的方向夹角阈值")
    parser.add_argument("--consensus_sample_spacing", type=float, default=8.0,
                        help="共识过滤：沿线段采样间隔")
    parser.add_argument("--consensus_min_supported_fraction", type=float, default=0.25,
                        help="共识过滤：采样点中至少多少比例需要其他 patch 支持")
    parser.add_argument("--consensus_long_keep_length", type=float, default=55.0,
                        help="共识过滤：长于该值的线段直接保留，避免误删长主路")
    parser.add_argument("--support_spur_filter", type=int, default=1,
                        help="是否启用重叠证据短悬挂枝过滤：1 启用，0 关闭")
    parser.add_argument("--support_spur_length", type=float, default=16.0,
                        help="重叠证据短悬挂枝过滤：候选枝最大长度")
    parser.add_argument("--support_spur_max_edges", type=int, default=5,
                        help="重叠证据短悬挂枝过滤：候选枝最多边数")
    parser.add_argument("--support_spur_min_patches", type=int, default=2,
                        help="重叠证据短悬挂枝过滤：至少需要多少个支持 patch")
    parser.add_argument("--support_spur_min_supported_fraction", type=float, default=0.20,
                        help="重叠证据短悬挂枝过滤：平均支持比例下限")
    parser.add_argument("--spur_length", type=float, default=16.0,
                        help="短悬挂枝剪除长度阈值，<=0 表示关闭")
    parser.add_argument("--spur_max_edges", type=int, default=5,
                        help="短悬挂枝最多包含的边数")
    parser.add_argument("--spur_rounds", type=int, default=2,
                        help="短悬挂枝迭代剪除轮数")
    parser.add_argument("--min_component_edges", type=int, default=2,
                        help="小连通分量最少边数；少于该值会被删除，<=0 表示关闭")
    parser.add_argument("--min_component_length", type=float, default=20.0,
                        help="小连通分量最短总长度；短于该值会被删除，<=0 表示关闭")
    parser.add_argument("--parallel_prune_dist", type=float, default=1.6,
                        help="平行/重叠冗余边去除距离阈值，<=0 表示关闭")
    parser.add_argument("--parallel_prune_angle_deg", type=float, default=10.0,
                        help="判定平行的最大夹角（度）")
    parser.add_argument("--parallel_overlap_ratio", type=float, default=0.70,
                        help="判定重叠冗余的最小投影重叠比例")
    parser.add_argument("--junction_enable", type=int, default=1,
                        help="是否启用 junction-aware 图修正：1 启用，0 关闭")
    parser.add_argument("--junction_snap_node_dist", type=float, default=5.0,
                        help="Junction 修正：将附近节点吸附到 junction 的距离阈值")
    parser.add_argument("--junction_split_edge_dist", type=float, default=4.0,
                        help="Junction 修正：junction 在线段附近时切分边的距离阈值")
    parser.add_argument("--junction_endpoint_bridge_dist", type=float, default=0.0,
                        help="Junction 修正：悬挂端点桥接到附近 junction 的距离阈值")
    parser.add_argument("--junction_min_patch_support", type=int, default=2,
                        help="Junction 聚类：至少需要多少个 patch 支持才保留")
    parser.add_argument("--junction_cluster_dist", type=float, default=5.0,
                        help="Junction 聚类：重复 junction 合并距离")
    parser.add_argument("--enable_viz", type=int, default=0)
    parser.add_argument("--viz_only", type=int, default=0,
                        help="只基于已保存的 graph/*.p 生成可视化，不重新拼接 graph")
    parser.add_argument("--viz_graph_dir", default="",
                        help="viz_only 模式下读取 graph 的目录；空值表示 output_dir/graph")
    parser.add_argument("--viz_output_dir", default="",
                        help="viz_only 模式下保存可视化的目录；空值表示 output_dir/viz")
    parser.add_argument("--viz_draw_junctions", type=int, default=1,
                        help="可视化时是否叠加聚类后的 junction 点")
    parser.add_argument(
        "--patch_image_dir",
        default=None,
        help="Patch image directory used to stitch background.",
    )
    parser.add_argument(
        "--image_root",
        default=os.path.join(PROJECT_ROOT, "cityscale", "20cities"),
        help="Fallback directory for region-level satellite images.",
    )
    parser.add_argument("--image_pattern", default="region_{}_sat.png")
    args = parser.parse_args()
    apply_param_preset(args)
    selected_region_set = parse_region_list(args.regions)

    print(f"[config] {summarize_args(args)}")

    with open(args.input_json, "r", encoding="utf-8") as f:
        coco = json.load(f)

    if bool(args.viz_only):
        region_metas = build_region_patch_meta(coco)
        region_junctions = {}
        if bool(args.viz_draw_junctions) and bool(args.junction_enable) and args.junction_json:
            raw_region_junctions = load_junction_points(
                args.junction_json,
                crop_size_orig=args.crop_size_orig,
                patch_size_model=args.patch_size_model,
            )
            region_junctions = cluster_region_junctions(
                raw_region_junctions,
                cluster_dist=args.junction_cluster_dist,
                min_patch_support=args.junction_min_patch_support,
            )
            total_junctions = sum(int(v.shape[0]) for v in region_junctions.values())
            print(
                f"[junction] regions={len(region_junctions)} "
                f"clustered_junctions={total_junctions}"
            )
        run_viz_only(args, coco, region_metas, region_junctions)
        return

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
        consensus_filter=bool(args.consensus_filter),
        consensus_support_dist=args.consensus_support_dist,
        consensus_angle_deg=args.consensus_angle_deg,
        consensus_sample_spacing=args.consensus_sample_spacing,
        consensus_min_supported_fraction=args.consensus_min_supported_fraction,
        consensus_long_keep_length=args.consensus_long_keep_length,
    )
    graph_support_index = None
    if bool(args.support_spur_filter):
        graph_support_index = build_region_support_index(
            coco,
            score_threshold=args.score_threshold,
            category_id=args.category_id,
            crop_size_orig=args.crop_size_orig,
            patch_size_model=args.patch_size_model,
            support_dist=args.consensus_support_dist,
            sample_spacing=args.consensus_sample_spacing,
        )

    region_junctions = {}
    if bool(args.junction_enable) and args.junction_json:
        raw_region_junctions = load_junction_points(
            args.junction_json,
            crop_size_orig=args.crop_size_orig,
            patch_size_model=args.patch_size_model,
        )
        region_junctions = cluster_region_junctions(
            raw_region_junctions,
            cluster_dist=args.junction_cluster_dist,
            min_patch_support=args.junction_min_patch_support,
        )
        total_junctions = sum(int(v.shape[0]) for v in region_junctions.values())
        print(
            f"[junction] regions={len(region_junctions)} "
            f"clustered_junctions={total_junctions}"
        )

    # 构建每个 region 的 patch 元信息（用于拼接背景）
    region_metas = build_region_patch_meta(coco)

    os.makedirs(args.output_dir, exist_ok=True)
    graph_save_dir = os.path.join(args.output_dir, "graph")
    os.makedirs(graph_save_dir, exist_ok=True)
    viz_save_dir = os.path.join(args.output_dir, "viz")
    if bool(args.enable_viz):
        os.makedirs(viz_save_dir, exist_ok=True)

    summary = []
    selected_region_ids = [
        rid for rid in sorted(region_graphs.keys())
        if (
            (args.single_region is None or rid == args.single_region) and
            (selected_region_set is None or rid in selected_region_set)
        )
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

        if args.parallel_prune_dist > 0:
            edges = prune_redundant_parallel_edges(
                nodes,
                edges,
                parallel_dist=args.parallel_prune_dist,
                angle_deg=args.parallel_prune_angle_deg,
                overlap_ratio=args.parallel_overlap_ratio,
            )
            edges = deduplicate_undirected_edges(np.asarray(edges, dtype=np.int32))

        if bool(args.support_spur_filter):
            nodes, edges = prune_unsupported_spurs(
                nodes,
                edges,
                region_id=region_id,
                support_index=graph_support_index,
                spur_length=args.support_spur_length,
                max_spur_edges=args.support_spur_max_edges,
                support_dist=args.consensus_support_dist,
                angle_deg=args.consensus_angle_deg,
                sample_spacing=args.consensus_sample_spacing,
                min_supported_fraction=args.support_spur_min_supported_fraction,
                min_support_patches=args.support_spur_min_patches,
            )

        if bool(args.junction_enable):
            nodes, edges = apply_junction_guidance(
                nodes,
                edges,
                region_junctions.get(region_id, np.zeros((0, 2), dtype=np.float32)),
                snap_node_dist=args.junction_snap_node_dist,
                split_edge_dist=args.junction_split_edge_dist,
                endpoint_bridge_dist=args.junction_endpoint_bridge_dist,
            )

        nodes, edges = prune_short_spurs(
            nodes,
            edges,
            spur_length=args.spur_length,
            max_spur_edges=args.spur_max_edges,
            rounds=args.spur_rounds,
        )
        nodes, edges = remove_small_components(
            nodes,
            edges,
            min_component_edges=args.min_component_edges,
            min_component_length=args.min_component_length,
        )

        sat2graph = convert_to_sat2graph_format(nodes, edges)
        output_path = os.path.join(graph_save_dir, f"{region_id}.p")
        with open(output_path, "wb") as f:
            pickle.dump(sat2graph, f)
        save_dt = time.perf_counter() - region_t0
        print(f"[region] graph_saved region={region_id} elapsed={save_dt:.2f}s")

        if bool(args.enable_viz):
            viz_t0 = time.perf_counter()
            after_path = save_region_viz(
                region_id,
                nodes,
                edges,
                region_metas=region_metas,
                patch_image_dir=args.patch_image_dir,
                crop_size_orig=args.crop_size_orig,
                image_root=args.image_root,
                image_pattern=args.image_pattern,
                viz_save_dir=viz_save_dir,
                junctions=region_junctions.get(region_id),
                draw_junctions=bool(args.viz_draw_junctions),
            )
            viz_dt = time.perf_counter() - viz_t0
            print(f"[region] viz_done region={region_id} path={after_path} elapsed={viz_dt:.2f}s")

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


