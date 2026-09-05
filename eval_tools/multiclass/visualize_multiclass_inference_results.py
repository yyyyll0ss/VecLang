import argparse
import pickle
import importlib.util
import json
import os
import random
import re
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_ROAD_STITCH_SCRIPT = (
    SCRIPT_DIR / "road" / "2.stitch_coco_polylines.py"
)

SAVE_DPI = 600

BUILDING_OUTLINE = (182, 47, 54, 255)   # #B62F36
BUILDING_FILL = (243, 185, 184, 150)    # #F3B9B8
BUILDING_NODE = BUILDING_OUTLINE
BUILDING_NODE_RADIUS = 4
BUILDING_OUTLINE_THICKNESS = 2

WATER_OUTLINE = (68, 114, 196, 255)     # #4472C4
WATER_FILL = (133, 193, 229, 150)       # #85C1E5
WATER_NODE = WATER_OUTLINE
WATER_NODE_RADIUS = 4
WATER_OUTLINE_THICKNESS = 2

ROAD_AREA_FILL = (255, 221, 168, 125)
ROAD_AREA_OUTLINE = (224, 133, 20, 180)
ROAD_LINE = (253, 160, 15, 255)
ROAD_NODE = (255, 255, 0, 255)
ROAD_LINE_THICKNESS = 2
ROAD_AREA_BUFFER_THICKNESS = 14
ROAD_NODE_RADIUS = 3

PATCH_NAME_PATTERN = re.compile(
    r"^(?P<region>\d+)_x(?P<x>\d+)_y(?P<y>\d+)\.(?:png|jpg|jpeg|tif|tiff|bmp)$",
    re.IGNORECASE,
)


def path_arg(path):
    return str(path) if path is not None else None


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def read_image_unicode(path):
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def write_image_with_dpi(path, image, dpi=SAVE_DPI):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ext = path.suffix.lower() or ".jpg"

    try:
        from PIL import Image

        format_by_ext = {
            ".jpg": "JPEG",
            ".jpeg": "JPEG",
            ".png": "PNG",
            ".tif": "TIFF",
            ".tiff": "TIFF",
            ".bmp": "BMP",
        }
        image_format = format_by_ext.get(ext)
        if image_format is not None:
            if image.ndim == 2:
                pil_img = Image.fromarray(image)
            elif image.shape[2] == 4:
                pil_img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA))
            else:
                pil_img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

            save_kwargs = {"dpi": (dpi, dpi)}
            if image_format == "JPEG":
                save_kwargs.update({"quality": 95, "subsampling": 0})
                if pil_img.mode != "RGB":
                    pil_img = pil_img.convert("RGB")

            with open(path, "wb") as f:
                pil_img.save(f, format=image_format, **save_kwargs)
            return True
    except ImportError:
        print("Warning: Pillow is not installed; saving without DPI metadata.")
    except Exception as exc:
        print(f"Warning: Pillow save failed for {path}: {exc}; falling back to cv2.imencode.")

    ok, encoded = cv2.imencode(ext, image)
    if not ok:
        return False
    encoded.tofile(str(path))
    return True


def rgba_to_bgr(rgba):
    r, g, b, _ = rgba
    return (b, g, r)


def blend_overlay(base, overlay, rgba):
    alpha = rgba[3] / 255.0
    if alpha >= 1.0:
        return overlay
    if alpha <= 0.0:
        return base
    return cv2.addWeighted(overlay, alpha, base, 1.0 - alpha, 0)


def iter_segmentation_arrays(segmentation, min_points):
    if isinstance(segmentation, str):
        try:
            segmentation = json.loads(segmentation)
        except Exception:
            return

    if not isinstance(segmentation, list) or len(segmentation) == 0:
        return

    if all(isinstance(item, (int, float)) for item in segmentation):
        segments = [segmentation]
    else:
        segments = segmentation

    for segment in segments:
        if not isinstance(segment, list) or len(segment) < min_points * 2:
            continue
        if len(segment) % 2 == 1:
            segment = segment[:-1]
        try:
            pts = np.asarray(segment, dtype=np.float32).reshape(-1, 2)
        except Exception:
            continue
        if pts.shape[0] >= min_points:
            yield pts


def clip_points(points, width, height):
    clipped = points.copy()
    clipped[:, 0] = np.clip(clipped[:, 0], 0, max(0, width - 1))
    clipped[:, 1] = np.clip(clipped[:, 1], 0, max(0, height - 1))
    return np.round(clipped).astype(np.int32)


def is_near_any(value, candidates, tolerance):
    return any(abs(float(value) - float(candidate)) <= tolerance for candidate in candidates)


def point_on_internal_boundary(point, internal_boundaries, tolerance):
    if not internal_boundaries:
        return False
    x, y = point
    return (
        is_near_any(x, internal_boundaries.get("x", []), tolerance)
        or is_near_any(y, internal_boundaries.get("y", []), tolerance)
    )


def segment_on_internal_boundary(p0, p1, internal_boundaries, tolerance):
    if not internal_boundaries:
        return False
    x0, y0 = p0
    x1, y1 = p1
    return (
        is_near_any(x0, internal_boundaries.get("x", []), tolerance)
        and is_near_any(x1, internal_boundaries.get("x", []), tolerance)
        or is_near_any(y0, internal_boundaries.get("y", []), tolerance)
        and is_near_any(y1, internal_boundaries.get("y", []), tolerance)
    )


def draw_polyline_segments(
    image,
    pts_int,
    color,
    thickness,
    is_closed,
    internal_boundaries=None,
    boundary_tolerance=1,
):
    if pts_int.shape[0] < 2:
        return

    points = [tuple(pt) for pt in pts_int]
    last_idx = len(points) if is_closed else len(points) - 1
    for idx in range(last_idx):
        p0 = points[idx]
        p1 = points[(idx + 1) % len(points)]
        if segment_on_internal_boundary(p0, p1, internal_boundaries, boundary_tolerance):
            continue
        cv2.line(image, p0, p1, color, thickness, cv2.LINE_AA)


def draw_polygon_layer(
    image,
    polygons,
    fill_rgba,
    outline_rgba,
    node_rgba,
    outline_thickness,
    node_radius,
    close_outline=True,
    draw_nodes=True,
    internal_boundaries=None,
    boundary_tolerance=1,
):
    out = image.copy()
    height, width = out.shape[:2]
    draw_polygons = []

    for pts in polygons:
        pts_int = clip_points(pts, width, height)
        if pts_int.shape[0] >= 3:
            draw_polygons.append(pts_int)

    if not draw_polygons:
        return out

    fill_overlay = out.copy()
    cv2.fillPoly(fill_overlay, draw_polygons, rgba_to_bgr(fill_rgba), lineType=cv2.LINE_AA)
    out = blend_overlay(out, fill_overlay, fill_rgba)

    outline_overlay = out.copy()
    for pts_int in draw_polygons:
        draw_polyline_segments(
            outline_overlay,
            pts_int,
            rgba_to_bgr(outline_rgba),
            outline_thickness,
            close_outline,
            internal_boundaries=internal_boundaries,
            boundary_tolerance=boundary_tolerance,
        )
    out = blend_overlay(out, outline_overlay, outline_rgba)

    if draw_nodes and node_rgba is not None:
        node_overlay = out.copy()
        for pts_int in draw_polygons:
            for pt in pts_int:
                if point_on_internal_boundary(pt, internal_boundaries, boundary_tolerance):
                    continue
                cv2.circle(
                    node_overlay,
                    tuple(pt),
                    node_radius,
                    rgba_to_bgr(node_rgba),
                    -1,
                    lineType=cv2.LINE_AA,
                )
        out = blend_overlay(out, node_overlay, node_rgba)
    return out


def draw_road_graph(
    image,
    nodes,
    edges,
    line_rgba=ROAD_LINE,
    node_rgba=ROAD_NODE,
    line_thickness=ROAD_LINE_THICKNESS,
    node_radius=ROAD_NODE_RADIUS,
    draw_nodes=True,
):
    out = image.copy()
    if nodes is None or edges is None or len(nodes) == 0:
        return out

    height, width = out.shape[:2]
    xy_nodes = np.asarray(nodes, dtype=np.float32)[:, ::-1]

    line_overlay = out.copy()
    for u, v in np.asarray(edges, dtype=np.int32):
        p0 = xy_nodes[int(u)]
        p1 = xy_nodes[int(v)]
        x0 = int(round(np.clip(p0[0], 0, width - 1)))
        y0 = int(round(np.clip(p0[1], 0, height - 1)))
        x1 = int(round(np.clip(p1[0], 0, width - 1)))
        y1 = int(round(np.clip(p1[1], 0, height - 1)))
        cv2.line(
            line_overlay,
            (x0, y0),
            (x1, y1),
            rgba_to_bgr(line_rgba),
            line_thickness,
            cv2.LINE_AA,
        )
    out = blend_overlay(out, line_overlay, line_rgba)

    if draw_nodes and node_rgba is not None:
        node_overlay = out.copy()
        for pt in xy_nodes:
            x = int(round(np.clip(pt[0], 0, width - 1)))
            y = int(round(np.clip(pt[1], 0, height - 1)))
            cv2.circle(
                node_overlay,
                (x, y),
                node_radius,
                rgba_to_bgr(node_rgba),
                -1,
                lineType=cv2.LINE_AA,
            )
        out = blend_overlay(out, node_overlay, node_rgba)

    return out


def draw_road_graph_buffer(image, nodes, edges, fill_rgba=ROAD_AREA_FILL, thickness=ROAD_AREA_BUFFER_THICKNESS):
    out = image.copy()
    if nodes is None or edges is None or len(nodes) == 0:
        return out

    height, width = out.shape[:2]
    xy_nodes = np.asarray(nodes, dtype=np.float32)[:, ::-1]
    area_overlay = out.copy()

    for u, v in np.asarray(edges, dtype=np.int32):
        p0 = xy_nodes[int(u)]
        p1 = xy_nodes[int(v)]
        x0 = int(round(np.clip(p0[0], 0, width - 1)))
        y0 = int(round(np.clip(p0[1], 0, height - 1)))
        x1 = int(round(np.clip(p1[0], 0, width - 1)))
        y1 = int(round(np.clip(p1[1], 0, height - 1)))
        cv2.line(
            area_overlay,
            (x0, y0),
            (x1, y1),
            rgba_to_bgr(fill_rgba),
            thickness,
            cv2.LINE_AA,
        )

    return blend_overlay(out, area_overlay, fill_rgba)


def draw_polylines_layer(
    image,
    polylines,
    line_rgba,
    node_rgba=None,
    thickness=2,
    node_radius=3,
    draw_nodes=True,
):
    out = image.copy()
    if not polylines:
        return out

    height, width = out.shape[:2]
    line_overlay = out.copy()
    clipped_lines = []

    for pts in polylines:
        if pts is None or len(pts) < 2:
            continue
        pts_int = clip_points(np.asarray(pts, dtype=np.float32), width, height)
        if pts_int.shape[0] < 2:
            continue
        clipped_lines.append(pts_int)
        cv2.polylines(
            line_overlay,
            [pts_int],
            isClosed=False,
            color=rgba_to_bgr(line_rgba),
            thickness=thickness,
            lineType=cv2.LINE_AA,
        )
    out = blend_overlay(out, line_overlay, line_rgba)

    if draw_nodes and node_rgba is not None:
        node_overlay = out.copy()
        for pts_int in clipped_lines:
            for pt in pts_int:
                cv2.circle(
                    node_overlay,
                    tuple(pt),
                    node_radius,
                    rgba_to_bgr(node_rgba),
                    -1,
                    lineType=cv2.LINE_AA,
                )
        out = blend_overlay(out, node_overlay, node_rgba)

    return out


def parse_region_id_from_source(source_image):
    if not source_image:
        return None
    stem = Path(str(source_image)).stem
    try:
        return int(stem)
    except ValueError:
        return None


def parse_patch_file_name(file_name):
    match = PATCH_NAME_PATTERN.match(str(file_name))
    if match is None:
        return None
    return {
        "region_id": int(match.group("region")),
        "patch_x": int(match.group("x")),
        "patch_y": int(match.group("y")),
    }


def load_patch_image_meta(patch_coco_path):
    coco = load_json(patch_coco_path)
    image_meta = {}
    region_sizes = {}
    region_to_image_ids = defaultdict(list)
    region_patch_spans = defaultdict(list)

    for image_info in coco.get("images", []):
        image_id = int(image_info["id"])
        file_name = image_info.get("file_name", "")
        parsed = parse_patch_file_name(file_name) or {}
        region_id = parse_region_id_from_source(image_info.get("source_image"))
        if region_id is None:
            region_id = parsed.get("region_id")
        if region_id is None:
            continue

        patch_x = int(image_info.get("patch_x", parsed.get("patch_x", 0)))
        patch_y = int(image_info.get("patch_y", parsed.get("patch_y", 0)))
        source_width = int(image_info.get("source_width", patch_x + image_info.get("width", 0)))
        source_height = int(image_info.get("source_height", patch_y + image_info.get("height", 0)))

        image_meta[image_id] = {
            "region_id": region_id,
            "file_name": file_name,
            "patch_x": patch_x,
            "patch_y": patch_y,
            "width": int(image_info.get("width", 512)),
            "height": int(image_info.get("height", 512)),
            "source_width": source_width,
            "source_height": source_height,
        }
        region_to_image_ids[region_id].append(image_id)
        region_patch_spans[region_id].append(
            (
                patch_x,
                patch_y,
                int(image_info.get("width", 512)),
                int(image_info.get("height", 512)),
            )
        )

        old_size = region_sizes.get(region_id, (0, 0))
        region_sizes[region_id] = (
            max(old_size[0], source_width),
            max(old_size[1], source_height),
        )

    region_internal_boundaries = {}
    for region_id, spans in region_patch_spans.items():
        source_width, source_height = region_sizes.get(region_id, (0, 0))
        x_lines = set()
        y_lines = set()
        for patch_x, patch_y, patch_w, patch_h in spans:
            for x in (patch_x, patch_x + patch_w):
                if 0 < x < source_width:
                    x_lines.add(int(x))
            for y in (patch_y, patch_y + patch_h):
                if 0 < y < source_height:
                    y_lines.add(int(y))
        region_internal_boundaries[region_id] = {
            "x": sorted(x_lines),
            "y": sorted(y_lines),
        }

    return image_meta, region_sizes, region_to_image_ids, region_internal_boundaries


def load_prediction_annotations(path):
    data = load_json(path)
    if isinstance(data, list):
        return data
    return data.get("annotations", [])


def parse_category_ids(value):
    if value is None:
        return None
    value = str(value).strip()
    if not value or value.lower() in {"all", "*"}:
        return None
    return {int(item.strip()) for item in value.split(",") if item.strip()}


def build_region_polygons(pred_path, patch_image_meta, category_ids, score_threshold):
    region_polygons = defaultdict(list)
    if not pred_path or not os.path.exists(pred_path):
        print(f"Warning: prediction file not found: {pred_path}")
        return region_polygons

    annotations = load_prediction_annotations(pred_path)
    kept = 0
    skipped_missing_image = 0

    for ann in annotations:
        if category_ids is not None and int(ann.get("category_id", -1)) not in category_ids:
            continue
        if float(ann.get("score", 1.0)) < score_threshold:
            continue

        try:
            image_id = int(ann.get("image_id"))
        except Exception:
            continue
        meta = patch_image_meta.get(image_id)
        if meta is None:
            skipped_missing_image += 1
            continue

        offset = np.asarray([meta["patch_x"], meta["patch_y"]], dtype=np.float32)
        for pts in iter_segmentation_arrays(ann.get("segmentation", []), min_points=3):
            region_polygons[meta["region_id"]].append(pts + offset)
            kept += 1

    print(
        f"Loaded {kept} polygons from {pred_path}"
        f" | skipped_missing_image={skipped_missing_image}"
    )
    return region_polygons


def build_region_polygons_from_coco(coco_path, patch_image_meta, category_ids, score_threshold=0.0):
    region_polygons = defaultdict(list)
    if not coco_path or not os.path.exists(coco_path):
        print(f"Warning: COCO file not found: {coco_path}")
        return region_polygons

    data = load_json(coco_path)
    annotations = data if isinstance(data, list) else data.get("annotations", [])
    kept = 0
    skipped_missing_image = 0

    for ann in annotations:
        if category_ids is not None and int(ann.get("category_id", -1)) not in category_ids:
            continue
        if float(ann.get("score", 1.0)) < score_threshold:
            continue

        try:
            image_id = int(ann.get("image_id"))
        except Exception:
            continue
        meta = patch_image_meta.get(image_id)
        if meta is None:
            skipped_missing_image += 1
            continue

        offset = np.asarray([meta["patch_x"], meta["patch_y"]], dtype=np.float32)
        for pts in iter_segmentation_arrays(ann.get("segmentation", []), min_points=3):
            region_polygons[meta["region_id"]].append(pts + offset)
            kept += 1

    print(
        f"Loaded {kept} GT polygons from {coco_path}"
        f" | categories={'all' if category_ids is None else sorted(category_ids)}"
        f" | skipped_missing_image={skipped_missing_image}"
    )
    return region_polygons


def iter_geojson_line_coords(geometry):
    if not isinstance(geometry, dict):
        return

    geom_type = geometry.get("type")
    coords = geometry.get("coordinates")

    if geom_type == "LineString":
        if isinstance(coords, list) and len(coords) >= 2:
            yield coords
    elif geom_type == "MultiLineString":
        if isinstance(coords, list):
            for line in coords:
                if isinstance(line, list) and len(line) >= 2:
                    yield line
    elif geom_type == "GeometryCollection":
        for geom in geometry.get("geometries", []):
            yield from iter_geojson_line_coords(geom)


def build_gt_roadlines(roadline_dir):
    region_lines = defaultdict(list)
    if not roadline_dir or not os.path.isdir(roadline_dir):
        print(f"Warning: GT roadline directory not found: {roadline_dir}")
        return region_lines

    total_lines = 0
    for path in sorted(Path(roadline_dir).glob("*.geojson")):
        try:
            region_id = int(path.stem)
        except ValueError:
            continue

        data = load_json(path)
        for feature in data.get("features", []):
            for coords in iter_geojson_line_coords(feature.get("geometry")):
                try:
                    pts = np.asarray([[float(x), float(y)] for x, y, *_ in coords], dtype=np.float32)
                except Exception:
                    continue
                if pts.shape[0] >= 2:
                    region_lines[region_id].append(pts)
                    total_lines += 1

    print(f"Loaded {total_lines} GT road lines from {roadline_dir} | regions={len(region_lines)}")
    return region_lines


def load_road_stitch_module(script_path):
    script_path = Path(script_path)
    spec = importlib.util.spec_from_file_location("road_stitch_hard_v2", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load road stitch script: {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_road_graphs(args):
    if args.road_graph_dir:
        paths = sorted(Path(args.road_graph_dir).glob("*.p"))
        if not paths:
            raise FileNotFoundError(f"No stitched graphs in {args.road_graph_dir}")
        graphs = {}
        for path in paths:
            with path.open("rb") as f:
                adjacency = pickle.load(f)
            nodes = list(dict.fromkeys(list(adjacency) + [v for values in adjacency.values() for v in values]))
            index = {node: i for i, node in enumerate(nodes)}
            edges = {tuple(sorted((index[u], index[v]))) for u, values in adjacency.items() for v in values if u != v}
            graphs[int(path.stem)] = (np.asarray(nodes, dtype=float).reshape(-1, 2), sorted(edges))
        return graphs
    if not args.road_dt or not os.path.exists(args.road_dt):
        print(f"Warning: road prediction file not found: {args.road_dt}")
        return {}

    road_mod = load_road_stitch_module(args.road_stitch_script)
    road_coco = load_json(args.road_dt)
    category_id = None if args.road_category_id < 0 else args.road_category_id

    region_graphs = road_mod.build_region_graph(
        road_coco,
        score_threshold=args.road_score_threshold,
        category_id=category_id,
        crop_size_orig=args.road_crop_size_orig,
        patch_size_model=args.road_patch_size_model,
        stride=args.road_stride,
        core_keep_mode=args.road_core_keep_mode,
        core_margin_scale=args.road_core_margin_scale,
        core_buffer_px=args.road_core_buffer_px,
    )
    region_metas = road_mod.build_region_patch_meta(road_coco)

    if not args.merge_road:
        return region_graphs

    merged_graphs = {}
    for region_id, (raw_nodes, raw_edges) in region_graphs.items():
        nodes, edges = road_mod.merge_region_graph(
            raw_nodes,
            raw_edges,
            merge_node_dist=args.road_merge_node_dist,
            split_edge_dist=args.road_split_edge_dist,
            endpoint_snap_dist=args.road_endpoint_snap_dist,
        )
        if args.road_patch_boundary_bridge_dist > 0:
            edges = road_mod.bridge_endpoints_near_patch_boundaries(
                nodes,
                edges,
                region_metas.get(region_id, []),
                crop_size_orig=args.road_crop_size_orig,
                near_boundary_px=args.road_patch_boundary_near_px,
                bridge_dist=args.road_patch_boundary_bridge_dist,
            )
        merged_graphs[region_id] = (nodes, edges)

    print(
        f"Loaded road graphs from {args.road_dt}"
        f" | regions={len(merged_graphs)} | merged={args.merge_road}"
    )
    return merged_graphs


def resolve_region_image(region_id, region_image_dir):
    region_image_dir = Path(region_image_dir)
    candidates = [
        region_image_dir / f"{region_id}.png",
        region_image_dir / f"{region_id}.jpg",
        region_image_dir / f"{region_id}.jpeg",
        region_image_dir / f"region_{region_id}.png",
        region_image_dir / f"region_{region_id}.jpg",
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def create_blank_canvas(
    region_id,
    region_sizes,
    building_polygons,
    water_polygons,
    road_graphs,
    road_polygons=None,
    extra_polygon_layers=None,
    extra_line_layers=None,
):
    road_polygons = road_polygons or {}
    extra_polygon_layers = extra_polygon_layers or []
    extra_line_layers = extra_line_layers or []
    width, height = region_sizes.get(region_id, (0, 0))
    max_x = float(width)
    max_y = float(height)

    polygon_layers = [
        building_polygons.get(region_id, []),
        water_polygons.get(region_id, []),
        road_polygons.get(region_id, []),
    ]
    for layer in extra_polygon_layers:
        polygon_layers.append(layer.get(region_id, []))

    for polygons in polygon_layers:
        for pts in polygons:
            if pts.size:
                max_x = max(max_x, float(np.max(pts[:, 0])) + 1)
                max_y = max(max_y, float(np.max(pts[:, 1])) + 1)

    for layer in extra_line_layers:
        for pts in layer.get(region_id, []):
            if pts.size:
                max_x = max(max_x, float(np.max(pts[:, 0])) + 1)
                max_y = max(max_y, float(np.max(pts[:, 1])) + 1)

    if region_id in road_graphs:
        nodes, _ = road_graphs[region_id]
        if nodes is not None and len(nodes) > 0:
            max_y = max(max_y, float(np.max(nodes[:, 0])) + 1)
            max_x = max(max_x, float(np.max(nodes[:, 1])) + 1)

    width = max(512, int(np.ceil(max_x)))
    height = max(512, int(np.ceil(max_y)))
    return np.zeros((height, width, 3), dtype=np.uint8)


def render_multiclass_overlay(
    base,
    region_id,
    building_polygons,
    water_polygons,
    road_graphs=None,
    road_polygons=None,
    road_lines=None,
    draw_building=True,
    draw_water=True,
    draw_road=True,
    draw_roadline_nodes=True,
    road_area_buffer_thickness=ROAD_AREA_BUFFER_THICKNESS,
    internal_boundaries=None,
    boundary_tolerance=1,
    draw_legend_flag=False,
):
    vis = base.copy()
    road_graphs = road_graphs or {}
    road_polygons = road_polygons or {}
    road_lines = road_lines or {}

    if draw_water:
        vis = draw_polygon_layer(
            vis,
            water_polygons.get(region_id, []),
            fill_rgba=WATER_FILL,
            outline_rgba=WATER_OUTLINE,
            node_rgba=WATER_NODE,
            outline_thickness=WATER_OUTLINE_THICKNESS,
            node_radius=WATER_NODE_RADIUS,
            internal_boundaries=internal_boundaries,
            boundary_tolerance=boundary_tolerance,
        )

    if draw_road:
        if road_polygons.get(region_id):
            vis = draw_polygon_layer(
                vis,
                road_polygons.get(region_id, []),
                fill_rgba=ROAD_AREA_FILL,
                outline_rgba=ROAD_AREA_OUTLINE,
                node_rgba=None,
                outline_thickness=ROAD_LINE_THICKNESS,
                node_radius=ROAD_NODE_RADIUS,
                close_outline=False,
                draw_nodes=False,
                internal_boundaries=internal_boundaries,
                boundary_tolerance=boundary_tolerance,
            )
        if region_id in road_graphs:
            nodes, edges = road_graphs[region_id]
            vis = draw_road_graph_buffer(
                vis,
                nodes,
                edges,
                thickness=road_area_buffer_thickness,
            )

    if draw_road:
        if region_id in road_graphs:
            nodes, edges = road_graphs[region_id]
            vis = draw_road_graph(vis, nodes, edges, draw_nodes=draw_roadline_nodes)
        if road_lines.get(region_id):
            vis = draw_polylines_layer(
                vis,
                road_lines.get(region_id, []),
                line_rgba=ROAD_LINE,
                node_rgba=ROAD_NODE,
                thickness=ROAD_LINE_THICKNESS,
                node_radius=ROAD_NODE_RADIUS,
                draw_nodes=draw_roadline_nodes,
            )

    if draw_building:
        vis = draw_polygon_layer(
            vis,
            building_polygons.get(region_id, []),
            fill_rgba=BUILDING_FILL,
            outline_rgba=BUILDING_OUTLINE,
            node_rgba=BUILDING_NODE,
            outline_thickness=BUILDING_OUTLINE_THICKNESS,
            node_radius=BUILDING_NODE_RADIUS,
            internal_boundaries=internal_boundaries,
            boundary_tolerance=boundary_tolerance,
        )

    if draw_legend_flag:
        draw_legend(vis)

    return vis


def draw_legend(image):
    rows = [
        ("building", BUILDING_OUTLINE, BUILDING_FILL),
        ("water", WATER_OUTLINE, WATER_FILL),
        ("road area", ROAD_AREA_OUTLINE, ROAD_AREA_FILL),
        ("road line", ROAD_LINE, ROAD_LINE),
    ]
    x0, y0 = 10, 10
    row_h = 24
    box_w = 150
    box_h = 12 + row_h * len(rows)

    overlay = image.copy()
    cv2.rectangle(overlay, (0, 0), (box_w, box_h), (0, 0, 0), -1)
    image[:] = cv2.addWeighted(overlay, 0.45, image, 0.55, 0)

    for idx, (name, outline, fill) in enumerate(rows):
        y = y0 + idx * row_h
        cv2.rectangle(
            image,
            (x0, y + 3),
            (x0 + 14, y + 17),
            rgba_to_bgr(fill),
            -1,
            cv2.LINE_AA,
        )
        cv2.rectangle(
            image,
            (x0, y + 3),
            (x0 + 14, y + 17),
            rgba_to_bgr(outline),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            image,
            name,
            (x0 + 22, y + 16),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )


def choose_region_ids(all_region_ids, predicted_region_ids, args):
    if args.single_region is not None:
        return [args.single_region]

    candidates = sorted(all_region_ids if args.include_empty else predicted_region_ids)
    if args.max_regions > 0 and len(candidates) > args.max_regions:
        rng = random.Random(args.seed)
        candidates = sorted(rng.sample(candidates, args.max_regions))
    return candidates


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize building, waterbody and road predictions on one IRSAMap region image."
    )
    parser.add_argument("--building_dt", default=None, help="Building COCO detections; required unless --no_building")
    parser.add_argument("--water_dt", default=None, help="Waterbody COCO detections; required unless --no_water")
    parser.add_argument("--road_graph_dir", help="Directory of stitched graph/*.p used for evaluation (preferred)")
    parser.add_argument("--road_dt", default=None, help="Road full COCO detections; required unless --no_road")
    parser.add_argument("--patch_coco", required=True, help="Patch-level COCO JSON with region-aware file names")
    parser.add_argument("--gt", default=None, help="Optional multiclass COCO ground truth")
    parser.add_argument("--gt_out_dir", default=None)
    parser.add_argument("--gt_roadline_dir", default=None)
    parser.add_argument("--no_gt", action="store_true")
    parser.add_argument("--no_gt_roadline", action="store_true")
    parser.add_argument("--no_roadline_nodes", action="store_true")
    parser.add_argument("--draw_gt_roadline_nodes", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--no_gt_roadline_nodes", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--region_image_dir", required=True, help="Directory containing region-level images")
    parser.add_argument("--out_dir", required=True, help="Visualization output directory")
    parser.add_argument("--output_ext", default=".jpg", choices=[".jpg", ".jpeg", ".png", ".tif", ".tiff"])
    parser.add_argument("--dpi", type=int, default=SAVE_DPI)
    parser.add_argument("--max_regions", type=int, default=500)
    parser.add_argument("--single_region", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--include_empty", action="store_true")
    parser.add_argument("--draw_legend", action="store_true")

    parser.add_argument("--building_category_ids", default="1")
    parser.add_argument("--water_category_ids", default="2")
    parser.add_argument("--building_score_threshold", type=float, default=0.0)
    parser.add_argument("--water_score_threshold", type=float, default=0.0)

    parser.add_argument("--no_building", action="store_true")
    parser.add_argument("--no_water", action="store_true")
    parser.add_argument("--no_road", action="store_true")
    parser.add_argument("--draw_road_nodes", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--no_road_nodes", action="store_true", help=argparse.SUPPRESS)

    parser.add_argument("--road_stitch_script", default=path_arg(DEFAULT_ROAD_STITCH_SCRIPT))
    parser.add_argument("--road_score_threshold", type=float, default=0.0)
    parser.add_argument("--road_category_id", type=int, default=-1, help="-1 means all road categories.")
    parser.add_argument("--merge_road", action="store_true", default=True)
    parser.add_argument("--no_merge_road", action="store_false", dest="merge_road")
    parser.add_argument("--road_merge_node_dist", type=float, default=10.0)
    parser.add_argument("--road_split_edge_dist", type=float, default=3.0)
    parser.add_argument("--road_endpoint_snap_dist", type=float, default=10.0)
    parser.add_argument("--road_patch_boundary_bridge_dist", type=float, default=4.0)
    parser.add_argument("--road_patch_boundary_near_px", type=float, default=4.0)
    parser.add_argument("--road_crop_size_orig", type=int, default=128)
    parser.add_argument("--road_patch_size_model", type=int, default=256)
    parser.add_argument("--road_stride", type=int, default=64)
    parser.add_argument(
        "--road_core_keep_mode",
        default="endpoint_or_midpoint",
        choices=["midpoint", "endpoint_or_midpoint"],
    )
    parser.add_argument("--road_core_margin_scale", type=float, default=0.5)
    parser.add_argument("--road_core_buffer_px", type=float, default=2.0)
    parser.add_argument("--road_area_buffer_thickness", type=int, default=ROAD_AREA_BUFFER_THICKNESS)
    parser.add_argument(
        "--patch_seam_tolerance",
        type=int,
        default=1,
        help="Hide polygon outline/node segments on internal 512-patch seams within this pixel tolerance.",
    )
    parser.add_argument("--keep_patch_seams", action="store_true")
    args = parser.parse_args()
    if not args.no_building and not args.building_dt:
        parser.error("--building_dt is required unless --no_building is set")
    if not args.no_water and not args.water_dt:
        parser.error("--water_dt is required unless --no_water is set")
    if not args.no_road and not (args.road_dt or args.road_graph_dir):
        parser.error("--road_graph_dir or --road_dt is required unless --no_road is set")
    if not args.no_gt and not args.gt:
        parser.error("--gt is required unless --no_gt is set")
    if not args.no_gt and not args.no_gt_roadline and not args.gt_roadline_dir:
        parser.error("--gt_roadline_dir is required unless --no_gt or --no_gt_roadline is set")
    if args.draw_gt_roadline_nodes:
        args.no_roadline_nodes = False
    if args.draw_road_nodes:
        args.no_roadline_nodes = False
    if args.no_gt_roadline_nodes:
        args.no_roadline_nodes = True
    if args.no_road_nodes:
        args.no_roadline_nodes = True
    return args


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    gt_out_dir = Path(args.gt_out_dir) if args.gt_out_dir else Path(str(out_dir) + "_gt")
    if not args.no_gt:
        gt_out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Patch COCO: {args.patch_coco}")
    (
        patch_image_meta,
        region_sizes,
        region_to_image_ids,
        region_internal_boundaries,
    ) = load_patch_image_meta(args.patch_coco)
    print(f"Patch images: {len(patch_image_meta)} | regions: {len(region_to_image_ids)}")

    building_category_ids = parse_category_ids(args.building_category_ids)
    water_category_ids = parse_category_ids(args.water_category_ids)

    building_polygons = defaultdict(list)
    water_polygons = defaultdict(list)
    road_graphs = {}
    gt_building_polygons = defaultdict(list)
    gt_water_polygons = defaultdict(list)
    gt_road_polygons = defaultdict(list)
    gt_road_lines = defaultdict(list)

    if not args.no_building:
        building_polygons = build_region_polygons(
            args.building_dt,
            patch_image_meta,
            building_category_ids,
            args.building_score_threshold,
        )
    if not args.no_water:
        water_polygons = build_region_polygons(
            args.water_dt,
            patch_image_meta,
            water_category_ids,
            args.water_score_threshold,
        )
    if not args.no_road:
        road_graphs = build_road_graphs(args)

    if not args.no_gt:
        if not args.no_building:
            gt_building_polygons = build_region_polygons_from_coco(
                args.gt,
                patch_image_meta,
                {1},
            )
        if not args.no_water:
            gt_water_polygons = build_region_polygons_from_coco(
                args.gt,
                patch_image_meta,
                {2},
            )
        if not args.no_road:
            gt_road_polygons = build_region_polygons_from_coco(
                args.gt,
                patch_image_meta,
                {3},
            )
            if not args.no_gt_roadline:
                gt_road_lines = build_gt_roadlines(args.gt_roadline_dir)

    all_region_ids = set(region_to_image_ids.keys())
    all_region_ids.update(region_sizes.keys())
    all_region_ids.update(building_polygons.keys())
    all_region_ids.update(water_polygons.keys())
    all_region_ids.update(road_graphs.keys())
    all_region_ids.update(gt_building_polygons.keys())
    all_region_ids.update(gt_water_polygons.keys())
    all_region_ids.update(gt_road_polygons.keys())
    all_region_ids.update(gt_road_lines.keys())

    predicted_region_ids = set(building_polygons.keys())
    predicted_region_ids.update(water_polygons.keys())
    predicted_region_ids.update(road_graphs.keys())
    predicted_region_ids.update(gt_building_polygons.keys())
    predicted_region_ids.update(gt_water_polygons.keys())
    predicted_region_ids.update(gt_road_polygons.keys())
    predicted_region_ids.update(gt_road_lines.keys())

    selected_region_ids = choose_region_ids(all_region_ids, predicted_region_ids, args)
    print(f"Visualizing regions: {len(selected_region_ids)}")

    saved = 0
    gt_saved = 0
    skipped = 0
    gt_skipped = 0
    for idx, region_id in enumerate(selected_region_ids, start=1):
        image_path = resolve_region_image(region_id, args.region_image_dir)
        if image_path is not None:
            base = read_image_unicode(image_path)
        else:
            base = None

        if base is None:
            base = create_blank_canvas(
                region_id,
                region_sizes,
                building_polygons,
                water_polygons,
                road_graphs,
                gt_road_polygons,
                extra_polygon_layers=[
                    gt_building_polygons,
                    gt_water_polygons,
                ],
                extra_line_layers=[gt_road_lines],
            )
            print(f"Warning: using blank canvas for region={region_id}")

        vis = render_multiclass_overlay(
            base,
            region_id,
            building_polygons=building_polygons,
            water_polygons=water_polygons,
                road_graphs=road_graphs,
                road_polygons=None,
                road_lines=None,
                draw_building=not args.no_building,
                draw_water=not args.no_water,
                draw_road=not args.no_road,
                draw_roadline_nodes=not args.no_roadline_nodes,
                road_area_buffer_thickness=args.road_area_buffer_thickness,
                internal_boundaries=None if args.keep_patch_seams else region_internal_boundaries.get(region_id),
                boundary_tolerance=args.patch_seam_tolerance,
                draw_legend_flag=args.draw_legend,
            )

        out_name = f"{idx:03d}_region_{region_id}_multiclass{args.output_ext}"
        out_path = out_dir / out_name
        if write_image_with_dpi(out_path, vis, dpi=args.dpi):
            saved += 1
        else:
            skipped += 1
            print(f"Warning: failed to save {out_path}")

        if not args.no_gt:
            gt_vis = render_multiclass_overlay(
                base,
                region_id,
                building_polygons=gt_building_polygons,
                water_polygons=gt_water_polygons,
                road_graphs=None,
                road_polygons=gt_road_polygons,
                road_lines=gt_road_lines,
                draw_building=not args.no_building,
                draw_water=not args.no_water,
                draw_road=not args.no_road,
                draw_roadline_nodes=not args.no_roadline_nodes,
                road_area_buffer_thickness=args.road_area_buffer_thickness,
                internal_boundaries=None if args.keep_patch_seams else region_internal_boundaries.get(region_id),
                boundary_tolerance=args.patch_seam_tolerance,
                draw_legend_flag=args.draw_legend,
            )
            gt_out_name = f"{idx:03d}_region_{region_id}_multiclass_gt{args.output_ext}"
            gt_out_path = gt_out_dir / gt_out_name
            if write_image_with_dpi(gt_out_path, gt_vis, dpi=args.dpi):
                gt_saved += 1
            else:
                gt_skipped += 1
                print(f"Warning: failed to save {gt_out_path}")

    print(f"Done. pred_saved={saved}, pred_skipped={skipped}, out_dir={out_dir}")
    if not args.no_gt:
        print(f"GT done. gt_saved={gt_saved}, gt_skipped={gt_skipped}, gt_out_dir={gt_out_dir}")


if __name__ == "__main__":
    main()
