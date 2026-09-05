import argparse
import json
import math
import pickle
from collections import defaultdict
from pathlib import Path

from tqdm import tqdm




def is_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def normalize_line_coords(coords):
    if not isinstance(coords, list) or len(coords) < 2:
        return []

    if all(is_number(v) for v in coords):
        pairs = []
        for i in range(0, len(coords) - 1, 2):
            pairs.append((float(coords[i]), float(coords[i + 1])))
        return pairs

    pairs = []
    for pt in coords:
        if (
            isinstance(pt, (list, tuple))
            and len(pt) >= 2
            and is_number(pt[0])
            and is_number(pt[1])
        ):
            pairs.append((float(pt[0]), float(pt[1])))
    return pairs


def iter_lines_from_geometry(geometry):
    if not isinstance(geometry, dict):
        return

    geom_type = geometry.get("type")
    coords = geometry.get("coordinates")

    if geom_type == "LineString":
        line = normalize_line_coords(coords)
        if len(line) >= 2:
            yield line
    elif geom_type == "MultiLineString":
        for item in coords or []:
            line = normalize_line_coords(item)
            if len(line) >= 2:
                yield line
    elif geom_type == "GeometryCollection":
        for item in geometry.get("geometries", []) or []:
            yield from iter_lines_from_geometry(item)


def clip_value(value, min_value, max_value):
    return min(max(value, min_value), max_value)


def point_to_node(point, clip, clip_min, clip_max):
    x, y = point
    if clip:
        x = clip_value(x, clip_min, clip_max)
        y = clip_value(y, clip_min, clip_max)
    return int(round(y)), int(round(x))


def add_edge(neighbors, node_a, node_b):
    if node_a == node_b:
        return False
    neighbors[node_a].add(node_b)
    neighbors[node_b].add(node_a)
    return True


def geojson_to_sat2graph(geojson_path, clip=True, clip_min=0.0, clip_max=1024.0):
    with open(geojson_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if data.get("type") == "FeatureCollection":
        features = data.get("features", [])
    elif data.get("type") == "Feature":
        features = [data]
    else:
        features = [{"geometry": data}]

    neighbors = defaultdict(set)
    line_count = 0
    edge_count = 0
    skipped_lines = 0

    for feature in features:
        geometry = feature.get("geometry", {}) if isinstance(feature, dict) else {}
        for line in iter_lines_from_geometry(geometry):
            nodes = [point_to_node(pt, clip, clip_min, clip_max) for pt in line]
            nodes = [node for node in nodes if all(math.isfinite(v) for v in node)]
            if len(nodes) < 2:
                skipped_lines += 1
                continue

            line_count += 1
            for node in nodes:
                neighbors.setdefault(node, set())
            for node_a, node_b in zip(nodes[:-1], nodes[1:]):
                if add_edge(neighbors, node_a, node_b):
                    edge_count += 1

    sat2graph = {
        node: sorted(adjacent)
        for node, adjacent in sorted(neighbors.items(), key=lambda item: item[0])
    }
    unique_edges = sum(len(v) for v in sat2graph.values()) // 2
    return sat2graph, {
        "line_count": line_count,
        "node_count": len(sat2graph),
        "edge_count": unique_edges,
        "raw_edge_count": edge_count,
        "skipped_lines": skipped_lines,
    }


def convert_directory(geojson_dir, output_root, overwrite=True, clip=True, clip_max=1024.0):
    geojson_dir = Path(geojson_dir)
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    geojson_files = sorted(geojson_dir.glob("*.geojson"), key=lambda p: int(p.stem) if p.stem.isdigit() else p.stem)
    if not geojson_files:
        raise FileNotFoundError(f"No .geojson files found in: {geojson_dir}")

    summary = {
        "geojson_dir": str(geojson_dir),
        "output_root": str(output_root),
        "clip": clip,
        "clip_max": clip_max,
        "files": [],
        "total_nodes": 0,
        "total_edges": 0,
        "converted": 0,
        "skipped_existing": 0,
    }

    for geojson_path in tqdm(geojson_files, desc="Converting GT graphs"):
        region_id = geojson_path.stem
        output_path = output_root / f"region_{region_id}_graph_gt.pickle"
        if output_path.exists() and not overwrite:
            summary["skipped_existing"] += 1
            continue

        sat2graph, stats = geojson_to_sat2graph(
            geojson_path,
            clip=clip,
            clip_min=0.0,
            clip_max=clip_max,
        )
        with open(output_path, "wb") as f:
            pickle.dump(sat2graph, f)

        item = {
            "region_id": region_id,
            "output": str(output_path),
            **stats,
        }
        summary["files"].append(item)
        summary["total_nodes"] += stats["node_count"]
        summary["total_edges"] += stats["edge_count"]
        summary["converted"] += 1

    summary_path = output_root / "conversion_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"Converted files: {summary['converted']}")
    print(f"Total nodes: {summary['total_nodes']}")
    print(f"Total edges: {summary['total_edges']}")
    print(f"Summary saved to: {summary_path}")
    return summary


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert IRSAMap roadline GeoJSON labels to CityScale-style sat2graph GT pickles."
    )
    parser.add_argument(
        "--geojson-dir",
        required=True,
    )
    parser.add_argument(
        "--output-root",
        required=True,
        help="Output root. Uses CityScale graph filename format: region_xxx_graph_gt.pickle.",
    )
    parser.add_argument("--clip-max", type=float, default=1024.0)
    parser.add_argument("--no-clip", action="store_true")
    parser.add_argument("--no-overwrite", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    convert_directory(
        geojson_dir=args.geojson_dir,
        output_root=args.output_root,
        overwrite=not args.no_overwrite,
        clip=not args.no_clip,
        clip_max=args.clip_max,
    )
