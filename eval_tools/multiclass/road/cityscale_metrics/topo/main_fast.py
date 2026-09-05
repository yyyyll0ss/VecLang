import argparse
import math
import os
import pickle
import time
from pathlib import Path

import graph_fast as splfy
import topo_fast as topo



LAT_TOP_LEFT = 41.0
LON_TOP_LEFT = -71.0


def parse_tiles_arg(tiles_arg):
    if not tiles_arg or not tiles_arg.strip():
        raise ValueError("Pass explicit IRSAMap -tiles or use 3.run_irsamap_metrics.py")
    return [int(x.strip()) for x in tiles_arg.split(",") if x.strip()]


def resolve_savedir(savedir_arg):
    base_dir = Path(__file__).resolve().parents[1]
    project_dir = base_dir.parent
    p = Path(savedir_arg)
    if p.is_absolute():
        return p
    return (project_dir / p).resolve()


def xy2latlon(x, y):
    lat = LAT_TOP_LEFT - x / 111111.0
    lon = LON_TOP_LEFT + (y / 111111.0) / math.cos(math.radians(LAT_TOP_LEFT))
    return lat, lon


def load_pickle_graph(graph_path):
    with open(graph_path, "rb") as f:
        return pickle.load(f)


def create_graph(neighbors):
    graph = splfy.RoadGraph()
    node_id_map = {}
    next_id = 0
    min_lat = LAT_TOP_LEFT
    max_lon = LON_TOP_LEFT

    for n1, adj_nodes in neighbors.items():
        lat1, lon1 = xy2latlon(n1[0], n1[1])
        min_lat = min(min_lat, lat1)
        max_lon = max(max_lon, lon1)

        if n1 not in node_id_map:
            node_id_map[n1] = next_id
            next_id += 1
        id1 = node_id_map[n1]

        for n2 in adj_nodes:
            lat2, lon2 = xy2latlon(n2[0], n2[1])
            if n2 not in node_id_map:
                node_id_map[n2] = next_id
                next_id += 1
            id2 = node_id_map[n2]
            graph.addEdge(id1, lat1, lon1, id2, lat2, lon2)

    graph.ReverseDirectionLink()

    for node in graph.nodes.keys():
        graph.nodeScore[node] = 100
    for edge in graph.edges.keys():
        graph.edgeScore[edge] = 100

    return graph, min_lat, max_lon


def evaluate_graph_pair(graph_gt_path, graph_prop_path, output_path, matching_threshold, topo_interval):
    map_gt = load_pickle_graph(graph_gt_path)
    map_prop = load_pickle_graph(graph_prop_path)

    graph_gt, min_lat_gt, max_lon_gt = create_graph(map_gt)
    graph_prop, min_lat_prop, max_lon_prop = create_graph(map_prop)

    min_lat = min(min_lat_gt, min_lat_prop)
    max_lon = max(max_lon_gt, max_lon_prop)

    region = [
        min_lat - 300.0 / 111111.0,
        LON_TOP_LEFT - 500.0 / 111111.0,
        LAT_TOP_LEFT + 300.0 / 111111.0,
        max_lon + 500.0 / 111111.0,
    ]

    graph_gt.region = region
    graph_prop.region = region

    losm = topo.TOPOGenerateStartingPoints(
        graph_gt,
        region=region,
        image="NULL",
        check=False,
        direction=False,
        metaData=None,
    )
    lmap = topo.TOPOGeneratePairs(
        graph_prop,
        graph_gt,
        losm,
        threshold=0.00010,
        region=region,
    )

    r = 0.00300
    if LAT_TOP_LEFT - min_lat < 0.01000:
        r = 0.00150

    topo_result = topo.TOPOWithPairs(
        graph_prop,
        graph_gt,
        lmap,
        losm,
        r=r,
        step=topo_interval,
        threshold=matching_threshold,
        outputfile=str(output_path),
        one2oneMatching=True,
        metaData=None,
    )

    with open(str(output_path).replace("txt", "topo.p"), "wb") as f:
        pickle.dump([losm, topo_result, region], f)


def run_tile(tile_idx, save_root, gt_root, matching_threshold, topo_interval):
    graph_prop_path = save_root / "graph" / f"{tile_idx}.p"
    graph_gt_path = gt_root / f"region_{tile_idx}_graph_gt.pickle"
    output_path = save_root / "results" / "topo" / f"{tile_idx}.txt"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not graph_prop_path.exists():
        print(f"[skip] missing prediction graph: {graph_prop_path}")
        return False
    if not graph_gt_path.exists():
        print(f"[skip] missing gt graph: {graph_gt_path}")
        return False

    print(f"[run] tile={tile_idx}")
    start = time.perf_counter()
    evaluate_graph_pair(
        graph_gt_path,
        graph_prop_path,
        output_path,
        matching_threshold,
        topo_interval,
    )
    print(f"[done] tile={tile_idx} elapsed={time.perf_counter() - start:.2f}s")
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-graph_gt", default="", action="store", dest="graph_gt", type=str)
    parser.add_argument("-graph_prop", default="", action="store", dest="graph_prop", type=str)
    parser.add_argument("-output", default="", action="store", dest="output", type=str)
    parser.add_argument(
        "-matching_threshold",
        action="store",
        dest="matching_threshold",
        type=float,
        default=0.00010,
    )
    parser.add_argument(
        "-interval",
        action="store",
        dest="topo_interval",
        type=float,
        default=0.00005,
    )
    parser.add_argument(
        "-savedir",
        type=str,
        required=True,
    )
    parser.add_argument(
        "-gt_root",
        type=str,
        required=True,
    )
    parser.add_argument(
        "-tiles",
        type=str,
        default="",
        help="Comma-separated tile list. Required for batch evaluation; the IRSAMap runner discovers regions automatically.",
    )
    args = parser.parse_args()
    print(args)

    if args.graph_gt and args.graph_prop and args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        evaluate_graph_pair(
            Path(args.graph_gt),
            Path(args.graph_prop),
            output_path,
            args.matching_threshold,
            args.topo_interval,
        )
        return

    save_root = resolve_savedir(args.savedir)
    gt_root = Path(args.gt_root).resolve()
    tile_list = parse_tiles_arg(args.tiles)

    for tile_idx in tile_list:
        run_tile(tile_idx, save_root, gt_root, args.matching_threshold, args.topo_interval)


if __name__ == "__main__":
    main()
