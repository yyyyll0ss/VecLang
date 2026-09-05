import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np


TILES = [8, 9, 19, 28, 29, 39, 48, 49, 59, 68, 69, 79, 88, 89, 99, 108, 109, 119, 128, 129, 139, 148, 149, 159, 168, 169, 179]


def run_cmd(cmd, cwd):
    print("[run]", " ".join(cmd))
    start = time.perf_counter()
    subprocess.run(cmd, cwd=str(cwd), check=True)
    elapsed = time.perf_counter() - start
    print(f"[done] {elapsed:.2f}s")
    return elapsed


def parse_tiles_arg(tiles_arg):
    if not tiles_arg or not tiles_arg.strip():
        return TILES
    return [int(x.strip()) for x in tiles_arg.split(",") if x.strip()]


def resolve_savedir(project_dir, savedir_arg):
    p = Path(savedir_arg)
    if p.is_absolute():
        return p.resolve()
    return (project_dir / p).resolve()


def resolve_go_bin(go_bin_arg):
    p = Path(go_bin_arg)
    if p.is_absolute() and p.exists():
        return str(p)

    found = shutil.which(go_bin_arg)
    if found:
        return found

    for candidate in (Path(r"C:\Program Files\Go\bin\go.exe"), Path(r"C:\Go\bin\go.exe")):
        if candidate.exists():
            return str(candidate)

    raise FileNotFoundError(
        "未找到 Go 可执行文件。请安装 Go，或通过 --go-bin 传入完整路径，例如："
        '--go-bin "C:\\Program Files\\Go\\bin\\go.exe"'
    )


def resolve_worker_count(requested):
    if requested and requested > 0:
        return requested
    cpu_count = os.cpu_count() or 1
    return max(1, min(4, cpu_count))


def get_mtime(path):
    return path.stat().st_mtime if path.exists() else None


def is_up_to_date(output_paths, input_paths):
    outputs = [Path(p) for p in output_paths]
    inputs = [Path(p) for p in input_paths]

    if not outputs or not inputs:
        return False
    if any(not p.exists() for p in outputs):
        return False
    if any(not p.exists() for p in inputs):
        return False

    oldest_output = min(get_mtime(p) for p in outputs)
    newest_input = max(get_mtime(p) for p in inputs)
    return oldest_output >= newest_input


def aggregate_apls_subset(save_root, tile_list):
    apls_dir = save_root / "results" / "apls"
    if not apls_dir.exists():
        raise FileNotFoundError(f"APLS result directory not found: {apls_dir}")

    values = []
    valid_tiles = []
    for tile in tile_list:
        fp = apls_dir / f"{tile}.txt"
        if not fp.exists():
            print(f"[skip] missing APLS result: {fp}")
            continue
        with open(fp, "r", encoding="utf-8") as f:
            lines = f.readlines()
        values.append(float(lines[0].split(" ")[-1][:-2]))
        valid_tiles.append(tile)

    if not values:
        raise RuntimeError("未匹配到任何 APLS 结果，请检查 --tiles 参数和结果目录。")

    mean_apls = float(np.mean(values))
    print("APLS mean", mean_apls)
    score_dir = save_root / "score"
    score_dir.mkdir(parents=True, exist_ok=True)
    with open(score_dir / "apls.json", "w", encoding="utf-8") as jf:
        json.dump({"apls": values, "final_APLS": mean_apls, "tiles": valid_tiles}, jf)


def aggregate_topo_subset(save_root, tile_list):
    topo_dir = save_root / "results" / "topo"
    if not topo_dir.exists():
        raise FileNotFoundError(f"TOPO result directory not found: {topo_dir}")

    topo_vals = []
    prec_vals = []
    rec_vals = []
    valid_tiles = []
    for tile in tile_list:
        fp = topo_dir / f"{tile}.txt"
        if not fp.exists():
            print(f"[skip] missing TOPO result: {fp}")
            continue
        with open(fp, "r", encoding="utf-8") as f:
            lines = f.readlines()
        p = float(lines[-1].split(" ")[0].split("=")[-1])
        r = float(lines[-1].split(" ")[-1].split("=")[-1])
        f1 = 0.0 if (p + r) == 0 else (2 * p * r / (p + r))
        topo_vals.append(f1)
        prec_vals.append(p)
        rec_vals.append(r)
        valid_tiles.append(tile)

    if not topo_vals:
        raise RuntimeError("未匹配到任何 TOPO 结果，请检查 --tiles 参数和结果目录。")

    topo_mean = float(np.mean(topo_vals))
    prec_mean = float(np.mean(prec_vals))
    rec_mean = float(np.mean(rec_vals))
    print("TOPO", topo_mean, "Precision", prec_mean, "Recall", rec_mean)

    score_dir = save_root / "score"
    score_dir.mkdir(parents=True, exist_ok=True)
    with open(score_dir / "topo.json", "w", encoding="utf-8") as jf:
        json.dump(
            {
                "mean topo": [topo_mean, prec_mean, rec_mean],
                "prec": prec_vals,
                "recall": rec_vals,
                "f1": topo_vals,
                "tiles": valid_tiles,
            },
            jf,
        )


def build_apls_binary(metrics_dir, go_bin, apls_subdir):
    apls_dir = metrics_dir / apls_subdir
    bin_name = "apls_eval_rtreego.exe" if "rtreego" in apls_subdir else "apls_eval.exe"
    bin_path = apls_dir / bin_name
    source_files = list(apls_dir.rglob("*.go"))
    for extra_name in ("go.mod", "go.sum"):
        extra = apls_dir / extra_name
        if extra.exists():
            source_files.append(extra)

    if is_up_to_date([bin_path], source_files):
        print(f"[cache] reuse APLS binary: {bin_path}")
        return bin_path

    run_cmd([go_bin, "build", "-o", str(bin_path), "."], cwd=apls_dir)
    return bin_path


def run_topo_one_tile(metrics_dir, save_root, gt_root, tile):
    pred = save_root / "graph" / f"{tile}.p"
    gt = gt_root / f"region_{tile}_graph_gt.pickle"
    result_dir = save_root / "results" / "topo"
    result_dir.mkdir(parents=True, exist_ok=True)
    out_txt = result_dir / f"{tile}.txt"
    out_pickle = result_dir / f"{tile}.topo.p"
    topo_script = metrics_dir / "topo" / "main_fast.py"
    topo_deps = [topo_script, metrics_dir / "topo" / "topo_fast.py", metrics_dir / "topo" / "graph_fast.py"]

    if not pred.exists():
        print(f"[skip] missing prediction graph: {pred}")
        return tile, "missing_pred"
    if not gt.exists():
        print(f"[skip] missing gt graph: {gt}")
        return tile, "missing_gt"

    if is_up_to_date([out_txt, out_pickle], [pred, gt, *topo_deps]):
        print(f"[cache] reuse TOPO tile={tile}")
        return tile, "cached"

    run_cmd(
        [
            sys.executable,
            str(topo_script),
            "-savedir",
            str(save_root),
            "-gt_root",
            str(gt_root),
            "-tiles",
            str(tile),
        ],
        cwd=metrics_dir,
    )
    return tile, "done"


def run_topo(metrics_dir, save_root, gt_root, tile_list, workers=1):
    worker_count = resolve_worker_count(workers)
    start = time.perf_counter()
    if worker_count <= 1:
        for tile in tile_list:
            run_topo_one_tile(metrics_dir, save_root, gt_root, tile)
    else:
        print(f"[info] TOPO 并行 worker 数: {worker_count}")
        with ThreadPoolExecutor(max_workers=worker_count) as ex:
            futs = [ex.submit(run_topo_one_tile, metrics_dir, save_root, gt_root, tile) for tile in tile_list]
            for fut in as_completed(futs):
                tile_done, status = fut.result()
                print(f"[done] TOPO tile={tile_done} status={status}")

    aggregate_topo_subset(save_root, tile_list)
    print(f"[info] TOPO total elapsed: {time.perf_counter() - start:.2f}s")


def run_apls(metrics_dir, gt_root, save_root, go_bin, workers=1, apls_subdir="apls_rtreego_fast", tile_list=None):
    start = time.perf_counter()
    pred_base = save_root / "graph"
    result_dir = save_root / "results" / "apls"
    result_dir.mkdir(parents=True, exist_ok=True)
    apls_bin = build_apls_binary(metrics_dir, go_bin, apls_subdir)
    convert_script = metrics_dir / apls_subdir / "convert.py"

    active_tiles = tile_list if tile_list is not None else TILES
    tasks = []
    for tile in active_tiles:
        pred = pred_base / f"{tile}.p"
        gt = gt_root / f"region_{tile}_graph_gt.pickle"
        if not pred.exists():
            print(f"[skip] missing prediction graph: {pred}")
            continue
        if not gt.exists():
            print(f"[skip] missing gt graph: {gt}")
            continue
        tasks.append((tile, gt, pred))

    def run_one_tile(item):
        tile, gt, pred = item
        gt_json = result_dir / f"{tile}_gt.json"
        prop_json = result_dir / f"{tile}_prop.json"
        out_txt = result_dir / f"{tile}.txt"

        if not is_up_to_date([gt_json], [gt, convert_script]):
            run_cmd([sys.executable, str(convert_script), str(gt), str(gt_json)], cwd=metrics_dir)
        else:
            print(f"[cache] reuse gt json tile={tile}")

        if not is_up_to_date([prop_json], [pred, convert_script]):
            run_cmd([sys.executable, str(convert_script), str(pred), str(prop_json)], cwd=metrics_dir)
        else:
            print(f"[cache] reuse pred json tile={tile}")

        if is_up_to_date([out_txt], [gt_json, prop_json, apls_bin]):
            print(f"[cache] reuse APLS tile={tile}")
            return tile, "cached"

        run_cmd([str(apls_bin), str(gt_json), str(prop_json), str(out_txt)], cwd=metrics_dir)
        return tile, "done"

    worker_count = resolve_worker_count(workers)
    if worker_count <= 1:
        for task in tasks:
            run_one_tile(task)
    else:
        print(f"[info] APLS 并行 worker 数: {worker_count}")
        with ThreadPoolExecutor(max_workers=worker_count) as ex:
            futs = [ex.submit(run_one_tile, t) for t in tasks]
            for fut in as_completed(futs):
                tile_done, status = fut.result()
                print(f"[done] APLS tile={tile_done} status={status}")

    aggregate_apls_subset(save_root, active_tiles)
    print(f"[info] APLS total elapsed: {time.perf_counter() - start:.2f}s")


def main():
    parser = argparse.ArgumentParser(description="Run CityScale TOPO/APLS fast metrics in one command.")
    parser.add_argument(
        "--savedir",
        required=True,
        help="Prediction output directory (absolute path or path relative to graph_based_eval).",
    )
    parser.add_argument(
        "--gt-root",
        required=True,
        help="Ground-truth graph root directory containing region_xxx_graph_gt.pickle.",
    )
    parser.add_argument("--only", choices=["all", "topo", "apls"], default="all")
    parser.add_argument("--go-bin", default="go", help="Go executable name/path, default: go")
    parser.add_argument("--workers", type=int, default=4, help="APLS 并行 tile 数；传 0 表示自动选择")
    parser.add_argument("--topo-workers", type=int, default=4, help="TOPO 并行 tile 数；传 0 表示自动选择")
    parser.add_argument("--tiles", type=str, default="", help="逗号分隔样本列表，例如 9,19,29,59,89,129")
    parser.add_argument(
        "--apls-version",
        choices=["rtreego_fast"],
        default="rtreego_fast",
        help="APLS 版本选择：offline=无外部依赖版本，rtreego=使用 rtreego 版本",
    )
    args = parser.parse_args()

    overall_start = time.perf_counter()
    project_dir = Path(__file__).resolve().parent
    metrics_dir = project_dir / "cityscale_metrics"
    save_root = resolve_savedir(project_dir, args.savedir)
    gt_root = Path(args.gt_root).resolve()
    go_bin = resolve_go_bin(args.go_bin) if args.only in ("all", "apls") else args.go_bin
    tile_list = parse_tiles_arg(args.tiles)

    print(f"[info] metrics_dir={metrics_dir}")
    print(f"[info] project_dir={project_dir}")
    print(f"[info] save_root={save_root}")
    print(f"[info] gt_root={gt_root}")
    print(f"[info] tiles={tile_list}")
    if args.only in ("all", "topo"):
        print(f"[info] topo_workers={resolve_worker_count(args.topo_workers)}")
    if args.only in ("all", "apls"):
        print(f"[info] go_bin={go_bin}")
        print(f"[info] apls_workers={resolve_worker_count(args.workers)}")

    if args.only in ("all", "topo"):
        run_topo(
            metrics_dir,
            save_root,
            gt_root,
            tile_list,
            workers=args.topo_workers,
        )

    if args.only in ("all", "apls"):
        apls_subdir = "apls_rtreego_fast"
        run_apls(
            metrics_dir,
            gt_root,
            save_root,
            go_bin,
            workers=args.workers,
            apls_subdir=apls_subdir,
            tile_list=tile_list,
        )

    print(f"[done] metric evaluation finished. total={time.perf_counter() - overall_start:.2f}s")


if __name__ == "__main__":
    main()
