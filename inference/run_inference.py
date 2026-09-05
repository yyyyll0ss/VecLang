#!/usr/bin/env python3
"""Run reproducible VecLang inference through LLaMA-Factory.

The source manifests under ``VecLang/dataset/test`` are never rewritten. A
small LLaMA-Factory ``dataset_info.json`` and a resolved YAML config are created
inside each timestamped run directory instead.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml


INFERENCE_DIR = Path(__file__).resolve().parent
VECLANG_DIR = INFERENCE_DIR.parent
REPO_DIR = VECLANG_DIR.parent
CONFIG_DIR = INFERENCE_DIR / "configs"
DEFAULT_CATALOG = CONFIG_DIR / "datasets.yaml"
DEFAULT_OUTPUT_ROOT = Path(
    os.getenv("VECLANG_OUTPUT_ROOT", str(VECLANG_DIR.parent / "VecLang_outputs"))
)


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    manifest: Path
    records: int


@dataclass(frozen=True)
class TaskSpec:
    name: str
    config: Path
    media_dir: Path
    datasets: tuple[DatasetSpec, ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--task",
        choices=("object_detection", "attributes_generation", "all"),
        default="all",
        help="Task group to run. The default runs both groups sequentially.",
    )
    parser.add_argument(
        "--datasets",
        nargs="*",
        help="Optional dataset names, separated by spaces or commas. Use --list-datasets to inspect names.",
    )
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--model-path",
        default=os.getenv("VECLANG_MODEL_PATH"),
        help=(
            "Local checkpoint path or Hugging Face repo ID. It overrides "
            "model_name_or_path in the task config."
        ),
    )
    parser.add_argument("--run-name", help="Optional run directory name; defaults to a timestamp.")
    parser.add_argument("--max-samples", type=int, help="Maximum examples per selected dataset.")
    parser.add_argument("--gpus", help="CUDA device list, for example 0 or 0,1,2,3.")
    parser.add_argument("--llamafactory-cli", type=Path, help="Explicit LLaMA-Factory executable.")
    parser.add_argument("--list-datasets", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Validate and print configs without inference.")
    parser.add_argument("--check-all-images", action="store_true", help="Stat every referenced image during preflight.")
    parser.add_argument("--skip-gpu-check", action="store_true", help="Only for launchers where nvidia-smi is unavailable.")
    parser.add_argument("--no-split", action="store_true", help="Do not split combined predictions by dataset.")
    return parser.parse_args()


def absolute_from(base: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def load_catalog(path: Path) -> dict[str, TaskSpec]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or not isinstance(raw.get("tasks"), dict):
        raise ValueError(f"Invalid dataset catalog: {path}")
    dataset_root = absolute_from(path.parent, raw["dataset_root"])
    tasks: dict[str, TaskSpec] = {}
    all_names: set[str] = set()
    for task_name, task_raw in raw["tasks"].items():
        datasets: list[DatasetSpec] = []
        for row in task_raw.get("datasets", []):
            name = str(row["name"])
            if name in all_names:
                raise ValueError(f"Duplicate dataset name in catalog: {name}")
            all_names.add(name)
            datasets.append(
                DatasetSpec(
                    name=name,
                    manifest=absolute_from(dataset_root, row["manifest"]),
                    records=int(row["records"]),
                )
            )
        tasks[task_name] = TaskSpec(
            name=task_name,
            config=absolute_from(CONFIG_DIR, task_raw["config"]),
            media_dir=absolute_from(dataset_root, task_raw["media_dir"]),
            datasets=tuple(datasets),
        )
    return tasks


def requested_names(values: list[str] | None) -> set[str] | None:
    if not values:
        return None
    names = {name.strip() for value in values for name in value.split(",") if name.strip()}
    return names or None


def select_tasks(tasks: dict[str, TaskSpec], task_name: str, names: set[str] | None) -> list[TaskSpec]:
    selected_task_names = list(tasks) if task_name == "all" else [task_name]
    known_names = {dataset.name for task in tasks.values() for dataset in task.datasets}
    if names:
        unknown = sorted(names - known_names)
        if unknown:
            raise ValueError(f"Unknown dataset name(s): {', '.join(unknown)}")
    selected: list[TaskSpec] = []
    for name in selected_task_names:
        task = tasks[name]
        datasets = tuple(dataset for dataset in task.datasets if names is None or dataset.name in names)
        if datasets:
            selected.append(TaskSpec(task.name, task.config, task.media_dir, datasets))
    if not selected:
        raise ValueError("No datasets selected for the requested task.")
    return selected


def validate_model(config: dict[str, Any]) -> None:
    model_value = str(config["model_name_or_path"])
    model_dir = Path(model_value).expanduser()
    if not model_dir.is_absolute() and not model_dir.exists():
        # A non-local value such as "yyyllll/VecLang-4B" is resolved by
        # Transformers/Hugging Face Hub when inference starts.
        if "/" in model_value and not model_value.startswith(("./", "../")):
            return
        model_dir = (REPO_DIR / model_dir).resolve()
    else:
        model_dir = model_dir.resolve()
    index_path = model_dir / "model.safetensors.index.json"
    if not index_path.is_file():
        raise FileNotFoundError(f"Model index not found: {index_path}")
    index = json.loads(index_path.read_text(encoding="utf-8"))
    missing = sorted({name for name in index.get("weight_map", {}).values() if not (model_dir / name).is_file()})
    if missing:
        raise FileNotFoundError(f"Missing model shards: {missing}")


def manifest_rows(spec: DatasetSpec, media_dir: Path, check_all_images: bool) -> int:
    if not spec.manifest.is_file():
        raise FileNotFoundError(f"Manifest not found: {spec.manifest}")
    data = json.loads(spec.manifest.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"Manifest is not a JSON list: {spec.manifest}")
    if len(data) != spec.records:
        raise ValueError(f"Catalog count mismatch for {spec.name}: expected={spec.records}, actual={len(data)}")
    indexes = range(len(data)) if check_all_images else sorted({0, max(0, len(data) // 2), max(0, len(data) - 1)})
    for index in indexes:
        images = data[index].get("images") if isinstance(data[index], dict) else None
        if not images or not isinstance(images[0], str):
            raise ValueError(f"Missing image path in {spec.name}, record {index}")
        image_path = Path(images[0])
        image_path = image_path if image_path.is_absolute() else media_dir / image_path
        if not image_path.is_file():
            raise FileNotFoundError(f"Image not found in {spec.name}, record {index}: {image_path}")
    return len(data)


def dataset_info(datasets: tuple[DatasetSpec, ...]) -> dict[str, Any]:
    shared = {
        "formatting": "sharegpt",
        "columns": {"messages": "messages", "images": "images"},
        "tags": {
            "role_tag": "role",
            "content_tag": "content",
            "user_tag": "user",
            "assistant_tag": "assistant",
        },
    }
    return {dataset.name: {"file_name": str(dataset.manifest), **shared} for dataset in datasets}


def find_cli(explicit: Path | None) -> Path:
    if explicit:
        candidate = explicit.expanduser().resolve()
    elif os.getenv("LLAMAFACTORY_CLI"):
        candidate = Path(os.environ["LLAMAFACTORY_CLI"]).expanduser().resolve()
    elif shutil.which("llamafactory-cli"):
        candidate = Path(shutil.which("llamafactory-cli") or "").resolve()
    else:
        raise FileNotFoundError(
            "llamafactory-cli was not found on PATH. Activate the inference "
            "environment or set LLAMAFACTORY_CLI."
        )
    if not candidate.is_file() or not os.access(candidate, os.X_OK):
        raise FileNotFoundError(f"llamafactory-cli is not executable: {candidate}")
    return candidate


def gpu_preflight(skip: bool) -> None:
    if skip:
        return
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,name,memory.total", "--format=csv,noheader"],
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0 or not result.stdout.strip():
        detail = result.stderr.strip() or "no visible GPU"
        raise RuntimeError(f"NVIDIA GPU preflight failed: {detail}")


def build_config(
    task: TaskSpec,
    dataset_dir: Path,
    output_dir: Path,
    max_samples: int | None,
    model_path: str | None,
) -> dict[str, Any]:
    config = yaml.safe_load(task.config.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError(f"Invalid task config: {task.config}")
    config.update(
        {
            "eval_dataset": ",".join(dataset.name for dataset in task.datasets),
            "dataset_dir": str(dataset_dir),
            "media_dir": str(task.media_dir),
            "output_dir": str(output_dir),
            "overwrite_output_dir": False,
        }
    )
    if model_path:
        candidate = Path(model_path).expanduser()
        config["model_name_or_path"] = (
            str(candidate.resolve()) if candidate.exists() else model_path
        )
    if max_samples is not None:
        if max_samples <= 0:
            raise ValueError("--max-samples must be positive.")
        config["max_samples"] = max_samples
    validate_model(config)
    return config


def split_predictions(raw_path: Path, output_dir: Path, datasets: tuple[DatasetSpec, ...], max_samples: int) -> dict[str, Any]:
    expected = {dataset.name: min(dataset.records, max_samples) for dataset in datasets}
    actual_total = sum(1 for line in raw_path.open(encoding="utf-8") if line.strip())
    expected_total = sum(expected.values())
    if actual_total != expected_total:
        raise ValueError(f"Prediction count mismatch: expected={expected_total}, actual={actual_total}")
    output_dir.mkdir(parents=True, exist_ok=False)
    summary: dict[str, Any] = {"source": str(raw_path), "total": actual_total, "datasets": {}}
    with raw_path.open(encoding="utf-8") as source:
        for dataset in datasets:
            path = output_dir / f"{dataset.name}.jsonl"
            written = 0
            with path.open("w", encoding="utf-8") as target:
                while written < expected[dataset.name]:
                    line = source.readline()
                    if not line:
                        raise ValueError(f"Unexpected EOF while splitting {dataset.name}")
                    if line.strip():
                        target.write(line)
                        written += 1
            summary["datasets"][dataset.name] = {"records": written, "path": str(path)}
    return summary


def run_task(task: TaskSpec, cli: Path, run_root: Path, args: argparse.Namespace) -> None:
    for dataset in task.datasets:
        count = manifest_rows(dataset, task.media_dir, args.check_all_images)
        print(f"[preflight] {task.name}/{dataset.name}: {count} records")

    task_dir = run_root / task.name
    registry_dir = task_dir / "dataset_registry"
    output_dir = task_dir / "llamafactory_output"
    config = build_config(task, registry_dir, output_dir, args.max_samples, args.model_path)
    print(f"\n[{task.name}] resolved config:\n{yaml.safe_dump(config, sort_keys=False, allow_unicode=True)}")
    if args.dry_run:
        return

    task_dir.mkdir(parents=True, exist_ok=False)
    registry_dir.mkdir()
    (registry_dir / "dataset_info.json").write_text(
        json.dumps(dataset_info(task.datasets), ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    resolved_config = task_dir / "resolved_config.yaml"
    resolved_config.write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8")

    env = os.environ.copy()
    # LLaMA-Factory launches distributed inference through the first `torchrun`
    # on PATH. Keep it in the same environment as llamafactory-cli; otherwise a
    # globally activated Conda environment can silently spawn incompatible
    # Python workers.
    env["PATH"] = str(cli.parent) + os.pathsep + env.get("PATH", "")
    source_dir = REPO_DIR / "src"
    env["PYTHONPATH"] = str(source_dir) + os.pathsep + env.get("PYTHONPATH", "")
    if args.gpus:
        env["CUDA_VISIBLE_DEVICES"] = args.gpus
        gpu_count = len([value for value in args.gpus.split(",") if value.strip()])
        if gpu_count > 1:
            env["FORCE_TORCHRUN"] = "1"
            env["NPROC_PER_NODE"] = str(gpu_count)
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    env.setdefault("HF_HOME", "/tmp/veclang_hf_cache")
    env.setdefault("MPLCONFIGDIR", "/tmp/veclang_mpl_cache")
    subprocess.run([str(cli), "train", str(resolved_config)], cwd=REPO_DIR, env=env, check=True)

    summary: dict[str, Any] = {
        "task": task.name,
        "config": str(resolved_config),
        "datasets": [dataset.name for dataset in task.datasets],
        "max_samples_per_dataset": int(config["max_samples"]),
    }
    raw_predictions = output_dir / "generated_predictions.jsonl"
    if not args.no_split and raw_predictions.is_file():
        summary["split"] = split_predictions(
            raw_predictions,
            task_dir / "predictions_by_dataset",
            task.datasets,
            int(config["max_samples"]),
        )
    (task_dir / "run_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def main() -> None:
    args = parse_args()
    tasks = load_catalog(args.catalog.resolve())
    if args.list_datasets:
        for task in tasks.values():
            print(f"{task.name}:")
            for dataset in task.datasets:
                print(f"  {dataset.name:<55} {dataset.records:>8} records")
        return

    selected = select_tasks(tasks, args.task, requested_names(args.datasets))
    cli = find_cli(args.llamafactory_cli)
    if not args.dry_run:
        gpu_preflight(args.skip_gpu_check)
    run_name = args.run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
    run_root = args.output_root.expanduser().resolve() / run_name
    if not args.dry_run and run_root.exists():
        raise FileExistsError(f"Run directory already exists: {run_root}")
    for task in selected:
        run_task(task, cli, run_root, args)
    if args.dry_run:
        print("Dry run completed; no inference files were written.")
    else:
        print(f"Inference completed: {run_root}")


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, FileExistsError, RuntimeError, ValueError, subprocess.CalledProcessError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(2) from error
