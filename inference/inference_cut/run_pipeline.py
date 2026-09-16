#!/usr/bin/env python3
"""Continue from detection JSONL through cropping and attribute inference."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from build_attribute_manifest import build_manifest
from crop_instances import crop_instances
from detection_to_coco import convert_predictions


SCRIPT_DIR = Path(__file__).resolve().parent
INFERENCE_DIR = SCRIPT_DIR.parent
VECLANG_DIR = INFERENCE_DIR.parent
DEFAULT_CONFIG = SCRIPT_DIR / "configs" / "profiles.yaml"
DEFAULT_OUTPUT_ROOT = Path(
    os.getenv("VECLANG_OUTPUT_ROOT", str(VECLANG_DIR.parent / "VecLang_outputs"))
)


def _absolute(base: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def _load_config(path: Path, dataset_root_override: Path | None) -> tuple[dict[str, Any], Path, Path]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or not isinstance(raw.get("profiles"), dict):
        raise ValueError(f"Invalid inference-cut config: {path}")
    dataset_root = (
        dataset_root_override.expanduser().resolve()
        if dataset_root_override else _absolute(path.parent, raw["dataset_root"])
    )
    attribute_config = _absolute(path.parent, raw["attribute_config"])
    return raw, dataset_root, attribute_config


def _find_cli(explicit: Path | None) -> Path:
    if explicit:
        candidate = explicit.expanduser().resolve()
    elif os.getenv("LLAMAFACTORY_CLI"):
        candidate = Path(os.environ["LLAMAFACTORY_CLI"]).expanduser().resolve()
    elif shutil.which("llamafactory-cli"):
        candidate = Path(shutil.which("llamafactory-cli") or "").resolve()
    else:
        raise FileNotFoundError("llamafactory-cli not found; activate the environment or set LLAMAFACTORY_CLI.")
    if not candidate.is_file() or not os.access(candidate, os.X_OK):
        raise FileNotFoundError(f"llamafactory-cli is not executable: {candidate}")
    return candidate


def _dataset_registry(name: str, manifest: Path) -> dict[str, Any]:
    return {
        name: {
            "file_name": str(manifest),
            "formatting": "sharegpt",
            "columns": {"messages": "messages", "images": "images"},
            "tags": {
                "role_tag": "role",
                "content_tag": "content",
                "user_tag": "user",
                "assistant_tag": "assistant",
            },
        }
    }


def _run_attributes(
    profile: dict[str, Any],
    attribute_config: Path,
    manifest: Path,
    crop_dir: Path,
    output_dir: Path,
    model_path: str | None,
    cli: Path,
    gpus: str | None,
) -> None:
    registry_dir = output_dir / "dataset_registry"
    inference_output = output_dir / "attribute_inference"
    registry_dir.mkdir(parents=True)
    (registry_dir / "dataset_info.json").write_text(
        json.dumps(_dataset_registry(profile["output_dataset"], manifest), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    config = yaml.safe_load(attribute_config.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError(f"Invalid attribute config: {attribute_config}")
    config.update(
        {
            "eval_dataset": profile["output_dataset"],
            "dataset_dir": str(registry_dir),
            "media_dir": str(crop_dir),
            "output_dir": str(inference_output),
            "overwrite_output_dir": False,
        }
    )
    selected_model = model_path or os.getenv("VECLANG_MODEL_PATH")
    if selected_model:
        candidate = Path(selected_model).expanduser()
        config["model_name_or_path"] = str(candidate.resolve()) if candidate.exists() else selected_model
    resolved_config = output_dir / "attribute_config.yaml"
    resolved_config.write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8")

    env = os.environ.copy()
    env["PATH"] = str(cli.parent) + os.pathsep + env.get("PATH", "")
    source_dir = VECLANG_DIR.parent / "src"
    env["PYTHONPATH"] = str(source_dir) + os.pathsep + env.get("PYTHONPATH", "")
    if gpus:
        env["CUDA_VISIBLE_DEVICES"] = gpus
        gpu_count = len([value for value in gpus.split(",") if value.strip()])
        if gpu_count > 1:
            env["FORCE_TORCHRUN"] = "1"
            env["NPROC_PER_NODE"] = str(gpu_count)
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    subprocess.run([str(cli), "train", str(resolved_config)], cwd=VECLANG_DIR.parent, env=env, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", help="Profile name from configs/profiles.yaml.")
    parser.add_argument("--detection-predictions", type=Path)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--dataset-root", type=Path, help="Override dataset/test from the profile config.")
    parser.add_argument("--output-dir", type=Path, help="Defaults to ../VecLang_outputs/<timestamp>_<profile>.")
    parser.add_argument("--model-path", help="Local checkpoint or Hugging Face model ID for attributes.")
    parser.add_argument("--llamafactory-cli", type=Path)
    parser.add_argument("--gpus", help="CUDA device list, for example 0 or 0,1,2,3.")
    parser.add_argument("--max-instances", type=int, default=0, help="0 processes every detected instance.")
    parser.add_argument("--prepare-only", action="store_true", help="Stop after creating crops and manifest.")
    parser.add_argument("--list-profiles", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = args.config.expanduser().resolve()
    raw, dataset_root, attribute_config = _load_config(config_path, args.dataset_root)
    if args.list_profiles:
        for name, profile in raw["profiles"].items():
            print(f"{name:<22} {profile['output_dataset']}")
        return
    if not args.profile or not args.detection_predictions:
        raise ValueError("--profile and --detection-predictions are required unless --list-profiles is used.")
    if args.profile not in raw["profiles"]:
        raise ValueError(f"Unknown profile: {args.profile}")
    profile = raw["profiles"][args.profile]
    crop_config = raw.get("crop", {})
    prediction_file = args.detection_predictions.expanduser().resolve()
    if not prediction_file.is_file():
        raise FileNotFoundError(f"Detection predictions not found: {prediction_file}")
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir else DEFAULT_OUTPUT_ROOT / f"{datetime.now():%Y%m%d_%H%M%S}_{args.profile}"
    )
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_file = _absolute(dataset_root, profile["detection_manifest"])
    image_dir = _absolute(dataset_root, profile["image_dir"])
    prompt_manifest = _absolute(dataset_root, profile["prompt_manifest"])
    full_coco = convert_predictions(
        prediction_file=prediction_file,
        manifest_file=manifest_file,
        media_dir=manifest_file.parent,
        output_file=output_dir / "detections_coco.json",
        category_id=int(profile["category_id"]),
        category_name=str(profile["category_name"]),
        category_aliases=[str(value) for value in profile.get("category_aliases", [])],
        model_max=float(crop_config.get("model_coordinate_max", 1000)),
    )
    crop_dir = output_dir / "instance_crops"
    count = crop_instances(
        coco_file=full_coco,
        image_dir=image_dir,
        output_dir=crop_dir,
        category_id=int(profile["category_id"]),
        category_name=str(profile["category_name"]),
        class_code=int(profile["class_code"]),
        score_threshold=float(crop_config.get("score_threshold", 0.0)),
        min_area=float(crop_config.get("min_area", 0.0)),
        scale_factor=float(crop_config.get("scale_factor", 1.3)),
        output_size=int(crop_config.get("output_size", 256)),
        max_instances=args.max_instances,
    )
    if count == 0:
        raise ValueError("No valid instances were produced for this profile.")
    attribute_manifest = crop_dir / "attribute_manifest.json"
    build_manifest(
        annotation_dir=crop_dir / "annotations",
        image_dir=crop_dir / "images",
        output_path=attribute_manifest,
        prompt_manifest=prompt_manifest,
        image_prefix="images",
        property_class=str(profile["property_class"]),
    )
    if args.prepare_only:
        print(f"Prepared crops and manifest: {crop_dir}")
        return
    cli = _find_cli(args.llamafactory_cli)
    _run_attributes(
        profile, attribute_config, attribute_manifest, crop_dir, output_dir,
        args.model_path, cli, args.gpus,
    )
    print(f"Detection-to-attribute inference completed: {output_dir}")


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, FileExistsError, RuntimeError, ValueError, subprocess.CalledProcessError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(2) from error
