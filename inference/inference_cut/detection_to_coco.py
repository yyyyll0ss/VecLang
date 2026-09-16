#!/usr/bin/env python3
"""Convert VecLang detection JSONL to ordered COCO and full-COCO files."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from PIL import Image


def _strip_fence(text: str) -> str:
    text = text.strip()
    if not text.startswith("```"):
        return text
    lines = text.splitlines()[1:]
    if lines and lines[-1].strip().startswith("```"):
        lines.pop()
    return "\n".join(lines).strip()


def _complete_objects(text: str) -> list[dict[str, Any]]:
    objects: list[dict[str, Any]] = []
    buffer: list[str] = []
    depth = 0
    quoted = escaped = False
    for character in text:
        if escaped:
            escaped = False
        elif character == "\\" and quoted:
            escaped = True
        elif character == '"':
            quoted = not quoted
        elif not quoted and character == "{":
            if depth == 0:
                buffer = []
            depth += 1
        if depth:
            buffer.append(character)
        if not quoted and character == "}" and depth:
            depth -= 1
            if depth == 0:
                try:
                    value = json.loads("".join(buffer))
                    if isinstance(value, dict):
                        objects.append(value)
                except json.JSONDecodeError:
                    pass
    return objects


def _prediction_objects(record: dict[str, Any]) -> list[dict[str, Any]]:
    value: Any = record.get("predict")
    if value is None:
        for message in reversed(record.get("messages", [])):
            if isinstance(message, dict) and message.get("role") == "assistant":
                value = message.get("content")
                break
    if isinstance(value, dict):
        return [value]
    if isinstance(value, list):
        return [item for item in value if isinstance(item, dict)]
    if not isinstance(value, str):
        return []
    text = _strip_fence(value)
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return [parsed]
        if isinstance(parsed, list):
            return [item for item in parsed if isinstance(item, dict)]
    except json.JSONDecodeError:
        pass
    return _complete_objects(text)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}: {error}") from error
            if not isinstance(row, dict):
                raise ValueError(f"Expected a JSON object at {path}:{line_number}")
            rows.append(row)
    return rows


def _normalize(value: Any) -> str:
    return str(value or "").strip().lower().replace("-", "_").replace(" ", "_")


def _image_path(row: dict[str, Any], media_dir: Path) -> Path:
    images = row.get("images")
    if not isinstance(images, list) or not images or not isinstance(images[0], str):
        raise ValueError("Each source-manifest row must contain images[0].")
    path = Path(images[0])
    return path if path.is_absolute() else media_dir / path


def _bbox(item: dict[str, Any], width: int, height: int, model_max: float) -> list[float] | None:
    raw = next(
        (item.get(name) for name in ("bbox_2d", "bbox", "box") if isinstance(item.get(name), list)),
        None,
    )
    if not raw or len(raw) != 4:
        return None
    try:
        x1, y1, x2, y2 = (float(value) for value in raw)
    except (TypeError, ValueError):
        return None
    x1, x2 = x1 * width / model_max, x2 * width / model_max
    y1, y2 = y1 * height / model_max, y2 * height / model_max
    x1, x2 = max(0.0, min(width, x1)), max(0.0, min(width, x2))
    y1, y2 = max(0.0, min(height, y1)), max(0.0, min(height, y2))
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2 - x1, y2 - y1]


def convert_predictions(
    prediction_file: Path,
    manifest_file: Path,
    media_dir: Path,
    output_file: Path,
    category_id: int,
    category_name: str,
    category_aliases: list[str],
    model_max: float = 1000.0,
) -> Path:
    predictions = _load_jsonl(prediction_file)
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    if not isinstance(manifest, list) or len(predictions) > len(manifest):
        raise ValueError("Prediction count must not exceed the source-manifest count.")
    manifest = manifest[: len(predictions)]
    aliases = {_normalize(category_name), *(_normalize(value) for value in category_aliases)}
    images: list[dict[str, Any]] = []
    annotations: list[dict[str, Any]] = []
    detections: list[dict[str, Any]] = []
    invalid = unknown = 0
    for image_id, (source_row, prediction_row) in enumerate(zip(manifest, predictions), 1):
        path = _image_path(source_row, media_dir).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Source image not found: {path}")
        with Image.open(path) as image:
            width, height = image.size
        images.append({"id": image_id, "file_name": path.name, "width": width, "height": height})
        for item in _prediction_objects(prediction_row):
            label = next(
                (item.get(name) for name in ("label", "class", "category", "category_name") if item.get(name)),
                None,
            )
            if _normalize(label) not in aliases:
                unknown += 1
                continue
            bbox = _bbox(item, width, height, model_max)
            if bbox is None:
                invalid += 1
                continue
            x, y, box_width, box_height = bbox
            try:
                score = float(item.get("score", item.get("confidence", 1.0)))
            except (TypeError, ValueError):
                score = 1.0
            common = {
                "image_id": image_id, "category_id": category_id, "bbox": bbox,
                "area": box_width * box_height, "score": score,
            }
            detections.append(dict(common))
            annotations.append(
                {
                    "id": len(annotations) + 1, **common,
                    "segmentation": [[x, y, x + box_width, y, x + box_width, y + box_height, x, y + box_height]],
                    "iscrowd": 0,
                }
            )
    categories = [{"id": category_id, "name": category_name, "supercategory": "object"}]
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(json.dumps(detections, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    full_path = output_file.with_name(f"{output_file.stem}_full.json")
    full_path.write_text(
        json.dumps({"images": images, "annotations": annotations, "categories": categories}, ensure_ascii=False, indent=2)
        + "\n",
        encoding="utf-8",
    )
    print(
        f"[detection-to-coco] images={len(images)}, detections={len(detections)}, "
        f"invalid={invalid}, other_categories={unknown}"
    )
    return full_path
