#!/usr/bin/env python3
"""Crop one fixed-size image for each detected COCO instance."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from PIL import Image
from tqdm import tqdm


IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")


def _find_image(image_dir: Path, file_name: str) -> Path | None:
    direct = image_dir / Path(file_name).name
    if direct.is_file():
        return direct
    for extension in IMAGE_EXTENSIONS:
        candidate = image_dir / f"{Path(file_name).stem}{extension}"
        if candidate.is_file():
            return candidate
    return None


def _clip_bbox(raw: Any, width: int, height: int) -> list[float] | None:
    if not isinstance(raw, list) or len(raw) != 4:
        return None
    try:
        x, y, box_width, box_height = (float(value) for value in raw)
    except (TypeError, ValueError):
        return None
    x1, y1 = max(0.0, min(width, x)), max(0.0, min(height, y))
    x2, y2 = max(0.0, min(width, x + box_width)), max(0.0, min(height, y + box_height))
    return [x1, y1, x2 - x1, y2 - y1] if x2 > x1 and y2 > y1 else None


def _segmentation(raw: Any, bbox: list[float]) -> list[list[float]]:
    if isinstance(raw, list) and raw:
        values = [raw] if all(isinstance(value, (int, float)) for value in raw) else raw
        output: list[list[float]] = []
        for polygon in values:
            if isinstance(polygon, list) and len(polygon) >= 6:
                try:
                    output.append([float(value) for value in polygon])
                except (TypeError, ValueError):
                    pass
        if output:
            return output
    x, y, width, height = bbox
    return [[x, y, x + width, y, x + width, y + height, x, y + height]]


def _crop(
    image: Image.Image, bbox: list[float], scale_factor: float
) -> tuple[Image.Image, float, float, float, float]:
    x, y, width, height = bbox
    crop_width, crop_height = max(1.0, width * scale_factor), max(1.0, height * scale_factor)
    crop_x = x + width / 2.0 - crop_width / 2.0
    crop_y = y + height / 2.0 - crop_height / 2.0
    source_x1, source_y1 = max(0, int(crop_x)), max(0, int(crop_y))
    source_x2 = min(image.width, int(crop_x + crop_width))
    source_y2 = min(image.height, int(crop_y + crop_height))
    canvas = Image.new("RGB", (int(round(crop_width)), int(round(crop_height))))
    region = image.crop((source_x1, source_y1, source_x2, source_y2))
    target_x, target_y = max(0, source_x1 - int(crop_x)), max(0, source_y1 - int(crop_y))
    copy_width = min(region.width, canvas.width - target_x)
    copy_height = min(region.height, canvas.height - target_y)
    if copy_width > 0 and copy_height > 0:
        canvas.paste(region.crop((0, 0, copy_width, copy_height)), (target_x, target_y))
    return canvas, crop_x, crop_y, crop_width, crop_height


def _transform(
    polygons: list[list[float]], crop_x: float, crop_y: float, scale_x: float, scale_y: float
) -> list[list[float]]:
    output: list[list[float]] = []
    for polygon in polygons:
        transformed: list[float] = []
        for index in range(0, len(polygon) - 1, 2):
            transformed.extend(
                ((polygon[index] - crop_x) * scale_x, (polygon[index + 1] - crop_y) * scale_y)
            )
        if len(transformed) >= 6:
            output.append(transformed)
    return output


def crop_instances(
    coco_file: Path,
    image_dir: Path,
    output_dir: Path,
    category_id: int,
    category_name: str,
    class_code: int,
    score_threshold: float = 0.0,
    min_area: float = 0.0,
    scale_factor: float = 1.3,
    output_size: int = 256,
    max_instances: int = 0,
) -> int:
    coco = json.loads(coco_file.read_text(encoding="utf-8"))
    if not isinstance(coco, dict):
        raise ValueError(f"Expected full COCO data: {coco_file}")
    annotations_by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for annotation in coco.get("annotations", []):
        if isinstance(annotation, dict) and "image_id" in annotation:
            annotations_by_image[int(annotation["image_id"])].append(annotation)
    image_output = output_dir / "images"
    annotation_output = output_dir / "annotations"
    filtered_output = output_dir / "filtered_small"
    for directory in (image_output, annotation_output, filtered_output):
        directory.mkdir(parents=True, exist_ok=True)

    counts: defaultdict[str, int] = defaultdict(int)
    for image_info in tqdm(coco.get("images", []), desc=f"Crop {category_name} instances"):
        image_id = int(image_info["id"])
        image_path = _find_image(image_dir, str(image_info.get("file_name", "")))
        if image_path is None:
            counts["missing_images"] += 1
            continue
        try:
            with Image.open(image_path) as source:
                image = source.convert("RGB")
        except Exception:
            counts["image_errors"] += 1
            continue
        for annotation in annotations_by_image.get(image_id, []):
            if int(annotation.get("category_id", -1)) != category_id:
                continue
            if float(annotation.get("score", 1.0)) < score_threshold:
                counts["low_score"] += 1
                continue
            bbox = _clip_bbox(annotation.get("bbox"), image.width, image.height)
            if bbox is None:
                counts["invalid_bbox"] += 1
                continue
            area = float(annotation.get("area") or bbox[2] * bbox[3])
            polygons = _segmentation(annotation.get("segmentation"), bbox)
            cropped, crop_x, crop_y, crop_width, crop_height = _crop(image, bbox, scale_factor)
            resized = cropped.resize((output_size, output_size), Image.Resampling.BILINEAR)
            annotation_id = int(annotation.get("id", counts["success"] + 1))
            output_name = f"{image_path.stem}_inst_{annotation_id}"
            if area < min_area:
                resized.save(filtered_output / f"{output_name}_area_{int(area)}.png")
                counts["filtered_small"] += 1
                continue
            scale_x, scale_y = output_size / crop_width, output_size / crop_height
            transformed = _transform(polygons, crop_x, crop_y, scale_x, scale_y)
            x, y, box_width, box_height = bbox
            output_annotation = {
                "id": annotation_id,
                "image_id": image_id,
                "instance_index": annotation_id,
                "category_id": category_id,
                "category_name": category_name,
                "category_code": class_code,
                "score": float(annotation.get("score", 1.0)),
                "segmentation": transformed,
                "bbox": [(x - crop_x) * scale_x, (y - crop_y) * scale_y, box_width * scale_x, box_height * scale_y],
                "area": area,
                "iscrowd": int(annotation.get("iscrowd", 0)),
                "original_bbox": bbox,
                "original_segmentation": polygons,
                "original_image": image_path.name,
                "original_coco": coco_file.name,
                "crop_info": {
                    "crop_x": crop_x, "crop_y": crop_y, "crop_w": crop_width, "crop_h": crop_height,
                    "scale_factor": scale_factor, "output_size": output_size,
                },
            }
            resized.save(image_output / f"{output_name}.png")
            (annotation_output / f"{output_name}.json").write_text(
                json.dumps(output_annotation, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )
            counts["success"] += 1
            if max_instances and counts["success"] >= max_instances:
                break
        if max_instances and counts["success"] >= max_instances:
            break
    (output_dir / "crop_summary.json").write_text(
        json.dumps({"category": category_name, **dict(counts)}, indent=2) + "\n", encoding="utf-8"
    )
    print(f"[crop-instances] {dict(counts)}")
    return counts["success"]
