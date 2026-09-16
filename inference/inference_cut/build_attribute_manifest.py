#!/usr/bin/env python3
"""Build a ShareGPT attribute manifest from detected instance crops."""

from __future__ import annotations

import json
from pathlib import Path, PurePosixPath
from typing import Any


def _user_prompt(path: Path) -> str:
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"Prompt manifest is empty or invalid: {path}")
    for message in rows[0].get("messages", []):
        if isinstance(message, dict) and message.get("role") == "user":
            content = message.get("content")
            if isinstance(content, str) and content.strip():
                return content
    raise ValueError(f"No user prompt found in: {path}")


def _feature(annotation: dict[str, Any], property_class: str, canvas_size: int) -> list[dict[str, Any]]:
    output_size = float(annotation.get("crop_info", {}).get("output_size", 256))
    polygons = annotation.get("segmentation", [])
    if polygons and isinstance(polygons[0], (int, float)):
        polygons = [polygons]
    converted: list[list[int]] = []
    for polygon in polygons if isinstance(polygons, list) else []:
        if not isinstance(polygon, list) or len(polygon) < 6:
            continue
        points: list[tuple[int, int]] = []
        for index in range(0, len(polygon) - 1, 2):
            point = (
                max(0, min(canvas_size, round(float(polygon[index]) * canvas_size / output_size))),
                max(0, min(canvas_size, round(float(polygon[index + 1]) * canvas_size / output_size))),
            )
            if not points or point != points[-1]:
                points.append(point)
        if len(points) > 1 and points[0] == points[-1]:
            points.pop()
        if len(points) >= 3:
            converted.append([coordinate for point in points for coordinate in point])
    if not converted:
        return []
    return [
        {
            "type": "Feature",
            "geometry": {"type": "Polygon", "coordinates": converted[0]},
            "properties": {"class": property_class, "holes": converted[1:]},
        }
    ]


def build_manifest(
    annotation_dir: Path,
    image_dir: Path,
    output_path: Path,
    prompt_manifest: Path,
    image_prefix: str,
    property_class: str,
    canvas_size: int = 1000,
) -> int:
    prompt = _user_prompt(prompt_manifest)
    rows: list[dict[str, Any]] = []
    skipped = 0
    for annotation_path in sorted(annotation_dir.glob("*.json")):
        image_name = f"{annotation_path.stem}.png"
        if not (image_dir / image_name).is_file():
            skipped += 1
            continue
        annotation = json.loads(annotation_path.read_text(encoding="utf-8"))
        target = _feature(annotation, property_class, canvas_size)
        rows.append(
            {
                "messages": [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": json.dumps(target, ensure_ascii=False, separators=(",", ": "))},
                ],
                "images": [str(PurePosixPath(image_prefix) / image_name)],
            }
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(rows, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"[build-attribute-manifest] records={len(rows)}, skipped={skipped}")
    return len(rows)
