"""
Convert road extraction predictions in JSONL format to COCO-style outputs.

The script supports GeoJSON-like Feature objects whose geometry is usually
MultiLineString, and keeps compatibility with Polygon outputs as fallback.
"""

import argparse
import json
import math
import os
import time

from tqdm import tqdm


def strip_code_fence(text):
    """Remove optional markdown code fences from model outputs."""
    text = text.strip()
    if not text.startswith("```"):
        return text

    lines = text.splitlines()
    if not lines:
        return text

    if lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip().startswith("```"):
        lines = lines[:-1]
    return "\n".join(lines).strip()


def extract_json_array_from_string(text):
    """Extract a JSON array substring from a larger string."""
    text = text.strip()

    if text.startswith("["):
        return text

    start_idx = text.find("[")
    if start_idx == -1:
        raise ValueError("Could not find JSON array start '['")

    end_idx = max(text.rfind("}"), text.rfind("]"))
    if end_idx == -1:
        raise ValueError("Could not find JSON array end")

    return text[start_idx:end_idx + 1]


def parse_prediction_objects(predict_value):
    """
    Normalize the predict field to a list of objects.

    Supported inputs:
    - JSON array string: "[{...}]"
    - JSON object string: "{...}"
    - list
    - dict
    - markdown fenced JSON
    """
    if predict_value is None:
        return []

    if isinstance(predict_value, list):
        return predict_value

    if isinstance(predict_value, dict):
        return [predict_value]

    if not isinstance(predict_value, str):
        return []

    text = strip_code_fence(predict_value)
    if not text:
        return []

    try:
        data = json.loads(text)
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            return [data]
    except Exception:
        pass

    try:
        extracted = extract_json_array_from_string(text)
        data = json.loads(extracted)
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            return [data]
    except Exception:
        pass

    start_idx = text.find("{")
    end_idx = text.rfind("}")
    if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
        try:
            data = json.loads(text[start_idx:end_idx + 1])
            if isinstance(data, list):
                return data
            if isinstance(data, dict):
                return [data]
        except Exception:
            pass

    return []


def is_explicit_empty_prediction(predict_value):
    """Return True for valid empty model outputs such as [] or fenced []."""
    if isinstance(predict_value, list) and not predict_value:
        return True
    if not isinstance(predict_value, str):
        return False

    text = strip_code_fence(predict_value).strip()
    if text == "[]":
        return True

    try:
        return json.loads(text) == []
    except Exception:
        return False


def safe_pair_coords(coord_list):
    """Convert flattened coordinates [x1, y1, ...] to [(x, y), ...]."""
    if not isinstance(coord_list, list):
        return []

    coord_count = len(coord_list) // 2
    pairs = []
    for i in range(0, coord_count * 2, 2):
        try:
            pairs.append((float(coord_list[i]), float(coord_list[i + 1])))
        except Exception:
            continue
    return pairs


def scale_coord_pairs(coord_pairs, scale):
    return [(x * scale, y * scale) for x, y in coord_pairs]


def parse_junction_coords(junction_data):
    """Parse junctions from either [[x, y], ...] or [x1, y1, ...]."""
    if not isinstance(junction_data, list) or not junction_data:
        return []

    if all(isinstance(item, (list, tuple)) for item in junction_data):
        pairs = []
        for item in junction_data:
            if len(item) < 2:
                continue
            try:
                pairs.append((float(item[0]), float(item[1])))
            except Exception:
                continue
        return pairs

    return safe_pair_coords(junction_data)


def build_scaled_polyline(flat_coord_list, scale):
    """
    Parse, scale, flatten, compute bbox and total length in one pass.
    """
    if not isinstance(flat_coord_list, list):
        return None

    coord_count = len(flat_coord_list) // 2
    if coord_count < 2:
        return None

    flat_coords = []
    min_x = float("inf")
    min_y = float("inf")
    max_x = float("-inf")
    max_y = float("-inf")
    length = 0.0
    prev_x = None
    prev_y = None
    valid_count = 0

    for i in range(0, coord_count * 2, 2):
        try:
            x = float(flat_coord_list[i]) * scale
            y = float(flat_coord_list[i + 1]) * scale
        except Exception:
            continue

        valid_count += 1
        flat_coords.append(round(x, 2))
        flat_coords.append(round(y, 2))
        min_x = min(min_x, x)
        min_y = min(min_y, y)
        max_x = max(max_x, x)
        max_y = max(max_y, y)

        if prev_x is not None:
            length += math.hypot(x - prev_x, y - prev_y)
        prev_x = x
        prev_y = y

    if valid_count < 2:
        return None

    bbox = [
        round(min_x, 2),
        round(min_y, 2),
        round(max_x - min_x, 2),
        round(max_y - min_y, 2),
    ]
    return flat_coords, bbox, round(length, 4)


def build_scaled_polygon(flat_coord_list, scale):
    """
    Parse, scale, flatten, compute bbox and polygon area in one pass.
    """
    coord_pairs = safe_pair_coords(flat_coord_list)
    if len(coord_pairs) < 3:
        return None

    scaled_pairs = []
    flat_coords = []
    min_x = float("inf")
    min_y = float("inf")
    max_x = float("-inf")
    max_y = float("-inf")

    for x, y in coord_pairs:
        sx = x * scale
        sy = y * scale
        scaled_pairs.append((sx, sy))
        flat_coords.append(round(sx, 2))
        flat_coords.append(round(sy, 2))
        min_x = min(min_x, sx)
        min_y = min(min_y, sy)
        max_x = max(max_x, sx)
        max_y = max(max_y, sy)

    if len(scaled_pairs) < 3:
        return None

    area = 0.0
    point_count = len(scaled_pairs)
    for i in range(point_count):
        x1, y1 = scaled_pairs[i]
        x2, y2 = scaled_pairs[(i + 1) % point_count]
        area += x1 * y2 - x2 * y1

    bbox = [
        round(min_x, 2),
        round(min_y, 2),
        round(max_x - min_x, 2),
        round(max_y - min_y, 2),
    ]
    return flat_coords, bbox, round(abs(area) / 2.0, 4)


def iter_prediction_records(inference_file):
    """
    Yield parsed prediction objects line by line to avoid storing all
    intermediate predictions in memory.
    """
    with open(inference_file, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                yield line_no, [], None
                continue

            try:
                data = json.loads(line)
                predict_value = data.get("predict", "")
                if not predict_value:
                    yield line_no, [], None
                    continue

                objects = parse_prediction_objects(predict_value)
                if objects:
                    yield line_no, objects, None
                elif is_explicit_empty_prediction(predict_value):
                    yield line_no, [], None
                else:
                    preview = str(predict_value)[:200].replace("\n", " ")
                    yield line_no, [], f"Unable to parse predict field: {preview}"
            except Exception as exc:
                yield line_no, [], str(exc)


def get_json_dump_kwargs(pretty=False):
    if pretty:
        return {"indent": 2, "ensure_ascii": False}
    return {"ensure_ascii": False, "separators": (",", ":")}


def convert_road_predictions_to_coco(
    inference_file,
    output_file,
    gt_file,
    model_max=1000,
    default_score=1.0,
    pretty_json=False,
    save_full_output=True,
    save_junction_output=True,
    manifest=None,
):
    """
    Convert road extraction predictions to COCO-style outputs.

    Performance notes:
    - Stream predictions instead of storing all intermediate records first.
    - Use compact JSON by default because pretty printing is much slower.
    - Reduce repeated coordinate passes when building annotations.
    """
    total_start = time.perf_counter()

    print(f"\n1. Loading GT from: {gt_file}")
    stage_start = time.perf_counter()
    with open(gt_file, "r", encoding="utf-8") as f:
        gt_data = json.load(f)
    gt_images = gt_data.get("images", [])
    if model_max <= 0:
        raise ValueError("model_max must be positive")
    if manifest:
        with open(manifest, encoding="utf-8") as f:
            records = json.load(f)
        def key(name):
            return os.path.basename(name).removeprefix("region_")
        by_name = {key(im["file_name"]): im for im in gt_images}
        names = [key(record["images"][0]) for record in records]
        if len(names) != len(set(names)) or len(by_name) != len(gt_images):
            raise ValueError("Duplicate patch names")
        gt_images = [by_name[name] for name in names]
    if not gt_images:
        raise ValueError("No road patch images")
    if any(im["width"] != im["height"] for im in gt_images):
        raise ValueError("Road extraction expects square model patches")
    print(f"   GT images: {len(gt_images)}")
    print(f"   GT load time: {time.perf_counter() - stage_start:.2f}s")

    print(f"\n2. Reading and converting predictions from: {inference_file}")
    stage_start = time.perf_counter()
    error_lines = []
    record_iter = iter_prediction_records(inference_file)

    coco_results = []
    coco_annotations = []
    patch_junctions = []
    annotation_id = 1
    processed_prediction_count = 0

    stats = {
        "total_images": len(gt_images),
        "images_with_pred": 0,
        "empty_predictions": 0,
        "total_lines": 0,
        "total_junctions": 0,
        "skipped_short_lines": 0,
    }

    for gt_img in tqdm(gt_images, total=len(gt_images), desc="Processing"):
        try:
            line_no, pred_list, parse_error = next(record_iter)
            processed_prediction_count += 1
            if parse_error:
                error_lines.append((line_no, parse_error))
        except StopIteration:
            print(
                f"ERROR: prediction count ({processed_prediction_count}) "
                f"!= GT image count ({len(gt_images)})"
            )
            print("       Please ensure JSONL line order matches GT images.")
            raise ValueError("Prediction count does not match GT image count")

        image_id = gt_img["id"]
        image_name = gt_img.get("file_name", str(image_id))
        img_w = gt_img.get("width", 256)
        img_h = gt_img.get("height", 256)
        scale = img_w / model_max
        image_junctions = []
        has_valid = False

        if not pred_list:
            stats["empty_predictions"] += 1
            patch_junctions.append(
                {
                    "image_id": image_id,
                    "file_name": image_name,
                    "width": img_w,
                    "height": img_h,
                    "junction_count": 0,
                    "junction": [],
                }
            )
            continue

        for obj in pred_list:
            if not isinstance(obj, dict) or obj.get("type") != "Feature":
                continue

            geometry = obj.get("geometry", {})
            properties = obj.get("properties", {})
            geom_type = geometry.get("type", "")
            score = float(properties.get("score", default_score))

            if geom_type in ("MultiLineString", "LineString"):
                lines = geometry.get("coordinates", [])
                if geom_type == "LineString":
                    lines = [lines]
                for line_coords in lines:
                    line_data = build_scaled_polyline(line_coords, scale)
                    if line_data is None:
                        stats["skipped_short_lines"] += 1
                        continue

                    flat_coords, bbox, area = line_data
                    pred_item = {
                        "image_id": image_id,
                        "category_id": 1,
                        "segmentation": [flat_coords],
                        "bbox": bbox,
                        "area": area,
                        "score": score,
                    }
                    coco_results.append(pred_item)
                    coco_annotations.append(
                        {
                            **pred_item,
                            "id": annotation_id,
                            "iscrowd": 0,
                        }
                    )
                    annotation_id += 1
                    stats["total_lines"] += 1
                    has_valid = True

                junction_pairs = parse_junction_coords(properties.get("junction", []))
                if junction_pairs:
                    stats["total_junctions"] += len(junction_pairs)
                    image_junctions.extend(scale_coord_pairs(junction_pairs, scale))

            elif geom_type == "Polygon":
                polygon_data = build_scaled_polygon(geometry.get("coordinates", []), scale)
                if polygon_data is None:
                    continue

                flat_coords, bbox, area = polygon_data
                pred_item = {
                    "image_id": image_id,
                    "category_id": 1,
                    "segmentation": [flat_coords],
                    "bbox": bbox,
                    "area": area,
                    "score": score,
                }
                coco_results.append(pred_item)
                coco_annotations.append(
                    {
                        **pred_item,
                        "id": annotation_id,
                        "iscrowd": 0,
                    }
                )
                annotation_id += 1
                stats["total_lines"] += 1
                has_valid = True

        if has_valid:
            stats["images_with_pred"] += 1

        unique_junctions = sorted({(round(x, 2), round(y, 2)) for x, y in image_junctions})
        patch_junctions.append(
            {
                "image_id": image_id,
                "file_name": image_name,
                "width": img_w,
                "height": img_h,
                "junction_count": len(unique_junctions),
                "junction": [[x, y] for x, y in unique_junctions],
            }
        )

    extra_prediction_count = sum(1 for _ in record_iter)
    if extra_prediction_count > 0:
        print(
            f"ERROR: prediction count ({processed_prediction_count + extra_prediction_count}) "
            f"!= GT image count ({len(gt_images)})"
        )
        print("       Please ensure JSONL line order matches GT images.")
        raise ValueError("Prediction count does not match GT image count")

    print(f"   Parsed predictions: {processed_prediction_count}")
    if error_lines:
        print(f"   Parse errors: {len(error_lines)} (first 5: {error_lines[:5]})")
    print(f"   Process time: {time.perf_counter() - stage_start:.2f}s")

    print("\n3. Saving results ...")
    stage_start = time.perf_counter()
    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
    dump_kwargs = get_json_dump_kwargs(pretty=pretty_json)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(coco_results, f, **dump_kwargs)
    print(f"   Eval predictions saved to: {output_file}")

    if save_full_output:
        full_output_path = output_file.replace(".json", "_full.json")
        full_coco = {
            "images": gt_images,
            "annotations": coco_annotations,
            "categories": [{"id": 1, "name": "road", "supercategory": "road"}],
        }
        with open(full_output_path, "w", encoding="utf-8") as f:
            json.dump(full_coco, f, **dump_kwargs)
        print(f"   Full COCO format saved to: {full_output_path}")

    if save_junction_output:
        junction_output_path = output_file.replace(".json", "_junctions.json")
        with open(junction_output_path, "w", encoding="utf-8") as f:
            json.dump(patch_junctions, f, **dump_kwargs)
        print(f"   Patch junctions saved to: {junction_output_path}")
    print(f"   Save time: {time.perf_counter() - stage_start:.2f}s")

    print(f"\n{'=' * 60}")
    print("=== Conversion Summary ===")
    print(f"{'=' * 60}")
    print(f"  Total images:             {stats['total_images']}")
    print(f"  Images with predictions:  {stats['images_with_pred']}")
    print(f"  Empty predictions:        {stats['empty_predictions']}")
    print(f"  Total line annotations:   {stats['total_lines']}")
    print(f"  Total junctions:          {stats['total_junctions']}")
    print(f"  Skipped short lines (<2): {stats['skipped_short_lines']}")
    pct = stats["images_with_pred"] / max(stats["total_images"], 1) * 100
    avg = stats["total_lines"] / max(stats["images_with_pred"], 1)
    print(f"  Prediction coverage:      {pct:.1f}%")
    print(f"  Avg lines per image:      {avg:.2f}")
    print(f"  Total runtime:            {time.perf_counter() - total_start:.2f}s")
    print(f"{'=' * 60}")

    return coco_results


def main():
    parser = argparse.ArgumentParser(
        description="Convert road polyline predictions (JSONL) to patch-level COCO results."
    )
    parser.add_argument("--inference-file", required=True, help="Input model predictions in JSONL format")
    parser.add_argument("--gt-file", required=True, help="Patch-level COCO ground-truth JSON")
    parser.add_argument("--output-file", required=True, help="Output COCO detection JSON")
    parser.add_argument("--model-max", type=float, default=1000, help="Maximum model coordinate value")
    parser.add_argument("--default-score", type=float, default=1.0)
    parser.add_argument("--pretty-json", action="store_true", help="Indent output JSON (larger and slower)")
    parser.add_argument("--no-full-output", action="store_true", help="Do not write the *_full.json file")
    parser.add_argument("--no-junction-output", action="store_true", help="Do not write the *_junctions.json file")
    parser.add_argument("--manifest", help="Ordered SFT JSON manifest, matching prediction order")
    args = parser.parse_args()

    output_parent = os.path.dirname(os.path.abspath(args.output_file))
    os.makedirs(output_parent, exist_ok=True)
    convert_road_predictions_to_coco(
        inference_file=args.inference_file,
        output_file=args.output_file,
        gt_file=args.gt_file,
        model_max=args.model_max,
        manifest=args.manifest,
        default_score=args.default_score,
        pretty_json=args.pretty_json,
        save_full_output=not args.no_full_output,
        save_junction_output=not args.no_junction_output,
    )
    print("\nConversion completed!")


if __name__ == "__main__":
    main()
