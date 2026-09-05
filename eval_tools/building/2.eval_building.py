import argparse
import copy
import json
import os
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from pycocotools import mask as cocomask
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from tqdm import tqdm

try:
    from shapely import geometry
    from shapely.geometry import Polygon

    SHAPELY_AVAILABLE = True
except ImportError:
    geometry = None
    Polygon = None
    SHAPELY_AVAILABLE = False


EPS = 1e-9


def load_detection_annotations(dt_path: str) -> List[dict]:
    with open(dt_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    raise ValueError(f"Detection file must be a JSON list: {dt_path}")


def detect_iou_type(dt_anns: Sequence[dict]) -> str:
    """Automatically detect whether detections contain segmentation or only bbox."""
    if not dt_anns:
        return "bbox"
    ann = dt_anns[0]
    if has_valid_segmentation(ann):
        return "segm"
    return "bbox"


def has_valid_segmentation(ann: dict) -> bool:
    segmentation = ann.get("segmentation")
    if isinstance(segmentation, dict):
        return "counts" in segmentation and "size" in segmentation
    return len(get_valid_segments(segmentation)) > 0


def get_valid_segments(segmentation) -> List[List[float]]:
    if not isinstance(segmentation, list):
        return []

    valid_segments: List[List[float]] = []
    for seg in segmentation:
        if not isinstance(seg, list):
            continue
        if len(seg) < 6 or len(seg) % 2 != 0:
            continue
        valid_segments.append(seg)
    return valid_segments


def count_vertices(ann: dict) -> int:
    return sum(len(seg) // 2 for seg in get_valid_segments(ann.get("segmentation")))


def calc_mask_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    inter = np.logical_and(mask_a, mask_b).sum()
    union = np.logical_or(mask_a, mask_b).sum()
    if union == 0:
        return 1.0
    return float(inter / (union + EPS))


def poly_to_bbox(poly: np.ndarray) -> List[float]:
    lt_x = float(np.min(poly[:, 0]))
    lt_y = float(np.min(poly[:, 1]))
    w = float(np.max(poly[:, 0]) - lt_x)
    h = float(np.max(poly[:, 1]) - lt_y)
    return [lt_x, lt_y, w, h]


def generate_coco_ann(polys, scores, labels, img_id):
    sample_ann = []
    for i, polygon in enumerate(polys):
        if polygon.shape[0] < 4:
            continue

        vec_poly = polygon.ravel().tolist()
        poly_bbox = poly_to_bbox(polygon)
        ann_per_building = {
            "image_id": img_id,
            "category_id": int(labels[i]),
            "segmentation": [vec_poly],
            "bbox": poly_bbox,
            "score": float(scores[i]),
        }
        sample_ann.append(ann_per_building)

    return sample_ann


def decode_annotation_mask(ann: dict, height: int, width: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=bool)
    segmentation = ann.get("segmentation")

    if isinstance(segmentation, dict):
        decoded = cocomask.decode(segmentation)
    else:
        segments = get_valid_segments(segmentation)
        if not segments:
            return mask
        rle = cocomask.frPyObjects(segments, height, width)
        decoded = cocomask.decode(rle)

    if decoded.ndim == 2:
        return decoded.astype(bool)
    return np.any(decoded, axis=2)


def build_image_mask_and_vertices(coco_api: COCO, img_id: int) -> Tuple[np.ndarray, int]:
    img = coco_api.loadImgs([img_id])[0]
    height, width = img["height"], img["width"]
    mask = np.zeros((height, width), dtype=bool)
    vertex_count = 0

    ann_ids = coco_api.getAnnIds(imgIds=[img_id])
    for ann in coco_api.loadAnns(ann_ids):
        mask = np.logical_or(mask, decode_annotation_mask(ann, height, width))
        vertex_count += count_vertices(ann)

    return mask, vertex_count


def compute_map(
    gt_path: str,
    dt_anns: Sequence[dict],
    iou_type: Optional[str] = None,
    max_dets: Sequence[int] = (1, 10, 100),
) -> Dict[str, dict]:
    """Compute COCO mAP / mAR metrics."""
    if not os.path.exists(gt_path):
        raise FileNotFoundError(f"GT file not found: {gt_path}")

    print(f"Loading ground-truth from: {gt_path}")
    coco_gt = COCO(gt_path)

    gt_anns = coco_gt.loadAnns(coco_gt.getAnnIds())
    gt_has_full_segm = all(has_valid_segmentation(ann) for ann in gt_anns)
    dt_has_segm = any(has_valid_segmentation(ann) for ann in dt_anns)

    if iou_type is None:
        detected_type = detect_iou_type(dt_anns)
        if detected_type == "segm" and gt_has_full_segm:
            iou_types = ["bbox", "segm"]
        else:
            iou_types = ["bbox"]
            if dt_has_segm and not gt_has_full_segm:
                print("Note: GT contains annotations without valid segmentation, skipping segm mAP.")
    else:
        iou_types = [iou_type]

    if not dt_anns:
        print("No detections found, COCO mAP will be skipped.")
        return {
            "available": False,
            "reason": "Detection result list is empty.",
            "iou_types": iou_types,
        }

    print("Loading detections...")
    coco_dt = coco_gt.loadRes(copy.deepcopy(list(dt_anns)))
    all_results: Dict[str, dict] = {}

    for current_iou_type in iou_types:
        print(f"\nEvaluating COCO metrics with IoU type: {current_iou_type}")
        coco_eval = COCOeval(coco_gt, coco_dt, iouType=current_iou_type)
        coco_eval.params.maxDets = list(max_dets)
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        stats = coco_eval.stats.tolist() if hasattr(coco_eval.stats, "tolist") else list(coco_eval.stats)
        all_results[current_iou_type] = {
            "AP_50_95": stats[0],
            "AP_50": stats[1],
            "AP_75": stats[2],
            "AP_small": stats[3],
            "AP_medium": stats[4],
            "AP_large": stats[5],
            "AR_1": stats[6],
            "AR_10": stats[7],
            "AR_100": stats[8],
            "AR_small": stats[9],
            "AR_medium": stats[10],
            "AR_large": stats[11],
        }

    return {
        "available": True,
        "iou_types": iou_types,
        "metrics": all_results,
    }


def compute_iou_and_ciou(coco_gt: COCO, coco_dt: COCO) -> Dict[str, float]:
    img_ids = list(sorted(coco_gt.imgs.keys()))
    list_iou: List[float] = []
    list_ciou: List[float] = []
    list_ps: List[float] = []

    for img_id in tqdm(img_ids, desc="Computing IoU / C-IoU"):
        dt_mask, dt_vertices = build_image_mask_and_vertices(coco_dt, img_id)
        gt_mask, gt_vertices = build_image_mask_and_vertices(coco_gt, img_id)

        ps = 1.0 - abs(dt_vertices - gt_vertices) / (dt_vertices + gt_vertices + EPS)
        iou = calc_mask_iou(dt_mask, gt_mask)

        list_iou.append(iou)
        list_ciou.append(iou * ps)
        list_ps.append(ps)

    return {
        "IoU": float(np.mean(list_iou)) if list_iou else 0.0,
        "C-IoU": float(np.mean(list_ciou)) if list_ciou else 0.0,
        "PS": float(np.mean(list_ps)) if list_ps else 0.0,
        "image_count": len(img_ids),
    }


def bounding_box(points: np.ndarray) -> List[float]:
    bot_left_x, bot_left_y = float("inf"), float("inf")
    top_right_x, top_right_y = float("-inf"), float("-inf")
    for x, y in points:
        bot_left_x = min(bot_left_x, x)
        bot_left_y = min(bot_left_y, y)
        top_right_x = max(top_right_x, x)
        top_right_y = max(top_right_y, y)
    return [bot_left_x, bot_left_y, top_right_x - bot_left_x, top_right_y - bot_left_y]


def make_polygon(points: np.ndarray):
    if not SHAPELY_AVAILABLE or points.shape[0] < 3:
        return None

    poly = Polygon(points)
    if poly.is_empty:
        return None
    if not poly.is_valid:
        poly = poly.buffer(0)
    if poly.is_empty or poly.geom_type != "Polygon":
        return None
    return poly


def compare_polys(poly_a, poly_b) -> float:
    boundary_a, boundary_b = poly_a.exterior, poly_b.exterior
    return polis(boundary_a.coords, boundary_b) + polis(boundary_b.coords, boundary_a)


def polis(coords, boundary) -> float:
    total = 0.0
    for pt in (geometry.Point(coord) for coord in coords[:-1]):
        total += boundary.distance(pt)
    return total / float(2 * len(coords))


def get_instance_polygons(anns: Sequence[dict]) -> Tuple[List[np.ndarray], List[List[float]]]:
    polygons: List[np.ndarray] = []
    bboxes: List[List[float]] = []

    for ann in anns:
        segments = get_valid_segments(ann.get("segmentation"))
        if not segments:
            continue
        points = np.asarray(segments[0], dtype=np.float64).reshape(-1, 2)
        if points.shape[0] < 3:
            continue
        polygons.append(points)
        bboxes.append(bounding_box(points))

    return polygons, bboxes


def compute_polis(coco_gt: COCO, coco_dt: COCO, match_iou: float = 0.5) -> Dict[str, Optional[float]]:
    if not SHAPELY_AVAILABLE:
        return {
            "PoLiS": None,
            "matched_gt_count": 0,
            "valid_image_count": 0,
            "reason": "Shapely is not installed, PoLiS is unavailable.",
        }

    img_ids = list(sorted(coco_gt.imgs.keys()))
    polis_total = 0.0
    valid_image_count = 0
    matched_gt_count = 0

    for img_id in tqdm(img_ids, desc="Computing PoLiS"):
        gt_anns = coco_gt.loadAnns(coco_gt.getAnnIds(imgIds=[img_id]))
        dt_anns = coco_dt.loadAnns(coco_dt.getAnnIds(imgIds=[img_id]))
        if not gt_anns or not dt_anns:
            continue

        gt_polygons, gt_bboxes = get_instance_polygons(gt_anns)
        dt_polygons, dt_bboxes = get_instance_polygons(dt_anns)
        if not gt_polygons or not dt_polygons:
            continue

        iscrowd = [0] * len(gt_bboxes)
        ious = cocomask.iou(dt_bboxes, gt_bboxes, iscrowd)
        if ious.size == 0:
            continue

        image_polis_sum = 0.0
        image_match_count = 0

        for gt_idx, gt_points in enumerate(gt_polygons):
            matched_dt_idx = int(np.argmax(ious[:, gt_idx]))
            matched_iou = float(ious[matched_dt_idx, gt_idx])
            if matched_iou <= match_iou:
                continue

            gt_poly = make_polygon(gt_points)
            dt_poly = make_polygon(dt_polygons[matched_dt_idx])
            if gt_poly is None or dt_poly is None:
                continue

            image_polis_sum += compare_polys(gt_poly, dt_poly)
            image_match_count += 1

        if image_match_count == 0:
            continue

        polis_total += image_polis_sum / image_match_count
        matched_gt_count += image_match_count
        valid_image_count += 1

    if valid_image_count == 0:
        return {
            "PoLiS": None,
            "matched_gt_count": 0,
            "valid_image_count": 0,
            "reason": f"No matched polygon pairs found with IoU > {match_iou:.2f}.",
        }

    return {
        "PoLiS": float(polis_total / valid_image_count),
        "matched_gt_count": matched_gt_count,
        "valid_image_count": valid_image_count,
        "match_iou_threshold": match_iou,
    }


def compute_polygon_metrics(gt_path: str, dt_anns: Sequence[dict], polis_match_iou: float = 0.5) -> Dict[str, object]:
    if not os.path.exists(gt_path):
        raise FileNotFoundError(f"GT file not found: {gt_path}")

    coco_gt = COCO(gt_path)
    gt_anns = coco_gt.loadAnns(coco_gt.getAnnIds())
    gt_has_segm = any(has_valid_segmentation(ann) for ann in gt_anns)
    dt_has_segm = any(has_valid_segmentation(ann) for ann in dt_anns)

    if not gt_has_segm:
        return {
            "available": False,
            "reason": "Ground-truth annotations do not contain valid polygon segmentation.",
        }

    if not dt_has_segm:
        return {
            "available": False,
            "reason": "Detection results do not contain valid polygon segmentation.",
        }

    coco_dt = coco_gt.loadRes(copy.deepcopy(list(dt_anns)))
    overlap_metrics = compute_iou_and_ciou(coco_gt, coco_dt)
    polis_metrics = compute_polis(coco_gt, coco_dt, match_iou=polis_match_iou)

    metrics = {}
    metrics.update(overlap_metrics)
    metrics.update(polis_metrics)
    return {
        "available": True,
        "metrics": metrics,
    }


def print_results(results: Dict[str, object]) -> None:
    print("\n" + "=" * 80)
    print("Evaluation Summary")
    print("=" * 80)

    coco_results = results.get("coco_metrics", {})
    if coco_results.get("available"):
        print("\n[COCO Metrics]")
        for iou_type, metrics in coco_results["metrics"].items():
            print(f"  {iou_type}:")
            for key, value in metrics.items():
                print(f"    {key:12s}: {value:.4f}")
    else:
        print("\n[COCO Metrics]")
        print(f"  Skipped: {coco_results.get('reason', 'Unknown reason')}")

    polygon_results = results.get("polygon_metrics", {})
    if polygon_results.get("available"):
        print("\n[Polygon Metrics]")
        for key, value in polygon_results["metrics"].items():
            if isinstance(value, float):
                print(f"  {key:18s}: {value:.4f}")
            else:
                print(f"  {key:18s}: {value}")
    else:
        print("\n[Polygon Metrics]")
        print(f"  Skipped: {polygon_results.get('reason', 'Unknown reason')}")

    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate building predictions with COCO mAP and polygon metrics (IoU, C-IoU, PoLiS)."
    )
    parser.add_argument(
        "--gt",
        required=True,
        help="Path to ground-truth COCO json",
    )
    parser.add_argument(
        "--dt",
        required=True,
        help="Path to detection results json",
    )
    parser.add_argument("--iou", choices=["bbox", "segm"], default=None, help="Force COCO IoU type")
    parser.add_argument(
        "--max-dets",
        nargs=3,
        type=int,
        default=(1, 10, 100),
        metavar=("AR1", "AR10", "AR100"),
        help="Three maxDet values used by COCOeval",
    )
    parser.add_argument(
        "--polis-match-iou",
        type=float,
        default=0.5,
        help="Minimum bbox IoU used to match GT/DT instances before computing PoLiS",
    )
    parser.add_argument("--skip-map", action="store_true", help="Skip COCO mAP evaluation")
    parser.add_argument("--skip-polygon", action="store_true", help="Skip IoU / C-IoU / PoLiS evaluation")
    parser.add_argument("--output", type=str, default=None, help="Path to save results json (optional)")
    args = parser.parse_args()

    if not os.path.exists(args.gt):
        raise FileNotFoundError(f"GT file not found: {args.gt}")
    if not os.path.exists(args.dt):
        raise FileNotFoundError(f"Detection file not found: {args.dt}")

    dt_anns = load_detection_annotations(args.dt)
    results: Dict[str, object] = {
        "config": {
            "gt": args.gt,
            "dt": args.dt,
            "iou": args.iou,
            "max_dets": list(args.max_dets),
            "polis_match_iou": args.polis_match_iou,
            "skip_map": args.skip_map,
            "skip_polygon": args.skip_polygon,
        }
    }

    if args.skip_map:
        results["coco_metrics"] = {"available": False, "reason": "Skipped by --skip-map"}
    else:
        results["coco_metrics"] = compute_map(
            args.gt,
            dt_anns,
            iou_type=args.iou,
            max_dets=args.max_dets,
        )

    if args.skip_polygon:
        results["polygon_metrics"] = {"available": False, "reason": "Skipped by --skip-polygon"}
    else:
        results["polygon_metrics"] = compute_polygon_metrics(
            args.gt,
            dt_anns,
            polis_match_iou=args.polis_match_iou,
        )

    print_results(results)

    if args.output:
        output_dir = os.path.dirname(os.path.abspath(args.output))
        os.makedirs(output_dir, exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
