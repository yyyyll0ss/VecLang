#!/usr/bin/env python3
"""
Analyze and visualize prediction performance based on Segmentation IoU scores.
Color-code detections by performance level to identify problematic predictions.
"""
import json
import os
import cv2
import numpy as np
from pathlib import Path
from collections import defaultdict
import argparse
import random
from pycocotools.coco import COCO
from pycocotools.mask import decode as mask_decode, encode as mask_encode
from pycocotools import mask as maskUtils


# Performance levels with colors (BGR for OpenCV) - more vibrant colors
PERFORMANCE_LEVELS = {
    'excellent': {'range': (0.75, 1.00), 'color': (0, 255, 0), 'color_overlay': (80, 220, 80), 'name': 'Excellent (>=0.75)', 'thickness': 3},
    'good': {'range': (0.50, 0.75), 'color': (255, 255, 0), 'color_overlay': (220, 220, 0), 'name': 'Good (0.50-0.75)', 'thickness': 2},
    'fair': {'range': (0.30, 0.50), 'color': (0, 255, 255), 'color_overlay': (0, 220, 220), 'name': 'Fair (0.30-0.50)', 'thickness': 2},
    'poor': {'range': (0.10, 0.30), 'color': (0, 165, 255), 'color_overlay': (0, 140, 220), 'name': 'Poor (0.10-0.30)', 'thickness': 2},
    'missing': {'range': (0.00, 0.10), 'color': (0, 0, 255), 'color_overlay': (0, 0, 220), 'name': 'Pred Missing (<0.10)', 'thickness': 2},
}

# Style for GT objects that are not matched by any prediction above threshold
GT_MISSING_STYLE = {
    'color': (255, 0, 255),           # BGR: magenta outline
    'color_overlay': (200, 0, 200),   # overlay: purple
    'name': 'GT Missing',
    'thickness': 2,
}


def compute_bbox_iou(box1, box2):
    x, y, w, h = box1
    a, b, c, d = box2
    intersection = max(0, min(x+w, a+c)-max(x, a)) * max(0, min(y+h, b+d)-max(y, b))
    union = w*h + c*d - intersection
    return intersection / union if union > 0 else 0.0


def polygon_to_bbox(polygon):
    """Convert polygon (list of x,y coords) to bbox [x, y, w, h]."""
    if not polygon or len(polygon) < 2:
        return None
    coords = np.array(polygon).reshape(-1, 2)
    x_min, y_min = coords.min(axis=0)
    x_max, y_max = coords.max(axis=0)
    return [float(x_min), float(y_min), float(x_max - x_min), float(y_max - y_min)]


def rle_to_bbox(rle, img_h, img_w):
    """Convert RLE segmentation to bbox."""
    if not rle or not rle.get('counts'):
        return None
    try:
        mask = mask_decode(rle)
        coords = np.where(mask)
        if len(coords[0]) == 0:
            return None
        y_min, y_max = coords[0].min(), coords[0].max()
        x_min, x_max = coords[1].min(), coords[1].max()
        return [float(x_min), float(y_min), float(x_max - x_min + 1), float(y_max - y_min + 1)]
    except:
        return None


def segm_to_mask(seg, h, w):
    # 支持RLE和polygon
    if seg is None:
        return None
    try:
        if isinstance(seg, dict):
            return mask_decode(seg)
        elif isinstance(seg, list):
            rles = maskUtils.frPyObjects(seg, h, w)
            merged = maskUtils.merge(rles)
            return mask_decode(merged)
    except Exception:
        return None
    return None

def compute_segm_iou(mask1, mask2):
    # mask1, mask2: numpy array, 0/1
    if mask1 is None or mask2 is None:
        return 0.0
    inter = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    if union == 0:
        return 0.0
    return inter / union

def compute_detection_ious(gt_annotations, dt_annotations, img_h, img_w):
    """
    用segmentation IoU计算每个预测与GT的最大IoU
    返回: dt_index -> {'iou': float, 'gt_id': int or None}
    """
    dt_ious = {}
    gt_masks = []
    for gt_ann in gt_annotations:
        gt_masks.append(segm_to_mask(gt_ann.get('segmentation'), img_h, img_w))
    for dt_idx, dt_ann in enumerate(dt_annotations):
        dt_mask = segm_to_mask(dt_ann.get('segmentation'), img_h, img_w)
        best_iou = 0.0
        best_gt_id = None
        for gt_idx, gt_mask in enumerate(gt_masks):
            iou = compute_segm_iou(dt_mask, gt_mask)
            if iou > best_iou:
                best_iou = iou
                best_gt_id = gt_annotations[gt_idx].get('id')
        dt_ious[dt_idx] = {'iou': best_iou, 'gt_id': best_gt_id}
    return dt_ious


def get_performance_level(iou):
    """Get performance level and color for an IoU value."""
    for level_name, level_info in PERFORMANCE_LEVELS.items():
        iou_min, iou_max = level_info['range']
        if iou_min <= iou < iou_max:
            return level_name, level_info
    # Default: excellent if >= 0.75
    if iou >= 0.75:
        return 'excellent', PERFORMANCE_LEVELS['excellent']
    return 'missing', PERFORMANCE_LEVELS['missing']


def draw_mask_on_image(img, mask, color, alpha=0.35, outline_thickness=2):
    if mask is None:
        return
    mask_bool = mask.astype(bool)
    if mask_bool.ndim == 3:
        mask_bool = mask_bool.squeeze()
    if not mask_bool.any():
        return
    # 仅在掩膜区域进行混合，避免整图变暗
    color_arr = np.array(color, dtype=np.uint8).reshape(1, 1, 3)
    overlay_region = np.zeros_like(img, dtype=np.uint8)
    overlay_region[mask_bool] = color_arr
    blended = img.copy()
    blended[mask_bool] = cv2.addWeighted(overlay_region[mask_bool], alpha, img[mask_bool], 1 - alpha, 0)
    img[:] = blended
    # 轮廓描边，增强可见性
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img, contours, -1, (int(color[0]), int(color[1]), int(color[2])), outline_thickness)

def draw_bbox_on_image(img, bbox, color, thickness, label_text=''):
    if bbox is None or len(bbox) < 4:
        return
    x, y, w, h = bbox
    x, y, w, h = int(x), int(y), int(w), int(h)
    x2 = x + w
    y2 = y + h
    x, y = max(0, x), max(0, y)
    x2, y2 = min(img.shape[1], x2), min(img.shape[0], y2)
    cv2.rectangle(img, (x, y), (x2, y2), color, thickness)
    if label_text:
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        text_thickness = 2
        (text_width, text_height), _ = cv2.getTextSize(label_text, font, font_scale, text_thickness)
        # 文本背景与阴影提升可读性
        cv2.rectangle(img, (x, y - text_height - 6), (x + text_width, y), (0, 0, 0), -1)
        cv2.putText(img, label_text, (x+1, y - 1), font, font_scale, (0, 0, 0), text_thickness)
        cv2.putText(img, label_text, (x, y - 2), font, font_scale, (255, 255, 255), text_thickness)


def analyze_and_visualize(
    gt_path,
    dt_path,
    image_dir,
    output_dir,
    sample_size=200,
    category_id=None,
    prefer_dt=True,
    only_with_dt=True,
    seed=42,
    alpha=0.35,
    outline=2,
    brighten=0,
    gamma=1.0,
    match_thr=0.10,
):
    """
    Analyze predictions and create visualization with color-coded performance.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    coco_gt = COCO(gt_path)
    if category_id is None:
        raise ValueError("category_id is required for multiclass analysis")
    with open(dt_path, 'r', encoding='utf-8') as f:
        dt_anns = json.load(f)
    if isinstance(dt_anns, dict):
        dt_anns = dt_anns["annotations"]
    dt_anns = [a for a in dt_anns if a["category_id"] == category_id]
    
    # Group detections by image_id
    dt_by_img = defaultdict(list)
    def normalize_img_id(x):
        try:
            return int(x)
        except Exception:
            return x
    for dt_idx, dt_ann in enumerate(dt_anns):
        iid = normalize_img_id(dt_ann.get('image_id'))
        dt_by_img[iid].append((dt_idx, dt_ann))
    
    # Statistics
    stats = defaultdict(int)
    
    # Decide candidate images: prefer those with DT to ensure targets present
    dt_img_ids = [normalize_img_id(i) for i in dt_by_img.keys()]
    gt_img_ids = [normalize_img_id(i) for i in coco_gt.imgs.keys()]
    candidates = dt_img_ids if prefer_dt else gt_img_ids
    if not candidates:
        candidates = gt_img_ids
    if only_with_dt:
        candidates = [iid for iid in candidates if iid in dt_by_img]

    # Random sample
    random.seed(seed)
    if len(candidates) <= sample_size:
        img_ids = candidates
    else:
        img_ids = random.sample(candidates, sample_size)
    
    for img_id in img_ids:
        img_info = coco_gt.imgs[img_id]
        img_h, img_w = img_info.get('height', 0), img_info.get('width', 0)
        
        # Load image
        img_path = os.path.join(image_dir, img_info.get('file_name', ''))
        if not os.path.exists(img_path):
            print(f'  Skipping {img_path} (not found)')
            continue
        
        img = cv2.imread(img_path)
        if img is None:
            print(f'  Failed to load {img_path}')
            continue
        # 亮度/伽马增强，提升整体可见性
        if brighten != 0 or gamma != 1.0:
            if brighten != 0:
                img = cv2.convertScaleAbs(img, alpha=1.0, beta=int(brighten))
            if gamma != 1.0:
                invGamma = 1.0 / float(gamma)
                table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype('uint8')
                img = cv2.LUT(img, table)
        
        # Get GT and DT annotations for this image
        # Fetch GT anns via API to ensure full set
        gt_ann_ids = coco_gt.getAnnIds(imgIds=[img_id], catIds=[category_id])
        gt_anns = coco_gt.loadAnns(gt_ann_ids) if gt_ann_ids else []
        # All DT anns for this image
        dt_anns_for_img = [ann for _, ann in dt_by_img.get(img_id, [])]
        
        # Compute IoUs (segm)
        dt_ious = compute_detection_ious(gt_anns, dt_anns_for_img, img_h, img_w)
        # Cache DT masks for GT-missing check
        dt_masks_cache = []
        # Draw detections with color-coding (mask+框)
        drawn_count = 0
        for dt_idx, dt_ann in enumerate(dt_anns_for_img):
            iou_info = dt_ious.get(dt_idx, {'iou': 0.0})
            iou = iou_info['iou']
            level_name, level_info = get_performance_level(iou)
            stats[level_name] += 1
            # mask可视化
            mask = segm_to_mask(dt_ann.get('segmentation'), img_h, img_w)
            dt_masks_cache.append(mask)
            if mask is None:
                # fallback: use bbox to create a rectangular mask for stronger visibility
                bb = dt_ann.get('bbox')
                if bb is not None and len(bb) >= 4:
                    x, y, w, h = map(int, bb[:4])
                    x2, y2 = x + w, y + h
                    x, y = max(0, x), max(0, y)
                    x2, y2 = min(img.shape[1], x2), min(img.shape[0], y2)
                    rect_mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
                    rect_mask[y:y2, x:x2] = 1
                    draw_mask_on_image(img, rect_mask, level_info['color_overlay'], alpha=alpha, outline_thickness=outline)
                    drawn_count += 1
            else:
                draw_mask_on_image(img, mask, level_info['color_overlay'], alpha=alpha, outline_thickness=outline)
                drawn_count += 1
            # 框可视化
            dt_bbox = dt_ann.get('bbox')
            label = f'{level_name} {iou:.2f}'
            draw_bbox_on_image(img, dt_bbox, level_info['color'], level_info['thickness'], label)

        # Visualize GT objects that are missing (no DT IoU >= match_thr)
        gt_missing_count = 0
        for gt_ann in gt_anns:
            gt_mask = segm_to_mask(gt_ann.get('segmentation'), img_h, img_w)
            if gt_mask is None:
                continue
            best_iou = 0.0
            for dt_m in dt_masks_cache:
                best_iou = max(best_iou, compute_segm_iou(gt_mask, dt_m))
            if best_iou < match_thr:
                draw_mask_on_image(img, gt_mask, GT_MISSING_STYLE['color_overlay'], alpha=alpha, outline_thickness=outline)
                stats['gt_missing'] += 1
                gt_missing_count += 1

        # Header text: counts
        header = f"DT:{len(dt_anns_for_img)} Drawn:{drawn_count} GT:{len(gt_anns)} GT-miss:{gt_missing_count} | IoU: segm"
        cv2.putText(img, header, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 3)
        cv2.putText(img, header, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        
        # Draw GT mask轮廓（白色）
        for gt_ann in gt_anns:
            mask = segm_to_mask(gt_ann.get('segmentation'), img_h, img_w)
            if mask is not None:
                contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(img, contours, -1, (255,255,255), 1)
        
        # Add legend
        legend_y = 30
        for level_name in ['excellent', 'good', 'fair', 'poor', 'missing']:
            level_info = PERFORMANCE_LEVELS[level_name]
            text = f'{level_info["name"]}'
            color = level_info['color']
            cv2.putText(img, text, (10, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            legend_y += 25
        # GT-missing legend
        cv2.putText(img, f'{GT_MISSING_STYLE["name"]} (<{match_thr:.2f})', (10, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, GT_MISSING_STYLE['color'], 2)
        legend_y += 25
        
        # Save visualization
        filename = Path(img_info.get('file_name', f'img_{img_id}')).stem
        output_path = os.path.join(output_dir, f'{filename}_analysis.jpg')
        cv2.imwrite(output_path, img)
        print(f'  Saved: {output_path}')
    
    # Print statistics
    print(f'\n{"="*80}')
    print('Performance Statistics:')
    print(f'{"="*80}')
    total = sum(stats.values())
    for level_name in ['excellent', 'good', 'fair', 'poor', 'missing', 'gt_missing']:
        count = stats.get(level_name, 0)
        pct = (count / total * 100) if total > 0 else 0
        print(f'{level_name:15} : {count:6d} ({pct:6.2f}%)')
    print(f'{"="*80}')
    
    return stats


def main():
    parser = argparse.ArgumentParser(description='Analyze and visualize prediction performance by IoU')
    parser.add_argument('--gt', required=True, help='Ground truth COCO json')
    parser.add_argument('--dt', required=True, help='Detection results json')
    parser.add_argument('--images', required=True, help='Image directory')
    parser.add_argument('--output', required=True, help='Output directory for visualizations')
    parser.add_argument('--sample-size', type=int, default=100, help='Random sample size of images')
    parser.add_argument('--prefer-dt', action='store_true', default=False, help='Prefer images that have detections')
    parser.add_argument('--only-with-dt', action='store_true', default=False, help='Only visualize images that have detections')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for sampling')
    parser.add_argument('--alpha', type=float, default=0.35, help='Mask overlay alpha (lower is lighter)')
    parser.add_argument('--outline', type=int, default=2, help='Mask contour outline thickness')
    parser.add_argument('--brighten', type=int, default=15, help='Increase image brightness by this value')
    parser.add_argument('--gamma', type=float, default=1.1, help='Gamma correction (>1.0 brightens mid-tones)')
    parser.add_argument('--match-thr', type=float, default=0.10, help='IoU threshold to consider a GT as matched (for GT-missing visualization)')
    parser.add_argument("--category-id", type=int, default=2)
    args = parser.parse_args()
    
    print(f'Analyzing predictions with IoU-based performance coloring...\n')
    stats = analyze_and_visualize(
        args.gt,
        args.dt,
        args.images,
        args.output,
        sample_size=args.sample_size,
        category_id=args.category_id,
        prefer_dt=args.prefer_dt,
        only_with_dt=args.only_with_dt,
        seed=args.seed,
        alpha=args.alpha,
        outline=args.outline,
        brighten=args.brighten,
        gamma=args.gamma,
        match_thr=args.match_thr,
    )
    
    print(f'\nVisualizations saved to: {os.path.abspath(args.output)}')


if __name__ == '__main__':
    main()
