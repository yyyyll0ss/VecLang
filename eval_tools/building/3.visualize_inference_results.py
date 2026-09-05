import os
import json
import argparse
from collections import defaultdict
import random

import cv2
import numpy as np


SAVE_DPI = 600
BUILDING_OUTLINE = (182, 47, 54, 255)
BUILDING_FILL = (243, 185, 184, 150)
BUILDING_NODE = BUILDING_OUTLINE
BUILDING_NODE_RADIUS = 3
BUILDING_OUTLINE_THICKNESS = 2


def cv2_imwrite_chinese_path(filepath, img):
    """
    cv2.imwrite 无法处理中文路径，使用 cv2.imencode + 文件写入来解决
    """
    ext = os.path.splitext(filepath)[1]
    result, encoded = cv2.imencode(ext, img)
    if result:
        with open(filepath, 'wb') as f:
            f.write(encoded.tobytes())
        return True
    return False


def load_dt(dt_path):
    with open(dt_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def load_gt_annotations(gt_path):
    if not os.path.exists(gt_path):
        return {}
    with open(gt_path, 'r', encoding='utf-8') as f:
        gt = json.load(f)
    img2anns = defaultdict(list)
    for ann in gt.get('annotations', []):
        image_id = ann.get('image_id')
        img2anns[image_id].append(ann)
    return img2anns


def build_id2filename_from_gt(gt_path, image_dir=None):
    if not os.path.exists(gt_path):
        return {}
    with open(gt_path, 'r', encoding='utf-8') as f:
        gt = json.load(f)
    id2fn = {}
    for img in gt.get('images', []):
        fname = img.get('file_name', '')
        if fname:
            # use basename so we can join with provided image_dir
            id2fn[img['id']] = os.path.basename(fname)
    return id2fn


def try_locate_image(image_dir, basename_stem):
    # try common extensions
    exts = ['.tif', '.tiff', '.jpg', '.jpeg', '.png', '.bmp']
    for ext in exts:
        candidate = os.path.join(image_dir, f"{basename_stem}{ext}")
        if os.path.exists(candidate):
            return candidate
    # as fallback try the stem itself (maybe contains extension)
    candidate = os.path.join(image_dir, basename_stem)
    if os.path.exists(candidate):
        return candidate
    return None


def draw_dt_on_image(img, predictions):
    """绘制仅包含预测结果的可视化"""
    out = img.copy()
    
    # DT 颜色配置（更醒目）
    dt_edge_color = (0, 255, 255)     # 黄色用于 DT 轮廓线
    dt_node_color = (0, 0, 255)       # 红色用于 DT 节点
    dt_node_radius = 2                # DT 节点圆点半径
    dt_edge_thickness = 2             # DT 轮廓线粗细

    # 绘制 DT（预测）注释
    for pred in predictions:
        # segmentation may be list of lists
        segm = pred.get('segmentation', [])
        if isinstance(segm, list) and len(segm) > 0:
            # common COCO style: segmentation is [ [x1,y1,...] ]
            seg_coords = segm[0] if isinstance(segm[0], list) else segm
            try:
                pts = np.array(seg_coords, dtype=np.float32).reshape(-1, 2)
            except Exception:
                continue
            
            pts_int = np.round(pts).astype(np.int32)
            
            # 绘制 DT 轮廓线
            cv2.polylines(out, [pts_int], isClosed=True, color=dt_edge_color, thickness=dt_edge_thickness)
            
            # 绘制 DT 顶点节点
            for pt in pts_int:
                cv2.circle(out, tuple(pt), dt_node_radius, dt_node_color, -1)

    return out


def draw_gt_on_image(img, gt_annotations):
    """绘制仅包含 GT 的可视化"""
    out = img.copy()
    
    # GT 颜色配置（更醒目）
    gt_edge_color = (0, 165, 255)     # 橙色用于 GT 轮廓线
    gt_node_color = (255, 0, 255)     # 洋红用于 GT 节点
    gt_node_radius = 2                # GT 节点圆点半径
    gt_edge_thickness = 2             # GT 轮廓线粗细

    # 绘制 GT 注释
    for gt_ann in gt_annotations:
        segm = gt_ann.get('segmentation', [])
        if isinstance(segm, list) and len(segm) > 0:
            seg_coords = segm[0] if isinstance(segm[0], list) else segm
            try:
                pts = np.array(seg_coords, dtype=np.float32).reshape(-1, 2)
            except Exception:
                continue
            
            pts_int = np.round(pts).astype(np.int32)
            
            # 绘制 GT 轮廓线
            cv2.polylines(out, [pts_int], isClosed=True, color=gt_edge_color, thickness=gt_edge_thickness)
            
            # 绘制 GT 顶点节点
            for pt in pts_int:
                cv2.circle(out, tuple(pt), gt_node_radius, gt_node_color, -1)

    return out


def rgba_to_bgr(rgba):
    r, g, b, _ = rgba
    return (b, g, r)


def blend_overlay(base, overlay, rgba):
    alpha = rgba[3] / 255.0
    if alpha >= 1.0:
        return overlay
    if alpha <= 0.0:
        return base
    return cv2.addWeighted(overlay, alpha, base, 1.0 - alpha, 0)


def iter_segmentation_polygons(annotation):
    segm = annotation.get('segmentation', [])
    if not isinstance(segm, list) or len(segm) == 0:
        return

    if all(isinstance(item, (int, float)) for item in segm):
        polygons = [segm]
    else:
        polygons = segm

    for seg_coords in polygons:
        try:
            pts = np.array(seg_coords, dtype=np.float32).reshape(-1, 2)
        except Exception:
            continue
        if len(pts) < 3:
            continue
        yield np.round(pts).astype(np.int32)


def draw_buildings_on_image(img, annotations):
    out = img.copy()
    polygons = []
    for ann in annotations:
        polygons.extend(iter_segmentation_polygons(ann))

    if not polygons:
        return out

    fill_overlay = out.copy()
    cv2.fillPoly(fill_overlay, polygons, color=rgba_to_bgr(BUILDING_FILL), lineType=cv2.LINE_AA)
    out = blend_overlay(out, fill_overlay, BUILDING_FILL)

    outline_overlay = out.copy()
    for pts_int in polygons:
        cv2.polylines(
            outline_overlay,
            [pts_int],
            isClosed=True,
            color=rgba_to_bgr(BUILDING_OUTLINE),
            thickness=BUILDING_OUTLINE_THICKNESS,
            lineType=cv2.LINE_AA,
        )
    out = blend_overlay(out, outline_overlay, BUILDING_OUTLINE)

    node_overlay = out.copy()
    for pts_int in polygons:
        for pt in pts_int:
            cv2.circle(
                node_overlay,
                tuple(pt),
                BUILDING_NODE_RADIUS,
                rgba_to_bgr(BUILDING_NODE),
                -1,
                lineType=cv2.LINE_AA,
            )
    out = blend_overlay(out, node_overlay, BUILDING_NODE)
    return out


def draw_dt_on_image(img, predictions):
    return draw_buildings_on_image(img, predictions)


def draw_gt_on_image(img, gt_annotations):
    return draw_buildings_on_image(img, gt_annotations)


def cv2_imwrite_chinese_path(filepath, img, dpi=SAVE_DPI):
    ext = os.path.splitext(filepath)[1]
    try:
        from PIL import Image

        format_by_ext = {
            '.jpg': 'JPEG',
            '.jpeg': 'JPEG',
            '.png': 'PNG',
            '.tif': 'TIFF',
            '.tiff': 'TIFF',
            '.bmp': 'BMP',
        }
        image_format = format_by_ext.get(ext.lower())
        if image_format is not None:
            if img.ndim == 2:
                pil_img = Image.fromarray(img)
            elif img.shape[2] == 4:
                pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA))
            else:
                pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

            save_kwargs = {'dpi': (dpi, dpi)}
            if image_format == 'JPEG':
                save_kwargs.update({'quality': 95, 'subsampling': 0})
                if pil_img.mode != 'RGB':
                    pil_img = pil_img.convert('RGB')

            with open(filepath, 'wb') as f:
                pil_img.save(f, format=image_format, **save_kwargs)
            return True
    except ImportError:
        print('Warning: Pillow is not installed; saving without DPI metadata.')
    except Exception as exc:
        print(f'Warning: Pillow save failed for {filepath}: {exc}; falling back to cv2.imencode.')

    result, encoded = cv2.imencode(ext, img)
    if result:
        with open(filepath, 'wb') as f:
            f.write(encoded.tobytes())
        return True
    return False


def main():
    parser = argparse.ArgumentParser(description='Visualize COCO-format inference results on images')
    parser.add_argument('--dt', required=True, help='COCO-format detections JSON')
    parser.add_argument('--gt', default=None, help='Optional GT COCO JSON used to map image_id to file_name')
    parser.add_argument('--image_dir', required=True, help='Directory containing images')
    parser.add_argument('--out_dir', required=True, help='Directory to save visualizations')
    parser.add_argument(
        '--max_images',
        type=int,
        default=0,
        help='Maximum number of images to visualize; 0 means all images (default)',
    )
    parser.add_argument('--seed', type=int, default=42, help='Unused legacy argument kept for compatibility')
    args = parser.parse_args()

    if args.max_images < 0:
        parser.error('--max_images must be 0 (all images) or a positive integer')
    if not os.path.isfile(args.dt):
        parser.error(f'Detection JSON not found: {args.dt}')
    if args.gt and not os.path.isfile(args.gt):
        parser.error(f'GT JSON not found: {args.gt}')
    if not os.path.isdir(args.image_dir):
        parser.error(f'Image directory not found: {args.image_dir}')

    os.makedirs(args.out_dir, exist_ok=True)
    # 创建 GT 可视化专用文件夹
    gt_out_dir = args.out_dir + '_gt'
    os.makedirs(gt_out_dir, exist_ok=True)

    print('Loading detections from', args.dt)
    dt = load_dt(args.dt)
    print('Total detections:', len(dt))

    # group by image_id
    img2preds = defaultdict(list)
    for det in dt:
        img2preds[det.get('image_id')].append(det)

    print('Unique image ids in detections:', len(img2preds))

    id2fn = build_id2filename_from_gt(args.gt, args.image_dir) if os.path.exists(args.gt) else {}
    
    # 加载 GT 注释
    img2gt_anns = load_gt_annotations(args.gt) if os.path.exists(args.gt) else {}
    if img2gt_anns:
        print('Loaded GT annotations for {} images'.format(len(img2gt_anns)))

    # Build ordered list of images to visualize and ensure one-to-one with input images
    images_to_process = []  # list of (image_id_or_None, filename_basename)

    if os.path.exists(args.gt):
        # use GT images order to guarantee correspondence
        with open(args.gt, 'r', encoding='utf-8') as f:
            gt = json.load(f)
        for img in gt.get('images', []):
            fname = img.get('file_name', '')
            if fname:
                images_to_process.append((img.get('id'), os.path.basename(fname)))
    else:
        # fallback: enumerate files in image_dir (sorted)
        if os.path.exists(args.image_dir):
            files = [fn for fn in sorted(os.listdir(args.image_dir)) if os.path.splitext(fn)[1].lower() in ['.tif', '.tiff', '.jpg', '.jpeg', '.png', '.bmp']]
            for idx, fn in enumerate(files):
                # try to infer image_id from numeric stem, otherwise None
                stem = os.path.splitext(fn)[0]
                try:
                    image_id = int(stem) if stem.isdigit() else None
                except Exception:
                    image_id = None
                images_to_process.append((image_id, fn))
        else:
            # no GT and no image_dir: fall back to detections' image ids
            all_image_ids = sorted(list(img2preds.keys()))
            for iid in all_image_ids:
                images_to_process.append((iid, str(iid)))

    total_images = len(images_to_process)

    # max_images=0 means no limit. A positive value keeps the first N entries.
    if args.max_images > 0 and total_images > args.max_images:
        images_to_process = images_to_process[:args.max_images]

    print(f'Images available: {total_images}')
    print(f'Visualizing: {len(images_to_process)} images')
    print(f'DT predictions -> {args.out_dir} (building fill, outline and nodes; {SAVE_DPI} DPI)')
    if img2gt_anns:
        print(f'GT annotations -> {gt_out_dir} (building fill, outline and nodes; {SAVE_DPI} DPI)')

    skipped = 0
    save_failed = 0
    saved_dt = 0
    saved_gt = 0
    for index, (image_id, basename) in enumerate(images_to_process, start=1):
        # resolve filename path
        filename = None
        # if GT mapping available, prefer that
        if image_id is not None and image_id in id2fn:
            filename = os.path.join(args.image_dir, id2fn[image_id])
        else:
            filename = os.path.join(args.image_dir, basename)
            if not os.path.exists(filename):
                # try locate by stem without extension
                filename = try_locate_image(args.image_dir, os.path.splitext(basename)[0])

        if not filename or not os.path.exists(filename):
            print(f'Warning: image file for entry ({image_id}, {basename}) not found (tried {filename})')
            skipped += 1
            continue

        img = cv2.imread(filename)
        if img is None:
            print('Warning: failed to read', filename)
            skipped += 1
            continue

        # get preds (may be empty) and GT anns (may be empty)
        preds = img2preds.get(image_id, []) if image_id is not None else []
        gt_anns = img2gt_anns.get(image_id, []) if image_id is not None else []

        base = os.path.splitext(os.path.basename(filename))[0]

        # always save DT visualization (even if preds empty)
        vis_dt = draw_dt_on_image(img, preds)
        out_path_dt = os.path.join(args.out_dir, f"{base}_vis_dt.jpg")
        if cv2_imwrite_chinese_path(out_path_dt, vis_dt):
            saved_dt += 1
        else:
            print('Warning: failed to save', out_path_dt)
            save_failed += 1

        # always save GT visualization if GT file provided (even if empty, produce same image)
        if os.path.exists(args.gt):
            vis_gt = draw_gt_on_image(img, gt_anns)
            out_path_gt = os.path.join(gt_out_dir, f"{base}_vis_gt.jpg")
            if cv2_imwrite_chinese_path(out_path_gt, vis_gt):
                saved_gt += 1
            else:
                print('Warning: failed to save', out_path_gt)
                save_failed += 1

        if index % 100 == 0 or index == len(images_to_process):
            print(f'Progress: {index}/{len(images_to_process)}')

    print(f'Done. DT saved: {saved_dt}; GT saved: {saved_gt}; skipped: {skipped}; save failed: {save_failed}')


if __name__ == '__main__':
    main()
