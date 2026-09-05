"""
实例级patch预测结果转换为512x512 patch级COCO格式

功能：
- 读取实例级patch的预测结果（jsonl格式）
- 使用实例级注释中的crop_info进行逆向坐标转换
- 按512x512 patch（original_image）聚合预测结果
- 生成与validation_patches_512.json对应的COCO格式输出
"""

import argparse
import os
import json
import cv2
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict


def safe_load_json_list(raw_text):
    """
    从可能被截断的 JSON 文本中，尽量解析出完整的 dict 列表，
    自动丢弃最后一个不完整的对象
    """
    raw_text = raw_text.strip()

    if not raw_text.startswith("["):
        raise ValueError("raw_text 不是 JSON list")

    # 尝试直接解析
    try:
        data = json.loads(raw_text)
        if isinstance(data, list):
            return data
    except Exception:
        pass

    # ===== 截断修复：逐个 {} 解析 =====
    objs = []
    buf = ""
    depth = 0
    started = False

    for ch in raw_text:
        if ch == "{":
            started = True
            depth += 1
        if started:
            buf += ch
        if ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    obj = json.loads(buf)
                    objs.append(obj)
                except Exception:
                    pass
                buf = ""
                started = False

    return objs


def strip_code_fence(text):
    """移除模型输出中可能携带的 markdown code fence。"""
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


def parse_prediction_objects(predict_value):
    """
    将 predict 字段统一解析为对象列表。

    兼容以下情况：
    - JSON 数组字符串: "[{...}]"
    - 单个 JSON 对象字符串: "{...}"
    - 已经是 list / dict
    - 被 ```json ... ``` 包裹的输出
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

    # 优先直接解析完整 JSON，兼容数组和单对象。
    try:
        data = json.loads(text)
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            return [data]
    except Exception:
        pass

    # 向后兼容旧逻辑：尝试从字符串中抽取数组。
    try:
        extracted = extract_json_array_from_string(text)
        data = json.loads(extracted)
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            return [data]
    except Exception:
        pass

    # 最后尝试抽取单个对象。
    start_idx = text.find('{')
    end_idx = text.rfind('}')
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


def extract_json_array_from_string(text):
    """
    从字符串中提取 JSON 数组
    """
    text = text.strip()
    
    if text.startswith('['):
        return text
    
    start_idx = text.find('[')
    if start_idx == -1:
        raise ValueError("找不到 JSON 数组开始符号 '['")
    
    end_idx = max(text.rfind('}'), text.rfind(']'))
    if end_idx == -1:
        raise ValueError("找不到 JSON 结束符号")
    
    extracted = text[start_idx:end_idx+1]
    return extracted


def safe_pair_coords(coord_list):
    """保证坐标成对"""
    if not isinstance(coord_list, list):
        return []

    if len(coord_list) % 2 == 1:
        coord_list = coord_list[:-1]

    pairs = []
    for i in range(0, len(coord_list), 2):
        try:
            x = float(coord_list[i])
            y = float(coord_list[i + 1])
            pairs.append((x, y))
        except Exception:
            continue

    return pairs


def inverse_transform_point(x, y, crop_info):
    """
    将裁剪图像上的单个点坐标转换回原始图像坐标
    
    Args:
        x: 裁剪图像上的x坐标 (0-256)
        y: 裁剪图像上的y坐标 (0-256)
        crop_info: 裁剪信息字典，包含 crop_x, crop_y, crop_w, crop_h, output_size
    
    Returns:
        (original_x, original_y): 原始图像上的坐标
    """
    crop_x = crop_info['crop_x']
    crop_y = crop_info['crop_y']
    crop_w = crop_info['crop_w']
    crop_h = crop_info['crop_h']
    output_size = crop_info['output_size']
    
    # 计算缩放比例的逆
    scale_x_inv = crop_w / output_size
    scale_y_inv = crop_h / output_size
    
    # 逆向变换: original = new * (crop_size / output_size) + crop_offset
    original_x = x * scale_x_inv + crop_x
    original_y = y * scale_y_inv + crop_y
    
    return original_x, original_y


def inverse_transform_coords(coords, crop_info):
    """
    将裁剪图像上的坐标列表转换回原始图像坐标
    
    Args:
        coords: 坐标对列表 [(x1, y1), (x2, y2), ...]
        crop_info: 裁剪信息字典
    
    Returns:
        转换后的坐标对列表
    """
    transformed = []
    for x, y in coords:
        orig_x, orig_y = inverse_transform_point(x, y, crop_info)
        transformed.append((orig_x, orig_y))
    return transformed


def polygon_to_bbox(coords):
    """从多边形坐标计算 bbox"""
    if not coords or len(coords) < 2:
        return None
    
    if isinstance(coords[0], (list, tuple)):
        xs = [c[0] for c in coords]
        ys = [c[1] for c in coords]
    else:
        # 平铺格式
        coords = safe_pair_coords(coords)
        if not coords:
            return None
        xs = [c[0] for c in coords]
        ys = [c[1] for c in coords]
    
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    
    width = x_max - x_min
    height = y_max - y_min
    
    return [x_min, y_min, width, height]


def polygon_area(coords):
    """计算多边形面积（鞋带公式）"""
    if isinstance(coords[0], (list, tuple)):
        points = coords
    else:
        points = safe_pair_coords(coords)
    
    if len(points) < 3:
        return 0.0
    
    area = 0.0
    n = len(points)
    for i in range(n):
        x1, y1 = points[i]
        x2, y2 = points[(i + 1) % n]
        area += x1 * y2 - x2 * y1
    
    return abs(area) / 2.0


def get_image_dimensions(image_path):
    """获取图像的宽和高"""
    try:
        img = cv2.imread(image_path)
        if img is not None:
            h, w = img.shape[:2]
            return w, h
    except Exception:
        pass
    
    return 512, 512  # WHU数据集默认尺寸


def load_instance_annotations(anno_dir):
    """
    加载实例级注释文件
    
    Returns:
        anno_list: 按文件名排序的注释列表
        anno_dict: 文件名到注释的字典
    """
    anno_files = sorted([f for f in os.listdir(anno_dir) if f.endswith('.json')])
    print(f"Found {len(anno_files)} instance annotation files")
    
    anno_list = []
    anno_dict = {}
    
    for anno_file in tqdm(anno_files, desc="Loading annotations"):
        anno_path = os.path.join(anno_dir, anno_file)
        with open(anno_path, 'r', encoding='utf-8') as f:
            anno = json.load(f)
        anno['_anno_file'] = anno_file
        anno_list.append(anno)
        anno_dict[anno_file] = anno
    
    return anno_list, anno_dict


def convert_instance_predictions_to_coco(
    inference_file,
    anno_dir,
    image_dir,
    output_file,
    model_max=1000,
    gt_file=None
):
    """
    将实例级patch预测结果转换为512x512 patch级COCO格式
    
    Args:
        inference_file: 推理结果文件（jsonl格式）
        anno_dir: 实例级注释目录
        image_dir: 512x512 patch图像目录
        output_file: 输出COCO格式文件路径
        model_max: 模型输出坐标范围（默认1000）
        gt_file: 512x512 patch的GT文件（validation_patches_512.json），用于获取正确的image_id映射
    """
    
    # 1. 加载实例级注释
    print(f"\n1. Loading instance annotations from: {anno_dir}")
    anno_list, anno_dict = load_instance_annotations(anno_dir)
    
    # 2. 读取预测结果
    print(f"\n2. Reading predictions from: {inference_file}")
    predictions = []
    error_lines = []
    
    with open(inference_file, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if line:
                try:
                    data = json.loads(line)
                    if isinstance(data, dict) and "predict" in data:
                        inferences = parse_prediction_objects(data["predict"])
                        if not inferences and data["predict"]:
                            preview = str(data["predict"])[:200].replace('\n', ' ')
                            error_lines.append((idx + 1, f"Unable to parse predict field: {preview}"))
                        predictions.append(inferences)
                    else:
                        predictions.append([])
                except Exception as e:
                    error_lines.append((idx+1, str(e)))
                    predictions.append([])
    
    print(f"Loaded {len(predictions)} predictions")
    if error_lines:
        print(f"Parse errors in {len(error_lines)} lines")
    
    # 3. 检查数量匹配
    if len(predictions) != len(anno_list):
        print(f"ERROR: prediction count ({len(predictions)}) != annotation count ({len(anno_list)})")
        return
    
    # 4. 加载GT获取image_id映射（如果提供）
    gt_image_id_map = {}  # original_image -> image_id
    if gt_file and os.path.exists(gt_file):
        print(f"\n3. Loading GT for image_id mapping from: {gt_file}")
        with open(gt_file, 'r', encoding='utf-8') as f:
            gt_data = json.load(f)
        for img in gt_data.get('images', []):
            fname = os.path.basename(img['file_name'])
            gt_image_id_map[fname] = img['id']
        print(f"Built mapping for {len(gt_image_id_map)} images from GT")
    
    # 5. 按原始图像聚合预测结果
    print(f"\n4. Converting predictions with inverse transform...")
    
    # 存储按原始图像聚合的预测
    image_predictions = defaultdict(list)  # original_image -> [predictions]
    image_dimensions = {}  # original_image -> (w, h)
    
    valid_predictions = 0
    empty_predictions = 0
    
    for idx, (anno, pred_list) in enumerate(tqdm(zip(anno_list, predictions), 
                                                   total=len(anno_list),
                                                   desc="Processing")):
        original_image = anno.get('original_image', '')
        crop_info = anno.get('crop_info')
        
        if not original_image or not crop_info:
            continue
        
        # 记录图像尺寸（只需要一次）
        if original_image not in image_dimensions:
            img_path = os.path.join(image_dir, original_image)
            w, h = get_image_dimensions(img_path)
            image_dimensions[original_image] = (w, h)
        
        # 如果没有预测结果，跳过
        if not pred_list:
            empty_predictions += 1
            continue
        
        # 处理每个预测
        for obj in pred_list:
            if not isinstance(obj, dict):
                continue
            
            # 解析GeoJSON Feature格式
            if obj.get("type") == "Feature":
                geometry = obj.get("geometry", {})
                properties = obj.get("properties", {})
                
                if geometry.get("type") != "Polygon":
                    continue
                
                # coordinates 是平铺的 [x1, y1, x2, y2, ...]
                coords = geometry.get("coordinates", [])
                coord_pairs = safe_pair_coords(coords)
                
                if len(coord_pairs) < 3:
                    continue
                
                # Step 1: 从模型坐标(0-1000)转到裁剪图像坐标(0-256)
                output_size = crop_info['output_size']  # 256
                scale_to_crop = output_size / model_max  # 256/1000
                
                crop_coords = [(x * scale_to_crop, y * scale_to_crop) for x, y in coord_pairs]
                
                # Step 2: 逆向变换到原始图像坐标
                original_coords = inverse_transform_coords(crop_coords, crop_info)
                
                # 转为平铺格式
                segmentation_coords = []
                for x, y in original_coords:
                    segmentation_coords.append(x)
                    segmentation_coords.append(y)
                
                # 计算bbox和area
                bbox = polygon_to_bbox(original_coords)
                if bbox is None:
                    continue
                
                area = polygon_area(original_coords)
                score = properties.get("score", 1.0)
                
                pred_item = {
                    "segmentation": [segmentation_coords],
                    "bbox": bbox,
                    "area": area,
                    "score": score
                }
                
                image_predictions[original_image].append(pred_item)
                valid_predictions += 1
    
    print(f"\nProcessed {valid_predictions} valid predictions")
    print(f"Empty prediction instances: {empty_predictions}")
    print(f"Unique images with predictions: {len(image_predictions)}")
    
    # 6. 生成COCO格式输出
    print(f"\n5. Generating COCO format output...")
    
    coco_images = []
    coco_results = []
    annotation_id = 1
    
    # 处理所有有预测的图像
    processed_images = set()
    
    for original_image, preds in image_predictions.items():
        # 获取image_id
        if original_image in gt_image_id_map:
            image_id = gt_image_id_map[original_image]
        else:
            # 如果没有GT映射，使用文件名的hash作为image_id
            # 保留完整的patch信息（如20000_0），不要只取20000
            fname_stem = original_image.split('.')[0]  # "20000_0"
            # 生成唯一的image_id：基于完整文件名
            image_id = hash(fname_stem) % 10000000
            print(f"Warning: No GT mapping for {original_image}, using hash-based id: {image_id}")
        
        w, h = image_dimensions.get(original_image, (512, 512))
        
        # 添加图像信息
        if original_image not in processed_images:
            coco_images.append({
                "id": image_id,
                "file_name": original_image,
                "width": w,
                "height": h
            })
            processed_images.add(original_image)
        
        # 添加预测结果
        for pred in preds:
            coco_pred = {
                "id": annotation_id,
                "image_id": image_id,
                "category_id": 1,
                "segmentation": pred["segmentation"],
                "bbox": pred["bbox"],
                "area": pred["area"],
                "score": pred["score"],
                "iscrowd": 0
            }
            coco_results.append(coco_pred)
            annotation_id += 1
    
    # 7. 保存结果
    print(f"\n6. Saving results...")
    
    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
    
    # 保存预测结果（用于COCO评估API）- 不含id和iscrowd
    eval_results = []
    for r in coco_results:
        eval_results.append({
            "image_id": r["image_id"],
            "category_id": r["category_id"],
            "segmentation": r["segmentation"],
            "bbox": r["bbox"],
            "area": r["area"],
            "score": r["score"]
        })
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(eval_results, f, indent=2, ensure_ascii=False)
    print(f"Predictions saved to: {output_file}")
    
    # 保存完整COCO格式
    full_output_path = output_file.replace('.json', '_full.json')
    full_coco = {
        "images": coco_images,
        "annotations": coco_results,
        "categories": [{"id": 1, "name": "building"}]
    }
    
    with open(full_output_path, 'w', encoding='utf-8') as f:
        json.dump(full_coco, f, indent=2, ensure_ascii=False)
    print(f"Full COCO format saved to: {full_output_path}")
    
    # 打印统计信息
    print(f"\n=== Conversion Summary ===")
    print(f"Total images: {len(coco_images)}")
    print(f"Total predictions: {len(coco_results)}")
    print(f"Average predictions per image: {len(coco_results) / max(len(coco_images), 1):.2f}")
    
    return coco_images, coco_results


def main():
    parser = argparse.ArgumentParser(
        description="Convert instance-level building predictions (JSONL) to patch-level COCO results."
    )
    parser.add_argument("--inference-file", required=True, help="Input model predictions in JSONL format")
    parser.add_argument("--anno-dir", required=True, help="Instance-crop annotation directory")
    parser.add_argument("--image-dir", required=True, help="Patch image directory")
    parser.add_argument("--gt-file", required=True, help="Patch-level COCO ground-truth JSON")
    parser.add_argument("--output-file", required=True, help="Output COCO detection JSON")
    parser.add_argument("--model-max", type=float, default=1000, help="Maximum model coordinate value")
    args = parser.parse_args()

    Path(args.output_file).parent.mkdir(parents=True, exist_ok=True)
    convert_instance_predictions_to_coco(
        inference_file=args.inference_file,
        anno_dir=args.anno_dir,
        image_dir=args.image_dir,
        output_file=args.output_file,
        model_max=args.model_max,
        gt_file=args.gt_file,
    )
    print("\nConversion completed!")


if __name__ == "__main__":
    main()
