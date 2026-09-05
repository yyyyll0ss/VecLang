"""
实例级patch预测结果转换为512x512 patch级COCO格式

功能：
- 读取实例级patch的预测结果（jsonl格式）
- 使用实例级注释中的crop_info进行逆向坐标转换
- 按512x512 patch（original_image）聚合预测结果
- 生成与validation_patches_512.json对应的COCO格式输出
"""

import os
import json
import cv2
import re
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict

from shapely.affinity import translate
from shapely.geometry import GeometryCollection, MultiPolygon, Polygon, box
from shapely.geometry.polygon import orient


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


def normalize_polygon_coord_pairs(coords):
    """Normalize common model/GeoJSON polygon coordinate formats into point pairs."""
    if not isinstance(coords, list) or not coords:
        return []

    if all(isinstance(v, (int, float)) for v in coords):
        return safe_pair_coords(coords)

    if all(
        isinstance(pt, (list, tuple))
        and len(pt) >= 2
        and isinstance(pt[0], (int, float))
        and isinstance(pt[1], (int, float))
        for pt in coords
    ):
        return [(float(pt[0]), float(pt[1])) for pt in coords]

    first = coords[0]
    if isinstance(first, list):
        if first and all(isinstance(v, (int, float)) for v in first):
            return safe_pair_coords(first)
        if all(
            isinstance(pt, (list, tuple))
            and len(pt) >= 2
            and isinstance(pt[0], (int, float))
            and isinstance(pt[1], (int, float))
            for pt in first
        ):
            return [(float(pt[0]), float(pt[1])) for pt in first]

    return []
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

PATCH_NAME_PATTERN = re.compile(
    r"^(?P<source>.+)_x(?P<x>-?\d+)_y(?P<y>-?\d+)(?P<ext>\.[^.]+)$"
)


def to_int_or_none(value):
    if value is None:
        return None
    try:
        return int(round(float(value)))
    except Exception:
        return None


def build_gt_image_maps(gt_data):
    exact_image_info_map = {}
    source_patch_index = defaultdict(list)
    categories = gt_data.get("categories") or [{"id": 1, "name": "water"}]

    for img in gt_data.get("images", []):
        file_name = os.path.basename(img.get("file_name", ""))
        if not file_name:
            continue

        image_info = {
            "id": img["id"],
            "file_name": file_name,
            "width": int(img.get("width", 512)),
            "height": int(img.get("height", 512)),
        }
        exact_image_info_map[file_name] = image_info

        source_image = img.get("source_image")
        patch_x = to_int_or_none(img.get("patch_x"))
        patch_y = to_int_or_none(img.get("patch_y"))

        if source_image is None or patch_x is None or patch_y is None:
            match = PATCH_NAME_PATTERN.match(file_name)
            if match:
                source_image = "{}{}".format(match.group("source"), match.group("ext"))
                patch_x = int(match.group("x"))
                patch_y = int(match.group("y"))

        if source_image is None or patch_x is None or patch_y is None:
            continue

        patch_info = dict(image_info)
        patch_info.update({
            "patch_x": patch_x,
            "patch_y": patch_y,
            "source_width": to_int_or_none(img.get("source_width")),
            "source_height": to_int_or_none(img.get("source_height")),
        })
        source_patch_index[os.path.basename(source_image)].append(patch_info)

    for patches in source_patch_index.values():
        patches.sort(key=lambda item: (item["patch_y"], item["patch_x"], item["file_name"]))

    return exact_image_info_map, source_patch_index, categories


def resolve_target_category_id(categories, preferred_name="water"):
    aliases = {"water", "waterbody", "water_body", "waterbodies"}
    for category in categories:
        name = str(category.get("name", "")).lower()
        if name == preferred_name.lower() or name in aliases:
            return int(category["id"])
    if categories:
        return int(categories[0]["id"])
    return 1


def fix_geometry(geometry):
    if geometry is None or geometry.is_empty:
        return None
    if geometry.is_valid:
        return geometry
    fixed = geometry.buffer(0)
    if fixed.is_empty:
        return None
    return fixed


def pred_to_polygon(pred):
    segmentation = pred.get("segmentation") or []
    if not segmentation:
        return None
    points = safe_pair_coords(segmentation[0])
    if len(points) < 3:
        return None
    try:
        return fix_geometry(Polygon(points))
    except Exception:
        return None


def iter_polygons(geometry):
    if geometry is None or geometry.is_empty:
        return
    if isinstance(geometry, Polygon):
        yield geometry
        return
    if isinstance(geometry, MultiPolygon):
        for polygon in geometry.geoms:
            yield polygon
        return
    if isinstance(geometry, GeometryCollection) or hasattr(geometry, "geoms"):
        for sub_geometry in geometry.geoms:
            yield from iter_polygons(sub_geometry)


def clamp(value, lower, upper):
    return min(max(value, lower), upper)


def polygon_to_coco_flat(polygon, width, height, decimals=2):
    polygon = orient(polygon, sign=1.0)
    coords = list(polygon.exterior.coords)
    if len(coords) >= 2 and coords[0] == coords[-1]:
        coords = coords[:-1]

    ring = []
    previous = None
    for x, y in coords:
        x = round(clamp(float(x), 0.0, float(width)), decimals)
        y = round(clamp(float(y), 0.0, float(height)), decimals)
        point = (x, y)
        if point != previous:
            ring.append(point)
            previous = point

    if len(ring) >= 2 and ring[0] == ring[-1]:
        ring = ring[:-1]
    if len(set(ring)) < 3:
        return None

    flat = []
    for x, y in ring:
        flat.extend([x, y])
    return flat


def bbox_from_bounds(bounds, width, height, decimals=2):
    minx, miny, maxx, maxy = bounds
    minx = clamp(float(minx), 0.0, float(width))
    miny = clamp(float(miny), 0.0, float(height))
    maxx = clamp(float(maxx), 0.0, float(width))
    maxy = clamp(float(maxy), 0.0, float(height))
    return [
        round(minx, decimals),
        round(miny, decimals),
        round(maxx - minx, decimals),
        round(maxy - miny, decimals),
    ]


def clip_prediction_to_gt_patches(pred, patch_infos, min_area=1.0, decimals=2):
    source_polygon = pred_to_polygon(pred)
    if source_polygon is None:
        return []

    minx, miny, maxx, maxy = source_polygon.bounds
    clipped_predictions = []

    for patch_info in patch_infos:
        patch_x = patch_info["patch_x"]
        patch_y = patch_info["patch_y"]
        patch_w = patch_info["width"]
        patch_h = patch_info["height"]
        source_w = patch_info.get("source_width")
        source_h = patch_info.get("source_height")
        patch_x1 = patch_x + patch_w if source_w is None else min(patch_x + patch_w, source_w)
        patch_y1 = patch_y + patch_h if source_h is None else min(patch_y + patch_h, source_h)

        if maxx <= patch_x or maxy <= patch_y or minx >= patch_x1 or miny >= patch_y1:
            continue

        try:
            clipped = source_polygon.intersection(
                box(float(patch_x), float(patch_y), float(patch_x1), float(patch_y1))
            )
        except Exception:
            continue

        clipped = fix_geometry(clipped)
        if clipped is None or clipped.area < min_area:
            continue

        shifted = fix_geometry(translate(clipped, xoff=-float(patch_x), yoff=-float(patch_y)))
        if shifted is None or shifted.area < min_area:
            continue

        segmentations = []
        total_area = 0.0
        for polygon in iter_polygons(shifted):
            if polygon.area < min_area:
                continue
            flat = polygon_to_coco_flat(polygon, patch_w, patch_h, decimals=decimals)
            if flat:
                segmentations.append(flat)
                total_area += float(polygon.area)

        if not segmentations or total_area < min_area:
            continue

        bbox = bbox_from_bounds(shifted.bounds, patch_w, patch_h, decimals=decimals)
        if bbox[2] <= 0 or bbox[3] <= 0:
            continue

        clipped_predictions.append({
            "image_id": patch_info["id"],
            "file_name": patch_info["file_name"],
            "width": patch_w,
            "height": patch_h,
            "segmentation": segmentations,
            "bbox": bbox,
            "area": round(total_area, decimals),
            "score": pred["score"],
        })

    return clipped_predictions


def stable_image_id(file_name):
    value = 0
    for ch in file_name:
        value = (value * 131 + ord(ch)) % 10000000
    return value


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
    gt_file=None,
    category_id=None,
    manifest=None
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
    if model_max <= 0:
        raise ValueError("model_max must be positive")
    if not gt_file or not os.path.isfile(gt_file):
        raise FileNotFoundError("A valid GT COCO file is required for evaluation image/category mapping")
    with open(gt_file, encoding="utf-8") as f:
        categories = json.load(f)["categories"]
    aliases = {'waterbodies', 'waterbody', 'water'}
    matches = [c["id"] for c in categories if str(c["name"]).lower().replace("_", "") in aliases]
    if category_id is None:
        if len(matches) != 1:
            raise ValueError("Cannot uniquely resolve target category; pass --category-id")
        category_id = matches[0]
    if category_id not in [c["id"] for c in categories]:
        raise ValueError("category_id does not exist in GT")
    if manifest:
        with open(manifest, encoding="utf-8") as f:
            records = json.load(f)
        names = [Path(r["images"][0]).stem + ".json" for r in records]
        if len(names) != len(set(names)):
            raise ValueError("Duplicate crop names in manifest")
        anno_list = [anno_dict[n] for n in names]

    
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
        raise ValueError(f"prediction count ({len(predictions)}) != annotation count ({len(anno_list)})")
    
    # 4. 加载GT获取image_id映射（如果提供）
    gt_image_info_map = {}  # file_name -> image info
    gt_source_patch_index = defaultdict(list)  # source image -> patch image infos
    gt_categories = [{"id": 1, "name": "water"}]
    target_category_id = 1
    if gt_file and os.path.exists(gt_file):
        print(f"\n3. Loading GT for image_id mapping from: {gt_file}")
        with open(gt_file, 'r', encoding='utf-8') as f:
            gt_data = json.load(f)
        gt_image_info_map, gt_source_patch_index, gt_categories = build_gt_image_maps(gt_data)
        target_category_id = category_id
        patch_count = sum(len(items) for items in gt_source_patch_index.values())
        print(f"Built exact mapping for {len(gt_image_info_map)} images from GT")
        print(f"Built source-to-patch mapping for {len(gt_source_patch_index)} source images / {patch_count} patches")
        print(f"Using target category_id={target_category_id} for water predictions")
    
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
            raise ValueError(f'Missing original_image/crop_info in {anno["_anno_file"]}')
        
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
                
                # coordinates may be flat, point-list, or standard GeoJSON ring format
                coords = geometry.get("coordinates", [])
                coord_pairs = normalize_polygon_coord_pairs(coords)
                
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
    direct_mapped_images = 0
    patch_mapped_images = 0
    unmatched_images = 0
    clipped_patch_predictions = 0
    
    for original_image, preds in image_predictions.items():
        original_key = os.path.basename(original_image)

        if original_key in gt_image_info_map:
            direct_mapped_images += 1
            image_info = gt_image_info_map[original_key]
            image_id = image_info["id"]
            file_name = image_info["file_name"]

            if file_name not in processed_images:
                coco_images.append({
                    "id": image_id,
                    "file_name": file_name,
                    "width": image_info["width"],
                    "height": image_info["height"]
                })
                processed_images.add(file_name)

            for pred in preds:
                coco_results.append({
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": target_category_id,
                    "segmentation": pred["segmentation"],
                    "bbox": pred["bbox"],
                    "area": pred["area"],
                    "score": pred["score"],
                    "iscrowd": 0
                })
                annotation_id += 1
            continue

        if original_key in gt_source_patch_index:
            patch_mapped_images += 1
            patch_infos = gt_source_patch_index[original_key]

            for pred in preds:
                clipped_items = clip_prediction_to_gt_patches(pred, patch_infos)
                for item in clipped_items:
                    file_name = item["file_name"]
                    if file_name not in processed_images:
                        coco_images.append({
                            "id": item["image_id"],
                            "file_name": file_name,
                            "width": item["width"],
                            "height": item["height"]
                        })
                        processed_images.add(file_name)

                    coco_results.append({
                        "id": annotation_id,
                        "image_id": item["image_id"],
                        "category_id": target_category_id,
                        "segmentation": item["segmentation"],
                        "bbox": item["bbox"],
                        "area": item["area"],
                        "score": item["score"],
                        "iscrowd": 0
                    })
                    annotation_id += 1
                    clipped_patch_predictions += 1
            continue

        unmatched_images += 1
        if gt_file and os.path.exists(gt_file):
            raise ValueError(f"No GT exact/patch mapping for {original_image}")
            continue

        image_id = stable_image_id(original_key)
        w, h = image_dimensions.get(original_image, (512, 512))
        print(f"Warning: No GT mapping for {original_image}, using stable fallback id: {image_id}")

        if original_key not in processed_images:
            coco_images.append({
                "id": image_id,
                "file_name": original_key,
                "width": w,
                "height": h
            })
            processed_images.add(original_key)

        for pred in preds:
            coco_results.append({
                "id": annotation_id,
                "image_id": image_id,
                "category_id": target_category_id,
                "segmentation": pred["segmentation"],
                "bbox": pred["bbox"],
                "area": pred["area"],
                "score": pred["score"],
                "iscrowd": 0
            })
            annotation_id += 1
        continue

    # 7. 保存结果
    print(f"Direct GT-mapped images: {direct_mapped_images}")
    print(f"Source-to-patch mapped images: {patch_mapped_images}")
    print(f"Patch-clipped predictions: {clipped_patch_predictions}")
    print(f"Unmatched images: {unmatched_images}")
    
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
    
    coco_images = gt_data["images"]

    # 保存完整COCO格式
    full_output_path = output_file.replace('.json', '_full.json')
    full_coco = {
        "images": coco_images,
        "annotations": coco_results,
        "categories": gt_categories
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
    import argparse
    parser = argparse.ArgumentParser(description="Convert IRSAMap instance crops to GT-aligned COCO patches.")
    for flag in ("inference-file", "anno-dir", "image-dir", "gt-file", "output-file"):
        parser.add_argument("--" + flag, required=True)
    parser.add_argument("--model-max", type=float, default=1000)
    parser.add_argument("--category-id", type=int, default=None)
    parser.add_argument("--manifest", help="Ordered SFT JSON manifest; otherwise annotations use sorted filenames")
    args = parser.parse_args()
    convert_instance_predictions_to_coco(**vars(args))

if __name__ == "__main__":
    main()
