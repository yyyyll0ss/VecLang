# 建筑评估

## 运行顺序

1. `1.convert_to_coco_format_geojson_instance_patch.py`
2. `2.eval_building.py`
3. `3.visualize_inference_results.py`（可选）

## 1. 转换预测结果

将实例裁剪级 JSONL 预测还原到 patch 坐标，并输出标准 COCO detection list 和带完整元信息的 `*_full.json`。`--anno-dir` 中的实例标注用于逆变换坐标，`--gt-file` 用于保持预测与 GT 的 `image_id` 一致。

若预测来自 `inference/inference_cut`，分别使用
`<run>/attribute_inference/generated_predictions.jsonl` 和
`<run>/instance_crops/annotations/`；`--image-dir` 指向原始检测 patch 影像，不是
裁出的 256×256 实例图。完整路径对应表见[评估总览](../README.md#数据约定)。
这里的 `--inference-file` 是第二阶段建筑矢量属性 JSONL，不是第一阶段 bbox 检测
JSONL。

```bash
python 1.convert_to_coco_format_geojson_instance_patch.py \
  --inference-file path/to/predictions.jsonl \
  --anno-dir path/to/instance_annotations \
  --image-dir path/to/patch_images \
  --gt-file path/to/gt.json \
  --output-file outputs/building_predictions.json
```

模型坐标范围不是 0–1000 时，通过 `--model-max` 修改。

## 2. 计算指标

计算 COCO bbox/segm mAP，以及逐实例 IoU、C-IoU 和 PoLiS。检测文件应使用第 1 步生成的 detection list，而不是 `*_full.json`。

```bash
python 2.eval_building.py \
  --gt path/to/gt.json \
  --dt outputs/building_predictions.json \
  --output outputs/building_metrics.json
```

常用选项：`--iou bbox|segm`、`--max-dets 1 10 100`、`--polis-match-iou 0.5`、`--skip-map`、`--skip-polygon`。

## 3. 可视化

在 patch 影像上绘制建筑预测；若提供 `--gt`，同时生成 GT 对照。

```bash
python 3.visualize_inference_results.py \
  --dt outputs/building_predictions.json \
  --gt path/to/gt.json \
  --image_dir path/to/patch_images \
  --out_dir outputs/building_visualization \
  --max_images 100
```

`--max_images 0` 表示处理全部影像。
