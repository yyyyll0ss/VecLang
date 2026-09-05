# 水体评估

## 运行顺序

1. `1.convert_to_coco_format_geojson_instance_patch.py`
2. `2.eval_waterbody.py`
3. `3.visualize_inference_results.py`（可选）

## 1. 转换预测结果

将实例裁剪级 JSONL 预测还原到 patch 坐标，输出 COCO detection list 和 `*_full.json`。

```bash
python 1.convert_to_coco_format_geojson_instance_patch.py \
  --inference-file path/to/predictions.jsonl \
  --anno-dir path/to/instance_annotations \
  --image-dir path/to/patch_images \
  --gt-file path/to/gt.json \
  --output-file outputs/waterbody_predictions.json
```

模型坐标范围不是 0–1000 时，通过 `--model-max` 修改。

## 2. 计算指标

计算 COCO bbox/segm mAP，以及 IoU、C-IoU 和 PoLiS。`--dt` 使用第 1 步的 detection list。

```bash
python 2.eval_waterbody.py \
  --gt path/to/gt.json \
  --dt outputs/waterbody_predictions.json \
  --output outputs/waterbody_metrics.json
```

常用选项与建筑评估一致：`--iou`、`--max-dets`、`--polis-match-iou`、`--skip-map`、`--skip-polygon`。

## 3. 可视化

```bash
python 3.visualize_inference_results.py \
  --dt outputs/waterbody_predictions.json \
  --gt path/to/gt.json \
  --image_dir path/to/patch_images \
  --out_dir outputs/waterbody_visualization \
  --category_ids 1 \
  --max_images 100
```

`--category_ids all` 绘制所有类别，`--max_images 0` 处理全部影像。
