# 道路评估

## 运行顺序

1. `1.extract_road_instance_patch.py`：预测 JSONL → patch 级 COCO。
2. `2.stitch_coco_polylines.py`：将重叠 patch 道路拼成区域级图。
3. `3.run_cityscale_metrics.py`：计算 TOPO 和 APLS。

`cityscale_metrics/` 是第 3 步所需的内部实现，不需要单独调用。

## 1. 转换道路预测

```bash
python 1.extract_road_instance_patch.py \
  --inference-file path/to/predictions.jsonl \
  --gt-file path/to/patch_gt.json \
  --output-file outputs/road_predictions.json
```

默认生成：

- `road_predictions.json`：COCO detection list；
- `road_predictions_full.json`：包含 `images` 和 `categories`，供第 2 步使用；
- `road_predictions_junctions.json`：patch 级交点信息。

可用 `--no-full-output` 或 `--no-junction-output` 关闭附加输出。模型坐标范围通过 `--model-max` 设置。

## 2. 拼接区域级道路图

输入影像文件名必须符合 `region_<region>_patch_<patch>_<x0>_<y0>.png`，脚本据此恢复 patch 的区域位置。

```bash
python 2.stitch_coco_polylines.py \
  --input_json outputs/road_predictions_full.json \
  --output_dir outputs/stitched_roads \
  --crop_size_orig 128 \
  --patch_size_model 256 \
  --stride 64
```

输出图保存为 `outputs/stitched_roads/graph/<region>.p`。默认不生成可视化；如需检查拼接效果，增加 `--enable_viz 1 --patch_image_dir path/to/patch_images`（也可提供 `--image_root` 作为区域影像回退目录）。

拼接阈值均已开放为参数，建议先保持默认值复现实验，再调整 `--merge_node_dist`、`--endpoint_snap_dist`、`--patch_boundary_bridge_dist` 等选项。

## 3. 计算 CityScale TOPO/APLS

GT 目录应包含 `region_<tile>_graph_gt.pickle`，`--savedir` 指向第 2 步输出目录。

```bash
python 3.run_cityscale_metrics.py \
  --savedir outputs/stitched_roads \
  --gt-root path/to/cityscale_graph_gt \
  --only all \
  --workers 4 \
  --topo-workers 4
```

结果写入 `<savedir>/results/` 和 `<savedir>/score/`。`--tiles 8,9,19` 可只评估指定区域，`--only topo` 或 `--only apls` 可单独运行指标。APLS 需要 Go 1.21+；Go 不在 `PATH` 时通过 `--go-bin /path/to/go` 指定。

TOPO 额外依赖 `rtree` 和 `hopcroftkarp`。部分 Linux 环境安装 `rtree` 前需要系统包 `libspatialindex`。
