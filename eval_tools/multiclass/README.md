# IRSAMap 多类别评测

本目录提供完整的建筑、水体、道路评测和区域级联合可视化。所有运行所需的 Python 和 Go 源码均保存在本目录，建筑、水体的相同实现也分别保留，可单独复制本目录使用，无需相邻的 `../building`、`../waterbody` 或 `../road`。

## 安装

Python 3.9+，在本目录执行：

```bash
python -m pip install -r requirements.txt
```

道路 APLS 另需 Go 1.21+。首次执行会依据本地 `road/cityscale_metrics/apls_rtreego_fast/go.mod`、`go.sum` 下载依赖并编译。TOPO 不需要 Go。输入数据、推理结果和编译产物不随代码发布。

## 代码结构

| 目录/脚本 | 内容 |
| --- | --- |
| `building/1…5*.py` | 实例坐标还原与跨 patch 裁剪、COCO/多边形指标、可视化、IoU 诊断、逐场景评测 |
| `waterbody/1…5*.py` | 水体的完整独立实现，按 GT 类别映射 |
| `road/0.convert_irsamap_road_geojson_to_graph_gt.py` | IRSAMap GeoJSON 道路 GT 转图 |
| `road/1.extract_road_instance_patch.py` | 模型预测转 patch COCO 与 junction 文件 |
| `road/2.stitch_coco_polylines.py` | 与整理好的单类别版本对应的完整拼接流程 |
| `road/2.stitch_coco_polylines_junction.py` | 原多类别版本的完整 junction/consensus 增强拼接流程 |
| `road/3.run_irsamap_metrics.py` | 自动发现 IRSAMap 区域并运行 TOPO/APLS |
| `road/cityscale_metrics/` | 完整本地 TOPO 和 Go APLS 后端；目录名保留其评测协议来源 |
| `visualize_multiclass_inference_results.py` | 建筑、水体、拼接道路的联合展示 |
| `tests/test_pipeline.py` | 无数据集依赖的回归测试 |

## 数据约定

- 模型 JSONL 每条记录含 `predict`，其内容为 SVL/GeoJSON Feature。无有效几何的模型输出记为空预测，保留顺序并输出解析错误统计，不替换为 GT。
- 建筑、水体的实例 annotation JSON 含 `original_image` 和 `crop_info`（`crop_x/crop_y/crop_w/crop_h/output_size`）。预测坐标默认 0–1000，经实例 crop 逆变换到源图，再根据 GT 元数据裁剪到评测 patch。
- GT COCO 含 `images/annotations/categories`。IRSAMap patch 元数据建议明确提供 `source_image/patch_x/patch_y`；亦支持 `100_x0000_y0000.png` 命名。转换产物中的 `image_id`、`category_id` 与 GT 对齐，`*_full.json` 保留全部 GT 图像，包括无预测图像。
- 建筑、水体类别默认按 GT 名称解析，水体支持 `water/waterbody/water_body/waterbodies`。有歧义时传 `--category-id`，不要假定所有数据版本都用建筑 1、水体 2。
- **顺序必须与推理一致**：推荐传入生成推理数据时的 `--manifest`（SFT JSON 数组，每项含 `images`）。建筑/水体按图片 stem 匹配实例 annotation JSON；不传时按 annotation 文件名字典序。道路不传时按 GT `images` 数组顺序。数量不符会报错；仅数量相同不能证明顺序正确。
- 道路必须使用真实推理 patch 索引，文件名支持 `region_100_patch_0_0_0.png` 或 `100_patch_0_0_0.png`。末两项是源图偏移；默认原始 crop 128、模型 patch 256、stride 64。不能用建筑/水体的 512 patch GT 替代道路索引。
- 道路图采用 `(row, col)` 像素坐标；GT GeoJSON 输入为 `(x, y)` 像素坐标，不是经纬度。GT 转换默认将顶点限制在 `[0,1024]` 并取整，保持原实现；其他尺寸应指定 `--clip-max` 或 `--no-clip`。

## 1. 建筑、水体

以下命令从本目录运行，路径均为示例，替换为自己的真实数据路径。

```bash
python building/1.convert_to_coco_format_instance_patch.py \
  --inference-file inputs/building_predictions.jsonl \
  --anno-dir inputs/building_crop_annotations \
  --manifest inputs/building_test_sft.json \
  --image-dir inputs/source_images \
  --gt-file inputs/test_patches_512_coco.json \
  --output-file outputs/building.json

python building/2.eval_building.py \
  --gt inputs/test_patches_512_coco.json --dt outputs/building.json \
  --iou segm --output outputs/building_metrics.json

python waterbody/1.convert_to_coco_format_instance_patch.py \
  --inference-file inputs/water_predictions.jsonl \
  --anno-dir inputs/water_crop_annotations \
  --manifest inputs/water_test_sft.json \
  --image-dir inputs/source_images \
  --gt-file inputs/test_patches_512_coco.json \
  --output-file outputs/water.json

python waterbody/2.eval_waterbody.py \
  --gt inputs/test_patches_512_coco.json --dt outputs/water.json \
  --iou segm --output outputs/water_metrics.json
```

历史单类别预测若以 `category_id=1` 存储水体，评测时使用 `--category water --dt-category-id 1`；转换后的正确多类别 ID 无需重映射。建筑/水体各自包含 `3.visualize_inference_results.py`、`4.analysis_vis.py`、`5.eval_by_scene.py`，具体命令见子目录 README。

## 2. 道路

```bash
python road/0.convert_irsamap_road_geojson_to_graph_gt.py \
  --geojson-dir inputs/roadline_geojson --output-root outputs/road_gt

python road/1.extract_road_instance_patch.py \
  --inference-file inputs/road_predictions.jsonl \
  --gt-file inputs/road_patch_index.json \
  --manifest inputs/road_test_sft.json --output-file outputs/road.json

python road/2.stitch_coco_polylines.py \
  --input_json outputs/road_full.json --output_dir outputs/road_stitched \
  --crop_size_orig 128 --patch_size_model 256 --stride 64 --enable_viz 0

python road/3.run_irsamap_metrics.py \
  --savedir outputs/road_stitched --gt-root outputs/road_gt --only all
```

增强拼接是第二步的另一种配置，不要在已拼接图上再执行它：

```bash
python road/2.stitch_coco_polylines_junction.py \
  --input_json outputs/road_full.json --junction_json outputs/road_junctions.json \
  --output_dir outputs/road_junction_stitched --preset balanced --enable_viz 0
```

`balanced/precision/recall` 会应用原实现预设；如需逐项自定义阈值，用 `--preset custom`。两种算法和参数会影响分数，应记录实际命令并在全部测试区域统一使用，不根据测试得分逐样本选择。完整参数可用 `--help` 查看。

输出图为 `graph/<region>.p`；指标为 `score/topo.json`、`score/apls.json`。默认从 GT 目录发现所有 `region_<id>_graph_gt.pickle`，不使用 CityScale 固定测试编号。可用 `--tiles 100,102,308` 明确指定子集；缺失预测/GT 会报错，不能把缺失文件当成已评估。无道路预测的区域应保留空图。只运行 TOPO 可传 `--only topo`。

## 3. 联合展示

推荐读取与评测完全相同的最终道路图：

```bash
python visualize_multiclass_inference_results.py \
  --building_dt outputs/building.json --water_dt outputs/water.json \
  --road_graph_dir outputs/road_stitched/graph \
  --patch_coco inputs/test_patches_512_coco.json \
  --region_image_dir inputs/source_images --out_dir outputs/multiclass_vis --no_gt
```

默认可视化类别为建筑 1、水体 2；必要时调整 `--building_category_ids`、`--water_category_ids`。需要 GT 对照时移除 `--no_gt`，传 `--gt` 和 `--gt_roadline_dir`（或 `--no_gt_roadline`）。旧 `--road_dt` 入口保留，其绘图内的简化拼接不等同于完整评测流水线；正式展示应使用 `--road_graph_dir`。图 pickle 应由本流程生成。

## 指标口径与验证

建筑/水体使用原实现的 COCO mAP/mAR、逐图二值掩膜 IoU、C-IoU、PS 和 PoLiS。IoU 在 GT 的全部图像上取平均；C-IoU = IoU × PS，PS 衡量顶点数量差异。PoLiS 使用 bbox IoU 阈值匹配，属于匹配实例的几何误差，并非漏检惩罚；无匹配时不可解释为零误差。原多边形转换/PoLiS 以外环为主，不是带孔洞或所有 MultiPolygon 分量的完整 GIS 评测。空预测仍参与 COCO 和掩膜评测；COCO 不适用的面积档或无 GT 类别指标保留 `-1`。

逐场景脚本按 `source_image` 或 patch 文件名还原完整场景 ID，不使用前两位截断；保留无预测场景，输出每个场景指标。逐场景指标与全数据集 COCO AP 不可互换。诊断图中的颜色基于 mask IoU，不能代替正式 COCO 一对一匹配指标。

```bash
python -m unittest discover -s tests -v
```

已验证：类别映射、跨 patch 裁剪、完美/空预测 COCO、空预测 IoU、道路提取和两种拼接、最终图读取、GT 坐标转换、区域自动发现和缺失输入报错。TOPO 对相同测试道路图的 Precision/Recall/F1 为 1。当前整理环境未安装 Go，APLS 已保留完整源码但尚未完成编译运行验证；未重新运行整套 IRSAMap 基准。

源码来源与整理范围见 [SOURCES.md](SOURCES.md)。
