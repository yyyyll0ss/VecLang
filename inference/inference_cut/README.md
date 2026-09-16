# 检测结果裁剪与属性推理

本目录补充 VecLang 渐进式推理中原先缺失的一步：目标检测完成后，先按照检测框
裁出单实例图像，再将这些实例图像送入矢量属性推理。

原有的 `../run_inference.py`、`../run_inference.sh` 和任务配置没有改动。新增流程
读取原推理脚本产生的 `predictions_by_dataset/*.jsonl`，因此可以与现有代码直接
衔接，也可以单独使用。

整个项目的执行顺序以 [`../../REPRODUCE.md`](../../REPRODUCE.md) 为准；本文只
维护裁剪阶段的 profile、参数和输出细节。除“支持的配置”小节外，下面命令均从
仓库根目录执行。

## 文件说明

- `run_pipeline.sh`：推荐入口；
- `run_pipeline.py`：串联 COCO 转换、实例裁剪、属性数据构造和属性推理；
- `detection_to_coco.py`：按照原检测 manifest 的顺序将 JSONL 转为 COCO；
- `crop_instances.py`：按检测框放大 1.3 倍并裁为 256×256 单实例图像；
- `build_attribute_manifest.py`：生成 LLaMA-Factory 可读取的属性推理 manifest；
- `configs/profiles.yaml`：数据集路径、类别 ID、类别别名和裁剪参数。

所有代码文件、目录和运行结果名称均使用英文。脚本使用 Pillow，不要求额外安装
OpenCV。运行环境和 LLaMA-Factory 要求与上级推理目录相同。

## 支持的配置

```bash
bash run_pipeline.sh --list-profiles
```

当前包含：

- `whu_building`；
- `wb_waterbody`；
- `irsamap_building`；
- `irsamap_waterbody`。

若数据不在默认的 `dataset/test`，使用 `--dataset-root /path/to/dataset/test`。

## 完整运行顺序

以下命令均从仓库根目录执行。

### 1. 运行目标检测

```bash
bash inference/run_inference.sh \
  --task object_detection \
  --datasets WHU_test_patches_512 \
  --run-name whu_detection \
  --gpus 0,1,2,3
```

检测结果位于：

```text
../VecLang_outputs/whu_detection/object_detection/
└── predictions_by_dataset/WHU_test_patches_512.jsonl
```

### 2. 裁剪实例并运行属性推理

```bash
bash inference/inference_cut/run_pipeline.sh \
  --profile whu_building \
  --detection-predictions \
    ../VecLang_outputs/whu_detection/object_detection/predictions_by_dataset/WHU_test_patches_512.jsonl \
  --model-path /path/to/VecLang-4B \
  --gpus 0,1,2,3
```

也可以通过环境变量指定模型和 LLaMA-Factory：

```bash
export VECLANG_MODEL_PATH=/path/to/VecLang-4B
export VECLANG_PYTHON=/path/to/environment/bin/python
export LLAMAFACTORY_CLI=/path/to/environment/bin/llamafactory-cli
```

## 只检查裁剪结果

`--prepare-only` 不启动第二阶段模型，适合先确认检测转换和裁剪结果：

```bash
bash inference/inference_cut/run_pipeline.sh \
  --profile whu_building \
  --detection-predictions /path/to/WHU_test_patches_512.jsonl \
  --max-instances 10 \
  --prepare-only
```

`--max-instances 0` 表示处理全部检测实例。该参数只限制裁剪数量；正式推理应保持
默认值 0。

## 输出目录

默认输出到仓库同级的 `VecLang_outputs/<时间戳>_<profile>/`：

```text
<run>/
├── detections_coco.json
├── detections_coco_full.json
├── instance_crops/
│   ├── images/
│   ├── annotations/
│   ├── filtered_small/
│   ├── crop_summary.json
│   └── attribute_manifest.json
├── dataset_registry/dataset_info.json
├── attribute_config.yaml
└── attribute_inference/generated_predictions.jsonl
```

这些结果不应上传 GitHub。可使用 `--output-dir` 指定其他输出位置。

动态 manifest 中的 assistant 字段只是 LLaMA-Factory `do_predict` 所需的格式占位，
由检测框生成，不是真实矢量属性标注，也不会作为用户提示输入模型。第二阶段应以
`attribute_inference/generated_predictions.jsonl` 中的 `predict` 字段作为正式结果。

## IRSAMap 多类别说明

IRSAMap 的同一份目标检测 JSONL 同时包含建筑和水体。分别使用
`irsamap_building` 与 `irsamap_waterbody` 运行两次即可；两次运行会按类别过滤，
不会互相混合。
