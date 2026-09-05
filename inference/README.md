# VecLang 推理代码

本目录只保存可上传 GitHub 的推理入口、任务配置和复现说明。测试数据、模型权重、
完整预测结果以及运行缓存均不放在这里。

## 文件说明

- `run_inference.sh`：推荐入口，选择 Python 环境并自动发现本地模型；
- `run_inference.py`：数据预检、配置生成、LLaMA-Factory 启动和结果拆分；
- `configs/datasets.yaml`：测试集名称、任务类型、样本数和相对路径；
- `configs/object_detection.yaml`：目标检测生成参数；
- `configs/attributes_generation.yaml`：矢量属性生成参数；
- `SMOKE_TEST.md`：六卡小样例推理验证记录。

脚本不会改写 `../dataset/test`。每次运行会创建独立目录，保存解析后的配置、
LLaMA-Factory 原始输出、按数据集拆分的 JSONL 和运行摘要。

## LLaMA-Factory 要求

VecLang 不包含 LLaMA-Factory 本体；`run_inference.py` 是对其 `do_predict`
流程的封装，正式推理前必须能够调用 `llamafactory-cli`。

- 已实测兼容版本：`LLaMA-Factory 0.9.4.dev0`；
- 需要支持 `Qwen3-VL`、`qwen3_vl_nothink` 模板和多模态 `media_dir`；
- 建议使用本项目测试时对应的 LLaMA-Factory 源码版本。其他版本可能存在模板名、
  YAML 参数或预测输出文件名差异，尚未验证；
- `python`、`llamafactory-cli` 和多卡启动所用的 `torchrun` 必须来自同一个
  Conda/虚拟环境，否则分布式子进程可能加载到另一套 PyTorch 或依赖；
- 完整的 Python、PyTorch、Transformers、CUDA 及其他依赖版本见
  `../ENVIRONMENT.md`。

若当前工作区已经包含 LLaMA-Factory 源码，可在对应环境中进行可编辑安装：

```bash
cd /path/to/LLaMA-Factory
python -m pip install -e .
```

也可以单独安装官方 LLaMA-Factory，然后把 VecLang 作为独立目录使用。安装后先
确认以下命令均指向预期环境：

```bash
command -v python
command -v llamafactory-cli
command -v torchrun
llamafactory-cli version
llamafactory-cli env
```

版本检查应能看到 `0.9.4.dev0`（或经过验证的兼容版本）。若 CLI 不在当前
`PATH`，可以显式指定：

```bash
export VECLANG_PYTHON=/path/to/env/bin/python
export LLAMAFACTORY_CLI=/path/to/env/bin/llamafactory-cli
```

推理脚本启动多卡任务时，会将 `LLAMAFACTORY_CLI` 所在的 `bin` 目录放到
`PATH` 最前面，保证其调用同环境中的 `torchrun`。

## 模型地址

模型按以下优先级选择：

1. 命令行 `--model-path`；
2. 环境变量 `VECLANG_MODEL_PATH`；
3. 若存在 `../model_weights/Qwen3-VL-4B-VecLang-0502`，启动脚本自动使用它；
4. 否则使用配置中的 Hugging Face 仓库 `yyyllll/VecLang-4B`。

例如：

```bash
bash run_inference.sh \
  --task object_detection \
  --model-path /path/to/Qwen3-VL-4B-VecLang-0502 \
  --gpus 0,1,2,3
```

## 数据集与预检

```bash
# 先激活含 LLaMA-Factory 的环境
conda activate Qwen3VL-New

bash run_inference.sh --list-datasets
bash run_inference.sh --task all --dry-run
bash run_inference.sh --task object_detection --dry-run --check-all-images
```

共注册 3 个目标检测测试集和 10 个矢量属性生成测试集，数据根目录定义在
`configs/datasets.yaml`。若数据放在其他位置，可复制该文件后通过 `--catalog`
指定新配置。

## 正式推理

```bash
# 两类任务依次运行，使用当前可见 GPU
bash run_inference.sh --task all

# 仅运行目标检测并指定 4 张 GPU
bash run_inference.sh --task object_detection --gpus 0,1,2,3

# 每个指定数据集只测试前 100 条
bash run_inference.sh \
  --task attributes_generation \
  --datasets WHU_building_instance_test_0129 Cityscales_road_instance_test_0301 \
  --max-samples 100 \
  --gpus 0
```

默认输出到仓库同级的 `VecLang_outputs/YYYYMMDD_HHMMSS/<task>/`，不会进入
GitHub 仓库。也可通过 `--output-root` 或 `VECLANG_OUTPUT_ROOT` 修改。

每个任务目录包含：

- `resolved_config.yaml`：实际使用的完整配置；
- `dataset_registry/dataset_info.json`：临时 LLaMA-Factory 数据注册；
- `llamafactory_output/`：LLaMA-Factory 原始输出；
- `predictions_by_dataset/`：按数据集拆分后的 JSONL；
- `run_summary.json`：数据集、条数和输出位置摘要。

环境入口可通过 `VECLANG_PYTHON` 覆盖，LLaMA-Factory 命令可通过
`LLAMAFACTORY_CLI` 或 `--llamafactory-cli` 覆盖。依赖版本见上级目录
`ENVIRONMENT.md`。

当前服务器也可以不激活环境，直接指定：

```bash
VECLANG_PYTHON=/path/to/Qwen3VL-New/bin/python \
LLAMAFACTORY_CLI=/path/to/Qwen3VL-New/bin/llamafactory-cli \
bash run_inference.sh --list-datasets
```
