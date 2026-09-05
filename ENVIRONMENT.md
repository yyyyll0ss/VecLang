# VecLang 推理环境

记录日期：2026-09-01（Asia/Shanghai）

## 安装与复现

以下版本表记录了原始 Linux 推理环境。推理脚本使用当前环境的 `python3` 和
`llamafactory-cli`，不要求原机器的绝对路径。建议 Python 3.10 和支持 BF16 的 NVIDIA GPU。

```bash
conda create -n veclang python=3.10 -y
conda activate veclang
python -m pip install torch==2.4.0 torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cu121

# 使用兼容 Qwen3-VL / qwen3_vl_nothink 的 LLaMA-Factory 源码。
cd /path/to/LLaMA-Factory
python -m pip install -e .

python -m pip install transformers==4.57.1 tokenizers==0.22.2 accelerate==1.11.0 datasets==4.0.0 peft==0.17.1 trl==0.9.6 safetensors==0.5.3 sentencepiece==0.2.1 numpy==1.26.4 scipy==1.15.3 Pillow==11.3.0 PyYAML==6.0.3 tqdm==4.67.3 gradio==5.45.0 pydantic==2.10.6
python -m pip check
llamafactory-cli version
llamafactory-cli env
nvidia-smi
```

LLaMA-Factory 官方源码：<https://github.com/hiyouga/LlamaFactory>。已记录的版本为
`0.9.4.dev0`，但本次归档没有保存对应 Git commit，因此不能据此锁定完全相同的源码。
请使用已有的兼容源码；最新上游版本未在这里验证，若依赖检查失败，不要忽略冲突。
上述固定版本用于恢复已记录的核心依赖，不是所有传递依赖的完整锁文件。

Python、CLI 与 `torchrun` 应来自同一环境。需要显式指定时：

```bash
export VECLANG_PYTHON=/path/to/env/bin/python
export LLAMAFACTORY_CLI=/path/to/env/bin/llamafactory-cli
```

评估建议单独建环境，避免评估包升级影响推理版本。在 VecLang 仓库根目录运行：

```bash
conda create -n veclang-eval python=3.10 -y
conda activate veclang-eval
python -m pip install -r eval_tools/requirements.txt -r eval_tools/multiclass/requirements.txt
```

道路 APLS 还需 Go 1.21+，首次构建需访问 Go 模块源。Demo 单文件版无需 Python；
开发与构建需要 Node.js 18+，无第三方 npm 运行依赖。
完整步骤见 [REPRODUCE.md](REPRODUCE.md)。

## 原始测试环境记录

- Conda 环境：`/home/isalab301/.conda/envs/Qwen3VL-New`
- Python：`3.10.20`（conda-forge，GCC 14.3.0）
- LLaMA-Factory：`0.9.4.dev0`
- LLaMA-Factory CLI：`/home/isalab301/.conda/envs/Qwen3VL-New/bin/llamafactory-cli`
- LLaMA-Factory 源码：`/mnt/data/yyl/LLaMA-Factory-New/src/llamafactory`
- 平台：Linux x86_64，kernel `5.8.0-43-generic`，glibc 2.31
- pip：`26.0.1`

原始环境中的调用方式（新机器请使用自己的仓库位置和已激活环境）：

```bash
cd /mnt/data/yyl/LLaMA-Factory-New/VecLang/inference
bash run_inference.sh --list-datasets
```

## 核心依赖版本

| 组件 | 版本 |
|---|---|
| PyTorch | 2.4.0+cu121 |
| torchvision | 0.19.0+cu121 |
| CUDA（PyTorch 编译版本） | 12.1 |
| cuDNN | 9.1.0 |
| Transformers | 4.57.1 |
| Tokenizers | 0.22.2 |
| Accelerate | 1.11.0 |
| Datasets | 4.0.0 |
| PEFT | 0.17.1 |
| TRL | 0.9.6 |
| Safetensors | 0.5.3 |
| SentencePiece | 0.2.1 |
| NumPy | 1.26.4 |
| SciPy | 1.15.3 |
| Pillow | 11.3.0 |
| PyYAML | 6.0.3 |
| tqdm | 4.67.3 |
| Gradio | 5.45.0 |
| Pydantic | 2.10.6 |

## 可选库状态

当前环境中未安装：

- `deepspeed`
- `bitsandbytes`
- `flash-attn`
- `qwen-vl-utils`
- `vllm`
- `pycocotools`
- `opencv-python` / `opencv-python-headless`

当前配置使用 `flash_attn: auto`，未安装 FlashAttention 时会自动采用可用的普通注意力实现。配置保留历史实验中的 `quantization_method: bnb`，但没有设置 `quantization_bit`，因此当前完整 BF16 权重推理不依赖 bitsandbytes。

## GPU 状态

实际 GPU 运行环境中共有 6 张 NVIDIA GeForce RTX 3090，每张显存 24576 MiB。`Qwen3VL-New` 环境能够识别全部 6 张卡，已于 2026-09-01 使用 `cuda:0` 至 `cuda:5` 完成目标检测和矢量属性生成的分布式推理实测。详细结果见 `inference/SMOKE_TEST.md`。

受限的命令沙箱中可能出现 `torch.cuda.is_available() == False` 或 `nvidia-smi` 无法连接驱动；这不代表宿主机 GPU 不可用。正式推理仍会先执行 GPU 预检，不能访问 GPU 时安全退出。

## 版本复核命令

```bash
/home/isalab301/.conda/envs/Qwen3VL-New/bin/llamafactory-cli env
/home/isalab301/.conda/envs/Qwen3VL-New/bin/python -m pip list
nvidia-smi
```

注意：版本表描述的是上述指定环境，不是 shell 当前可能激活的 `/home/hnu3/anaconda3` 基础环境。
推理入口会把所选 CLI 的 `bin` 目录置于 `PATH` 首位，确保多卡子进程使用同一环境中的 `torchrun` 和 Python。
