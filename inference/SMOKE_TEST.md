# 六卡推理实测记录

测试日期：2026-09-01（Asia/Shanghai）

## 结论

`Qwen3-VL-4B-VecLang-0502` 权重可以在 `Qwen3VL-New` 环境中通过 LLaMA-Factory 真正执行图像推理，不是仅完成配置或模型加载。目标检测与矢量属性生成均已在 6 张 RTX 3090 上完成生成并写出预测结果。

推理使用 6 个分布式进程，对应 `cuda:0` 至 `cuda:5`。模型为 `Qwen3VLForConditionalGeneration`，参数量 4,437,815,808，计算类型为 BF16，注意力后端实际回退到 PyTorch SDPA。

## 样例结果

| 任务 / 数据集 | 样本数 | JSON 可解析 | 实测表现 |
|---|---:|---:|---|
| 目标检测 / WHU | 9 | 9/9 | 第 9 条为首个非空样本：22 个真值框，预测 20 个框；按该单图的一对一 IoU >= 0.5 简单匹配，20 个预测均匹配真值，precision=1.000、recall=0.909、F1=0.952 |
| 建筑属性 / WHU | 1 | 1/1 | 正确生成 `Polygon`、`Building` 及 6 个坐标；仅第一个坐标由真值 522 预测为 476，其余坐标一致 |
| 道路属性 / Cityscales | 1 | 1/1 | 正确生成 `MultiLineString`、`Road`、`junction` 的 JSON 结构，但道路几何和连接点与该条真值差异较明显 |

这里的样本量只适合验证推理链路和初步观察权重能力，不能替代完整测试集上的检测 mAP、矢量几何相似度和拓扑指标评估。LLaMA-Factory 输出的 BLEU/ROUGE 只衡量文本重叠，也不应作为空间任务的最终指标。

## 结果文件说明

为控制 GitHub 仓库大小，原始 smoke test 预测、临时数据注册和运行目录不随代码
上传。复现命令会在仓库同级的 `VecLang_outputs/` 中重新生成对应结果。

## 复现命令

```bash
cd /path/to/LLaMA-Factory-New

VecLang/inference/run_inference.sh \
  --task object_detection \
  --datasets WHU_test_patches_512 \
  --max-samples 9 \
  --gpus 0,1,2,3,4,5 \
  --run-name smoke_nonempty_detection_20260901

VecLang/inference/run_inference.sh \
  --task attributes_generation \
  --datasets Cityscales_road_instance_test_0301 \
  --max-samples 1 \
  --gpus 0,1,2,3,4,5 \
  --run-name smoke_cityscales_20260901
```

## 运行备注

第一次启动曾因当前 shell 的基础 Conda 环境抢占 `torchrun` 而失败。推理入口现已把所选 LLaMA-Factory CLI 的 `bin` 目录置于 `PATH` 首位，并把仓库 `src` 加入 `PYTHONPATH`，从而确保主进程和六个工作进程均使用 `Qwen3VL-New`。最终三次有效实测均正常退出，结束后六张 GPU 均为约 1 MiB 显存占用、0% 利用率，无遗留推理进程。
