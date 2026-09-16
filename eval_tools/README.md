# Vector-map evaluation tools

本目录汇总建筑、道路、水体和多类别结果的转换、评估与可视化代码，只保留运行所需源码，不包含历史评估结果、可视化图片、数据集、模型输出、缓存或已编译二进制文件。

## 目录

- `building/`：建筑实例结果转换、COCO/多边形指标评估、可视化。
- `road/`：道路结果转换、跨 patch 图拼接、CityScale TOPO/APLS 指标。
- `waterbody/`：水体实例结果转换、COCO/多边形指标评估、可视化。
- `multiclass/`：完整独立的 IRSAMap 建筑、水体、道路转换、评测及联合可视化，含本地 TOPO/APLS 后端。

每个子目录的 `README.md` 给出了代码功能、输入输出和推荐运行顺序。

本页只作为评估入口索引：完整的“环境 → 推理 → 裁剪 → 评估”顺序见
[`../REPRODUCE.md`](../REPRODUCE.md)，具体命令参数以相应任务子目录 README
为准。`inference/inference_cut` 负责生成实例裁剪和属性预测，不计算指标。

## 环境

建议使用 Python 3.9 或更高版本：

```bash
python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows PowerShell
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

道路 APLS 还需要 Go 1.21 或更高版本。首次运行会根据 `road/cityscale_metrics/apls_rtreego_fast/go.mod` 下载 Go 依赖并编译评估程序。

## 数据约定

- COCO GT 至少应包含 `images`、`annotations`、`categories`。
- 建筑、水体属性预测经任务转换脚本处理后会生成标准 COCO detection list，以及带 `images` 元信息的 `*_full.json`；两者用途不同。
- 输出目录由脚本自动创建，但输入文件和数据集不会随本目录发布。
- 所有数据路径均通过命令行传入，代码中不再绑定本机数据路径。

若使用检测后裁剪流程，路径对应关系为：

| 评估输入 | `inference_cut` 产物 |
| --- | --- |
| 属性预测 JSONL | `<run>/attribute_inference/generated_predictions.jsonl` |
| 实例 annotation | `<run>/instance_crops/annotations/` |
| 属性推理 manifest（多类别转换推荐） | `<run>/instance_crops/attribute_manifest.json` |

评估脚本的 `--image-dir` 仍应指向原始检测 patch 影像目录，而不是 256×256 的
`instance_crops/images/`。COCO GT 也必须另行提供；动态 manifest 中由检测框生成
的 assistant 占位内容不是真值。

## 开源前提醒

本目录尚未添加许可证。公开发布前请根据项目政策补充 `LICENSE`，并确认所使用数据集和第三方算法代码的许可证兼容性。
