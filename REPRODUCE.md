# Reproducing VecLang inference and evaluation

This release provides the merged VecLang-4B checkpoint, inference configurations, and evaluation code. The steps below reproduce checkpoint inference and downstream evaluation; training recipes are not included in this release.

## 1. Environment

Follow [ENVIRONMENT.md](ENVIRONMENT.md) to prepare Linux, an NVIDIA GPU, Python 3.10, and a compatible LLaMA-Factory installation. Activate that environment before running the commands below from the repository root.

## 2. Weights and data

- Weights: [yyyllll/VecLang-4B](https://huggingface.co/yyyllll/VecLang-4B). The inference configs use this Hugging Face ID by default.
- Data: [VecLang on Baidu Netdisk](https://pan.baidu.com/s/1b9clAjJT_YWdQBrTR99MKg?pwd=paqs), extraction code **paqs**.

Extract the test manifests and their referenced images so the following paths exist. Preserve the image paths relative to each task's media directory; a manifest alone is not sufficient.

```text
VecLang/
├── dataset/test/
│   ├── object_detection/
│   │   ├── WHU_test_patches_512.json
│   │   ├── WB_test_patches_512.json
│   │   └── IRSAMap_test_patches_512.json
│   └── attributes_generation/
│       ├── WHU_building_instance_test_0129.json
│       ├── Cityscales_road_instance_test_0301.json
│       └── ...
├── inference/
└── eval_tools/
```

The complete list of 13 manifests and expected record counts is in [datasets.yaml](inference/configs/datasets.yaml). For a different data location, set `dataset_root` in a copy of this catalog and pass `--catalog /absolute/path/to/datasets.yaml`. Relative roots resolve against the catalog file.

For offline inference, download the full checkpoint in advance, then pass `--model-path /absolute/path/to/VecLang-4B` or set `VECLANG_MODEL_PATH`. The checkpoint must include configuration, tokenizer/processor files, and all weight shards.

## 3. Preflight and a small run

```bash
bash inference/run_inference.sh --list-datasets
bash inference/run_inference.sh --task all --dry-run --check-all-images

bash inference/run_inference.sh \
  --task object_detection --datasets WHU_test_patches_512 \
  --max-samples 9 --gpus 0

bash inference/run_inference.sh \
  --task attributes_generation --datasets WHU_building_instance_test_0129 \
  --max-samples 1 --gpus 0
```

Listing datasets does not need downloaded data or a GPU. Dry-run validates the local inputs and prints resolved configurations without model inference. The recorded multi-GPU smoke test is described in [SMOKE_TEST.md](inference/SMOKE_TEST.md); small runs establish pipeline functionality, not benchmark accuracy.

## 4. Full inference

```bash
bash inference/run_inference.sh --task object_detection --gpus 0,1,2,3
bash inference/run_inference.sh --task attributes_generation --gpus 0,1,2,3
# Alternatively, run both groups in sequence:
# bash inference/run_inference.sh --task all --gpus 0,1,2,3
```

Adjust GPU IDs to your machine. The catalog includes 3 detection datasets and 10 vector-attribute datasets; use `--datasets` to select a subset. The supplied configurations retain the recorded generation settings:

| Setting | Object detection | Vector attributes |
|---|---:|---:|
| `cutoff_len` | 2300 | 800 |
| `max_new_tokens` | 2100 | 512 |
| Per-device evaluation batch size | 2 | 16 |
| `temperature` | 0.95 | 0.95 |
| `top_p` | 0.7 | 0.7 |

Both tasks use `qwen3_vl_nothink`. Preserve the supplied YAML files when comparing runs. If memory is insufficient, reduce `per_device_eval_batch_size` and record the change.

Outputs default to `../VecLang_outputs/<timestamp>/<task>/`, including `resolved_config.yaml`, `run_summary.json`, the raw LLaMA-Factory output, and `predictions_by_dataset/*.jsonl`. Use `--output-root` to choose another location. Input manifests and images are not rewritten.

## 5. Evaluation

Install the evaluation dependencies in a separate environment as described in [ENVIRONMENT.md](ENVIRONMENT.md). Evaluation requires the corresponding COCO ground truth, crop annotations, and images in addition to generated JSONL. Replace the example paths below with the downloaded data and your run's prediction file.

```bash
python eval_tools/building/1.convert_to_coco_format_geojson_instance_patch.py \
  --inference-file /path/to/predictions.jsonl \
  --anno-dir /path/to/instance_annotations \
  --image-dir /path/to/patch_images \
  --gt-file /path/to/gt.json \
  --output-file outputs/building_predictions.json

python eval_tools/building/2.eval_building.py \
  --gt /path/to/gt.json \
  --dt outputs/building_predictions.json \
  --output outputs/building_metrics.json
```

Follow the task-specific conversion and evaluation order:

| Task | Guide | Pipeline |
|---|---|---|
| Buildings | [building](eval_tools/building/README.md) | Restore crop coordinates → COCO/polygon metrics |
| Water bodies | [waterbody](eval_tools/waterbody/README.md) | Restore crop coordinates → COCO/polygon metrics |
| Roads | [road](eval_tools/road/README.md) | Extract patches → stitch graph → TOPO/APLS |
| IRSAMap multiclass | [multiclass](eval_tools/multiclass/README.md) | Per-class conversion/evaluation → joint visualization |

Use the standard detection-list JSON for polygon evaluation, not the companion `*_full.json`. Road APLS requires Go 1.21+ and downloads its Go modules on first build. Generated text BLEU/ROUGE and the demo's selected scenes do not substitute for spatial benchmark metrics.

## 6. Demo

Download [VecLang-Standalone.html](https://github.com/yyyyll0ss/VecLang/raw/refs/heads/main/demo/VecLang-Standalone.html) and open it in a browser. It contains precomputed examples and needs no model or GPU. To run the source version, install Node.js 18+ and use:

```bash
cd demo
npm run dev
```

Open `http://127.0.0.1:4173`. See [demo/README.md](demo/README.md) for interaction and data-provenance details.
