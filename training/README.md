# VecLang training information

The supervised fine-tuning (SFT) stage uses [LLaMA-Factory](https://github.com/hiyouga/LlamaFactory). The reinforcement learning (RL) stage uses [ms-swift](https://github.com/modelscope/ms-swift), with a custom Structured Vector Language reward function in [My_Reward_v9.py](My_Reward_v9.py). The published checkpoint is intended for inference as described in [REPRODUCE.md](../REPRODUCE.md); this directory documents the training frameworks, data contract, and reward implementation. Complete training configurations and training checkpoints are not part of this code release.

## SFT data

The released data archive contains training manifests under `dataset/train/object_detection/` and `dataset/train/attributes_generation/`. Each manifest is a JSON array of examples with `messages` and `images` fields. A typical example has this shape:

```json
{
  "messages": [
    {"role": "user", "content": "<image>\n...task instructions..."},
    {"role": "assistant", "content": "[{\"type\":\"Feature\",...}]"}
  ],
  "images": ["WHU_building_instance_train_0129/example.png"]
}
```

The object-detection target is a JSON array of `{ "bbox_2d": [x1, y1, x2, y2], "label": "..." }` objects. The attribute-generation target is a JSON array of GeoJSON-like `Feature` objects: polygons for buildings and water bodies, and `MultiLineString` geometry with optional `junction` points for roads. Coordinates in the released prompts and targets are patch-local and normalized to `[0, 1000]`. The image path must resolve against the configured media directory; check the paths in your downloaded data before registering a dataset with LLaMA-Factory. See the actual manifests in the [released data archive](../README.md#todo) for prompts and examples rather than treating the shortened example above as a full training record.

The inference environment and tested LLaMA-Factory version are recorded in [ENVIRONMENT.md](../ENVIRONMENT.md). Those versions describe the published inference setup; they do not lock every package used during SFT training.

## RL data and reward

The V9 reward plugin uses ms-swift's `ORM` interface. RL records provide `messages` and `images` as model inputs and a separate `solution` string containing the reference vector-language JSON. Unlike SFT records, the reference is not an assistant message: ms-swift passes `solution` to the reward function alongside generated `completions`. A minimal record shape is:

```json
{
  "messages": [{"role": "user", "content": "<image>\n...task instructions..."}],
  "images": ["/path/to/example.png"],
  "solution": "[{\"type\":\"Feature\",...}]"
}
```

The reward scores three components: vector-language validity, vector-content consistency, and vector-execution fidelity. The combined `vector_language_reward_v9` score weights them `0.05`, `0.10`, and `0.85`. The source also registers individual component rewards; the following names and weights match a recorded V9 ms-swift GRPO run (`ms-swift 4.1.0.dev0`):

```bash
swift rlhf --rlhf_type grpo \
  --model /path/to/sft-checkpoint \
  --dataset /path/to/rl-data.json \
  --external_plugins training/My_Reward_v9.py \
  --reward_funcs \
    vector_language_validity_reward_v9 \
    vector_content_reward_v9 \
    vector_execution_reward_v9 \
    vector_execution_iou_reward_v9 \
    vector_execution_alignment_reward_v9 \
    vector_execution_connectivity_reward_v9 \
  --reward_weights 0.05 0.10 0.85 0 0 0
```

Run this example from the repository root after installing a compatible ms-swift environment and replacing the model and dataset paths. Add the remaining training and hardware arguments for your own setup; this is the reward-plugin invocation, not a complete reproduction of the original GRPO run. For the combined reward as a single function, use `--reward_funcs vector_language_reward_v9` instead of the six component names and omit `--reward_weights`.

The plugin imports `swift.rewards` and uses Shapely for polygon overlap, boundary alignment, and road geometry scores. Install Shapely in the ms-swift training environment; without it, geometry-related rewards fall back to low scores. The plugin writes `training/my_reward_v9_debug.log` when imported. This runtime log is ignored by Git.
