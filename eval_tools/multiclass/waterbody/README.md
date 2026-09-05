# IRSAMap waterbody

本目录包含完整独立实现。总流程、输入格式及指标口径见 [上级说明](../README.md)。以下从 multiclass 目录执行；类别 ID 2 为常见配置，请按自己的 GT 调整。

```bash
python waterbody/3.visualize_inference_results.py --dt outputs/waterbody.json --gt inputs/test_patches_512_coco.json --image_dir inputs/test_patches_512 --out_dir outputs/waterbody_vis --category_ids 2

python waterbody/4.analysis_vis.py --gt inputs/test_patches_512_coco.json --dt outputs/waterbody.json --images inputs/test_patches_512 --output outputs/waterbody_analysis --category-id 2 --sample-size 100

python waterbody/5.eval_by_scene.py --gt inputs/test_patches_512_coco.json --dt outputs/waterbody.json --category-id 2 --output outputs/waterbody_scenes.json
```

`4.analysis_vis.py` 默认从全部 GT 图像抽样，包含无预测图像；`--only-with-dt` 可用于展示筛选，不应作为正式评测子集。逐场景评测可用 `--dt-category-id` 显式映射历史预测类别。
