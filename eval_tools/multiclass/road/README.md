# IRSAMap 道路

完整运行顺序及示例见 [上级说明](../README.md#2-道路)。

1. `0.convert_irsamap_road_geojson_to_graph_gt.py` 生成区域 GT 图。
2. `1.extract_road_instance_patch.py` 还原真实道路 patch 的预测坐标，输出 COCO 和 junction。
3. `2.stitch_coco_polylines.py` 或 `2.stitch_coco_polylines_junction.py` 生成最终区域图。
4. `3.run_irsamap_metrics.py` 运行本目录内完整 TOPO/APLS 后端。

运行器的相对路径按当前工作目录解析。APLS 依赖 Go 1.21+；使用 `--go-bin` 指定 Go 可执行文件。结果按区域缓存，输入或后端代码更新后会重新计算；更换评测配置建议使用新的输出目录。
