# 源码来源与整理范围

所有来源均为同一仓库内用户提供的代码。原始目录保持不变。

- 建筑转换：`eval_tools_multi_class/0_building_eval_tools/1.convert_to_coco_format_instance_patch_IRSAMap_final.py`。
- 水体转换：`eval_tools_multi_class/0_WB_eval_tools/1.convert_to_coco_format_geojson_instance_patch_final.py`。
- 建筑/水体评测：上述目录中的 `2.eval_*_final.py`，保留完整本地 COCO、IoU、C-IoU、PS、PoLiS 实现。逐场景脚本也包含完整评测函数，不跨目录导入。
- 建筑/水体可视化与诊断：上述目录中的 `3.visualize_inference_results.py` 和 `4.analysis_vis.py`。
- 道路 GT 转换、junction 增强拼接：`eval_tools_multi_class/0_road_eval_tools/0.convert_irsamap_road_geojson_to_graph_gt.py`、`1.stitch_coco_polylines_hard_v2_junction.py`。
- 道路提取补入原 `1.extract_road_instance_patch_0304.py` 的显式空预测识别，避免将有效 `[]` 标为解析错误。
- 道路提取、常规拼接、指标运行器及 TOPO/APLS 后端：整理好的 `VecLang/eval_tools/road/`，完整复制到本目录；运行器改为 IRSAMap 区域发现与完整性检查。
- 联合可视化：原开源目录中的脚本，改用本目录道路代码，并支持直接加载最终拼接图。

整理包含：显式 CLI 路径、类别解析与映射、可选 manifest 顺序对齐、空预测评测、逐场景完整 ID、错误输入检查、文档与回归测试。删除了不可执行的旧分支和本机默认路径；历史输出、缓存、重复备份脚本及未使用的实验性评测变体不属于运行依赖。所采用流程的重复代码均完整保留，没有使用软链接、占位函数或对相邻单类别目录的导入。
