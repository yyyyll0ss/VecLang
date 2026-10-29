import json
import logging
import math
from pathlib import Path
from statistics import fmean
from typing import Dict, List, Optional, Tuple

from swift.rewards import ORM, orms


END_TOKENS = ("<|im_end|>", "<|endoftext|>")


def _build_logger() -> logging.Logger:
    logger = logging.getLogger("StructuredVectorLanguageRewardV9")
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("[MyRewardV9] %(message)s")

    log_path = Path(__file__).resolve().with_name("my_reward_v9_debug.log")
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    logger.propagate = False
    return logger


logger = _build_logger()


class StructuredVectorLanguageRewardV9(ORM):
    """
    V9 reward keeps the three-level structure from V8 while rebuilding road fidelity:
    1) Vector Language Validity
    2) Vector Content Consistency
    3) Vector Execution Fidelity

    Key changes:
    - Closed classes keep the V8 boundary-sensitive design.
    - Road fidelity becomes centerline-F1 dominant instead of average overlap dominant.
    - Road topology/connectivity is preserved as a second major execution signal.
    - Road IoU is retained only as a low-weight stabilizer.
    """

    CLOSED_CLASSES = {"building", "water", "waterbody", "farmland"}
    OPEN_CLASSES = {"road"}

    ROAD_BUFFER_DISTANCE = 3.0
    ROAD_ATTACH_TOL = 3.0
    JUNCTION_MATCH_TOL = 3.0
    ROAD_NODE_MERGE_TOL = 2.0
    ROAD_NODE_MATCH_TOL = 4.0
    ROAD_MIN_EDGE_LENGTH = 1e-3
    ROAD_CONNECTIVITY_MODE = "light"  # options: "light", "graph"
    IOU_EPS = 1e-6
    MAX_POLYGON_VERTICES = 40
    MAX_ROAD_SEGMENTS = 24

    VALIDITY_WEIGHT = 0.05
    CONTENT_WEIGHT = 0.10
    FIDELITY_WEIGHT = 0.85

    # Level 1: syntax-only legality
    VALIDITY_JSON_WEIGHT = 0.25
    VALIDITY_SCHEMA_WEIGHT = 0.35
    VALIDITY_FIELD_WEIGHT = 0.40

    # Level 2: content consistency
    CONTENT_TYPE_WEIGHT = 0.35
    CLOSED_CONTENT_VALIDITY_WEIGHT = 0.40
    CLOSED_CONTENT_COMPACTNESS_WEIGHT = 0.25
    OPEN_CONTENT_VALIDITY_WEIGHT = 0.30
    OPEN_CONTENT_COMPACTNESS_WEIGHT = 0.15
    OPEN_CONTENT_RELATION_WEIGHT = 0.20

    # Level 3: execution fidelity
    CLOSED_IOU_WEIGHT = 0.50
    CLOSED_ALIGNMENT_WEIGHT = 0.50
    OPEN_IOU_WEIGHT = 0.10
    OPEN_ALIGNMENT_WEIGHT = 0.60
    OPEN_CONNECTIVITY_WEIGHT = 0.30
    ALIGN_QUADRATIC_POWER = 2.0
    IOU_QUADRATIC_POWER = 2.0
    CLOSED_IOU_THRESHOLD = 0.20
    OPEN_IOU_THRESHOLD = 0.10
    IOU_LOW_SCORE = 0.05

    EMPTY_ROAD_CLASS = "road"
    EMPTY_ROAD_TYPE = "MultiLineString"
    EMPTY_ROAD_MISSING_JUNCTION_SCORE = 0.70
    EMPTY_ROAD_EXPLICIT_EMPTY_JUNCTION_SCORE = 1.00
    EMPTY_ROAD_FALSE_POSITIVE_PENALTY = 0.0
    BOUNDARY_BUFFER_RATIO = 0.02
    ROAD_ALIGNMENT_BUFFER = 2.5

    ROAD_GRAPH_NODE_WEIGHT = 0.35
    ROAD_GRAPH_EDGE_WEIGHT = 0.45
    ROAD_GRAPH_COMPONENT_WEIGHT = 0.20

    ENABLE_SUCCESS_FLOOR = False
    SUCCESS_FLOOR = 0.05

    ENABLE_LOG = False
    LOG_EVERY = 200
    ENABLE_DIAGNOSTIC_LOG = False
    DIAGNOSTIC_LOG_EVERY = 200

    _calls = 0
    _parse_fail = 0
    _validity_weak = 0
    _content_weak = 0
    _fidelity_weak = 0

    SCORE_MODE = "total"

    def __call__(self, completions, solution, **kwargs) -> List[float]:
        rewards: List[float] = []
        batch_breakdowns: List[Dict[str, float]] = []

        for idx, (content, sol) in enumerate(zip(completions, solution)):
            self.__class__._calls += 1
            breakdown = self._compute_breakdown(content, sol, idx=idx)
            reward = float(breakdown.get(self._score_key(), 0.0))
            rewards.append(reward)
            batch_breakdowns.append(breakdown)

            if self.ENABLE_LOG:
                self._log_sample_breakdown(idx, breakdown)

            self._maybe_log_diagnostics()

        if self.ENABLE_LOG and rewards and (self.__class__._calls % max(1, int(self.LOG_EVERY)) == 0):
            self._log_batch_breakdown(batch_breakdowns)
        return rewards

    @classmethod
    def _score_key(cls) -> str:
        mode_to_key = {
            "total": "reward_total",
            "validity": "validity",
            "content": "content",
            "fidelity": "fidelity",
            "fidelity_iou": "fidelity_iou",
            "fidelity_alignment": "fidelity_alignment",
            "fidelity_connectivity": "fidelity_connectivity",
        }
        return mode_to_key.get(str(cls.SCORE_MODE).strip().lower(), "reward_total")

    def _maybe_log_diagnostics(self) -> None:
        if not self.ENABLE_DIAGNOSTIC_LOG:
            return

        n = self.__class__._calls
        if n <= 0 or n % max(1, int(self.DIAGNOSTIC_LOG_EVERY)) != 0:
            return

        def pct(x: int) -> float:
            return 100.0 * x / n

        logger.info(
            "diag "
            f"calls={n}, parse_fail={self.__class__._parse_fail}({pct(self.__class__._parse_fail):.2f}%), "
            f"validity_weak={self.__class__._validity_weak}({pct(self.__class__._validity_weak):.2f}%), "
            f"content_weak={self.__class__._content_weak}({pct(self.__class__._content_weak):.2f}%), "
            f"fidelity_weak={self.__class__._fidelity_weak}({pct(self.__class__._fidelity_weak):.2f}%)"
        )

    def _compute_breakdown(self, content: str, sol: str, idx: Optional[int] = None) -> Dict[str, float]:
        breakdown = self._zero_breakdown()
        gt_obj = self._parse_feature(sol)
        if gt_obj is None:
            self.__class__._parse_fail += 1
            breakdown["parse_fail"] = 1.0
            if self.ENABLE_LOG:
                logger.info(
                    f"gt_parse_fail idx={idx}, gt_head={str(sol)[:160].replace(chr(10), ' ')}"
                )
            return breakdown

        validity, pred_obj, validity_parts = self._vector_language_validity_breakdown(content)
        breakdown.update(validity_parts)
        breakdown["validity"] = validity

        if validity <= 0.0:
            self.__class__._parse_fail += 1
            breakdown["parse_fail"] = 1.0
            breakdown["reason_language_validity_zero"] = 1.0
            return breakdown

        consistency, content_parts = self._vector_content_consistency_breakdown(pred_obj, gt_obj)
        fidelity, fidelity_parts = self._vector_execution_fidelity_breakdown(pred_obj, gt_obj)
        breakdown.update(content_parts)
        breakdown.update(fidelity_parts)
        breakdown["content"] = consistency
        breakdown["fidelity"] = fidelity

        if validity < 0.5:
            self.__class__._validity_weak += 1
        if consistency < 0.5:
            self.__class__._content_weak += 1
        if fidelity < 0.5:
            self.__class__._fidelity_weak += 1

        reward_total = (
            self.VALIDITY_WEIGHT * validity
            + self.CONTENT_WEIGHT * consistency
            + self.FIDELITY_WEIGHT * fidelity
        )
        if self.ENABLE_SUCCESS_FLOOR:
            reward_total = max(self.SUCCESS_FLOOR, reward_total)
        breakdown["reward_total"] = max(0.0, min(1.0, float(reward_total)))
        return breakdown

    def _zero_breakdown(self) -> Dict[str, float]:
        return {
            "reward_total": 0.0,
            "validity": 0.0,
            "content": 0.0,
            "fidelity": 0.0,
            "validity_json": 0.0,
            "validity_schema": 0.0,
            "validity_field": 0.0,
            "content_type_match": 0.0,
            "content_geometry_validity": 0.0,
            "content_compactness": 0.0,
            "content_relation": 0.0,
            "fidelity_iou": 0.0,
            "fidelity_iou_raw": 0.0,
            "fidelity_alignment": 0.0,
            "fidelity_alignment_raw": 0.0,
            "fidelity_alignment_precision": 0.0,
            "fidelity_alignment_recall": 0.0,
            "fidelity_connectivity": 0.0,
            "fidelity_graph_node": 0.0,
            "fidelity_graph_edge": 0.0,
            "fidelity_graph_component": 0.0,
            "parse_fail": 0.0,
            "reason_language_validity_zero": 0.0,
        }

    def _log_sample_breakdown(self, idx: int, breakdown: Dict[str, float]) -> None:
        logger.info(
            "sample "
            f"mode={self.SCORE_MODE} idx={idx}, total={breakdown['reward_total']:.4f}, "
            f"validity={breakdown['validity']:.4f}"
            f"(json={breakdown['validity_json']:.4f}, schema={breakdown['validity_schema']:.4f}, field={breakdown['validity_field']:.4f}), "
            f"content={breakdown['content']:.4f}"
            f"(type={breakdown['content_type_match']:.4f}, geom={breakdown['content_geometry_validity']:.4f}, "
            f"compact={breakdown['content_compactness']:.4f}, relation={breakdown['content_relation']:.4f}), "
            f"fidelity={breakdown['fidelity']:.4f}"
            f"(iou={breakdown['fidelity_iou']:.4f}, iou_raw={breakdown['fidelity_iou_raw']:.4f}, "
            f"align={breakdown['fidelity_alignment']:.4f}, align_raw={breakdown['fidelity_alignment_raw']:.4f}, "
            f"align_p={breakdown['fidelity_alignment_precision']:.4f}, "
            f"align_r={breakdown['fidelity_alignment_recall']:.4f}, "
            f"conn={breakdown['fidelity_connectivity']:.4f}, graph_node={breakdown['fidelity_graph_node']:.4f}, "
            f"graph_edge={breakdown['fidelity_graph_edge']:.4f}, graph_component={breakdown['fidelity_graph_component']:.4f}), "
            f"parse_fail={breakdown['parse_fail']:.0f}, validity_zero={breakdown['reason_language_validity_zero']:.0f}"
        )

    def _log_batch_breakdown(self, batch_breakdowns: List[Dict[str, float]]) -> None:
        keys = [
            "reward_total",
            "validity",
            "validity_json",
            "validity_schema",
            "validity_field",
            "content",
            "content_type_match",
            "content_geometry_validity",
            "content_compactness",
            "content_relation",
            "fidelity",
            "fidelity_iou",
            "fidelity_iou_raw",
            "fidelity_alignment",
            "fidelity_alignment_raw",
            "fidelity_alignment_precision",
            "fidelity_alignment_recall",
            "fidelity_connectivity",
            "fidelity_graph_node",
            "fidelity_graph_edge",
            "fidelity_graph_component",
            "parse_fail",
            "reason_language_validity_zero",
        ]
        summary = ", ".join(
            f"{key}={fmean(item[key] for item in batch_breakdowns):.4f}"
            for key in keys
        )
        scores = [float(item[self._score_key()]) for item in batch_breakdowns]
        logger.info(
            f"batch_summary mode={self.SCORE_MODE} count={len(batch_breakdowns)}, "
            f"min={min(scores):.4f}, max={max(scores):.4f}, avg={fmean(scores):.4f}, {summary}"
        )

    def _strip_special_tokens(self, text: str) -> str:
        s = text.strip()
        for tok in END_TOKENS:
            if s.endswith(tok):
                s = s[:-len(tok)].rstrip()
        return s

    def _parse_json_text(self, text: str):
        if not isinstance(text, str):
            return None

        raw = self._strip_special_tokens(text)
        try:
            return json.loads(raw)
        except Exception:
            return None

    def _extract_feature_from_parsed(self, obj) -> Optional[Dict]:
        if isinstance(obj, dict):
            return obj
        if isinstance(obj, list) and not obj:
            return self._make_empty_road_feature()
        if isinstance(obj, list) and obj and isinstance(obj[0], dict):
            return obj[0]
        return None

    def _parse_feature(self, text: str) -> Optional[Dict]:
        parsed = self._parse_json_text(text)
        return self._extract_feature_from_parsed(parsed)

    def _make_empty_road_feature(self) -> Dict:
        return {
            "type": "Feature",
            "geometry": {"type": self.EMPTY_ROAD_TYPE, "coordinates": []},
            "properties": {
                "class": self.EMPTY_ROAD_CLASS,
                "junction": [],
                "_empty_road": True,
            },
        }

    def _is_empty_road_feature(self, feature: Optional[Dict]) -> bool:
        if not isinstance(feature, dict):
            return False
        props = feature.get("properties")
        if isinstance(props, dict) and props.get("_empty_road") is True:
            return True
        if self._extract_class_name(feature) != self.EMPTY_ROAD_CLASS:
            return False
        geom = feature.get("geometry")
        if not isinstance(geom, dict):
            return False
        if str(geom.get("type", "")).strip().lower() != "multilinestring":
            return False
        coords = geom.get("coordinates")
        return isinstance(coords, list) and len(coords) == 0

    def _extract_class_name(self, feature: Optional[Dict]) -> str:
        cls = feature.get("properties", {}).get("class", "") if isinstance(feature, dict) else ""
        if not isinstance(cls, str):
            return ""
        return cls.strip().lower()

    def _extract_geometry_type(self, feature: Optional[Dict]) -> str:
        gtype = feature.get("geometry", {}).get("type", "") if isinstance(feature, dict) else ""
        if not isinstance(gtype, str):
            return ""
        return gtype.strip().lower()

    def _expected_geometry_type(self, cls_name: str) -> str:
        if cls_name in self.CLOSED_CLASSES:
            return "polygon"
        if cls_name in self.OPEN_CLASSES:
            return "multilinestring"
        return ""

    def _vector_language_validity(self, text: str) -> Tuple[float, Optional[Dict]]:
        score, pred_obj, _ = self._vector_language_validity_breakdown(text)
        return score, pred_obj

    def _vector_language_validity_breakdown(self, text: str) -> Tuple[float, Optional[Dict], Dict[str, float]]:
        parsed = self._parse_json_text(text)
        r_json = 1.0 if parsed is not None else 0.0
        if r_json <= 0.0:
            return 0.0, None, {
                "validity_json": 0.0,
                "validity_schema": 0.0,
                "validity_field": 0.0,
            }

        pred_obj = self._extract_feature_from_parsed(parsed)
        r_schema = self._language_schema_score(pred_obj)
        r_field = self._language_field_usable_score(pred_obj)

        score = (
            self.VALIDITY_JSON_WEIGHT * r_json
            + self.VALIDITY_SCHEMA_WEIGHT * r_schema
            + self.VALIDITY_FIELD_WEIGHT * r_field
        )
        return max(0.0, min(1.0, float(score))), pred_obj, {
            "validity_json": r_json,
            "validity_schema": r_schema,
            "validity_field": r_field,
        }

    def _language_schema_score(self, pred_obj: Optional[Dict]) -> float:
        if not isinstance(pred_obj, dict):
            return 0.0

        score = 0.0
        if pred_obj.get("type") == "Feature":
            score += 0.25
        geometry = pred_obj.get("geometry")
        properties = pred_obj.get("properties")
        if isinstance(geometry, dict):
            score += 0.25
        if isinstance(properties, dict):
            score += 0.25
        if isinstance(geometry, dict) and isinstance(geometry.get("type"), str):
            score += 0.15
        if isinstance(geometry, dict) and geometry.get("coordinates") is not None:
            score += 0.10
        return score

    def _language_field_usable_score(self, pred_obj: Optional[Dict]) -> float:
        if not isinstance(pred_obj, dict):
            return 0.0

        geometry = pred_obj.get("geometry")
        properties = pred_obj.get("properties")
        if not isinstance(geometry, dict) or not isinstance(properties, dict):
            return 0.0

        gtype = geometry.get("type")
        coords = geometry.get("coordinates")
        cls_name = properties.get("class", "")
        if not isinstance(gtype, str):
            return 0.0

        gtype_l = gtype.strip().lower()
        class_score = 1.0 if isinstance(cls_name, str) and cls_name.strip() else 0.0

        if gtype_l == "polygon":
            coord_score = 1.0 if self._to_polygon_coords(coords) is not None else 0.0
            return 0.25 * class_score + 0.75 * coord_score

        if gtype_l == "multilinestring":
            coord_score = 1.0 if self._to_multilinestring_coords(coords) is not None else 0.0
            junction_score = self._optional_junction_field_score(pred_obj)
            return 0.20 * class_score + 0.60 * coord_score + 0.20 * junction_score

        return 0.0

    def _optional_junction_field_score(self, pred_obj: Dict) -> float:
        props = pred_obj.get("properties")
        if not isinstance(props, dict):
            return 0.0
        if "junction" not in props:
            return 1.0
        return 1.0 if self._parse_junctions(pred_obj) is not None else 0.0

    def _type_match_score(self, pred_obj: Optional[Dict], gt_obj: Dict) -> float:
        pred_cls = self._extract_class_name(pred_obj)
        gt_cls = self._extract_class_name(gt_obj)
        pred_gtype = self._extract_geometry_type(pred_obj)
        expected_gtype = self._expected_geometry_type(gt_cls)

        score = 0.0
        if pred_cls == gt_cls and gt_cls:
            score += 0.5
        if pred_gtype == expected_gtype and expected_gtype:
            score += 0.5
        return score

    def _field_integrity_score(self, pred_obj: Optional[Dict]) -> float:
        if not isinstance(pred_obj, dict):
            return 0.0

        pred_cls = self._extract_class_name(pred_obj)
        coords = pred_obj.get("geometry", {}).get("coordinates")

        if pred_cls in self.CLOSED_CLASSES:
            return 1.0 if self._to_polygon_coords(coords) is not None else 0.0

        if pred_cls in self.OPEN_CLASSES:
            score = 0.0
            if self._to_multilinestring_coords(coords) is not None:
                score += 0.5
            if self._parse_junctions(pred_obj) is not None:
                score += 0.5
            return score

        return 0.0

    def _vector_content_consistency(self, pred_obj: Optional[Dict], gt_obj: Dict) -> float:
        score, _ = self._vector_content_consistency_breakdown(pred_obj, gt_obj)
        return score

    def _vector_content_consistency_breakdown(
        self, pred_obj: Optional[Dict], gt_obj: Dict
    ) -> Tuple[float, Dict[str, float]]:
        gt_cls = self._extract_class_name(gt_obj)
        if gt_cls in self.CLOSED_CLASSES:
            return self._closed_content_consistency_breakdown(pred_obj, gt_obj)
        if gt_cls in self.OPEN_CLASSES:
            return self._open_content_consistency_breakdown(pred_obj, gt_obj)
        return 0.0, {
            "content_type_match": 0.0,
            "content_geometry_validity": 0.0,
            "content_compactness": 0.0,
            "content_relation": 0.0,
        }

    def _closed_content_consistency(self, pred_obj: Optional[Dict], gt_obj: Dict) -> float:
        score, _ = self._closed_content_consistency_breakdown(pred_obj, gt_obj)
        return score

    def _closed_content_consistency_breakdown(
        self, pred_obj: Optional[Dict], gt_obj: Dict
    ) -> Tuple[float, Dict[str, float]]:
        r_type_match = self._type_match_score(pred_obj, gt_obj)
        normalized = self._normalize_geometry(pred_obj)
        if normalized is None or normalized.get("type") != "Polygon":
            score = self.CONTENT_TYPE_WEIGHT * r_type_match
            return score, {
                "content_type_match": r_type_match,
                "content_geometry_validity": 0.0,
                "content_compactness": 0.0,
                "content_relation": 0.0,
            }

        r_geometry_validity = self._polygon_geometry_validity_score(normalized)
        r_compactness = self._vertex_budget_score(normalized)
        score = (
            self.CONTENT_TYPE_WEIGHT * r_type_match
            + self.CLOSED_CONTENT_VALIDITY_WEIGHT * r_geometry_validity
            + self.CLOSED_CONTENT_COMPACTNESS_WEIGHT * r_compactness
        )
        return score, {
            "content_type_match": r_type_match,
            "content_geometry_validity": r_geometry_validity,
            "content_compactness": r_compactness,
            "content_relation": 0.0,
        }

    def _open_content_consistency(self, pred_obj: Optional[Dict], gt_obj: Dict) -> float:
        score, _ = self._open_content_consistency_breakdown(pred_obj, gt_obj)
        return score

    def _open_content_consistency_breakdown(
        self, pred_obj: Optional[Dict], gt_obj: Dict
    ) -> Tuple[float, Dict[str, float]]:
        r_type_match = self._type_match_score(pred_obj, gt_obj)
        if self._is_empty_road_feature(gt_obj):
            empty_score = self._empty_road_match_score(pred_obj)
            score = (
                self.CONTENT_TYPE_WEIGHT * r_type_match
                + (1.0 - self.CONTENT_TYPE_WEIGHT) * empty_score
            )
            return score, {
                "content_type_match": r_type_match,
                "content_geometry_validity": empty_score,
                "content_compactness": empty_score,
                "content_relation": empty_score,
            }

        normalized = self._normalize_geometry(pred_obj)
        if normalized is None or normalized.get("type") != "MultiLineString":
            score = self.CONTENT_TYPE_WEIGHT * r_type_match
            return score, {
                "content_type_match": r_type_match,
                "content_geometry_validity": 0.0,
                "content_compactness": 0.0,
                "content_relation": 0.0,
            }

        gt_normalized = self._normalize_geometry(gt_obj)
        r_geometry_validity = self._line_geometry_validity_score(normalized)
        fragment_score = self._road_relative_fragment_score(normalized, gt_normalized)
        junctions = self._parse_junctions(pred_obj)
        junction_score = self._road_junction_structure_score(pred_obj, normalized, gt_obj, junctions)
        score = (
            self.CONTENT_TYPE_WEIGHT * r_type_match
            + self.OPEN_CONTENT_VALIDITY_WEIGHT * r_geometry_validity
            + self.OPEN_CONTENT_COMPACTNESS_WEIGHT * fragment_score
            + self.OPEN_CONTENT_RELATION_WEIGHT * junction_score
        )
        return score, {
            "content_type_match": r_type_match,
            "content_geometry_validity": r_geometry_validity,
            "content_compactness": fragment_score,
            "content_relation": junction_score,
        }

    def _vector_execution_fidelity(self, pred_obj: Optional[Dict], gt_obj: Dict) -> float:
        score, _ = self._vector_execution_fidelity_breakdown(pred_obj, gt_obj)
        return score

    def _vector_execution_fidelity_breakdown(
        self, pred_obj: Optional[Dict], gt_obj: Dict
    ) -> Tuple[float, Dict[str, float]]:
        gt_cls = self._extract_class_name(gt_obj)
        if gt_cls in self.CLOSED_CLASSES:
            return self._closed_execution_fidelity_breakdown(pred_obj, gt_obj)
        if gt_cls in self.OPEN_CLASSES:
            return self._open_execution_fidelity_breakdown(pred_obj, gt_obj)
        return 0.0, {
            "fidelity_iou": 0.0,
            "fidelity_iou_raw": 0.0,
            "fidelity_alignment": 0.0,
            "fidelity_alignment_raw": 0.0,
            "fidelity_alignment_precision": 0.0,
            "fidelity_alignment_recall": 0.0,
            "fidelity_connectivity": 0.0,
            "fidelity_graph_node": 0.0,
            "fidelity_graph_edge": 0.0,
            "fidelity_graph_component": 0.0,
        }

    def _closed_execution_fidelity(self, pred_obj: Optional[Dict], gt_obj: Dict) -> float:
        score, _ = self._closed_execution_fidelity_breakdown(pred_obj, gt_obj)
        return score

    def _closed_execution_fidelity_breakdown(
        self, pred_obj: Optional[Dict], gt_obj: Dict
    ) -> Tuple[float, Dict[str, float]]:
        raw_iou = self._geometry_iou_raw(pred_obj, gt_obj)
        r_iou = self._shape_iou_score(raw_iou, gt_obj)
        raw_alignment = self._boundary_overlap_score_raw(pred_obj, gt_obj)
        r_alignment = self._shape_alignment_score(raw_alignment)
        score = self.CLOSED_IOU_WEIGHT * r_iou + self.CLOSED_ALIGNMENT_WEIGHT * r_alignment
        return score, {
            "fidelity_iou": r_iou,
            "fidelity_iou_raw": raw_iou,
            "fidelity_alignment": r_alignment,
            "fidelity_alignment_raw": raw_alignment,
            "fidelity_alignment_precision": raw_alignment,
            "fidelity_alignment_recall": raw_alignment,
            "fidelity_connectivity": 0.0,
            "fidelity_graph_node": 0.0,
            "fidelity_graph_edge": 0.0,
            "fidelity_graph_component": 0.0,
        }

    def _open_execution_fidelity(self, pred_obj: Optional[Dict], gt_obj: Dict) -> float:
        score, _ = self._open_execution_fidelity_breakdown(pred_obj, gt_obj)
        return score

    def _open_execution_fidelity_breakdown(
        self, pred_obj: Optional[Dict], gt_obj: Dict
    ) -> Tuple[float, Dict[str, float]]:
        if self._is_empty_road_feature(gt_obj):
            empty_score = self._empty_road_match_score(pred_obj)
            return empty_score, {
                "fidelity_iou": empty_score,
                "fidelity_iou_raw": empty_score,
                "fidelity_alignment": empty_score,
                "fidelity_alignment_raw": empty_score,
                "fidelity_alignment_precision": empty_score,
                "fidelity_alignment_recall": empty_score,
                "fidelity_connectivity": empty_score,
                "fidelity_graph_node": 0.0,
                "fidelity_graph_edge": 0.0,
                "fidelity_graph_component": 0.0,
            }

        raw_iou = self._geometry_iou_raw(pred_obj, gt_obj)
        r_iou = self._shape_iou_score(raw_iou, gt_obj)
        raw_alignment, alignment_parts = self._road_centerline_f1_score_breakdown(pred_obj, gt_obj)
        r_alignment = self._shape_alignment_score(raw_alignment)
        r_connectivity, connectivity_parts = self._road_connectivity_score_breakdown(pred_obj, gt_obj)
        score = (
            self.OPEN_IOU_WEIGHT * r_iou
            + self.OPEN_ALIGNMENT_WEIGHT * r_alignment
            + self.OPEN_CONNECTIVITY_WEIGHT * r_connectivity
        )
        parts = {
            "fidelity_iou": r_iou,
            "fidelity_iou_raw": raw_iou,
            "fidelity_alignment": r_alignment,
            "fidelity_alignment_raw": raw_alignment,
            "fidelity_alignment_precision": alignment_parts["fidelity_alignment_precision"],
            "fidelity_alignment_recall": alignment_parts["fidelity_alignment_recall"],
            "fidelity_connectivity": r_connectivity,
            "fidelity_graph_node": 0.0,
            "fidelity_graph_edge": 0.0,
            "fidelity_graph_component": 0.0,
        }
        parts.update(connectivity_parts)
        return score, parts

    def _closed_alignment_score(self, pred_obj: Optional[Dict], gt_obj: Dict) -> float:
        raw_score = self._boundary_overlap_score_raw(pred_obj, gt_obj)
        return self._shape_alignment_score(raw_score)

    def _open_alignment_score(self, pred_obj: Optional[Dict], gt_obj: Dict) -> float:
        score, _ = self._open_alignment_score_breakdown(pred_obj, gt_obj)
        return score

    def _open_alignment_score_breakdown(
        self, pred_obj: Optional[Dict], gt_obj: Dict
    ) -> Tuple[float, Dict[str, float]]:
        gt_cls = self._extract_class_name(gt_obj)
        if gt_cls in self.OPEN_CLASSES:
            raw_score, parts = self._road_centerline_f1_score_breakdown(pred_obj, gt_obj)
            return self._shape_alignment_score(raw_score), parts

        score = self._closed_alignment_score(pred_obj, gt_obj)
        return score, {
            "fidelity_alignment_precision": score,
            "fidelity_alignment_recall": score,
        }

    def _shape_alignment_score(self, raw_score: float) -> float:
        score = max(0.0, min(1.0, float(raw_score)))
        return max(0.0, min(1.0, score ** self.ALIGN_QUADRATIC_POWER))

    def _shape_iou_score(self, raw_score: float, gt_obj: Optional[Dict] = None) -> float:
        score = max(0.0, min(1.0, float(raw_score)))
        threshold = self.CLOSED_IOU_THRESHOLD
        gt_cls = self._extract_class_name(gt_obj)
        if gt_cls in self.OPEN_CLASSES:
            threshold = self.OPEN_IOU_THRESHOLD

        if score < threshold:
            return self.IOU_LOW_SCORE

        denom = max(1e-6, 1.0 - threshold)
        normalized = (score - threshold) / denom
        shaped = self.IOU_LOW_SCORE + (1.0 - self.IOU_LOW_SCORE) * (normalized ** self.IOU_QUADRATIC_POWER)
        return max(0.0, min(1.0, shaped))

    def _normalize_geometry(self, feature: Optional[Dict]) -> Optional[Dict]:
        if not isinstance(feature, dict):
            return None

        geom = feature.get("geometry")
        if not isinstance(geom, dict):
            return None

        gtype = geom.get("type")
        coords = geom.get("coordinates")
        if not isinstance(gtype, str):
            return None

        gtype_l = gtype.strip().lower()
        if gtype_l == "polygon":
            normalized_coords = self._to_polygon_coords(coords)
            if normalized_coords is None:
                return None
            return {"type": "Polygon", "coordinates": normalized_coords}
        if gtype_l == "multilinestring":
            normalized_coords = self._to_multilinestring_coords(coords)
            if normalized_coords is None:
                return None
            return {"type": "MultiLineString", "coordinates": normalized_coords}
        return None

    def _to_polygon_coords(self, coords) -> Optional[List[List[List[float]]]]:
        if isinstance(coords, list) and coords and all(isinstance(v, (int, float)) for v in coords):
            if len(coords) < 6 or len(coords) % 2 != 0:
                return None
            ring = [[float(coords[i]), float(coords[i + 1])] for i in range(0, len(coords), 2)]
            ring = self._close_ring(ring)
            return [ring]

        if isinstance(coords, list) and coords and all(
            isinstance(pt, list) and len(pt) >= 2 and all(isinstance(v, (int, float)) for v in pt[:2])
            for pt in coords
        ):
            ring = [[float(pt[0]), float(pt[1])] for pt in coords]
            if len(ring) < 3:
                return None
            ring = self._close_ring(ring)
            return [ring]

        if isinstance(coords, list) and coords and isinstance(coords[0], list):
            first = coords[0]
            if isinstance(first, list) and first and all(isinstance(pt, list) and len(pt) >= 2 for pt in first):
                rings = []
                for ring in coords:
                    if not isinstance(ring, list) or len(ring) < 3:
                        return None
                    new_ring = []
                    for pt in ring:
                        if not (
                            isinstance(pt, list)
                            and len(pt) >= 2
                            and isinstance(pt[0], (int, float))
                            and isinstance(pt[1], (int, float))
                        ):
                            return None
                        new_ring.append([float(pt[0]), float(pt[1])])
                    rings.append(self._close_ring(new_ring))
                return rings
        return None

    def _to_multilinestring_coords(self, coords) -> Optional[List[List[List[float]]]]:
        if isinstance(coords, list) and len(coords) == 0:
            return []

        if isinstance(coords, list) and coords and all(isinstance(v, (int, float)) for v in coords):
            if len(coords) < 4 or len(coords) % 2 != 0:
                return None
            line = [[float(coords[i]), float(coords[i + 1])] for i in range(0, len(coords), 2)]
            return [line]

        if isinstance(coords, list) and coords and all(
            isinstance(pt, list) and len(pt) >= 2 and all(isinstance(v, (int, float)) for v in pt[:2])
            for pt in coords
        ):
            line = [[float(pt[0]), float(pt[1])] for pt in coords]
            if len(line) < 2:
                return None
            return [line]

        if isinstance(coords, list) and coords:
            lines: List[List[List[float]]] = []
            for line in coords:
                if isinstance(line, list) and line and all(isinstance(v, (int, float)) for v in line):
                    if len(line) < 4 or len(line) % 2 != 0:
                        return None
                    lines.append([[float(line[i]), float(line[i + 1])] for i in range(0, len(line), 2)])
                    continue

                if isinstance(line, list) and line and all(
                    isinstance(pt, list) and len(pt) >= 2 and all(isinstance(v, (int, float)) for v in pt[:2])
                    for pt in line
                ):
                    new_line = [[float(pt[0]), float(pt[1])] for pt in line]
                    if len(new_line) < 2:
                        return None
                    lines.append(new_line)
                    continue

                return None

            return lines if lines else None
        return None

    def _parse_junctions(self, feature: Optional[Dict]) -> Optional[List[List[float]]]:
        if not isinstance(feature, dict):
            return None

        props = feature.get("properties")
        if not isinstance(props, dict):
            return None

        junctions = props.get("junction")
        if junctions is None:
            return None
        if not isinstance(junctions, list):
            return None

        out: List[List[float]] = []
        for pt in junctions:
            if not (
                isinstance(pt, list)
                and len(pt) >= 2
                and isinstance(pt[0], (int, float))
                and isinstance(pt[1], (int, float))
            ):
                return None
            out.append([float(pt[0]), float(pt[1])])
        return out

    def _close_ring(self, ring: List[List[float]]) -> List[List[float]]:
        if len(ring) < 3:
            return ring
        if ring[0] != ring[-1]:
            ring = ring + [ring[0]]
        return ring

    def _polygon_geometry_validity_score(self, geom: Dict) -> float:
        shape_obj = self._raw_shape_from_geom(geom)
        if shape_obj is None or shape_obj.is_empty:
            return 0.0
        if not shape_obj.is_valid:
            return 0.0
        return 1.0 if shape_obj.area > 1e-6 else 0.0

    def _vertex_budget_score(self, geom: Dict) -> float:
        try:
            ring = geom["coordinates"][0]
            n = max(0, len(ring) - 1)
        except Exception:
            return 0.0

        if n < 3:
            return 0.0
        if n <= self.MAX_POLYGON_VERTICES:
            return 1.0
        excess = n - self.MAX_POLYGON_VERTICES
        return max(0.0, 1.0 - excess / max(1.0, float(self.MAX_POLYGON_VERTICES)))

    def _line_geometry_validity_score(self, geom: Dict) -> float:
        if geom.get("type") == "MultiLineString" and geom.get("coordinates") == []:
            return 1.0
        shape_obj = self._raw_shape_from_geom(geom)
        if shape_obj is None or shape_obj.is_empty:
            return 0.0
        if not shape_obj.is_valid:
            return 0.0
        return 1.0 if shape_obj.length > 1e-6 else 0.0

    def _road_relative_fragment_score(self, pred_geom: Dict, gt_geom: Optional[Dict]) -> float:
        try:
            pred_count = len(pred_geom["coordinates"])
        except Exception:
            return 0.0

        if pred_count <= 0:
            return 0.0

        if gt_geom is None:
            return 1.0

        try:
            gt_count = len(gt_geom["coordinates"])
        except Exception:
            return 0.0

        if gt_count <= 0:
            return 1.0

        if pred_count <= gt_count:
            return 1.0
        return gt_count / float(pred_count)

    def _empty_road_match_score(self, pred_obj: Optional[Dict]) -> float:
        if self._is_empty_road_feature(pred_obj):
            return 1.0

        normalized = self._normalize_geometry(pred_obj)
        if normalized is None or normalized.get("type") != "MultiLineString":
            return self.EMPTY_ROAD_FALSE_POSITIVE_PENALTY

        coords = normalized.get("coordinates")
        if isinstance(coords, list) and len(coords) == 0:
            return 1.0

        shape_obj = self._shape_from_geom(normalized)
        if shape_obj is None or shape_obj.is_empty:
            return 1.0

        try:
            total_len = float(shape_obj.length)
        except Exception:
            return self.EMPTY_ROAD_FALSE_POSITIVE_PENALTY
        if total_len <= 1e-6:
            return 1.0
        return self.EMPTY_ROAD_FALSE_POSITIVE_PENALTY

    def _road_junction_structure_score(
        self,
        pred_obj: Optional[Dict],
        geom: Dict,
        gt_obj: Dict,
        junctions: Optional[List[List[float]]],
    ) -> float:
        gt_junctions = self._parse_junctions(gt_obj)
        gt_has_junction = bool(gt_junctions)

        if junctions is None:
            return (
                self.EMPTY_ROAD_MISSING_JUNCTION_SCORE
                if not gt_has_junction
                else 0.10
            )

        if not junctions:
            return (
                self.EMPTY_ROAD_EXPLICIT_EMPTY_JUNCTION_SCORE
                if not gt_has_junction
                else 0.15
            )

        attach_score = self._junction_attach_score(geom, junctions)
        match_score = self._junction_match_score(pred_obj, gt_obj)
        if not gt_has_junction:
            return 0.20 * attach_score
        return 0.35 * attach_score + 0.65 * match_score

    def _junction_attach_score(self, geom: Dict, junctions: Optional[List[List[float]]]) -> float:
        if junctions is None or not junctions:
            return 0.0

        shape_obj = self._shape_from_geom(geom)
        if shape_obj is None or shape_obj.is_empty:
            return 0.0

        try:
            from shapely.geometry import Point
        except Exception:
            return 0.0

        attached = 0
        for pt in junctions:
            dist = shape_obj.distance(Point(pt[0], pt[1]))
            if dist <= self.ROAD_ATTACH_TOL:
                attached += 1
        return attached / len(junctions)

    def _shape_from_geom(self, geom: Dict):
        try:
            from shapely.geometry import shape
            from shapely.ops import unary_union
            try:
                from shapely.validation import make_valid
            except Exception:
                make_valid = None
        except Exception:
            return None

        def _safe_make_valid(g):
            if g is None or g.is_empty:
                return g
            if g.is_valid:
                return g
            if make_valid is not None:
                try:
                    fixed = make_valid(g)
                    if fixed is not None and not fixed.is_empty:
                        return fixed
                except Exception:
                    pass
            try:
                fixed = g.buffer(0)
                if fixed is not None and not fixed.is_empty:
                    return fixed
            except Exception:
                pass
            try:
                fixed = unary_union(g)
                if fixed is not None and not fixed.is_empty:
                    return fixed
            except Exception:
                pass
            return g

        try:
            g = shape(geom)
        except Exception:
            return None
        return _safe_make_valid(g)

    def _raw_shape_from_geom(self, geom: Dict):
        try:
            from shapely.geometry import shape
        except Exception:
            return None

        try:
            return shape(geom)
        except Exception:
            return None

    def _geometry_iou(self, pred_obj: Optional[Dict], gt_obj: Dict) -> float:
        raw_score = self._geometry_iou_raw(pred_obj, gt_obj)
        return self._shape_iou_score(raw_score, gt_obj)

    def _geometry_iou_raw(self, pred_obj: Optional[Dict], gt_obj: Dict) -> float:
        pred_geom = self._normalize_geometry(pred_obj)
        gt_geom = self._normalize_geometry(gt_obj)
        if pred_geom is None or gt_geom is None:
            return 0.0

        if self._is_empty_road_feature(gt_obj):
            return self._empty_road_match_score(pred_obj)

        pg = self._shape_from_geom(pred_geom)
        gg = self._shape_from_geom(gt_geom)
        if pg is None or gg is None or pg.is_empty or gg.is_empty:
            return 0.0

        pred_cls = self._extract_class_name(pred_obj)
        gt_cls = self._extract_class_name(gt_obj)
        if pred_cls in self.OPEN_CLASSES and gt_cls in self.OPEN_CLASSES:
            try:
                pg = pg.buffer(self.ROAD_BUFFER_DISTANCE)
                gg = gg.buffer(self.ROAD_BUFFER_DISTANCE)
            except Exception:
                return 0.0
            pg = self._shape_from_geom(pg.__geo_interface__)
            gg = self._shape_from_geom(gg.__geo_interface__)
            if pg is None or gg is None or pg.is_empty or gg.is_empty:
                return 0.0

        try:
            inter = pg.intersection(gg).area
            union = pg.union(gg).area
        except Exception:
            return 0.0
        if union <= 0:
            return 0.0
        return float(inter / (union + self.IOU_EPS))

    def _boundary_alignment_score(self, pred_obj: Optional[Dict], gt_obj: Dict) -> float:
        pred_geom = self._normalize_geometry(pred_obj)
        gt_geom = self._normalize_geometry(gt_obj)
        if pred_geom is None or gt_geom is None:
            return 0.0

        pg = self._shape_from_geom(pred_geom)
        gg = self._shape_from_geom(gt_geom)
        if pg is None or gg is None or pg.is_empty or gg.is_empty:
            return 0.0

        try:
            dist = pg.boundary.hausdorff_distance(gg.boundary)
        except Exception:
            return 0.0

        bbox = self._bbox_from_geom(gt_geom)
        if bbox is None:
            return 0.0
        scale = math.hypot(max(1e-6, bbox[2] - bbox[0]), max(1e-6, bbox[3] - bbox[1]))
        if scale <= 0:
            return 0.0
        return max(0.0, 1.0 - dist / (0.25 * scale))

    def _boundary_overlap_score(self, pred_obj: Optional[Dict], gt_obj: Dict) -> float:
        raw_score = self._boundary_overlap_score_raw(pred_obj, gt_obj)
        return self._shape_alignment_score(raw_score)

    def _boundary_overlap_score_raw(self, pred_obj: Optional[Dict], gt_obj: Dict) -> float:
        pred_geom = self._normalize_geometry(pred_obj)
        gt_geom = self._normalize_geometry(gt_obj)
        if pred_geom is None or gt_geom is None:
            return 0.0

        pg = self._shape_from_geom(pred_geom)
        gg = self._shape_from_geom(gt_geom)
        if pg is None or gg is None or pg.is_empty or gg.is_empty:
            return 0.0

        bbox = self._bbox_from_geom(gt_geom)
        if bbox is None:
            return 0.0
        scale = math.hypot(max(1e-6, bbox[2] - bbox[0]), max(1e-6, bbox[3] - bbox[1]))
        tol = max(1e-3, self.BOUNDARY_BUFFER_RATIO * scale)
        try:
            pred_band = pg.boundary.buffer(tol)
            gt_band = gg.boundary.buffer(tol)
            inter = pred_band.intersection(gt_band).area
            union = pred_band.union(gt_band).area
        except Exception:
            return 0.0
        if union <= 0:
            return 0.0
        return max(0.0, min(1.0, float(inter / (union + self.IOU_EPS))))

    def _junction_match_score(self, pred_obj: Optional[Dict], gt_obj: Dict) -> float:
        pred_j = self._parse_junctions(pred_obj)
        gt_j = self._parse_junctions(gt_obj)
        if self._is_empty_road_feature(gt_obj):
            if pred_j is None:
                return self.EMPTY_ROAD_MISSING_JUNCTION_SCORE
            return 1.0 if not pred_j else 0.0
        if gt_j is None and pred_j is None:
            return 1.0
        if gt_j is None or pred_j is None:
            return 0.0
        if not gt_j and not pred_j:
            return 1.0
        if not gt_j or not pred_j:
            return 0.0

        remaining = pred_j[:]
        matched = 0
        for g in gt_j:
            best_idx = -1
            best_dist = None
            for i, p in enumerate(remaining):
                d = math.hypot(g[0] - p[0], g[1] - p[1])
                if best_dist is None or d < best_dist:
                    best_dist = d
                    best_idx = i
            if best_dist is not None and best_dist <= self.JUNCTION_MATCH_TOL:
                matched += 1
                remaining.pop(best_idx)

        denom = max(len(gt_j), len(pred_j))
        return matched / denom if denom > 0 else 0.0

    def _road_centerline_alignment_score(self, pred_obj: Optional[Dict], gt_obj: Dict) -> float:
        score, _ = self._road_centerline_f1_score_breakdown(pred_obj, gt_obj)
        return self._shape_alignment_score(score)

    def _road_centerline_f1_score_breakdown(
        self, pred_obj: Optional[Dict], gt_obj: Dict
    ) -> Tuple[float, Dict[str, float]]:
        pred_geom = self._normalize_geometry(pred_obj)
        gt_geom = self._normalize_geometry(gt_obj)
        if pred_geom is None or gt_geom is None:
            return 0.0, {
                "fidelity_alignment_precision": 0.0,
                "fidelity_alignment_recall": 0.0,
            }

        if self._is_empty_road_feature(gt_obj):
            score = self._empty_road_match_score(pred_obj)
            return score, {
                "fidelity_alignment_precision": score,
                "fidelity_alignment_recall": score,
            }

        pg = self._shape_from_geom(pred_geom)
        gg = self._shape_from_geom(gt_geom)
        if pg is None or gg is None or pg.is_empty or gg.is_empty:
            return 0.0, {
                "fidelity_alignment_precision": 0.0,
                "fidelity_alignment_recall": 0.0,
            }

        try:
            precision = pg.intersection(gg.buffer(self.ROAD_ALIGNMENT_BUFFER)).length / max(pg.length, 1e-6)
            recall = gg.intersection(pg.buffer(self.ROAD_ALIGNMENT_BUFFER)).length / max(gg.length, 1e-6)
        except Exception:
            return 0.0, {
                "fidelity_alignment_precision": 0.0,
                "fidelity_alignment_recall": 0.0,
            }

        precision = max(0.0, min(1.0, float(precision)))
        recall = max(0.0, min(1.0, float(recall)))
        if precision + recall <= 1e-8:
            f1 = 0.0
        else:
            f1 = 2.0 * precision * recall / (precision + recall)

        return max(0.0, min(1.0, f1)), {
            "fidelity_alignment_precision": precision,
            "fidelity_alignment_recall": recall,
        }

    def _road_connectivity_score(self, pred_obj: Optional[Dict], gt_obj: Dict) -> float:
        score, _ = self._road_connectivity_score_breakdown(pred_obj, gt_obj)
        return score

    def _road_connectivity_score_breakdown(
        self, pred_obj: Optional[Dict], gt_obj: Dict
    ) -> Tuple[float, Dict[str, float]]:
        mode = str(self.ROAD_CONNECTIVITY_MODE).strip().lower()
        if mode == "graph":
            score, node_score, edge_score, component_score = self._road_connectivity_graph_score_breakdown(
                pred_obj, gt_obj
            )
            return score, {
                "fidelity_graph_node": node_score,
                "fidelity_graph_edge": edge_score,
                "fidelity_graph_component": component_score,
            }
        score = self._road_connectivity_light_score(pred_obj, gt_obj)
        return score, {
            "fidelity_graph_node": 0.0,
            "fidelity_graph_edge": 0.0,
            "fidelity_graph_component": 0.0,
        }

    def _road_connectivity_light_score(self, pred_obj: Optional[Dict], gt_obj: Dict) -> float:
        pred_geom = self._normalize_geometry(pred_obj)
        gt_geom = self._normalize_geometry(gt_obj)
        if pred_geom is None or gt_geom is None:
            return 0.0

        pred_count = self._component_count(pred_geom, buffer_distance=self.ROAD_BUFFER_DISTANCE)
        gt_count = self._component_count(gt_geom, buffer_distance=self.ROAD_BUFFER_DISTANCE)
        if pred_count is None or gt_count is None:
            return 0.0
        return 1.0 / (1.0 + abs(pred_count - gt_count))

    def _road_connectivity_graph_score(self, pred_obj: Optional[Dict], gt_obj: Dict) -> float:
        score, _, _, _ = self._road_connectivity_graph_score_breakdown(pred_obj, gt_obj)
        return score

    def _road_connectivity_graph_score_breakdown(
        self, pred_obj: Optional[Dict], gt_obj: Dict
    ) -> Tuple[float, float, float, float]:
        pred_graph = self._build_road_graph(pred_obj)
        gt_graph = self._build_road_graph(gt_obj)
        if pred_graph is None or gt_graph is None:
            return 0.0, 0.0, 0.0, 0.0

        node_score, pred_to_gt = self._road_graph_node_score(pred_graph, gt_graph)
        edge_score = self._road_graph_edge_score(pred_graph, gt_graph, pred_to_gt)
        component_score = 1.0 / (
            1.0 + abs(int(pred_graph["components"]) - int(gt_graph["components"]))
        )

        score = (
            self.ROAD_GRAPH_NODE_WEIGHT * node_score
            + self.ROAD_GRAPH_EDGE_WEIGHT * edge_score
            + self.ROAD_GRAPH_COMPONENT_WEIGHT * component_score
        )
        return score, node_score, edge_score, component_score

    def _component_count(self, geom: Dict, buffer_distance: float) -> Optional[int]:
        shape_obj = self._shape_from_geom(geom)
        if shape_obj is None or shape_obj.is_empty:
            return None
        try:
            buffered = shape_obj.buffer(buffer_distance)
        except Exception:
            return None
        if buffered.is_empty:
            return None
        if hasattr(buffered, "geoms"):
            return len(list(buffered.geoms))
        return 1

    def _build_road_graph(self, feature: Optional[Dict]) -> Optional[Dict]:
        normalized = self._normalize_geometry(feature)
        if normalized is None or normalized.get("type") != "MultiLineString":
            return None

        lines = normalized.get("coordinates")
        if not isinstance(lines, list) or not lines:
            return None

        junctions = self._parse_junctions(feature) or []
        nodes: List[List[float]] = []
        anchors_by_line: List[List[Tuple[float, int]]] = [[] for _ in lines]

        def register_node(point: List[float]) -> int:
            return self._merge_point_into_nodes(nodes, point, self.ROAD_NODE_MERGE_TOL)

        line_geoms = self._build_linestrings(lines)
        if line_geoms is None:
            return None

        for line_idx, line in enumerate(lines):
            total_len = self._polyline_length(line)
            if total_len <= self.ROAD_MIN_EDGE_LENGTH:
                continue
            start_id = register_node(line[0])
            end_id = register_node(line[-1])
            anchors_by_line[line_idx].append((0.0, start_id))
            anchors_by_line[line_idx].append((total_len, end_id))

        for pt in junctions:
            attached = False
            for line_idx, line in enumerate(lines):
                dist, offset = self._project_point_to_polyline(pt, line)
                if dist <= self.ROAD_ATTACH_TOL:
                    node_id = register_node(pt)
                    anchors_by_line[line_idx].append((offset, node_id))
                    attached = True
            if not attached:
                continue

        for line_i in range(len(line_geoms)):
            for line_j in range(line_i + 1, len(line_geoms)):
                inter_geom = line_geoms[line_i].intersection(line_geoms[line_j])
                inter_points = self._extract_points_from_geometry(inter_geom)
                for pt in inter_points:
                    node_id = register_node(pt)
                    dist_i, offset_i = self._project_point_to_polyline(pt, lines[line_i])
                    dist_j, offset_j = self._project_point_to_polyline(pt, lines[line_j])
                    if dist_i <= self.ROAD_ATTACH_TOL:
                        anchors_by_line[line_i].append((offset_i, node_id))
                    if dist_j <= self.ROAD_ATTACH_TOL:
                        anchors_by_line[line_j].append((offset_j, node_id))

        edges: List[Dict] = []
        for line_idx, line in enumerate(lines):
            anchors = self._dedupe_sorted_anchors(anchors_by_line[line_idx])
            if len(anchors) < 2:
                continue
            for i in range(1, len(anchors)):
                prev_offset, prev_node = anchors[i - 1]
                curr_offset, curr_node = anchors[i]
                if prev_node == curr_node:
                    continue
                edge_length = max(0.0, curr_offset - prev_offset)
                if edge_length <= self.ROAD_MIN_EDGE_LENGTH:
                    continue
                edges.append({"u": prev_node, "v": curr_node, "length": edge_length, "line_idx": line_idx})

        if not nodes or not edges:
            return {
                "nodes": nodes,
                "edges": edges,
                "degrees": [0 for _ in nodes],
                "components": len(nodes),
            }

        adjacency = [set() for _ in nodes]
        for edge in edges:
            u = int(edge["u"])
            v = int(edge["v"])
            if u == v:
                continue
            adjacency[u].add(v)
            adjacency[v].add(u)

        degrees = [len(nei) for nei in adjacency]
        components = self._count_graph_components(adjacency)
        return {
            "nodes": nodes,
            "edges": edges,
            "degrees": degrees,
            "components": components,
        }

    def _build_linestrings(self, lines: List[List[List[float]]]):
        try:
            from shapely.geometry import LineString
        except Exception:
            return None

        out = []
        try:
            for line in lines:
                if not isinstance(line, list) or len(line) < 2:
                    return None
                out.append(LineString(line))
        except Exception:
            return None
        return out

    def _merge_point_into_nodes(self, nodes: List[List[float]], point: List[float], tol: float) -> int:
        for idx, existing in enumerate(nodes):
            if math.hypot(existing[0] - point[0], existing[1] - point[1]) <= tol:
                return idx
        nodes.append([float(point[0]), float(point[1])])
        return len(nodes) - 1

    def _polyline_length(self, line: List[List[float]]) -> float:
        total = 0.0
        for i in range(1, len(line)):
            total += math.hypot(line[i][0] - line[i - 1][0], line[i][1] - line[i - 1][1])
        return total

    def _project_point_to_polyline(self, point: List[float], line: List[List[float]]) -> Tuple[float, float]:
        best_dist = float("inf")
        best_offset = 0.0
        cumulative = 0.0

        for i in range(1, len(line)):
            x1, y1 = line[i - 1]
            x2, y2 = line[i]
            seg_dx = x2 - x1
            seg_dy = y2 - y1
            seg_len2 = seg_dx * seg_dx + seg_dy * seg_dy
            seg_len = math.sqrt(seg_len2)
            if seg_len2 <= 1e-12:
                continue

            t = ((point[0] - x1) * seg_dx + (point[1] - y1) * seg_dy) / seg_len2
            t = max(0.0, min(1.0, t))
            proj_x = x1 + t * seg_dx
            proj_y = y1 + t * seg_dy
            dist = math.hypot(point[0] - proj_x, point[1] - proj_y)
            if dist < best_dist:
                best_dist = dist
                best_offset = cumulative + t * seg_len
            cumulative += seg_len

        if best_dist == float("inf"):
            return float("inf"), 0.0
        return best_dist, best_offset

    def _dedupe_sorted_anchors(self, anchors: List[Tuple[float, int]]) -> List[Tuple[float, int]]:
        if not anchors:
            return []

        anchors = sorted(anchors, key=lambda x: (x[0], x[1]))
        deduped: List[Tuple[float, int]] = []
        for offset, node_id in anchors:
            if not deduped:
                deduped.append((offset, node_id))
                continue

            prev_offset, prev_node = deduped[-1]
            if node_id == prev_node and abs(offset - prev_offset) <= self.ROAD_MIN_EDGE_LENGTH:
                continue
            if abs(offset - prev_offset) <= self.ROAD_MIN_EDGE_LENGTH:
                keep_offset = min(prev_offset, offset)
                deduped[-1] = (keep_offset, prev_node)
                continue
            deduped.append((offset, node_id))
        return deduped

    def _extract_points_from_geometry(self, geom) -> List[List[float]]:
        points: List[List[float]] = []
        if geom is None or geom.is_empty:
            return points

        geom_type = getattr(geom, "geom_type", "")
        if geom_type == "Point":
            x, y = geom.coords[0]
            return [[float(x), float(y)]]
        if geom_type == "MultiPoint":
            for sub in geom.geoms:
                points.extend(self._extract_points_from_geometry(sub))
            return points
        if geom_type in {"LineString", "LinearRing"}:
            coords = list(geom.coords)
            if coords:
                x1, y1 = coords[0]
                x2, y2 = coords[-1]
                points.append([float(x1), float(y1)])
                if len(coords) > 1:
                    points.append([float(x2), float(y2)])
            return points
        if hasattr(geom, "geoms"):
            for sub in geom.geoms:
                points.extend(self._extract_points_from_geometry(sub))
        return points

    def _count_graph_components(self, adjacency: List[set]) -> int:
        visited = set()
        components = 0
        for start in range(len(adjacency)):
            if start in visited:
                continue
            components += 1
            stack = [start]
            visited.add(start)
            while stack:
                node = stack.pop()
                for nei in adjacency[node]:
                    if nei not in visited:
                        visited.add(nei)
                        stack.append(nei)
        return components

    def _road_graph_node_score(self, pred_graph: Dict, gt_graph: Dict) -> Tuple[float, Dict[int, int]]:
        pred_nodes = pred_graph.get("nodes", [])
        gt_nodes = gt_graph.get("nodes", [])
        pred_degrees = pred_graph.get("degrees", [])
        gt_degrees = gt_graph.get("degrees", [])

        if not pred_nodes and not gt_nodes:
            return 1.0, {}
        if not pred_nodes or not gt_nodes:
            return 0.0, {}

        pairs: List[Tuple[float, int, int]] = []
        for i, p in enumerate(pred_nodes):
            for j, g in enumerate(gt_nodes):
                dist = math.hypot(p[0] - g[0], p[1] - g[1])
                if dist <= self.ROAD_NODE_MATCH_TOL:
                    pairs.append((dist, i, j))
        pairs.sort(key=lambda x: x[0])

        used_pred = set()
        used_gt = set()
        pred_to_gt: Dict[int, int] = {}
        total = 0.0
        for dist, pred_idx, gt_idx in pairs:
            if pred_idx in used_pred or gt_idx in used_gt:
                continue
            used_pred.add(pred_idx)
            used_gt.add(gt_idx)
            pred_to_gt[pred_idx] = gt_idx

            spatial_score = max(0.0, 1.0 - dist / max(self.ROAD_NODE_MATCH_TOL, 1e-6))
            degree_gap = abs(int(pred_degrees[pred_idx]) - int(gt_degrees[gt_idx]))
            degree_score = 1.0 / (1.0 + degree_gap)
            total += 0.5 * spatial_score + 0.5 * degree_score

        denom = max(len(pred_nodes), len(gt_nodes))
        return total / denom if denom > 0 else 0.0, pred_to_gt

    def _road_graph_edge_score(self, pred_graph: Dict, gt_graph: Dict, pred_to_gt: Dict[int, int]) -> float:
        pred_edges = pred_graph.get("edges", [])
        gt_edges = gt_graph.get("edges", [])
        if not pred_edges and not gt_edges:
            return 1.0
        if not pred_edges or not gt_edges:
            return 0.0

        gt_edge_buckets: Dict[Tuple[int, int], List[float]] = {}
        for edge in gt_edges:
            key = tuple(sorted((int(edge["u"]), int(edge["v"]))))
            gt_edge_buckets.setdefault(key, []).append(float(edge["length"]))

        for lengths in gt_edge_buckets.values():
            lengths.sort()

        matched_quality = 0.0
        for edge in pred_edges:
            u = int(edge["u"])
            v = int(edge["v"])
            if u not in pred_to_gt or v not in pred_to_gt:
                continue

            gt_key = tuple(sorted((pred_to_gt[u], pred_to_gt[v])))
            candidate_lengths = gt_edge_buckets.get(gt_key)
            if not candidate_lengths:
                continue

            pred_len = float(edge["length"])
            best_idx = -1
            best_score = -1.0
            for idx, gt_len in enumerate(candidate_lengths):
                ratio = min(pred_len, gt_len) / max(pred_len, gt_len, 1e-6)
                if ratio > best_score:
                    best_score = ratio
                    best_idx = idx

            if best_idx >= 0:
                matched_quality += max(0.0, best_score)
                candidate_lengths.pop(best_idx)

        denom = max(len(pred_edges), len(gt_edges))
        return matched_quality / denom if denom > 0 else 0.0

    def _bbox_from_geom(self, geom: Dict) -> Optional[Tuple[float, float, float, float]]:
        try:
            points: List[List[float]] = []
            gtype = geom.get("type")
            coords = geom.get("coordinates")
            if gtype == "Polygon":
                for ring in coords:
                    for pt in ring:
                        points.append(pt)
            elif gtype == "MultiLineString":
                for line in coords:
                    for pt in line:
                        points.append(pt)
            else:
                return None
            if not points:
                return None
            xs = [p[0] for p in points]
            ys = [p[1] for p in points]
            return (min(xs), min(ys), max(xs), max(ys))
        except Exception:
            return None


class StructuredVectorLanguageValidityRewardV9(StructuredVectorLanguageRewardV9):
    SCORE_MODE = "validity"


class StructuredVectorContentRewardV9(StructuredVectorLanguageRewardV9):
    SCORE_MODE = "content"


class StructuredVectorExecutionRewardV9(StructuredVectorLanguageRewardV9):
    SCORE_MODE = "fidelity"


class StructuredVectorExecutionIoURewardV9(StructuredVectorLanguageRewardV9):
    SCORE_MODE = "fidelity_iou"


class StructuredVectorExecutionAlignmentRewardV9(StructuredVectorLanguageRewardV9):
    SCORE_MODE = "fidelity_alignment"


class StructuredVectorExecutionConnectivityRewardV9(StructuredVectorLanguageRewardV9):
    SCORE_MODE = "fidelity_connectivity"


orms["vector_language_reward_v9"] = StructuredVectorLanguageRewardV9
orms["vector_language_validity_reward_v9"] = StructuredVectorLanguageValidityRewardV9
orms["vector_content_reward_v9"] = StructuredVectorContentRewardV9
orms["vector_execution_reward_v9"] = StructuredVectorExecutionRewardV9
orms["vector_execution_iou_reward_v9"] = StructuredVectorExecutionIoURewardV9
orms["vector_execution_alignment_reward_v9"] = StructuredVectorExecutionAlignmentRewardV9
orms["vector_execution_connectivity_reward_v9"] = StructuredVectorExecutionConnectivityRewardV9
