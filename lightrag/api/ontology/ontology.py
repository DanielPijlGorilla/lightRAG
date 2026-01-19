from __future__ import annotations
from typing import Any, Dict, List


def compile_ontology(raw: dict[str, Any] | None) -> dict[str, Any]:
    if not raw or not isinstance(raw, dict):
        return {
            "name": "unknown",
            "version": "0.0.0",
            "node_types": [],
            "relation_predicates": [],
            "allowed_relations": {},
            "inverse_of": {},
            "global_constraints": [],
            "design_notes": [],
            "source_schema": raw,
        }

    # ---- helpers ----
    def _as_list(x: Any) -> List[str]:
        if x is None:
            return []
        if isinstance(x, list):
            return [str(i).strip() for i in x if i is not None and str(i).strip()]
        s = str(x).strip()
        return [s] if s else []

    def _norm_token(x: Any) -> str:
        return str(x).strip().upper()

    def _norm_type(x: Any) -> str:
        return str(x).strip().upper()

    def _first_nonempty(*vals: Any) -> Any:
        for v in vals:
            if v is None:
                continue
            if isinstance(v, str) and not v.strip():
                continue
            if isinstance(v, list) and len(v) == 0:
                continue
            return v
        return None

    include_inverses_in_predicates = bool(
        raw.get("include_inverses_in_predicates", True)
    )

    # ---- base ontology metadata ----
    onto_meta = raw.get("ontology") or {}
    name = str(onto_meta.get("name") or raw.get("name") or "unknown")
    version = str(onto_meta.get("version") or raw.get("version") or "0.0.0")

    node_types_raw = _as_list(
        _first_nonempty(
            onto_meta.get("nodeTypes"),
            onto_meta.get("node_types"),
            raw.get("nodeTypes"),
            raw.get("node_types"),
        )
    )
    node_types = sorted({_norm_type(t) for t in node_types_raw})

    # ---- compile edges ----
    allowed_relations: Dict[str, Dict[str, Any]] = {}
    inverse_of: Dict[str, str] = {}
    real_predicates: set[str] = set()

    edge_types = raw.get("edgeTypes") or raw.get("edges") or raw.get("edge_types") or []
    if not isinstance(edge_types, list):
        edge_types = []

    for edge in edge_types:
        if not isinstance(edge, dict):
            continue

        pred_raw = _first_nonempty(edge.get("name"), edge.get("predicate"), edge.get("type"))
        if pred_raw is None:
            continue
        pred = _norm_token(pred_raw)
        if not pred:
            continue

        src_types = [_norm_type(t) for t in _as_list(_first_nonempty(
            edge.get("from"),
            edge.get("src"),
            edge.get("source_types"),
            edge.get("src_types"),
            edge.get("source"),
        ))]
        tgt_types = [_norm_type(t) for t in _as_list(_first_nonempty(
            edge.get("to"),
            edge.get("tgt"),
            edge.get("target_types"),
            edge.get("tgt_types"),
            edge.get("target"),
        ))]

        inv_raw = _first_nonempty(edge.get("inverse"), edge.get("inverse_of"))
        inv = _norm_token(inv_raw) if inv_raw else None

        constraints = edge.get("constraints") if isinstance(edge.get("constraints"), dict) else {}

        allowed_relations[pred] = {
            "src_types": src_types,
            "tgt_types": tgt_types,
            "inverse": inv,
            "constraints": constraints,
            "is_placeholder": False,
        }
        real_predicates.add(pred)

        if inv:
            inverse_of[pred] = inv
            inverse_of[inv] = pred

            if inv not in allowed_relations:
                allowed_relations[inv] = {
                    "src_types": tgt_types,
                    "tgt_types": src_types,
                    "inverse": pred,
                    "constraints": {},
                    "is_placeholder": True,
                }

    # ---- derived lists ----
    if include_inverses_in_predicates:
        relation_predicates = sorted(list(allowed_relations.keys()))
    else:
        relation_predicates = sorted(list(real_predicates))

    # ---- passthrough extras ----
    global_constraints = raw.get("globalConstraints") or raw.get("global_constraints") or []
    if not isinstance(global_constraints, list):
        global_constraints = []

    design_notes = raw.get("designNotes") or raw.get("design_notes") or []
    if not isinstance(design_notes, list):
        design_notes = []

    return {
        "name": name,
        "version": version,
        "node_types": node_types,
        "relation_predicates": relation_predicates,
        "allowed_relations": allowed_relations,
        "inverse_of": inverse_of,
        "global_constraints": global_constraints,
        "design_notes": design_notes,
        "source_schema": raw,
    }
