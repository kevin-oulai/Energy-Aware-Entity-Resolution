from pathlib import Path
from typing import Dict, List, Set, Tuple

from igraph import Graph


Pair = Tuple[str, str]


def _normalize_rid(rid: str) -> str:
    rid = rid.strip()
    if rid.startswith("idx_") and not rid.startswith("idx__"):
        return rid.replace("idx_", "idx__", 1)
    return rid


def _get_ground_truth(ground_truth_file: str) -> Dict[str, List[str]]:
    matches: Dict[str, List[str]] = {}
    with open(ground_truth_file, "r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            left, right = [part.strip() for part in line.split(",", 1)]
            left = _normalize_rid(left)
            right = _normalize_rid(right)
            matches.setdefault(left, []).append(right)
    if not matches:
        raise IOError("Matches file is empty.")
    return matches


def _get_similarity_file(configuration: dict) -> str:
    if configuration.get("similarity_file"):
        return configuration["similarity_file"]
    state_cfg = configuration.get("state_management", {}) if isinstance(configuration, dict) else {}
    save_dir = state_cfg.get("predicted_match-dir", "data/predicted")
    name = state_cfg.get("predicted_match-name") or configuration.get("version_name", "test")
    output_format = configuration.get("output_format", "graphml")
    ext = ".graphml" if output_format == "graphml" else ".txt"
    return str(Path(save_dir) / f"{name}{ext}")


def _ground_truth_pairs(ground_truth_file: str) -> Tuple[Set[Pair], Set[str]]:
    matches = _get_ground_truth(ground_truth_file)
    pair_set: Set[Pair] = set()
    left_target_rids: Set[str] = set()

    for left_rid, right_rids in matches.items():
        left_target_rids.add(left_rid)
        for right_rid in right_rids:
            if left_rid != right_rid:
                pair_set.add((left_rid, right_rid))

    return pair_set, left_target_rids


def _predicted_pairs_from_graphml(similarity_file: str) -> Set[Pair]:
    if not Path(similarity_file).exists():
        raise FileNotFoundError(f"Similarity file not found: {similarity_file}")

    graph = Graph.Read_GraphML(similarity_file)
    predicted_pairs: Set[Pair] = set()
    for edge in graph.es:
        source = str(graph.vs[edge.source]["name"])
        target = str(graph.vs[edge.target]["name"])
        if source == target:
            continue
        predicted_pairs.add((source, target))

    for vertex in graph.vs:
        members = vertex["members"].split(",") if "members" in vertex.attributes() and vertex["members"] else []
        if len(members) >= 2:
            for idx, left in enumerate(members):
                for right in members[idx + 1:]:
                    if left != right:
                        predicted_pairs.add((str(left), str(right)))
    return predicted_pairs


def _compute_metrics(predicted_pairs: Set[Pair], actual_pairs: Set[Pair]) -> Dict[str, float]:
    correct = len(predicted_pairs & actual_pairs)
    total_predicted = len(predicted_pairs)
    total_actual = len(actual_pairs)
    precision = correct / total_predicted if total_predicted else 0.0
    recall = correct / total_actual if total_actual else 0.0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "correct_matches": correct,
        "total_predicted_matches": total_predicted,
        "total_true_matches": total_actual,
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
    }


def compare_ground_truth(configuration: dict) -> Dict[str, Dict[str, float]]:
    ground_truth_file = configuration.get("match_file") or configuration.get("ground_truth")
    if not ground_truth_file:
        raise ValueError("Evaluation requires 'match_file' or 'ground_truth' in the configuration.")
    similarity_file = _get_similarity_file(configuration)
    output_format = configuration.get("output_format", "graphml")
    if output_format != "graphml":
        raise ValueError("Current evaluation implementation supports graphml output only.")

    actual_pairs, left_target_rids = _ground_truth_pairs(ground_truth_file)
    predicted_pairs = _predicted_pairs_from_graphml(similarity_file)

    all_metrics = _compute_metrics(predicted_pairs, actual_pairs)

    predicted_pairs_left_only = {pair for pair in predicted_pairs if pair[0] in left_target_rids}
    actual_pairs_left_only = {pair for pair in actual_pairs if pair[0] in left_target_rids}
    left_only_metrics = _compute_metrics(predicted_pairs_left_only, actual_pairs_left_only)

    print(f'[Result] version: {configuration["version_name"]}, all_mutual_top1: {all_metrics}, left_target_only: {left_only_metrics}')

    return {
        "all_mutual_top1": all_metrics,
        "left_target_only": left_only_metrics,
    }


__all__ = ["compare_ground_truth"]