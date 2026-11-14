"""
Scoring for Benchmark A.

Computes per-label and macro-averaged metrics from saved per-clip result JSONs:
  - Precision, Recall, F1, Accuracy per label
  - Macro averages across all labels

Result file layout
------------------
  results/benchmark_a/
      {model_key}/
          {clip_id}.json              ← per-clip result (written by runner)
          summary_{model_key}.json    ← this model's scores (written atomically)
      leaderboard.json                ← merged ranking (assembled on demand)

Public API
----------
  score_model(results_dir, model_key) -> dict
      Read all per-clip JSONs, compute metrics. Pure computation, no I/O side effects.

  score_and_save_model(results_dir, model_key) -> dict
      score_model + write summary_{model_key}.json atomically.
      Called by the runner after each model's inference completes.

  assemble_leaderboard(results_dir, model_keys=None) -> dict
      Glob all summary_*.json, build ranking, write leaderboard.json atomically.
      When models were scored on different clip sets, every model is re-scored on
      the clips all of them completed, so the ranked macros are comparable.
      Call after all models are done, or with --score-only.

  build_summary(results_dir, model_keys) -> dict
      Convenience wrapper: score_and_save all models + assemble leaderboard.
      Returns the leaderboard dict.

  print_summary_table(leaderboard) -> None
      Print a readable leaderboard to stdout.
"""

from __future__ import annotations

import glob
import json
import os
from dataclasses import dataclass
from typing import Optional

from benchmark.models.base import LABEL_KEYS

# The nine ASD behavior labels, i.e. everything except "Background".
#
# "Background" is scored as a positive class like the others, but its true value
# means "no ASD behavior is present" — an absence, semantically inverted relative
# to the other nine, yet weighted identically at 1/10 of every macro average. So
# "macro F1 over 10 labels" reads as ten behavior-detection scores when one of
# them is an absence-detection score.
#
# The macro_*_behaviors fields below average over these nine only. The original
# macro_* fields keep all ten labels and are unchanged, so existing leaderboards
# and the ranking remain exactly comparable.
BEHAVIOR_LABEL_KEYS: list[str] = [k for k in LABEL_KEYS if k != "Background"]


@dataclass
class LabelMetrics:
    label: str
    tp: int = 0
    fp: int = 0
    tn: int = 0
    fn: int = 0
    abstained: int = 0  # clips where the model returned None for this label

    @property
    def assessed(self) -> int:
        """Clips the model actually answered (non-abstained)."""
        return self.tp + self.fp + self.tn + self.fn

    @property
    def total(self) -> int:
        """All clips seen for this label, including abstentions."""
        return self.assessed + self.abstained

    @property
    def assessability_rate(self) -> Optional[float]:
        return self.assessed / self.total if self.total > 0 else None

    @property
    def precision(self) -> float:
        """0.0 when the model made no positive predictions (zero_division=0)."""
        denom = self.tp + self.fp
        return self.tp / denom if denom > 0 else 0.0

    @property
    def recall(self) -> float:
        """0.0 when there are no positive ground-truth examples (zero_division=0)."""
        denom = self.tp + self.fn
        return self.tp / denom if denom > 0 else 0.0

    @property
    def f1(self) -> float:
        """Always defined; 0.0 when either precision or recall is 0."""
        denom = self.precision + self.recall
        return 2 * self.precision * self.recall / denom if denom > 0 else 0.0

    @property
    def accuracy(self) -> Optional[float]:
        return (self.tp + self.tn) / self.assessed if self.assessed > 0 else None

    def to_dict(self) -> dict:
        return {
            "label": self.label,
            "tp": self.tp,
            "fp": self.fp,
            "tn": self.tn,
            "fn": self.fn,
            "assessed": self.assessed,
            "abstained": self.abstained,
            "assessability_rate": _round(self.assessability_rate),
            "precision": _round(self.precision),
            "recall": _round(self.recall),
            "f1": _round(self.f1),
            "accuracy": _round(self.accuracy),
        }


def _round(v: Optional[float], ndigits: int = 4) -> Optional[float]:
    return round(v, ndigits) if v is not None else None


def _macro_avg(values: list[Optional[float]]) -> Optional[float]:
    """Average over non-None values; used for optional metrics like accuracy/assessability."""
    valid = [v for v in values if v is not None]
    return sum(valid) / len(valid) if valid else None


def score_model(
    results_dir: str,
    model_key: str,
    clip_ids: Optional[set[str]] = None,
) -> dict:
    """
    Read all per-clip JSONs for a model and compute aggregate metrics.

    Skips:
      - summary_*.json files (not clip results)
      - *.tmp files (incomplete atomic writes)
      - Files with {"status": "in_progress"} (orphaned placeholders)
      - Files with {"error": ...} (counted as failed_clips, not scored)

    Args:
        clip_ids: When provided, only clips whose ID is in this set are scored.
                  Useful for comparing models on an identical subset (e.g. first N clips).

    Returns:
        Dict with per_label metrics + macro averages, suitable for the summary file.
    """
    model_dir = os.path.join(results_dir, "benchmark_a", model_key)
    if not os.path.isdir(model_dir):
        raise FileNotFoundError(f"No results directory found at '{model_dir}'")

    clip_files = [
        f for f in os.listdir(model_dir)
        if f.endswith(".json")
        and not f.startswith("summary")
        and not f.endswith(".tmp")
        and (clip_ids is None or f.removesuffix(".json") in clip_ids)
    ]
    if not clip_files:
        raise FileNotFoundError(f"No clip result JSONs found in '{model_dir}'")

    label_metrics: dict[str, LabelMetrics] = {k: LabelMetrics(label=k) for k in LABEL_KEYS}
    total_clips = 0
    failed_clips = 0
    # Self-consistency of the Background label against the nine behavior labels.
    # In the ground truth, Background is exactly NOT(any behavior) — verified on
    # all 928 rows, zero exceptions. Nothing forces a model's own output to respect
    # that, so a prediction can be internally impossible; the scorer would still
    # count it as a TP in one bucket and an FP in another. These counters measure
    # how often each model contradicts itself.
    consistency_evaluable = 0
    consistency_violations_both = 0   # Background true while a behavior is also true
    consistency_violations_neither = 0  # Background false while no behavior is true
    timing_values: list[float] = []
    input_token_values: list[int] = []
    output_token_values: list[int] = []

    for fname in clip_files:
        fpath = os.path.join(model_dir, fname)
        with open(fpath, encoding="utf-8") as f:
            data = json.load(f)

        # Skip orphaned in-progress placeholders
        if data.get("status") == "in_progress":
            continue

        if data.get("error"):
            failed_clips += 1
            continue

        total_clips += 1
        if "timing_s" in data:
            timing_values.append(data["timing_s"])

        costs = data.get("token_costs") or {}
        if costs.get("input_tokens") is not None:
            input_token_values.append(costs["input_tokens"])
        if costs.get("output_tokens") is not None:
            output_token_values.append(costs["output_tokens"])

        prediction: dict = data.get("asd_behavioral_metrics", {})
        ground_truth: dict = data.get("ground_truth", {})

        # Only judge consistency when the model committed to every label — an
        # abstention makes the constraint unfalsifiable rather than violated.
        def _as_bool(v):
            if isinstance(v, bool): return v
            if isinstance(v, str): return v.lower() == "true"
            return None

        bg = _as_bool(prediction.get("Background"))
        beh = [_as_bool(prediction.get(k)) for k in BEHAVIOR_LABEL_KEYS]
        if bg is not None and all(b is not None for b in beh):
            consistency_evaluable += 1
            if bg and any(beh):
                consistency_violations_both += 1
            elif not bg and not any(beh):
                consistency_violations_neither += 1

        for label in LABEL_KEYS:
            lm = label_metrics[label]
            pred_val = prediction.get(label)
            gt_val = ground_truth.get(label)

            # Normalize pred_val: JSON bool, string, null, or missing
            if pred_val is None:
                lm.abstained += 1
                continue
            if isinstance(pred_val, str):
                pred_bool = pred_val.lower() == "true"
            else:
                pred_bool = bool(pred_val)

            gt_bool = bool(gt_val)

            if pred_bool and gt_bool:
                lm.tp += 1
            elif pred_bool and not gt_bool:
                lm.fp += 1
            elif not pred_bool and not gt_bool:
                lm.tn += 1
            else:
                lm.fn += 1

    per_label = {k: label_metrics[k].to_dict() for k in LABEL_KEYS}

    # Macro metrics always averaged over all 10 labels (zero_division=0 convention).
    macro_f1 = sum(label_metrics[k].f1 for k in LABEL_KEYS) / len(LABEL_KEYS)
    macro_precision = sum(label_metrics[k].precision for k in LABEL_KEYS) / len(LABEL_KEYS)
    macro_recall = sum(label_metrics[k].recall for k in LABEL_KEYS) / len(LABEL_KEYS)
    macro_accuracy = _macro_avg([label_metrics[k].accuracy for k in LABEL_KEYS])
    macro_assessability_rate = _macro_avg([label_metrics[k].assessability_rate for k in LABEL_KEYS])

    # Same convention, restricted to the nine behavior labels — see BEHAVIOR_LABEL_KEYS.
    n_beh = len(BEHAVIOR_LABEL_KEYS)
    macro_f1_behaviors = sum(label_metrics[k].f1 for k in BEHAVIOR_LABEL_KEYS) / n_beh
    macro_precision_behaviors = sum(label_metrics[k].precision for k in BEHAVIOR_LABEL_KEYS) / n_beh
    macro_recall_behaviors = sum(label_metrics[k].recall for k in BEHAVIOR_LABEL_KEYS) / n_beh
    macro_accuracy_behaviors = _macro_avg([label_metrics[k].accuracy for k in BEHAVIOR_LABEL_KEYS])

    mean_timing = round(sum(timing_values) / len(timing_values), 2) if timing_values else None
    total_input_tokens = sum(input_token_values) if input_token_values else None
    total_output_tokens = sum(output_token_values) if output_token_values else None

    return {
        "model_key": model_key,
        "total_clips": total_clips,
        "failed_clips": failed_clips,
        "per_label": per_label,
        "macro_f1": _round(macro_f1),
        "macro_recall": _round(macro_recall),
        "macro_precision": _round(macro_precision),
        "macro_accuracy": _round(macro_accuracy),
        "macro_assessability_rate": _round(macro_assessability_rate),
        # Nine behavior labels only (excludes "Background"). Not used for ranking.
        "macro_f1_behaviors": _round(macro_f1_behaviors),
        "macro_recall_behaviors": _round(macro_recall_behaviors),
        "macro_precision_behaviors": _round(macro_precision_behaviors),
        "macro_accuracy_behaviors": _round(macro_accuracy_behaviors),
        # Background self-consistency (prediction-side only; see counters above)
        "background_consistency_evaluable": consistency_evaluable,
        "background_violations_both": consistency_violations_both,
        "background_violations_neither": consistency_violations_neither,
        "background_consistency_rate": _round(
            1 - (consistency_violations_both + consistency_violations_neither)
            / consistency_evaluable
        ) if consistency_evaluable else None,
        "mean_timing_s": mean_timing,
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
    }


def score_and_save_model(results_dir: str, model_key: str) -> dict:
    """
    Score a model from its per-clip JSONs and write summary_{model_key}.json
    atomically (tmp → os.replace) to results/benchmark_a/{model_key}/.

    Each runner owns exactly one output file. No two runners ever write to the
    same path, so there is no race when multiple models run in parallel.

    Returns the scores dict (same as score_model).
    """
    scores = score_model(results_dir, model_key)

    summary_path = os.path.join(
        results_dir, "benchmark_a", model_key, f"summary_{model_key}.json"
    )
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    tmp_path = summary_path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(scores, f, indent=2, ensure_ascii=False)
    os.replace(tmp_path, summary_path)

    return scores


def scored_clip_ids(results_dir: str, model_key: str) -> set[str]:
    """
    Clip IDs that score_model() would count for this model: finished results,
    excluding errors and in-progress placeholders.
    """
    model_dir = os.path.join(results_dir, "benchmark_a", model_key)
    ids: set[str] = set()
    if not os.path.isdir(model_dir):
        return ids
    for fname in os.listdir(model_dir):
        if not fname.endswith(".json") or fname.startswith("summary"):
            continue
        try:
            with open(os.path.join(model_dir, fname), encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            continue
        if data.get("status") == "in_progress" or data.get("error"):
            continue
        ids.add(fname.removesuffix(".json"))
    return ids


def assemble_leaderboard(results_dir: str, model_keys: Optional[list[str]] = None) -> dict:
    """
    Glob all summary_{model_key}.json files under results/benchmark_a/*/
    and build a ranked leaderboard dict. Writes leaderboard.json atomically.

    Comparable macros: when the models were scored on different clip sets (an
    interrupted run, a pilot on 100 clips, a dev-split run), every model is
    re-scored on the clips ALL of them completed and ranked on that. Otherwise a
    model run only on an easy or hard subset would be ranked against others
    measured on different clips. Each model's all-clips scores are kept under
    "models_all_clips" for reference.

    Args:
        model_keys: restrict the leaderboard to these models (default: every
                    model with a summary). The common clip set is the
                    intersection over the included models only, so excluding a
                    model that ran on few clips widens the comparison.

    Safe to call while other model runs are in progress — it reads only
    completed summary files and finished per-clip results.

    Returns the leaderboard dict:
        models             scores used for ranking (common clips when they differ)
        models_all_clips   each model's scores on all of its own clips
        ranking_basis      "common_clips" or "all_clips"
        common_clips       number of clips in the common set
        per_model_clip_counts, ranking, errors
    """
    pattern = os.path.join(results_dir, "benchmark_a", "*", "summary_*.json")
    summary_files = glob.glob(pattern)

    all_clip_scores: dict[str, dict] = {}
    errors: dict[str, str] = {}

    for sf in summary_files:
        try:
            with open(sf, encoding="utf-8") as f:
                data = json.load(f)
            # Use model_key field if present, otherwise derive from filename
            model_key = data.get("model_key") or (
                os.path.basename(sf)
                .removeprefix("summary_")
                .removesuffix(".json")
            )
            all_clip_scores[model_key] = data
        except Exception as e:
            errors[sf] = str(e)

    if model_keys is not None:
        unknown = [k for k in model_keys if k not in all_clip_scores]
        if unknown:
            errors["model_keys"] = f"no summary for: {', '.join(unknown)}"
        all_clip_scores = {k: v for k, v in all_clip_scores.items() if k in model_keys}

    clip_sets = {k: scored_clip_ids(results_dir, k) for k in all_clip_scores}
    common = set.intersection(*clip_sets.values()) if clip_sets else set()

    models_scores = all_clip_scores
    ranking_basis = "all_clips"
    if len(clip_sets) > 1 and len({frozenset(s) for s in clip_sets.values()}) > 1:
        if common:
            ranking_basis = "common_clips"
            models_scores = {}
            for k in all_clip_scores:
                try:
                    models_scores[k] = score_model(results_dir, k, clip_ids=common)
                except Exception as e:
                    errors[f"common_{k}"] = str(e)
                    models_scores[k] = all_clip_scores[k]
        else:
            errors["common_clips"] = (
                "The models share no completed clips; ranking on each model's "
                "own clips, which are NOT comparable."
            )

    ranking = sorted(
        models_scores.keys(),
        key=lambda k: models_scores[k].get("macro_f1") or -1,
        reverse=True,
    )

    leaderboard = {
        "models": models_scores,
        "models_all_clips": all_clip_scores,
        "ranking_basis": ranking_basis,
        "common_clips": len(common) if clip_sets else 0,
        "per_model_clip_counts": {k: len(s) for k, s in clip_sets.items()},
        "ranking": ranking,
        "errors": errors,
    }

    leaderboard_path = os.path.join(results_dir, "benchmark_a", "leaderboard.json")
    os.makedirs(os.path.dirname(leaderboard_path), exist_ok=True)
    tmp_path = leaderboard_path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(leaderboard, f, indent=2, ensure_ascii=False)
    os.replace(tmp_path, leaderboard_path)

    return leaderboard


def build_summary(results_dir: str, model_keys: list[str]) -> dict:
    """
    Convenience wrapper: score_and_save all listed models then assemble
    and return the leaderboard. Used by the runner after all inference is done.
    """
    for model_key in model_keys:
        try:
            score_and_save_model(results_dir, model_key)
        except Exception as e:
            print(f"  [scoring] Could not score model '{model_key}': {e}")

    return assemble_leaderboard(results_dir)


def print_summary_table(leaderboard: dict) -> None:
    """Print a readable leaderboard to stdout."""
    models = leaderboard.get("models", {})
    ranking = leaderboard.get("ranking", [])

    header = (
        f"{'Rank':<5} {'Model':<30} {'Macro F1':>10} {'Recall':>10} "
        f"{'Precision':>10} {'Accuracy':>10} {'Avg s/clip':>11} "
        f"{'In Tokens':>12} {'Out Tokens':>12}"
    )
    print("\n" + "=" * len(header))
    if leaderboard.get("ranking_basis") == "common_clips":
        print(f"Benchmark A Results — all models scored on the same "
              f"{leaderboard.get('common_clips')} clips (completed by every model below)")
    else:
        print("Benchmark A Results")
    print("=" * len(header))
    print(header)
    print("-" * len(header))

    for i, key in enumerate(ranking, 1):
        m = models[key]
        in_tok = m.get("total_input_tokens")
        out_tok = m.get("total_output_tokens")
        print(
            f"{i:<5} {key:<30} "
            f"{_fmt(m.get('macro_f1')):>10} "
            f"{_fmt(m.get('macro_recall')):>10} "
            f"{_fmt(m.get('macro_precision')):>10} "
            f"{_fmt(m.get('macro_accuracy')):>10} "
            f"{_fmt_secs(m.get('mean_timing_s')):>11} "
            f"{str(in_tok) if in_tok is not None else 'N/A':>12} "
            f"{str(out_tok) if out_tok is not None else 'N/A':>12}"
        )
    print("=" * len(header) + "\n")
    print("  Avg s/clip is mean per-clip request latency, not wall-clock throughput —\n"
          "  clips run concurrently, so elapsed time per model is roughly\n"
          "  (avg s/clip x total_clips) / that model's max_concurrent.\n")

    _print_behavior_only_macro(models, ranking)
    _print_background_consistency(models, ranking)

    if leaderboard.get("ranking_basis") == "common_clips":
        _print_common_clip_note(leaderboard, ranking)
    else:
        _warn_uneven_clip_counts(models, ranking)


def _print_common_clip_note(leaderboard: dict, ranking: list) -> None:
    """
    Explain the common-clip basis: each model's own clip count, how many of its
    clips were left out, and its all-clips macro F1 (not comparable, for reference).
    Points out which model limits the common set when it is much smaller than
    the largest run.
    """
    counts = leaderboard.get("per_model_clip_counts", {})
    all_scores = leaderboard.get("models_all_clips", {})
    common = leaderboard.get("common_clips", 0)
    print(f"  Scores above use the {common} clips every model completed, so they are "
          f"directly comparable.")
    print(f"    {'model':<30}{'own clips':>10}{'left out':>10}{'all-clips F1':>14}")
    for k in ranking:
        n = counts.get(k, 0)
        f1 = all_scores.get(k, {}).get("macro_f1")
        print(f"    {k:<30}{n:>10}{n - common:>10}{_fmt(f1):>14}")
    print("    all-clips F1 is each model on its own clips — for reference only, "
          "not comparable across models.")
    largest = max(counts.values()) if counts else 0
    if largest and common < 0.8 * largest:
        smallest = sorted(counts, key=counts.get)[:2]
        print(f"  The common set ({common}) is limited by the smallest run(s): "
              f"{', '.join(f'{k} ({counts[k]})' for k in smallest)}. "
              f"Use --leaderboard-models to compare a subset on more clips.")
    print()


def _print_behavior_only_macro(models: dict, ranking: list) -> None:
    """
    Show the nine-behavior macro alongside the all-ten headline, and flag any
    model whose rank would change if the leaderboard were sorted on it instead.
    Kept out of the main table to avoid widening it further.
    """
    rows = [(k, models[k].get("macro_f1"), models[k].get("macro_f1_behaviors"))
            for k in ranking]
    if not any(b is not None for _, _, b in rows):
        return  # summaries predate the field; re-run --score-only to populate

    print("  Macro F1 excluding 'Background' (9 behavior labels only):")
    for k, all10, beh in rows:
        if beh is None:
            print(f"    {k:<30}      (not scored — re-run --score-only)")
            continue
        delta = beh - all10 if all10 is not None else None
        print(f"    {k:<30} {beh:.4f}   (all 10: {all10:.4f}"
              + (f", {delta:+.4f})" if delta is not None else ")"))

    scored = [(k, b) for k, _, b in rows if b is not None]
    reordered = [k for k, _ in sorted(scored, key=lambda kv: kv[1], reverse=True)]
    if reordered != [k for k, _ in scored]:
        print(f"\n    NOTE: ranking on the behavior-only macro would reorder to: "
              f"{' > '.join(reordered)}")
    print()


def _print_background_consistency(models: dict, ranking: list) -> None:
    """
    Report how often each model's Background prediction contradicts its own nine
    behavior predictions. Ground truth satisfies Background == NOT(any behavior)
    on every row, so a violation is a logically impossible answer, not a near miss.
    """
    rows = [(k, models[k]) for k in ranking
            if models[k].get("background_consistency_rate") is not None]
    if not rows:
        return

    print("  Background self-consistency (Background must equal NOT(any behavior)):")
    print(f"    {'model':<30}{'consistent':>11}{'both':>7}{'neither':>9}{'of':>7}")
    for k, m in rows:
        print(f"    {k:<30}"
              f"{m['background_consistency_rate']:>11.4f}"
              f"{m['background_violations_both']:>7}"
              f"{m['background_violations_neither']:>9}"
              f"{m['background_consistency_evaluable']:>7}")
    print("    both    = said Background AND a behavior were both present")
    print("    neither = said Background absent while predicting no behavior at all\n")


def _warn_uneven_clip_counts(models: dict, ranking: list) -> None:
    """
    Warn when the ranked models were not scored on the same number of clips.

    A summary is written even when a run is interrupted, so an aborted or
    partially-resumed run leaves structurally valid summaries at different
    denominators. The table above would rank them against each other with no
    indication that the comparison is unequal — this makes that visible.
    """
    counts = {k: models[k].get("total_clips") for k in ranking}
    present = {k: v for k, v in counts.items() if v is not None}
    if len(set(present.values())) <= 1:
        return

    top = max(present.values())
    print("  [WARNING] Models were scored on different clip counts — "
          "macro metrics above are NOT directly comparable:")
    for k, v in sorted(present.items(), key=lambda kv: kv[1]):
        failed = models[k].get("failed_clips")
        suffix = f", {failed} failed" if failed else ""
        gap = "" if v == top else f"  ← {top - v} fewer than the leader"
        print(f"    {k:<30} scored {v} clip(s){suffix}{gap}")
    missing = [k for k, v in counts.items() if v is None]
    if missing:
        print(f"    no total_clips recorded: {', '.join(missing)}")
    print("  Re-run the same command to finish the short runs (completed clips are cached).\n")


def _fmt(v: Optional[float]) -> str:
    return f"{v:.4f}" if v is not None else "  N/A"


def _fmt_secs(v: Optional[float]) -> str:
    """Seconds with one decimal — four decimals are noise for a latency in the 5-60s range."""
    return f"{v:.1f}s" if v is not None else "N/A"
