"""
Scoring for Benchmark C.

Three-dimension rubric (Safety, Clinical Actionability, Relational Empathy).
Evidence alignment (clinical_accuracy) is excluded from the composite: a
single-stage report has no observation record to be checked against.

Cross-model composite formula (0–1):
    score = 0.40 × (safety / 3)
          + 0.40 × (clinical_actionability / 7)
          + 0.20 × ((relational_empathy - 7) / 28)

Judge file format ({results_dir}/judge/{clip_id}.json):
  {
    "clip_id": "...",
    "model_key_to_label": {"single-stage-matched": "Model A", ...},
    "judge_keys": ["muse-c-judge", "deepseek-v4-pro-c-judge", "claude-opus-5-c-judge"],
    "judge_scores": {
      "muse-c-judge": {
        "Model A": {
          "relational_empathy":    {"total": int, "reasoning": str, ...sub-scores},
          "clinical_actionability":{"total": int, "reasoning": str, ...sub-scores},
          "safety":                {"score": int, "reasoning": str}
        }, ...
      }, ...
    }
  }

Clinical accuracy file format (results/benchmark_c/clinical_accuracy/{clip_id}.json):
  {
    "clip_id": "...",
    "judge_scores": {
      "<judge key>": {"total": int, "a1_...": int, ..., "reasoning": str}, ...
    },
    "timing_s": float,
    "error": null
  }

Public API:
    score_model(results_dir, model_key) -> dict
    score_and_save_model(results_dir, model_key) -> dict
    score_kidooo_clinical_accuracy(results_dir) -> dict
    assemble_leaderboard(results_dir) -> dict
    print_summary_table(leaderboard) -> None
"""

from __future__ import annotations

import glob
import json
import math
import os
from typing import Optional


# ---------------------------------------------------------------------------
# Cross-model composite formula
# ---------------------------------------------------------------------------

def _cross_model_score(
    safety: float,                  # 0–3
    clinical_actionability: float,  # 0–7
    relational_empathy: float,      # 7–35
) -> float:
    s_norm   = safety / 3.0
    act_norm = clinical_actionability / 7.0
    emp_norm = (relational_empathy - 7.0) / 28.0
    return round(
        0.40 * s_norm
        + 0.40 * act_norm
        + 0.20 * emp_norm,
        4,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _round(v: Optional[float], n: int = 4) -> Optional[float]:
    return round(v, n) if v is not None else None


def _mean(values: list[float]) -> Optional[float]:
    return sum(values) / len(values) if values else None


def _std(values: list[float]) -> Optional[float]:
    if len(values) < 2:
        return None
    m = sum(values) / len(values)
    return math.sqrt(sum((x - m) ** 2 for x in values) / (len(values) - 1))


def _bench_c_dir(results_dir: str) -> str:
    return os.path.join(results_dir, "benchmark_c")


def _save_atomic(path: str, data: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


def _word_count(text: str) -> int:
    return len(text.split()) if text else 0


def _extract_dim_score(model_scores: dict, dim: str) -> Optional[float]:
    val = model_scores.get(dim)
    if val is None:
        return None
    if isinstance(val, dict):
        raw = val.get("score") if dim == "safety" else val.get("total")
        return float(raw) if raw is not None else None
    return float(val)


# ---------------------------------------------------------------------------
# Krippendorff's Alpha (interval, pure Python) — shared with benchmark_b
# ---------------------------------------------------------------------------

def _krippendorff_alpha(ratings_matrix: list[list[Optional[float]]]) -> Optional[float]:
    all_vals: list[float] = [v for unit in ratings_matrix for v in unit if v is not None]
    n = len(all_vals)
    if n < 2:
        return None

    do_sum = 0.0
    do_count = 0
    for unit in ratings_matrix:
        vals = [v for v in unit if v is not None]
        for i in range(len(vals)):
            for j in range(i + 1, len(vals)):
                do_sum += (vals[i] - vals[j]) ** 2
                do_count += 1

    if do_count == 0:
        return None
    D_o = do_sum / do_count

    sum_v  = sum(all_vals)
    sum_v2 = sum(v ** 2 for v in all_vals)
    de_num = n * sum_v2 - sum_v ** 2
    de_den = n * (n - 1) / 2
    if de_den == 0:
        return None
    D_e = de_num / de_den

    if D_e == 0:
        return 1.0 if D_o == 0 else None
    return _round(1 - D_o / D_e)


# ---------------------------------------------------------------------------
# Common clip set enforcement (fairness across models)
# ---------------------------------------------------------------------------

def _common_clip_set(
    results_dir: str, model_keys: list[str]
) -> tuple[set[str], dict[str, set[str]]]:
    """
    Return the set of clip_ids where every model in model_keys was judged,
    along with per-model coverage sets.

    A clip is included in the common set only when all model_keys appear in
    its judge file's model_key_to_label.  This prevents a flaky model from
    being scored on a different (potentially easier) subset of clips than its
    peers, which would make the leaderboard unfair.
    """
    bench_dir = _bench_c_dir(results_dir)
    judge_dir = os.path.join(bench_dir, "judge")
    if not os.path.isdir(judge_dir):
        return set(), {mk: set() for mk in model_keys}

    per_model: dict[str, set[str]] = {mk: set() for mk in model_keys}

    for fname in sorted(os.listdir(judge_dir)):
        if not fname.endswith(".json") or fname.endswith(".tmp"):
            continue
        fpath = os.path.join(judge_dir, fname)
        try:
            with open(fpath, encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            continue
        if data.get("error"):
            continue
        clip_id: str = data.get("clip_id") or fname.removesuffix(".json")
        label_map: dict[str, str] = data.get("model_key_to_label", {})
        for mk in model_keys:
            if mk in label_map:
                per_model[mk].add(clip_id)

    if not model_keys:
        return set(), per_model

    common = set.intersection(*(per_model[mk] for mk in model_keys))
    return common, per_model


# ---------------------------------------------------------------------------
# Core per-model scoring
# ---------------------------------------------------------------------------

def score_model(
    results_dir: str,
    model_key: str,
    clip_ids: Optional[set[str]] = None,
) -> dict:
    """
    Aggregate multi-judge scores for one model across all clips.

    Args:
        clip_ids: If provided, only clips in this set are included.
                  Pass the common clip set from assemble_leaderboard to
                  ensure all models are scored on an identical subset.

    Returns a summary dict with per-dimension means, IRR, and composite score.
    """
    bench_dir = _bench_c_dir(results_dir)
    judge_dir = os.path.join(bench_dir, "judge")

    if not os.path.isdir(judge_dir):
        raise FileNotFoundError(f"No judge results directory at '{judge_dir}'")

    dimensions = ["safety", "clinical_actionability", "relational_empathy"]

    per_clip_dim_means: dict[str, list[float]] = {d: [] for d in dimensions}
    per_clip_judge_scores: dict[str, list[list[float]]] = {d: [] for d in dimensions}
    final_scores: list[float] = []
    reasoning_wc: list[int] = []
    total_clips = 0
    failed_clips = 0
    excluded_clips = 0  # clips skipped because not in clip_ids

    for fname in sorted(os.listdir(judge_dir)):
        if not fname.endswith(".json") or fname.endswith(".tmp"):
            continue
        fpath = os.path.join(judge_dir, fname)
        with open(fpath, encoding="utf-8") as f:
            data = json.load(f)

        if data.get("error"):
            failed_clips += 1
            continue

        clip_id: str = data.get("clip_id") or fname.removesuffix(".json")
        if clip_ids is not None and clip_id not in clip_ids:
            excluded_clips += 1
            continue

        label_map: dict[str, str] = data.get("model_key_to_label", {})
        anon_label = label_map.get(model_key)
        if anon_label is None:
            continue

        judge_scores_map: dict[str, dict] = data.get("judge_scores", {})
        clip_dim_scores: dict[str, list[float]] = {d: [] for d in dimensions}

        for jk, jscores in judge_scores_map.items():
            model_scores = jscores.get(anon_label)
            if not model_scores:
                continue
            for dim in dimensions:
                val = _extract_dim_score(model_scores, dim)
                if val is not None:
                    clip_dim_scores[dim].append(val)
            for dim in dimensions:
                dim_val = model_scores.get(dim)
                if isinstance(dim_val, dict):
                    reasoning = dim_val.get("reasoning", "")
                    if reasoning:
                        reasoning_wc.append(_word_count(reasoning))

        if not any(clip_dim_scores[d] for d in dimensions):
            continue

        total_clips += 1
        clip_dim_means: dict[str, Optional[float]] = {}

        for dim in dimensions:
            vals = clip_dim_scores[dim]
            if vals:
                m = sum(vals) / len(vals)
                per_clip_dim_means[dim].append(m)
                per_clip_judge_scores[dim].append(vals)
                clip_dim_means[dim] = m
            else:
                clip_dim_means[dim] = None

        s   = clip_dim_means.get("safety")
        act = clip_dim_means.get("clinical_actionability")
        emp = clip_dim_means.get("relational_empathy")
        if None not in (s, act, emp):
            final_scores.append(_cross_model_score(s, act, emp))

    # --------------- Aggregate ---------------
    summary: dict = {
        "total_clips_judged": total_clips,
        "failed_clips": failed_clips,
        "excluded_clips": excluded_clips,
    }

    all_ij_stds: list[float] = []
    alpha_per_dim: list[float] = []

    for dim in dimensions:
        means = per_clip_dim_means[dim]
        summary[f"mean_{dim}"] = _round(_mean(means))
        summary[f"std_{dim}"]  = _round(_std(means))

        ij_stds = [
            s for clip_scores in per_clip_judge_scores[dim]
            if (s := _std(clip_scores)) is not None
        ]
        summary[f"mean_inter_judge_std_{dim}"] = _round(_mean(ij_stds))
        all_ij_stds.extend(ij_stds)

        alpha = _krippendorff_alpha(per_clip_judge_scores[dim])
        summary[f"krippendorff_alpha_{dim}"] = alpha
        if alpha is not None:
            alpha_per_dim.append(alpha)

    summary["mean_cross_model_score"]    = _round(_mean(final_scores))
    summary["std_cross_model_score"]     = _round(_std(final_scores))
    summary["mean_inter_judge_std"]      = _round(_mean(all_ij_stds))
    summary["krippendorff_alpha"]        = _round(_mean(alpha_per_dim)) if alpha_per_dim else None
    summary["mean_reasoning_word_count"] = _round(_mean([float(c) for c in reasoning_wc]), 1)

    return {"model_key": model_key, "judge": summary}


def score_and_save_model(
    results_dir: str,
    model_key: str,
    clip_ids: Optional[set[str]] = None,
) -> dict:
    scores = score_model(results_dir, model_key, clip_ids=clip_ids)
    path = os.path.join(_bench_c_dir(results_dir), f"summary_{model_key}.json")
    _save_atomic(path, scores)
    return scores


# ---------------------------------------------------------------------------
# Kidooo Clinical Accuracy (Stage 4 / Metric -1)
# ---------------------------------------------------------------------------

def score_kidooo_clinical_accuracy(results_dir: str) -> dict:
    """
    Aggregate B-judge clinical accuracy scores for Kidooo's reports.

    Reads results/benchmark_c/clinical_accuracy/{clip_id}.json files saved by Stage 4.
    Each file contains per-judge clinical_accuracy dicts with sub-scores and total (0–12).

    Returns a summary dict with mean, std, IRR, and Krippendorff's alpha.
    Scale: 0–12 (6 DSM-5 domains × 0–2 each).
    """
    acc_dir = os.path.join(_bench_c_dir(results_dir), "clinical_accuracy")
    if not os.path.isdir(acc_dir):
        return {
            "total_clips": 0,
            "failed_clips": 0,
            "mean_clinical_accuracy": None,
            "std_clinical_accuracy": None,
            "mean_inter_judge_std": None,
            "krippendorff_alpha": None,
            "note": "No clinical_accuracy directory found — Stage 4 not yet run",
        }

    per_clip_judge_scores: list[list[float]] = []
    per_clip_means: list[float] = []
    total_clips = 0
    failed_clips = 0

    for fname in sorted(os.listdir(acc_dir)):
        if not fname.endswith(".json") or fname.endswith(".tmp"):
            continue
        fpath = os.path.join(acc_dir, fname)
        try:
            with open(fpath, encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            continue

        if data.get("error"):
            failed_clips += 1
            continue

        judge_scores_map: dict = data.get("judge_scores", {})
        totals: list[float] = []
        for jk, ca in judge_scores_map.items():
            t = ca.get("total")
            if t is not None:
                totals.append(float(t))

        if not totals:
            continue

        total_clips += 1
        per_clip_judge_scores.append(totals)
        per_clip_means.append(sum(totals) / len(totals))

    alpha = _krippendorff_alpha(per_clip_judge_scores)
    ij_stds = [
        s for scores in per_clip_judge_scores
        if (s := _std(scores)) is not None
    ]

    return {
        "total_clips": total_clips,
        "failed_clips": failed_clips,
        "mean_clinical_accuracy": _round(_mean(per_clip_means)),
        "std_clinical_accuracy": _round(_std(per_clip_means)),
        "mean_inter_judge_std": _round(_mean(ij_stds)),
        "krippendorff_alpha": alpha,
    }


# ---------------------------------------------------------------------------
# Pairwise Win/Tie/Loss
# ---------------------------------------------------------------------------

def _compute_win_tie_loss(
    results_dir: str,
    model_keys: list[str],
    clip_ids: Optional[set[str]] = None,
) -> dict:
    bench_dir = _bench_c_dir(results_dir)
    judge_dir = os.path.join(bench_dir, "judge")

    video_scores: dict[str, dict[str, float]] = {}

    for fname in sorted(os.listdir(judge_dir)):
        if not fname.endswith(".json") or fname.endswith(".tmp"):
            continue
        fpath = os.path.join(judge_dir, fname)
        with open(fpath, encoding="utf-8") as f:
            data = json.load(f)
        if data.get("error"):
            continue

        clip_id: str = data.get("clip_id", fname.removesuffix(".json"))
        if clip_ids is not None and clip_id not in clip_ids:
            continue
        label_map = data.get("model_key_to_label", {})
        judge_scores_map = data.get("judge_scores", {})

        for mk in model_keys:
            anon = label_map.get(mk)
            if anon is None:
                continue
            finals: list[float] = []
            for jk, jscores in judge_scores_map.items():
                ms = jscores.get(anon, {})
                s   = _extract_dim_score(ms, "safety")
                act = _extract_dim_score(ms, "clinical_actionability")
                emp = _extract_dim_score(ms, "relational_empathy")
                if None not in (s, act, emp):
                    finals.append(_cross_model_score(s, act, emp))
            if finals:
                video_scores.setdefault(clip_id, {})[mk] = sum(finals) / len(finals)

    wtl: dict[str, dict] = {}
    for a in model_keys:
        for b in model_keys:
            if a == b:
                continue
            wins = ties = losses = 0
            for vid, scores in video_scores.items():
                if a not in scores or b not in scores:
                    continue
                sa, sb = scores[a], scores[b]
                if sa > sb:
                    wins += 1
                elif sa == sb:
                    ties += 1
                else:
                    losses += 1
            n = wins + ties + losses
            wtl[f"{a}_vs_{b}"] = {
                "wins": wins, "ties": ties, "losses": losses,
                "win_rate": _round(wins / n) if n > 0 else None,
                "n_clips": n,
            }
    return wtl


# ---------------------------------------------------------------------------
# Leaderboard assembly
# ---------------------------------------------------------------------------

def assemble_leaderboard(results_dir: str) -> dict:
    """
    Build and save the leaderboard.

    All models are re-scored on the *common clip set* — the intersection of
    clips where every model was judged.  This prevents a flaky model from
    being evaluated on a different (potentially easier) subset of clips than
    its peers.  Coverage stats (per-model clip counts, excluded clips) are
    included in the leaderboard for transparency.
    """
    pattern = os.path.join(_bench_c_dir(results_dir), "summary_*.json")
    summary_files = glob.glob(pattern)

    # Collect model keys from existing summary files
    model_keys: list[str] = []
    errors: dict[str, str] = {}
    for sf in summary_files:
        try:
            with open(sf, encoding="utf-8") as f:
                data = json.load(f)
            mk = data.get("model_key") or (
                os.path.basename(sf).removeprefix("summary_").removesuffix(".json")
            )
            model_keys.append(mk)
        except Exception as e:
            errors[sf] = str(e)

    # Compute common clip set and per-model coverage
    common_clip_ids: set[str] = set()
    per_model_clip_counts: dict[str, int] = {}
    coverage_warning: Optional[str] = None

    if len(model_keys) > 1:
        common_clip_ids, per_model_sets = _common_clip_set(results_dir, model_keys)
        per_model_clip_counts = {mk: len(s) for mk, s in per_model_sets.items()}
        excluded = {
            mk: len(per_model_sets[mk]) - len(common_clip_ids)
            for mk in model_keys
            if len(per_model_sets[mk]) > len(common_clip_ids)
        }
        if excluded:
            coverage_warning = (
                "Some models were judged on fewer clips than peers — scores reflect "
                "the common subset only. Excluded clips per model: "
                + ", ".join(f"{mk}={n}" for mk, n in excluded.items())
            )
            print(f"[WARNING] {coverage_warning}")
    elif model_keys:
        # Single model: no cross-model fairness constraint needed
        common_clip_ids = None  # type: ignore[assignment]

    # Re-score every model on the common set and save summaries
    models_scores: dict[str, dict] = {}
    for mk in model_keys:
        try:
            data = score_and_save_model(results_dir, mk, clip_ids=common_clip_ids)
            models_scores[mk] = data
        except Exception as e:
            errors[f"score_{mk}"] = str(e)

    ranking = sorted(
        models_scores.keys(),
        key=lambda k: models_scores[k].get("judge", {}).get("mean_cross_model_score") or -1,
        reverse=True,
    )

    win_tie_loss: dict = {}
    if len(models_scores) > 1:
        try:
            win_tie_loss = _compute_win_tie_loss(
                results_dir, list(models_scores.keys()), clip_ids=common_clip_ids
            )
        except Exception as e:
            errors["win_tie_loss"] = str(e)

    # Kidooo-only: Clinical Accuracy (Stage 4 / Metric -1)
    clinical_accuracy_summary: Optional[dict] = None
    try:
        clinical_accuracy_summary = score_kidooo_clinical_accuracy(results_dir)
    except Exception as e:
        errors["clinical_accuracy"] = str(e)

    leaderboard = {
        "models": models_scores,
        "ranking": ranking,
        "win_tie_loss": win_tie_loss,
        "common_clips": len(common_clip_ids) if common_clip_ids is not None else None,
        "per_model_clip_counts": per_model_clip_counts,
        "coverage_warning": coverage_warning,
        "kidooo_clinical_accuracy": clinical_accuracy_summary,
        "errors": errors,
    }
    path = os.path.join(_bench_c_dir(results_dir), "leaderboard.json")
    _save_atomic(path, leaderboard)
    return leaderboard


# ---------------------------------------------------------------------------
# Print summary
# ---------------------------------------------------------------------------

def print_summary_table(leaderboard: dict) -> None:
    models       = leaderboard.get("models", {})
    ranking      = leaderboard.get("ranking", [])
    wtl          = leaderboard.get("win_tie_loss", {})
    common_clips = leaderboard.get("common_clips")
    coverage_warning = leaderboard.get("coverage_warning")
    clinical_accuracy = leaderboard.get("kidooo_clinical_accuracy")

    header = (
        f"{'Rank':<5} {'Model':<28} {'Score':>7} {'StdDev':>7} "
        f"{'Safe':>5} {'Act':>5} {'Emp':>6} "
        f"{'IJ-Std':>7} {'α-IRR':>7} {'Reas-Len':>9}"
    )
    clips_note = f" ({common_clips} common clips)" if common_clips is not None else ""
    print("\n" + "=" * len(header))
    print(f"Table 5 rubric results{clips_note}")
    if coverage_warning:
        print(f"  *** {coverage_warning}")
    print("=" * len(header))
    print(header)
    print("-" * len(header))

    for i, key in enumerate(ranking, 1):
        m = models[key]
        j = m.get("judge", {})
        print(
            f"{i:<5} {key:<28} "
            f"{_fmt(j.get('mean_cross_model_score')):>7} "
            f"{_fmt(j.get('std_cross_model_score')):>7} "
            f"{_fmt(j.get('mean_safety')):>5} "
            f"{_fmt(j.get('mean_clinical_actionability')):>5} "
            f"{_fmt(j.get('mean_relational_empathy')):>6} "
            f"{_fmt(j.get('mean_inter_judge_std')):>7} "
            f"{_fmt(j.get('krippendorff_alpha')):>7} "
            f"{_fmt_int(j.get('mean_reasoning_word_count')):>9}"
        )

    print("=" * len(header))

    if wtl and len(ranking) > 1:
        print("\nHead-to-Head Win/Tie/Loss (on mean_cross_model_score):")
        print(f"  {'Matchup':<50} {'W':>4} {'T':>4} {'L':>4} {'Win%':>7}")
        print("  " + "-" * 73)
        for a in ranking:
            for b in ranking:
                if a == b:
                    continue
                key_ab = f"{a}_vs_{b}"
                if key_ab not in wtl:
                    continue
                entry = wtl[key_ab]
                win_pct = f"{entry['win_rate']*100:.1f}%" if entry.get("win_rate") is not None else "  N/A"
                print(
                    f"  {f'{a} vs {b}':<50} "
                    f"{entry['wins']:>4} {entry['ties']:>4} {entry['losses']:>4} "
                    f"{win_pct:>7}"
                )

    if clinical_accuracy and clinical_accuracy.get("total_clips", 0) > 0:
        print("\nKidooo Clinical Accuracy — Metric -1 (0–12, 6 DSM-5 domains × 0–2):")
        ca = clinical_accuracy
        print(
            f"  Clips: {ca['total_clips']}  "
            f"Mean: {_fmt(ca.get('mean_clinical_accuracy'))}  "
            f"Std: {_fmt(ca.get('std_clinical_accuracy'))}  "
            f"IJ-Std: {_fmt(ca.get('mean_inter_judge_std'))}  "
            f"α-IRR: {_fmt(ca.get('krippendorff_alpha'))}"
        )
    print()


def _fmt(v: Optional[float]) -> str:
    return f"{v:.4f}" if v is not None else "  N/A"


def _fmt_int(v: Optional[float]) -> str:
    return f"{v:.0f}" if v is not None else "  N/A"
