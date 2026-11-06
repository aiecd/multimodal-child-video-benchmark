"""
Scoring for Benchmark B.

Reads judge output files and aggregates scores across the AI judges and all
clips to produce per-model summaries and a leaderboard.

Judge file format (results/benchmark_b/judge/{language}/{clip_id}.json):
  {
    "judge_keys": ["gemini-3.7-judge", "muse-judge", "deepseek-v4-pro-judge"],
    "model_key_to_label": {"gpt-5.6-sol": "Model A", ...},
    "judge_scores": {
      "gemini-3.7-judge": {
        "Model A": {
          "clinical_accuracy":     {"total": int, "reasoning": str, ...sub-scores},
          "relational_empathy":    {"total": int, "reasoning": str, ...sub-scores},
          "clinical_actionability":{"total": int, "reasoning": str, ...sub-scores},
          "safety":                {"score": int, "reasoning": str}
        }, ...
      },
      "muse-judge":            {"Model A": {...}, ...},
      "deepseek-v4-pro-judge": {"Model A": {...}, ...}
    }
  }

Metrics per model (in summary_{language}_{model_key}.json):

  Per dimension (clinical_accuracy, safety, clinical_actionability, relational_empathy):
    mean_{dim}                — mean score across all (judge × video) pairs
    std_{dim}                 — std of per-video mean scores (cross-video spread)
    mean_inter_judge_std_{dim}— avg std across judges per video (judge disagreement)
    krippendorff_alpha_{dim}  — inter-rater reliability for this dimension (interval α)

  Composite (final score range 0–1):
    mean_final_judge_score    — weighted: 0.40×CA/12 + 0.25×S/3 + 0.25×Act/7 + 0.10×(E-7)/28
    std_final_judge_score     — cross-video std of composite
    mean_inter_judge_std      — overall avg judge disagreement across all dims/videos
    krippendorff_alpha        — IRR averaged across the 4 dimensions

  Qualitative:
    mean_reasoning_word_count — avg words in judge reasoning field (all judge×video pairs)

Leaderboard extras (in leaderboard.json):
  win_tie_loss              — pairwise W/T/L matrix on mean_final_judge_score

Interpreting Win/Tie/Loss:
  For pair (A, B): W = videos where A's mean score > B's mean score, etc.
  win_rate = W / (W + T + L)  — fraction of videos where A strictly outperformed B

Result layout:
   results/benchmark_b/
       judge/{language}/{clip_id}.json      ← per-clip judge scores (runner)
       summary_{language}_{model_key}.json  ← aggregated scores per model
       leaderboard_{language}.json          ← ranked + pairwise W/T/L matrix

Public API:
   score_model(results_dir, model_key) -> dict
   score_and_save_model(results_dir, model_key) -> dict
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
# Score formula
# ---------------------------------------------------------------------------

def _final_judge_score(
    clinical_accuracy: float,       # 0–12
    safety: float,                  # 0–3
    clinical_actionability: float,  # 0–7
    relational_empathy: float,      # 7–35 (minimum 7, effective range 28)
) -> float:
    # Normalize each dimension to [0, 1] before applying weights.
    # relational_empathy starts at 7 (7 items × min 1), so subtract the floor.
    # Weights: 40% clinical accuracy, 25% safety, 25% actionability, 10% empathy.
    # Final score range: 0–1.
    ca_norm  = clinical_accuracy / 12.0
    s_norm   = safety / 3.0
    act_norm = clinical_actionability / 7.0
    emp_norm = (relational_empathy - 7.0) / 28.0
    return round(
        0.40 * ca_norm
        + 0.25 * s_norm
        + 0.25 * act_norm
        + 0.10 * emp_norm,
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


def _bench_b_dir(results_dir: str) -> str:
    return os.path.join(results_dir, "benchmark_b")


def _save_atomic(path: str, data: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


def _word_count(text: str) -> int:
    return len(text.split()) if text else 0


def _extract_dim_score(model_scores: dict, dim: str) -> Optional[float]:
    """Extract a raw dimension score from the nested judge output.

    Handles both the current nested format (each dimension is a dict with
    'total' or 'score') and the legacy flat format (dimension value is a number).
    """
    val = model_scores.get(dim)
    if val is None:
        return None
    if isinstance(val, dict):
        raw = val.get("score") if dim == "safety" else val.get("total")
        return float(raw) if raw is not None else None
    # Legacy flat format
    return float(val)


# ---------------------------------------------------------------------------
# Krippendorff's Alpha (interval metric, pure Python)
# ---------------------------------------------------------------------------

def _krippendorff_alpha(ratings_matrix: list[list[Optional[float]]]) -> Optional[float]:
    """
    Krippendorff's Alpha for interval data (d² = (v_k - v_l)²).

    ratings_matrix: list of units; each unit is a list of ratings (one per judge).
                    Use None for missing ratings (e.g. a judge that failed).

    Returns alpha in [-1, 1]:
      1.0  = perfect agreement
      0.0  = no better than chance
      < 0  = systematic disagreement (rare in practice)
      None = insufficient data to compute

    Formula:
      α = 1 - D_o / D_e
      D_o = mean pairwise squared diff within units  (observed disagreement)
      D_e = mean pairwise squared diff of marginal   (expected if random)
           = (n·Σv² - (Σv)²) / (n·(n-1)/2)
    """
    all_vals: list[float] = [v for unit in ratings_matrix for v in unit if v is not None]
    n = len(all_vals)
    if n < 2:
        return None

    # Observed disagreement: pairwise squared diffs within each unit
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

    # Expected disagreement: marginal distribution
    sum_v = sum(all_vals)
    sum_v2 = sum(v ** 2 for v in all_vals)
    de_numerator = n * sum_v2 - sum_v ** 2      # = Σ_{i<j} (v_i - v_j)²
    de_denominator = n * (n - 1) / 2             # number of unordered pairs
    if de_denominator == 0:
        return None
    D_e = de_numerator / de_denominator

    if D_e == 0:
        # All raters gave the same value everywhere → perfect agreement
        return 1.0 if D_o == 0 else None

    return _round(1 - D_o / D_e)


# ---------------------------------------------------------------------------
# Core per-model scoring
# ---------------------------------------------------------------------------

def _stage2_token_costs(results_dir: str, model_key: str, language: str) -> dict:
    """
    Sum the Stage 2 token cost for one report model across all its videos.

    Only Stage 2 is attributed to a model. Stage 1 runs once per video and its
    observations feed every report model, and one Stage 3 judge call scores all
    models jointly on anonymized reports — so neither is divisible by model.
    Folding them in would count the same shared spend once per model and make the
    per-model figures both inflated and incomparable. They are reported separately
    by pipeline_token_costs().
    """
    report_dir = os.path.join(_bench_b_dir(results_dir), "reports", language, model_key)
    totals = {"input": 0, "output": 0, "videos": 0, "videos_missing_cost": 0}
    if not os.path.isdir(report_dir):
        return totals

    for fname in sorted(os.listdir(report_dir)):
        if not fname.endswith(".json") or fname.endswith(".tmp"):
            continue
        try:
            with open(os.path.join(report_dir, fname), encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            continue
        if data.get("error") or data.get("status") == "in_progress":
            continue
        costs = data.get("token_costs") or {}
        tin, tout = costs.get("input_tokens"), costs.get("output_tokens")
        if tin is None and tout is None:
            totals["videos_missing_cost"] += 1
            continue
        totals["input"] += tin or 0
        totals["output"] += tout or 0
        totals["videos"] += 1
    return totals


def pipeline_token_costs(results_dir: str, language: str = "en") -> dict:
    """
    Token cost of the stages that are shared across every report model:
    Stage 1 observation extraction, and Stage 3 judging (broken out per judge).

    These are pipeline-level costs, not per-model ones — see _stage2_token_costs.
    """
    bench_dir = _bench_b_dir(results_dir)
    out: dict = {
        "stage1_observations": {"input": 0, "output": 0, "videos": 0},
        "stage3_judges": {},
    }

    obs_dir = os.path.join(bench_dir, "observations")
    if os.path.isdir(obs_dir):
        for fname in sorted(os.listdir(obs_dir)):
            if not fname.endswith(".json") or fname.endswith(".tmp"):
                continue
            try:
                with open(os.path.join(obs_dir, fname), encoding="utf-8") as f:
                    data = json.load(f)
            except Exception:
                continue
            if data.get("error") or data.get("status") == "in_progress":
                continue
            costs = data.get("token_costs") or {}
            out["stage1_observations"]["input"] += costs.get("input_tokens") or 0
            out["stage1_observations"]["output"] += costs.get("output_tokens") or 0
            out["stage1_observations"]["videos"] += 1

    judge_dir = os.path.join(bench_dir, "judge", language)
    if os.path.isdir(judge_dir):
        for fname in sorted(os.listdir(judge_dir)):
            if not fname.endswith(".json") or fname.endswith(".tmp"):
                continue
            try:
                with open(os.path.join(judge_dir, fname), encoding="utf-8") as f:
                    data = json.load(f)
            except Exception:
                continue
            if data.get("error"):
                continue
            for jk, costs in (data.get("judge_token_costs") or {}).items():
                slot = out["stage3_judges"].setdefault(
                    jk, {"input": 0, "output": 0, "videos": 0})
                slot["input"] += (costs or {}).get("input_tokens") or 0
                slot["output"] += (costs or {}).get("output_tokens") or 0
                slot["videos"] += 1

    return out


def score_model(results_dir: str, model_key: str, language: str = "en") -> dict:
    """
    Aggregate multi-judge scores for one model across all videos.

    Returns a summary dict with judge metrics, IRR (Krippendorff's alpha),
    qualitative signals (reasoning word count), and Stage 2 token cost.
    """
    bench_dir = _bench_b_dir(results_dir)
    judge_dir = os.path.join(bench_dir, "judge", language)

    if not os.path.isdir(judge_dir):
        raise FileNotFoundError(f"No judge results directory found at '{judge_dir}'")

    dimensions = ["clinical_accuracy", "safety", "clinical_actionability", "relational_empathy"]
    all_dimensions = dimensions

    # per-video mean score per dimension (for cross-video mean/std)
    per_video_dim_means: dict[str, list[float]] = {d: [] for d in all_dimensions}
    # per-video list of judge scores per dimension (for inter-judge std + Krippendorff)
    per_video_judge_scores: dict[str, list[list[float]]] = {d: [] for d in all_dimensions}

    final_scores: list[float] = []
    reasoning_word_counts: list[int] = []
    total_videos = 0
    failed_videos = 0

    for fname in sorted(os.listdir(judge_dir)):
        if not fname.endswith(".json") or fname.endswith(".tmp"):
            continue
        fpath = os.path.join(judge_dir, fname)
        with open(fpath, encoding="utf-8") as f:
            data = json.load(f)

        if data.get("error"):
            failed_videos += 1
            continue

        label_map: dict[str, str] = data.get("model_key_to_label", {})
        anon_label = label_map.get(model_key)
        if anon_label is None:
            continue

        # Collect scores from each judge for this video
        judge_scores_map: dict[str, dict] = data.get("judge_scores", {})
        if not judge_scores_map:
            legacy = data.get("scores", {})
            if legacy:
                judge_scores_map = {"legacy": legacy}

        video_dim_scores: dict[str, list[float]] = {d: [] for d in all_dimensions}
        for jk, jscores in judge_scores_map.items():
            model_scores = jscores.get(anon_label)
            if not model_scores:
                continue
            for dim in all_dimensions:
                val = _extract_dim_score(model_scores, dim)
                if val is not None:
                    video_dim_scores[dim].append(val)
            # Collect reasoning word counts from all dimension reasoning fields
            for dim in all_dimensions:
                dim_val = model_scores.get(dim)
                if isinstance(dim_val, dict):
                    reasoning = dim_val.get("reasoning", "")
                    if reasoning:
                        reasoning_word_counts.append(_word_count(reasoning))

        if not any(video_dim_scores[d] for d in dimensions):
            continue

        total_videos += 1

        video_dim_means: dict[str, Optional[float]] = {}
        for dim in all_dimensions:
            vals = video_dim_scores[dim]
            if vals:
                m = sum(vals) / len(vals)
                per_video_dim_means[dim].append(m)
                per_video_judge_scores[dim].append(vals)
                video_dim_means[dim] = m
            else:
                video_dim_means[dim] = None

        ca  = video_dim_means.get("clinical_accuracy")
        s   = video_dim_means.get("safety")
        act = video_dim_means.get("clinical_actionability")
        e   = video_dim_means.get("relational_empathy")
        if None not in (ca, s, act, e):
            final_scores.append(_final_judge_score(ca, s, act, e))

    # --------------- Aggregate ---------------
    judge_summary: dict = {
        "total_videos_judged": total_videos,
        "failed_videos": failed_videos,
    }

    all_inter_judge_stds: list[float] = []
    alpha_per_dim: list[float] = []

    for dim in all_dimensions:
        means = per_video_dim_means[dim]
        judge_summary[f"mean_{dim}"] = _round(_mean(means))
        judge_summary[f"std_{dim}"] = _round(_std(means))

        # Inter-judge std per video, averaged over videos
        ij_stds = [
            s for video_scores in per_video_judge_scores[dim]
            if (s := _std(video_scores)) is not None
        ]
        judge_summary[f"mean_inter_judge_std_{dim}"] = _round(_mean(ij_stds))
        all_inter_judge_stds.extend(ij_stds)

        # Krippendorff's Alpha — treat each video as a unit, judges as raters
        alpha = _krippendorff_alpha(per_video_judge_scores[dim])
        judge_summary[f"krippendorff_alpha_{dim}"] = alpha
        if alpha is not None:
            alpha_per_dim.append(alpha)

    judge_summary["mean_final_judge_score"] = _round(_mean(final_scores))
    judge_summary["std_final_judge_score"] = _round(_std(final_scores))
    judge_summary["mean_inter_judge_std"] = _round(_mean(all_inter_judge_stds))
    # Overall IRR: mean alpha across dimensions (omit None)
    judge_summary["krippendorff_alpha"] = _round(_mean(alpha_per_dim)) if alpha_per_dim else None
    # Qualitative: avg words in judge reasoning field
    judge_summary["mean_reasoning_word_count"] = _round(_mean([float(c) for c in reasoning_word_counts]), 1)

    tok = _stage2_token_costs(results_dir, model_key, language)
    n_tok = tok["videos"]

    return {
        "model_key": model_key,
        "judge": judge_summary,
        # Stage 2 only — this model's own report-generation spend. Field names match
        # Benchmark A's summary so downstream reporting can treat them alike.
        "total_input_tokens": tok["input"] if n_tok else None,
        "total_output_tokens": tok["output"] if n_tok else None,
        "mean_input_tokens": _round(tok["input"] / n_tok, 1) if n_tok else None,
        "mean_output_tokens": _round(tok["output"] / n_tok, 1) if n_tok else None,
        "token_cost_videos": n_tok,
        # Providers that report no usage are counted here rather than silently
        # averaged as zero, which would understate the model.
        "token_cost_videos_missing": tok["videos_missing_cost"],
    }


def score_and_save_model(results_dir: str, model_key: str, language: str = "en") -> dict:
    """Score model and write summary_{language}_{model_key}.json atomically."""
    scores = score_model(results_dir, model_key, language=language)
    path = os.path.join(_bench_b_dir(results_dir), f"summary_{language}_{model_key}.json")
    _save_atomic(path, scores)
    return scores


# ---------------------------------------------------------------------------
# Pairwise Win/Tie/Loss
# ---------------------------------------------------------------------------

def _compute_win_tie_loss(
    results_dir: str,
    model_keys: list[str],
    language: str = "en",
) -> dict:
    """
    For each ordered pair (A, B), count videos where A's mean judge score
    (final_judge_score averaged across all judges) is strictly greater than B's (W),
    equal (T), or less (L).

    Output format:
      {
        "model_a_vs_model_b": {
          "wins": int, "ties": int, "losses": int,
          "win_rate": float,   # W / (W+T+L)
          "n_videos": int
        },
        ...
      }
    """
    bench_dir = _bench_b_dir(results_dir)
    judge_dir = os.path.join(bench_dir, "judge", language)

    # Build per-video per-model final score: {video_id: {model_key: score}}
    video_scores: dict[str, dict[str, float]] = {}

    for fname in sorted(os.listdir(judge_dir)):
        if not fname.endswith(".json") or fname.endswith(".tmp"):
            continue
        fpath = os.path.join(judge_dir, fname)
        with open(fpath, encoding="utf-8") as f:
            data = json.load(f)

        if data.get("error"):
            continue

        video_id: str = data.get("video_id", fname.removesuffix(".json"))
        label_map: dict[str, str] = data.get("model_key_to_label", {})
        judge_scores_map: dict[str, dict] = data.get("judge_scores", {})
        if not judge_scores_map:
            legacy = data.get("scores", {})
            if legacy:
                judge_scores_map = {"legacy": legacy}

        for mk in model_keys:
            anon = label_map.get(mk)
            if anon is None:
                continue
            # Collect per-judge final scores for this model and video
            all_judge_finals: list[float] = []
            for jk, jscores in judge_scores_map.items():
                ms = jscores.get(anon, {})
                ca  = _extract_dim_score(ms, "clinical_accuracy")
                s   = _extract_dim_score(ms, "safety")
                act = _extract_dim_score(ms, "clinical_actionability")
                e   = _extract_dim_score(ms, "relational_empathy")
                if None not in (ca, s, act, e):
                    all_judge_finals.append(_final_judge_score(ca, s, act, e))
            if all_judge_finals:
                video_scores.setdefault(video_id, {})[mk] = sum(all_judge_finals) / len(all_judge_finals)

    # Compute W/T/L for all ordered pairs
    wtl: dict[str, dict] = {}
    for i, a in enumerate(model_keys):
        for j, b in enumerate(model_keys):
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
                "wins": wins,
                "ties": ties,
                "losses": losses,
                "win_rate": _round(wins / n) if n > 0 else None,
                "n_videos": n,
            }

    return wtl


# ---------------------------------------------------------------------------
# Leaderboard assembly
# ---------------------------------------------------------------------------

def assemble_leaderboard(results_dir: str, language: str = "en") -> dict:
    """
    Glob all summary_{language}_*.json files, rank by mean_final_judge_score, and compute
    pairwise Win/Tie/Loss matrix.
    Writes leaderboard_{language}.json atomically.
    """
    pattern = os.path.join(_bench_b_dir(results_dir), f"summary_{language}_*.json")
    summary_files = glob.glob(pattern)

    models_scores: dict[str, dict] = {}
    errors: dict[str, str] = {}

    for sf in summary_files:
        try:
            with open(sf, encoding="utf-8") as f:
                data = json.load(f)
            model_key = data.get("model_key") or (
                os.path.basename(sf).removeprefix("summary_").removesuffix(".json")
            )
            models_scores[model_key] = data
        except Exception as e:
            errors[sf] = str(e)

    ranking = sorted(
        models_scores.keys(),
        key=lambda k: models_scores[k].get("judge", {}).get("mean_final_judge_score") or -1,
        reverse=True,
    )

    win_tie_loss: dict = {}
    if len(models_scores) > 1:
        try:
            win_tie_loss = _compute_win_tie_loss(results_dir, list(models_scores.keys()), language=language)
        except Exception as e:
            errors["win_tie_loss"] = str(e)

    try:
        pipeline_tokens = pipeline_token_costs(results_dir, language=language)
    except Exception as e:
        pipeline_tokens = {}
        errors["pipeline_token_costs"] = str(e)

    leaderboard = {
        "models": models_scores,
        "ranking": ranking,
        "win_tie_loss": win_tie_loss,
        # Stage 1 + Stage 3 spend, shared across every model rather than attributed
        # to any one of them.
        "pipeline_token_costs": pipeline_tokens,
        "errors": errors,
    }

    path = os.path.join(_bench_b_dir(results_dir), f"leaderboard_{language}.json")
    _save_atomic(path, leaderboard)
    return leaderboard


# ---------------------------------------------------------------------------
# Print summary
# ---------------------------------------------------------------------------

def print_summary_table(leaderboard: dict) -> None:
    """Print a readable leaderboard to stdout."""
    models = leaderboard.get("models", {})
    ranking = leaderboard.get("ranking", [])
    wtl = leaderboard.get("win_tie_loss", {})

    header = (
        f"{'Rank':<5} {'Model':<28} {'Score':>7} {'StdDev':>7} "
        f"{'CA':>6} {'Safe':>5} {'Act':>5} {'Emp':>6}"
        f" "
        f"{'IJ-Std':>7} {'α-IRR':>7} {'Reas-Len':>9} "
        f"{'In Tokens':>11} {'Out Tokens':>11}"
    )
    print("\n" + "=" * len(header))
    print("Benchmark B Results")
    print("=" * len(header))
    print(header)
    print("-" * len(header))

    for i, key in enumerate(ranking, 1):
        m = models[key]
        j = m.get("judge", {})
        print(
            f"{i:<5} {key:<28} "
            f"{_fmt(j.get('mean_final_judge_score')):>7} "
            f"{_fmt(j.get('std_final_judge_score')):>7} "
            f"{_fmt(j.get('mean_clinical_accuracy')):>6} "
            f"{_fmt(j.get('mean_safety')):>5} "
            f"{_fmt(j.get('mean_clinical_actionability')):>5} "
            f"{_fmt(j.get('mean_relational_empathy')):>6}"
            f" "
            f"{_fmt(j.get('mean_inter_judge_std')):>7} "
            f"{_fmt(j.get('krippendorff_alpha')):>7} "
            f"{_fmt_int(j.get('mean_reasoning_word_count')):>9} "
            f"{_fmt_tokens(m.get('total_input_tokens')):>11} "
            f"{_fmt_tokens(m.get('total_output_tokens')):>11}"
        )

    print("=" * len(header))
    _print_token_costs(leaderboard, models, ranking)

    # Win/Tie/Loss summary (only if data available)
    if wtl and len(ranking) > 1:
        print("\nHead-to-Head Win/Tie/Loss (on mean_final_judge_score):")
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
                matchup = f"{a} vs {b}"
                print(
                    f"  {matchup:<50} "
                    f"{entry['wins']:>4} {entry['ties']:>4} {entry['losses']:>4} "
                    f"{win_pct:>7}"
                )
    print()


def _print_token_costs(leaderboard: dict, models: dict, ranking: list) -> None:
    """
    Explain the token columns and report the shared Stage 1 / Stage 3 spend.

    Printed rather than folded into the table because these costs belong to the
    pipeline, not to any model: one observation extraction per video serves every
    report model, and one judge call scores them all at once.
    """
    print("\n  In/Out Tokens above are Stage 2 only — this model's own report "
          "generation.")

    missing = [(k, models[k].get("token_cost_videos_missing", 0)) for k in ranking]
    missing = [(k, n) for k, n in missing if n]
    if missing:
        print("  Videos whose provider returned no usage data (excluded from the "
              "totals above):")
        for k, n in missing:
            print(f"    {k:<28} {n}")

    pt = leaderboard.get("pipeline_token_costs") or {}
    obs = pt.get("stage1_observations") or {}
    judges = pt.get("stage3_judges") or {}
    if not obs.get("videos") and not judges:
        return

    print("\n  Shared pipeline cost (not attributable to any single model):")
    print(f"    {'stage':<30}{'in':>13}{'out':>13}{'calls':>8}")
    if obs.get("videos"):
        print(f"    {'Stage 1 observations':<30}"
              f"{obs['input']:>13,}{obs['output']:>13,}{obs['videos']:>8}")
    for jk, c in sorted(judges.items()):
        print(f"    {'Stage 3 judge: ' + jk:<30}"
              f"{c['input']:>13,}{c['output']:>13,}{c['videos']:>8}")

    grand_in = obs.get("input", 0) + sum(c["input"] for c in judges.values()) \
        + sum(models[k].get("total_input_tokens") or 0 for k in ranking)
    grand_out = obs.get("output", 0) + sum(c["output"] for c in judges.values()) \
        + sum(models[k].get("total_output_tokens") or 0 for k in ranking)
    print(f"    {'TOTAL (all stages, all models)':<30}{grand_in:>13,}{grand_out:>13,}")
    print()


def _fmt_tokens(v: Optional[float]) -> str:
    return f"{int(v):,}" if v is not None else "N/A"


def _fmt(v: Optional[float]) -> str:
    return f"{v:.4f}" if v is not None else "  N/A"


def _fmt_int(v: Optional[float]) -> str:
    return f"{v:.0f}" if v is not None else "  N/A"
