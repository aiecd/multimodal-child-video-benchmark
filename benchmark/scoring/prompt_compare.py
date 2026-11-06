"""
Scoring for the prompt comparisons and the end-to-end comparison.

  compare_a()        — labeling prompts on the same model
  score_b_prompts()  — report rubric, one report per arm (Table 4 panel B)
  score_e2e()        — Table 5 rubric, reports judged side by side

Every comparison is paired: an arm is only compared with the other on clips where
both have a result, so a flaky arm cannot be measured on an easier subset.
Deltas are always reported as (other arm − baseline arm).

Statistics
----------
  mean_diff        mean of per-clip differences
  ci95_mean_diff   percentile bootstrap (10,000 resamples, fixed seed)
  wins/ties/losses clips where the other arm scored higher / equal / lower
  sign_test_p      two-sided exact binomial test on wins vs losses (ties dropped)
  mcnemar_p        exact McNemar on discordant per-label correctness (labeling prompts)

Composite formulas are imported from the B and C scoring modules unchanged, so a
score here is directly comparable to the main Benchmark B / C leaderboards.
"""

from __future__ import annotations

import json
import math
import os
import random
from typing import Callable, Optional

from benchmark.models.base import LABEL_KEYS
from benchmark.scoring.benchmark_b import (
    _extract_dim_score,
    _final_judge_score,
    _krippendorff_alpha,
)
from benchmark.scoring.benchmark_c import _cross_model_score

B_DIMENSIONS = ["clinical_accuracy", "safety", "clinical_actionability", "relational_empathy"]
C_DIMENSIONS = ["safety", "clinical_actionability", "relational_empathy"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _round(v: Optional[float], n: int = 4) -> Optional[float]:
    return round(v, n) if v is not None else None


def _mean(values: list[float]) -> Optional[float]:
    return sum(values) / len(values) if values else None


def _save_atomic(path: str, data: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


def _iter_json(dir_path: str):
    """Yield (stem, data) for every readable, finished, error-free result file."""
    if not os.path.isdir(dir_path):
        return
    for fname in sorted(os.listdir(dir_path)):
        if not fname.endswith(".json") or fname.startswith("summary_"):
            continue
        try:
            with open(os.path.join(dir_path, fname), encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            continue
        if data.get("status") == "in_progress" or data.get("error") is not None:
            continue
        yield fname.removesuffix(".json"), data


def binomial_two_sided_p(k: int, n: int) -> Optional[float]:
    """Exact two-sided binomial test of k successes in n trials against p = 0.5."""
    if n == 0:
        return None
    tail = sum(math.comb(n, i) for i in range(min(k, n - k) + 1)) / 2 ** n
    return round(min(1.0, 2 * tail), 6)


def paired_comparison(
    baseline: dict[str, float],
    other: dict[str, float],
    n_boot: int = 10_000,
    seed: int = 0,
) -> dict:
    """Paired comparison of two per-clip score maps on their common clips."""
    common = sorted(set(baseline) & set(other))
    diffs = [other[c] - baseline[c] for c in common]
    wins = sum(1 for d in diffs if d > 0)
    losses = sum(1 for d in diffs if d < 0)
    ties = len(diffs) - wins - losses

    ci = None
    if len(diffs) >= 2:
        rng = random.Random(seed)
        n = len(diffs)
        boots = sorted(sum(rng.choices(diffs, k=n)) / n for _ in range(n_boot))
        ci = [_round(boots[int(0.025 * n_boot)]), _round(boots[int(0.975 * n_boot) - 1])]

    return {
        "n_clips": len(common),
        "mean_baseline": _round(_mean([baseline[c] for c in common])),
        "mean_other": _round(_mean([other[c] for c in common])),
        "mean_diff": _round(_mean(diffs)),
        "ci95_mean_diff": ci,
        "wins": wins,
        "ties": ties,
        "losses": losses,
        "sign_test_p": binomial_two_sided_p(wins, wins + losses),
    }


# ---------------------------------------------------------------------------
# Labeling prompt comparison
# ---------------------------------------------------------------------------

def _as_bool(v) -> Optional[bool]:
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        return v.lower() == "true"
    return None


def _f1(tp: int, fp: int, fn: int) -> float:
    # Zero-division → 0, matching scoring/benchmark_a.py
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return 2 * p * r / (p + r) if p + r else 0.0


def compare_a(results_dir: str, baseline_key: str, other_key: str) -> dict:
    """
    Paired comparison of two Benchmark A runs on the clips both completed.

    Per label: F1 for each arm, and exact McNemar on correctness (a clip-label is
    correct when the prediction is non-null and equals ground truth; an abstention
    counts as incorrect). Macro F1 averages over all 10 labels, as in Benchmark A.
    """
    bench_dir = os.path.join(results_dir, "benchmark_a")
    runs = {
        key: dict(_iter_json(os.path.join(bench_dir, key)))
        for key in (baseline_key, other_key)
    }
    common = sorted(set(runs[baseline_key]) & set(runs[other_key]))

    per_label: dict[str, dict] = {}
    pooled_b_only = pooled_o_only = 0
    exact_b: dict[str, float] = {}
    exact_o: dict[str, float] = {}

    for label in LABEL_KEYS:
        counts = {k: {"tp": 0, "fp": 0, "fn": 0} for k in runs}
        b_only = o_only = 0
        for clip in common:
            gt = bool(runs[baseline_key][clip].get("ground_truth", {}).get(label))
            correct = {}
            for key in runs:
                pred = _as_bool(runs[key][clip].get("asd_behavioral_metrics", {}).get(label))
                c = counts[key]
                if pred is True and gt:
                    c["tp"] += 1
                elif pred is True and not gt:
                    c["fp"] += 1
                elif pred is not True and gt:
                    c["fn"] += 1
                correct[key] = pred is not None and pred == gt
            if correct[baseline_key] and not correct[other_key]:
                b_only += 1
            elif correct[other_key] and not correct[baseline_key]:
                o_only += 1
        pooled_b_only += b_only
        pooled_o_only += o_only
        f1_b = _f1(**counts[baseline_key])
        f1_o = _f1(**counts[other_key])
        per_label[label] = {
            "f1_baseline": _round(f1_b),
            "f1_other": _round(f1_o),
            "f1_diff": _round(f1_o - f1_b),
            "support": counts[baseline_key]["tp"] + counts[baseline_key]["fn"],
            "baseline_only_correct": b_only,
            "other_only_correct": o_only,
            "mcnemar_p": binomial_two_sided_p(o_only, b_only + o_only),
        }

    # Clip-level: fraction of the 10 labels each arm got right, paired across clips
    for clip in common:
        for key, out in ((baseline_key, exact_b), (other_key, exact_o)):
            data = runs[key][clip]
            gt = data.get("ground_truth", {})
            preds = data.get("asd_behavioral_metrics", {})
            right = sum(
                1 for label in LABEL_KEYS
                if (p := _as_bool(preds.get(label))) is not None and p == bool(gt.get(label))
            )
            out[clip] = right / len(LABEL_KEYS)

    macro_b = _mean([v["f1_baseline"] for v in per_label.values()])
    macro_o = _mean([v["f1_other"] for v in per_label.values()])
    return {
        "baseline": baseline_key,
        "other": other_key,
        "n_common_clips": len(common),
        "n_clips": {k: len(v) for k, v in runs.items()},
        "macro_f1_baseline": _round(macro_b),
        "macro_f1_other": _round(macro_o),
        "macro_f1_diff": _round(macro_o - macro_b) if common else None,
        "pooled_mcnemar": {
            "baseline_only_correct": pooled_b_only,
            "other_only_correct": pooled_o_only,
            "p": binomial_two_sided_p(pooled_o_only, pooled_b_only + pooled_o_only),
        },
        "per_clip_label_accuracy": paired_comparison(exact_b, exact_o),
        "per_label": per_label,
    }


def print_compare_a(result: dict) -> None:
    b, o = result["baseline"], result["other"]
    print(f"\nLabeling prompt comparison  ({result['n_common_clips']} common clips)")
    print(f"  baseline: {b}   other: {o}")
    print(f"  Macro F1: {_fmt(result['macro_f1_baseline'])} → {_fmt(result['macro_f1_other'])}"
          f"  (Δ {_fmt_signed(result['macro_f1_diff'])})")
    pm = result["pooled_mcnemar"]
    print(f"  Pooled McNemar (all labels): baseline-only correct {pm['baseline_only_correct']}, "
          f"other-only correct {pm['other_only_correct']}, p={_fmt(pm['p'])}")
    acc = result["per_clip_label_accuracy"]
    print(f"  Per-clip label accuracy: Δ {_fmt_signed(acc['mean_diff'])} "
          f"CI95 {acc['ci95_mean_diff']}  W/T/L {acc['wins']}/{acc['ties']}/{acc['losses']}"
          f"  sign p={_fmt(acc['sign_test_p'])}")
    print(f"\n  {'Label':<45} {'F1 base':>8} {'F1 other':>9} {'Δ':>8} {'supp':>5} {'McNemar p':>10}")
    print("  " + "-" * 89)
    for label, v in result["per_label"].items():
        print(f"  {label:<45} {_fmt(v['f1_baseline']):>8} {_fmt(v['f1_other']):>9} "
              f"{_fmt_signed(v['f1_diff']):>8} {v['support']:>5} {_fmt(v['mcnemar_p']):>10}")
    print()


# ---------------------------------------------------------------------------
# Report-prompt and end-to-end comparisons — judged report quality
# ---------------------------------------------------------------------------

def _arm_scores(
    judge_scores: dict[str, dict],
    dimensions: list[str],
    composite: Callable[..., float],
) -> tuple[dict[str, Optional[float]], dict[str, list[float]], dict[str, float]]:
    """
    Aggregate one report's scores across judges.

    judge_scores: {judge_key: {dim: {...}}} for a single report.
    Returns (per-dim judge means, per-dim judge values, per-judge composites).
    The clip composite is computed from the judge-mean dimensions, as in B and C.
    """
    dim_vals: dict[str, list[float]] = {d: [] for d in dimensions}
    per_judge: dict[str, float] = {}
    for jk, scores in judge_scores.items():
        vals = {d: _extract_dim_score(scores or {}, d) for d in dimensions}
        for d, v in vals.items():
            if v is not None:
                dim_vals[d].append(v)
        if None not in vals.values():
            per_judge[jk] = composite(*(vals[d] for d in dimensions))
    dim_means = {d: _mean(v) for d, v in dim_vals.items()}
    return dim_means, dim_vals, per_judge


def _summarize_arms(
    per_clip: dict[str, dict[str, dict[str, dict]]],
    arms: list[str],
    baseline_arm: str,
    dimensions: list[str],
    composite: Callable[..., float],
) -> dict:
    """
    per_clip: {clip_id: {arm: {judge_key: scores}}}

    Returns per-arm means over the common clip set (clips where every arm has a
    complete composite), per-dimension and per-judge paired deltas vs the baseline.
    """
    clip_composite: dict[str, dict[str, float]] = {a: {} for a in arms}
    clip_dims: dict[str, dict[str, dict[str, float]]] = {a: {d: {} for d in dimensions} for a in arms}
    clip_judge: dict[str, dict[str, dict[str, float]]] = {a: {} for a in arms}
    judge_matrix: dict[str, dict[str, list[list[float]]]] = {a: {d: [] for d in dimensions} for a in arms}

    for clip, by_arm in per_clip.items():
        for arm in arms:
            if arm not in by_arm:
                continue
            dim_means, dim_vals, per_judge = _arm_scores(by_arm[arm], dimensions, composite)
            if None in dim_means.values():
                continue
            clip_composite[arm][clip] = composite(*(dim_means[d] for d in dimensions))
            for d in dimensions:
                clip_dims[arm][d][clip] = dim_means[d]
                judge_matrix[arm][d].append(dim_vals[d])
            for jk, v in per_judge.items():
                clip_judge[arm].setdefault(jk, {})[clip] = v

    common = set.intersection(*(set(clip_composite[a]) for a in arms)) if arms else set()

    def _restrict(m: dict[str, float]) -> dict[str, float]:
        return {c: v for c, v in m.items() if c in common}

    summary_arms: dict[str, dict] = {}
    for arm in arms:
        summary_arms[arm] = {
            "clips_scored": len(clip_composite[arm]),
            "mean_composite": _round(_mean(list(_restrict(clip_composite[arm]).values()))),
            **{
                f"mean_{d}": _round(_mean(list(_restrict(clip_dims[arm][d]).values())))
                for d in dimensions
            },
            "per_judge_mean_composite": {
                jk: _round(_mean(list(_restrict(m).values())))
                for jk, m in sorted(clip_judge[arm].items())
            },
            # Inter-rater reliability over every clip this arm has scores for
            "krippendorff_alpha": {d: _krippendorff_alpha(judge_matrix[arm][d]) for d in dimensions},
        }

    comparisons: dict[str, dict] = {}
    for arm in arms:
        if arm == baseline_arm:
            continue
        comp = {"composite": paired_comparison(
            _restrict(clip_composite[baseline_arm]), _restrict(clip_composite[arm]))}
        for d in dimensions:
            comp[d] = paired_comparison(
                _restrict(clip_dims[baseline_arm][d]), _restrict(clip_dims[arm][d]))
        # Per-judge composite deltas: a delta that only one judge sees is weak evidence
        comp["per_judge_composite"] = {
            jk: paired_comparison(
                _restrict(clip_judge[baseline_arm].get(jk, {})),
                _restrict(clip_judge[arm].get(jk, {})),
            )
            for jk in sorted(set(clip_judge[baseline_arm]) | set(clip_judge[arm]))
        }
        comparisons[f"{arm}_vs_{baseline_arm}"] = comp

    return {
        "baseline_arm": baseline_arm,
        "common_clips": len(common),
        "arms": summary_arms,
        "comparisons": comparisons,
    }


def stage_token_costs(paths_by_arm: dict[str, list[str]]) -> dict:
    """Sum token_costs and timing_s over the Stage 1/2 files that make up each arm."""
    out: dict = {}
    for arm, dirs in paths_by_arm.items():
        stages = []
        for d in dirs:
            tin = tout = n = 0
            secs: list[float] = []
            for _, data in _iter_json(d):
                costs = data.get("token_costs") or {}
                tin += costs.get("input_tokens") or 0
                tout += costs.get("output_tokens") or 0
                if data.get("timing_s") is not None:
                    secs.append(data["timing_s"])
                n += 1
            stages.append({
                "stage_dir": os.path.relpath(d, os.path.dirname(os.path.dirname(d))),
                "files": n,
                "input_tokens": tin,
                "output_tokens": tout,
                "mean_timing_s": _round(_mean(secs), 2),
            })
        out[arm] = stages
    return out


def score_b_prompts(
    judge_dir: str,
    arms: list[str],
    baseline_arm: str,
    out_path: str,
    stage_dirs: dict[str, list[str]],
) -> dict:
    """
    Judge files: {judge_dir}/{arm}/{clip_id}.json, each holding one report
    scored against its own arm's observations on the Benchmark B rubric.
    Composite = Benchmark B mean_final_judge_score formula.
    """
    per_clip: dict[str, dict[str, dict]] = {}
    for arm in arms:
        for clip, data in _iter_json(os.path.join(judge_dir, arm)):
            scores = data.get("judge_scores") or {}
            if scores:
                per_clip.setdefault(clip, {})[arm] = scores
    result = _summarize_arms(per_clip, arms, baseline_arm, B_DIMENSIONS, _final_judge_score)
    result["composite_formula"] = "0.40×CA/12 + 0.25×S/3 + 0.25×Act/7 + 0.10×(E−7)/28 (Benchmark B)"
    result["stage_costs"] = stage_token_costs(stage_dirs)
    _save_atomic(out_path, result)
    return result


def score_e2e(
    judge_dir: str,
    arms: list[str],
    baseline_arm: str,
    out_path: str,
    stage_dirs: dict[str, list[str]],
) -> dict:
    """
    Judge files: {judge_dir}/{clip_id}.json in the Table 5 format
    (both arms judged side by side, anonymized). Composite = Benchmark C
    cross-model score.
    """
    per_clip: dict[str, dict[str, dict]] = {}
    for clip, data in _iter_json(judge_dir):
        label_map: dict[str, str] = data.get("model_key_to_label", {})
        for arm in arms:
            label = label_map.get(arm)
            if label is None:
                continue
            scores = {
                jk: js.get(label)
                for jk, js in (data.get("judge_scores") or {}).items()
                if js and js.get(label)
            }
            if scores:
                per_clip.setdefault(clip, {})[arm] = scores
    result = _summarize_arms(per_clip, arms, baseline_arm, C_DIMENSIONS, _cross_model_score)
    result["composite_formula"] = "0.40×S/3 + 0.40×Act/7 + 0.20×(E−7)/28 (Benchmark C)"
    result["stage_costs"] = stage_token_costs(stage_dirs)
    _save_atomic(out_path, result)
    return result


def print_judged_comparison(result: dict, title: str) -> None:
    arms = result["arms"]
    dims = [k.removeprefix("mean_") for k in next(iter(arms.values()), {})
            if k.startswith("mean_") and k != "mean_composite"]
    print(f"\n{title}  ({result['common_clips']} common clips)")
    print(f"  {result['composite_formula']}")
    header = f"  {'Arm':<28} {'Composite':>9} " + " ".join(f"{d[:12]:>12}" for d in dims) + f" {'scored':>7}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for arm, s in arms.items():
        print(f"  {arm:<28} {_fmt(s['mean_composite']):>9} "
              + " ".join(f"{_fmt(s.get('mean_' + d)):>12}" for d in dims)
              + f" {s['clips_scored']:>7}")

    for name, comp in result["comparisons"].items():
        print(f"\n  {name}  (Δ = other − baseline)")
        for key in ["composite", *dims]:
            c = comp[key]
            print(f"    {key:<24} Δ {_fmt_signed(c['mean_diff']):>8}  CI95 {str(c['ci95_mean_diff']):<20} "
                  f"W/T/L {c['wins']}/{c['ties']}/{c['losses']}  sign p={_fmt(c['sign_test_p'])}")
        print("    per-judge composite Δ:")
        for jk, c in comp["per_judge_composite"].items():
            print(f"      {jk:<26} Δ {_fmt_signed(c['mean_diff']):>8}  n={c['n_clips']}  "
                  f"W/T/L {c['wins']}/{c['ties']}/{c['losses']}")

    print("\n  Stage cost per arm (tokens summed over cached files):")
    for arm, stages in result.get("stage_costs", {}).items():
        for st in stages:
            print(f"    {arm:<28} {st['stage_dir']:<52} in {st['input_tokens']:>12,}  "
                  f"out {st['output_tokens']:>11,}  files {st['files']:>4}  "
                  f"mean {st['mean_timing_s']}s")
    print()


def _fmt(v: Optional[float]) -> str:
    return f"{v:.4f}" if v is not None else "N/A"


def _fmt_signed(v: Optional[float]) -> str:
    return f"{v:+.4f}" if v is not None else "N/A"
