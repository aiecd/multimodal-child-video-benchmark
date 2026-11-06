"""
Compute the paper's tables from the released outputs. No model calls.

Statistics (Methods, Statistical Analysis): 95% percentile CIs from 10,000
source-video-clustered bootstrap replicates. Source videos are resampled with
replacement, all clips of a sampled video are kept, and statistics and paired
differences are recomputed in every replicate (label counts rebuilt for macro-F1).

Subcommands and outputs (all under benchmark/results/paper_tables/):
  (none)    stage1/  Table 1, Supplementary Tables S5 and S6, Figure 2 counts,
                     Supplementary Note S1 label agreement
  table2    table2/  Table 2 and Supplementary Table S3
  table4a   table4a/ Table 4 panel A and the Supplementary Note S1 per-label results
  table4b   table4b/ Table 4 panel B
  table5    table5/  Table 5 and token counts

Run every subcommand with: bash reproduce.sh
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import Counter
from typing import Optional

import numpy as np
import pandas as pd

from benchmark.data.loader import parse_clip_id
from benchmark.models.base import LABEL_KEYS

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
NINE = [l for l in LABEL_KEYS if l != "Background"]
STAGE1_MODELS = {
    "gemini-3.7-flash": "Gemini 3.7 Flash",
    "muse-spark-1.2": "Muse Spark 1.2",
    "qwen-3.8-max": "Qwen 3.8 Max",
    "pegasus-1.5": "Pegasus 1.5",
    "kimi-k3": "Kimi K3",
}
EYE = "Absence or Avoidance of Eye Contact"
NR = "Non-Responsiveness to Verbal Interaction"
TP, FP, FN, TN = range(4)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def path(rel: str) -> str:
    return rel if os.path.isabs(rel) else os.path.join(ROOT, rel)


def source_video(clip_id: str) -> str:
    return parse_clip_id(clip_id)[0]


def read_ids(csv_rel: str) -> list[str]:
    return pd.read_csv(path(csv_rel))["Video_ID"].astype(str).str.strip().tolist()


def load_ground_truth() -> dict[str, dict[str, bool]]:
    df = pd.read_csv(path("dataset/csvs/dataset.csv"))
    df["Video_ID"] = df["Video_ID"].astype(str).str.strip()
    return {r["Video_ID"]: {l: bool(int(r[l])) for l in LABEL_KEYS} for _, r in df.iterrows()}


def load_results(model_dir: str) -> tuple[dict[str, dict], set[str]]:
    """(valid records by clip, clip IDs that errored). Placeholders are ignored."""
    valid, failed = {}, set()
    if not os.path.isdir(model_dir):
        raise FileNotFoundError(f"No results at {model_dir}")
    for fname in os.listdir(model_dir):
        if not fname.endswith(".json") or fname.startswith(("summary", "leaderboard")):
            continue
        try:
            with open(os.path.join(model_dir, fname), encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            continue
        clip = fname[:-5]
        if data.get("status") == "in_progress":
            continue
        if data.get("error"):
            failed.add(clip)
        else:
            valid[clip] = data
    return valid, failed


def as_bool(v):
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        return v.strip().lower() == "true"
    return None


# ---------------------------------------------------------------------------
# Confusion counts and metrics
# ---------------------------------------------------------------------------

def clip_counts(records: dict, clips: list[str], labels: list[str],
                gt: dict, failures_as_errors: bool = False) -> np.ndarray:
    """
    Per-clip, per-label outcome indicators, shape (n_clips, n_labels, 4) = TP/FP/FN/TN.

    A null prediction for a label (abstention) contributes nothing for that label,
    matching benchmark/scoring/benchmark_a.py. With failures_as_errors, a clip the
    system failed to process is scored as wrong on every label.
    """
    out = np.zeros((len(clips), len(labels), 4), dtype=np.int32)
    for i, c in enumerate(clips):
        rec = records.get(c)
        for j, l in enumerate(labels):
            truth = gt[c][l]
            if rec is None:
                if not failures_as_errors:
                    continue
                pred = not truth
            else:
                pred = as_bool(rec.get("asd_behavioral_metrics", {}).get(l))
                if pred is None:
                    continue
            out[i, j, (TP if truth else FP) if pred else (FN if truth else TN)] = 1
    return out


def metrics(counts: np.ndarray) -> dict[str, np.ndarray]:
    """
    Macro and per-label metrics from summed counts of shape (..., n_labels, 4).
    Zero-division gives 0, as in the benchmark scorer. Works on bootstrap stacks.
    """
    tp, fp, fn, tn = (counts[..., k].astype(float) for k in range(4))
    with np.errstate(divide="ignore", invalid="ignore"):
        prec = np.where(tp + fp > 0, tp / (tp + fp), 0.0)
        rec = np.where(tp + fn > 0, tp / (tp + fn), 0.0)
        f1 = np.where(prec + rec > 0, 2 * prec * rec / (prec + rec), 0.0)
        n = tp + fp + fn + tn
        acc = np.where(n > 0, (tp + tn) / n, np.nan)
        fpr = np.where(fp + tn > 0, fp / (fp + tn), np.nan)
    return {
        "macro_f1": f1.mean(axis=-1),
        "macro_precision": prec.mean(axis=-1),
        "macro_recall": rec.mean(axis=-1),
        "macro_accuracy": np.nanmean(acc, axis=-1),
        "per_label_precision": prec, "per_label_recall": rec,
        "per_label_f1": f1, "per_label_fpr": fpr,
    }


class ClusteredBootstrap:
    """Source-video-clustered resampling weights for a fixed list of clips."""

    def __init__(self, clips: list[str], n_boot: int, seed: int):
        videos = sorted({source_video(c) for c in clips})
        index = {v: i for i, v in enumerate(videos)}
        self.clip_video = np.array([index[source_video(c)] for c in clips])
        self.n_videos = len(videos)
        rng = np.random.default_rng(seed)
        # weights[b, v] = how many times video v is drawn in resample b
        self.weights = rng.multinomial(
            self.n_videos, np.full(self.n_videos, 1 / self.n_videos), size=n_boot
        ).astype(np.float64)

    def resample(self, per_clip: np.ndarray) -> np.ndarray:
        """Sum per-clip arrays (n_clips, ...) within videos, then within resamples."""
        shape = per_clip.shape[1:]
        per_video = np.zeros((self.n_videos,) + shape, dtype=np.float64)
        np.add.at(per_video, self.clip_video, per_clip)
        flat = per_video.reshape(self.n_videos, -1)
        return (self.weights @ flat).reshape((-1,) + shape)


def ci(samples: np.ndarray) -> tuple[float, float]:
    lo, hi = np.nanpercentile(samples, [2.5, 97.5])
    return float(lo), float(hi)




# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

# Internal field names -> terms used in the manuscript
PAPER_TERMS = {"clinical_accuracy": "evidence_alignment", "clinical_actionability": "actionability",
               "macro_accuracy": "label_wise_accuracy", "clips_processed": "valid_outputs"}


def _paper_term(s):
    if not isinstance(s, str):
        return s
    for old, new in PAPER_TERMS.items():
        if s == old or s.startswith(old + "_"):
            return new + s[len(old):]
    return s


def write_csv(out_dir: str, name: str, rows: list[dict]) -> None:
    if not rows:
        return
    rows = [{_paper_term(k): _paper_term(v) for k, v in x.items()} for x in rows]
    with open(os.path.join(out_dir, name), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def write_run_info(out_dir: str, info: dict) -> None:
    with open(os.path.join(out_dir, "run_info.json"), "w") as f:
        json.dump({**info, "python": sys.version.split()[0], "numpy": np.__version__,
                   "pandas": pd.__version__}, f, indent=2)


def r(v, nd=9):
    return None if v is None or (isinstance(v, float) and np.isnan(v)) else round(float(v), nd)






# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------

def screening_tables(out_dir: str) -> tuple[list[str], list[str]]:
    included = [l.strip() for l in open(path("dataset/clips_screened_pass.txt")) if l.strip()]
    excl_rows = list(csv.DictReader(open(path("dataset/clips_screened_excluded.txt")), delimiter="\t"))
    excluded = [r_["clip_id"] for r_ in excl_rows]
    failed_by = {r_["clip_id"]: [c.strip() for c in r_["failed_criteria"].split(",")] for r_ in excl_rows}
    probe = json.load(open(path("dataset/clips_video/.probe_cache.json")))
    props = {k[:-4]: v for k, v in probe.items()}

    criteria = [  # (label in the excluded file, probe field, cut, unit, scale, format)
        ("Duration", "duration", "Bottom 2.5% and top 2.5%", "s", 1, "{:g}"),
        ("Bitrate", "bitrate", "Bottom 5%", "kbps", 1 / 1000, "{:,.0f}"),
        ("Frame count", "n_frames", "Bottom 2.5% and top 2.5%", "frames", 1, "{:,.0f}"),
        ("Frame rate", "fps", "Bottom 5%", "fps", 1, "{:.2f}"),
    ]
    rows = []
    for name, field, cut, unit, scale, f in criteria:
        vals = [props[c][field] * scale for c in included if props.get(c, {}).get(field) is not None]
        outside = sum(1 for c in excluded if name in failed_by[c])
        sole = sum(1 for c in excluded if failed_by[c] == [name])
        rows.append({"property": name, "percentile_cut": cut,
                     "retained_range": f"{f.format(min(vals))} to {f.format(max(vals))} {unit}",
                     "clips_outside_range": outside, "sole_reason": sole})
    multi = sum(1 for c in excluded if len(failed_by[c]) > 1)
    total = len(included) + len(excluded)
    rows += [
        {"property": "Excluded on more than one criterion", "percentile_cut": "", "retained_range": "",
         "clips_outside_range": multi, "sole_reason": ""},
        {"property": "Excluded on any criterion", "percentile_cut": "", "retained_range": "",
         "clips_outside_range": f"{len(excluded)} ({len(excluded) / total:.1%})", "sole_reason": ""},
        {"property": "Retained", "percentile_cut": "", "retained_range": "",
         "clips_outside_range": f"{len(included)} ({len(included) / total:.1%})", "sole_reason": ""},
    ]
    write_csv(out_dir, "table_s6_screening.csv", rows)
    return included, excluded


def stage1_tables(out_dir: str, gt: dict, included: list[str], args) -> list[str]:
    runs = {m: load_results(os.path.join(path(args.stage1_dir), m)) for m in STAGE1_MODELS}
    inc = set(included)
    valid = {m: {c: x for c, x in runs[m][0].items() if c in inc} for m in STAGE1_MODELS}
    common = sorted(set.intersection(*(set(v) for v in valid.values())))

    # Table 1 is the primary_common_subset / ten-label rows; the other rows are
    # Supplementary Table S5a (nine labels, each model's own clips, failures as errors).
    analyses = [
        ("primary_common_subset", common, False),
        ("own_successful_clips", None, False),
        ("failures_as_errors", sorted(inc), True),
    ]
    rows, ranks = [], {}
    for name, clip_set, fail_err in analyses:
        for labels_name, labels in (("nine", NINE), ("ten", LABEL_KEYS)):
            point = {}
            for k, m in enumerate(STAGE1_MODELS):
                clips = sorted(valid[m]) if clip_set is None else clip_set
                C = clip_counts(valid[m], clips, labels, gt, failures_as_errors=fail_err)
                bs = ClusteredBootstrap(clips, args.n_boot, args.seed + 10 + k)
                est = metrics(C.sum(axis=0))
                boot = metrics(bs.resample(C))
                point[m] = float(est["macro_f1"])
                # Processing time on this analysis's clips (clips the model failed have no time).
                # The clustered CI uses seed args.seed for every model.
                tm = np.array([[valid[m][c]["timing_s"], 1.0] if c in valid[m]
                               and valid[m][c].get("timing_s") is not None else [0.0, 0.0]
                               for c in clips])
                if tm[:, 1].sum() > 0:
                    tb = ClusteredBootstrap(clips, args.n_boot, args.seed).resample(tm)
                    t_lo, t_hi = ci(tb[:, 0] / tb[:, 1])
                    t_mean = tm[:, 0].sum() / tm[:, 1].sum()
                else:
                    t_lo = t_hi = t_mean = None
                row = {"analysis": name, "labels": labels_name, "model": STAGE1_MODELS[m],
                       "n_clips": len(clips), "n_source_videos": bs.n_videos,
                       "clips_processed": len(valid[m]),
                       "failure_rate": r(1 - len(valid[m]) / len(inc)),
                       "mean_time_s": r(t_mean), "mean_time_ci_lo": r(t_lo),
                       "mean_time_ci_hi": r(t_hi)}
                for met in ("macro_f1", "macro_precision", "macro_recall", "macro_accuracy"):
                    lo, hi = ci(boot[met])
                    row[met], row[met + "_ci_lo"], row[met + "_ci_hi"] = r(est[met]), r(lo), r(hi)
                rows.append(row)
            order = sorted(point, key=point.get, reverse=True)
            ranks[(name, labels_name)] = order
            for row in rows:
                if row["analysis"] == name and row["labels"] == labels_name:
                    key = next(k for k, v in STAGE1_MODELS.items() if v == row["model"])
                    row["rank_by_macro_f1"] = order.index(key) + 1
    write_csv(out_dir, "table1_and_s5a.csv", rows)

    # Results: paired difference, first minus second model, 10-label macro-F1, common subset
    a, b = ranks[("primary_common_subset", "ten")][:2]
    Ca = clip_counts(valid[a], common, LABEL_KEYS, gt)
    Cb = clip_counts(valid[b], common, LABEL_KEYS, gt)
    bs = ClusteredBootstrap(common, args.n_boot, args.seed)
    d = metrics(bs.resample(Ca))["macro_f1"] - metrics(bs.resample(Cb))["macro_f1"]
    lo, hi = ci(d)
    write_csv(out_dir, "table1_paired_difference.csv", [{
        "labels": "ten", "model_a": STAGE1_MODELS[a], "model_b": STAGE1_MODELS[b], "n_clips": len(common),
        "macro_f1_difference": r(metrics(Ca.sum(axis=0))["macro_f1"] - metrics(Cb.sum(axis=0))["macro_f1"]),
        "ci_lo": r(lo), "ci_hi": r(hi)}])

    # Per-category counts for every model and Gemini false-positive rates (Table S5b)
    cat_rows = []
    for m in STAGE1_MODELS:
        C = clip_counts(valid[m], common, LABEL_KEYS, gt).sum(axis=0)
        mt = metrics(C)
        for j, l in enumerate(LABEL_KEYS):
            cat_rows.append({"model": STAGE1_MODELS[m], "label": l, "n_clips": len(common),
                             "tp": int(C[j, TP]), "fp": int(C[j, FP]), "fn": int(C[j, FN]),
                             "tn": int(C[j, TN]), "precision": r(mt["per_label_precision"][j]),
                             "recall": r(mt["per_label_recall"][j]), "f1": r(mt["per_label_f1"][j]),
                             "false_positive_rate": r(mt["per_label_fpr"][j])})
    write_csv(out_dir, "table_s5b_per_category_counts.csv", cat_rows)

    C = clip_counts(valid["gemini-3.7-flash"], common, NINE, gt)
    boot = metrics(ClusteredBootstrap(common, args.n_boot, args.seed + 20).resample(C))
    est = metrics(C.sum(axis=0))
    fpr_rows = []
    for j, l in enumerate(NINE):
        lo, hi = ci(boot["per_label_fpr"][:, j])
        fpr_rows.append({"label": l, "definition_type": "Absence-defined" if l in (EYE, NR)
                         else "Presence-defined", "false_positive_rate": r(est["per_label_fpr"][j]),
                         "ci_lo": r(lo), "ci_hi": r(hi)})
    write_csv(out_dir, "table_s5b_false_positive_rates.csv", fpr_rows)

    # Supplementary Note S1: overall label agreement of Gemini 3.7 Flash (formative review frame)
    agree = Counter()
    for c, x in valid["gemini-3.7-flash"].items():
        pred = x.get("asd_behavioral_metrics", {})
        agree[sum(1 for l in LABEL_KEYS if as_bool(pred.get(l)) is gt[c][l])] += 1
    n = sum(agree.values())
    t1 = [{"overall_label_agreement": f"{k}/10", "included_clips_n": agree.get(k, 0),
           "included_clips_pct": round(agree.get(k, 0) / n * 100, 1)} for k in range(10, -1, -1)]
    t1.append({"overall_label_agreement": "Total", "included_clips_n": n, "included_clips_pct": 100.0})
    write_csv(out_dir, "note_s1_label_agreement.csv", t1)
    return common


def paired_rows(name_a: str, name_b: str, rec_a, rec_b, clips: list[str],
                gt: dict, args, seed: int) -> list[dict]:
    """
    Paired comparison b − a on the same clips and the same bootstrap resamples.

    rec_a / rec_b are one run (a dict of records by clip) or a list of runs. With
    several runs, each system's value is the mean of its per-run metric, computed on
    the same clips and, within each resample, the same resampled clips for every run.
    """
    runs_a = rec_a if isinstance(rec_a, list) else [rec_a]
    runs_b = rec_b if isinstance(rec_b, list) else [rec_b]
    bs = ClusteredBootstrap(clips, args.n_boot, seed)
    out = []
    specs = [("Macro-F1, nine behavior labels", NINE, "macro_f1", None),
             ("Macro-precision", NINE, "macro_precision", None),
             ("Macro-recall", NINE, "macro_recall", None),
             ("Accuracy", NINE, "macro_accuracy", None),
             ("False-positive rate: absence or avoidance of eye contact", NINE, "per_label_fpr", EYE),
             ("False-positive rate: non-responsiveness to verbal interaction", NINE, "per_label_fpr", NR),
             ("Macro-F1, all 10 labels (sensitivity)", LABEL_KEYS, "macro_f1", None)]
    cache = {}
    for outcome, labels, key, label in specs:
        tag = "nine" if labels is NINE else "ten"
        if tag not in cache:
            def per_run(runs):
                Cs = [clip_counts(rec, clips, labels, gt) for rec in runs]
                return ([metrics(C.sum(axis=0)) for C in Cs], [metrics(bs.resample(C)) for C in Cs])
            cache[tag] = (per_run(runs_a), per_run(runs_b))
        (ea, ba), (eb, bb) = cache[tag]
        j = labels.index(label) if label else None
        pick = (lambda d: d[key][..., j]) if label else (lambda d: d[key])
        mean_runs = lambda ds: np.mean([pick(d) for d in ds], axis=0)
        va, vb = float(mean_runs(ea)), float(mean_runs(eb))
        boot_a, boot_b = mean_runs(ba), mean_runs(bb)
        diffs = boot_b - boot_a
        lo, hi = ci(diffs)
        alo, ahi = ci(boot_a)
        blo, bhi = ci(boot_b)
        out.append({"outcome": outcome, "n_clips": len(clips), "n_source_videos": bs.n_videos,
                    f"{name_a}_runs": len(runs_a), f"{name_b}_runs": len(runs_b),
                    name_a: r(va), f"{name_a}_ci_lo": r(alo), f"{name_a}_ci_hi": r(ahi),
                    name_b: r(vb), f"{name_b}_ci_lo": r(blo), f"{name_b}_ci_hi": r(bhi),
                    "paired_difference": r(vb - va), "diff_ci_lo": r(lo), "diff_ci_hi": r(hi)})
    return out


def per_label_rows(name_a: str, name_b: str, rec_a, rec_b, clips: list[str],
                   gt: dict, args, seed: int) -> list[dict]:
    """Per-label F1, b − a, for the nine behavior labels (same resamples as paired_rows)."""
    runs_a = rec_a if isinstance(rec_a, list) else [rec_a]
    runs_b = rec_b if isinstance(rec_b, list) else [rec_b]
    bs = ClusteredBootstrap(clips, args.n_boot, seed)

    def stats(runs):
        Cs = [clip_counts(rec, clips, NINE, gt) for rec in runs]
        est = np.mean([metrics(C.sum(axis=0))["per_label_f1"] for C in Cs], axis=0)
        boot = np.mean([metrics(bs.resample(C))["per_label_f1"] for C in Cs], axis=0)
        return est, boot

    (ea, ba), (eb, bb) = stats(runs_a), stats(runs_b)
    out = []
    for j, label in enumerate(NINE):
        diffs = bb[:, j] - ba[:, j]
        lo, hi = ci(diffs)
        out.append({"label": label, "n_clips": len(clips),
                    "positives": int(sum(bool(gt[c][label]) for c in clips)),
                    name_a: r(ea[j]), name_b: r(eb[j]), "paired_difference": r(eb[j] - ea[j]),
                    "diff_ci_lo": r(lo), "diff_ci_hi": r(hi)})
    return out


def figure2_counts(out_dir: str, included: list[str], excluded: list[str], gt: dict,
                   common: list[str], args) -> None:
    dev, evaluation = read_ids(args.dev_csv), read_ids(args.eval_csv)
    nv = lambda ids: len({source_video(c) for c in ids})
    retrievable = included + excluded
    rows = [
        {"step": "Published AV-ASD corpus", "clips": len(gt), "source_videos": nv(gt)},
        {"step": "Clips unavailable", "clips": len(gt) - len(retrievable), "source_videos": ""},
        {"step": "Retrievable clips", "clips": len(retrievable), "source_videos": nv(retrievable)},
        {"step": "Clips excluded on technical criteria", "clips": len(excluded), "source_videos": nv(excluded)},
        {"step": "Source videos with both included and excluded clips", "clips": "",
         "source_videos": len({source_video(c) for c in included} & {source_video(c) for c in excluded})},
        {"step": "Included corpus", "clips": len(included), "source_videos": nv(included)},
        {"step": "Stage 1 benchmark, common valid outputs", "clips": len(common), "source_videos": nv(common)},
        {"step": "Development subset", "clips": len(dev), "source_videos": nv(dev)},
        {"step": "Evaluation subset", "clips": len(evaluation), "source_videos": nv(evaluation)},
    ]
    write_csv(out_dir, "figure2_counts.csv", rows)


def table4a_main(argv) -> None:
    """
    Table 4 panel A: original vs revised labeling prompt on the evaluation subset, three
    runs each. Each system's value is the mean of its per-run metrics, on the clips valid
    in every run of both systems, with the same bootstrap resamples for every run.
    """
    ap = argparse.ArgumentParser(prog="python -m benchmark.scripts.paper_tables table4a",
                                 description="Table 4 panel A (labeling prompts)")
    ap.add_argument("--clips", required=True, help="CSV of clip IDs (Video_ID column)")
    ap.add_argument("--system", action="append", required=True,
                    help="NAME=RUN_DIR[,RUN_DIR...]; repeat per system")
    ap.add_argument("--primary", required=True, help="CANDIDATE:BASELINE (difference = candidate − baseline)")
    ap.add_argument("--reference", action="append", default=[],
                    help="NAME=RUN_DIR[,RUN_DIR...]: a system reported on the same clips, without a "
                         "paired difference")
    ap.add_argument("--per-label", action="store_true", help="also report per-label F1 differences")
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=20261027)
    ap.add_argument("--out", default="benchmark/results/paper_tables/table4a")
    args = ap.parse_args(argv)

    systems: dict[str, list[str]] = {}
    references: list[str] = []
    for spec in args.system + args.reference:
        name, _, dirs = spec.partition("=")
        if not dirs:
            ap.error(f"--system/--reference needs NAME=RUN_DIR[,RUN_DIR...]: {spec}")
        systems[name.strip()] = [path(d.strip()) for d in dirs.split(",") if d.strip()]
        if spec in args.reference:
            references.append(name.strip())
    cand, base = args.primary.split(":")
    for side in (cand, base):
        if side not in systems:
            ap.error(f"unknown system '{side}' in {args.primary}; defined: {sorted(systems)}")

    gt = load_ground_truth()
    ids = set(read_ids(args.clips))
    runs, processing = {}, []
    for name, dirs in systems.items():
        runs[name] = []
        for d in dirs:
            valid, failed = load_results(d)
            valid = {c: x for c, x in valid.items() if c in ids}
            runs[name].append(valid)
            processing.append({"system": name, "run_dir": os.path.relpath(d, ROOT), "clips": len(ids),
                               "valid": len(valid), "failed": len(failed & ids),
                               "not_run": len(ids) - len(valid) - len(failed & ids)})

    out_dir = path(args.out)
    os.makedirs(out_dir, exist_ok=True)
    clips = sorted(set.intersection(*(set(r_) for r_ in runs[cand] + runs[base])))
    seed = args.seed + 300
    rows = [{"candidate": cand, "baseline": base,
             **{k.replace(base, "baseline_value").replace(cand, "candidate_value")
                if k in (base, cand) or k.startswith((base + "_", cand + "_")) else k: v
                for k, v in x.items()}}
            for x in paired_rows(base, cand, runs[base], runs[cand], clips, gt, args, seed)]
    write_csv(out_dir, "table4a.csv", rows)
    write_csv(out_dir, "table4a_valid_outputs.csv", processing)
    if args.per_label:
        label_rows = [{"candidate": cand, "baseline": base, "label": x["label"], "n_clips": x["n_clips"],
                       "positives": x["positives"], "baseline_f1": x[base], "candidate_f1": x[cand],
                       **{k: x[k] for k in ("paired_difference", "diff_ci_lo", "diff_ci_hi")}}
                      for x in per_label_rows(base, cand, runs[base], runs[cand], clips, gt, args, seed)]
        write_csv(out_dir, "note_s1_per_label.csv", label_rows)
    ref_rows = []
    for name in references:
        ref_clips = sorted(set(clips).intersection(*(set(r_) for r_ in runs[name])))
        for x in paired_rows(name, base, runs[name], runs[base], ref_clips, gt, args, seed):
            ref_rows.append({"reference": name, "outcome": x["outcome"], "n_clips": x["n_clips"],
                             "n_source_videos": x["n_source_videos"], "runs": len(runs[name]),
                             "value": x[name], "ci_lo": x[name + "_ci_lo"], "ci_hi": x[name + "_ci_hi"]})
    write_csv(out_dir, "table4a_benchmark_prompt_reference.csv", ref_rows)
    write_run_info(out_dir, {"clips": args.clips, "systems": {k: [os.path.relpath(d, ROOT) for d in v]
                                                              for k, v in systems.items()},
                             "primary": args.primary, "references": references,
                             "n_boot": args.n_boot, "seed": args.seed})
    print(f"Wrote Table 4 panel A to {os.path.relpath(out_dir, ROOT)}/ ({len(clips)} clips)")


# Report rubric items (Supplementary Table S2b): dimension totals are recomputed from
# their items (alignment 0–2 each, actionability 0–1, empathy 1–5); safety is the
# direct 0–3 rating. Item keys are the judges' output keys.
B_ITEMS = {
    "clinical_accuracy": (["a1_social_emotional_reciprocity", "a2_nonverbal_communication",
                           "a3_relationship_development", "b4_repetitive_motor_behaviors",
                           "b5_insistence_on_sameness", "b6_sensory_processing"], 0, 2),
    "clinical_actionability": (["s1_context", "s2_recent_findings", "s3_fact_only", "s4_concrete_action",
                                "s5_followup_plan", "s6_guidance", "s7_professional_clarity"], 0, 1),
    "relational_empathy": (["e1_making_at_ease", "e2_whole_person", "e3_understanding_concerns",
                            "e4_care_and_compassion", "e5_being_positive", "e6_empowerment",
                            "e7_partnership"], 1, 5),
}
B_OUTCOMES = [("Composite score (0–1)", "composite"), ("Evidence alignment (0–12)", "clinical_accuracy"),
              ("Judge-rated safety (0–3)", "safety"), ("Actionability (0–7)", "clinical_actionability"),
              ("Relational empathy (7–35)", "relational_empathy")]


def _is_int(v) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def report_dims(scores: dict, canonical: bool) -> Optional[dict[str, float]]:
    """One judge's four dimension scores for one report, or None if any is unusable."""
    out = {}
    for dim, (items, lo, hi) in B_ITEMS.items():
        block = scores.get(dim)
        if not isinstance(block, dict):
            return None
        if canonical:
            vals = [block.get(i) for i in items]
            if not all(_is_int(v) and lo <= v <= hi for v in vals):
                return None
            out[dim] = float(sum(vals))
        else:
            t = block.get("total")
            if not isinstance(t, (int, float)) or isinstance(t, bool):
                return None
            out[dim] = float(t)
    s = (scores.get("safety") or {}).get("score") if isinstance(scores.get("safety"), dict) else None
    if not (_is_int(s) and 0 <= s <= 3):
        return None
    out["safety"] = float(s)
    out["composite"] = (0.40 * out["clinical_accuracy"] / 12 + 0.25 * out["safety"] / 3
                        + 0.25 * out["clinical_actionability"] / 7
                        + 0.10 * (out["relational_empathy"] - 7) / 28)
    return out


def table4b_main(argv) -> None:
    """
    Table 4 panel B: GPT-5.6 Sol reports written with the original and revised report
    prompts from the same observation records, each scored by the three-judge panel.
    Paired by clip; scores rebuilt from rubric items.
    """
    ap = argparse.ArgumentParser(prog="python -m benchmark.scripts.paper_tables table4b",
                                 description="Table 4 panel B (report prompts)")
    ap.add_argument("--judge-dir", default="benchmark/results/table4b_report_prompts/judge")
    ap.add_argument("--baseline-arm", default="report-original")
    ap.add_argument("--candidate-arm", default="report-v2")
    ap.add_argument("--judges", default="muse-judge,deepseek-v4-pro-judge,claude-opus-5-judge")
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=20261027)
    ap.add_argument("--out", default="benchmark/results/paper_tables/table4b")
    args = ap.parse_args(argv)
    judges = [j.strip() for j in args.judges.split(",")]
    arms = [args.baseline_arm, args.candidate_arm]

    raw: dict[str, dict[str, dict]] = {a: {} for a in arms}
    for a in arms:
        d = path(os.path.join(args.judge_dir, a))
        for fname in os.listdir(d):
            if fname.endswith(".json"):
                with open(os.path.join(d, fname), encoding="utf-8") as f:
                    raw[a][fname[:-5]] = json.load(f).get("judge_scores") or {}
    both = sorted(set(raw[arms[0]]) & set(raw[arms[1]]))

    # Clips with every rubric item from every judge, in both arms: {clip: {arm: {dim: judge mean}}}
    means = {}
    for c in both:
        per_arm = {}
        for a in arms:
            per_judge = [report_dims(raw[a][c].get(j) or {}, True) for j in judges]
            if any(x is None for x in per_judge):
                break
            per_arm[a] = {k: float(np.mean([x[k] for x in per_judge])) for k in per_judge[0]}
        else:
            means[c] = per_arm

    out_dir = path(args.out)
    os.makedirs(out_dir, exist_ok=True)
    clips = sorted(means)
    bs = ClusteredBootstrap(clips, args.n_boot, args.seed)
    n_b = bs.resample(np.ones(len(clips)))
    rows = []
    for outcome, key in B_OUTCOMES:
        xa = np.array([means[c][arms[0]][key] for c in clips])
        xb = np.array([means[c][arms[1]][key] for c in clips])
        ba, bb = bs.resample(xa) / n_b, bs.resample(xb) / n_b
        (alo, ahi), (blo, bhi), (lo, hi) = ci(ba), ci(bb), ci(bb - ba)
        rows.append({"outcome": outcome, "n_clips": len(clips), "n_source_videos": bs.n_videos,
                     "original": r(xa.mean()), "original_ci_lo": r(alo), "original_ci_hi": r(ahi),
                     "revised": r(xb.mean()), "revised_ci_lo": r(blo), "revised_ci_hi": r(bhi),
                     "paired_difference": r(xb.mean() - xa.mean()), "diff_ci_lo": r(lo),
                     "diff_ci_hi": r(hi)})
    write_csv(out_dir, "table4b.csv", rows)
    write_run_info(out_dir, {"judge_dir": args.judge_dir, "arms": arms, "judges": judges,
                             "clips_with_judge_files": len(both), "n_boot": args.n_boot, "seed": args.seed})
    print(f"Wrote Table 4 panel B to {os.path.relpath(out_dir, ROOT)}/ ({len(clips)} clips)")


C_ITEMS = {
    "clinical_actionability": ([f"s{i}_{n}" for i, n in enumerate(
        ["context", "recent_findings", "fact_only", "concrete_action", "followup_plan", "guidance",
         "professional_clarity"], 1)], 0, 1),
    "relational_empathy": ([f"e{i}_{n}" for i, n in enumerate(
        ["making_at_ease", "whole_person", "understanding_concerns", "care_and_compassion",
         "being_positive", "empowerment", "partnership"], 1)], 1, 5),
}
C_OUTCOMES = [("Report composite (0–1)", "composite"), ("Judge-rated safety (0–3)", "safety"),
              ("Actionability (0–7)", "clinical_actionability"),
              ("Relational empathy (7–35)", "relational_empathy")]
T5_CONTRASTS = [("delta_architecture", "two-stage-same-model", "single-stage-matched"),   # primary
                ("delta_model", "two-stage-deployed", "two-stage-same-model")]


def c_report_dims(scores: dict) -> Optional[dict[str, float]]:
    """One judge's Table 5 rubric scores for one report (from items), or None if unusable."""
    out = {}
    for dim, (items, lo, hi) in C_ITEMS.items():
        block = scores.get(dim)
        vals = [block.get(i) for i in items] if isinstance(block, dict) else []
        if len(vals) != len(items) or not all(_is_int(v) and lo <= v <= hi for v in vals):
            return None
        out[dim] = float(sum(vals))
    s = (scores.get("safety") or {}).get("score") if isinstance(scores.get("safety"), dict) else None
    if not (_is_int(s) and 0 <= s <= 3):
        return None
    out["safety"] = float(s)
    out["composite"] = (0.40 * out["safety"] / 3 + 0.40 * out["clinical_actionability"] / 7
                        + 0.20 * (out["relational_empathy"] - 7) / 28)
    return out


def table5_main(argv) -> None:
    """
    Table 5: matched end-to-end comparison on the evaluation subset. Judge outcomes are
    rubric scores rebuilt from items, mean over the three judges per report. Behavior
    agreement compares each report's behavior mentions with the AV-ASD labels. Eligible
    clips: every condition scored by every judge and every condition's mentions present.
    """
    ap = argparse.ArgumentParser(prog="python -m benchmark.scripts.paper_tables table5",
                                 description="Table 5 (matched end-to-end comparison)")
    ap.add_argument("--results-dir", default="benchmark/results/table5_end_to_end")
    ap.add_argument("--records-dir", default="benchmark/results/records",
                    help="observation and report records, read for token counts")
    ap.add_argument("--judges", default="muse-c-judge,deepseek-v4-pro-c-judge,claude-opus-5-c-judge")
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=20261027)
    ap.add_argument("--out", default="benchmark/results/paper_tables/table5")
    args = ap.parse_args(argv)
    judges = [j.strip() for j in args.judges.split(",")]
    conds = ["single-stage-matched", "two-stage-same-model", "two-stage-deployed"]
    res = path(args.results_dir)
    gt = load_ground_truth()

    def _load(d):
        out = {}
        if os.path.isdir(d):
            for fname in sorted(os.listdir(d)):
                if fname.endswith(".json"):
                    with open(os.path.join(d, fname), encoding="utf-8") as f:
                        out[fname[:-5]] = json.load(f)
        return out

    judge_files = _load(os.path.join(res, "judge"))
    mention_files = {c: _load(os.path.join(res, "mentions", c)) for c in conds}

    dims, ment, excluded = {}, {}, {}
    for clip, data in judge_files.items():
        k2l = data.get("model_key_to_label") or {}
        per_cond = {}
        for c in conds:
            per_judge = [c_report_dims(((data.get("judge_scores") or {}).get(j) or {}).get(k2l.get(c), {}) or {})
                         for j in judges]
            if any(x is None for x in per_judge):
                break
            per_cond[c] = {k: float(np.mean([x[k] for x in per_judge])) for k in per_judge[0]}
        else:
            m = {c: (mention_files[c].get(clip) or {}).get("mentions") for c in conds}
            if all(isinstance(v, dict) for v in m.values()) and clip in gt:
                dims[clip] = per_cond
                ment[clip] = {c: np.array([bool(m[c][b]["mentioned"]) for b in NINE]) for c in conds}
                continue
            excluded[clip] = "mentions missing"
            continue
        excluded[clip] = "judge scores incomplete"

    clips = sorted(dims)
    bs = ClusteredBootstrap(clips, args.n_boot, args.seed)
    n_b = bs.resample(np.ones(len(clips)))
    truth = np.array([[gt[c][b] for b in NINE] for c in clips])

    def counts(cond):
        pred = np.stack([ment[c][cond] for c in clips])
        return np.stack([pred & truth, pred & ~truth, ~pred & truth, ~pred & ~truth], axis=-1).astype(float)

    point, boot = {c: {} for c in conds}, {c: {} for c in conds}
    for c in conds:
        for _, key in C_OUTCOMES:
            x = np.array([dims[cl][c][key] for cl in clips])
            point[c][key], boot[c][key] = x.mean(), bs.resample(x) / n_b
        cnt = counts(c)
        point[c]["macro_f1"] = float(metrics(cnt.sum(axis=0))["macro_f1"])
        boot[c]["macro_f1"] = metrics(bs.resample(cnt))["macro_f1"]
        unlabeled = cnt[..., 1].sum(axis=1)   # mentioned behaviors not in the clip's AV-ASD labels
        point[c]["unlabeled_mentions"] = unlabeled.mean()
        boot[c]["unlabeled_mentions"] = bs.resample(unlabeled) / n_b

    outcomes = C_OUTCOMES + [("Behavior agreement with AV-ASD labels (nine-label macro-F1)", "macro_f1"),
                             ("Mentions not in the reference labels, per report", "unlabeled_mentions")]
    out_dir = path(args.out)
    os.makedirs(out_dir, exist_ok=True)
    rows = []
    for outcome, key in outcomes:
        row = {"outcome": outcome, "n_clips": len(clips), "n_source_videos": bs.n_videos}
        for c in conds:
            lo, hi = ci(boot[c][key])
            row.update({c: r(point[c][key]), f"{c}_ci_lo": r(lo), f"{c}_ci_hi": r(hi)})
        for name, a, b in T5_CONTRASTS:
            lo, hi = ci(boot[a][key] - boot[b][key])
            row.update({name: r(point[a][key] - point[b][key]), f"{name}_ci_lo": r(lo), f"{name}_ci_hi": r(hi)})
        rows.append(row)
    write_csv(out_dir, "table5.csv", rows)
    write_csv(out_dir, "table5_excluded_clips.csv",
              [{"clip_id": k, "reason": v} for k, v in sorted(excluded.items())])

    # Token use per clip on the eligible clips; two-stage conditions add both calls.
    records = path(args.records_dir)
    calls = {"two-stage-same-model": ["observations/gemini-3.7-obs-v3d",
                                      "reports/en/gemini-3.7-obs-v3d__gemini-3.7-report-v2"],
             "two-stage-deployed": ["observations/gemini-3.7-obs-v3d",
                                    "reports/en/gemini-3.7-obs-v3d__gpt-5.6-sol-report-v2"],
             "single-stage-matched": ["reports/en/gemini-3.7-matched-single"]}
    tok_rows = []
    for c, parts in calls.items():
        tin, tout, missing = [], [], 0
        for clip in clips:
            recs = []
            for part in parts:
                p = os.path.join(records, part, clip + ".json")
                recs.append(json.load(open(p, encoding="utf-8")) if os.path.exists(p) else None)
            if any(x is None or x.get("error") or not x.get("token_costs") for x in recs):
                missing += 1
                continue
            tin.append(sum(x["token_costs"]["input_tokens"] for x in recs))
            tout.append(sum(x["token_costs"]["output_tokens"] for x in recs))
        tok_rows.append({"condition": c, "n_clips": len(tin), "clips_without_token_record": missing,
                         "mean_input_tokens": r(np.mean(tin)) if tin else None,
                         "mean_output_tokens": r(np.mean(tout)) if tout else None})
    write_csv(out_dir, "table5_tokens.csv", tok_rows)
    write_run_info(out_dir, {"results_dir": args.results_dir, "conditions": conds, "judges": judges,
                             "n_boot": args.n_boot, "seed": args.seed})
    print(f"Wrote Table 5 to {os.path.relpath(out_dir, ROOT)}/ ({len(clips)} clips)")


# ---------------------------------------------------------------------------
# Table 2 and Supplementary Table S3: report-model benchmark
# ---------------------------------------------------------------------------

REPORT_MODELS = {
    "gpt-5.6-sol": "GPT-5.6 Sol",
    "gpt-5.6-terra": "GPT-5.6 Terra",
    "kimi-k3": "Kimi K3",
    "claude-opus-5": "Claude Opus 5",
    "qwen-3.8-max": "Qwen 3.8 Max",
}
REPORT_SHORT = {"gpt-5.6-sol": "Sol", "gpt-5.6-terra": "Terra", "kimi-k3": "Kimi",
                "claude-opus-5": "Claude", "qwen-3.8-max": "Qwen"}
TABLE2_JUDGES = {"gemini-3.7-judge": "Gemini 3.7 Flash", "muse-judge": "Muse Spark 1.2",
                 "deepseek-v4-pro-judge": "DeepSeek V4 Pro"}
B_DIMS = ["composite", "clinical_accuracy", "safety", "clinical_actionability", "relational_empathy"]


def equal_weight(d: dict) -> float:
    return (d["clinical_accuracy"] / 12 + d["safety"] / 3 + d["clinical_actionability"] / 7
            + (d["relational_empathy"] - 7) / 28) / 4


def krippendorff_interval(values: np.ndarray, weights: np.ndarray) -> float:
    """Interval alpha (squared distance) for units x coders with no missing values.
    weights: how many times each unit is drawn (1 for the point estimate)."""
    m = values.shape[1]
    pair = ((values[:, :, None] - values[:, None, :]) ** 2).sum(axis=(1, 2)) / (m - 1)
    n = (weights * m).sum()
    d_o = (weights * pair).sum() / n
    s1 = (weights[:, None] * values).sum()
    s2 = (weights[:, None] * values ** 2).sum()
    d_e = (2 * n * s2 - 2 * s1 ** 2) / (n * (n - 1))
    return float(1 - d_o / d_e)


def table2_main(argv) -> None:
    """
    Table 2, Supplementary Table S3 and the Supplementary Note S1 report denominators.
    Scores are rebuilt from rubric items (report_dims, canonical=True); judge-returned
    totals are the sensitivity analysis.
    """
    ap = argparse.ArgumentParser(prog="python -m benchmark.scripts.paper_tables table2",
                                 description="Table 2 and Supplementary Table S3")
    ap.add_argument("--judge-dir", default="benchmark/results/table2_report_models/judge/en")
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=20261027)
    ap.add_argument("--out", default="benchmark/results/paper_tables/table2")
    args = ap.parse_args(argv)
    models, judges = list(REPORT_MODELS), list(TABLE2_JUDGES)

    files = {}
    d = path(args.judge_dir)
    for fname in sorted(os.listdir(d)):
        if fname.endswith(".json"):
            with open(os.path.join(d, fname), encoding="utf-8") as f:
                files[fname[:-5]] = json.load(f)

    def scores(rec, m, j):
        lab = (rec.get("model_key_to_label") or {}).get(m)
        return ((rec.get("judge_scores") or {}).get(j) or {}).get(lab) if lab else None

    def dims(rec, canonical, js=None):
        """{model: {judge: dims}} for clips where every model and judge is usable, else None."""
        out = {}
        for m in models:
            out[m] = {}
            for j in (js or judges):
                x = report_dims(scores(rec, m, j) or {}, canonical)
                if x is None:
                    return None
                out[m][j] = x
        return out

    def judge_mean(dm, m, key, js=None, fn=None):
        js = js or judges
        return float(np.mean([fn(dm[m][j]) if fn else dm[m][j][key] for j in js]))

    out_dir = path(args.out)
    os.makedirs(out_dir, exist_ok=True)
    nv = lambda ids: len({source_video(c) for c in ids})

    # Denominators (Supplementary Note S1): reports judged per model, complete per model
    den = [{"item": "Report archive (judge files)", "clips": len(files), "source_videos": nv(files)}]
    for m in models:
        have = [c for c, x in files.items() if m in (x.get("model_key_to_label") or {})]
        complete = [c for c in have if all(report_dims(scores(files[c], m, j) or {}, True) for j in judges)]
        den.append({"item": f"{REPORT_MODELS[m]}: reports judged", "clips": len(have), "source_videos": nv(have)})
        den.append({"item": f"{REPORT_MODELS[m]}: every rubric item from all three judges",
                    "clips": len(complete), "source_videos": nv(complete)})
    write_csv(out_dir, "note_s1_report_denominators.csv", den)

    cohorts = {}
    for label, canonical in (("canonical", True), ("returned_totals", False)):
        dm = {c: dims(x, canonical) for c, x in files.items()}
        cohorts[label] = {c: v for c, v in dm.items() if v is not None}
    prim = cohorts["canonical"]
    clips = sorted(prim)
    bs = ClusteredBootstrap(clips, args.n_boot, args.seed)
    n_b = bs.resample(np.ones(len(clips)))

    def mean_ci(x):
        return (float(np.mean(x)), *ci(bs.resample(np.asarray(x, dtype=float)) / n_b))

    # Table 2
    t2 = []
    for m in models:
        row = {"model": REPORT_MODELS[m], "n_clips": len(clips), "n_source_videos": bs.n_videos}
        for k in B_DIMS:
            v, lo, hi = mean_ci([judge_mean(prim[c], m, k) for c in clips])
            row[k], row[k + "_ci_lo"], row[k + "_ci_hi"] = r(v), r(lo), r(hi)
        t2.append(row)
    write_csv(out_dir, "table2.csv", t2)

    # Contrasts and win rate printed in Results. Ties count as non-wins.
    comp = {m: np.array([judge_mean(prim[c], m, "composite") for c in clips]) for m in models}
    con = []
    for a, b in (("gpt-5.6-sol", "gpt-5.6-terra"), ("kimi-k3", "claude-opus-5")):
        v, lo, hi = mean_ci(comp[a] - comp[b])
        con.append({"contrast": f"{REPORT_MODELS[a]} minus {REPORT_MODELS[b]}", "statistic": "composite difference",
                    "estimate": r(v), "ci_lo": r(lo), "ci_hi": r(hi)})
    v, lo, hi = mean_ci((comp["gpt-5.6-sol"] > comp["gpt-5.6-terra"]).astype(float) * 100)
    con.append({"contrast": "GPT-5.6 Sol above GPT-5.6 Terra",
                "statistic": "percent of clips (ties are not wins)", "estimate": r(v),
                "ci_lo": r(lo), "ci_hi": r(hi)})
    write_csv(out_dir, "table2_contrasts.csv", con)

    # S3a: interval Krippendorff alpha; units are clip-model reports, clustered by source video
    units_video = np.repeat(bs.clip_video, len(models))
    s3a = []
    for k in B_DIMS:
        vals = np.array([[prim[c][m][j][k] for j in judges] for c in clips for m in models])
        point = krippendorff_interval(vals, np.ones(len(vals)))
        boot = np.array([krippendorff_interval(vals, bs.weights[b][units_video]) for b in range(args.n_boot)])
        lo, hi = ci(boot)
        s3a.append({"dimension": k, "n_reports": len(vals), "krippendorff_alpha": r(point),
                    "ci_lo": r(lo), "ci_hi": r(hi)})
    write_csv(out_dir, "table_s3a_alpha.csv", s3a)

    # S3b: rank orders under leave-one-judge-out panels and equal weights
    def order(fn):
        means = {m: np.mean([fn(c, m) for c in clips]) for m in models}
        return " > ".join(REPORT_SHORT[m] for m in sorted(means, key=means.get, reverse=True))
    s3c = [{"aggregation": "All three judges (primary)", "rank_order": order(lambda c, m: judge_mean(prim[c], m, "composite"))}]
    for j in judges:
        js = [x for x in judges if x != j]
        s3c.append({"aggregation": f"Excluding {TABLE2_JUDGES[j]}",
                    "rank_order": order(lambda c, m, js=js: judge_mean(prim[c], m, "composite", js))})
    write_csv(out_dir, "table_s3b_leave_one_judge_out.csv", s3c)
    s6 = []
    for name, fn in (("Primary weights", lambda dd: dd["composite"]), ("Equal weights", equal_weight)):
        row = {"scoring_variant": name, "rank_order": order(lambda c, m, fn=fn: judge_mean(prim[c], m, None, fn=fn))}
        for m in models:
            v, lo, hi = mean_ci([judge_mean(prim[c], m, None, fn=fn) for c in clips])
            row[REPORT_MODELS[m]], row[REPORT_MODELS[m] + "_ci_lo"], row[REPORT_MODELS[m] + "_ci_hi"] = r(v), r(lo), r(hi)
        s6.append(row)
    write_csv(out_dir, "table_s3b_equal_weights.csv", s6)

    # S3 note: judge-returned totals checked against item sums
    def audit(cohort_clips):
        by_dim, reports, clip_set, missing_blocks, missing_clips = Counter(), set(), set(), 0, set()
        for c in cohort_clips:
            for m in models:
                for j in judges:
                    s = scores(files[c], m, j) or {}
                    for dim, (items, lo, hi) in B_ITEMS.items():
                        blk = s.get(dim) or {}
                        vals = [blk.get(i) for i in items]
                        if not all(_is_int(v) for v in vals):
                            missing_blocks += 1
                            missing_clips.add(c)
                            continue
                        if blk.get("total") != sum(vals):
                            by_dim[dim] += 1
                            reports.add((c, m))
                            clip_set.add(c)
        return {"inconsistent_totals": sum(by_dim.values()),
                "clinical_actionability": by_dim["clinical_actionability"],
                "relational_empathy": by_dim["relational_empathy"],
                "clinical_accuracy": by_dim["clinical_accuracy"], "reports": len(reports),
                "clips": len(clip_set), "item_blocks_missing_a_key": missing_blocks,
                "clips_with_missing_key": len(missing_clips)}
    aud = [{"cohort": f"Primary, canonical items complete (n={len(clips)})", **audit(clips)},
           {"cohort": f"Returned totals complete (n={len(cohorts['returned_totals'])})",
            **audit(sorted(cohorts["returned_totals"]))}]
    write_csv(out_dir, "table_s3_note_scoring_audit.csv", aud)

    # S3 note: rank order with judge-returned totals, primary and broader cohorts
    sens = []
    for name, coh, canonical in (("Primary cohort, item sums", clips, True),
                                 ("Primary cohort, returned totals", clips, False),
                                 ("Returned-total cohort, returned totals", sorted(cohorts["returned_totals"]), False)):
        dm = {c: dims(files[c], canonical) for c in coh}
        means = {m: np.mean([judge_mean(dm[c], m, "composite") for c in coh]) for m in models}
        sens.append({"analysis": name, "n_clips": len(coh), "n_source_videos": nv(coh),
                     "rank_order": " > ".join(REPORT_SHORT[m]
                                              for m in sorted(means, key=means.get, reverse=True)),
                     **{REPORT_MODELS[m]: r(means[m]) for m in models}})
    write_csv(out_dir, "table_s3_note_returned_totals.csv", sens)

    write_run_info(out_dir, {"judge_dir": args.judge_dir, "judges": judges, "models": models,
                             "n_boot": args.n_boot, "seed": args.seed})
    print(f"Wrote Table 2 and Supplementary Table S3 to {os.path.relpath(out_dir, ROOT)}/ "
          f"({len(clips)} clips, {bs.n_videos} source videos)")


def main(argv=None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] == "table4a":
        return table4a_main(argv[1:])
    if argv and argv[0] == "table4b":
        return table4b_main(argv[1:])
    if argv and argv[0] == "table5":
        return table5_main(argv[1:])
    if argv and argv[0] == "table2":
        return table2_main(argv[1:])
    ap = argparse.ArgumentParser(description="Table 1, Supplementary Tables S5 and S6, Figure 2 counts")
    ap.add_argument("--stage1-dir", default="benchmark/results/table1_video_models")
    ap.add_argument("--dev-csv", default="dataset/csvs/a80_dev.csv")
    ap.add_argument("--eval-csv", default="dataset/csvs/a80_test.csv")
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=20261027)
    ap.add_argument("--out", default="benchmark/results/paper_tables/stage1")
    args = ap.parse_args(argv)

    out_dir = path(args.out)
    os.makedirs(out_dir, exist_ok=True)
    gt = load_ground_truth()
    included, excluded = screening_tables(out_dir)
    common = stage1_tables(out_dir, gt, included, args)
    figure2_counts(out_dir, included, excluded, gt, common, args)
    write_run_info(out_dir, {"stage1_dir": args.stage1_dir, "dev_csv": args.dev_csv,
                             "eval_csv": args.eval_csv, "n_boot": args.n_boot, "seed": args.seed})
    print(f"Wrote Table 1, S5, S6 and Figure 2 counts to {os.path.relpath(out_dir, ROOT)}/")


if __name__ == "__main__":
    main()
