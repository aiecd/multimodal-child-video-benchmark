"""
Table 5: matched end-to-end comparison on the evaluation subset.

Conditions, per clip (settings.yaml -> e2e_matched.conditions):
  single-stage-matched   Gemini 3.7 Flash watches the clip and writes the report in one
                         call (benchmark_e2e_matched_single_stage.py)
  two-stage-same-model   Gemini 3.7 Flash writes an observation record with the revised
                         observation prompt, then a report from that record alone with
                         the revised report prompt
  two-stage-deployed     the same record -> GPT-5.6 Sol report (revised report prompt)

The three reports are normalized to one plain-text schema, labeled Model A-C in an order
shuffled per clip (e2e_matched.shuffle_seed), and scored side by side by Muse Spark 1.2,
DeepSeek V4 Pro and Claude Opus 5 without the video or the record. DeepSeek V4 Pro then
records which of the nine AV-ASD behaviors each report says the child showed.

Usage (from the repository root):
    python -m benchmark.runners.run_e2e_matched
    python -m benchmark.runners.run_e2e_matched --score-only

Result layout:
    <compare_cache_dir>/observations/{extractor}/{clip_id}.json
    <compare_cache_dir>/reports/en/{extractor}__{report_model}/{clip_id}.json
    <compare_cache_dir>/reports/en/{video_report_model}/{clip_id}.json
    <results_dir>/mentions/{condition}/{clip_id}.json
    <results_dir>/judge/{clip_id}.json
"""

from __future__ import annotations

import argparse
import asyncio
import os
import pathlib
import random
import sys
from typing import Optional

import pandas as pd
import yaml
from dotenv import load_dotenv

load_dotenv(pathlib.Path(__file__).resolve().parents[2] / ".env")

from benchmark.data.loader_c import ClipRecord, load_clips
from benchmark.runners.compare_pipeline import (
    cached_stage,
    clean_stale,
    component_limit,
    instantiate,
    judge_stage,
    make_ctx,
    obs_path,
    pipeline_arm_key,
    report_cleanup,
    report_path,
    resolve_path,
    run_clips,
)
from benchmark.scoring.prompt_compare import print_judged_comparison, score_e2e
from benchmark.utils.normalize_report_c import normalize_report

_TITLE = "Table 5 — matched end-to-end comparison"
_LANGUAGE = "en"


def _load_yaml(rel: str) -> dict:
    with open(resolve_path(rel), encoding="utf-8") as f:
        return yaml.safe_load(f)


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=_TITLE)
    parser.add_argument("--clips-csv", default=None,
                        help="Override e2e_matched.clips_csv")
    parser.add_argument("--results-dir", default=None,
                        help="Override e2e_matched.results_dir")
    parser.add_argument("--max-clips", type=int, default=None, help="Limit number of clips")
    parser.add_argument("--score-only", action="store_true",
                        help="Skip inference; rebuild comparison.json from judge files")
    args = parser.parse_args(argv)

    settings = _load_yaml("benchmark/config/settings.yaml")
    models_config = _load_yaml("benchmark/config/models.yaml")
    cfg = settings["e2e_matched"]
    common = settings["compare_common"]

    conditions: dict[str, dict] = cfg["conditions"]
    arms = list(conditions)
    baseline = arms[0]
    judge_keys: list[str] = cfg["judges"]
    mentions_key: str = cfg["mentions_model"]
    seed = cfg.get("shuffle_seed", 0)
    results_dir = resolve_path(args.results_dir or cfg["results_dir"])
    judge_dir = os.path.join(results_dir, "judge")
    cache_dir = resolve_path(common["compare_cache_dir"])

    def _report_file(cond: str, clip_id: str) -> str:
        c = conditions[cond]
        key = c["video_report"] if "video_report" in c else pipeline_arm_key(c["extractor"], c["report"])
        return report_path(cache_dir, _LANGUAGE, key, clip_id)

    stage_dirs = {
        cond: ([os.path.dirname(obs_path(cache_dir, c["extractor"], "x"))] if "extractor" in c else [])
        + [os.path.dirname(_report_file(cond, "x"))]
        for cond, c in conditions.items()
    }

    def _score() -> None:
        result = score_e2e(judge_dir, arms, baseline,
                           os.path.join(results_dir, "comparison.json"), stage_dirs)
        print_judged_comparison(result, _TITLE)
        print(f"Comparison saved: {os.path.relpath(results_dir, resolve_path('.'))}/comparison.json")

    if args.score_only:
        _score()
        return

    clips_csv = args.clips_csv or cfg["clips_csv"]
    wanted = set(pd.read_csv(resolve_path(clips_csv))["Video_ID"].astype(str).str.strip())
    clips = [c for c in load_clips(resolve_path(common["clips_dir"])) if c.clip_id in wanted]
    if args.max_clips is not None:
        clips = clips[:args.max_clips]
    if not clips:
        print("No clips loaded. Check compare_common.clips_dir and the clip list.")
        sys.exit(1)
    print(f"  {len(clips)} of {len(wanted)} listed clips found ({clips_csv})")

    keys = sorted({v for c in conditions.values() for v in c.values()})
    try:
        models = {}
        for k in keys:
            field = "report_class" if models_config["models"][k].get("role") == "report" else "class"
            models[k] = instantiate(models_config, k, common, field_name=field)
        mentions = instantiate(models_config, mentions_key, common)
        judges = {jk: instantiate(models_config, jk, common) for jk in judge_keys}
    except Exception as e:
        print(f"[ERROR] Cannot instantiate models: {e}")
        sys.exit(1)

    # Only this comparison's own directories
    own_dirs = {d for ds in stage_dirs.values() for d in ds} | {results_dir}
    for root in sorted(own_dirs):
        report_cleanup(_TITLE, clean_stale(root, retry_failed=False))

    judge_default = common.get("judge_max_concurrent", 5)
    limits = {
        **{k: component_limit(models_config, k, 5) for k in keys},
        mentions_key: component_limit(models_config, mentions_key, 16),
        **{jk: component_limit(models_config, jk, judge_default) for jk in judge_keys},
    }
    ctx = make_ctx(common, limits, False)
    obs_tasks: dict[tuple[str, str], asyncio.Task] = {}

    async def _observations(ext: str, clip: ClipRecord) -> Optional[dict]:
        # Both two-stage conditions share one record per clip: run the extractor once
        key = (ext, clip.clip_id)
        if key not in obs_tasks:
            model = models[ext]
            obs_tasks[key] = asyncio.ensure_future(cached_stage(
                ctx, obs_path(cache_dir, ext, clip.clip_id), "observations",
                {"clip_id": clip.clip_id, "model": ext},
                ext, model, lambda: model.extract_async(clip.video_path),
                f"{clip.clip_id}/obs",
            ))
        return await obs_tasks[key]

    async def _report(cond: str, clip: ClipRecord) -> Optional[dict]:
        c = conditions[cond]
        if "video_report" in c:
            k = c["video_report"]
            model = models[k]
            return await cached_stage(
                ctx, _report_file(cond, clip.clip_id), "report_output",
                {"clip_id": clip.clip_id, "model": k},
                k, model, lambda: model.predict_async(clip.video_path),
                f"{clip.clip_id}/{cond}/report",
            )
        observations = await _observations(c["extractor"], clip)
        if observations is None:
            return None
        model = models[c["report"]]
        return await cached_stage(
            ctx, _report_file(cond, clip.clip_id), "report_output",
            {"clip_id": clip.clip_id, "model": c["report"], "extractor": c["extractor"]},
            c["report"], model, lambda: model.generate_report_async(observations, _LANGUAGE),
            f"{clip.clip_id}/{cond}/report",
        )

    async def _mentions(cond: str, clip: ClipRecord, report: dict) -> None:
        await cached_stage(
            ctx, os.path.join(results_dir, "mentions", cond, f"{clip.clip_id}.json"), "mentions",
            {"clip_id": clip.clip_id, "condition": cond, "model": mentions_key},
            mentions_key, mentions, lambda: mentions.extract_mentions_async(normalize_report(report)),
            f"{clip.clip_id}/{cond}/mentions",
        )

    async def _process(clip: ClipRecord) -> None:
        reports = dict(zip(arms, await asyncio.gather(*[_report(c, clip) for c in arms])))
        await asyncio.gather(*[_mentions(c, clip, r) for c, r in reports.items() if r is not None])
        if any(r is None for r in reports.values()):
            print(f"  [{clip.clip_id}] judge: SKIP — needs all three reports", flush=True)
            return

        def _base_record(reuse: Optional[dict]) -> dict:
            if reuse is not None:
                k2l = reuse["model_key_to_label"]
            else:
                order = list(arms)
                random.Random(f"{seed}:{clip.clip_id}").shuffle(order)
                k2l = {k: f"Model {chr(ord('A') + i)}" for i, k in enumerate(order)}
            return {
                "clip_id": clip.clip_id,
                "model_key_to_label": k2l,
                "label_to_model_key": {v: k for k, v in k2l.items()},
                "model_order": sorted(k2l.values()),
            }

        def _make_call(jk, judge, record):
            anon = {label: normalize_report(reports[key])
                    for label, key in sorted(record["label_to_model_key"].items())}
            return lambda: judge.score_async({}, anon)

        await judge_stage(
            ctx, os.path.join(judge_dir, f"{clip.clip_id}.json"), judges,
            f"{clip.clip_id}/judge",
            compatible=lambda ex: set(ex.get("model_key_to_label", {})) == set(arms),
            base_record=_base_record,
            make_call=_make_call,
        )

    print(f"\n{'=' * 60}\n{_TITLE} — {len(clips)} clips")
    for cond, c in conditions.items():
        print(f"  {cond}: {c}")
    print(f"Judges: {judge_keys}   Mentions: {mentions_key}\n{'=' * 60}\n")
    interrupted = asyncio.run(run_clips(
        clips, _process, ctx,
        common.get("max_concurrent_clips", 8),
        common.get("graceful_shutdown_timeout_s", 30),
        _TITLE,
    ))
    if interrupted:
        print("Interrupted — re-run the same command to resume (completed stages are cached).")
    _score()


if __name__ == "__main__":
    main()
