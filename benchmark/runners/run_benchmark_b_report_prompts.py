"""
Table 4 panel B: original vs revised report prompt on 100 development clips.

Per clip:
  Stage 1  Gemini 3.7 Flash writes one observation record with the revised observation
           prompt (benchmark_b_observation_v3d.py); both arms use the same record
  Stage 2  GPT-5.6 Sol writes one report per arm: benchmark_b_report.py (original)
           and benchmark_b_report_v2.py (revised)
  Stage 3  Muse Spark 1.2, DeepSeek V4 Pro and Claude Opus 5 each score each report
           alone against the record it was written from

Clips: prompt_compare_b2.clips_csv (dataset/csvs/b2_dev100.csv).

Usage (from the repository root):
    python -m benchmark.runners.run_benchmark_b_report_prompts
    python -m benchmark.runners.run_benchmark_b_report_prompts --score-only

Result layout:
    <compare_cache_dir>/observations/{extractor}/{clip_id}.json
    <compare_cache_dir>/reports/{language}/{extractor}__{report_model}/{clip_id}.json
    <results_dir>/judge/{arm}/{clip_id}.json
"""

from __future__ import annotations

import argparse
import asyncio
import os
import pathlib
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
from benchmark.scoring.prompt_compare import print_judged_comparison, score_b_prompts

_TITLE = "Table 4 panel B — report prompt comparison"
_LABEL = "Model A"   # single report per judge call


def _load_yaml(rel: str) -> dict:
    with open(resolve_path(rel), encoding="utf-8") as f:
        return yaml.safe_load(f)


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=_TITLE)
    parser.add_argument("--max-clips", type=int, default=None,
                        help="Limit number of clips")
    parser.add_argument("--score-only", action="store_true",
                        help="Skip inference; rebuild comparison.json from judge files")
    args = parser.parse_args(argv)

    settings = _load_yaml("benchmark/config/settings.yaml")
    models_config = _load_yaml("benchmark/config/models.yaml")
    cfg = settings["prompt_compare_b2"]
    common = settings["compare_common"]

    ext_key: str = cfg["extractor"]
    arms: dict[str, str] = cfg["arms"]                     # arm → report model key
    baseline_arm: str = cfg.get("baseline_arm", next(iter(arms)))
    judge_keys: list[str] = cfg["judges"]
    language: str = cfg.get("language", "en")
    results_dir = resolve_path(cfg["results_dir"])
    judge_dir = os.path.join(results_dir, "judge")
    cache_dir = resolve_path(common["compare_cache_dir"])

    obs_dir = os.path.dirname(obs_path(cache_dir, ext_key, "x"))
    report_dirs = {
        arm: os.path.dirname(report_path(cache_dir, language, pipeline_arm_key(ext_key, rk), "x"))
        for arm, rk in arms.items()
    }
    stage_dirs = {arm: [obs_dir, report_dirs[arm]] for arm in arms}

    def _score() -> None:
        result = score_b_prompts(
            judge_dir, list(arms), baseline_arm,
            os.path.join(results_dir, "comparison.json"), stage_dirs,
        )
        print_judged_comparison(result, _TITLE)
        print(f"Comparison saved: {os.path.relpath(results_dir, resolve_path('.'))}/comparison.json")

    if args.score_only:
        _score()
        return

    wanted = set(pd.read_csv(resolve_path(cfg["clips_csv"]))["Video_ID"].astype(str).str.strip())
    clips = [c for c in load_clips(resolve_path(common["clips_dir"])) if c.clip_id in wanted]
    if args.max_clips is not None:
        clips = clips[:args.max_clips]
    if not clips:
        print("No clips loaded. Check compare_common.clips_dir and prompt_compare_b2.clips_csv.")
        sys.exit(1)
    print(f"  {len(clips)} of {len(wanted)} listed clips found")

    try:
        extractor = instantiate(models_config, ext_key, common)
        report_models = {arm: instantiate(models_config, rk, common, field_name="report_class")
                         for arm, rk in arms.items()}
        judges = {jk: instantiate(models_config, jk, common) for jk in judge_keys}
    except Exception as e:
        print(f"[ERROR] Cannot instantiate models: {e}")
        sys.exit(1)

    # Only this comparison's own directories: never touch other runs' cache entries
    for root in (obs_dir, *report_dirs.values(), results_dir):
        report_cleanup(_TITLE, clean_stale(root, retry_failed=False))

    judge_default = common.get("judge_max_concurrent", 5)
    limits = {
        ext_key: component_limit(models_config, ext_key, 5),
        **{rk: component_limit(models_config, rk, 5) for rk in arms.values()},
        **{jk: component_limit(models_config, jk, judge_default) for jk in judge_keys},
    }
    ctx = make_ctx(common, limits, False)

    async def _process(clip: ClipRecord) -> None:
        observations = await cached_stage(
            ctx, obs_path(cache_dir, ext_key, clip.clip_id), "observations",
            {"clip_id": clip.clip_id, "model": ext_key},
            ext_key, extractor, lambda: extractor.extract_async(clip.video_path),
            f"{clip.clip_id}/obs",
        )
        if observations is None:
            return

        async def _arm(arm: str) -> None:
            rk = arms[arm]
            report_model = report_models[arm]
            report = await cached_stage(
                ctx, report_path(cache_dir, language, pipeline_arm_key(ext_key, rk), clip.clip_id),
                "report_output",
                {"clip_id": clip.clip_id, "model": rk, "extractor": ext_key},
                rk, report_model,
                lambda: report_model.generate_report_async(observations, language),
                f"{clip.clip_id}/{arm}/report",
            )
            if report is None:
                return

            def _make_call(jk, judge, record):
                async def _call():
                    scores, raw, costs = await judge.score_async(observations, {_LABEL: report}, language)
                    if not isinstance(scores, dict) or not isinstance(scores.get(_LABEL), dict):
                        raise ValueError(f"judge output has no '{_LABEL}' entry: {str(scores)[:200]}")
                    return scores[_LABEL], raw, costs
                return _call

            await judge_stage(
                ctx, os.path.join(judge_dir, arm, f"{clip.clip_id}.json"), judges,
                f"{clip.clip_id}/{arm}/judge",
                compatible=lambda ex: ex.get("extractor") == ext_key and ex.get("report_model") == rk,
                base_record=lambda _: {
                    "clip_id": clip.clip_id, "arm": arm, "extractor": ext_key,
                    "report_model": rk, "language": language,
                },
                make_call=_make_call,
            )

        await asyncio.gather(*[_arm(arm) for arm in arms])

    print(f"\n{'=' * 60}\n{_TITLE} — {len(clips)} clips")
    print(f"Observations: {ext_key}   Arms: {arms}  (baseline: {baseline_arm})")
    print(f"Judges: {judge_keys}\n{'=' * 60}\n")
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
