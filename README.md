# Multimodal Foundation Models That Describe Autism-Related Behaviors in Children's Videos for Caregivers: Stage-Wise Benchmark Study

Code, prompts, settings and released outputs for the paper. Two commands regenerate every table from the released outputs, without any model calls.

## Reproduce the tables

```bash
pip install -r requirements.txt   # Python 3.12 or later
bash reproduce.sh                 # needs network access to download the AV-ASD annotations
```

`reproduce.sh` makes no model calls. It downloads the AV-ASD annotation files at a fixed commit, checks their SHA-256 checksums, and writes the tables to `benchmark/results/paper_tables/`, with numeric values rounded to nine decimals. The committed tables were all generated with Python 3.12.4, numpy 2.4.4 and pandas 3.1.1. Each `run_info.json` records the versions of the run that wrote it, so a run in another environment changes those five files and their checksums; the tables themselves do not change.

`MANIFEST.sha256` lists the SHA-256 checksum of every file in the repository except itself. Check it with `shasum -a 256 -c MANIFEST.sha256`.

| Paper item | File in `benchmark/results/paper_tables/` |
|---|---|
| Table 1 | `stage1/table1_and_s5a.csv` (`analysis` = primary_common_subset, `labels` = ten), `stage1/table1_paired_difference.csv` |
| Table 2 | `table2/table2.csv`, `table2/table2_contrasts.csv` |
| Table 4, panel A | `table4a/table4a.csv`, `table4a/table4a_benchmark_prompt_reference.csv`, `table4a/table4a_valid_outputs.csv` |
| Table 4, panel B | `table4b/table4b.csv`, `table4b/table4b_excluded_clips.csv` |
| Table 5 | `table5/table5.csv`, `table5/table5_tokens.csv`, `table5/table5_excluded_clips.csv` |
| Figure 2 | Corpus, screening and partition counts: `stage1/figure2_counts.csv`. Report archive (693 clips): `table2/note_s1_report_denominators.csv`. Returned-total cohort (525) and primary report cohort (455): `table2/table_s3_note_returned_totals.csv`. Comparison sizes (134, 95, 133): the `n_clips` column of `table4a/table4a.csv`, `table4b/table4b.csv` and `table5/table5.csv`. The 50-clip formative review is not part of the released outputs. |
| Supplementary Table S3 | `table2/table_s3a_alpha.csv`, `table2/table_s3b_*.csv`, `table2/table_s3_note_*.csv` |
| Supplementary Table S5 | `stage1/table1_and_s5a.csv` (other rows), `stage1/table_s5b_*.csv` |
| Supplementary Table S6 | `stage1/table_s6_screening.csv` |
| Supplementary Note S1 | `stage1/note_s1_label_agreement.csv`, `table2/note_s1_report_denominators.csv`, `table4a/note_s1_per_label.csv` |

## Contents

| Path | Contents |
|---|---|
| `reproduce.sh` | Regenerates every table (see above) |
| `benchmark/scripts/paper_tables.py` | All analyses |
| `benchmark/scripts/fetch_labels.py` | Downloads and checks the AV-ASD annotation files |
| `benchmark/prompts/` | Every prompt (Supplementary Table S4), including the judge prompts with rubric anchors |
| `benchmark/config/` | Models, providers, and inference settings (Supplementary Table S1) |
| `benchmark/runners/`, `benchmark/models/` | Code that called the models and AI judges |
| `benchmark/scoring/` | Per-run summaries printed by the runners (the paper's tables come from `paper_tables.py`) |
| `benchmark/data/`, `benchmark/utils/` | Clip loading and cutting, report normalization, retries |
| `benchmark/results/table*/` | Released outputs for each table: labels, judge item scores, behavior-mention flags, timing and token counts |
| `benchmark/results/records/` | Observation records and reports of Table 5, with their text withheld (token counts and timing kept) |
| `benchmark/results/paper_tables/` | The regenerated tables |
| `dataset/clips_screened_pass.txt` | The 701 included clips |
| `dataset/clips_screened_excluded.txt` | The 122 excluded clips and the criteria they failed |
| `dataset/clips_video/.probe_cache.json` | Screening properties of the 823 retrievable clips, measured by `scripts/analyze_clips.py` |
| `dataset/csvs/a80_dev.csv`, `dataset/csvs/a80_test.csv` | The 562 development clips and the 139 evaluation clips (clip ids only) |
| `dataset/csvs/b2_dev100.csv` | The 100 development clips of the report-prompt comparison (Table 4, panel B) |
| `scripts/` | Clip download, cutting, property measurement, and upload |

In the code, `benchmark_a` is Stage 1 labeling (Tables 1 and 4, panel A), `benchmark_b` is Stage 2 caregiver reports and their AI-judge ratings (Tables 2 and 4, panel B), and `benchmark_c` is the rubric the AI judges applied to the anonymized reports in Table 5. Kidooo AI is the caregiver-facing application whose research configuration the paper evaluates. `gemini-3.7-flash-kidooo` is the original labeling prompt and `gemini-3.7-flash-kidooo-v3d` the revised labeling prompt; `gemini-3.7-obs-v3d` writes observation records with the revised observation prompt.

## Reading the released files

- **Rubric names.** Judge files use the internal names `clinical_accuracy` for evidence alignment and `clinical_actionability` for actionability. `paper_tables.py` renames them in its output.
- **Clip ids.** Table 2 judge files name the clip `video_id`; all other files use `clip_id`. Both hold the clip id `{source video}_{start}_{end}`.
- **Failed calls.** A failed model call has an `error` string. For judges, `judge_errors` is `true` where the call failed, because the error messages could contain report text. Failed calls were not run again.
- **Excluded clips.** `table4b/table4b_excluded_clips.csv` and `table5/table5_excluded_clips.csv` list the clips each comparison leaves out and why. In Table 4, panel B, the two clips with no judge file in either arm are the two for which Gemini 3.7 Flash returned an empty observation record.
- **Display names.** Table 4, panel A files carry the display names in use when the runs were made ("A Kidooo prompt", "A Kidooo v3d prompt"); they are the original and revised labeling prompts.
- **An unreadable clip.** `mkin5up15XM_57_63` was downloaded as a 262-byte file that ffprobe could not read. It is counted among the 823 retrievable clips and is excluded on all four screening criteria (Supplementary Table S6).
- **Language placeholder in prompts.** The report prompts end with "Write all report text in the language specified at the top of this prompt". The language slot is empty for English, which every run in the paper used. The prompts are kept exactly as the models received them.

## Data not deposited

The released outputs contain the models' labels, the judges' rubric item scores, the behavior-mention flags, processing times and token counts. The observation records and reports written by the models and the judges' written rationales are not deposited because they describe individual children in identifiable public videos; in the released files, those fields read `[removed in the public release]` or are omitted. They are available from the corresponding author on reasonable request.

Neither the AV-ASD videos nor the annotations are redistributed. `reproduce.sh` downloads the annotation files from [ShijianDeng/AV-ASD](https://github.com/ShijianDeng/AV-ASD).

## Running the models again

1. Install `requirements-models.txt`, ffmpeg and yt-dlp, and set the API keys named by `api_key_env` in `benchmark/config/models.yaml` (for example in a `.env` file).
2. Download the annotations with `python -m benchmark.scripts.fetch_labels`.
3. Download and cut the clips into `dataset/clips_video/` with `scripts/data_preparation/fetch_clips.py`. The runners use only the 701 clips in `dataset/clips_screened_pass.txt`; Supplementary Table S6 gives the screening thresholds. `scripts/analyze_clips.py` measures duration, bitrate, frame count and frame rate.
4. Qwen 3.8 Max and Pegasus 1.5 received each clip as a URL. Upload the clips with `scripts/upload_clips_to_s3.py` and set `s3_bucket` for both models in `models.yaml` (and `cloudfront_domain` if the clips are served through CloudFront; Qwen does not accept pre-signed S3 links).

New runs write to `benchmark/results/rerun/` (settings.yaml) and never to the released folders. The released records in `benchmark/results/records/` have their text withheld, so the runners do not reuse them.

| Paper item | Command (`python -m benchmark.runners.…`) | Output | Tables from the new output (`python -m benchmark.scripts.paper_tables …`) |
|---|---|---|---|
| Table 1 | `run_benchmark_a --all-models` | `rerun/benchmark_a/<model>/` | `--stage1-dir benchmark/results/rerun/benchmark_a --out <folder>` |
| Table 2 | `run_benchmark_b` | `rerun/benchmark_b/` | `table2 --judge-dir benchmark/results/rerun/benchmark_b/judge/en --out <folder>` |
| Table 4, panel A (three runs) | `run_benchmark_a --models gemini-3.7-flash-kidooo,gemini-3.7-flash-kidooo-v3d --csv dataset/csvs/a80_test.csv --results-dir benchmark/results/rerun/table4a/run1` (then `run2`, `run3`) | `rerun/table4a/runN/benchmark_a/<model>/` | `table4a` with `--system` folders as in `reproduce.sh` and `--out <folder>` |
| Table 4, panel B | `run_benchmark_b_report_prompts` | `rerun/table4b_report_prompts/`, records in `rerun/records/` | `table4b --judge-dir benchmark/results/rerun/table4b_report_prompts/judge --out <folder>` |
| Table 5 | `run_e2e_matched` | `rerun/table5_end_to_end/`, records in `rerun/records/` | `table5 --results-dir benchmark/results/rerun/table5_end_to_end --records-dir benchmark/results/rerun/records --out <folder>` |

Runners skip clips that already have a result, so give each repeated run its own folder (`--results-dir`, and `--cache-dir` for Table 4, panel B and Table 5). Failed calls are kept and skipped on the next run; `--retry-failed` runs them again. `--score-only` prints a summary from existing judge files, for example of the released Table 5 scores with `run_e2e_matched --score-only --results-dir benchmark/results/table5_end_to_end --cache-dir benchmark/results/records`.

## License and citation

Code: MIT License (`LICENSE`). Released outputs and documentation: Creative Commons Attribution 4.0 International (`LICENSE-DATA`). The AV-ASD videos and annotation files remain under their owners' terms. Please also cite AV-ASD: Deng S, Kosloski EE, Patel S, et al. Hear me, see me, understand me: audio-visual autism behavior recognition. IEEE Trans Multimedia. 2025;27:2335-2346. doi:10.1109/TMM.2024.3521838
