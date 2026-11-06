# Multimodal Foundation Models That Describe Autism-Related Behaviors in Children's Videos for Caregivers: Stage-Wise Benchmark Study

Code, prompts, settings and released outputs for the paper. One command regenerates every table.

## Reproduce the tables

```bash
pip install -r requirements.txt   # Python 3.12 or later
bash reproduce.sh
```

`reproduce.sh` makes no model calls. It downloads the AV-ASD annotation files at a fixed commit, checks their SHA-256 checksums, and writes the tables to `benchmark/results/paper_tables/` with values to nine decimals. `MANIFEST.sha256` lists the checksum of every file.

| Paper item | File in `benchmark/results/paper_tables/` |
|---|---|
| Table 1 | `stage1/table1_and_s5a.csv` (`analysis` = primary_common_subset, `labels` = ten), `stage1/table1_paired_difference.csv` |
| Table 2 | `table2/table2.csv`, `table2/table2_contrasts.csv` |
| Table 4, panel A | `table4a/table4a.csv`, `table4a/table4a_benchmark_prompt_reference.csv`, `table4a/table4a_valid_outputs.csv` |
| Table 4, panel B | `table4b/table4b.csv` |
| Table 5 | `table5/table5.csv`, `table5/table5_tokens.csv`, `table5/table5_excluded_clips.csv` |
| Figure 2 | `stage1/figure2_counts.csv` |
| Supplementary Table S3 | `table2/table_s3a_alpha.csv`, `table2/table_s3b_*.csv`, `table2/table_s3_note_*.csv` |
| Supplementary Table S5 | `stage1/table1_and_s5a.csv` (other rows), `stage1/table_s5b_*.csv` |
| Supplementary Table S6 | `stage1/table_s6_screening.csv` |
| Supplementary Note S1 | `stage1/note_s1_label_agreement.csv`, `table2/note_s1_report_denominators.csv`, `table4a/note_s1_per_label.csv` |

## Contents

| Path | Contents |
|---|---|
| `benchmark/scripts/paper_tables.py` | All analyses |
| `benchmark/prompts/` | Every prompt (Supplementary Table S4), including the judge prompts with rubric anchors |
| `benchmark/config/` | Models, providers, and inference settings (Supplementary Table S1) |
| `benchmark/runners/`, `benchmark/models/`, `benchmark/scoring/` | Code that called the models and AI judges |
| `benchmark/results/` | Released outputs for each table, and the regenerated tables |
| `dataset/clips_screened_pass.txt` | The 701 included clips |
| `dataset/clips_screened_excluded.txt` | The 122 excluded clips and the criteria they failed |
| `dataset/clips_video/.probe_cache.json` | Screening properties of the 823 retrievable clips, measured by `scripts/analyze_clips.py` |
| `dataset/csvs/a80_dev.csv`, `dataset/csvs/a80_test.csv` | The 562 development clips and the 139 evaluation clips |
| `dataset/csvs/b2_dev100.csv` | The 100 development clips of the report-prompt comparison (Table 4, panel B) |
| `scripts/` | Clip download, cutting, property measurement, and upload |

In the code, `benchmark_a` is Stage 1 labeling (Tables 1 and 4, panel A), `benchmark_b` is Stage 2 caregiver reports and their AI-judge ratings (Tables 2 and 4, panel B), and `benchmark_c` is the rubric the AI judges applied to the anonymized reports in Table 5. `gemini-3.7-flash-kidooo` is the original labeling prompt and `gemini-3.7-flash-kidooo-v3d` the revised labeling prompt.

## Data not deposited

The released outputs contain the models' labels, the judges' rubric item scores, the behavior-mention flags, processing times and token counts. The observation records and reports written by the models and the judges' written rationales are not deposited because they describe individual children in identifiable public videos; in the released files, those fields read `[removed in the public release]` or are omitted. They are available from the corresponding author on reasonable request.

Neither the AV-ASD videos nor the annotations are redistributed. `reproduce.sh` downloads the annotation files from [ShijianDeng/AV-ASD](https://github.com/ShijianDeng/AV-ASD).

## Running the models again

Install `requirements-models.txt`, ffmpeg and yt-dlp, and set the API keys named by `api_key_env` in `benchmark/config/models.yaml` (for example in a `.env` file). Download and cut the clips with `scripts/data_preparation/fetch_clips.py`, and measure their duration, bitrate, frame count and frame rate with `scripts/analyze_clips.py`. Supplementary Table S6 gives the screening thresholds, and `dataset/clips_screened_pass.txt` lists the 701 included clips. Qwen 3.8 Max and Pegasus 1.5 received each clip as a URL: upload the clips with `scripts/upload_clips_to_s3.py`, serve them through CloudFront, and set `s3_bucket` and `cloudfront_domain` for both models.

| Paper item | Command (`python -m benchmark.runners.…`) |
|---|---|
| Table 1 | `run_benchmark_a --all-models` |
| Table 2 | `run_benchmark_b` |
| Table 4, panel A (three runs) | `run_benchmark_a --models gemini-3.7-flash-kidooo,gemini-3.7-flash-kidooo-v3d --csv dataset/csvs/a80_test.csv --results-dir <folder>` |
| Table 4, panel B | `run_benchmark_b_report_prompts` |
| Table 5 | `run_e2e_matched` |

Runners skip clips that already have a result, so write each run to a new results folder.

## License and citation

Code: MIT License (`LICENSE`). Released outputs and documentation: CC BY 4.0. The AV-ASD videos and annotation files remain under their owners' terms. Please also cite AV-ASD: Deng S, Kosloski EE, Patel S, et al. Hear me, see me, understand me: audio-visual autism behavior recognition. IEEE Trans Multimedia. 2025;27:2335-2346. doi:10.1109/TMM.2024.3521838
