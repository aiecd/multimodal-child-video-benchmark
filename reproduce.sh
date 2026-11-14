#!/usr/bin/env bash
# Regenerate every table in the paper from the released outputs. No model calls.
# Run from the repository root. Outputs go to benchmark/results/paper_tables/.
set -euo pipefail
export PYTHONPATH="$PWD"
R=benchmark/results
runs() { echo "$R/table4a_labeling_prompts/run1/benchmark_a/$1,$R/table4a_labeling_prompts/run2/benchmark_a/$1,$R/table4a_labeling_prompts/run3/benchmark_a/$1"; }

# AV-ASD annotations, downloaded from the dataset authors' repository and checksum-verified
python -m benchmark.scripts.fetch_labels

# Table 1, Supplementary Tables S5 and S6, Figure 2 counts
python -m benchmark.scripts.paper_tables

# Table 2 and Supplementary Table S3
python -m benchmark.scripts.paper_tables table2

# Table 4 panel A
python -m benchmark.scripts.paper_tables table4a --clips dataset/csvs/a80_test.csv \
  --system original="$(runs gemini-3.7-flash-kidooo)" \
  --system revised="$(runs gemini-3.7-flash-kidooo-v3d)" \
  --reference benchmark_prompt="$R/table1_video_models/gemini-3.7-flash" \
  --primary revised:original --per-label

# Table 4 panel B
python -m benchmark.scripts.paper_tables table4b

# Table 5
python -m benchmark.scripts.paper_tables table5
