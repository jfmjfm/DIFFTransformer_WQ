#!/usr/bin/env bash
set -euo pipefail

# Reproduce the lightweight baseline comparison used for:
# - global_metrics_comparison_2x2_macro_nse_clipped_horizontal_labels_nse2.png
# - variable_metrics_2x2_idiff_labels_nse2.png

source /data/soft/miniconda3/etc/profile.d/conda.sh
conda activate leak_audio_env

COMMON_ARGS=(
  --is_training 1
  --root_path ./dataset/zhaoxu/
  --data_path merged_all.csv
  --data custom
  --features M
  --seq_len 288
  --label_len 48
  --pred_len 48
  --enc_in 14
  --dec_in 14
  --c_out 14
  --d_model 64
  --n_heads 2
  --e_layers 1
  --d_layers 1
  --d_ff 64
  --batch_size 32
  --train_epochs 3
  --learning_rate 0.0001
  --loss MSE
  --target TOC
  --inverse
  --freq h
  --seed 2023
  --itr 1
  --des zhaoxu_baselines
)

run_model() {
  local model="$1"
  python run.py \
    --model_id "${model}_seq288_pred48_e1_seed2023" \
    --model "${model}" \
    "${COMMON_ARGS[@]}"
}

run_model LSTM
run_model GRU
run_model Transformer
run_model iDiffTransformer

python run_persistence_baseline_zhaoxu.py \
  --root_path ./dataset/zhaoxu/ \
  --data_path merged_all.csv \
  --seq_len 288 \
  --pred_len 48 \
  --d_model 64 \
  --n_heads 2 \
  --e_layers 1 \
  --d_ff 64 \
  --des zhaoxu_baselines \
  --seed 2023

python analyze_ablation_zhaoxu.py \
  --models Persistence LSTM GRU Transformer iDiffTransformer \
  --seq_len 288 \
  --pred_len 48 \
  --e_layers 1 \
  --seeds 2023 \
  --d_model 64 \
  --n_heads 2 \
  --d_ff 64 \
  --des zhaoxu_baselines \
  --results_dir results \
  --out_dir results/zhaoxu_baselines_summary
