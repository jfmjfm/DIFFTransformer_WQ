# DIFFTransformer_WQ

This repository contains the source code, processed data, and experiment scripts
for hourly water-quality forecasting with an iDiffTransformer model using a
differential attention mechanism. The study focuses on the Huangtaiqiao
monitoring section of the Xiaoqing River, Jinan, China, and predicts multiple
water-quality and algal indicators from multivariate hourly time series.

The codebase is adapted from the iTransformer time-series forecasting framework
and extends it with differential attention modules, water-quality data
preprocessing scripts, recurrent baselines, a persistence baseline, and
publication-oriented evaluation utilities.

## Repository Contents

```text
.
├── run.py                                      # main training/testing entry point
├── run_zhaoxu_lightweight_baseline_comparison.sh
│                                                # reproduces baseline comparison experiments
├── run_persistence_baseline_zhaoxu.py           # persistence baseline
├── analyze_ablation_zhaoxu.py                   # global and per-indicator metrics
├── model/
│   ├── DiffTransformer.py                      # proposed DiffTransformer model
│   ├── iTransformer.py                          # iTransformer baseline
│   ├── Transformer.py                           # Transformer baseline
│   ├── LSTM.py                                  # LSTM baseline
│   └── GRU.py                                   # GRU baseline
├── layers/
│   ├── SelfAttention_Family.py                  # FullAttention and DiffAttention modules
│   ├── Embed.py
│   └── Transformer_EncDec.py
├── data_provider/
│   ├── data_loader.py                           # dataset split, scaling, time features
│   └── data_factory.py
├── experiments/
│   ├── exp_long_term_forecasting.py             # training, validation, testing
│   └── exp_long_term_forecasting_partial.py
├── scripts/                                     # data audit, cleaning, smoothing, plotting
├── dataset/zhaoxu/                              # Huangtaiqiao data files
├── results/zhaoxu_baselines_summary/            # comparison tables and figures
├── checkpoints/                                 # saved model checkpoints
├── docs/                                        # data audit and revision notes
└── requirements.txt
```

## Data

The main processed dataset is:

```text
dataset/zhaoxu/merged_all.csv
```

It contains 62,013 hourly records from `2017-05-03 13:00:00` to
`2024-05-30 09:00:00`. The variables include:

- Water-quality and algal indicators: `BLUEGREEN`, `CHA`, `DO`, `COD`, `ELE`,
  `PH`, `NH4`, `TN`, `TP`, `TUR`, `TEMP`, `TOC`
- Hydrometeorological inputs: `flow`, `prec`

Raw and intermediate files are also stored in `dataset/zhaoxu/`, including the
yearly Huangtaiqiao water-quality workbooks, discharge data, precipitation data,
cleaned files, smoothed files, and audit outputs.

Descriptive statistics for manuscript tables are available at:

```text
dataset/zhaoxu/analysis/data_description_stats_english_publication.csv
```

## Environment

Install the required Python packages with:

```bash
pip install -r requirements.txt
```

The original experiments in this workspace were run in a Conda environment:

```bash
source /data/soft/miniconda3/etc/profile.d/conda.sh
conda activate leak_audio_env
```

Main dependencies include PyTorch, NumPy, pandas, scikit-learn, matplotlib,
SciPy, reformer-pytorch, scienceplots, and openpyxl.

## Reproducing the Main Baseline Comparison

Run:

```bash
./run_zhaoxu_lightweight_baseline_comparison.sh
```

This script trains/tests:

- Persistence
- LSTM
- GRU
- Transformer
- iDiffTransformer

The lightweight comparison uses the following explicit settings:

```text
seq_len = 288
label_len = 48
pred_len = 48
enc_in = dec_in = c_out = 14
d_model = 64
n_heads = 2
e_layers = 1
d_layers = 1
d_ff = 64
batch_size = 32
learning_rate = 1e-4
loss = MSE
seed = 2023
features = M
```

These values are passed at runtime and are different from several defaults in
`run.py`.

## Individual Training Example

Example command for iDiffTransformer:

```bash
python run.py \
  --is_training 1 \
  --root_path ./dataset/zhaoxu/ \
  --data_path merged_all.csv \
  --model_id iDiffTransformer_seq288_pred48_e1_seed2023 \
  --model iDiffTransformer \
  --data custom \
  --features M \
  --seq_len 288 \
  --label_len 48 \
  --pred_len 48 \
  --enc_in 14 \
  --dec_in 14 \
  --c_out 14 \
  --d_model 64 \
  --n_heads 2 \
  --e_layers 1 \
  --d_layers 1 \
  --d_ff 64 \
  --batch_size 32 \
  --learning_rate 0.0001 \
  --target TOC \
  --inverse \
  --freq h \
  --seed 2023 \
  --itr 1 \
  --des zhaoxu_baselines
```

## Metrics and Figures

After model prediction files have been generated, summarize global and
per-indicator metrics with:

```bash
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
```

The main output tables include:

```text
results/zhaoxu_baselines_summary/ablation_metrics_global.csv
results/zhaoxu_baselines_summary/ablation_metrics_by_variable.csv
results/zhaoxu_baselines_summary/publication/variable_metrics_publication_wide.csv
results/zhaoxu_baselines_summary/publication/variable_metrics_publication_long.csv
```

The main publication figures include:

```text
results/zhaoxu_baselines_summary/publication/global_metrics_comparison_2x2_macro_nse_clipped_horizontal_labels_nse2.png
results/zhaoxu_baselines_summary/publication/variable_metrics_2x2_idiff_labels_nse2.png
results/zhaoxu_baselines_summary/publication/loss_curve/idifftransformer_loss_curve.png
results/zhaoxu_baselines_summary/publication/noise_suppression_2x2/noise_suppression_two_cases_2x2.png
```

## Evaluation Metrics

The experiments report:

- NSE
- MSE
- MAE
- RMSE

For multivariate evaluation, macro-average NSE is computed by first calculating
NSE for each indicator and then averaging across indicators. This prevents
large-scale variables from dominating the overall evaluation.

`flow` and `prec` are used as auxiliary inputs but are excluded from the
per-indicator water-quality performance tables and figures.

## Data Preprocessing

The preprocessing workflow includes:

1. Hourly timestamp alignment.
2. Missing-value imputation using causal forward filling for short gaps and
   training-period month-hour medians for longer gaps.
3. Physical/instrument-range screening.
4. Half-detection-limit substitution for selected low-concentration indicators.
5. Isolated spike correction using historical robust median rules.
6. Three-hour causal EWMA smoothing for water-quality variables except
   precipitation.
7. Standardization with parameters fitted only on the training set.

Related scripts and audit files:

```text
scripts/audit_zhaoxu_data.py
scripts/clean_zhaoxu_bluegreen.py
scripts/clean_smooth_zhaoxu_cha.py
scripts/clean_smooth_zhaoxu_remaining.py
scripts/process_zhaoxu_extremes.py
scripts/replace_zhaoxu_extremes_30d_median.py
docs/zhaoxu_data_audit.md
docs/zhaoxu_cleaning_policy.md
dataset/zhaoxu/analysis/
dataset/zhaoxu/cleaning_audit/
```

## Notes on Differential Attention

The proposed model is implemented in:

```text
model/iDiffTransformer.py
layers/SelfAttention_Family.py
```

The differential attention mechanism follows the idea of subtracting two
attention distributions to reduce redundant or unstable attention responses.
The repository also includes ablation and residual-spectrum outputs to support
the noise-suppression analysis.

## Citation and Attribution

This repository builds on the iTransformer framework:

```text
iTransformer: Inverted Transformers Are Effective for Time Series Forecasting
ICLR 2024
https://arxiv.org/abs/2310.06625
```

The differential attention module is based on the Differential Transformer
formulation:

```text
Differential Transformer
https://arxiv.org/abs/2410.05258
```

If you use this repository, please cite the corresponding manuscript and the
underlying iTransformer and Differential Transformer papers as appropriate.

## Software and Data Availability

The source code and processed datasets are publicly available at:

```text
https://github.com/jfmjfm/DIFFTransformer_WQ
```

Original monitoring data ownership and access conditions should follow the
requirements of the relevant data provider or monitoring authority.


The repository is organized as follows:
model/iDiffTransformer.py: implementation of the proposed iDiffTransformer model.
layers/SelfAttention_Family.py: implementation of the attention mechanisms, including FullAttention and DiffAttention.
model/LSTM.py, model/GRU.py, model/Transformer.py: baseline model implementations.
run.py: main entry point for model training, validation, testing, and prediction.
run_zhaoxu_lightweight_baseline_comparison.sh: script for reproducing the baseline comparison experiments.
run_persistence_baseline_zhaoxu.py: implementation and execution script for the Persistence baseline.
analyze_ablation_zhaoxu.py: script for calculating global and per-indicator metrics and summarizing model comparison results.
scripts/: data preprocessing and diagnostic scripts.
data_provider/: dataset loading, chronological splitting, scaling, and time-feature construction.
experiments/: experiment runners for long-term forecasting.
utils/: metric calculation and training utilities.
dataset/zhaoxu/: raw and processed Huangtaiqiao water-quality, discharge, and precipitation datasets.
results/zhaoxu_baselines_summary/: generated comparison tables and figures.
checkpoints/: saved model checkpoints.
