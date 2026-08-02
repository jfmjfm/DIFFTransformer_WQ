import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


VARIABLE_NAMES = ['BLUEGREEN', 'CHA', 'flow', 'prec', 'DO', 'COD', 'ELE', 'PH', 'NH4', 'TN', 'TP', 'TUR', 'TEMP', 'TOC']


def setting_name(model, seq_len, pred_len, e_layers, seed, d_model, n_heads, d_ff, des):
    model_id = f"{model}_seq{seq_len}_pred{pred_len}_e{e_layers}_seed{seed}"
    return (
        f"{model_id}_{model}_custom_M_ft{seq_len}_sl48_ll{pred_len}_pl{d_model}_"
        f"dm{n_heads}_nh{e_layers}_el1_dl{d_ff}_df1_fctimeF_ebTrue_dt{des}_projection_0"
    )


def variable_metrics(preds, trues, names):
    rows = []
    for idx, name in enumerate(names):
        pred = preds[:, :, idx].reshape(-1)
        true = trues[:, :, idx].reshape(-1)
        error = pred - true
        denominator = np.sum((true - np.mean(true)) ** 2)
        rows.append({
            'variable': name,
            'NSE': np.nan if denominator == 0 else 1 - np.sum((true - pred) ** 2) / denominator,
            'MSE': np.mean(error ** 2),
            'MAE': np.mean(np.abs(error)),
            'RMSE': np.sqrt(np.mean(error ** 2)),
            'Bias': np.mean(error),
        })
    return pd.DataFrame(rows)


def residual_spectrum(preds, trues, names):
    rows = []
    for idx, name in enumerate(names):
        residual = (preds[:, :, idx] - trues[:, :, idx]).reshape(-1)
        spectrum = np.abs(np.fft.rfft(residual)) ** 2
        if spectrum.size <= 1:
            high_freq_ratio = np.nan
        else:
            split = max(1, int(spectrum.size * 0.5))
            high_freq_ratio = spectrum[split:].sum() / spectrum[1:].sum()
        rows.append({'variable': name, 'high_freq_residual_ratio': high_freq_ratio})
    return pd.DataFrame(rows)


def global_metrics(preds, trues):
    pred = preds.reshape(-1)
    true = trues.reshape(-1)
    error = pred - true
    denominator = np.sum((true - np.mean(true)) ** 2)
    return {
        'NSE': np.nan if denominator == 0 else 1 - np.sum((true - pred) ** 2) / denominator,
        'MSE': np.mean(error ** 2),
        'MAE': np.mean(np.abs(error)),
        'RMSE': np.sqrt(np.mean(error ** 2)),
        'Bias': np.mean(error),
    }


def save_metric_plot(df, metric, out_path):
    pivot = df.pivot(index='variable', columns='model', values=metric)
    ax = pivot.plot(kind='bar', figsize=(14, 5), width=0.8)
    ax.set_title(metric)
    ax.set_xlabel('Variable')
    ax.set_ylabel(metric)
    ax.grid(axis='y', alpha=0.3)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Summarize ZhaoXu ablation results.')
    parser.add_argument('--models', nargs='+', default=['Persistence', 'LSTM', 'GRU', 'Transformer', 'iDiffTransformer'])
    parser.add_argument('--seq_len', type=int, default=288)
    parser.add_argument('--pred_len', type=int, default=48)
    parser.add_argument('--e_layers', type=int, default=1)
    parser.add_argument('--seeds', nargs='+', type=int, default=[2023])
    parser.add_argument('--d_model', type=int, default=64)
    parser.add_argument('--n_heads', type=int, default=2)
    parser.add_argument('--d_ff', type=int, default=64)
    parser.add_argument('--des', type=str, default='zhaoxu_baselines')
    parser.add_argument('--results_dir', type=str, default='results')
    parser.add_argument('--out_dir', type=str, default='results/zhaoxu_ablation_summary')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    all_metrics = []
    all_spectrum = []
    all_global = []
    missing = []

    for model in args.models:
        for seed in args.seeds:
            setting = setting_name(model, args.seq_len, args.pred_len, args.e_layers, seed, args.d_model, args.n_heads, args.d_ff, args.des)
            folder = os.path.join(args.results_dir, setting)
            pred_path = os.path.join(folder, 'pred.npy')
            true_path = os.path.join(folder, 'true.npy')
            if not os.path.exists(pred_path) or not os.path.exists(true_path):
                missing.append(folder)
                continue

            preds = np.load(pred_path)
            trues = np.load(true_path)
            names = VARIABLE_NAMES[:preds.shape[-1]]
            metrics_df = variable_metrics(preds, trues, names)
            metrics_df.insert(0, 'seed', seed)
            metrics_df.insert(0, 'model', model)
            all_metrics.append(metrics_df)

            global_row = global_metrics(preds, trues)
            global_row['model'] = model
            global_row['seed'] = seed
            all_global.append(global_row)

            spectrum_df = residual_spectrum(preds, trues, names)
            spectrum_df.insert(0, 'seed', seed)
            spectrum_df.insert(0, 'model', model)
            all_spectrum.append(spectrum_df)

    if missing:
        with open(os.path.join(args.out_dir, 'missing_results.txt'), 'w') as f:
            f.write('\n'.join(missing))

    if not all_metrics:
        raise FileNotFoundError('No pred.npy/true.npy pairs found. Check missing_results.txt.')

    metrics = pd.concat(all_metrics, ignore_index=True)
    spectrum = pd.concat(all_spectrum, ignore_index=True)
    metrics.to_csv(os.path.join(args.out_dir, 'ablation_metrics_by_variable_seed.csv'), index=False)
    spectrum.to_csv(os.path.join(args.out_dir, 'ablation_residual_spectrum_seed.csv'), index=False)
    global_df = pd.DataFrame(all_global)
    global_df.to_csv(os.path.join(args.out_dir, 'ablation_metrics_global_seed.csv'), index=False)

    summary = metrics.groupby(['model', 'variable'], as_index=False).agg({
        'NSE': ['mean', 'std'],
        'MSE': ['mean', 'std'],
        'MAE': ['mean', 'std'],
        'RMSE': ['mean', 'std'],
        'Bias': ['mean', 'std'],
    })
    summary.columns = ['_'.join(col).strip('_') for col in summary.columns.values]
    summary.to_csv(os.path.join(args.out_dir, 'ablation_metrics_by_variable.csv'), index=False)

    overall = metrics.groupby('model', as_index=False).agg({
        'NSE': ['mean', 'std'],
        'MSE': ['mean', 'std'],
        'MAE': ['mean', 'std'],
        'RMSE': ['mean', 'std'],
        'Bias': ['mean', 'std'],
    })
    overall.columns = ['_'.join(col).strip('_') for col in overall.columns.values]
    overall.to_csv(os.path.join(args.out_dir, 'ablation_metrics_overall.csv'), index=False)
    global_summary = global_df.groupby('model', as_index=False).agg({
        'NSE': ['mean', 'std'],
        'MSE': ['mean', 'std'],
        'MAE': ['mean', 'std'],
        'RMSE': ['mean', 'std'],
        'Bias': ['mean', 'std'],
    })
    global_summary.columns = ['_'.join(col).strip('_') for col in global_summary.columns.values]
    global_summary.to_csv(os.path.join(args.out_dir, 'ablation_metrics_global.csv'), index=False)

    plot_df = metrics.groupby(['model', 'variable'], as_index=False)[['NSE', 'MSE', 'MAE', 'RMSE']].mean()
    for metric in ['NSE', 'MSE', 'MAE', 'RMSE']:
        save_metric_plot(plot_df, metric, os.path.join(args.out_dir, f'{metric.lower()}_by_variable.png'))

    spectrum_plot = spectrum.groupby(['model', 'variable'], as_index=False)['high_freq_residual_ratio'].mean()
    save_metric_plot(spectrum_plot, 'high_freq_residual_ratio', os.path.join(args.out_dir, 'high_freq_residual_ratio_by_variable.png'))

    print('Wrote summary to', args.out_dir)
    print(global_summary)


if __name__ == '__main__':
    main()
