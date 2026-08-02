import argparse
import os

import numpy as np
import pandas as pd

from analyze_ablation_zhaoxu import VARIABLE_NAMES, variable_metrics, global_metrics


def setting_name(args):
    model_id = f"Persistence_seq{args.seq_len}_pred{args.pred_len}_e{args.e_layers}_seed{args.seed}"
    return (
        f"{model_id}_Persistence_custom_M_ft{args.seq_len}_sl48_ll{args.pred_len}_pl{args.d_model}_"
        f"dm{args.n_heads}_nh{args.e_layers}_el1_dl{args.d_ff}_df1_fctimeF_ebTrue_dt{args.des}_projection_0"
    )


def main():
    parser = argparse.ArgumentParser(description='Persistence baseline for ZhaoXu data.')
    parser.add_argument('--root_path', type=str, default='./dataset/zhaoxu/')
    parser.add_argument('--data_path', type=str, default='merged_all.csv')
    parser.add_argument('--seq_len', type=int, default=288)
    parser.add_argument('--pred_len', type=int, default=48)
    parser.add_argument('--e_layers', type=int, default=1)
    parser.add_argument('--seed', type=int, default=2023)
    parser.add_argument('--d_model', type=int, default=64)
    parser.add_argument('--n_heads', type=int, default=2)
    parser.add_argument('--d_ff', type=int, default=64)
    parser.add_argument('--des', type=str, default='zhaoxu_baselines')
    parser.add_argument('--results_dir', type=str, default='results')
    args = parser.parse_args()

    df = pd.read_csv(os.path.join(args.root_path, args.data_path), encoding='utf-8-sig')
    data = df[VARIABLE_NAMES].values.astype(np.float32)

    num_train = int(len(data) * 0.7)
    num_test = int(len(data) * 0.2)
    border1 = len(data) - num_test - args.seq_len
    border2 = len(data)
    test_data = data[border1:border2]

    preds = []
    trues = []
    for index in range(len(test_data) - args.seq_len - args.pred_len + 1):
        s_end = index + args.seq_len
        true = test_data[s_end:s_end + args.pred_len]
        pred = np.repeat(test_data[s_end - 1:s_end], args.pred_len, axis=0)
        preds.append(pred)
        trues.append(true)

    preds = np.asarray(preds, dtype=np.float32)
    trues = np.asarray(trues, dtype=np.float32)

    folder = os.path.join(args.results_dir, setting_name(args))
    os.makedirs(folder, exist_ok=True)
    np.save(os.path.join(folder, 'pred.npy'), preds)
    np.save(os.path.join(folder, 'true.npy'), trues)

    per_variable = variable_metrics(preds, trues, VARIABLE_NAMES)
    per_variable.to_csv(os.path.join(folder, 'per_variable_metrics.csv'), index=False)
    global_row = global_metrics(preds, trues)
    np.save(
        os.path.join(folder, 'metrics.npy'),
        np.array([global_row['MAE'], global_row['MSE'], global_row['RMSE'], np.nan, np.nan, global_row['NSE']])
    )

    print('saved:', folder)
    print(global_row)


if __name__ == '__main__':
    main()
