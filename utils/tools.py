import os

import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.dates as mdates
import scienceplots

plt.style.use(['science', 'no-latex'])
plt.rcParams['font.sans-serif'] = ['Noto Sans CJK SC', 'DejaVu Sans']
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.unicode_minus'] = False
plt.switch_backend('agg')


def adjust_learning_rate(optimizer, epoch, args):
    # lr = args.learning_rate * (0.2 ** (epoch // 2))
    if args.lradj == 'type1':
        lr_adjust = {epoch: args.learning_rate * (0.5 ** ((epoch - 1) // 1))}
    elif args.lradj == 'type2':
        lr_adjust = {
            2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
            10: 5e-7, 15: 1e-7, 20: 5e-8
        }
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print('Updating learning rate to {}'.format(lr))


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), path + '/' + 'checkpoint.pth')
        self.val_loss_min = val_loss


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class StandardScaler():
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


def visual(true, preds=None, name='./pic/test.pdf'):
    """
    Results visualization
    """
    plt.figure()
    plt.plot(true, label='GroundTruth', linewidth=2)
    if preds is not None:
        plt.plot(preds, label='Prediction', linewidth=2)
    plt.legend()
    plt.savefig(name, bbox_inches='tight')

def visualbytime(true, preds, timestamps, name='./pic/test.pdf', title='变量名'):
    """
    结果可视化,包含实际时间戳和NSE值
    """
    plt.figure(figsize=(20, 10))
    
    # 确保时间戳是datetime对象
    if not isinstance(timestamps[0], pd.Timestamp):
        timestamps = pd.to_datetime(timestamps)
    
    # 检查维度是否匹配
    if len(timestamps) != len(true) or len(timestamps) != len(preds):
        raise ValueError(f"维度不匹配: timestamps {len(timestamps)}, preds {len(preds)}, true {len(true)}")
    
    # 计算NSE
    nse_value = nse(true, preds)
    
    # 绘制预测数据
    plt.plot(timestamps, preds, label='预测值', color='blue', linewidth=2)
    
    # 绘制真实数据
    plt.plot(timestamps, true, label='观测值', color='red', linewidth=2)
    
    plt.legend()
    plt.xlabel('时间')
    plt.ylabel('值')
    plt.title(f'{title} 时间序列预测 (NSE: {nse_value:.4f})')
    
    # 设置x轴日期格式
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
    plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
    
    # 自动旋转和对齐日期标签
    plt.gcf().autofmt_xdate()
    
    # 添加网格以便更容易读取
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # 添加NSE评价标准说明
    nse_text = (
        "NSE评价标准:\n"
        "0.75 < NSE ≤ 1.00: 很好 (Very good)\n"
        "0.65 < NSE ≤ 0.75: 良好 (Good)\n"
        "0.50 < NSE ≤ 0.65: 满意 (Satisfactory)\n"
        "NSE ≤ 0.50: 不满意 (Unsatisfactory)"
    )
    plt.text(0.02, 0.98, nse_text, transform=plt.gca().transAxes, fontsize=10, 
             verticalalignment='top', bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
    
    # 调整布局以确保所有元素可见
    plt.tight_layout()
    
    plt.savefig(name, bbox_inches='tight', dpi=300)
    plt.close()

def nse(observations, predictions):
    """计算Nash-Sutcliffe效率系数"""
    return 1 - (np.sum((observations - predictions) ** 2) / 
                np.sum((observations - np.mean(observations)) ** 2))

def adjustment(gt, pred):
    anomaly_state = False
    for i in range(len(gt)):
        if gt[i] == 1 and pred[i] == 1 and not anomaly_state:
            anomaly_state = True
            for j in range(i, 0, -1):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
            for j in range(i, len(gt)):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
        elif gt[i] == 0:
            anomaly_state = False
        if anomaly_state:
            pred[i] = 1
    return gt, pred


def cal_accuracy(y_pred, y_true):
    return np.mean(y_pred == y_true)
