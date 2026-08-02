from data_provider.data_factory import data_provider
from experiments.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, nse as nse_function
from utils.metrics import metric
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
import pandas as pds
import time
warnings.filterwarnings('ignore')

def calculate_variable_metrics(preds, trues, variable_names):
    rows = []
    for idx, name in enumerate(variable_names):
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
            'R2': np.nan if denominator == 0 else 1 - np.sum((true - pred) ** 2) / denominator,
            'Bias': np.mean(error),
        })
    return pds.DataFrame(rows)

class Exp_Long_Term_Forecast(Exp_Basic):
    def __init__(self, args):
        super(Exp_Long_Term_Forecast, self).__init__(args)

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, batch_x_time, batch_y_time) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                if 'PEMS' in self.args.data or 'Solar' in self.args.data:
                    batch_x_mark = None
                    batch_y_mark = None
                else:
                    batch_x_mark = batch_x_mark.float().to(self.device)
                    batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()

                loss = criterion(pred, true)

                total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        loss_history = []
        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, batch_x_time, batch_y_time) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                if 'PEMS' in self.args.data or 'Solar' in self.args.data:
                    batch_x_mark = None
                    batch_y_mark = None
                else:
                    batch_x_mark = batch_x_mark.float().to(self.device)
                    batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                        f_dim = -1 if self.args.features == 'MS' else 0
                        outputs = outputs[:, -self.args.pred_len:, f_dim:]
                        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                        loss = criterion(outputs, batch_y)
                        train_loss.append(loss.item())
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                    loss = criterion(outputs, batch_y)
                    train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            loss_history.append({
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'vali_loss': vali_loss,
                'test_loss': test_loss,
            })
            loss_dir = os.path.join('./results', setting)
            os.makedirs(loss_dir, exist_ok=True)
            pds.DataFrame(loss_history).to_csv(os.path.join(loss_dir, 'loss_history.csv'), index=False)
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

            # get_cka(self.args, setting, self.model, train_loader, self.device, epoch)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model
    #在测试阶段，批次默认为1
    def test(self, setting, test=0):
        import os 
        #测试使用中文，预测使用英文

        #旧版数据-总站注意数据带不带降雨      
        #总站：variable_names = ['降雨','pH', '溶解氧', '水温', '电导率', '浊度', '高锰酸盐指数', '总氮', '总磷', '氨氮']
        
        #赵旭：merged_df = merged_df[['BLUEGREEN', 'CHA', 'flow', 'prec', 'DO', 'COD', 'ELE', 'PH', 'NH4', 'TN', 'TP', 'TUR', 'TEMP', 'TOC']]
        #variable_names = ['蓝绿藻', '叶绿素','流量','降雨','溶解氧','高锰酸盐指数','电导率','pH','氨氮','总氮','总磷','浊度','水温','TOC']
        variable_names = ['BLUEGREEN', 'CHA', 'flow', 'prec', 'DO', 'COD', 'ELE', 'PH', 'NH4', 'TN', 'TP', 'TUR', 'TEMP', 'TOC']
        #新版数据-总站NH4,COD,TP,prec,TN,DO,Tur,Ele,WT,Ph
        #target变量必须放置在最后一个位置，否则Dataset_Custom和Dataset_Pred会强制将target变量放置在最后一个位置，导致乱序
        #variable_names = ['氨氮', '高锰酸盐指数', '总磷', '降雨', '总氮', '溶解氧', '浊度', '电导率', '水温', 'pH']
        test_data, test_loader = self._get_data(flag='test')
        if test:
            print('加载模型')
            
            starttime = time.time()
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))
            endtime = time.time()
            
            loadtime = endtime - starttime
            print(f'模型加载时间: {loadtime:.4f} 秒')

        preds = []
        trues = []
        self.model.eval()
        
        with torch.no_grad():
            total_nse_count = 0
            nse_above_0_5_count = 0
            
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, batch_x_time, batch_y_time) in enumerate(test_loader):
                #print(batch_x.shape, batch_y.shape, batch_x_mark.shape, batch_y_mark.shape, batch_x_time.shape, batch_y_time.shape)
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                if 'PEMS' in self.args.data or 'Solar' in self.args.data:
                    batch_x_mark = None
                    batch_y_mark = None
                else:
                    batch_x_mark = batch_x_mark.float().to(self.device)
                    batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]

                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()
                if test_data.scale and self.args.inverse:
                    shape = outputs.shape
                    outputs = test_data.inverse_transform(outputs.squeeze(0)).reshape(shape)
                    batch_y = test_data.inverse_transform(batch_y.squeeze(0)).reshape(shape)

                pred = outputs
                true = batch_y
                #print(pred.shape,true.shape)

                preds.append(pred)
                trues.append(true)

                if i % 100 == 0:
                    print(f"正在处理第 {i} 个批次")
                    input = batch_x.detach().cpu().numpy()
                    if test_data.scale and self.args.inverse:
                        shape = input.shape
                        input = test_data.inverse_transform(input.squeeze(0)).reshape(shape)
                    for var_idx in range(len(variable_names)):
                        var_gt = np.concatenate((input[0, :, var_idx], true[0, :, var_idx]), axis=0)
                        var_pd = np.concatenate((input[0, :, var_idx], pred[0, :, var_idx]), axis=0)
                        nse_value = nse_function(var_gt, var_pd)
                        total_nse_count += 1
                        if nse_value > 0.5:
                            nse_above_0_5_count += 1
        # 计算并打印NSE大于0.5的比例
        nse_above_0_5_ratio = nse_above_0_5_count / total_nse_count if total_nse_count > 0 else 0
        print(f"NSE > 0.5 的比例: {nse_above_0_5_ratio:.2%}")
        
        #[1,96,21]
        preds = np.array(preds)
        trues = np.array(trues)
        print('before reshape preds.shape:',preds.shape,'trues.shape:',trues.shape)
        #[4431,1,96,9]
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
         #[4431,96,9]
        print('after reshape preds.shape:',preds.shape,'trues.shape:',trues.shape)
        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        mae, mse, rmse, mape, mspe,nse,r2 = metric(preds, trues)
        result_variable_names = variable_names[-preds.shape[-1]:] if self.args.features == 'MS' else variable_names[:preds.shape[-1]]
        per_variable_metrics = calculate_variable_metrics(preds, trues, result_variable_names)
        per_variable_metrics.to_csv(folder_path + 'per_variable_metrics.csv', index=False)
        print('mse:{}, mae:{},nse:{},r2:{}，nse>0.5的比例：{}'.format(mse, mae,nse,r2,nse_above_0_5_ratio))
        f = open("result_long_term_forecast.txt", 'a')
        f.write(setting + "  \n")
        f.write('mse:{}, mae:{},nse:{} ,r2:{},nse>0.5的比例：{}'.format(mse, mae,nse,r2,nse_above_0_5_ratio))
        f.write('\n')
        f.write('{},{},{},{},{}'.format(self.args.model,self.args.seq_len,self.args.pred_len,self.args.e_layers,nse_above_0_5_ratio))
        f.write('\n')
        f.write('\n')
        f.close()

        np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe, nse]))
        np.save(folder_path + 'pred.npy', preds)
        np.save(folder_path + 'true.npy', trues)

        return

    #在预测阶段，批次默认为1
    def predict(self, setting, load=False):
        pred_data, pred_loader = self._get_data(flag='pred')

        if load:
            path = os.path.join(self.args.checkpoints, setting)
            best_model_path = path + '/' + 'checkpoint.pth'
            self.model.load_state_dict(torch.load(best_model_path))

        preds = []
        combined_timestamps = None
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, batch_x_time, batch_y_time) in enumerate(pred_loader):
                batch_x = batch_x.float().to(self.device)#输入序列的历史数据[B,seq_len,D]
                batch_y = batch_y.float().to(self.device)#在预测阶段，通常是一个占位符，前label包含已知的最近数据，用于教师强制，后pre填充为0或者其它占位符,模型逐步填充pred长度的占位符
                batch_x_mark = batch_x_mark.float().to(self.device)#输入序列的时间编码[B,SEQ_LEN,D]，提供额外时间信息，帮助模型理解输入数据的时间结构
                batch_y_mark = batch_y_mark.float().to(self.device)#提供预测时间段的时间信息，[B,label+pred,D]
                #batch_y前半部分label用于初始化预测过程
                # decoder input
                #label+pred两部分，前部分label保留，后部分清0
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                #后部分清零后再和label合并
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                #print('final dim:',batch_x.shape, batch_x_mark.shape, dec_inp.shape, batch_y_mark.shape)
                outputs = outputs.detach().cpu().numpy()
                if pred_data.scale and self.args.inverse:
                    shape = outputs.shape
                    outputs = pred_data.inverse_transform(outputs.squeeze(0)).reshape(shape)
                preds.append(outputs)

                #拼接时间戳
                input_timestamps = batch_x_time[0, -self.args.seq_len:]
                output_timestamps = batch_y_time[0, -self.args.pred_len:]
                #包括输入input_timestamps和输出output_timestamps
                #combined_timestamps = np.concatenate([input_timestamps, output_timestamps])
                #仅包括输出output_timestamps
                combined_timestamps = output_timestamps
                # 转换为 datetime 对象

        preds = np.array(preds)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        np.save(folder_path + 'real_prediction.npy', preds)
        #Add by MJF
        #打印preds的维度
        #print(preds.shape)
        # 定义变量名列表
        #总站:['NH4','COD','TP','prec','TN','DO','Tur','Ele','WT','Ph']
        #target变量必须放置在最后一个位置，否则Dataset_Custom和Dataset_Pred会强制将target变量放置在最后一个位置，导致乱序
        #variable_names = ['氨氮', '高锰酸盐指数', '总磷', '降雨', '总氮', '溶解氧', '浊度', '电导率', '水温', 'pH']
        #variable_names = ['NH4', 'COD', 'TP', 'prec', 'TN', 'DO', 'Tur', 'Ele', 'WT', 'Ph']
        preds=preds.squeeze(0)
        #赵旭:['蓝绿藻', '叶绿素','流量','降雨','溶解氧','高锰酸盐指数','电导率','pH','氨氮','总氮','总磷','浊度','水温','TOC']
        #variable_names = ['蓝绿藻', '叶绿素','流量','降雨','溶解氧','高锰酸盐指数','电导率','pH','氨氮','总氮','总磷','浊度','水温','TOC']
        variable_names = ['BLUEGREEN', 'CHA', 'flow', 'prec', 'DO', 'COD', 'ELE', 'PH', 'NH4', 'TN', 'TP', 'TUR', 'TEMP', 'TOC']
        
        # 将时间戳转换为datetime格式
        timestamps = pds.to_datetime(combined_timestamps, unit='s')
        
        # 创建DataFrame
        df = pds.DataFrame(preds, columns=variable_names)
        
        # 添加时间戳列
        df.insert(0, 'date', timestamps)  # 在第一列插入时间戳
        
        # 保存为CSV文件
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        
        df.to_csv(folder_path + 'real_prediction.csv', index=False)
        
        return
