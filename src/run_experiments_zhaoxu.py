import subprocess
import time
import GPUtil
import itertools
import multiprocessing
import os

def get_free_gpu():
    GPUs = GPUtil.getGPUs()
    for gpu in GPUs:       
        #赵旭：merged_df = merged_df[['BLUEGREEN', 'CHA', 'flow', 'prec', 'DO', 'COD', 'ELE', 'PH', 'NH4', 'TN', 'TP', 'TUR', 'TEMP', 'TOC']]
        variable_names = ['蓝绿藻', '叶绿素','流量','降雨','溶解氧','高锰酸盐指数','电导率','pH','氨氮','总氮','总磷','浊度','水温','TOC']
        variable_names_en = ['BLUEGREEN', 'CHA', 'flow', 'prec', 'DO', 'COD', 'ELE', 'PH', 'NH4', 'TN', 'TP', 'TUR', 'TEMP', 'TOC'] 
        print('--------------------------------')
        print('zhaoxu测试使用中文名称')
        print(variable_names)
        print('--------------------------------')
        print('zhaoxu预测使用英文名称')
        print(variable_names_en)
        print(f"GPU {gpu.id} 剩余内存: {gpu.memoryFree} MB")
        print(f"GPU {gpu.id} 已使用内存: {gpu.memoryUsed} MB")
        if gpu.memoryFree > 700:  # 假设剩余内存大于1000MB为空闲
            return gpu.id
    return None

def run_experiment(args):
    model, seq_len, pred_len, e_layers, seed = args
    gpu_id = None
    while gpu_id is None:
        gpu_id = get_free_gpu()
        if gpu_id is None:
            print(f"等待可用GPU: {model}, seq_len={seq_len}, pred_len={pred_len}, e_layers={e_layers}, seed={seed}")
            time.sleep(60)  # 等待1分钟后再次检查
    
    model_id = f"{model}_seq{seq_len}_pred{pred_len}_e{e_layers}_seed{seed}"
    print(f"运行 {model_id} 模型，使用 GPU {gpu_id}")
    
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    subprocess.run([
        "python", "run.py",
        "--is_training", "1",
        "--root_path", "./dataset/zhaoxu/",
        "--data_path", "merged_all.csv",
        "--model_id", model_id,
        "--model", model,
        "--data", "custom",
        "--features", "M",
        "--seq_len", str(seq_len),
        "--pred_len", str( pred_len),
        "--e_layers", str(e_layers),
        "--enc_in", "14",
        "--dec_in", "14",
        "--c_out", "14",
        "--des", "zhaoxu",
        "--d_model", "512",
        "--d_ff", "512",
        "--do_predict", 
        "--target", "TOC",
        "--inverse",
        "--freq", "h",
        "--seed", str(seed),
        "--itr", "1"
    ], env=env)
    
    # 打印剩余GPU内存
    GPUs = GPUtil.getGPUs()
    for gpu in GPUs:
        print(f"GPU {gpu.id} 剩余内存: {gpu.memoryFree} MB")

if __name__ == "__main__":
    start_time = time.time()  # 记录开始时间
    
    models = ['Transformer', 'iDiffTransformer']
    seq_lens = [288]
    pred_lens = [48]
    e_layers_list = [2]
    seeds = [2023, 2024, 2025]
    
    all_combinations = list(itertools.product(models, seq_lens, pred_lens, e_layers_list, seeds))
    total_experiments = len(all_combinations)
    batch_size = min(3, total_experiments)
    
    print(f"总共 {total_experiments} 个实验待运行")
    
    for i in range(0, total_experiments, batch_size):
        batch = all_combinations[i:i+batch_size]
        print(f"运行第 {i//batch_size + 1} 批实验，共 {len(batch)} 个")
        
        with multiprocessing.Pool(processes=len(batch)) as pool:
            pool.map(run_experiment, batch)
        
        print(f"第 {i//batch_size + 1} 批实验完成")
    
    end_time = time.time()  # 记录结束时间
    total_time = end_time - start_time  # 计算总时长
    
    hours, remainder = divmod(total_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    print("所有实验完成")
    print(f"总运行时间：{int(hours)}小时 {int(minutes)}分钟 {int(seconds)}秒")
