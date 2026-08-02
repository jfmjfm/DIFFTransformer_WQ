import subprocess
import time
import GPUtil

def run_experiment2(model_id):
    subprocess.run(["python", "run.py", "--model_id", model_id,
  "--is_training", 1,
  "--root_path", "./dataset/zongzhan/",
  "--data_path", "wqprocessed.csv",
  "--model_id", "wq_192_96",
  "--model", model_id,
  "--data", "custom",
  "--features", "M",
  "--seq_len", 192,
  "--pred_len", 96,
  "--e_layers", 8,
  "--enc_in", 9,
  "--dec_in", 9,
  "--c_out", 9,
  "--des", "Exp",
  "--d_model", 512,
  "--d_ff", 512,
  "--do_predict",
  "--target", "NH4",
  "--inverse",
  "--freq", "h",
  "--itr", 1])
import itertools
import multiprocessing
import subprocess
import time
import os
import GPUtil

def get_free_gpu():
    GPUs = GPUtil.getGPUs()
    for gpu in GPUs:
        if gpu.memoryUsed < 1000:  # 假设小于1000MB为空闲
            return gpu.id
    return None

def run_experiment(args):
    model, seq_len, pred_len, e_layers = args
    gpu_id = None
    while gpu_id is None:
        gpu_id = get_free_gpu()
        if gpu_id is None:
            print(f"等待可用GPU: {model}, seq_len={seq_len}, pred_len={pred_len}, e_layers={e_layers}")
            time.sleep(60)  # 等待1分钟后再次检查
    
    model_id = f"{model}_seq{seq_len}_pred{pred_len}_e{e_layers}"
    print(f"运行 {model_id} 模型，使用 GPU {gpu_id}")
    
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    subprocess.run([
        "python", "run.py",
        "--is_training", "1",
        "--root_path", "./dataset/zongzhan/",   
        "--data_path", "wqprocessedwithprec.csv",
        "--model_id", model_id,
        "--model", model,
        "--data", "custom",
        "--features", "M",
        "--seq_len", str(seq_len),
        "--pred_len", str( pred_len),
        "--e_layers", str(e_layers),
        "--enc_in", "10",
        "--dec_in", "10",
        "--c_out", "10",
        "--des", "yanjiacun_prec",
        "--d_model", "512",
        "--d_ff", "512",
        "--do_predict", 
        "--target", "NH4",
        "--inverse",
        "--freq", "h",
        "--itr", "1"
    ], env=env)
    
    # 打印剩余GPU内存
    GPUs = GPUtil.getGPUs()
    for gpu in GPUs:
        print(f"GPU {gpu.id} 剩余内存: {gpu.memoryFree} MB")

if __name__ == "__main__":
    start_time = time.time()  # 记录开始时间
    
    models = ['iTransformer', 'iDiffTransformer','Transformer']
    seq_lens = [96, 144, 192, 240, 288]
    pred_lens = [48, 72, 96, 120]
    e_layers_list = [2, 3, 4, 5, 6, 7, 8]
    
    all_combinations = list(itertools.product(models, seq_lens, pred_lens, e_layers_list))
    total_experiments = len(all_combinations)
    batch_size = 35
    
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
