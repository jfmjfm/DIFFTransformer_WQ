import csv

input_file = 'dataset/zongzhan/训练结果.txt'
output_file = 'metrics.csv'

metrics = []

with open(input_file, 'r', encoding='utf-8') as f:
    lines = f.readlines()
    for i in range(2, len(lines), 4):  # 从第3行开始,每4行一个循环
        if i < len(lines):
            metrics.append(lines[i].strip())

with open(output_file, 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['Model', 'Input_seq', 'Pred_seq', 'Num_heads', 'NSE_ratio'])  # 写入表头
    for line in metrics:
        parts = line.split(',')
        if len(parts) == 5:
            writer.writerow(parts)

print(f"数据已保存到 {output_file}")
