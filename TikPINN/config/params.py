import yaml
import copy
import os
import numpy as np

# 读取 params.yaml 文件
with open('D:/OneDrive/PC/Code Library/Tikhonov-PINNs/TikPINN/config/params.yaml', 'r', encoding='utf-8') as f:
    params = yaml.safe_load(f)

# ids = ['01', '02', '05', '10', '20', '30', '40', '50']
# lambdas = [1.0e-4, 1.0e-5, 1.0e-6, 1.0e-7, 1.0e-8, 1.0e-9, 0.0]
ids = ['01', '10', '20', '50']
lambdas = [1.0e-6, 1.0e-7, 1.0e-8, 1.0e-9, 0.0]
for noise_str in ids:
    for lamb in lambdas:
        params["task"]["noise_str"] = noise_str
        params["loss_params"]["lamb"] = lamb
        dir_path = f"D:/OneDrive/PC/Code Library/Tikhonov-PINNs/TikPINN/config/"
        # 保存为新的 yaml 文件
        if lamb == 0.0:
            path = os.path.join(dir_path, f"params_00_{noise_str}.yaml")
        else:
            lamb = int(-np.log10(lamb))
            path = os.path.join(dir_path, f"params_0{lamb}_{noise_str}.yaml")
        with open(path, 'w', encoding='utf-8') as f_out:
            yaml.dump(params, f_out, allow_unicode=True)

print("参数文件已批量生成。")

# 获取当前目录下所有以 .yaml 结尾的文件名
yaml_files = [f for f in os.listdir(dir_path) if f.endswith('.yaml')]

# 写入到 args.txt，每行一个文件名
with open(os.path.join(dir_path, 'args.txt'), 'w', encoding='utf-8') as f:
    for name in yaml_files:
        f.write(f"{name}\n")

print("所有 yaml 文件名已写入 args.txt。")
