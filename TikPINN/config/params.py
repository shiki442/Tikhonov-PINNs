import yaml
import copy
import os
import numpy as np


def bash_cfg_noise():
    # 读取 params.yaml 文件
    with open('./TikPINN/config/params.yaml', 'r', encoding='utf-8') as f:
        params = yaml.safe_load(f)

    # ids = ['15', '25', '35', '45']
    # lambdas = [1.0e-4, 1.0e-5, 1.0e-6, 1.0e-7, 1.0e-8, 1.0e-9, 0.0]
    ids = ['00', '05', '10', '15', '20', '25', '30', '35', '40', '45', '50']
    lambdas = [1.0e-7, 1.0e-8, 1.0e-9, 0.0]
    dir_path = ""
    for noise_str in ids:
        for lamb in lambdas:
            params["task"]["noise_str"] = noise_str
            params["loss_params"]["lamb"] = lamb
            dir_path = f"./TikPINN/config/"
            # 保存为新的 yaml 文件
            if lamb == 0.0:
                path = os.path.join(dir_path, f"params_00_{noise_str}.yaml")
            else:
                lamb = int(-np.log10(lamb))
                path = os.path.join(dir_path, f"params_0{lamb}_{noise_str}.yaml")
            with open(path, 'w', encoding='utf-8') as f_out:
                yaml.dump(params, f_out, allow_unicode=True)

    print("参数文件已批量生成。")


def bash_cfg_w():
    # 读取 params.yaml 文件
    with open('./TikPINN/config/params.yaml', 'r', encoding='utf-8') as f:
        params = yaml.safe_load(f)

    # seeds = [1e-3, 1e-4, 1e-5, 1e-6, 1e-7]
    widths = [8, 16, 24, 32, 40, 48, 56, 64, 72, 80]
    seeds = [41, 42, 43, 58, 47]
    dir_path = ""
    for seed in seeds:
        for w in widths:
            params["seed"] = seed
            params["q_net_params"]["width_list"] = [w, w]
            params["u_net_params"]["width_list"] = [w, w]
            dir_path = f"./TikPINN/config/"
            # 保存为新的 yaml 文件
            path = os.path.join(dir_path, f"params_{seed}_{w}.yaml")
            with open(path, 'w', encoding='utf-8') as f_out:
                yaml.dump(params, f_out, allow_unicode=True)

    print("参数文件已批量生成。")


def bash_cfg_n():
    with open('./TikPINN/config/params.yaml', 'r', encoding='utf-8') as f:
        params = yaml.safe_load(f)

    n_list = [10, 50, 100, 500, 1000, 5000, 10000, 50000]
    seeds = [41, 42, 43, 58, 47]
    dir_path = ""
    for n in n_list:
        for seed in seeds:
            params["seed"] = seed
            params["dataloader_params"]["n_samples"] = n

            dir_path = f"./TikPINN/config/"
            path = os.path.join(dir_path, f"params_{n}_{seed}.yaml")

            # 保存为新的 yaml 文件
            with open(path, 'w', encoding='utf-8') as f_out:
                yaml.dump(params, f_out, allow_unicode=True)


def bash_cfg_reg():
    with open('./TikPINN/config/params.yaml', 'r', encoding='utf-8') as f:
        params = yaml.safe_load(f)

    regs = ['0', 'L2', 'H2']
    lambdas = [1e-3, 1e-4, 1e-5, 1e-6, 1e-7]
    dir_path = ""
    for reg in regs:
        for lamb in lambdas:
            params["loss_params"]["lamb"] = lamb
            params["loss_params"]["regularization"] = reg

            dir_path = f"./TikPINN/config/"
            lamb = int(-np.log10(lamb))
            path = os.path.join(dir_path, f"params_{reg}_0{lamb}.yaml")

            # 保存为新的 yaml 文件
            with open(path, 'w', encoding='utf-8') as f_out:
                yaml.dump(params, f_out, allow_unicode=True)


def bash_args():
    dir_path = f"./TikPINN/config/"
    # 获取当前目录下所有以 .yaml 结尾的文件名
    yaml_files = [f for f in os.listdir(dir_path) if f.endswith('.yaml')]

    # 写入到 args.txt，每行一个文件名
    with open(os.path.join(dir_path, 'args.txt'), 'w', encoding='utf-8') as f:
        for name in yaml_files:
            f.write(f"{name}\n")

    print("所有 yaml 文件名已写入 args.txt。")


if __name__ == '__main__':
    bash_cfg_w()
    bash_args()
