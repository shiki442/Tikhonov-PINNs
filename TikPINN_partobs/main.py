from argparse import ArgumentParser

import torch
import yaml
import numpy as np

from TikPINN.data import get_dataloader
from TikPINN.loss import get_loss
from TikPINN.nn import get_network
from TikPINN.optim import get_optimizer, get_pretrain_optimizer
from TikPINN.train import train
from TikPINN.utils import output_path

if __name__ == "__main__":
    # parameters
    parser = ArgumentParser(description="Basic paser")
    parser.add_argument("--config_path", type=str,
                        help="Path to the configuration file")
    args = parser.parse_args()
    config_file = args.config_path
    with open(config_file, 'r') as stream:
        config = yaml.load(stream, yaml.FullLoader)
    print(config)

    # problem parameters
    idx = "01"
    noise_str = "01"
    output_path(config, noise_str)

    # model
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Load data ...")
    dataloader = get_dataloader(
        idx=idx, noise_str=noise_str, **config["dataloader_params"])
    print(f"Construct neural networks ...")
    q_net = get_network(**config["q_net_params"]).to(device)
    u_net = get_network(**config["u_net_params"]).to(device)
    print(f"Define loss function ...")
    tikpinn_loss = get_loss(**config["loss_params"], idx=idx,
                          noise_str=noise_str, data_path=config["dataloader_params"]["data_path"])
    print(f"Define optimizer ...")
    pretrain_u_optimizer = get_pretrain_optimizer(u_net, **config["pretrain_optim_params"])
    optimizer = get_optimizer(q_net, u_net, **config["optim_params"])

    # train
    print(f"Begin training ...")
    loss_log = train(device, dataloader, q_net, u_net, tikpinn_loss,
                     pretrain_u_optimizer, optimizer,
                     **config["train_params"])
