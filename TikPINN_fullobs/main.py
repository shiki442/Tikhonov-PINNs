from argparse import ArgumentParser

import torch
import yaml

from model.data import get_dataloader
from model.loss import get_loss
from model.nn import get_network
from model.optim import get_optimizer, get_pretrain_optimizer
from model.train import train

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
    idx = "05"
    noise_str = "50"

    # model
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Load data ...")
    dataloader = get_dataloader(
        idx=idx, noise_str=noise_str, **config["dataloader_params"])
    print(f"Construct neural networks ...")
    q_net = get_network(**config["q_net_params"]).to(device)
    u_net = get_network(**config["u_net_params"]).to(device)
    print(f"Define loss function ...")
    tikpinn_loss = get_loss(**config["loss_params"])
    print(f"Define optimizer ...")
    pretrain_u_optimizer = get_pretrain_optimizer(u_net, **config["pretrain_optim_params"])
    optimizer = get_optimizer(q_net, u_net, **config["optim_params"])

    # train
    print(f"Begin training ...")
    loss_log = train(device, dataloader, q_net, u_net, tikpinn_loss,
                     pretrain_u_optimizer, optimizer,
                     **config["train_params"])
