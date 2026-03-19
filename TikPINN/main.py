from argparse import ArgumentParser

import torch
import torch.distributed as dist
import yaml
import os

from model.data import get_dataloader, get_ddp_dataloader
from model.loss import get_loss
from model.nn import get_network
from model.optim import get_optimizer, get_pretrain_optimizer, get_scheduler
from model.train import train
from model.utils import check_config, set_seed, save_config

import torch.multiprocessing as mp
from torch.utils.data import DataLoader, DistributedSampler, Dataset
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter

os.environ['MASTER_ADDR'] = '127.0.0.1'
os.environ['MASTER_PORT'] = '64060'


def get_problem_class(idx):
    """Get problem class by task index."""
    if idx == "01":
        from GenerateData.problems import Example01Problem

        return Example01Problem
    elif idx == "02":
        from GenerateData.problems import Example02Problem

        return Example02Problem
    elif idx == "06":
        from GenerateData.problems import Example06Problem

        return Example06Problem
    elif idx == "03":
        from GenerateData.problems import Example03Problem

        return Example03Problem
    else:
        raise ValueError(f"Unknown task idx: {idx}")


def main(rank, world_size, config, config_file_path=None):
    # problem parameters
    idx = config['task']['idx']
    noise_str = config['task']['noise_str']
    check_config(config, config_file_path=config_file_path)

    if world_size > 1:
        # Initialize the process group for distributed training
        dist.init_process_group("nccl", init_method="env://", rank=rank, world_size=world_size)
        torch.cuda.set_device(rank)
        set_seed(rank=rank, seed=config["seed"])

        print(f"Construct neural networks ...")
        q_net = get_network(**config["q_net_params"]).to(rank)
        u_net = get_network(**config["u_net_params"]).to(rank)
        ddp_q_net = DDP(q_net, device_ids=[rank])
        ddp_u_net = DDP(u_net, device_ids=[rank])
        q_net = ddp_q_net
        u_net = ddp_u_net

        print(f"Load data on device {rank}...")
        dataloader = get_ddp_dataloader(idx=idx, noise_str=noise_str, world_size=world_size, rank=rank, **config["dataloader_params"])

    elif str(rank) in ['cpu', 'cuda', 'cuda:0']:
        set_seed(rank=0, seed=config["seed"])
        print(f"Construct neural networks ...")
        q_net = get_network(**config["q_net_params"]).to(rank)
        u_net = get_network(**config["u_net_params"]).to(rank)

        print(f"Load data ...")
        dataloader = get_dataloader(idx=idx, noise_str=noise_str, **config["dataloader_params"])
    else:
        raise ValueError(f"Invalid rank: {rank}. It should be 'cpu', 'cuda', or an integer for distributed training.")

    print(f"Define loss function ...")
    tikpinn_loss = get_loss(**config["loss_params"])
    print(f"Define optimizer ...")
    pretrain_u_optimizer = get_pretrain_optimizer(u_net, **config["pretrain_optim_params"])
    optimizer_adam = get_optimizer(q_net, u_net, **config["optim_params_adam"])
    optimizer_lbfgs = get_optimizer(q_net, u_net, **config["optim_params_lbfgs"])
    scheduler_adam = get_scheduler(optimizer_adam, **config["scheduler_params"])
    optimizers = [optimizer_adam, optimizer_lbfgs]
    schedulers = [scheduler_adam, None]  # Only Adam has scheduler, LBFGS doesn't

    # Get checkpoint path from config if exists
    results_path = config["train_params"]["results_path"]
    checkpoint_path = config.get("checkpoint_path", None)
    if checkpoint_path is not None and not os.path.isabs(checkpoint_path):
        checkpoint_path = os.path.join(results_path, checkpoint_path)

    # Get checkpoint save parameters
    ckpt_every_n_epochs = config.get("checkpoint_params", {}).get("every_n_epochs", 100)
    ckpt_save_last = config.get("checkpoint_params", {}).get("save_last", True)

    # Create problem instance for error computation
    ProblemClass = get_problem_class(idx)
    problem = ProblemClass()

    # Create TensorBoard writer (only on main process)
    writer = None
    if world_size == 1 or rank == 0:
        writer = SummaryWriter(log_dir=results_path)

    print(f"Begin training ...")
    train(
        rank,
        dataloader,
        q_net,
        u_net,
        tikpinn_loss,
        pretrain_u_optimizer,
        optimizers,
        schedulers,
        config["train_params"]["pretrain_epochs_u"],
        config["train_params"]["num_epochs"],
        results_path,
        problem,
        writer=writer,
        eval_points=config.get("eval_points", 101),
        checkpoint_path=checkpoint_path,
        ckpt_every_n_epochs=ckpt_every_n_epochs,
        ckpt_save_last=ckpt_save_last,
        heatmap_every_n_epochs=config.get("heatmap_every_n_epochs", 100),
    )

    # Close TensorBoard writer
    if writer is not None:
        writer.close()

    # Save final config after training
    if world_size == 1 or rank == 0:
        save_config(config, results_path, 'config_final.yaml')

    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    # parameters
    parser = ArgumentParser(description="Basic paser")
    parser.add_argument("--config_path", type=str, help="Path to the configuration file", default='params.yaml')
    args = parser.parse_args()
    config_file = args.config_path
    with open(config_file, 'r') as stream:
        config = yaml.load(stream, yaml.FullLoader)
    print(config)

    # model
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    world_size = torch.cuda.device_count()

    # Get absolute path for config file
    config_file_abs = os.path.abspath(config_file)

    # train
    print(f"Begin training ...")
    if world_size > 1:
        mp.spawn(main, args=(world_size, config, config_file_abs), nprocs=world_size, join=True)
    else:
        main(device, world_size, config, config_file_abs)  # If only one GPU, run main directly
