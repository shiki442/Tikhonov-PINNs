import math
import torch
from torch.optim import Adam
from torch.optim import AdamW
from torch.optim import LBFGS


def get_pretrain_optimizer(u_net, pretrain_u_lr, pretrain_u_reg):
    pretrain_optimizer_u = Adam(u_net.parameters(), pretrain_u_lr, weight_decay=pretrain_u_reg)
    return pretrain_optimizer_u


def get_optimizer(q_net, u_net, **kwargs):
    q_lr, u_lr, name = kwargs.get('q_lr', 1.0e-4), kwargs.get('u_lr', 1.0e-4), kwargs.get('name', 'adam')
    if name == 'adam':
        return Adam([{'params': q_net.parameters(), 'lr': q_lr}, {'params': u_net.parameters(), 'lr': u_lr}], **kwargs.get('options', {}))
    elif name == 'adamw':
        return AdamW([{'params': q_net.parameters(), 'lr': q_lr}, {'params': u_net.parameters(), 'lr': u_lr}], **kwargs.get('options', {}))
    elif name == 'lbfgs':
        return LBFGS(list(q_net.parameters()) + list(u_net.parameters()), lr=min(q_lr, u_lr), **kwargs.get('options', {}))
    else:
        raise ValueError(f"Unknown optimizer name: {name}. Supported optimizers are 'adam', 'adamw', and 'lbfgs'.")


def get_scheduler(optimizer, warmup_steps=1000, cosine_annealing=True, total_steps=10000, last_epoch=-1):

    def lr_lambda(epoch):
        # warmup
        if epoch < warmup_steps:
            return float(epoch) / float(max(1, warmup_steps))

        if cosine_annealing:
            # cosine annealing
            progress = float(epoch - warmup_steps) / float(max(1, total_steps - warmup_steps))
            return 0.5 * (1.0 + math.cos(math.pi * progress))
        else:
            return 1.0

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda, last_epoch=last_epoch)
