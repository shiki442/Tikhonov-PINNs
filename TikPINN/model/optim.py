import math
import torch
from torch.optim import Adam
from torch.optim import AdamW
from torch.optim import LBFGS


def get_pretrain_optimizer(u_net, pretrain_u_lr, pretrain_u_reg):
    pretrain_optimizer_u = Adam(u_net.parameters(), pretrain_u_lr, weight_decay=pretrain_u_reg)
    return pretrain_optimizer_u


def get_optimizer(q_net, u_net, q_lr, u_lr, weight_decay=0.0, name='adam'):
    if name == 'adam':
        return Adam([{'params': q_net.parameters(), 'lr': q_lr}, {'params': u_net.parameters(), 'lr': u_lr}], weight_decay=weight_decay)
    elif name == 'adamw':
        return AdamW([{'params': q_net.parameters(), 'lr': q_lr}, {'params': u_net.parameters(), 'lr': u_lr}], weight_decay=weight_decay)
    elif name == 'lbfgs':
        return LBFGS(
            list(q_net.parameters()) + list(u_net.parameters()),
            lr=min(q_lr, u_lr),
            max_iter=20,
            history_size=100,
            line_search_fn='strong_wolfe',
        )
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


# class OptimizerManager:
#     def __init__(self, q_net, u_net, q_lr, u_lr, weight_decay=0.0, warmup_steps=1000, total_steps=10000, last_epoch=-1):
#         self.optimizer = Adam(
#             [{'params': q_net.parameters(), 'lr': q_lr}, {'params': u_net.parameters(), 'lr': u_lr}], weight_decay=weight_decay
#         )
#         self.warmup_steps = warmup_steps
#         self.total_steps = total_steps
#         self.global_step = 0

#         def lr_lambda(current_step):
#             # warmup
#             if current_step < warmup_steps:
#                 return float(current_step) / float(max(1, warmup_steps))
#             # cosine annealing
#             progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
#             return 0.5 * (1.0 + math.cos(math.pi * progress))

#         self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lr_lambda, last_epoch=last_epoch)

#     def step(self):
#         self.optimizer.step()
#         self.scheduler.step()
#         self.global_step += 1

#     def zero_grad(self):
#         self.optimizer.zero_grad()

#     def get_lr(self):
#         return self.optimizer.param_groups[0]['lr']
