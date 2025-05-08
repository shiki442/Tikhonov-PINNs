from torch.optim import Adam


def get_pretrain_optimizer(u_net, pretrain_u_lr, pretrain_u_reg):
    pretrain_optimizer_u = Adam(u_net.parameters(), pretrain_u_lr,
                                weight_decay=pretrain_u_reg)
    return pretrain_optimizer_u


def get_optimizer(q_net, u_net, q_lr, u_lr, weight_decay):
    return Adam([{'params': q_net.parameters(), 'lr': q_lr},
                 {'params': u_net.parameters(), 'lr': u_lr}],
                weight_decay=weight_decay)
