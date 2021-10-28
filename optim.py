import torch


def make_optimizer(cfg, model):
    optimizer = None
    if cfg.solver.optimizer == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.solver.learning_rate,
                                     weight_decay=cfg.solver.weight_decay)
    return optimizer
