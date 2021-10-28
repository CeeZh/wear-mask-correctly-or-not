import logging
import os
import time
import torch.nn as nn
import argparse
from tqdm import tqdm
import shutil
import pdb

import utils
from utils import setup_logger
from model import build_model
from trainer import BaseTrainer
from dataset import make_dataloader



def parse_args():
    parser = argparse.ArgumentParser(description="ReID training")
    parser.add_argument('-c', '--config_file', type=str,
                        help='the path to the training config')
    parser.add_argument('-t', '--test', action='store_true',
                        default=False, help='Model test')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    if args.test:
        test(args)
    else:
        train(args)


def train(args):
    cfg = utils.process_cfg(args.config_file)
    output_dir = os.path.join(cfg.exp_base, cfg.exp_name, str(time.time()))
    cfg.output_dir = output_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    shutil.copy(args.config_file, cfg.output_dir)
    setup_logger(output_dir)
    logger = logging.getLogger('train')
    logger.info('Train with config:\n{}'.format(cfg))

    train_dl = make_dataloader(cfg, 'train')
    val_dl = make_dataloader(cfg, 'validation')

    model = build_model(cfg)

    loss_func = nn.CrossEntropyLoss()

    trainer = BaseTrainer(cfg, model, train_dl, val_dl, loss_func)

    for epoch in range(trainer.epochs):
        for batch in tqdm(trainer.train_dl):
            trainer.step(batch)
            trainer.finish_batch()
        trainer.finish_epoch()


def test(cfg):
    pass


if __name__ == '__main__':
    main()

