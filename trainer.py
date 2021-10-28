import torch
from tqdm import tqdm
import numpy as np
import logging
from utils import AverageMeter
import os
from optim import make_optimizer
from torch.utils.tensorboard import SummaryWriter


class BaseTrainer(object):
    def __init__(self, cfg, model, train_dl, val_dl, loss_func):
        self.cfg = cfg
        self.model = model
        self.train_dl = train_dl
        self.val_dl = val_dl
        self.loss_func = loss_func

        self.loss_avg = AverageMeter()
        self.acc_avg = AverageMeter()
        self.loss_batch = 0
        self.acc_batch = 0
        self.min_loss = np.inf
        self.cur_epoch = 0  # start from 0
        self.cur_batch = 0  # start from 0
        self.steps = 0  # total batch

        self.logger = logging.getLogger('train')
        self.log_period = cfg.solver.log_period
        self.checkpoint_period = cfg.solver.checkpoint_period
        self.eval_period = cfg.solver.eval_period

        self.output_dir = cfg.output_dir
        self.epochs = cfg.solver.num_epochs
        self.device = cfg.device
        if self.device != 'cpu':
            self.model.cuda()
        self.optim = make_optimizer(cfg, self.model)

        self.writer = SummaryWriter(self.output_dir)
        self.logger.info('Trainer Built.')

    def finish_batch(self):
        if self.steps % self.log_period == 0 and self.steps != 0:
            self.writer.add_scalar('loss/train', self.loss_batch, self.steps)
            self.writer.add_scalar('acc/train', self.acc_batch, self.steps)
        if self.steps % self.checkpoint_period == 0 and self.steps != 0:
            self.save()
        if self.steps % self.eval_period == 0 and self.steps != 0:
            loss, acc = self.evaluate()
            self.logger.info('Validation Result:')
            self.logger.info('loss: {}, acc: {}'.format(loss, acc))
            self.logger.info('-' * 20)
            self.writer.add_scalar('loss/test', loss, self.steps)
            self.writer.add_scalar('acc/test', acc, self.steps)
            if loss < self.min_loss:
                self.save(True)
        self.cur_batch += 1
        self.steps += 1

    def finish_epoch(self):
        self.cur_batch = 0
        self.logger.info('Epoch {} done'.format(self.cur_epoch))
        self.logger.info('loss: {}, acc: {}'.format(self.loss_avg.avg, self.acc_avg.avg))
        self.logger.info('-' * 20)
        self.cur_epoch += 1

    def step(self, batch):
        self.model.train()
        self.optim.zero_grad()
        [img, label] = batch
        if self.device != 'cpu':
            img, label = img.cuda(), label.cuda()
        probs = self.model(img)
        loss = self.loss_func(probs, label)
        loss.backward()
        self.optim.step()

        acc = torch.mean(torch.eq(torch.argmax(probs, dim=1), label).float())

        self.loss_batch, self.acc_batch = loss.cpu().item(), acc.cpu().item()
        self.loss_avg.update(self.loss_batch)
        self.acc_avg.update(self.acc_batch)

    def evaluate(self):
        self.model.eval()
        loss_avg, acc_avg = AverageMeter(), AverageMeter()
        for data, label in self.val_dl:
            probs = self.model(data)
            loss = self.loss_func(probs, label)
            acc = torch.mean(torch.eq(torch.argmax(probs, dim=1), label).float())
            loss_avg.update(loss.cpu().item(), len(data))
            acc_avg.update(acc.cpu().item(), len(data))
        loss, acc = loss_avg.avg, acc_avg.avg
        return loss, acc

    def save(self, is_best=False):
        if is_best:
            torch.save(self.model.state_dict(),
                       os.path.join(self.output_dir,
                                    'best_step_{}_epoch_{}_batch_{}.pth'.format(
                                        self.steps, self.cur_epoch, self.cur_batch)))
        else:
            torch.save(self.model.state_dict(),
                       os.path.join(self.output_dir,
                                    'checkpoint_step_{}_epoch_{}_batch_{}.pth'.format(
                                        self.steps, self.cur_epoch, self.cur_batch)))


