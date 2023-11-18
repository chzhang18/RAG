import os
import sys
import time
import glob
import numpy as np
import torch
import logging
import argparse
import torch.nn as nn
import torch.utils
import torch.nn.functional as F
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn

import automl.darts_utils_cnn as utils

from torch.autograd import Variable
from automl.darts_basicmodel import BasicNetwork
from automl.darts_model import Network
from automl.darts_architecture import Architect


class AutoSearch(object):
    # Implements a NAS methods Darts
    def __init__(self, num_cells, num_class=10, input_size=None, lr=0.025, lr_a=3e-4, lr_min=0.001, momentum=0.9,
                 weight_decay=3e-4, weight_decay_a=1e-3, grad_clip=5, unrolled=False,
                 device='cuda: 0', writer=None, exp_name=None, save_name='EXP', args=None):
        self.num_cells = num_cells
        self.num_classes = num_class
        self.input_size = input_size

        self.device = device
        self.writer = writer
        self.exp_name = exp_name

        self.lr = lr
        self.lr_a = lr_a
        self.lr_min = lr_min
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.weight_decay_a = weight_decay_a
        if args.mode == 'search':
            self.weight_decay = args.lamb_w0
            self.weight_decay_a = args.lamb_wa0
            self.lr = args.lr0
            self.lr_a = args.lr_s0

        self.grad_clip = grad_clip
        self.unrolled = unrolled
        self.save_name = save_name

        self.criterion = nn.CrossEntropyLoss().to(device)
        self.model = BasicNetwork(self.input_size[0], 16, self.num_classes, self.num_cells, self.criterion,
                                  device=self.device).to(device)
        logging.info("param size = %fMB", utils.count_parameters_in_MB(self.model))

        params = []
        for name, p in self.model.named_parameters():
            if name not in ['alphas_normal', 'alphas_reduce']:
                params.append(p)

        self.optimizer = torch.optim.SGD(params=params, lr=self.lr, momentum=self.momentum,
                                         weight_decay=self.weight_decay)

        self.architect = Architect(self.model, lr_a=self.lr_a, momentum=self.momentum,
                                   weight_decay=self.weight_decay, weight_decay_a=self.weight_decay_a)

    def search(self, train_data, valid_data, batch_size, nepochs):
        """ search a model genotype for the given task

        :param train_data: the dataset of training data of the given task
        :param valid_data: the dataset of valid data of the given task
        :param batch_size: the batch size of training
        :param nepochs: the number of training epochs
        :return:
            genotype: the selected architecture for the given task
        """
        # dataloader of training data
        num_train = len(train_data)
        indices = list(range(num_train))
        split = int(np.floor(0.5 * num_train))

        train_queue = torch.utils.data.DataLoader(
            train_data, batch_size=batch_size,
            sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[:split]),
            pin_memory=True, num_workers=4)
        # dataloader of valid date
        valid_queue = torch.utils.data.DataLoader(
            train_data, batch_size=batch_size,
            sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[split:num_train]),
            pin_memory=True, num_workers=4)
        # the scheduler of learning rate of model parameters optimizer
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, nepochs,
                                                               eta_min=self.lr_min)
        best_loss = np.inf
        best_model = utils.get_model(self.model)

        for epoch in range(nepochs):
            lr = scheduler.get_lr()[0]
            logging.info('epoch %d lr %e', epoch, lr)
            print('epoch: {} lr: {}'.format(epoch, lr))

            genotype = self.model.genotype()
            logging.info('genotype = %s', genotype)

            arch_n = F.softmax(self.model.arch_parameters()["alphas_normal"], dim=-1)
            arch_r = F.softmax(self.model.arch_parameters()["alphas_reduce"], dim=-1)
            print(arch_n)
            # self.writer.add_images('CellArchValue/Normal',
            #                        arch_n, global_step=epoch, dataformats='HW')
            self.writer.add_histogram('CellArchHist/Normal',
                                     arch_n, global_step=epoch)
            print(arch_r)
            # self.writer.add_images('CellArchValue/Reduce',
            #                        arch_r, global_step=epoch, dataformats='HW')
            self.writer.add_histogram('CellArchHist/Reduce',
                                     arch_r, global_step=epoch)

            # training
            train_acc, train_obj = self.train(train_queue, valid_queue, lr)
            print('train_acc: {}'.format(train_acc))

            # validation
            valid_acc, valid_obj = self.eval(valid_queue)
            print('valid_acc: {}'.format(valid_acc))

            # logging
            self.writer.add_scalars('Search_Loss/Task: 0',
                                    {'train_loss': train_obj, 'valid_loss': valid_obj},
                                    global_step=epoch)
            self.writer.add_scalars('Search_Accuracy/Task: 0',
                                    {'train_acc': train_acc, 'valid_acc': valid_acc},
                                    global_step=epoch)

            # adjust learning according the scheduler
            scheduler.step()

            if valid_obj < best_loss:
                best_model = utils.get_model(self.model)
                best_loss = valid_obj

        # the best model and its architecture
        utils.set_model(self.model, best_model)
        print("The best architecture is", self.model.genotype())
        return self.model.genotype()

    def train(self, train_queue, valid_queue, lr):
        objs = utils.AverageMeter()
        top1 = utils.AverageMeter()
        # top5 = utils.AverageMeter()

        for step, (x, y) in enumerate(train_queue):
            self.model.train()
            n = x.size(0)

            x, y = x.to(self.device), y.to(self.device)

            # get a random mini batch from the search queue with replacement
            x_s, y_s = next(iter(valid_queue))
            x_s, y_s = x_s.to(self.device), y_s.to(self.device)

            self.architect.step(x, y, x_s, y_s, lr, self.optimizer, unrolled=self.unrolled)

            self.optimizer.zero_grad()
            logits = self.model(x)
            loss = self.criterion(logits, y)

            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.optimizer.step()

            prec1 = utils.accuracy(logits, y, topk=1)
            objs.update(loss.item(), n)
            top1.update(prec1.item(), n)
            # top5.update(prec5.item(), n)

            if step % 100 == 0:
                logging.info('train %03d %e %f %f', step, objs.avg, top1.avg)
                print('train: {} loss: {} acc: {}'.format(step, objs.avg, top1.avg))

        return top1.avg, objs.avg

    def eval(self, valid_queue):
        objs = utils.AverageMeter()
        top1 = utils.AverageMeter()
        # top5 = utils.AverageMeter()
        self.model.eval()

        for step, (x, y) in enumerate(valid_queue):

            x, y = x.to(self.device), y.to(self.device)

            logits = self.model(x)
            loss = self.criterion(logits, y)

            prec1 = utils.accuracy(logits, y, topk=1)
            n = x.size(0)
            objs.update(loss.item(), n)
            top1.update(prec1.item(), n)
            # top5.update(prec5.item(), n)

            if step % 100 == 0:
                logging.info('valid %03d %e %f %f', step, objs.avg, top1.avg)
                print('valid: {} loss: {} acc: {}'.format(step, objs.avg, top1.avg))

        return top1.avg, objs.avg

    def create_model(self, archi, num_cells, input_size, task_classes, init_channel):

        return Network(input_size, task_classes, num_cells, init_channel, archi, device=self.device)

