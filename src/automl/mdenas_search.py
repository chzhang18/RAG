import numpy as np
import torch
import logging
import torch.nn as nn
import torch.nn.functional as F
import torch.utils

import automl.darts_utils_cnn as utils

from utilstool.metrics import D1_metric, Thres_metric, EPE_metric
from utilstool.experiment import *

from automl.mdenas_basicmodel import BasicNetwork


class AutoSearch(object):
    # Implements a NAS methods MdeNAS
    def __init__(self, lr_min=0.001, momentum=0.9, weight_decay=3e-4, grad_clip=5, device='cuda: 0', writer=None, exp_name=None, save_name='EXP', args=None):

        self.device = device
        self.writer = writer
        self.exp_name = exp_name

        #import pdb; pdb.set_trace()
        self.lr_min = lr_min
        self.momentum = momentum
        self.weight_decay = weight_decay

        #import pdb; pdb.set_trace()
        
        if args.mode == 'search':
            self.lr = args.c_lr
            self.lr_a = args.c_lr_a
            self.weight_decay = args.c_lamb
            self.c_epochs = args.c_epochs
            self.c_batch = args.c_batch

        self.grad_clip = grad_clip
        self.save_name = save_name

        self.max_disp = 192

        self.model = BasicNetwork().to(device)
        logging.info("param size = %fMB", utils.count_parameters_in_MB(self.model))

        self.optimizer = torch.optim.SGD(params=self.model.parameters(), lr=self.lr, momentum=self.momentum,
                                         weight_decay=self.weight_decay)

    def search(self, t, train_data, valid_data, batch_size, nepochs):
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

        h_e = {
            'normal': torch.full((self.model.num_edges, self.model.num_ops), 0, dtype=torch.long),
            'reduce': torch.full((self.model.num_edges, self.model.num_ops), 0, dtype=torch.long)
        }
        h_a = {
            'normal': torch.full((self.model.num_edges, self.model.num_ops), 0.0),
            'reduce': torch.full((self.model.num_edges, self.model.num_ops), 0.0)
        }
        for epoch in range(nepochs):
            # 0 prepare
            lr = scheduler.get_lr()[0]
            logging.info('epoch %d lr %e', epoch, lr)
            print('epoch: {} lr: {}'.format(epoch, lr))
            genotype = self.model.genotype()
            logging.info('genotype = %s', genotype)
            # 1 sample
            p_n = self.model.probability()["normal"]
            p_r = self.model.probability()["reduce"]
            self.writer.add_histogram('CellArchHist/Normal',
                                      p_n, global_step=epoch)
            self.writer.add_histogram('CellArchHist/Reduce',
                                      p_r, global_step=epoch)
            selected_ops = {
                'normal': torch.multinomial(p_n, 1).view(-1),
                'reduce': torch.multinomial(p_r, 1).view(-1)
            }
            # 2 train
            fea_ops = selected_ops['normal']
            mat_ops = selected_ops['reduce']
            train_d1, train_obj, train_epe = self.train(train_queue, batch_size, fea_ops, mat_ops)
            #print('train_d1: {}'.format(train_d1))
            valid_d1, valid_obj, valid_epe = self.eval(valid_queue, batch_size, fea_ops, mat_ops)
            #print('valid_d1: {}'.format(valid_d1))
            #valid_d1, valid_obj = train_d1, train_obj
            # logging
            self.writer.add_scalars('Search_Cell_Loss/Task: {}'.format(t),
                                    {'train_loss': train_obj, 'valid_loss': valid_obj},
                                    global_step=epoch)
            self.writer.add_scalars('Search_Cell_D1_All/Task: {}'.format(t),
                                    {'train_d1': train_d1, 'valid_d1': valid_d1},
                                    global_step=epoch)
            self.writer.add_scalars('Search_Cell_EPE/Task: {}'.format(t),
                                    {'train_epe': train_epe, 'valid_epe': valid_epe},
                                    global_step=epoch)

            # 3 update h_e and h_a
            for cell_type in ['normal', 'reduce']:
                # for each edge
                for i, idx in enumerate(selected_ops[cell_type]):
                    h_e[cell_type][i][idx] += 1
                    h_a[cell_type][i][idx] = 1 - valid_d1

            # 4 update the probability
            for k in range(self.model.num_edges):
                dh_e_k = {
                    'normal': torch.reshape(h_e['normal'][k], (1, -1)) - torch.reshape(h_e['normal'][k], (-1, 1)),
                    'reduce': torch.reshape(h_e['reduce'][k], (1, -1)) - torch.reshape(h_e['reduce'][k], (-1, 1))
                }
                dh_a_k = {
                    'normal': torch.reshape(h_a['normal'][k], (1, -1)) - torch.reshape(h_a['normal'][k], (-1, 1)),
                    'reduce': torch.reshape(h_a['reduce'][k], (1, -1)) - torch.reshape(h_a['reduce'][k], (-1, 1))
                }
                for cell_type in ['normal', 'reduce']:
                    # vector1 = torch.sum((dh_e_k[cell_type] < 0) * (dh_a_k[cell_type] > 0), dim=1)
                    vector1 = torch.sum((dh_e_k[cell_type] < 0) * (dh_a_k[cell_type] > 0), dim=0)
                    # vector2 = torch.sum((dh_e_k[cell_type] > 0) * (dh_a_k[cell_type] < 0), dim=1)
                    vector2 = torch.sum((dh_e_k[cell_type] > 0) * (dh_a_k[cell_type] < 0), dim=0)
                    self.model.p[cell_type][k] += (self.lr_a * (vector1-vector2).float())
                    self.model.p[cell_type][k] = F.softmax(self.model.p[cell_type][k])

            # adjust learning according the scheduler
            scheduler.step()

            if valid_obj < best_loss:
                best_model = utils.get_model(self.model)
                best_loss = valid_obj

        # the best model and its architecture
        utils.set_model(self.model, best_model)
        print("The best architecture is", self.model.genotype())
        return self.model.genotype()

    def train(self, train_queue, batch_size, fea_ops, mat_ops):
        avg_train_scalars = AverageMeterDict()
        for step, sample in enumerate(train_queue):
            self.model.train()
            imgL, imgR, disp_gt = sample['left'].to(self.device), sample['right'].to(self.device), sample['disparity'].to(self.device)
            self.optimizer.zero_grad()
            disp_est = self.model(imgL, imgR, fea_ops, mat_ops)
            mask = (disp_gt < self.max_disp) & (disp_gt > 0)
            loss = F.smooth_l1_loss(disp_est[mask], disp_gt[mask], size_average=True)
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.optimizer.step()

            scalar_outputs = {"loss": loss}
            with torch.no_grad():
                epe = EPE_metric(disp_est, disp_gt, mask)
                scalar_outputs["EPE"] = epe
                d1 = D1_metric(disp_est, disp_gt, mask)
                scalar_outputs["D1"] = d1
                scalar_outputs["Thres1"] = Thres_metric(disp_est, disp_gt, mask, 1.0)
                scalar_outputs["Thres2"] = Thres_metric(disp_est, disp_gt, mask, 2.0)
                scalar_outputs["Thres3"] = Thres_metric(disp_est, disp_gt, mask, 3.0)
                scalar_outputs = tensor2float(scalar_outputs)
                avg_train_scalars.update(scalar_outputs)
                if step % 5 == 0:
                    logging.info('train %03d %e %f %f %f', step, loss.item(), tensor2float(epe), tensor2float(d1))
                    print('train: {} loss: {} EPE: {} D1: {}'.format(step, loss.item(), tensor2float(epe), tensor2float(d1)))
                #if step * batch_size > 200:
                #    break
                #if step > 10:
                #    break
        avg_train_scalars = avg_train_scalars.mean()
        loss_avg = avg_train_scalars["loss"]
        d1_avg = avg_train_scalars["D1"]
        epe_avg = avg_train_scalars["EPE"]

        return d1_avg, loss_avg, epe_avg

    def eval(self, valid_queue, batch_size, fea_ops, mat_ops):
        avg_eval_scalars = AverageMeterDict()
        self.model.eval()
        with torch.no_grad():
            for step, sample in enumerate(valid_queue):
                imgL, imgR, disp_gt = sample['left'].to(self.device), sample['right'].to(self.device), sample['disparity'].to(self.device)
                disp_est = self.model(imgL, imgR, fea_ops, mat_ops)
                mask = (disp_gt < self.max_disp) & (disp_gt > 0)
                loss = F.smooth_l1_loss(disp_est[mask], disp_gt[mask], size_average=True)
                scalar_outputs = {"loss": loss}
                epe = EPE_metric(disp_est, disp_gt, mask)
                scalar_outputs["EPE"] = epe
                d1 = D1_metric(disp_est, disp_gt, mask)
                scalar_outputs["D1"] = d1
                scalar_outputs["Thres1"] = Thres_metric(disp_est, disp_gt, mask, 1.0)
                scalar_outputs["Thres2"] = Thres_metric(disp_est, disp_gt, mask, 2.0)
                scalar_outputs["Thres3"] = Thres_metric(disp_est, disp_gt, mask, 3.0)
                scalar_outputs = tensor2float(scalar_outputs)
                avg_eval_scalars.update(scalar_outputs)
                
                if step % 5 == 0:
                    logging.info('valid %03d %e %f %f %f', step, loss.item(), tensor2float(epe), tensor2float(d1))
                    print('valid: {} loss: {} EPE: {} D1: {}'.format(step, loss.item(), tensor2float(epe), tensor2float(d1)))
                #if step * batch_size > 200:
                #    break
                #if step > 10:
                #    break

        avg_eval_scalars = avg_eval_scalars.mean()
        loss_avg = avg_eval_scalars["loss"]
        d1_avg = avg_eval_scalars["D1"]
        epe_avg = avg_eval_scalars["EPE"]

        return d1_avg, loss_avg, epe_avg

    


