import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn

import utils

from copy import deepcopy

from automl.mdenas_search import AutoSearch
from models.rag_model import Network
from automl.darts_genotypes import mdenas_pmnist, mdenas_mixture

from automl.genotypes_2d import Genotype

from utilstool.metrics import D1_metric, Thres_metric, EPE_metric
from utilstool.experiment import *

def compute_errors(gt, pred):
    thresh = np.maximum((gt / pred), (pred / gt))
    d1 = (thresh < 1.25).mean()
    d2 = (thresh < 1.25 ** 2).mean()
    d3 = (thresh < 1.25 ** 3).mean()

    rms = (gt - pred) ** 2
    rms = np.sqrt(rms.mean())

    log_rms = (np.log(gt) - np.log(pred)) ** 2
    log_rms = np.sqrt(log_rms.mean())

    abs_rel = np.mean(np.abs(gt - pred) / gt)
    sq_rel = np.mean(((gt - pred) ** 2) / gt)

    err = np.log(pred) - np.log(gt)
    silog = np.sqrt(np.mean(err ** 2) - np.mean(err) ** 2) * 100

    err = np.abs(np.log10(pred) - np.log10(gt))
    log10 = np.mean(err)

    return [silog, abs_rel, log10, rms, sq_rel, log_rms, d1, d2, d3]


class Appr(object):
    def __init__(self, clipgrad=5, writer=None, exp_name="None", device='cuda', args=None):
        # the number of cells
        self.model = None
        self.archis = []
        # the device and tensorboard writer for training
        self.device = device
        self.writer = writer
        self.exp_name = exp_name

        self.clipgrad = clipgrad

        self.args = args

        if args.mode == 'search':
            # mode: search the best hyper-parameter
            # the hyper parameters in cell search stage
            self.c_epochs = args.c_epochs
            self.c_batch = args.c_batch
            self.c_lr = args.c_lr
            self.c_lr_a = args.c_lr_a
            self.c_lamb = args.c_lamb
            # the hyper parameters in operation search stage
            self.o_epochs = args.o_epochs
            self.o_batch = args.o_batch
            self.o_lr = args.o_lr
            self.o_lr_a = args.o_lr_a
            self.o_lamb = args.o_lamb
            self.o_size = args.o_size
            # the hyper parameters in training stage
            self.epochs = args.epochs
            self.batch = args.batch
            self.lr = args.lr
            self.lamb = args.lamb

        #import pdb; pdb.set_trace()

        # define the search method
        self.auto_ml = AutoSearch(device=self.device, writer=self.writer, exp_name=self.exp_name, args=args)
        # define optimizer and loss function
        self.max_disp = 192
        self.optimizer = None
        self.optimizer_o = None

        self.silog_criterion = silog_loss()

    def _get_optimizer(self, lr):
        # optimizer to train the model parameters
        if lr is None:
            lr = self.lr

        return torch.optim.SGD(filter(lambda p: p.requires_grad, self.model.parameters()),
                               lr=lr, weight_decay=self.lamb, momentum=0.9)

    def _get_optimizer_o(self, lr=None):
        if lr is None:
            lr = self.o_lr
        params = self.model.get_param(self.model.new_models)

        return torch.optim.SGD(params=params, lr=lr, momentum=0.9, weight_decay=self.o_lamb)

    def train(self, t, train_data, valid_data, device='cuda'):
        # training network for task t
        # 1 search cell for task t
        #import pdb; pdb.set_trace()
        genotype = self.search_cell(t, train_data, valid_data, self.c_batch, self.c_epochs, device=device)
        # 2 search operation for task t > 0
        if t > 0:
            # 2.1 expand
            self.model.expand(t, genotype, device)
            # 2.2 freeze the model
            utils.freeze_model(self.model)
            self.model.modify_param(self.model.new_models, True)
            # 1.2.3 search the best expand action, the best action, and the best architecture
            self.search_t(t, train_data, valid_data, self.o_batch, self.o_epochs, device=device)
            best_archi = self.model.select(t)
            print("best_archi is {}".format(best_archi))
            self.writer.add_text("Archi for task {}".format(t),
                                "{}".format(best_archi))
            self.archis.append(best_archi)
            self.writer.add_text("ModelSize/Task_{}".format(t),
                                 "model size = {}".format(utils.get_model_size(self.model)))
            # 1.2.4 unfreeze the model that need to train
            utils.freeze_model(self.model)
            self.model.modify_param(self.model.model_to_train, True)

        # 3 training model for task t
        self.train_t(t, train_data, valid_data, self.batch, self.epochs, device)

    def train_t(self, t, train_data, valid_data, batch_size, epochs, device='cuda'):
        # training model for task t
        # 0 prepare
        print("Training stage of task {}".format(t))
        best_loss = np.inf
        best_model = utils.get_model(self.model)

        lr = self.lr
        self.optimizer = self._get_optimizer(lr)
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, patience=self.lr_patience,
        #                                                        factor=self.lr_factor, threshold=0.001)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, epochs)
        # 2 define the dataloader
        train_loader = torch.utils.data.DataLoader(
            train_data, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
        valid_loader = torch.utils.data.DataLoader(
            valid_data, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
        # 3 training the model
        for e in range(epochs):
            # 3.1 train
            self.train_epoch(t, train_loader, device=device)
            # 3.2 compute training loss
            train_loss, train_abs, train_log10, train_rmse, train_a1, train_a2, train_a3 = self.eval(t, train_loader, mode='train', device=device)
            # 3.3 compute valid loss
            valid_loss, valid_abs, valid_log10, valid_rmse, valid_a1, valid_a2, valid_a3 = self.eval(t, valid_loader, mode='train', device=device)
            # 3.4 logging
            print('| Epoch {:3d} | Train: loss={:.3f}, abs_rel={:.3f}, log10={:.3f}, rmse={:.3f}, delta1={:.3f}, delta2={:.3f}, delta3={:.3f} | \
                                   Valid: loss={:.3f}, abs_rel={:.3f}, log10={:.3f}, rmse={:.3f}, delta1={:.3f}, delta2={:.3f}, delta3={:.3f} |'.format(
                                   e, train_loss, train_abs, train_log10, train_rmse, train_a1, train_a2, train_a3, \
                                      valid_loss, valid_abs, valid_log10, valid_rmse, valid_a1, valid_a2, valid_a3))
            self.writer.add_scalars('Train_Loss/Task: {}'.format(t),
                                    {'train_loss': train_loss, 'valid_loss': valid_loss},
                                    global_step=e)
            self.writer.add_scalars('Train_Abs_Rel/Task: {}'.format(t),
                                    {'train_abs_rel': train_abs, 'valid_abs_rel': valid_abs},
                                    global_step=e)
            self.writer.add_scalars('Train_RMSE/Task: {}'.format(t),
                                    {'train_rmse': train_rmse, 'valid_rmse': valid_rmse},
                                    global_step=e)

            # 3.5 Adapt learning rate
            scheduler.step(valid_loss)
            # 3.6 update the best model
            if valid_loss < best_loss:
                best_loss = valid_loss
                best_model = utils.get_model(self.model)
        # 4 Restore best model
        utils.set_model_(self.model, best_model)
        return

    def train_epoch(self, t, train_loader, device='cuda'):
        self.model.train()
        #import pdb; pdb.set_trace()
        # set tht mode of models which are reused (BN)
        if t > 0:
            
            for i in range(self.model.length['stem_2d0']):
                if i not in self.model.model_to_train['stem_2d0']:
                    self.model.stem2d0[i].eval()
            for i in range(self.model.length['stem_2d1']):
                if i not in self.model.model_to_train['stem_2d1']:
                    self.model.stem2d1[i].eval()
            for i in range(self.model.length['stem_2d2']):
                if i not in self.model.model_to_train['stem_2d2']:
                    self.model.stem2d2[i].eval()
            for i in range(self.model.length['last_3_2d']):
                if i not in self.model.model_to_train['last_3_2d']:
                    self.model.last_3_2d[i].eval()
            for i in range(len(self.model.cells_2d)):
                for k in range(self.model.length['cell_2d' + str(i)]):
                    if k not in self.model.model_to_train['cell_2d' + str(i)]:
                        self.model.cells_2d[i][k].eval()
            
            
            for i in range(self.model.length['stem_3d0']):
                if i not in self.model.model_to_train['stem_3d0']:
                    self.model.stem3d0[i].eval()
            for i in range(self.model.length['stem_3d1']):
                if i not in self.model.model_to_train['stem_3d1']:
                    self.model.stem3d1[i].eval()

            for i in range(len(self.model.cells_3d)):
                for k in range(self.model.length['cell_3d' + str(i)]):
                    if k not in self.model.model_to_train['cell_3d' + str(i)]:
                        self.model.cells_3d[i][k].eval()
            

            for i in range(self.model.length['last_3_3d']):
                if i not in self.model.model_to_train['last_3_3d']:
                    self.model.last_3_3d[i].eval()
            for i in range(self.model.length['last_6_3d']):
                if i not in self.model.model_to_train['last_6_3d']:
                    self.model.last_6_3d[i].eval()
            for i in range(self.model.length['last_12_3d']):
                if i not in self.model.model_to_train['last_12_3d']:
                    self.model.last_12_3d[i].eval()

                        
        # Loop batch
        flag = 1
        for sample in train_loader:
            imgL, imgR, depth_gt = sample['left'].to(self.device), sample['right'].to(self.device), sample['disparity'].to(self.device)
            # forward
            depth_est = self.model.forward(imgL, imgR, t, self.archis[t])
            # todo test
            mask = depth_gt > 0
            loss = self.silog_criterion.forward(depth_est, depth_gt, mask.to(torch.bool))
            # backward
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clipgrad)
            self.optimizer.step()
            flag += 1
            #if flag > 20:
            #    break

    def search_cell(self, t, train_data, valid_data, batch_size, nepochs, device):

        print("Search cell for task {}".format(t))
        self.auto_ml = AutoSearch(device=self.device, writer=self.writer, exp_name=self.exp_name, args=self.args)
        genotype = deepcopy(self.auto_ml.search(t, train_data, valid_data, batch_size, nepochs))
        
        if t == 0:
            self.model = Network(genotype, device).to(device)
            self.archis.append(self.model.arch_init)

            self.writer.add_text("Task_0/genotype",
                                 "genotype = {}".format(genotype),
                                 global_step=0)
            self.writer.add_text("ModelSize/Task_0",
                                 "model size = {}".format(utils.get_model_size(self.model)))
        return genotype

    def search_t(self, t, train_data, valid_data, batch_size, epochs, device):
        # search operations for task t(t>0)
        # 0 prepare
        
        print("Search Stage of task {}".format(t))
        best_loss = np.inf
        best_model = utils.get_model(self.model)
        lr = self.o_lr
        # 1 define optimizers and scheduler
        self.optimizer_o = self._get_optimizer_o(lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer_o, epochs, eta_min=0.001)
        # 2 define the dataloader
        num_train = len(train_data)
        indices = list(range(num_train))
        split = int(np.floor(0.5 * num_train))
        train_loader = torch.utils.data.DataLoader(
            train_data, batch_size=batch_size,
            sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[:split]),
            num_workers=4, pin_memory=True)
        valid_loader = torch.utils.data.DataLoader(
            train_data, batch_size=batch_size,
            sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[split:num_train]),
            num_workers=4, pin_memory=True)

        h_e = [torch.full(pro.size(), 0, dtype=torch.long) for pro in self.model.p]
        h_a = [torch.full(pro.size(), 0.0, dtype=torch.float) for pro in self.model.p]
        
        for k in range(len(h_e)):
            h_e[k][0:-1] = self.o_size

        # 3 search the best model architecture
        for e in range(epochs):
            # 3.0 prepare
            for k, pro in enumerate(self.model.p):
                self.writer.add_histogram('Archi_task_{}/{}'.format(t, k),
                                          pro, global_step=e)
            # 3.1 sample
            selected_ops = [torch.multinomial(pro, 1).item() for pro in self.model.p]
            # print("Selected ops: {}".format(selected_ops))

            model_size = 0.0
            for i, idx in enumerate(selected_ops):
                model_size += (idx == t)
            if model_size < 1:
                model_size = 1
            model_size = 9 / model_size

            self.writer.add_text("Selected ops for task {}".format(t),
                                "{}".format(selected_ops))

            # 3.2 train
            train_loss, train_abs, train_rmse = self.search_epoch(t, train_loader, batch_size, selected_ops, device)
            valid_loss, valid_abs, valid_rmse, valid_a1 = self.search_eval(t, valid_loader, batch_size, selected_ops, device)
            # logging
            print('| Epoch {:3d} | Train: loss={:.3f}, abs_rel={:.3f}%, rmse={:.3f} | Valid: loss={:.3f}, abs_rel={:.3f}%, rmse={:.3f} |'.format(
                e, train_loss, train_abs, train_rmse, valid_loss, valid_abs, valid_rmse))
            self.writer.add_scalars('Search_Loss/Task: {}'.format(t),
                                    {'train_loss': train_loss, 'valid_loss': valid_loss},
                                    global_step=e)
            self.writer.add_scalars('Search_Abs_Rel/Task: {}'.format(t),
                                    {'train_abs_rel': train_abs, 'valid_abs_rel': valid_abs},
                                    global_step=e)
            self.writer.add_scalars('Search_RMSE/Task: {}'.format(t),
                                    {'train_rmse': train_rmse, 'valid_rmse': valid_rmse},
                                    global_step=e)
            # 3.3 update h_e and h_a
            for i, idx in enumerate(selected_ops):
                h_e[i][idx] += 1
                #h_a[i][idx] = 1 - valid_d1  # base
                h_a[i][idx] = np.sqrt(valid_a1) * np.log(model_size+1) / np.exp(1)

                # print("layer {}".format(i).center(50, "*"))
                # print("h_e in layer {}".format(i).center(50, "*"))
                # print(h_e[i])
                # print("h_a in layer {}".format(i).center(50, "*"))
                # print(h_a[i])

            # 3.4 update the probability
            for k in range(len(self.model.p)):
                dh_e_k = torch.reshape(h_e[k], (1, -1)) - torch.reshape(h_e[k], (-1, 1))
                dh_a_k = torch.reshape(h_a[k], (1, -1)) - torch.reshape(h_a[k], (-1, 1))

                # modify
                # vector1 = torch.sum((dh_e_k < 0) * (dh_a_k > 0), dim=1)
                vector1 = torch.sum((dh_e_k < 0) * (dh_a_k > 0), dim=0)
                # vector1[-1] /= self.o_size
                # print("vector1: {}".format(vector1))
                # vector2 = torch.sum((dh_e_k > 0) * (dh_a_k < 0), dim=1)
                vector2 = torch.sum((dh_e_k > 0) * (dh_a_k < 0), dim=0)
                # print("vector2: {}".format(vector2))
                update = (vector1 - vector2).float()
                # update[-1] /= self.o_size
                self.model.p[k] += (self.o_lr_a * update)
                self.model.p[k] = F.softmax(self.model.p[k])

            # 3.5 Adapt learning rate
            scheduler.step()
            # 3.6 update the best model
            if valid_loss < best_loss:
                best_loss = valid_loss
                best_model = utils.get_model(self.model)

        # 4 Restore best model
        utils.set_model_(self.model, best_model)
        return

    def search_epoch(self, t, train_loader, batch_size, selected_ops, device='cuda'):
        self.model.train()
        # set tht mode of models which are reused (BN)
        for i in range(self.model.length['stem_2d0']):
            self.model.stem2d0[i].eval()
        for i in range(self.model.length['stem_2d1']):
            self.model.stem2d1[i].eval()
        for i in range(self.model.length['stem_2d2']):
            self.model.stem2d2[i].eval()
        for i in range(self.model.length['last_3_2d']):
            self.model.last_3_2d[i].eval()
        for i in range(self.model.length['stem_3d0']):
            self.model.stem3d0[i].eval()
        for i in range(self.model.length['stem_3d1']):
            self.model.stem3d1[i].eval()
        

        for i in range(len(self.model.cells_2d)):
            for k in range(self.model.length['cell_2d' + str(i)]):
                self.model.cells_2d[i][k].eval()

        for i in range(len(self.model.cells_3d)):
            for k in range(self.model.length['cell_3d' + str(i)]):
                self.model.cells_3d[i][k].eval()
        
        # 2 Loop batches
        avg_train_scalars = AverageMeterDict()
        abs_rel_sum = 0
        rmse_sum = 0
        count = 0
        for step, sample in enumerate(train_loader):
            imgL, imgR, depth_gt = sample['left'].to(self.device), sample['right'].to(self.device), sample['disparity'].to(self.device)
            length = imgL.size()[0]
            # 2.2.3 Forward current model
            depth_est = self.model.search_forward(imgL, imgR, t, selected_ops)
            # todo test
            mask = depth_gt > 0
            loss = self.silog_criterion.forward(depth_est, depth_gt, mask.to(torch.bool))

            # 2.2.4 Backward
            self.optimizer_o.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clipgrad)
            self.optimizer_o.step()

            with torch.no_grad():
                scalar_outputs = {"loss": loss}
                scalar_outputs = tensor2float(scalar_outputs)
                avg_train_scalars.update(scalar_outputs)

                depth_gt = depth_gt.cpu().numpy()
                depth_est = depth_est.cpu().numpy()
                mask = depth_gt > 0
                depth_errors = compute_errors(depth_gt[mask], depth_est[mask])
                abs_rel_sum += depth_errors[1]
                rmse_sum += depth_errors[3]
                count += 1

            #if step > 20:
            #    break

        avg_train_scalars = avg_train_scalars.mean()
        loss_avg = avg_train_scalars["loss"]

        abs_rel_avg = abs_rel_sum / count
        rmse_avg = rmse_sum / count

        return loss_avg, abs_rel_avg, rmse_avg

    def eval(self, t, test_loader, mode, device):
        avg_eval_scalars = AverageMeterDict()
        abs_rel_sum = 0
        rmse_sum = 0
        log10_sum = 0
        a1_sum = 0
        a2_sum = 0
        a3_sum = 0
        count = 0
        self.model.eval()
        current_task = t
        with torch.no_grad():
            for step, sample in enumerate(test_loader):
                imgL, imgR, depth_gt = sample['left'].to(self.device), sample['right'].to(self.device), sample['disparity'].to(self.device)
                # forward
                depth_est = self.model.forward(imgL, imgR, t, self.archis[t])
                # compute loss
                mask = depth_gt > 0
                loss = self.silog_criterion.forward(depth_est, depth_gt, mask.to(torch.bool))

                scalar_outputs = {"loss": loss}
                scalar_outputs = tensor2float(scalar_outputs)
                avg_eval_scalars.update(scalar_outputs)

                depth_gt = depth_gt.cpu().numpy()
                depth_est = depth_est.cpu().numpy()
                mask = depth_gt > 0
                depth_errors = compute_errors(depth_gt[mask], depth_est[mask])
                abs_rel_sum += depth_errors[1]
                log10_sum += depth_errors[2]
                rmse_sum += depth_errors[3]
                a1_sum += depth_errors[6]
                a2_sum += depth_errors[7]
                a3_sum += depth_errors[8]
                count += 1

                #if step > 20:
                #    break

            avg_eval_scalars = avg_eval_scalars.mean()
            loss_avg = avg_eval_scalars["loss"]

            abs_rel_avg = abs_rel_sum / count
            log10_avg = log10_sum / count
            rmse_avg = rmse_sum / count
            a1_avg = a1_sum / count
            a2_avg = a2_sum / count
            a3_avg = a3_sum / count


        return loss_avg, abs_rel_avg, log10_avg, rmse_avg, a1_avg, a2_avg, a3_avg

    def search_eval(self, t, test_loader, batch_size, selected_ops, device):
        avg_eval_scalars = AverageMeterDict()
        abs_rel_sum = 0
        rmse_sum = 0
        a1_sum = 0
        count = 0
        self.model.eval()

        with torch.no_grad():
            for step, sample in enumerate(test_loader):
                imgL, imgR, depth_gt = sample['left'].to(self.device), sample['right'].to(self.device), sample['disparity'].to(self.device)
                length = imgL.size()[0]
                # forward
                depth_est = self.model.search_forward(imgL, imgR, t, selected_ops)
                # compute loss
                mask = depth_gt > 0
                loss = self.silog_criterion.forward(depth_est, depth_gt, mask.to(torch.bool))
                #loss = F.smooth_l1_loss(disp_est[mask], disp_gt[mask], size_average=True)
                
                scalar_outputs = {"loss": loss}
                scalar_outputs = tensor2float(scalar_outputs)
                avg_eval_scalars.update(scalar_outputs)

                depth_gt = depth_gt.cpu().numpy()
                depth_est = depth_est.cpu().numpy()
                mask = depth_gt > 0
                depth_errors = compute_errors(depth_gt[mask], depth_est[mask])
                abs_rel_sum += depth_errors[1]
                rmse_sum += depth_errors[3]
                a1_sum += depth_errors[6]
                count += 1

                
                #if step > 20:
                #    break


        avg_eval_scalars = avg_eval_scalars.mean()
        loss_avg = avg_eval_scalars["loss"]

        abs_rel_avg = abs_rel_sum / count
        rmse_avg = rmse_sum / count
        a1_avg = a1_sum / count

        return loss_avg, abs_rel_avg, rmse_avg, a1_avg
