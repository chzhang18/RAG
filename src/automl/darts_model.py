import numpy as np
import torch
import torch.nn as nn

import utils

from copy import deepcopy

from automl.darts_operation import *



class Cell(nn.Module):

    def __init__(self, genotype, C_prev_prev, C_prev, C, reduction, reduction_prev):
        super(Cell, self).__init__()
        print(C_prev_prev, C_prev, C)

        if reduction_prev:
            self.preprocess0 = FactorizedReduce(C_prev_prev, C)
        else:
            self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0)
        self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0)

        if reduction:
            op_names, indices = zip(*genotype.reduce)
            concat = genotype.reduce_concat
        else:
            op_names, indices = zip(*genotype.normal)
            concat = genotype.normal_concat
        self._compile(C, op_names, indices, concat, reduction)

    def _compile(self, C, op_names, indices, concat, reduction):
        assert len(op_names) == len(indices)
        self._steps = len(op_names) // 2
        self._concat = concat
        self.multiplier = len(concat)

        self._ops = nn.ModuleList()
        for name, index in zip(op_names, indices):
            stride = 2 if reduction and index < 2 else 1
            op = OPS[name](C, stride, True)
            self._ops += [op]
        self._indices = indices

    def forward(self, s0, s1):
        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)

        states = [s0, s1]
        # todo: test
        # print("num_cell: ", self._steps)
        # print("len_indices", len(self._indices))
        for i in range(self._steps):
            # print("idx: ", self._indices[2 * i])
            h1 = states[self._indices[2 * i]]
            # print("idx: ", self._indices[2 * i + 1])
            h2 = states[self._indices[2 * i + 1]]
            op1 = self._ops[2 * i]
            op2 = self._ops[2 * i + 1]
            h1 = op1(h1)
            h2 = op2(h2)
            s = h1 + h2
            states += [s]
        return torch.cat([states[i] for i in self._concat], dim=1)


class Network(nn.Module):

    def __init__(self, input_size, task_classes, cell_nums, init_channel, genotype, device):
        super(Network, self).__init__()
        self.device = device
        self.cell_nums = cell_nums
        self.input_size = input_size
        self.C = init_channel
        self.task_classes = task_classes
        self.genotype = []  # the genotypes which has been used in every model layer
        self.new_genotype = []  # the new genotypes in every model layer in search stage
        self.real_genotype = []  # the genotypes of each cell in every model layer

        self.stem_multiplier = 3
        self.length = {'stem': 1}
        self.arch_init = {'stem': [0], 'fc': [0]}
        self.mu_s = 2  # the number of mutate select

        C_curr = self.stem_multiplier * self.C
        self.stem = nn.ModuleList([nn.Sequential(
            nn.Conv2d(3, C_curr, 3, padding=1, bias=False),
            nn.BatchNorm2d(C_curr)
        )])

        C_prev_prev, C_prev, C_curr = C_curr, C_curr, self.C
        self.cells = nn.ModuleList([])
        reduction_prev = False
        for i in range(self.cell_nums):
            if i in [self.cell_nums // 3, 2 * self.cell_nums // 3]:
                C_curr *= 2
                reduction = True
            else:
                reduction = False
            cell = Cell(genotype, C_prev_prev, C_prev, C_curr, reduction, reduction_prev)
            reduction_prev = reduction
            self.cells.append(nn.ModuleList([cell]))
            C_prev_prev, C_prev = C_prev, cell.multiplier * C_curr
            self.arch_init['cell' + str(i)] = [0]
            self.length['cell'+str(i)] = 1
            # self.genotype[i] includes all kinds of genotypes that been used in the i cell layer
            self.genotype.append([genotype])
            self.new_genotype.append([])
            self.real_genotype.append([genotype])

        self.global_pooling = nn.AdaptiveAvgPool2d(1)

        self.classifier = nn.ModuleList([])
        for t, c in self.task_classes:
            self.classifier.append(nn.Linear(C_prev, c))

        # parameter for architecture search
        self.a = torch.nn.ParameterDict({})
        # maps
        self.map_length = []
        # The new models, model to train
        self.new_models = None
        self.model_to_train = None

    def forward(self, x, t, task_arch=None, path=None):
        arch_stem = None
        if task_arch is not None:
            arch_stem = task_arch['stem'][0]
        elif path is not None:
            arch_stem = path[0]
        s0 = s1 = self.stem[arch_stem](x)

        for i, cell in enumerate(self.cells):
            arch_cell = None
            if task_arch is not None:
                arch_cell = task_arch['cell'+str(i)][0]
            elif path is not None:
                arch_cell = path[i+1]
            s0, s1 = s1, cell[arch_cell](s0, s1)

        out = self.global_pooling(s1)

        logits = []
        for t, c in self.task_classes:
            logits.append(self.classifier[t](out.view(out.size(0), -1)))

        return logits

    def expand(self, t, device='cuda'):
        # expand the network to a super model
        # 1 expand stem
        # 1.1 reuse: reuse parameters and architecture
        # 1.2 update: reuse architecture but update parameter
        C_curr = self.stem_multiplier * self.C
        self.stem.append(nn.Sequential(nn.Conv2d(3, C_curr, 3, padding=1, bias=False),
                                       nn.BatchNorm2d(C_curr)
                                       ).to(device))
        # 1.3 generate action parameter
        num_l = self.length['stem'] + 1
        self.a['stem'] = nn.Parameter(torch.rand(num_l).to(device))
        self.map_length.append(num_l)

        # 2 expand cells
        C_prev_prev, C_prev, C_curr = C_curr, C_curr, self.C
        reduction_prev = False
        for i in range(self.cell_nums):
            if i in [self.cell_nums // 3, 2 * self.cell_nums // 3]:
                C_curr *= 2
                reduction = True
            else:
                reduction = False
            # 2.1 reuse: reuse parameters and architecture
            # 2.2 update and search
            multiplier = None
            self.new_genotype[i] = []  # clear the new genotypes of layer i
            # 2.2.1 update: reuse architecture but update parameter
            for genotype in self.genotype[i]:
                cell = Cell(genotype, C_prev_prev, C_prev, C_curr, reduction, reduction_prev).to(device)
                multiplier = cell.multiplier
                self.cells[i].append(cell)
            # 2.2.2 search: search new architecture for cell
            for genotype in self.genotype[i]:
                new_genotypes = self.mutate_select(genotype)
                for new_genotype in new_genotypes:
                    cell = Cell(new_genotype, C_prev_prev, C_prev, C_curr, reduction, reduction_prev).to(device)
                    self.cells[i].append(cell)
                    self.new_genotype[i].append(new_genotype)

            reduction_prev = reduction
            C_prev_prev, C_prev = C_prev, multiplier * C_curr
            # 2.3 generate action parameter
            # for each cell: reuse + update + mutate select
            num_l = self.length['cell'+str(i)] + (1+self.mu_s) * len(self.genotype[i])
            self.a['cell'+str(i)] = nn.Parameter(torch.rand(num_l).to(device))
            self.map_length.append(num_l)

        # 3 get the new modules
        self.get_new_model(t=t)

    def mutate_select(self, genotype):
        num = self.mu_s
        new_genotypes = []
        # (node2, [0,1]), (node3, [2,3]), (node4, [4,5]), (node5, [6,7]),
        mutation_node = np.random.randint(0, 8, 2*num)
        for i in range(num):
            new_genotype = deepcopy(genotype)
            prev_n = np.random.randint(0, mutation_node[i] // 2 + 2)
            op_n = list(OPS.keys())[np.random.randint(4, 8)]
            new_genotype.normal[mutation_node[i]] = (op_n, prev_n)

            prev_r = np.random.randint(0, mutation_node[i+num] // 2 + 2)
            op_r = list(OPS.keys())[np.random.randint(4, 8)]
            new_genotype.reduce[mutation_node[i+num]] = (op_r, prev_r)

            new_genotypes.append(new_genotype)

        return new_genotypes

    def mutate_evolve(self):
        pass

    def get_new_model(self, t):
        # get new model (update and search)
        new_models = {'stem': [], 'fc': []}
        # 1 stem
        c = self.length['stem']
        new_models['stem'].append(c)
        # 2 cells
        for i in range(self.cell_nums):
            c_1 = self.length['cell' + str(i)]
            c_2 = len(self.genotype[i])
            num_l = c_1 + c_2 * (1 + self.mu_s)
            new_models['cell' + str(i)] = []
            for k in range(c_1, num_l):
                new_models['cell' + str(i)].append(k)
        # # 3 pool
        # new_models['pool'].append(t)
        # 4 classifier
        new_models['fc'].append(t)
        self.new_models = new_models

    def get_archi_param(self):
        return self.a

    def get_param(self, models):
        params = []
        if 'stem' in models.keys():
            for idx in models['stem']:
                params.append({'params': self.stem[idx].parameters()})

        for i in range(self.cell_nums):
            if 'cell' + str(i) in models.keys():
                for idx in models['cell' + str(i)]:
                    params.append({'params': self.cells[i][idx].parameters()})

        # if 'pool' in models.keys():
        #     for idx in models['pool']:
        #         params.append({'params': self.global_pooling[idx].parameters()})

        if 'fc' in models.keys():
            for idx in models['fc']:
                params.append({'params': self.classifier[idx].parameters()})

        return params

    def modify_param(self, models, requires_grad=True):
        if 'stem' in models.keys():
            for idx in models['stem']:
                # print("Set stem {} as {}".format(idx, requires_grad))
                utils.modify_model(self.stem[idx], requires_grad)

        for i in range(self.cell_nums):
            if 'cell' + str(i) in models.keys():
                for idx in models['cell' + str(i)]:
                    utils.modify_model(self.cells[i][idx], requires_grad)
        # if 'pool' in models.keys():
        #     for idx in models['pool']:
        #         utils.modify_model(self.global_pooling[idx], requires_grad)

        if 'fc' in models.keys():
            for idx in models['fc']:
                # print("Set fc {} as {}".format(idx, requires_grad))
                utils.modify_model(self.classifier[idx], requires_grad)

    def modify_archi_param(self, requires_grad=True):
        params = self.get_archi_param()
        if requires_grad:
            utils.unfreeze_parameter(params)
        else:
            utils.freeze_parameter(params)

    def unfreeze_path(self, path, t):
        if path[0] >= self.length['stem']:
            utils.unfreeze_model(self.stem[path[0]])
        for i in range(len(self.cells)):
            if path[i+1] >= self.length['cell' + str(i)]:
                utils.unfreeze_model(self.cells[i][path[i+1]])
        utils.unfreeze_model(self.classifier[t])

    def regular_loss(self):
        loss = 0.0
        # loss of stem
        c = self.length['stem']
        g_stem = torch.exp(self.a['stem']) / torch.sum(torch.exp(self.a['stem']))
        loss += g_stem[c] * utils.model_size(self.stem[c])
        # loss of cells layer
        for i in range(self.cell_nums):
            # loss of cells layer i
            c_1 = self.length['cell' + str(i)]
            c_2 = len(self.genotype[i])
            g_cell = torch.exp(self.a['cell'+str(i)]) / torch.sum(torch.exp(self.a['cell'+str(i)]))
            num_l = c_1 + (1 + self.mu_s) * c_2
            for k in range(c_1, num_l):
                loss += g_cell[k] * utils.model_size(self.cells[i][k])

        return loss

    def search_forward(self, x, t):
        # 1 stem
        g_stem = torch.exp(self.a['stem']) / torch.sum(torch.exp(self.a['stem']))
        # 1.1 stem: update
        out_0 = out_1 = g_stem[-1] * self.stem[-1](x)
        # 1.2 stem: reuse
        c = self.length['stem']
        for i in range(c):
            out_0 += g_stem[i] * self.stem[i](x)
            out_1 += g_stem[i] * self.stem[i](x)

        # 2 cells layer
        for i, cell in enumerate(self.cells):
            g_cell = torch.exp(self.a['cell'+str(i)]) / torch.sum(torch.exp(self.a['cell'+str(i)]))
            c_1 = self.length['cell' + str(i)]
            c_2 = len(self.genotype[i])
            num_l = c_1 + (1 + self.mu_s) * c_2

            x_0, x_1 = out_0, out_1
            out_0 = x_1

            out_1 = g_cell[0] * cell[0](x_0, x_1)
            for k in range(1, num_l):
                out_1 += g_cell[k] * cell[k](x_0, x_1)

        out = self.global_pooling(out_1)
        logits = []
        for t, c in self.task_classes:
            logits.append(self.classifier[t](out.view(out.size(0), -1)))

        return logits

    def select(self, t, path=None):
        # 1 define the container of new models to train and the best submodel
        model_to_train = {'stem': [], 'fc': []}
        best_archi = {'stem': [], 'fc': []}
        best_op = {}  # reuse: 0, update: 1, mutate 2
        # 2 stem
        # 2.1 select the best architecture for stem
        if path is None:
            v, arg_v = torch.max(self.a['stem'].data, dim=0)
            idx = deepcopy(arg_v.item())
        else:
            idx = deepcopy(path[0])
        c = self.length['stem']
        if idx < c:  # reuse
            best_archi['stem'].append(idx)
            best_op['stem'] = 0
        elif idx == c:  # update
            best_archi['stem'].append(c)
            model_to_train['stem'].append(c)
            best_op['stem'] = 1
        # 2.2 delete for stem
        if idx != c:
            del self.stem[c]
        # 2.3 update the length
        self.length['stem'] = len(self.stem)

        # 3 cells layer
        for i in range(self.cell_nums):
            # todo: test
            # print("Select the cell: " + str(i))
            name = 'cell' + str(i)
            g_name = 'genotype' + str(i)
            model_to_train[name] = []
            best_archi[name] = []
            if path is None:
                v, arg_v = torch.max(self.a[name].data, dim=0)
                idx = deepcopy(arg_v.item())
            else:
                idx = deepcopy(path[i+1])
            c_1 = self.length[name]
            c_2 = len(self.genotype[i])
            num_l = c_1 + (1 + self.mu_s) * c_2
            # 3.1 select the best architecture for cell
            if idx < c_1:  # reuse the genotype of the idx cell in layer i
                best_archi[name].append(idx)
                best_op[name] = 0
                best_op[g_name] = deepcopy(self.real_genotype[i][idx])
                self.real_genotype[i].append(best_op[g_name])
            elif idx < c_1 + c_2:  # update the idx-c_1 genotype in the used genotype types set
                best_archi[name].append(c_1)
                model_to_train[name].append(c_1)
                best_op[name] = 1
                best_op[g_name] = deepcopy(self.genotype[i][idx-c_1])
                self.real_genotype[i].append(best_op[g_name])
            elif idx < num_l:  # mutate
                best_archi[name].append(c_1)
                model_to_train[name].append(c_1)
                self.genotype[i].append(deepcopy(self.new_genotype[i][idx-c_1-c_2]))
                best_op[name] = 2
                best_op[g_name] = deepcopy(self.new_genotype[i][idx-c_1-c_2])
                self.real_genotype[i].append(best_op[g_name])
            # 3.2 delete for cell
            cell_s = deepcopy(self.cells[i][idx])
            for k in range(c_1, num_l):
                del self.cells[i][-1]
            if idx >= c_1:
                self.cells[i].append(cell_s)

            # 3.3 update the length
            self.length[name] = len(self.cells[i])

        # 4 the classifier and pool layer
        model_to_train['fc'].append(t)
        best_archi['fc'].append(t)
        # model_to_train['pool'].append(t)
        # best_archi['pool'].append(t)

        # 5 update the model to train
        self.model_to_train = model_to_train

        return best_archi, best_op
