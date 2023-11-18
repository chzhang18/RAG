import torch
import torch.nn as nn
import torch.nn.functional as F

from copy import deepcopy

from automl.darts_genotypes import PRIMITIVES
from automl.darts_genotypes import Genotype
from automl.darts_operation import *


class MixedOp(nn.Module):
    # mixed operation in every edge
    def __init__(self, C, stride):
        super(MixedOp, self).__init__()
        self._ops = nn.ModuleList()
        for primitive in PRIMITIVES:
            op = OPS[primitive](C, stride, False)
            if 'pool' in primitive:
                op = nn.Sequential(op, nn.BatchNorm2d(C, affine=False))
            self._ops.append(op)

    def forward(self, x, weights):
        return sum(w * op(x) for w, op in zip(weights, self._ops))


class Cell(nn.Module):

    def __init__(self, steps, multiplier, C_prev_prev, C_prev, C, reduction, reduction_prev):
        super(Cell, self).__init__()
        self.reduction = reduction

        if reduction_prev:
            self.preprocess0 = FactorizedReduce(C_prev_prev, C, affine=False)
        else:
            self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0, affine=False)
        self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0, affine=False)
        self._steps = steps
        self._multiplier = multiplier

        self._ops = nn.ModuleList()
        self._bns = nn.ModuleList()
        for i in range(self._steps):
            for j in range(2 + i):
                stride = 2 if reduction and j < 2 else 1
                op = MixedOp(C, stride)
                self._ops.append(op)

    def forward(self, s0, s1, weights):
        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)

        states = [s0, s1]
        offset = 0
        for i in range(self._steps):
            # each intermediate node has multiple inputs
            s = sum(self._ops[offset + j](h, weights[offset + j]) for j, h in enumerate(states))
            offset += len(states)
            states.append(s)

        return torch.cat(states[-self._multiplier:], dim=1)


class BasicNetwork(nn.Module):

    def __init__(self, input_c, C, num_classes, layers, criterion, steps=4, multiplier=4, stem_multiplier=3, device='cuda:0'):
        super(BasicNetwork, self).__init__()
        self.input_c = input_c
        self._C = C
        self._num_classes = num_classes
        self._layers = layers
        self._criterion = criterion
        self._steps = steps  # the number of intermediate nodes
        self._multiplier = multiplier
        self.device = device

        C_curr = stem_multiplier * C
        self.stem = nn.Sequential(
            nn.Conv2d(self.input_c, C_curr, 3, padding=1, bias=False),
            nn.BatchNorm2d(C_curr)
        )

        C_prev_prev, C_prev, C_curr = C_curr, C_curr, C
        self.cells = nn.ModuleList()
        reduction_prev = False
        for i in range(layers):
            if i in [layers // 3, 2 * layers // 3]:
                C_curr *= 2
                reduction = True
            else:
                reduction = False
            cell = Cell(steps, multiplier, C_prev_prev, C_prev, C_curr, reduction, reduction_prev)
            reduction_prev = reduction
            self.cells += [cell]
            C_prev_prev, C_prev = C_prev, multiplier * C_curr

        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(C_prev, num_classes)

        self._initialize_alphas()

    def new(self):
        model_new = BasicNetwork(self.input_c, self._C, self._num_classes, self._layers, self._criterion).to(self.device)
        model_new._arch_parameters = deepcopy(self.arch_parameters())

        return model_new

    def forward(self, x):
        s0 = s1 = self.stem(x)
        for i, cell in enumerate(self.cells):
            if cell.reduction:
                weights = F.softmax(self._arch_parameters["alphas_reduce"], dim=-1)
            else:
                weights = F.softmax(self._arch_parameters["alphas_normal"], dim=-1)
            s0, s1 = s1, cell(s0, s1, weights)
        out = self.global_pooling(s1)
        logits = self.classifier(out.view(out.size(0), -1))
        return logits

    def loss(self, x, y):
        logits = self(x)
        return self._criterion(logits, y)

    def _initialize_alphas(self):
        k = sum(1 for i in range(self._steps) for n in range(2 + i))
        num_ops = len(PRIMITIVES)

        self._arch_parameters = nn.ParameterDict({
            "alphas_normal": nn.Parameter(1e-3 * torch.randn((k, num_ops), device=self.device, requires_grad=True)),
            "alphas_reduce": nn.Parameter(1e-3 * torch.randn((k, num_ops), device=self.device, requires_grad=True))
        })

    def arch_parameters(self):
        return self._arch_parameters

    def genotype(self):

        def _parse(weights):
            gene = []
            n = 2
            start = 0
            for i in range(self._steps):  # for each intermediate node
                # get the weights of the input edges of this intermediate node
                end = start + n
                w = weights[start:end].copy()
                # get the sorted edges and select the top-2
                edges = sorted(range(i + 2),
                               key=lambda x: -max(w[x][k] for k in range(len(w[x])) if k != PRIMITIVES.index('none')))[
                        :2]
                for j in edges:  # select the best operation for each edge
                    k_best = None
                    for k in range(len(w[j])):
                        if k != PRIMITIVES.index('none'):
                            if k_best is None or w[j][k] > w[j][k_best]:
                                k_best = k
                    # each intermediate node has two edges, the k_best is the operation of the edge and j is the input
                    # node of the edge
                    gene.append((PRIMITIVES[k_best], j))
                start = end
                n += 1
            return gene

        gene_normal = _parse(F.softmax(self._arch_parameters["alphas_normal"], dim=-1).cpu().detach().numpy())
        gene_reduce = _parse(F.softmax(self._arch_parameters["alphas_reduce"], dim=-1).cpu().detach().numpy())

        concat = range(2 + self._steps - self._multiplier, self._steps + 2)
        genotype = Genotype(
            normal=gene_normal, normal_concat=concat,
            reduce=gene_reduce, reduce_concat=concat
        )
        return genotype

