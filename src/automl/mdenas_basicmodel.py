import torch
import torch.nn as nn
import torch.nn.functional as F

from copy import deepcopy
import numpy as np


from automl.genotypes_2d import PRIMITIVES
from automl.genotypes_2d import Genotype
from automl.genotypes_3d import PRIMITIVES_3D
from automl.genotypes_3d import Genotype_3D

from automl.build_model_2d import AutoFeature
from automl.build_model_3d import AutoMatching


class DisparityRegression(nn.Module):
    def __init__(self, maxdisp):
        super(DisparityRegression, self).__init__()
        self.maxdisp = maxdisp

    def forward(self, x):
        assert(x.is_contiguous() == True)
        with torch.cuda.device_of(x):
            disp = torch.reshape(torch.arange(0, self.maxdisp, device=torch.cuda.current_device(), dtype=torch.float32),[1,self.maxdisp,1,1])
            disp = disp.repeat(x.size()[0], 1, x.size()[2], x.size()[3])
            out = torch.sum(x * disp, 1)
        return out


class Disp(nn.Module):
    def __init__(self, maxdisp=192):
        super(Disp, self).__init__()
        self.maxdisp = maxdisp
        self.softmax = nn.Softmin(dim=1)
        self.disparity = DisparityRegression(maxdisp=self.maxdisp)

    def forward(self, x):
        x = F.interpolate(x, [self.maxdisp, x.size()[3]*3, x.size()[4]*3], mode='trilinear', align_corners=False)
        x = torch.squeeze(x, 1)
        x = self.softmax(x)      
        x = self.disparity(x)
        return x



class BasicNetwork(nn.Module):

    #def __init__(self, input_c, C, num_classes, layers, criterion, steps=3, multiplier=4, stem_multiplier=3, device='cuda:0'):
    def __init__(self, steps=3, multiplier=4, stem_multiplier=3, device='cuda:0'):
        super(BasicNetwork, self).__init__()
        self._steps = steps  # the number of intermediate nodes
        self._multiplier = multiplier
        self.device = device
        self.num_ops = len(PRIMITIVES)
        self.num_edges = sum(1 for i in range(self._steps) for n in range(2 + i))


        self.feature = AutoFeature()
        self.matching = AutoMatching()

        self.maxdisp = 192
        self.disp = Disp(self.maxdisp)

        # initialize the probabilities
        self.p = None
        self._initialize_p()

    def new(self):
        model_new = BasicNetwork().to(self.device)
        model_new.p = deepcopy(self.probability())

        return model_new

    def forward(self, left, right, fea_ops, mat_ops):
        

        # stereo matching pipeline
        x = self.feature(left, fea_ops)       
        y = self.feature(right, fea_ops)

        with torch.cuda.device_of(x):
            cost = x.new().resize_(x.size()[0], x.size()[1]*2, int(self.maxdisp/3),  x.size()[2],  x.size()[3]).zero_() 
        for i in range(int(self.maxdisp/3)):
            if i > 0 : 
                cost[:,:x.size()[1], i,:,i:] = x[:,:,:,i:]
                cost[:,x.size()[1]:, i,:,i:] = y[:,:,:,:-i]
            else:
                cost[:,:x.size()[1],i,:,i:] = x
                cost[:,x.size()[1]:,i,:,i:] = y
        
        
        cost = self.matching(cost, mat_ops)     
        disp = self.disp(cost)

        return disp

    def _initialize_p(self):
        k = self.num_edges
        num_ops = self.num_ops
        self.p = {
            "normal": torch.full((k, num_ops), 1/num_ops),
            "reduce": torch.full((k, num_ops), 1/num_ops)
        }

    def probability(self):
        return self.p

    def genotype(self):

        def _parse(alphas, steps):
            gene = []
            start = 0
            n = 2
            for i in range(steps):
                end = start + n
                edges = sorted(range(start, end), key=lambda x: -np.max(alphas[x, 1:]))  # ignore none value
                top2edges = edges[:2]
                for j in top2edges:
                    best_op_index = np.argmax(alphas[j])  # this can include none op
                    gene.append([j, best_op_index])
                start = end
                n += 1
            return np.array(gene)

        normalized_fea = F.softmax(self.p['normal'], dim=-1).numpy()
        normalized_mat = F.softmax(self.p['reduce'], dim=-1).numpy()
        gene_normal = _parse(normalized_fea, self._steps)
        gene_reduce = _parse(normalized_mat, self._steps)
        genotype = Genotype(normal=gene_normal, normal_concat=None, reduce=gene_reduce, reduce_concat=None)

        return genotype

