import numpy as np
import torch
import torch.nn as nn

import utils

from copy import deepcopy

from automl.darts_operation import *

from automl.operations_2d import *
from automl.operations_3d import *
from automl.genotypes_2d import PRIMITIVES
from automl.genotypes_3d import PRIMITIVES_3D



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


class Cell_2d(nn.Module):
    def __init__(self, steps, block_multiplier, prev_prev_fmultiplier,
                 prev_filter_multiplier, genotype,
                 filter_multiplier, downup_sample):
        super(Cell_2d, self).__init__()
        self.genotype = genotype
        self.C_in = block_multiplier * filter_multiplier
        self.C_out = filter_multiplier
        self.C_prev = int(block_multiplier * prev_filter_multiplier)
        self.C_prev_prev = int(block_multiplier * prev_prev_fmultiplier)
        self.downup_sample = downup_sample
        self.pre_preprocess = ConvBR_2d(self.C_prev_prev, self.C_out, 1, 1, 0)
        self.preprocess = ConvBR_2d(self.C_prev, self.C_out, 1, 1, 0)
        self._steps = steps
        self.block_multiplier = block_multiplier
        self._ops = nn.ModuleList()
        if downup_sample == -1:
            self.scale = 0.5
        elif downup_sample == 1:
            self.scale = 2
        
        for x in self.genotype.normal: # self.cell_arch: tensor([[0,1], [1,0], [3,0], [2,0], [8,0], [6,0]])
            primitive = PRIMITIVES[x[1]] # 'conv_3×3', 'skip_connect'
            op = OPS_2d[primitive](self.C_out, stride=1)
            self._ops.append(op)
            

    def scale_dimension(self, dim, scale):
        return (int((float(dim) - 1.0) * scale + 1.0) if dim % 2 == 1 else int((float(dim) * scale)))

    def forward(self, prev_prev_input, prev_input):
        s0 = prev_prev_input
        s1 = prev_input
        if self.downup_sample != 0:
            feature_size_h = self.scale_dimension(s1.shape[2], self.scale)
            feature_size_w = self.scale_dimension(s1.shape[3], self.scale)
            s1 = F.interpolate(s1, [feature_size_h, feature_size_w], mode='bilinear', align_corners=True)
        if (s0.shape[2] != s1.shape[2]) or (s0.shape[3] != s1.shape[3]):
            s0 = F.interpolate(s0, (s1.shape[2], s1.shape[3]),
                                            mode='bilinear', align_corners=True)

        s0 = self.pre_preprocess(s0) if (s0.shape[1] != self.C_out) else s0
        s1 = self.preprocess(s1)

        states = [s0, s1]
        offset = 0
        ops_index = 0
        for i in range(self._steps):
            new_states = []
            for j, h in enumerate(states):
                branch_index = offset + j
                if branch_index in self.genotype.normal[:, 0]:
                    if prev_prev_input is None and j == 0:
                        ops_index += 1
                        continue
                    new_state = self._ops[ops_index](h)
                    new_states.append(new_state)
                    ops_index += 1

            s = sum(new_states)
            offset += len(states)
            states.append(s)

        concat_feature = torch.cat(states[-self.block_multiplier:], dim=1) 
        return prev_input, concat_feature


class Cell_3d(nn.Module):
    def __init__(self, steps, block_multiplier, prev_prev_fmultiplier,
                 prev_filter_multiplier, genotype,
                 filter_multiplier, downup_sample):
        super(Cell_3d, self).__init__()
        self.genotype = genotype
        self.C_in = block_multiplier * filter_multiplier
        self.C_out = filter_multiplier
        self.C_prev = int(block_multiplier * prev_filter_multiplier)
        self.C_prev_prev = int(block_multiplier * prev_prev_fmultiplier)
        self.downup_sample = downup_sample
        self.pre_preprocess = ConvBR_3d(self.C_prev_prev, self.C_out, 1, 1, 0)
        self.preprocess = ConvBR_3d(self.C_prev, self.C_out, 1, 1, 0)
        self._steps = steps
        self.block_multiplier = block_multiplier
        self._ops = nn.ModuleList()
        if downup_sample == -1:
            self.scale = 0.5
        elif downup_sample == 1:
            self.scale = 2
        for x in self.genotype.reduce:
            primitive = PRIMITIVES_3D[x[1]]
            op = OPS_3d[primitive](self.C_out, stride=1)
            self._ops.append(op)


    def scale_dimension(self, dim, scale):
        return (int((float(dim) - 1.0) * scale + 1.0) if dim % 2 == 1 else int((float(dim) * scale)))

    def forward(self, prev_prev_input, prev_input):
        s0 = prev_prev_input
        s1 = prev_input
        if self.downup_sample != 0:
            feature_size_d = self.scale_dimension(s1.shape[2], self.scale)
            feature_size_h = self.scale_dimension(s1.shape[3], self.scale)
            feature_size_w = self.scale_dimension(s1.shape[4], self.scale)
            s1 = F.interpolate(s1, [feature_size_d, feature_size_h, feature_size_w], mode='trilinear', align_corners=True)
        if (s0.shape[2] != s1.shape[2]) or (s0.shape[3] != s1.shape[3]) or (s0.shape[4] != s1.shape[4]):
            s0 = F.interpolate(s0, (s1.shape[2], s1.shape[3], s1.shape[4]),
                                            mode='trilinear', align_corners=True)
        s0 = self.pre_preprocess(s0) if (s0.shape[1] != self.C_out) else s0
        s1 = self.preprocess(s1)

        states = [s0, s1]
        offset = 0
        ops_index = 0
        for i in range(self._steps):
            new_states = []
            for j, h in enumerate(states):
                branch_index = offset + j
                if branch_index in self.genotype.reduce[:, 0]:
                    if prev_prev_input is None and j == 0:
                        ops_index += 1
                        continue
                    new_state = self._ops[ops_index](h)
                    new_states.append(new_state)
                    ops_index += 1

            s = sum(new_states)
            offset += len(states)
            states.append(s)

        concat_feature = torch.cat(states[-self.block_multiplier:], dim=1) 
        return prev_input, concat_feature



class Network(nn.Module):
    def __init__(self, genotype, device):
        super(Network, self).__init__()
        self.device = device
        
        # feature net
        self.cells_2d = nn.ModuleList()
        self._step = 3
        self._num_layers_2d = 4
        self._block_multiplier = 3 
        self._filter_multiplier = 4
        self._K_multiplier = 2

        initial_fm = self._filter_multiplier * self._block_multiplier
        half_initial_fm = initial_fm // 2

        self.length = {'stem_2d0': 1, 'stem_2d1': 1, 'stem_2d2': 1, 'last_3_2d': 1, 'stem_3d0': 1, 'stem_3d1': 1, 'last_3_3d': 1, 'last_6_3d': 1, 'last_12_3d': 1}
        self.arch_init = {'stem_2d0':[0], 'stem_2d1':[0], 'stem_2d2':[0], 'last_3_2d':[0], 'stem_3d0':[0], 'stem_3d1':[0], 'last_3_3d':[0], 'last_6_3d':[0], 'last_12_3d':[0]}

        self.stem2d0 = nn.ModuleList([ConvBR_2d(3, half_initial_fm, 3, stride=1, padding=1)])
        self.stem2d1 = nn.ModuleList([ConvBR_2d(half_initial_fm, initial_fm, 3, stride=3, padding=1)])
        self.stem2d2 = nn.ModuleList([ConvBR_2d(initial_fm, initial_fm, 3, stride=1, padding=1)])

        
        #import pdb; pdb.set_trace()
        # feature net structure [1, 0, 1, 0]
        for i in range(self._num_layers_2d):
            if i == 0:
                cell = Cell_2d(self._step, self._block_multiplier, initial_fm / self._block_multiplier,
                             initial_fm / self._block_multiplier, genotype, 8, -1)
            elif i == 1:
                cell = Cell_2d(self._step, self._block_multiplier, initial_fm / self._block_multiplier,
                             8, genotype, 4, 1)
            elif i == 2:
                cell = Cell_2d(self._step, self._block_multiplier, 8,
                             4, genotype, 8, -1)
            elif i == 3:
                cell = Cell_2d(self._step, self._block_multiplier, 4,
                             8, genotype, 4, 1)


            self.cells_2d.append(nn.ModuleList([cell]))

            self.arch_init['cell_2d' + str(i)] = [0]
            self.length['cell_2d'+str(i)] = 1

        self.last_3_2d  = nn.ModuleList([ConvBR_2d(initial_fm , initial_fm, 1, 1, 0, bn=False, relu=False)])
        

        # matching net
        self._num_layers_3d = 8
        self.cells_3d = nn.ModuleList()

        self.stem3d0 = nn.ModuleList([ConvBR_3d(initial_fm*2, initial_fm, 3, stride=1, padding=1)])
        self.stem3d1 = nn.ModuleList([ConvBR_3d(initial_fm, initial_fm, 3, stride=1, padding=1)])
        
        # matching net structure [0, 0, 0, 1, 2, 1, 2, 2]
        for i in range(self._num_layers_3d):
            if i == 0:
                cell = Cell_3d(self._step, self._block_multiplier, initial_fm / self._block_multiplier,
                             initial_fm / self._block_multiplier, genotype, 4 , 0)
            elif i == 1:
                cell = Cell_3d(self._step, self._block_multiplier, initial_fm / self._block_multiplier, 4, genotype, 4 , 0)

            elif i == 2:
                cell = Cell_3d(self._step, self._block_multiplier, 4, 4, genotype, 4, 0)

            elif i == 3:
                cell = Cell_3d(self._step, self._block_multiplier, 4, 4, genotype, 8, -1)

            elif i == 4:
                cell = Cell_3d(self._step, self._block_multiplier, 4, 8, genotype, 16, -1)

            elif i == 5:
                cell = Cell_3d(self._step, self._block_multiplier, 8, 16, genotype, 8, 1)

            elif i == 6:
                cell = Cell_3d(self._step, self._block_multiplier, 16, 8, genotype, 16, -1)

            elif i == 7:
                cell = Cell_3d(self._step, self._block_multiplier, 8, 16, genotype, 16, 0)

            self.cells_3d.append(nn.ModuleList([cell]))

            self.arch_init['cell_3d' + str(i)] = [0]
            self.length['cell_3d'+str(i)] = 1


        self.last_3_3d  = nn.ModuleList([ConvBR_3d(initial_fm, 1, 3, 1, 1,  bn=False, relu=False)]) 
        self.last_6_3d  = nn.ModuleList([ConvBR_3d(initial_fm*2 , initial_fm,    1, 1, 0)])
        self.last_12_3d = nn.ModuleList([ConvBR_3d(initial_fm*4 , initial_fm*2,  1, 1, 0)])


        self.maxdisp = 192
        self.disp = Disp(self.maxdisp)



        # parameter for architecture search
        self.p = None
        # The new models, model to train
        self.new_models = None
        self.model_to_train = None

    def feature(self, x, task_arch, path):
        
        arch_stem_2d0 = None
        arch_stem_2d1 = None
        arch_stem_2d2 = None
        arch_last_3_2d = None
        if task_arch is not None:
            arch_stem_2d0 = task_arch['stem_2d0'][0]
            arch_stem_2d1 = task_arch['stem_2d1'][0]
            arch_stem_2d2 = task_arch['stem_2d2'][0]
            arch_last_3_2d = task_arch['last_3_2d'][0]
        elif path is not None:
            arch_stem = path[0]

        stem0 = self.stem2d0[arch_stem_2d0](x)
        stem1 = self.stem2d1[arch_stem_2d1](stem0)
        stem2 = self.stem2d2[arch_stem_2d2](stem1)
        out = (stem1, stem2)

        

        for i, cell in enumerate(self.cells_2d):
            arch_cell = None
            if task_arch is not None:
                arch_cell = task_arch['cell_2d'+str(i)][0]
            elif path is not None:
                arch_cell = path[i+1]
            out = cell[arch_cell](out[0], out[1])

        last_output = out[-1]

        h, w = stem2.size()[2], stem2.size()[3]
        if last_output.size()[2] == h:
            fea = self.last_3_2d[arch_last_3_2d](last_output)
        else:
            print('this is a bug')
            print(x.shape, last_output.size()[2], stem2.size()[2])     

        return fea

    def matching(self, x, task_arch, path):
        arch_stem_3d0 = None
        arch_stem_3d1 = None
        arch_last_3_3d = None
        arch_last_6_3d = None
        arch_last_12_3d = None
        if task_arch is not None:
            arch_stem_3d0 = task_arch['stem_3d0'][0]
            arch_stem_3d1 = task_arch['stem_3d1'][0]
            arch_last_3_3d = task_arch['last_3_3d'][0]
            arch_last_6_3d = task_arch['last_6_3d'][0]
            arch_last_12_3d = task_arch['last_12_3d'][0]

        elif path is not None:
            arch_stem = path[0]

        stem0 = self.stem3d0[arch_stem_3d0](x)
        stem1 = self.stem3d1[arch_stem_3d1](stem0)
        out = (stem0, stem1)

        for i, cell in enumerate(self.cells_3d):
            arch_cell = None
            if task_arch is not None:
                arch_cell = task_arch['cell_3d'+str(i)][0]
            elif path is not None:
                arch_cell = path[i+1]
            out = cell[arch_cell](out[0], out[1])

        last_output = out[-1]


        d, h, w = x.size()[2], x.size()[3], x.size()[4]
        upsample_6  = nn.Upsample(size=x.size()[2:], mode='trilinear', align_corners=True)
        upsample_12 = nn.Upsample(size=[d//2, h//2, w//2], mode='trilinear', align_corners=True)

        if last_output.size()[3] == h:
            mat = self.last_3_3d[arch_last_3_3d](last_output)
        elif last_output.size()[3] == h//2:
            mat = self.last_3_3d[arch_last_3_3d](upsample_6(self.last_6_3d[arch_last_6_3d](last_output)))
        elif last_output.size()[3] == h//4:
            mat = self.last_3_3d[arch_last_3_3d](upsample_6(self.last_6_3d[arch_last_6_3d](upsample_12(self.last_12_3d[arch_last_12_3d](last_output)))))  
        return mat  


    def forward(self, left, right, t, task_arch=None, path=None):
        
        
        x = self.feature(left, task_arch, path)
        y = self.feature(right, task_arch, path)

        with torch.cuda.device_of(x):
            cost = x.new().resize_(x.size()[0], x.size()[1]*2, int(self.maxdisp/3),  x.size()[2],  x.size()[3]).zero_() 
        for i in range(int(self.maxdisp/3)):
            if i > 0 : 
                cost[:,:x.size()[1], i,:,i:] = x[:,:,:,i:]
                cost[:,x.size()[1]:, i,:,i:] = y[:,:,:,:-i]
            else:
                cost[:,:x.size()[1],i,:,i:] = x
                cost[:,x.size()[1]:,i,:,i:] = y

        cost = self.matching(cost, task_arch, path)     
        disp = self.disp(cost)   
        return disp
    
    
    ## both FeatureNet and MatchingNet are searched
    def expand(self, t, genotype, device='cuda'):
        # expand the network to a super model
        # 0 clean the probability
        self.p = []
        # 1 expand stem
        # 1.1 reuse: reuse blocks
        # 1.2 new: create a new block
        initial_fm = self._filter_multiplier * self._block_multiplier
        half_initial_fm = initial_fm // 2

        
        # 1.3 generate action parameter
        self.stem2d0.append(ConvBR_2d(3, half_initial_fm, 3, stride=1, padding=1).to(device))
        num_l = self.length['stem_2d0'] + 1
        temp = torch.full((num_l,), 1 / (self._K_multiplier * self.length['stem_2d0'] + 1))
        temp[:num_l-1] *= self._K_multiplier
        self.p.append(temp)

        self.stem2d1.append(ConvBR_2d(half_initial_fm, initial_fm, 3, stride=3, padding=1).to(device))
        num_l = self.length['stem_2d1'] + 1
        temp = torch.full((num_l,), 1 / (self._K_multiplier * self.length['stem_2d1'] + 1))
        temp[:num_l-1] *= self._K_multiplier
        self.p.append(temp)

        self.stem2d2.append(ConvBR_2d(initial_fm, initial_fm, 3, stride=1, padding=1).to(device))
        num_l = self.length['stem_2d2'] + 1
        temp = torch.full((num_l,), 1 / (self._K_multiplier * self.length['stem_2d2'] + 1))
        temp[:num_l-1] *= self._K_multiplier
        self.p.append(temp) # self.p 0,1,2

        # 2 expand cells
        for i in range(self._num_layers_2d):
            # 2.1 reuse: reuse blocks
            # 2.2 new: create new block according to the new cell
            if i == 0:
                cell = Cell_2d(self._step, self._block_multiplier, initial_fm / self._block_multiplier,
                             initial_fm / self._block_multiplier, genotype, 8, -1).to(device)
            elif i == 1:
                cell = Cell_2d(self._step, self._block_multiplier, initial_fm / self._block_multiplier,
                             8, genotype, 4, 1).to(device)
            elif i == 2:
                cell = Cell_2d(self._step, self._block_multiplier, 8,
                             4, genotype, 8, -1).to(device)
            elif i == 3:
                cell = Cell_2d(self._step, self._block_multiplier, 4,
                             8, genotype, 4, 1).to(device)

            self.cells_2d[i].append(cell)
            num_l = self.length['cell_2d'+str(i)] + 1
            temp = torch.full((num_l,), 1 / (self._K_multiplier * self.length['cell_2d'+str(i)] + 1))
            temp[:num_l-1] *= self._K_multiplier
            self.p.append(temp) # self.p 3,4,5,6


        self.last_3_2d.append(ConvBR_2d(initial_fm , initial_fm, 1, 1, 0, bn=False, relu=False).to(device))
        num_l = self.length['last_3_2d'] + 1
        temp = torch.full((num_l,), 1 / (self._K_multiplier * self.length['last_3_2d'] + 1))
        temp[:num_l-1] *= self._K_multiplier
        self.p.append(temp) # self.p 7



        self.stem3d0.append(ConvBR_3d(initial_fm*2, initial_fm, 3, stride=1, padding=1).to(device))
        num_l = self.length['stem_3d0'] + 1
        temp = torch.full((num_l,), 1 / (self._K_multiplier * self.length['stem_3d0'] + 1))
        temp[:num_l-1] *= self._K_multiplier
        self.p.append(temp) # self.p 8

        self.stem3d1.append(ConvBR_3d(initial_fm, initial_fm, 3, stride=1, padding=1).to(device))
        num_l = self.length['stem_3d1'] + 1
        temp = torch.full((num_l,), 1 / (self._K_multiplier * self.length['stem_3d1'] + 1))
        temp[:num_l-1] *= self._K_multiplier
        self.p.append(temp) # self.p 9


        for i in range(self._num_layers_3d):
            if i == 0:
                cell = Cell_3d(self._step, self._block_multiplier, initial_fm / self._block_multiplier,
                             initial_fm / self._block_multiplier, genotype, 4 , 0).to(device)
            elif i == 1:
                cell = Cell_3d(self._step, self._block_multiplier, initial_fm / self._block_multiplier, 4, genotype, 4 , 0).to(device)

            elif i == 2:
                cell = Cell_3d(self._step, self._block_multiplier, 4, 4, genotype, 4, 0).to(device)

            elif i == 3:
                cell = Cell_3d(self._step, self._block_multiplier, 4, 4, genotype, 8, -1).to(device)

            elif i == 4:
                cell = Cell_3d(self._step, self._block_multiplier, 4, 8, genotype, 16, -1).to(device)

            elif i == 5:
                cell = Cell_3d(self._step, self._block_multiplier, 8, 16, genotype, 8, 1).to(device)

            elif i == 6:
                cell = Cell_3d(self._step, self._block_multiplier, 16, 8, genotype, 16, -1).to(device)

            elif i == 7:
                cell = Cell_3d(self._step, self._block_multiplier, 8, 16, genotype, 16, 0).to(device)

            self.cells_3d[i].append(cell)

            # 2.3 generate action parameter
            # for each cell: reuse + create
            num_l = self.length['cell_3d'+str(i)] + 1
            temp = torch.full((num_l,), 1 / (self._K_multiplier * self.length['cell_3d'+str(i)] + 1))
            temp[:num_l-1] *= self._K_multiplier
            self.p.append(temp) # self.p 10,11,12,13,14,15,16,17


        self.last_3_3d.append(ConvBR_3d(initial_fm, 1, 3, 1, 1,  bn=False, relu=False).to(device))
        #num_l = self.length['last_3_3d'] + 1
        #temp = torch.full((num_l,), 1 / (self._K_multiplier * self.length['last_3_3d'] + 1))
        #temp[:num_l-1] *= self._K_multiplier
        #self.p.append(temp) # self.p 18

        self.last_6_3d.append(ConvBR_3d(initial_fm*2 , initial_fm,    1, 1, 0).to(device)) 
        #num_l = self.length['last_6_3d'] + 1
        #temp = torch.full((num_l,), 1 / (self._K_multiplier * self.length['last_6_3d'] + 1))
        #temp[:num_l-1] *= self._K_multiplier
        #self.p.append(temp) # self.p 19

        self.last_12_3d.append(ConvBR_3d(initial_fm*4 , initial_fm*2,  1, 1, 0).to(device))
        #num_l = self.length['last_12_3d'] + 1
        #temp = torch.full((num_l,), 1 / (self._K_multiplier * self.length['last_12_3d'] + 1))
        #temp[:num_l-1] *= self._K_multiplier
        #self.p.append(temp) # self.p 20

        #import pdb; pdb.set_trace()

        # 3 get the new modules
        self.get_new_model(t=t)
    

    def get_new_model(self, t):
        
        # get new model (update and search)
        new_models = {'stem_2d0': [], 'stem_2d1': [], 'stem_2d2': [], 'last_3_2d': [], 'stem_3d0': [], 'stem_3d1': [], 'last_3_3d': [], 'last_6_3d': [], 'last_12_3d': []}
        # 1 stem & last
        #new_models['stem_2d0'].append(t)
        new_models['stem_2d0'] = [self.length['stem_2d0']]
        new_models['stem_2d1'] = [self.length['stem_2d1']]
        new_models['stem_2d2'] = [self.length['stem_2d2']]
        new_models['last_3_2d'] = [self.length['last_3_2d']]
        new_models['stem_3d0'] = [self.length['stem_3d0']]
        new_models['stem_3d1'] = [self.length['stem_3d1']]
        #new_models['last_3_3d'] = [self.length['last_3_3d']]
        #new_models['last_6_3d'] = [self.length['last_6_3d']]
        #new_models['last_12_3d'] = [self.length['last_12_3d']]
        new_models['last_3_3d'].append(t)
        new_models['last_6_3d'].append(t)
        new_models['last_12_3d'].append(t)

        # 2 cells_2d & cells_3d
        for i in range(self._num_layers_2d):
            new_models['cell_2d' + str(i)] = [self.length['cell_2d' + str(i)]]

        for i in range(self._num_layers_3d):
            new_models['cell_3d' + str(i)] = [self.length['cell_3d' + str(i)]]

        self.new_models = new_models

    def get_param(self, models):
        params = []
        if 'stem_2d0' in models.keys():
            for idx in models['stem_2d0']:
                params.append({'params': self.stem2d0[idx].parameters()})
        if 'stem_2d1' in models.keys():
            for idx in models['stem_2d1']:
                params.append({'params': self.stem2d1[idx].parameters()})
        if 'stem_2d2' in models.keys():
            for idx in models['stem_2d2']:
                params.append({'params': self.stem2d2[idx].parameters()})
        if 'last_3_2d' in models.keys():
            for idx in models['last_3_2d']:
                params.append({'params': self.last_3_2d[idx].parameters()})
        if 'stem_3d0' in models.keys():
            for idx in models['stem_3d0']:
                params.append({'params': self.stem3d0[idx].parameters()})
        if 'stem_3d1' in models.keys():
            for idx in models['stem_3d1']:
                params.append({'params': self.stem3d1[idx].parameters()})
        if 'last_3_3d' in models.keys():
            for idx in models['last_3_3d']:
                params.append({'params': self.last_3_3d[idx].parameters()})
        if 'last_6_3d' in models.keys():
            for idx in models['last_6_3d']:
                params.append({'params': self.last_6_3d[idx].parameters()})
        if 'last_12_3d' in models.keys():
            for idx in models['last_12_3d']:
                params.append({'params': self.last_12_3d[idx].parameters()})


        for i in range(self._num_layers_2d):
            if 'cell_2d' + str(i) in models.keys():
                for idx in models['cell_2d' + str(i)]:
                    params.append({'params': self.cells_2d[i][idx].parameters()})

        for i in range(self._num_layers_3d):
            if 'cell_3d' + str(i) in models.keys():
                for idx in models['cell_3d' + str(i)]:
                    params.append({'params': self.cells_3d[i][idx].parameters()})


        return params

    def modify_param(self, models, requires_grad=True):
        if 'stem_2d0' in models.keys():
            for idx in models['stem_2d0']:
                utils.modify_model(self.stem2d0[idx], requires_grad)
        if 'stem_2d1' in models.keys():
            for idx in models['stem_2d1']:
                utils.modify_model(self.stem2d1[idx], requires_grad)
        if 'stem_2d2' in models.keys():
            for idx in models['stem_2d2']:
                utils.modify_model(self.stem2d2[idx], requires_grad)

        for i in range(self._num_layers_2d):
            if 'cell_2d' + str(i) in models.keys():
                for idx in models['cell_2d' + str(i)]:
                    utils.modify_model(self.cells_2d[i][idx], requires_grad)

        if 'last_3_2d' in models.keys():
            for idx in models['last_3_2d']:
                utils.modify_model(self.last_3_2d[idx], requires_grad)


        if 'stem_3d0' in models.keys():
            for idx in models['stem_3d0']:
                utils.modify_model(self.stem3d0[idx], requires_grad)
        if 'stem_3d1' in models.keys():
            for idx in models['stem_3d1']:
                utils.modify_model(self.stem3d1[idx], requires_grad)
        if 'last_3_3d' in models.keys():
            for idx in models['last_3_3d']:
                utils.modify_model(self.last_3_3d[idx], requires_grad)
        if 'last_6_3d' in models.keys():
            for idx in models['last_6_3d']:
                utils.modify_model(self.last_6_3d[idx], requires_grad)
        if 'last_12_3d' in models.keys():
            for idx in models['last_12_3d']:
                utils.modify_model(self.last_12_3d[idx], requires_grad)
        

        for i in range(self._num_layers_3d):
            if 'cell_3d' + str(i) in models.keys():
                for idx in models['cell_3d' + str(i)]:
                    utils.modify_model(self.cells_3d[i][idx], requires_grad)


    def search_feature(self, x, selected_ops):
        
        stem0 = self.stem2d0[selected_ops[0]](x)
        stem1 = self.stem2d1[selected_ops[1]](stem0)
        stem2 = self.stem2d2[selected_ops[2]](stem1)
        out = (stem1, stem2)

        for i, cell in enumerate(self.cells_2d):
            out = cell[selected_ops[i+3]](out[0], out[1])

        last_output = out[-1]

        h, w = stem2.size()[2], stem2.size()[3]
        if last_output.size()[2] == h:
            fea = self.last_3_2d[selected_ops[7]](last_output)
        else:
            print('this is a bug')
            print(x.shape, last_output.size()[2], stem2.size()[2])     

        return fea


    def search_matching(self, x, selected_ops, t):

        stem0 = self.stem3d0[selected_ops[8]](x)
        stem1 = self.stem3d1[selected_ops[9]](stem0)
        out = (stem0, stem1)

        for i, cell in enumerate(self.cells_3d):
            out = cell[selected_ops[i+10]](out[0], out[1])

        last_output = out[-1]


        d, h, w = x.size()[2], x.size()[3], x.size()[4]
        upsample_6  = nn.Upsample(size=x.size()[2:], mode='trilinear', align_corners=True)
        upsample_12 = nn.Upsample(size=[d//2, h//2, w//2], mode='trilinear', align_corners=True)

        if last_output.size()[3] == h:
            mat = self.last_3_3d[t](last_output)
        elif last_output.size()[3] == h//2:
            mat = self.last_3_3d[t](upsample_6(self.last_6_3d[t](last_output)))
        elif last_output.size()[3] == h//4:
            mat = self.last_3_3d[t](upsample_6(self.last_6_3d[t](upsample_12(self.last_12_3d[t](last_output)))))  
        return mat  


    def search_forward(self, left, right, t, selected_ops):


        x = self.search_feature(left, selected_ops)
        y = self.search_feature(right, selected_ops)

        with torch.cuda.device_of(x):
            cost = x.new().resize_(x.size()[0], x.size()[1]*2, int(self.maxdisp/3),  x.size()[2],  x.size()[3]).zero_() 
        for i in range(int(self.maxdisp/3)):
            if i > 0 : 
                cost[:,:x.size()[1], i,:,i:] = x[:,:,:,i:]
                cost[:,x.size()[1]:, i,:,i:] = y[:,:,:,:-i]
            else:
                cost[:,:x.size()[1],i,:,i:] = x
                cost[:,x.size()[1]:,i,:,i:] = y

        cost = self.search_matching(cost, selected_ops, t)     
        disp = self.disp(cost)   
        return disp


    def select(self, t):
        
        model_to_train = {'stem_2d0': [], 'stem_2d1': [], 'stem_2d2': [], 'last_3_2d': [], 'stem_3d0': [], 'stem_3d1': [], 'last_3_3d': [], 'last_6_3d': [], 'last_12_3d': []}
        best_archi = {'stem_2d0': [], 'stem_2d1': [], 'stem_2d2': [], 'last_3_2d': [], 'stem_3d0': [], 'stem_3d1': [], 'last_3_3d': [], 'last_6_3d': [], 'last_12_3d': []}
        # stem & last
        #model_to_train['stem_2d0'].append(t)
        #best_archi['stem_2d0'].append(t)

        v, idx = torch.max(self.p[0], dim=0)
        c = self.length['stem_2d0']
        if idx < c:  # reuse
            best_archi['stem_2d0'].append(idx)
        elif idx == c:  # update
            best_archi['stem_2d0'].append(c)
            model_to_train['stem_2d0'].append(c)
        # 2.2 delete for stem
        if idx != c:
            del self.stem2d0[c]
        # 2.3 update the length
        self.length['stem_2d0'] = len(self.stem2d0)

        v, idx = torch.max(self.p[1], dim=0)
        c = self.length['stem_2d1']
        if idx < c:
            best_archi['stem_2d1'].append(idx)
        elif idx == c:
            best_archi['stem_2d1'].append(c)
            model_to_train['stem_2d1'].append(c)
        if idx != c:
            del self.stem2d1[c]
        self.length['stem_2d1'] = len(self.stem2d1)

        v, idx = torch.max(self.p[2], dim=0)
        c = self.length['stem_2d2']
        if idx < c:
            best_archi['stem_2d2'].append(idx)
        elif idx == c:
            best_archi['stem_2d2'].append(c)
            model_to_train['stem_2d2'].append(c)
        if idx != c:
            del self.stem2d2[c]
        self.length['stem_2d2'] = len(self.stem2d2)


        # cells layer
        for i in range(self._num_layers_2d):
            name = 'cell_2d' + str(i)
            model_to_train[name] = []
            best_archi[name] = []

            v, idx = torch.max(self.p[i+3], dim=0)
            c = self.length[name]
            # 3.1 select the best architecture for cell
            if idx != c:  # reuse blocks
                best_archi[name].append(idx)
            elif idx == c:  # create new block
                best_archi[name].append(c)
                model_to_train[name].append(c)
            # 3.2 delete for cell
            if idx != c:
                del self.cells_2d[i][c]

            # 3.3 update the length
            self.length[name] = len(self.cells_2d[i])


        v, idx = torch.max(self.p[7], dim=0)
        c = self.length['last_3_2d']
        if idx < c:
            best_archi['last_3_2d'].append(idx)
        elif idx == c:
            best_archi['last_3_2d'].append(c)
            model_to_train['last_3_2d'].append(c)
        if idx != c:
            del self.last_3_2d[c]
        self.length['last_3_2d'] = len(self.last_3_2d)

        v, idx = torch.max(self.p[8], dim=0)
        c = self.length['stem_3d0']
        if idx < c:  # reuse
            best_archi['stem_3d0'].append(idx)
        elif idx == c:  # update
            best_archi['stem_3d0'].append(c)
            model_to_train['stem_3d0'].append(c)
        # 2.2 delete for stem
        if idx != c:
            del self.stem3d0[c]
        # 2.3 update the length
        self.length['stem_3d0'] = len(self.stem3d0)

        v, idx = torch.max(self.p[9], dim=0)
        c = self.length['stem_3d1']
        if idx < c:
            best_archi['stem_3d1'].append(idx)
        elif idx == c:
            best_archi['stem_3d1'].append(c)
            model_to_train['stem_3d1'].append(c)
        if idx != c:
            del self.stem3d1[c]
        self.length['stem_3d1'] = len(self.stem3d1)


        for i in range(self._num_layers_3d):
            name = 'cell_3d' + str(i)
            model_to_train[name] = []
            best_archi[name] = []

            v, idx = torch.max(self.p[i+10], dim=0)
            c = self.length[name]
            # 3.1 select the best architecture for cell
            if idx != c:  # reuse blocks
                best_archi[name].append(idx)
            elif idx == c:  # create new block
                best_archi[name].append(c)
                model_to_train[name].append(c)
            # 3.2 delete for cell
            if idx != c:
                del self.cells_3d[i][c]

            # 3.3 update the length
            self.length[name] = len(self.cells_3d[i])

        
        model_to_train['last_3_3d'].append(t)
        best_archi['last_3_3d'].append(t)

        model_to_train['last_6_3d'].append(t)
        best_archi['last_6_3d'].append(t)

        model_to_train['last_12_3d'].append(t)
        best_archi['last_12_3d'].append(t)


        # 5 update the model to train
        self.model_to_train = model_to_train

        return best_archi
