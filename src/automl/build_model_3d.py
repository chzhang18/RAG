import torch
import torch.nn as nn
import numpy as np
from automl.genotypes_3d import PRIMITIVES_3D
from automl.genotypes_3d import Genotype_3D
from automl.operations_3d import *
import torch.nn.functional as F
import numpy as np
import pdb

class MixedOp(nn.Module):

    def __init__(self, C, stride):
        super(MixedOp, self).__init__()
        self._ops = nn.ModuleList()
        for primitive in PRIMITIVES_3D:
            op = OPS_3d[primitive](C, stride)
            if 'pool' in primitive:
                op = nn.Sequential(op, nn.BatchNorm3d(C))
            self._ops.append(op)

    def forward(self, x, selected_op):
        return self._ops[selected_op](x)


class Cell(nn.Module):

    def __init__(self, steps, block_multiplier, prev_prev_fmultiplier,
                 prev_fmultiplier_down, prev_fmultiplier_same, prev_fmultiplier_up,
                 filter_multiplier):

        super(Cell, self).__init__()

        self.C_in = block_multiplier * filter_multiplier
        self.C_out = filter_multiplier

        self.C_prev_prev = int(prev_prev_fmultiplier * block_multiplier)
        self._prev_fmultiplier_same = prev_fmultiplier_same

        if prev_fmultiplier_down is not None:
            self.C_prev_down = int(prev_fmultiplier_down * block_multiplier)
            self.preprocess_down = ConvBR_3d(
                self.C_prev_down, self.C_out, 1, 1, 0)
        if prev_fmultiplier_same is not None:
            self.C_prev_same = int(prev_fmultiplier_same * block_multiplier)
            self.preprocess_same = ConvBR_3d(
                self.C_prev_same, self.C_out, 1, 1, 0)
        if prev_fmultiplier_up is not None:
            self.C_prev_up = int(prev_fmultiplier_up * block_multiplier)
            self.preprocess_up = ConvBR_3d(
                self.C_prev_up, self.C_out, 1, 1, 0)

        if prev_prev_fmultiplier != -1:
            self.pre_preprocess = ConvBR_3d(
                self.C_prev_prev, self.C_out, 1, 1, 0)

        self._steps = steps
        self.block_multiplier = block_multiplier
        self._ops = nn.ModuleList()

        for i in range(self._steps):
            for j in range(2 + i):
                stride = 1
                if prev_prev_fmultiplier == -1 and j == 0:
                    op = None
                else:
                    op = MixedOp(self.C_out, stride)
                self._ops.append(op)

        self._initialize_weights()

    def scale_dimension(self, dim, scale):
        assert isinstance(dim, int)
        return int((float(dim) - 1.0) * scale + 1.0) if dim % 2 else int(dim * scale)

    def prev_feature_resize(self, prev_feature, mode):
        if mode == 'down':
            feature_size_d = self.scale_dimension(prev_feature.shape[2], 0.5)
            feature_size_h = self.scale_dimension(prev_feature.shape[3], 0.5)
            feature_size_w = self.scale_dimension(prev_feature.shape[4], 0.5)
        elif mode == 'up':
            feature_size_d = self.scale_dimension(prev_feature.shape[2], 2)
            feature_size_h = self.scale_dimension(prev_feature.shape[3], 2)
            feature_size_w = self.scale_dimension(prev_feature.shape[4], 2)
        return F.interpolate(prev_feature, (feature_size_d, feature_size_h, feature_size_w), mode='trilinear', align_corners=True)

    def forward(self, s0, s1_down, s1_same, s1_up, n_alphas):

        if s1_down is not None:
            s1_down = self.prev_feature_resize(s1_down, 'down')
            s1_down = self.preprocess_down(s1_down)
            size_d, size_h, size_w = s1_down.shape[2], s1_down.shape[3], s1_down.shape[4]
        if s1_same is not None:
            s1_same = self.preprocess_same(s1_same)
            size_d, size_h, size_w = s1_same.shape[2], s1_same.shape[3], s1_same.shape[4]
        if s1_up is not None:
            s1_up = self.prev_feature_resize(s1_up, 'up')
            s1_up = self.preprocess_up(s1_up)
            size_d, size_h, size_w = s1_up.shape[2], s1_up.shape[3], s1_up.shape[4]
        all_states = []
        if s0 is not None:
            s0 = F.interpolate(s0, (size_d, size_h, size_w), mode='trilinear', align_corners=True) if (s0.shape[3] != size_h) or (s0.shape[4] != size_w) or (s0.shape[2] != size_d) else s0
            s0 = self.pre_preprocess(s0) if (s0.shape[1] != self.C_out) else s0
            if s1_down is not None:
                states_down = [s0, s1_down]
                all_states.append(states_down)
            if s1_same is not None:
                states_same = [s0, s1_same]
                all_states.append(states_same)
            if s1_up is not None:
                states_up = [s0, s1_up]
                all_states.append(states_up)
        else:
            if s1_down is not None:
                states_down = [0, s1_down]
                all_states.append(states_down)
            if s1_same is not None:
                states_same = [0, s1_same]
                all_states.append(states_same)
            if s1_up is not None:
                states_up = [0, s1_up]
                all_states.append(states_up)

        final_concates = []
        for states in all_states:
            offset = 0
            for i in range(self._steps):
                new_states = []
                for j, h in enumerate(states):
                    branch_index = offset + j
                    if self._ops[branch_index] is None:
                        continue
                    new_state = self._ops[branch_index](
                        h, n_alphas[branch_index])
                    new_states.append(new_state)

                s = sum(new_states)
                offset += len(states)
                states.append(s)

            concat_feature = torch.cat(states[-self.block_multiplier:], dim=1)
            final_concates.append(concat_feature)
        return final_concates


    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


class AutoMatching(nn.Module):
    def __init__(self, num_layers=8, filter_multiplier=4, block_multiplier=3, step=3, cell=Cell):        
        super(AutoMatching, self).__init__()

        self.cells = nn.ModuleList()
        self._step = step
        self._num_layers = num_layers
        self._block_multiplier = block_multiplier
        self._filter_multiplier = filter_multiplier
        
        f_initial = int(self._filter_multiplier)
        self._num_end = f_initial * self._block_multiplier

        self.stem0 = ConvBR_3d(self._num_end*2, self._num_end, 3, stride=1, padding=1)

        # network architecture [0, 0, 0, 1, 2, 1, 2, 2], 8 layers
        for i in range(self._num_layers):
            
            if i == 0:
                _cell = cell(self._step, self._block_multiplier, -1,
                             None, f_initial, None,
                             self._filter_multiplier)

            elif i == 1:
                _cell = cell(self._step, self._block_multiplier, f_initial,
                             None, self._filter_multiplier, self._filter_multiplier * 2,
                             self._filter_multiplier)

            elif i == 2:
                _cell = cell(self._step, self._block_multiplier, self._filter_multiplier,
                             None, self._filter_multiplier, self._filter_multiplier * 2,
                             self._filter_multiplier)

            elif i == 3:
                _cell = cell(self._step, self._block_multiplier, self._filter_multiplier,
                             self._filter_multiplier, self._filter_multiplier * 2, self._filter_multiplier * 4,
                             self._filter_multiplier * 2)
            elif i == 4:
                _cell = cell(self._step, self._block_multiplier, self._filter_multiplier,
                             self._filter_multiplier * 2, self._filter_multiplier * 4, self._filter_multiplier * 8,
                             self._filter_multiplier * 4)

            elif i == 5:
                _cell = cell(self._step, self._block_multiplier, self._filter_multiplier * 2,
                             self._filter_multiplier, self._filter_multiplier * 2, self._filter_multiplier * 4,
                             self._filter_multiplier * 2)

            elif i == 6:
                _cell = cell(self._step, self._block_multiplier, self._filter_multiplier * 4,
                             self._filter_multiplier * 2, self._filter_multiplier * 4, self._filter_multiplier * 8,
                             self._filter_multiplier * 4)

            else:
                _cell = cell(self._step, self._block_multiplier, self._filter_multiplier * 2,
                             self._filter_multiplier * 2, self._filter_multiplier * 4, self._filter_multiplier * 8,
                             self._filter_multiplier * 4)

            self.cells += [_cell]

        self.last_3  = ConvBR_3d(self._num_end, 1, 3, 1, 1,  bn=False, relu=False)  
        self.last_6  = ConvBR_3d(self._num_end*2 , self._num_end,    1, 1, 0)  
        self.last_12 = ConvBR_3d(self._num_end*4 , self._num_end*2,  1, 1, 0)  
        self.last_24 = ConvBR_3d(self._num_end*8 , self._num_end*4,  1, 1, 0)  
        

    def forward(self, x, n_alphas):
        self.level_3 = []
        self.level_6 = []
        self.level_12 = []
        self.level_24 = []

        

        stem = self.stem0(x) # [1, 12, 64, 64, 128]


        # network architecture [0, 0, 0, 1, 2, 1, 2, 2], 8 layers
        for layer in range(self._num_layers):
            if layer == 0:
                level3_new, = self.cells[layer](None, None, stem, None, n_alphas) # [1, 12, 64, 64, 128]

            elif layer == 1:
                level3_new_1, = self.cells[layer](stem, None, level3_new, None, n_alphas) # [1, 12, 64, 64, 128]

            elif layer == 2:
                level3_new_2, = self.cells[layer](level3_new, None, level3_new_1, None, n_alphas) # [1, 12, 64, 64, 128]

            elif layer == 3:
                level6_new, = self.cells[layer](level3_new_1, level3_new_2, None, None, n_alphas) # [1, 24, 32, 32, 64]

            elif layer == 4:
                level12_new, = self.cells[layer](level3_new_2, level6_new, None, None, n_alphas) # [1, 48, 16, 16, 32]

            elif layer == 5:
                level6_new, = self.cells[layer](level6_new, None, None, level12_new, n_alphas) # [1, 24, 32, 32, 64]

            elif layer == 6:
                level12_new_1, = self.cells[layer](level12_new, level6_new, None, None, n_alphas) # [1, 48, 16, 16, 32]

            elif layer == 7:
                level12_new_2, = self.cells[layer](level6_new, None, level12_new_1, None, n_alphas)


        last_output = level12_new_2

        #define upsampling
        d, h, w = x.size()[2], x.size()[3], x.size()[4]
        upsample_6  = nn.Upsample(size=x.size()[2:], mode='trilinear', align_corners=True)
        upsample_12 = nn.Upsample(size=[d//2, h//2, w//2], mode='trilinear', align_corners=True)
        upsample_24 = nn.Upsample(size=[d//4, h//4, w//4], mode='trilinear', align_corners=True)

        if last_output.size()[3] == h:
            mat = self.last_3(last_output)
        elif last_output.size()[3] == h//2:
            mat = self.last_3(upsample_6(self.last_6(last_output)))
        elif last_output.size()[3] == h//4:
            mat = self.last_3(upsample_6(self.last_6(upsample_12(self.last_12(last_output)))))
        elif last_output.size()[3] == h//8:
            mat = self.last_3(upsample_6(self.last_6(upsample_12(self.last_12(upsample_24(self.last_24(last_output)))))))      
        return mat  

