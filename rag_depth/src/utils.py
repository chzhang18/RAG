import os,sys
import math
import numpy as np
from copy import deepcopy
import torch
from tqdm import tqdm


def get_model_size(model, mode=None):
    count = 0
    for p in model.parameters():
        count += np.prod(p.size())
    human_count = None
    if mode is None:
        human_count = human_format(count)
    elif mode == 'M':
        human_count = human_format_m(count)

    return human_count


def print_model_report(model):
    print('-'*100)
    print(model)
    print('Dimensions =', end=' ')
    count = 0
    for p in model.parameters():
        print(p.size(), end=' ')
        count += np.prod(p.size())
    print()
    print('Num parameters = %s' % (human_format(count)))
    print('-'*100)
    return count


def human_format_m(num):

    return num / 1000000.0


def human_format(num):
    magnitude = 0
    while abs(num) >= 1000:
        magnitude += 1
        num /= 1000.0
    return '%.2f%s' % (num, ['', 'K', 'M', 'G', 'T', 'P'][magnitude])


def print_optimizer_config(optim):
    if optim is None:
        print(optim)
    else:
        print(optim,'=',end=' ')
        opt=optim.param_groups[0]
        for n in opt.keys():
            if not n.startswith('param'):
                print(n+':',opt[n],end=', ')
        print()
    return

########################################################################################################################


def get_model(model):
    return deepcopy(model.state_dict())


def set_model_(model,state_dict):
    model.load_state_dict(deepcopy(state_dict))
    return


def modify_model(model, requires_grad=True):
    for param in model.parameters():
        param.requires_grad = requires_grad


def freeze_model(model):
    for param in model.parameters():
        param.requires_grad = False
    return


def unfreeze_model(model):
    for param in model.parameters():
        param.requires_grad = True
    return


def freeze_parameter(params):
    for param in params.values():
        param.requires_grad = False


def unfreeze_parameter(params):
    for param in params.values():
        param.requires_grad = True

########################################################################################################################


def model_size(model):
    total_size = 0.0
    for param in model.parameters():
        size_list = param.size()
        size = 1
        for i in range(len(size_list)):
            size *= size_list[i]
        total_size += size
    log_size = math.log(1+total_size, 10)

    return log_size
########################################################################################################################


def compute_conv_output_size(Lin,kernel_size,stride=1,padding=0,dilation=1):
    return int(np.floor((Lin+2*padding-dilation*(kernel_size-1)-1)/float(stride)+1))

########################################################################################################################


def compute_mean_std_dataset(dataset):
    # dataset already put ToTensor
    mean=0
    std=0
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
    for image, _ in loader:
        mean+=image.mean(3).mean(2)
    mean /= len(dataset)

    mean_expanded=mean.view(mean.size(0),mean.size(1),1,1).expand_as(image)
    for image, _ in loader:
        std+=(image-mean_expanded).pow(2).sum(3).sum(2)

    std=(std/(len(dataset)*image.size(2)*image.size(3)-1)).sqrt()

    return mean, std

########################################################################################################################


def fisher_matrix_diag(t, train_loader, model, criterion, device, sbatch=20):
    # Init
    fisher = {}
    with torch.no_grad():
        for n, p in model.named_parameters():
            fisher[n] = 0 * p.data
    # Compute
    model.train()
    count = 0
    for sample in train_loader:
        imgL, imgR, disp_gt = sample['left'].to(device), sample['right'].to(device), sample['disparity'].to(device)
        # Forward and backward
        print(count)
        count += 1
        model.zero_grad()
        disp_est = model.forward(imgL, imgR)
        mask = (disp_gt < 192) & (disp_gt > 0)
        loss = criterion(t, disp_est[mask], disp_gt[mask])
        loss.backward()

        # Get gradients
        with torch.no_grad():
            for n, p in model.named_parameters():
                if p.grad is not None:
                    fisher[n] += sbatch * p.grad.data.pow(2)
    # Mean
    for n, _ in model.named_parameters():
        fisher[n] = fisher[n] / len(train_loader)

    return fisher

########################################################################################################################


def cross_entropy(outputs,targets,exp=1,size_average=True,eps=1e-5):
    out=torch.nn.functional.softmax(outputs)
    tar=torch.nn.functional.softmax(targets)
    if exp!=1:
        out=out.pow(exp)
        out=out/out.sum(1).view(-1,1).expand_as(out)
        tar=tar.pow(exp)
        tar=tar/tar.sum(1).view(-1,1).expand_as(tar)
    out=out+eps/out.size(1)
    out=out/out.sum(1).view(-1,1).expand_as(out)
    ce=-(tar*out.log()).sum(1)
    if size_average:
        ce=ce.mean()
    return ce

########################################################################################################################


def set_req_grad(layer, req_grad):
    if hasattr(layer, 'weight'):
        layer.weight.requires_grad=req_grad
    if hasattr(layer, 'bias'):
        layer.bias.requires_grad=req_grad
    return

########################################################################################################################


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass

    try:
        import unicodedata
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass

    return False
########################################################################################################################
