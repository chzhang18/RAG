import torch.nn.functional as F
import torch
import torch.nn as nn
from torch.autograd import Variable

def warp(x, disp):
    '''
    warp an image/tensor(im2) back to im1, according to the disparity

    x: [B, C, H, W] (im2)
    disp: [B, 1, H, W] (disparity)
        
    '''
    B, C, H, W = x.size()
    # mesh grid
    xx = torch.arange(0, W).view(1,-1).repeat(H,1)
    yy = torch.arange(0, H).view(-1,1).repeat(1,W)
    xx = xx.view(1,1,H,W).repeat(B,1,1,1)
    yy = yy.view(1,1,H,W).repeat(B,1,1,1)
    grid = torch.cat((xx,yy),1).float()
    if x.is_cuda:
        grid = grid.cuda()
    flow_y = torch.zeros([B, 1, H, W]).cuda()
    flow = torch.cat((disp, flow_y), 1)
    vgrid = Variable(grid) - flow
    # scale grid to [-1,1]
    vgrid[:,0,:,:] = 2.0*vgrid[:,0,:,:].clone() / max(W-1,1)-1.0
    vgrid[:,1,:,:] = 2.0*vgrid[:,1,:,:].clone() / max(H-1,1)-1.0
    vgrid = vgrid.permute(0,2,3,1)
    output = nn.functional.grid_sample(x, vgrid)
    mask = torch.autograd.Variable(torch.ones(x.size())).cuda()
    mask = nn.functional.grid_sample(mask, vgrid)
    mask[mask<0.9999] = 0
    mask[mask>0] = 1

    return output*mask


def gradient_x(img):
    # img : [B,C,H,W]
    gx = img[:,:,:,:-1] - img[:,:,:,1:]
    return gx

def gradient_y(img):
    gy = img[:,:,:-1,:] - img[:,:,1:,:]
    return gy

def charbonnier_function(x):
    alpha = 0.21
    eps = 0.001
    return (x.pow(2)+eps*eps).pow(alpha)


def l1(x,y,mask=None):
    """
    pixelwise reconstruction error
    Args:
        x: predicted image
        y: target image
        mask: compute only on this points
    """
    if mask is None:
        mask = torch.ones_like(x, dtype=torch.float32)
    return mask*torch.abs(x-y)

def mean_l1(x,y,mask=None):
    """
    Mean reconstruction error
    Args:
        x: predicted image
        y: target image
        mask: compute only on this points
    """
    if mask is None:
        mask = torch.ones_like(x, dtype=torch.float32)
    return ( (mask*torch.abs(x-y)).mean() ) / mask.mean()

def SSIM(x, y):
    """
    SSIM dissimilarity measure
    Args:
        x: predicted image
        y: target image
    """
    C1 = 0.01**2
    C2 = 0.03**2
    mu_x = F.avg_pool2d(x, kernel_size=3)
    mu_y = F.avg_pool2d(y, kernel_size=3)
    
    sigma_x = F.avg_pool2d(x**2, kernel_size=3) - mu_x**2
    sigma_y = F.avg_pool2d(y**2, kernel_size=3) - mu_y**2
    sigma_xy = F.avg_pool2d(x*y, kernel_size=3) - mu_x * mu_y

    SSIM_n = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
    SSIM_d = (mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2)

    SSIM = SSIM_n / SSIM_d

    return torch.clamp((1-SSIM)/2, 0 ,1)

def mean_SSIM(x,y):
    """
    Mean error over SSIM reconstruction
    """
    return SSIM(x,y).mean()

def mean_SSIM_L1(x, y):
    return 0.85 * mean_SSIM(x, y) + 0.15 * mean_l1(x, y)



def re_and_sm_loss(disp_est, left, right):
    all_losses = []
    
    disp_est = disp_est.unsqueeze(1)

    # reconstruction loss
    left_est = warp(right, disp_est)
    reconstruction_loss = mean_SSIM_L1(left, left_est)
    all_losses.append(reconstruction_loss)

    # smoothness loss
    disp_gradients_x = gradient_x(disp_est)
    disp_gradients_y = gradient_y(disp_est)

    image_gradients_x = torch.mean(gradient_x(left), 1, keepdim=True)
    image_gradients_y = torch.mean(gradient_y(left), 1, keepdim=True)

    weights_x = torch.exp(-torch.mean(torch.abs(image_gradients_x), 1, keepdim=True))
    weights_y = torch.exp(-torch.mean(torch.abs(image_gradients_y), 1, keepdim=True))

    smoothness_x = torch.abs(disp_gradients_x) * weights_x
    smoothness_y = torch.abs(disp_gradients_y) * weights_y

    smoothness_x = F.pad(smoothness_x, (0,1,0,0,0,0,0,0), mode='constant')
    smoothness_y = F.pad(smoothness_y, (0,0,0,1,0,0,0,0), mode='constant')

    smooth_loss = torch.mean(smoothness_x + smoothness_y)
    all_losses.append(0.1 * smooth_loss)

    return sum(all_losses)