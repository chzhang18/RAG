import sys, os, argparse, time
import logging
import json
import numpy as np
import torch
from torch.utils.tensorboard.writer import SummaryWriter

import utils
from dataloaders.stereo_dataset import StereoDataset

from approaches import rag as approach

tstart=time.time()

# Arguments
parser=argparse.ArgumentParser(description='xxx')
parser.add_argument('--seed', type=int, default=0, help='(default=%(default)d)')
parser.add_argument('--experiment', default='drivingstereo', type=str,help='(default=%(default)s)')
parser.add_argument('--approach', default='rag', type=str,help='(default=%(default)s)')
# mode: training or search the best hyper-parameter
parser.add_argument('--mode', default='train', type=str, required=False, choices=['train', 'search'],
                    help='(default=%(default)s)')
# if debug is true, only use a small dataset
parser.add_argument('--debug', default='False', type=str, required=False, choices=['False', 'True'],
                    help='(default=%(default)s)')
# model: the basic model
parser.add_argument('--output',default='', type=str, required=False, help='(default=%(default)s)')
parser.add_argument('--device', type=str, default='0', help='choose the device')
parser.add_argument('--id', type=str, default='0', help='the id of experiment')
#parser.add_argument('--search_layers', default=4, type=int, required=False, help='(default=%(default)d)')
#parser.add_argument('--eval_layers', default=6, type=int, required=False, help='(default=%(default)d)')
# hyper parameters in cell search stage
parser.add_argument('--c_epochs', default=100, type=int, required=False, help='(default=%(default)d)')
parser.add_argument('--c_batch', default=8, type=int, required=False, help='(default=%(default)d)')
parser.add_argument('--c_lr', default=0.025, type=float, required=False, help='(default=%(default)f)')
parser.add_argument('--c_lr_a', default=0.01, type=float, required=False, help='(default=%(default)f)')
parser.add_argument('--c_lamb', default=0.0003, type=float, required=False, help='(default=%(default)f)')

# hyper parameters in operation search stage
parser.add_argument('--o_epochs', default=100, type=int, required=False, help='(default=%(default)d)')
parser.add_argument('--o_batch', default=6, type=int, required=False, help='(default=%(default)d)')
parser.add_argument('--o_lr', default=0.025, type=float, required=False, help='(default=%(default)f)')
parser.add_argument('--o_lr_a', default=0.01, type=float, required=False, help='(default=%(default)f)')
parser.add_argument('--o_lamb', default=0.0003, type=float, required=False, help='(default=%(default)f)')
parser.add_argument('--o_lamb_a', default=0.0003, type=float, required=False, help='(default=%(default)f)')
parser.add_argument('--o_lamb_size', default=1, type=float, required=False, help='(default=%(default)f)')
parser.add_argument('--o_size', default=0, type=int, required=False,
help="the initial number of epochs for previous units")

# hyper parameters in training stage
parser.add_argument('--epochs', default=300, type=int, required=False, help='(default=%(default)d)')
parser.add_argument('--batch', default=4, type=int, required=False, help='(default=%(default)d)')
parser.add_argument('--lr', default=0.025, type=float, required=False, help='(default=%(default)f)')
parser.add_argument('--lamb', default=0.0003, type=float, required=False, help='(default=%(default)f)')
parser.add_argument('--lamb_ewc', default=10000, type=float, required=False, help='(default=%(default)f)')


parser.add_argument('--maxdisp', default=192, type=float, required=False, help='(default=%(default)f)')

args = parser.parse_args()

if args.output == '':
    args.output = '../res/'+args.experiment+'_'+args.approach+'_'+str(args.seed)+'_'+args.id+'.txt'

print('='*100)
print('Arguments =')
for arg in vars(args):
    print('\t'+arg+':', getattr(args, arg))
print('='*100)

########################################################################################################################

# Seed
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)
    device = torch.device('cuda:'+args.device)
else:
    device = torch.device('cpu')
    print('[CUDA unavailable]')
    sys.exit()

########################################################################################################################
# define logger
logger = logging.getLogger()

# define tensorboard writer
exp_name = args.experiment+'_'+args.approach+'_'+str(args.seed)+'_'+args.id

writer = SummaryWriter(log_dir='../logs/' + exp_name)


# Load date
print('Load data...')

# logging the experiment config
config_exp = ["name: {}".format(exp_name),
              "mode: {}".format(args.mode),
              "dataset: {}".format(args.experiment),
              "approach: {}".format(args.approach),
              "device: {}".format(args.device),
              "id: {}".format(args.id)]
for i, string in enumerate(config_exp):
    writer.add_text("Config_Experiment", string, i)
# logging the hyperparameter config
config_hyper_train = [
    "train_epochs: {}".format(args.epochs),
    "train_batch_size: {}".format(args.batch),
    "train_learning_rate: {}".format(args.lr),
    "train_weight_decay: {}".format(args.lamb)
]
# "learning_rate_patience: {}".format(args.lr_patience),
# "learning_factor: {}".format(args.lr_factor),


config_hyper_operation = [
    "operation_search_epochs: {}".format(args.o_epochs),
    "operation_search_batch_size: {}".format(args.o_batch),
    "operation_search_learning_rate_m: {}".format(args.o_lr),
    "operation_search_learning_rate_p: {}".format(args.o_lr_a),
    "operation_search_weight_decay_m: {}".format(args.o_lamb),
    "operation_search_weight_decay_p: {}".format(args.o_lamb_a),
    "operation_search_initial_epochs: {}".format(args.o_size)
]

config_hyper_cell = [
    "cell_search_epochs: {}".format(args.c_epochs),
    "cell_search_batch_size: {}".format(args.c_batch),
    "cell_search_learning_rate_m: {}".format(args.c_lr),
    "cell_search_learning_rate_p: {}".format(args.c_lr_a),
    "cell_search_weight_decay_m: {}".format(args.c_lamb),
]

for i, string in enumerate(config_hyper_train):
    writer.add_text("Config_Train", string, i)

for i, string in enumerate(config_hyper_operation):
    writer.add_text("Config_Operation", string, i)

for i, string in enumerate(config_hyper_cell):
    writer.add_text("Config_Cell", string, i)

# Inits
appr = approach.Appr(device=device, writer=writer, exp_name=exp_name, args=args)


# Loop tasks
d1 = np.zeros((4, 4), dtype=np.float32)
lss = np.zeros((4, 4), dtype=np.float32)
epe = np.zeros((4, 4), dtype=np.float32)
model_size = []


train_data_list = ['./filenames/drivingstereo/drivingstereo_cloudy_train.txt', './filenames/drivingstereo/drivingstereo_foggy_train.txt', 
                  './filenames/drivingstereo/drivingstereo_rainy_train.txt', './filenames/drivingstereo/drivingstereo_sunny_train.txt']
test_data_list = ['./filenames/drivingstereo/drivingstereo_cloudy_test.txt', './filenames/drivingstereo/drivingstereo_foggy_test.txt',
                  './filenames/drivingstereo/drivingstereo_rainy_test.txt', './filenames/drivingstereo/drivingstereo_sunny_test.txt']


for t in range(len(train_data_list)):
    print('*'*100)
    print('Task {:2d}'.format(t))
    print('*'*100)

    # get dataset
    train_data = StereoDataset(t, train_data_list, True)
    valid_data = StereoDataset(t, train_data_list, False)

    # Train
    appr.train(t, train_data, valid_data, device=device)
    print('-'*100)

    # Test
    for u in range(t+1):
        test_data = StereoDataset(u, test_data_list, False)
        test_loader = torch.utils.data.DataLoader(
            test_data, batch_size=1, shuffle=False, pin_memory=True, num_workers=4)

        test_loss, test_d1, test_epe, test_bad_1 = appr.eval(u, test_loader, mode='train', device=device)
        print('>>> Test on task {:2d} - {:15s}: loss={:.3f}, D1={:.3f}%, EPE={:.3f}, Bad_1={:.3f}% <<<'.format(
            u, 'Scene Flow', test_loss, 100*test_d1, test_epe, 100*test_bad_1))
        writer.add_scalars('Test/Loss',
                           {'task{}'.format(u): test_loss}, global_step=t)
        writer.add_scalars('Test/D1_All',
                           {'task{}'.format(u): test_d1 * 100}, global_step=t)
        writer.add_scalars('Test/EPE',
                           {'task{}'.format(u): test_epe}, global_step=t)

        d1[t, u] = test_d1
        lss[t, u] = test_loss
        epe[t, u] = test_epe

    checkpoint_data = {'task': t, 'model': appr.model.state_dict(), 'optimizer': appr.optimizer.state_dict()}
    save_path = "../logs/" + exp_name + "/checkpoint_task" + str(t) + ".ckpt"
    torch.save(checkpoint_data, save_path)

    model_size.append(utils.get_model_size(appr.model, mode='M'))
    writer.add_scalars('ModelParameter(M)',
                       {'ModelParameter(M)': utils.get_model_size(appr.model, 'M')},
                       global_step=t)

# Done, logging the experiment results
print('*'*100)
print('D1-All =')
for i in range(d1.shape[0]):
    print('\t', end='')
    for j in range(d1.shape[1]):
        writer.add_text("Results/D1_All", '{:5.1f}% '.format(100*d1[i, j]), i)
        print('{:5.1f}% '.format(100*d1[i, j]), end='')
    print()
print('*'*100)
print('Done!')

print('EPE =')
for i in range(epe.shape[0]):
    print('\t', end='')
    for j in range(epe.shape[1]):
        writer.add_text("Results/EPE", '{:.3f} '.format(epe[i, j]), i)
        print('{:.3f} '.format(epe[i, j]), end='')
    print()
print('*'*100)
print('Done!')

writer.add_text("Results/Time", '[Elapsed time = {:.1f} h]'.format((time.time()-tstart)/(60*60)))
writer.add_text("Results/Mean_D1_All", "Mean_D1_All = {}".format(np.mean(d1[-1]).tolist()))
writer.add_text("Results/Size", "Size at last = {}".format(model_size[-1]))
# store the result of experiment
result = {"model_size": model_size, "mean_D1": np.mean(d1[-1]).tolist(), "D1_All": d1.tolist(),
          "elapsed_time(h)": (time.time()-tstart)/(60*60)}

print("Results/Time", '[Elapsed time = {:.1f} h]'.format((time.time()-tstart)/(60*60)))
print("Results/Mean_D1_All", "Mean_D1_All = {}".format(np.mean(d1[-1]).tolist()))
print("Results/Size", "Size at last = {}".format(model_size[-1]))
