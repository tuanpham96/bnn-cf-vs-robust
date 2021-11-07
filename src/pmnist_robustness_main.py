import os
from pathlib import Path
import time
import shutil
from tqdm import tqdm

import itertools
import yaml
import argparse
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torchvision

import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('agg')

from models_utils import *
from pmnist_robustness_data_utils import TaskDataSet


# ------------------- HYPERPARAMETER PARSING -------------------

parser = argparse.ArgumentParser(description='BNN learning several tasks in a row, metaplasticity is controlled by the argument meta.')

# input related
parser.add_argument('--input', type = str, help='Source for meta-data of task sequence')
parser.add_argument('--batch-size', type = int, default = 100, help='(default: %(default)d) Batch size for data loader')
parser.add_argument('--num-workers', type = int, default = 2, help='(default: %(default)d) Number of workers for data loader')
parser.add_argument('--pin-memory', default = False, action = 'store_true', help='(default: %(default)d) Pin memory for data loader')

# model related
parser.add_argument('--hidden-layers', nargs = '+', type = int, help='Size of the hidden layers')
parser.add_argument('--init', type = str, default = 'uniform', help='(default: %(default)s) Weight initialisation type "uniform" or "gaussian"')
parser.add_argument('--init-width', type = float, default = 0.1, help='(default: %(default)f) Weight initialisation width')

# learning related
parser.add_argument('--lr', type = float, default = 0.005, help='(default: %(default)f) Learning rate')
parser.add_argument('--meta', type = float, nargs = '+', help='Metaplasticity coefficients layer wise')
parser.add_argument('--decay', type = float, default = 0.0, help='(default: %(default)f) Weight decay')
parser.add_argument('--gamma', type = float, default = 1.0, help='(default: %(default)f) Dividing factor for lr decay')
parser.add_argument('--epochs-per-task', type = int, default = 5, help='(default: %(default)d) Number of epochs per tasks')

# output related
parser.add_argument('--save-path', type = str, default = './data/output', help='(default: %(default)s) Save data path')
parser.add_argument('--output-name', type = str, default = '', help='(default: %(default)s) Name of the output directory to be concat to `--save-path`')

# device and platform related
parser.add_argument('--seed', type = int, default = None, help='(default: None) Seed for reproductibility')
parser.add_argument('--device', type = int, default = 0, help='(default: %(default)d) Choice of gpu')

args = parser.parse_args()

# These hyperparameters need to be fixed
args.scenario = 'task'
args.net = 'bnn'
args.in_size = 784
args.out_size = 10
args.norm = 'bn'
args.rnd_consolidation = False
args.ewc_lambda = 0.0
args.ewc = False
args.si_lambda = 0.0
args.si = False
args.bin_path = False
args.interleaved = False
args.beaker = False
# these are for beaker args from the original code but not using
# leave here just in case some legacy code uses them for checking
args.fb = None
args.n_bk = None
args.ratios = [None]
args.areas = [None]

# ------------------- TORCH SETUP -------------------

device = torch.device("cuda:"+str(args.device) if torch.cuda.is_available() else "cpu")

if args.seed is not None:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

# ------------------- OUTPUT PATH SETUP -------------------


if len(args.output_name) == 0:
    args.output_name = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')


main_path = Path(args.save_path) / args.output_name
main_path.mkdir(parents=True, exist_ok=True)

plot_path = main_path / 'plots'
plot_path.mkdir(parents=True, exist_ok=True)

model_path = main_path / 'models'
model_path.mkdir(parents=True, exist_ok=True)

exp_conf_path    = main_path / 'exp-config.yaml'
forget_perf_path = main_path / 'perf_forget.csv'
robust_perf_path = main_path / 'perf_robust.csv'

# ------------------- SAVE EXPERIMENT CONFIG -------------------

print('>>>>>>>>>>>>>>>>>>>> EXPERIMENT CONFIG <<<<<<<<<<<<<<<<<<\n')
exp_conf_dict = vars(args)
print(yaml.safe_dump(exp_conf_dict, default_flow_style=False))

with open(exp_conf_path, 'w') as file:
    documents = yaml.safe_dump(exp_conf_dict, file, default_flow_style=False)

# ------------------- INPUT SETUP -------------------

input_path = args.input
shutil.copy(input_path, str(main_path / 'input.yaml')) # copy for reference

with open(input_path, 'r') as file: # load it
    task_metadata = yaml.safe_load(file)

task_paths = task_metadata['task_paths']
task_ids = task_metadata['task_ids']
num_tasks = task_metadata['num_tasks']

# ------------------- DATA LOADER SETUP -------------------

common_dload_args = dict(
    batch_size  = args.batch_size,
    num_workers = args.num_workers,
    pin_memory  = args.pin_memory
)

transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=(0.0,), std=(1.0,))
])

def load_dataset(task_id, data_type, data_action=None):
    if data_action:
        data_key = '%s::%s' %(data_type, data_action)
        shuffle_on = data_action == 'train'
    else:
        data_key = data_type
        shuffle_on = False

    data_prefix = task_paths[task_id][data_key]['prefix']

    dset = TaskDataSet(data_prefix, transform=transform)
    data_loader = torch.utils.data.DataLoader(dset, shuffle=shuffle_on, **common_dload_args)
    return data_loader

train_loader_list = []
test_loader_list = []

for task_id in task_ids:
    train_loader_list.append(load_dataset(task_id, 'original', 'train'))
    test_loader_list.append(load_dataset(task_id, 'original', 'test'))

# ------------------- TRAINING AND MODEL SETUP -------------------

# MODEL CONSTRUCTION
archi = [args.in_size] + args.hidden_layers + [args.out_size]
model_args = dict(layers_dims = archi, init = args.init, width = args.init_width, norm = args.norm)
model = BNN(**model_args).to(device)

# TRAINING HYPERPARAMETERS
lr = args.lr
lrs = [lr*(args.gamma**(-i)) for i in range(num_tasks)] # scale by gamma
epochs = args.epochs_per_task

meta = {}
for n, p in model.named_parameters():
    index = int(n[9])
    p.newname = 'l'+str(index)
    if ('fc' in n) or ('cv' in n):
        meta[p.newname] = args.meta[index-1] if len(args.meta)>1 else args.meta[0]

print('>>>>>>>>>>>>>>>>>>>> MODEL CONSTRUCTION <<<<<<<<<<<<<<<<<<\n')

print(model)

# ------------------- DATA COLLECTION SETUP -------------------

arch = ''
for i in range(model.hidden_layers):
    arch = arch + '-' + str(model.layers_dims[i+1])

data = dict(
    net         = args.net,
    arch        = arch[1:],
    lr          = [],
    meta        = [],
    task_order  = [],
    task_id     = [],
    task_epoch  = [],
    glob_epoch  = [],
    train_acc   = [],
    train_loss  = [],
    train_time  = []
)
for task_id in task_ids:
    data['test_acc::'  + task_id] = []
    data['test_loss::' + task_id] = []
    data['test_time::' + task_id] = []

# batchnorm states
bn_states = []

# ------------------- DEFINE ARGS FOR TQDMS -------------------

parent_tqdm_args = dict(
    position=0,
    leave=True,
    ncols=100,
    colour='green'
)

child_tqdm_args = dict(
    position=1,
    leave=False,
    ncols=80
)
# ------------------- CATASTROPHIC FORGETTING - METAPLAST TRAINING -------------------

print('>>>>>>>>>>>>>>>>>>>> METAPLAST TRAINING <<<<<<<<<<<<<<<<<<\n')

glob_epoch = 1 # global epoch number

for task_ind, task_data in tqdm(enumerate(train_loader_list), desc='MAIN TRAIN', **parent_tqdm_args):
    # OPTIMIZER for each task
    optimizer = Adam_meta(model.parameters(), lr = lrs[task_ind], meta = meta, weight_decay = args.decay)
    task_id = task_ids[task_ind]

    # --- TRAIN AND TEST MODEL ---
    for epoch in tqdm(range(1, epochs+1), desc='+ Train ' + task_id, **child_tqdm_args):

        # TRAIN MODEL
        t0 = time.time()
        train(model, task_data, task_ind, optimizer, device, args)
        t1 = time.time()

        train_accuracy, train_loss = test(model, task_data, device)
        train_time = t1 - t0

        # SAVE GENERAL
        data['task_order'].append(task_ind+1)
        data['task_id'].append(task_id)
        data['task_epoch'].append(epoch)
        data['glob_epoch'].append(glob_epoch)
        glob_epoch += 1

        data['lr'].append(optimizer.param_groups[0]['lr'])
        data['meta'].append(meta)

        # SAVE TRAIN PERFORMANCE
        data['train_acc'].append(train_accuracy)
        data['train_loss'].append(train_loss)
        data['train_time'].append(train_time)

        print('\n\t - TRAIN[%s] | epoch = %02d | acc = %.2f %% | loss = %.4f | time = %.2f min' \
              %(task_id, epoch, train_accuracy, train_loss, train_time/60.0))

        # save batchnorm states
        current_bn_state = model.save_bn_states()

        # TESTING OTHER TASKS
        for other_task_ind, other_task_data in enumerate(test_loader_list):

            # load batchnorm states according to past or current task
            bnstate2load = current_bn_state if other_task_ind>=task_ind else bn_states[other_task_ind]
            model.load_bn_states(bnstate2load)

            # TEST FOR OTHER TASK
            t0 = time.time()
            test_accuracy, test_loss = test(model, other_task_data, device)
            t1 = time.time()
            test_time = t1 - t0

            # SAVE TEST PERFORMANCE
            other_task_id = task_ids[other_task_ind]
            data['test_acc::'  + other_task_id].append(test_accuracy)
            data['test_loss::' + other_task_id].append(test_loss)
            data['test_time::' + other_task_id].append(test_time)

            print('\t - TEST[%s] | epoch = %02d | acc = %.2f %% | loss = %.4f | time = %.2f min' \
                  %(other_task_id, epoch, test_accuracy, test_loss, test_time/60.0))

        # load back to current batchnorm states to continue training
        model.load_bn_states(current_bn_state)

    # --- AT THE END OF EACH TASK ---
    # SAVE PLOT PARAMETERS
    plot_parameters(model, save=True, save_path=str(plot_path / task_id))

    # SAVE MODEL STATES
    torch.save(dict(
        model_args      = model_args,
        model_states    = model.state_dict(),
        task_order      = task_ind+1,
        task_id         = task_id,
        task_epoch      = epoch,
        glob_epoch      = glob_epoch,
    ), model_path / (task_id + '.pt'))

    # append to atchnorm stats
    bn_states.append(current_bn_state)

# SAVE PERFORMANCE FOR CATASTROPHIC FORGETTING PROGRESS
df_data = pd.DataFrame(data)
df_data.to_csv(forget_perf_path, index = False)

# ------------------- ROBUSTNESSS TESTING -------------------

# delete to save memory
del train_loader_list
del test_loader_list

# create data
data = dict(
    source_task     = [],
    data_key        = [],
    train_phase     = [],
    test_acc        = [],
    test_loss       = [],
    test_time       = []
)

# only exclude original::train for testing
perturb_keys = [k for k in task_paths[task_ids[0]].keys() if k != 'original::train']
perturb_data_src = list(itertools.product(task_ids, perturb_keys))

print('>>>>>>>>>>>>>>>>>>>> ROBUSTNESSS TESTING <<<<<<<<<<<<<<<<<<\n')

for source_task, data_key in tqdm(perturb_data_src, desc='CORRUPTED DATA', **parent_tqdm_args):
    corrupted_dataloader = load_dataset(source_task, data_key)

    for train_phase in task_ids:
        # LOAD MODEL FOR EACH TRAINING PHASE
        ckpt = torch.load(model_path / (train_phase + '.pt'))

        model.load_state_dict(ckpt['model_states'])
        model.eval()

        # TEST PERFORMANCE
        t0 = time.time()
        test_acc, test_loss = test(model, corrupted_dataloader, device)
        t1 = time.time()
        test_time = t1 - t0

        # SAVE DATA
        data['source_task'].append(source_task)
        data['data_key'].append(data_key)
        data['train_phase'].append(train_phase)
        data['test_acc'].append(test_acc)
        data['test_loss'].append(test_loss)
        data['test_time'].append(test_time)
        print('\n source = %s | type = %s | train_at = %s || acc = %.2f %% | loss = %.4f | time = %.2f min' \
              %(source_task, data_key, train_phase, test_acc, test_loss, test_time/60.0))

df_data = pd.DataFrame(data)
df_data.to_csv(robust_perf_path, index = False)

print('\n>>>>>>>>>>>>>>>>>>>> FINISHED <<<<<<<<<<<<<<<<<<\n')
