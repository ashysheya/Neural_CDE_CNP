import argparse

import numpy as np
import stheno.torch as stheno
import torch
import matplotlib.pyplot as plt

import lib.data
from lib.neural_cde_cnp import get_net
from lib.losses import get_loss
from tensorboardX import SummaryWriter

import tqdm

from lib.utils import (
    device,
    report_loss,
    RunningAverage,
    generate_root,
    WorkingDirectory,
    save_checkpoint,
    update_learning_rate
)

plt.switch_backend('agg')

def validate(data, model, losses, report_freq=None, mode='mean'):
    """Compute the validation loss."""
    ravg = {'mse': RunningAverage(), 
            'nll': RunningAverage(), 
            'mse_extrap': RunningAverage(),
            'nll_extrap': RunningAverage()}

    model.eval()
    with torch.no_grad():
        for step, task in enumerate(data):

            output = model(task)

            task['mask_y'] = task['mask_y'] - task['mask_obs']

            if mode == 'mean':
                batch_size = 1
            else:
                batch_size = data.batch_size

            for loss in ['mse', 'nll']:
                index = task['extrap_index']

                loss_obj = losses[loss](task, output, start=0, end=index, mode=mode)

                ravg[loss].update(loss_obj.item(), batch_size)

                if report_freq:
                    report_loss(f'Validation {loss}', ravg[loss].avg, step, report_freq)

                if index < task['x'].size()[0]:
                    loss_obj = losses[loss](task, 
                                            output, 
                                            start=index,
                                            end=len(task['x']),
                                            mode=mode)

                    ravg[f'{loss}_extrap'].update(loss_obj.item(), batch_size)
                    if report_freq:
                        report_loss(f'Validation {loss} extrap',
                                    ravg[f'{loss}_extrap'].avg,
                                    step,
                                    report_freq)

    return {loss: ravg[loss].avg for loss in ravg}


def train(data, model, losses, opt, report_freq, mode='mean'):
    """Perform a training epoch."""
    if 'kl' in losses:
        ravg = {'kl': RunningAverage(), 
                'nll': RunningAverage(), 
                'obj': RunningAverage()}
    else:
        ravg = {'obj': RunningAverage()}

    model.train()
    for step, task in enumerate(data):

        if mode == 'mean':
            batch_size = 1
        else:
            batch_size = task['y'].size()[0]

        output = model(task)

        obj = losses['nll'](task, output, start=0, end=len(task['x']), mode=mode)

        obj.backward()
        opt.step()
        opt.zero_grad()
        ravg['obj'].update(obj.item(), batch_size)
        report_loss('Training', ravg['obj'].avg, step, report_freq)

    return {loss: ravg[loss].avg for loss in ravg}


def to_numpy(x):
    """Convert a PyTorch tensor to NumPy."""
    return x.squeeze().detach().cpu().numpy()


def plot_model_task(model, data, epoch, wd):
    """Plot validation samples."""
    for step, task in enumerate(data):
        model.eval()
        with torch.no_grad():
            output = model(task)

        pred = to_numpy(output['mean_pred'])
        pred_std = to_numpy(output['std_pred'])

        observations_mask = to_numpy(task['mask_obs'])
        ground_truth_data = to_numpy(task['y'])
        x = to_numpy(task['x'])

        for i in range(len(ground_truth_data)):
            # Plot context.
            fig = plt.figure()
            num_context_points = (x[observations_mask[i] == 1] <= task['extrap_time']).sum()
            plt.scatter(x[observations_mask[i] == 1][:num_context_points],
                        ground_truth_data[i, observations_mask[i] == 1][:num_context_points],
                        label='Context Set', color='indianred')
            plt.plot(x, pred[i], label='Predicted', color='navy')

            plt.fill_between(x, pred[i] + 2 * pred_std[i],
                pred[i] - 2 * pred_std[i], color='navy', alpha=0.1)

            plt.plot(x, ground_truth_data[i], label='Oracle GP', color='forestgreen')

            plt.legend()
            plt.savefig(wd.file('plots', f'epoch_{epoch + 1}_plot_{i + 1}.png'))
            plt.close()


# Parse arguments given to the script.
parser = argparse.ArgumentParser()
parser.add_argument('data',
                    choices=['eq',
                             'matern',
                             'noisy-mixture',
                             'weakly-periodic',
                             'sawtooth', 
                             'gauss-markov'],
                    help='Data set to train the CNP on. ')
parser.add_argument('--root',
                    help='Experiment root, which is the directory from which '
                         'the experiment will run. If it is not given, '
                         'a directory will be automatically created.')
parser.add_argument('--train',
                    action='store_true',
                    help='Perform training. If this is not specified, '
                         'the model will be attempted to be loaded from the '
                         'experiment root.')
parser.add_argument('--epochs',
                    default=100,
                    type=int,
                    help='Number of epochs to train for.')
parser.add_argument('--learning_rate',
                    default=1e-3,
                    type=float,
                    help='Learning rate.')
parser.add_argument('--weight_decay',
                    default=1e-5,
                    type=float,
                    help='Weight decay.')

# Model specification
parser.add_argument('--in_channels',
                    default=1,
                    type=int)
parser.add_argument('--conv_cnp_layers',
                    default=8,
                    type=int)
parser.add_argument('--conv_cnp_channels',
                    default=64,
                    type=int)
parser.add_argument('--bottleneck_function_channels',
                    default=1,
                    type=int)
parser.add_argument('--ode_hidden_state_channels',
                    default=20,
                    type=int)
parser.add_argument('--ode_func_channels',
                    default=100,
                    type=int)
parser.add_argument('--out_channels',
                    default=1,
                    type=int)
parser.add_argument('--points_per_unit',
                    default=64,
                    type=int)
parser.add_argument('--margin',
                    default=1.0,
                    type=float)
parser.add_argument('--receptive_field',
                    default=1.0,
                    type=float)
parser.add_argument('--length_scale_multiplier',
                    default=2.0,
                    type=float)
parser.add_argument('--extrapolation',
                    action='store_true')
parser.add_argument('--min_x',
                    default=-2.0,
                    type=float)
parser.add_argument('--max_x',
                    default=2.0,
                    type=float)

args = parser.parse_args()

# Load working directory.
if args.root:
    wd = WorkingDirectory(root=args.root)
else:
    experiment_name = f'latent_ode-{args.data}'
    wd = WorkingDirectory(root=generate_root(experiment_name))

# Load data generator.
if args.data == 'sawtooth':
    gen = lib.data.SawtoothGenerator()
    gen_val = lib.data.SawtoothGenerator(num_tasks=60, 
                                         extrapolation=args.extrapolation)
    gen_test = lib.data.SawtoothGenerator(num_tasks=1000, 
                                          extrapolation=args.extrapolation, 
                                          batch_size=1)
    gen_plot = lib.data.SawtoothGenerator(num_tasks=1,
                                          batch_size=3,
                                          plot=True,
                                          extrapolation=args.extrapolation)
else:
    if args.data == 'eq':
        kernel = stheno.EQ().stretch(0.25)
    elif args.data == 'matern':
        kernel = stheno.Matern52().stretch(0.25)
    elif args.data == 'noisy-mixture':
        kernel = stheno.EQ().stretch(1.) + \
                 stheno.EQ().stretch(.25) + \
                 0.001 * stheno.Delta()
    elif args.data == 'weakly-periodic':
        kernel = stheno.EQ().stretch(0.5) * stheno.EQ().periodic(period=0.25)
    elif args.data == 'gauss-markov':
    	a_inv = 1
    	b_sq = 0.25
    	kernel = (a_inv*b_sq/2)*stheno.Exp().stretch(a_inv)
    else:
        raise ValueError(f'Unknown data "{args.data}".')

    gen = lib.data.GPGenerator(kernel=kernel)
    gen_val = lib.data.GPGenerator(kernel=kernel, 
                                   num_tasks=60, 
                                   extrapolation=args.extrapolation)
    gen_test = lib.data.GPGenerator(kernel=kernel, 
                                    num_tasks=1000,
                                    batch_size=1,  
                                    extrapolation=args.extrapolation)
    gen_plot = lib.data.GPGenerator(kernel=kernel, num_tasks=1, 
                                    batch_size=3, plot=True, 
                                    extrapolation=args.extrapolation)

# Load model.
model = get_net(args).to(device)
print(model)

losses = get_loss(args)

# Perform training.
opt = torch.optim.Adam(model.parameters(), 
                       args.learning_rate, 
                       weight_decay=args.weight_decay)

writer = SummaryWriter(f'logs/{args.root}')

print(f'Number of trainable parameters: {model.num_params}')

if args.train:
    # Run the training loop, maintaining the best objective value.
    best_obj = np.inf
    for epoch in range(args.epochs):
        print('\nEpoch: {}/{}'.format(epoch + 1, args.epochs))

        # Compute training objective.
        train_obj = train(gen,
                          model, 
                          losses, 
                          opt,
                          report_freq=50)
        report_loss('Training', train_obj['obj'], 'epoch')

        for loss in train_obj:
            writer.add_scalar(f'train_{loss}', train_obj[loss], epoch)

        # Compute validation objective.
        val_obj = validate(gen_val, model, losses, report_freq=20)

        report_loss('Validation', val_obj['nll'], epoch)

        for loss in val_obj:
            writer.add_scalar(f'val_{loss}', val_obj[loss], epoch)

        if gen_plot is not None:
            plot_model_task(model, gen_plot, epoch, wd)

        update_learning_rate(opt, decay_rate=0.999, lowest=args.learning_rate/10)

        # Update the best objective value and checkpoint the model.
        is_best = False
        if val_obj['nll'] < best_obj:
            best_obj = val_obj['nll']
            is_best = True
        save_checkpoint(wd,
                        {'epoch': epoch + 1,
                         'state_dict': model.state_dict(),
                         'best_acc_top1': best_obj,
                         'optimizer': opt.state_dict()},
                        is_best=is_best)

else:
    # Load saved model.
    load_dict = torch.load(wd.file('model_best.pth.tar', exists=True))
    model.load_state_dict(load_dict['state_dict'])

# Perform final quality validation
means = {}
for _ in range(10):
    loss_dict = validate(gen_test, 
                         model, 
                         losses, 
                         mode='mean')
    for name in loss_dict:
        if name not in means:
            means[name] = [loss_dict[name]]
        else:
            means[name] += [loss_dict[name]]

for name in means:
    print(name, np.array(means[name]).mean(), np.array(means[name]).std()) 

# print(means)
# loss_dict = validate(gen_test, model, losses, mode='mean')
# for loss in loss_dict:
#     print(f'Model averages a {loss} of {loss_dict[loss]} on unseen tasks.')
#     with open(wd.file(f'{loss}.txt'), 'w') as f:
#         f.write(str(loss_dict[loss]))
