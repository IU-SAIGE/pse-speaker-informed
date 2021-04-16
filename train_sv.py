import os
import pathlib
import json
from typing import Optional, Sequence, Tuple, Union, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.io import wavfile
from pytorch_lightning import seed_everything
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune import CLIReporter
from torch.utils.data import DataLoader

import _datasets as D
import _models as M

EPS = 1e-8

def ray_train_sv(config, use_ray: bool = True):

    print(config)

    # fix seed
    seed_everything(0)

    # select device
    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda:0'

    network = M.NetSV(config['hidden_size'], config['num_layers']).to(device)
    optimizer = torch.optim.Adam(network.parameters(), lr=config['learning_rate'])
    criterion = torch.nn.BCEWithLogitsLoss(size_average=True)

    data_tr = D.MyDataset(D.speaker_ids_tr, 'free-sound', config['utterance_duration'], config['mixing_snr'])
    dl_tr = DataLoader(data_tr, batch_size=config['batch_size'])

    data_vl = D.MyDataset(D.speaker_ids_vl, 'free-sound', config['utterance_duration'], config['mixing_snr'])
    (vx_1, vx_2, vy) = next(iter(DataLoader(data_vl, batch_size=config['batch_size'])))
    vx_1 = vx_1.to(device)
    vx_2 = vx_2.to(device)
    vy = vy.to(device)


    # setup metrics + state dict
    (num_batches, current_epoch, best_epoch) = (0, 0, 0)
    best_result: float = 1e10
    best_accuracy: float = 0.
    best_state_dict = None
    errors, accuracies = [], []

    # loop indefinitely
    for (x_1, x_2, y) in dl_tr:

        current_epoch += 1
        num_batches += config['batch_size']

        optimizer.zero_grad()
        x_1 = x_1.to(device)
        x_2 = x_2.to(device)
        y = y.to(device)
        y_hat = network(x_1, x_2)

        loss = criterion(y_hat, y.squeeze())
        loss.backward()
        optimizer.step()

        # only validate every few epochs
        if current_epoch % 10:
            continue

        # validate
        with torch.no_grad():
            vy_hat = network(vx_1, vx_2)
            vl_loss = float(criterion(vy_hat, vy.squeeze()).item())
            vl_acc = float((torch.round(vy_hat) == vy).float().mean().item())
            errors.append(vl_loss)
            accuracies.append(vl_acc)

        if (current_epoch == 10) or (vl_loss < best_result):
            best_epoch = current_epoch
            best_result = vl_loss
            best_accuracy = vl_acc
            best_state_dict = network.state_dict()

            if use_ray:
                # save checkpoint using Ray Tune
                with tune.checkpoint_dir(current_epoch) as checkpoint_dir:
                    path = os.path.join(checkpoint_dir, 'checkpoint')
                    torch.save(network.state_dict(), path)
                    path = os.path.join(checkpoint_dir, 'errors.json')
                    with open(path, 'w') as fp:
                        json.dump(errors, fp)
                    path = os.path.join(checkpoint_dir, 'accuracies.json')
                    with open(path, 'w') as fp:
                        json.dump(accuracies, fp)

        tune.report(num_batches=num_batches, vl_loss=best_result, vl_acc=best_accuracy)

        # check for convergence
        if current_epoch - best_epoch > 1000:
            break

    print('exited train_sv with args = {{' + \
        ', '.join([f'{k}={v}' for (k,v) in config.items()]) + '}}')
    return {
        'state_dict': best_state_dict,
        'num_batches': best_epoch * config['batch_size'],
        'vl_loss': best_result,
        'vl_acc': best_accuracy,
    }


def main(num_gpus: float = .25):
    config = {
        'mixing_snr': (-5, 10),
        'utterance_duration': tune.grid_search([5, 3, 1]),
        'hidden_size': tune.grid_search([32, 64, 96]),
        'learning_rate': 1e-3, # tune.grid_search([1e-2, 1e-3, 1e-4]),
        'num_layers': 2,
        'batch_size': 128,
    }
    # scheduler = ASHAScheduler(
    #     metric='vl_loss',
    #     mode='min',
    #     max_t=10000,
    #     grace_period=100,
    #     reduction_factor=2
    # )
    scheduler = None
    result = tune.run(
        ray_train_sv,
        config=config,
        keep_checkpoints_num=1,
        log_to_file='log.txt',
        progress_reporter=CLIReporter(
            max_report_frequency=10,
            metric_columns=['num_batches', 'vl_loss', 'vl_acc'],
            parameter_columns=['hidden_size', 'utterance_duration', 'learning_rate'],
        ),
        resources_per_trial={'gpu': num_gpus},
        scheduler=scheduler,
        verbose=1
    )
    best_trial = result.get_best_trial(
        metric='vl_loss',
        mode='min',
        scope='last'
    )
    print('Best trial config: {}'.format(
        best_trial.config))
    print('Best trial final validation loss: {}'.format(
        best_trial.last_result['vl_loss']))


if __name__ == '__main__':
    main()
