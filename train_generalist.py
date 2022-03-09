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
from asteroid.losses.sdr import SingleSrcNegSDR as LossSDR

import _datasets as D
import _models as M


ROOT = os.path.dirname(os.path.realpath(__file__))
EARLY_STOPPING: int = 1000
DEFAULT_BATCH_SIZE: int = 128


def ray_train_generalist(config, use_ray: bool = True):

    print(config)

    # fix seed
    seed_everything(0)

    # select device
    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda:0'

    network = M.NetSE(config['hidden_size'], config['num_layers']).to(device)
    optimizer = torch.optim.Adam(network.parameters(), lr=config['learning_rate'])
    criterion = LossSDR('sisdr')

    data_tr = D.DatasetSE(
        speaker_ids=D.speaker_ids_tr,
        speech_subset='train',
        noise_subset='train',
        utterance_duration=config['utterance_duration'],
        mixture_snr=config['mixing_snr'])
    dl_tr = DataLoader(data_tr, batch_size=config['batch_size'])

    data_vl = D.DatasetSE(
        speaker_ids=D.speaker_ids_vl,
        speech_subset='train',
        noise_subset='train',
        utterance_duration=config['utterance_duration'],
        mixture_snr=config['mixing_snr'])
    (vx, vs, _) = next(iter(DataLoader(data_vl, batch_size=config['batch_size'])))
    vx = vx.to(device)
    vs = vs.to(device)


    # setup metrics + state dict
    (num_batches, current_epoch, best_epoch) = (0, 0, 0)
    best_result: float = 1e10
    best_sisdr: float = 0
    best_state_dict = None
    errors, denoising = [], []

    # loop indefinitely
    for (x, s, n) in dl_tr:

        current_epoch += 1
        num_batches += config['batch_size']

        optimizer.zero_grad()
        x = x.to(device)
        s = s.to(device)
        y = network(x)

        min_len = min(y.shape[-1], s.shape[-1])
        loss = criterion(y[..., :min_len], s[..., :min_len])
        loss.mean().backward()
        optimizer.step()

        # only validate every few epochs
        if current_epoch % 10:
            continue

        # validate
        with torch.no_grad():
            vy = network(vx)
            min_len = min(vy.shape[-1], vs.shape[-1])
            sisdr_in = criterion(vx[..., :min_len], vs[..., :min_len])
            sisdr_out = criterion(vy[..., :min_len], vs[..., :min_len])
            vl_loss = float(sisdr_out.mean())
            vl_sisdr = float((sisdr_in - sisdr_out).mean())
            errors.append(vl_loss)
            denoising.append(vl_sisdr)

        if (vl_loss < best_result):
            best_epoch = current_epoch
            best_result = vl_loss
            best_sisdr = vl_sisdr
            best_state_dict = network.state_dict()

        tune.report(num_batches=num_batches, vl_loss=best_result, vl_sisdr=best_sisdr)

        # check for convergence
        if current_epoch - best_epoch > EARLY_STOPPING:

            torch.cuda.empty_cache()

            if use_ray:
                # save checkpoint using Ray Tune
                with tune.checkpoint_dir(current_epoch) as checkpoint_dir:
                    path = os.path.join(checkpoint_dir, 'checkpoint')
                    torch.save(best_state_dict, path)
                    path = os.path.join(checkpoint_dir, 'errors.json')
                    with open(path, 'w') as fp:
                        json.dump(errors, fp)
                    path = os.path.join(checkpoint_dir, 'config.json')
                    with open(path, 'w') as fp:
                        json.dump(dict(
                            mixing_snr=config['mixing_snr'],
                            utterance_duration=int(config['utterance_duration']),
                            hidden_size=int(config['hidden_size']),
                            learning_rate=float(config['learning_rate']),
                            num_layers=int(config['num_layers']),
                            batch_size=int(config['batch_size']),
                            ), fp)
            break

    # run a test
    with torch.no_grad():
        seed_everything(0)
        data_te = D.DatasetSE(
            speaker_ids=D.speaker_ids_te,
            speech_subset='test',
            noise_subset='test',
            utterance_duration=5,
            mixture_snr=config['mixing_snr'])
        (x, s, _) = next(iter(DataLoader(data_te, batch_size=200)))
        x = x.to(device)
        s = s.to(device)
        y = network(x)
        min_len = min(y.shape[-1], s.shape[-1])
        sisdr_in = criterion(x[..., :min_len], s[..., :min_len])
        sisdr_out = criterion(y[..., :min_len], s[..., :min_len])
        te_sisdr = float((sisdr_in - sisdr_out).mean())

        if use_ray:
            print(te_sisdr)
            tune.report(te_sisdr=te_sisdr)
            with tune.checkpoint_dir(current_epoch) as checkpoint_dir:
                path = os.path.join(checkpoint_dir, 'test_sisdr.json')
                with open(path, 'w') as fp:
                    json.dump([te_sisdr], fp)

    print('exited train_specialist with args = {{' + \
        ', '.join([f'{k}={v}' for (k,v) in config.items()]) + '}}')
    return {
        # 'state_dict': best_state_dict,
        'num_batches': best_epoch * config['batch_size'],
        'vl_loss': best_result,
        'vl_sisdr': best_sisdr,
        'te_sisdr': te_sisdr,
    }


def find_learning_rates(num_gpus: float = .2, num_samples: int = 10):
    for hidden_size in [64]:
        config = {
            'mixing_snr': (-5, 10),
            'utterance_duration': 5,
            'num_layers': 2,
            'hidden_size': hidden_size,
            'learning_rate': tune.grid_search(
                        list(np.logspace(-4, -2, num_samples))),
            'batch_size': 128,
        }
        # scheduler = ASHAScheduler(
        #     metric='vl_loss',
        #     mode='min',
        #     reduction_factor=2
        # )
        scheduler = None
        result = tune.run(
            ray_train_generalist,
            name='ray_train_generalist_find_lr',
            config=config,
            keep_checkpoints_num=1,
            progress_reporter=CLIReporter(
                max_report_frequency=50,
                metric_columns=['num_batches', 'vl_sisdr', 'te_sisdr'],
                parameter_columns=['hidden_size', 'learning_rate'],
            ),
            scheduler=scheduler,
            resources_per_trial={'gpu': num_gpus},
            verbose=1
        )
        best_trial = result.get_best_trial(
            metric='vl_loss',
            mode='min',
            scope='last'
        )
        print('Best trial config: {}'.format(
            best_trial.config))
        print('Best trial final test results: {}'.format(
            best_trial.last_result['te_sisdr']))
        print('Best trial checkpoint path: {}'.format(
            os.path.join(best_trial.checkpoint.value, "checkpoint")))


if __name__ == '__main__':
    find_learning_rates()
