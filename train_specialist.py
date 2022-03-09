import os
import pathlib
import shutil
import json
import yaml
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


def ray_train_specialist(config, use_ray: bool = True):

    if config['specialist_index'] not in range(config['num_clusters']):
        return dict(num_batches=None, vl_loss=None, vl_sisdr=None)

    output_dir = None
    if config['save_to_local']:
        output_dir = os.path.join(
            ROOT, 'weights', 'specialists',
            'se-hs={:04d}_K={:02d}_k={:02d}'.format(
            config['hidden_size'], config['num_clusters'], config['specialist_index']))
        pathlib.Path(output_dir).mkdir(0o777, True, True)

    # find specialist mapping file
    mapping_filepath = os.path.join(
        ROOT, 'weights',
        'sv_mapping_K={:02d}.yaml'.format(config['num_clusters'])
    )
    with open(mapping_filepath, 'r') as fp:
        mapping = yaml.safe_load(fp)

    # grab a list of relevant speaker_ids
    relevant_speakers = sorted(mapping['specialists'][config['specialist_index']])

    # fix seed
    seed_everything(0)

    # select device
    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda:0'

    network = M.NetSE(config['hidden_size'], config['num_layers']).to(device)
    optimizer = torch.optim.Adam(network.parameters(),
                                 lr=config['learning_rate'])
    criterion = LossSDR('sisdr')

    data_tr = D.DatasetSE(relevant_speakers, 'train', 'train')
    dl_tr = DataLoader(data_tr, batch_size=DEFAULT_BATCH_SIZE)

    data_vl = D.DatasetSE(relevant_speakers, 'val', 'train')
    (vx, vs, _) = next(iter(DataLoader(data_vl, batch_size=DEFAULT_BATCH_SIZE)))
    vx = vx.to(device)
    vs = vs.to(device)

    print(config)

    # setup metrics + state dict
    (num_batches, current_epoch, best_epoch) = (0, 0, 0)
    best_result: float = 1e10
    best_sisdr: float = 0
    best_state_dict = None
    errors, denoising = [], []

    # loop indefinitely
    for (x, s, n) in dl_tr:

        current_epoch += 1
        num_batches += len(x)

        optimizer.zero_grad()
        x = x.to(device)
        s = s.to(device)
        y = network(x)

        min_len = min(y.shape[-1], s.shape[-1])
        loss = criterion(y[..., :min_len], s[..., :min_len]).mean()
        loss.backward()
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

        tune.report(
            num_batches=num_batches,
            vl_loss=best_result,
            vl_sisdr=best_sisdr
        )

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
                        json.dump(errors, fp, indent=2)
            break

    if output_dir:
        print('Config: {}'.format(config))
        print('Final Validation Loss: {}'.format(best_result))
        torch.save(best_state_dict, os.path.join(output_dir, 'checkpoint'))
        path = os.path.join(output_dir, 'config.json')
        with open(path, 'w') as fp:
            json.dump(config, fp, indent=2)
        path = os.path.join(output_dir, 'errors.json')
        with open(path, 'w') as fp:
            json.dump(errors, fp, indent=2)
        print('Model path: {}'.format(os.path.join(output_dir, 'checkpoint')))


    print('exited train_specialist with args = {{' + \
        ', '.join([f'{k}={v}' for (k,v) in config.items()]) + '}}')
    return {
        # 'state_dict': best_state_dict,
        'num_batches': best_epoch * DEFAULT_BATCH_SIZE,
        'vl_loss': best_result,
        'vl_sisdr': best_sisdr,
    }


def main_old(num_gpus: float = .33, num_samples: int = 12):

    # sweep hyperparameter space
    for hidden_size in [384, 512, 640, 768, 896, 1024, 1152, 1280]:
        for num_clusters in [2, 5, 10]:
            for specialist_index in range(num_clusters):

                output_dir = os.path.join(
                    'weights', 'specialists',
                    'se-hs={:04d}_K={:02d}_k={:02d}'.format(
                    hidden_size, num_clusters, specialist_index))
                pathlib.Path(output_dir).mkdir(0o777, True, True)

                if os.path.exists(os.path.join(output_dir, 'checkpoint')):
                    print(f'Skipping {output_dir}...')
                    continue

                config = {
                    'hidden_size': hidden_size,
                    'num_layers': 2,
                    'mixing_snr': (-5, 10),
                    'num_clusters': num_clusters,
                    'specialist_index': specialist_index,
                    'learning_rate': tune.grid_search(
                        list(np.logspace(-4, -2, num_samples))),
                }
                result = tune.run(
                    ray_train_specialist,
                    config=config,
                    keep_checkpoints_num=1,
                    progress_reporter=CLIReporter(
                        max_report_frequency=60,
                        metric_columns=[
                            'num_batches', 'vl_sisdr'
                        ],
                        parameter_columns=[
                            'num_clusters', 'specialist_index',
                            'hidden_size', 'learning_rate'
                        ],
                    ),
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
                print('Best trial final validation loss: {}'.format(
                    best_trial.last_result['vl_loss']))

                f_src = os.path.join(best_trial.checkpoint.value, 'checkpoint')
                print('Best trial checkpoint path: {}'.format(f_src))
                f_dest = os.path.join(output_dir, 'checkpoint')
                shutil.copy2(f_src, f_dest)
                print('Copied best checkpoint to: {}'.format(f_dest))
                f_dest = os.path.join(output_dir, 'config.json')
                with open(f_dest, 'w') as fp:
                    json.dump(best_trial.config, fp, indent=2)


def main(num_gpus: float = .33):

    config = {
        'save_to_local': True,
        'hidden_size': tune.grid_search([1152, 1280]),
        'num_layers': 2,
        'mixing_snr': (-5, 10),
        'num_clusters': tune.grid_search([2, 5, 10]),
        'specialist_index': tune.grid_search(list(range(10))[::-1]),
        'learning_rate': 1e-3,
    }
    tune.run(
        ray_train_specialist,
        config=config,
        keep_checkpoints_num=1,
        progress_reporter=CLIReporter(
            max_report_frequency=60,
            metric_columns=[
                'num_batches', 'vl_sisdr'
            ],
            parameter_columns=[
                'num_clusters', 'specialist_index',
                'hidden_size', 'learning_rate'
            ],
        ),
        resources_per_trial={'gpu': num_gpus},
        verbose=1
    )

if __name__ == '__main__':
    main()
