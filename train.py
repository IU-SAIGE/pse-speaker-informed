import json
import os
import pathlib
import shutil
import stat
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
import _models2 as M


ROOT = os.path.dirname(os.path.realpath(__file__))
EARLY_STOPPING: int = 1000
DEFAULT_BATCH_SIZE: int = 128


def copytree(src, dst, symlinks = False, ignore = None):
    if not os.path.exists(dst):
        os.makedirs(dst)
        shutil.copystat(src, dst)
    lst = os.listdir(src)
    if ignore:
        excl = ignore(src, lst)
        lst = [x for x in lst if x not in excl]
    for item in lst:
        s = os.path.join(src, item)
        d = os.path.join(dst, item)
        if symlinks and os.path.islink(s):
            if os.path.lexists(d):
                os.remove(d)
            os.symlink(os.readlink(s), d)
            try:
                st = os.lstat(s)
                mode = stat.S_IMODE(st.st_mode)
                os.lchmod(d, mode)
            except:
                pass # lchmod not available
        elif os.path.isdir(s):
            copytree(s, d, symlinks, ignore)
        else:
            shutil.copy2(s, d)



def run_test(network: torch.nn.Module):

    device = ('cuda:0' if torch.cuda.is_available() else 'cpu')
    criterion = LossSDR('sisdr')

    with torch.no_grad():
        network.eval()
        seed_everything(0)
        data_te = D.DatasetSE(D.speaker_ids_te, 'test', 'test')
        (x, s, _) = next(iter(DataLoader(data_te, batch_size=200)))
        x = x.to(device)
        s = s.to(device)
        y = network(x)
        ml = min(y.shape[-1], s.shape[-1])
        sisdr_in = criterion(x[..., :ml], s[..., :ml])
        sisdr_out = criterion(y[..., :ml], s[..., :ml])
        te_sisdr = float((sisdr_in - sisdr_out).mean())

    return te_sisdr


def ray_train_ensemble(config, use_ray: bool = True):


    # fix seed
    seed_everything(0)

    # select device
    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda:0'

    network = M.Ensemble(
        num_specialists=config['num_clusters'],
        specialist_hidden_size=config['specialist_size']
    ).to(device)
    # network.load_gating_network(
    #     ROOT + '/weights/sv'
    # )
    # network.load_speaker_vectors(
    #     ROOT + '/weights/sv_mapping_K={:02d}.yaml'.format(config['num_clusters']),
    #     ROOT + '/weights/sv_speakers.npy'
    # )
    # network.load_specialist_networks([
    #     ROOT + '/weights/specialists/se-hs={:04d}_K={:02d}_k={:02d}/checkpoint'.format(
    #         config['specialist_size'], config['num_clusters'], i
    #     ) for i in range(config['num_clusters'])
    # ])

    te_sisdr_before = run_test(network)
    print('w/o finetuning =', te_sisdr_before, 'dB')

    if not config['finetune']:
        return dict(num_batches=0, vl_loss=None, vl_sisdr=None,
                    te_sisdr_before=te_sisdr_before)

    print(config)

    optimizer = torch.optim.Adam(network.parameters(), lr=config['learning_rate'])
    criterion = LossSDR('sisdr')

    data_tr = D.DatasetSE(D.speaker_ids_tr, 'train', 'train')
    dl_tr = DataLoader(data_tr, batch_size=DEFAULT_BATCH_SIZE)

    data_vl = D.DatasetSE(D.speaker_ids_vl, 'train', 'train')
    (vx, vs, _) = next(iter(DataLoader(data_vl, batch_size=DEFAULT_BATCH_SIZE)))
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

        network.train()
        current_epoch += 1
        num_batches += DEFAULT_BATCH_SIZE

        optimizer.zero_grad()
        x = x.to(device).squeeze()
        s = s.to(device).squeeze()
        y = network(x).squeeze()

        min_len = min(y.shape[-1], s.shape[-1])
        loss = criterion(y[..., :min_len], s[..., :min_len])
        loss.mean().backward()
        optimizer.step()

        # only validate every few epochs
        if current_epoch % 10:
            continue

        # validate
        with torch.no_grad():
            network.eval()
            vy = network(vx).squeeze()
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
        if (current_epoch - best_epoch > EARLY_STOPPING) or num_batches > 1e6:

            torch.cuda.empty_cache()

            if use_ray:
                # save checkpoint using Ray Tune
                with tune.checkpoint_dir(current_epoch) as checkpoint_dir:
                    path = os.path.join(checkpoint_dir, 'checkpoint')
                    torch.save(best_state_dict, path)
                    path = os.path.join(checkpoint_dir, 'config.json')
                    with open(path, 'w') as fp:
                        json.dump(config, fp, indent=2)
                    path = os.path.join(checkpoint_dir, 'errors.json')
                    with open(path, 'w') as fp:
                        json.dump(errors, fp, indent=2)
            break

    # run a test
    te_sisdr_after = run_test(network)
    print('w/ finetuning =', te_sisdr_after, 'dB')

    if use_ray:
        # save checkpoint using Ray Tune
        with tune.checkpoint_dir(current_epoch) as checkpoint_dir:
            path = os.path.join(checkpoint_dir, 'test_results.json')
            with open(path, 'w') as fp:
                json.dump({
                    'te_sisdr_before': te_sisdr_before,
                    'te_sisdr_after': te_sisdr_after,
                }, fp, indent=2)

    print('exited train_ensemble with args = {{' + \
        ', '.join([f'{k}={v}' for (k,v) in config.items()]) + '}}')
    return {
        # 'state_dict': best_state_dict,
        'num_batches': best_epoch * DEFAULT_BATCH_SIZE,
        'vl_loss': best_result,
        'vl_sisdr': best_sisdr,
        'te_sisdr_before': te_sisdr_before,
        'te_sisdr_after': te_sisdr_after,
    }


def ray_train_classifier(config, use_ray: bool = True):


    # fix seed
    seed_everything(0)

    # select device
    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda:0'

    gating = M.NetSV(32, 2).to(device)
    gating.load_state_dict(torch.load(ROOT + '/weights/sv'), strict=True)
    gating.train(False)
    for p in gating.parameters():
        p.requires_grad = False
    classifier = M.NetClassifier(config['num_clusters']).to(device)

    print(config)

    optimizer = torch.optim.Adam(classifier.parameters(), lr=config['learning_rate'])
    criterion = torch.nn.BCEWithLogitsLoss(size_average=True)

    data_tr = D.DatasetClassification(D.speaker_ids_tr,
        ROOT + '/weights/sv_mapping_K={:02d}.yaml'.format(config['num_clusters']),
        'train', 'train')
    dl_tr = DataLoader(data_tr, batch_size=DEFAULT_BATCH_SIZE)

    data_vl = D.DatasetClassification(D.speaker_ids_tr,
        ROOT + '/weights/sv_mapping_K={:02d}.yaml'.format(config['num_clusters']),
        'test', 'train')
    (vx, vid) = next(iter(DataLoader(data_vl, batch_size=DEFAULT_BATCH_SIZE)))
    vx = vx.to(device)
    vid = vid.to(device)

    # setup metrics + state dict
    (num_batches, current_epoch, best_epoch) = (0, 0, 0)
    best_result: float = 1e10
    best_state_dict = None
    errors = []

    # loop indefinitely
    for (x, tid) in dl_tr:

        current_epoch += 1
        num_batches += DEFAULT_BATCH_SIZE

        optimizer.zero_grad()
        x = x.to(device).squeeze()
        tid = tid.to(device).squeeze()
        yid = classifier(gating.embedding(x).detach()).squeeze()

        loss = criterion(yid, tid)
        loss.backward()
        optimizer.step()

        # only validate every few epochs
        if current_epoch % 10:
            continue

        # validate
        with torch.no_grad():
            vyid = classifier(gating.embedding(vx).detach()).squeeze()
            vl_loss = float(criterion(vyid, vid))
            errors.append(vl_loss)

        if (vl_loss < best_result):
            best_epoch = current_epoch
            best_result = vl_loss
            accuracy = float((torch.argmax(vyid, dim=-1) == torch.argmax(vid, dim=-1)).float().mean())
            best_state_dict = classifier.state_dict()

        if use_ray:
            tune.report(num_batches=num_batches, vl_loss=best_result, vl_acc=accuracy)

        # check for convergence
        if (current_epoch - best_epoch > EARLY_STOPPING) or num_batches > 1e6:

            torch.cuda.empty_cache()

            if use_ray:
                # save checkpoint using Ray Tune
                with tune.checkpoint_dir(current_epoch) as checkpoint_dir:
                    path = os.path.join(checkpoint_dir, 'checkpoint')
                    torch.save(best_state_dict, path)
                    path = os.path.join(checkpoint_dir, 'config.json')
                    with open(path, 'w') as fp:
                        json.dump(config, fp, indent=2)
                    path = os.path.join(checkpoint_dir, 'errors.json')
                    with open(path, 'w') as fp:
                        json.dump(errors, fp, indent=2)
            break

    print('exited train_classifier with args = {{' + \
        ', '.join([f'{k}={v}' for (k,v) in config.items()]) + '}}')
    return {
        # 'state_dict': best_state_dict,
        'num_batches': best_epoch * DEFAULT_BATCH_SIZE,
        'vl_loss': best_result,
        'vl_acc': accuracy,
    }


def main_old(
        num_clusters: int,
        specialist_size: int,
        num_gpus: float = .5,
        num_samples: int = 10
    ):
    config = {
        'save_to_local': True,
        'finetune': True,
        'num_clusters': num_clusters,
        'specialist_size': specialist_size,
        'specialist_layers': 2,
        'learning_rate': tune.grid_search(
                        list(np.logspace(-5, -2, num_samples))),
    }
    if config['finetune']:

        output_dir = None
        if config['save_to_local']:
            output_dir = os.path.join(
                ROOT, 'weights', 'ensemble',
                'se-hs={:04d}_K={:02d}'.format(
                config['specialist_size'], config['num_clusters']))
            pathlib.Path(output_dir).mkdir(0o777, True, True)

        result = tune.run(
            ray_train_ensemble,
            config=config,
            keep_checkpoints_num=1,
            progress_reporter=CLIReporter(
                max_report_frequency=60,
                metric_columns=[
                    'num_batches', 'vl_sisdr', 'te_sisdr_before', 'te_sisdr_after'
                ],
                parameter_columns=[
                    'num_clusters', 'specialist_size', 'learning_rate'
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
        print('Best trial validation loss: {}'.format(
            best_trial.last_result['vl_sisdr']))
        print('Best trial final test results: {}'.format(
            best_trial.last_result['te_sisdr_after']))
        print('Best trial checkpoint path: {}'.format(
            os.path.join(best_trial.checkpoint.value, 'checkpoint')))

        if output_dir:
            copytree(best_trial.checkpoint.value, output_dir)
    else:
        ray_train_ensemble(config)


def main_ensemble(num_gpus: float = 1):
    config = {
        'save_to_local': False,
        'finetune': True,
        'num_clusters': tune.grid_search([10, 5, 2]),
        'specialist_size': tune.grid_search([64, 128, 256, 384, 512]),
        # 'specialist_size': tune.grid_search([64, 128, 256, 384, 512, 640, 768, 896, 1024, 1152, 1280]),
        'specialist_layers': 2,
        'learning_rate': 1e-4,
    }
    result = tune.run(
        ray_train_ensemble,
        config=config,
        keep_checkpoints_num=1,
        progress_reporter=CLIReporter(
            max_report_frequency=30,
            metric_columns=[
                'num_batches', 'vl_sisdr', 'te_sisdr_before', 'te_sisdr_after'
            ],
            parameter_columns=[
                'num_clusters', 'specialist_size', 'learning_rate'
            ],
        ),
        resources_per_trial={'gpu': num_gpus},
        verbose=1
    )


def main_classifier(num_gpus: float = 0.2):
    config = {
        'save_to_local': False,
        'num_clusters': tune.grid_search([10, 5, 2]),
        'learning_rate': tune.grid_search([5e-4, 2e-4, 1e-3, 1e-4]),
    }
    result = tune.run(
        ray_train_classifier,
        config=config,
        keep_checkpoints_num=1,
        progress_reporter=CLIReporter(
            max_report_frequency=30,
            metric_columns=[
                'num_batches', 'vl_loss', 'vl_acc'
            ],
            parameter_columns=[
                'num_clusters', 'learning_rate'
            ],
        ),
        resources_per_trial={'gpu': num_gpus},
        verbose=1
    )



if __name__ == '__main__':
    main_ensemble()
    # for i in [2, 5, 10]:
    #     for j in [128, 256, 384, 512, 640, 768, 896, 1024, 1152, 1280]:
    #         print(dict(num_clusters=i, specialist_size=j)); \
    #               main_old(num_clusters=i, specialist_size=j)
