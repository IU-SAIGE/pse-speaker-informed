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


EPS = 1e-8

_df_types = dict(
    channel=str, chapter_id=str, clip_id=str, data_type=str, duration=float,
    is_sparse=bool, set_id=str, speaker_id=str, utterance_id=str,
    freesound_id=str,
)

max_duration: int = 5  # seconds
sample_rate: int = 8000  # Hz


def create_df_librispeech(
    root_directory: str,
    csv_path: str = 'corpora/librispeech.csv'
):
    """Creates a Pandas DataFrame with files from the LibriSpeech corpus.
    Root directory should mimic archive-extracted folder structure.
    Dataset may be downloaded at `<https://www.openslr.org/12/>`_.
    """
    assert os.path.isdir(root_directory)
    df = pd.read_csv(csv_path, dtype=_df_types)
    df = df[(df.set_id == 'train-clean-100')
            & (df.duration > max_duration)]
    df.loc[:, 'max_offset'] = (df.duration - max_duration) * sample_rate
    df.loc[:, 'max_offset'] = df['max_offset'].astype(int)
    df.loc[:, 'split'] = 'pretrain'
    for speaker_id in df.speaker_id.unique():
        _mask = (df['speaker_id'] == speaker_id)
        _last_row = _mask[::-1].idxmax()
        df.loc[_last_row-25:_last_row-20, 'split'] = 'preval'
        df.loc[_last_row-20:_last_row-10, 'split'] = 'train'
        df.loc[_last_row-10:_last_row-5, 'split'] = 'val'
        df.loc[_last_row-5:_last_row, 'split'] = 'test'
    df.loc[:, 'filepath'] = (
        root_directory + '/' + df.set_id + '/' + df.speaker_id + '/'
        + df.chapter_id + '/' + df.speaker_id + '-' + df.chapter_id
        + '-' + df.utterance_id + '.wav'
    )
    # assert all(df.filepath.apply(os.path.isfile))
    return df


def create_df_musan(
    root_directory: str,
    csv_path: str = 'corpora/musan.csv'
):
    """Creates a Pandas DataFrame with files from the MUSAN corpus.
    Root directory should mimic archive-extracted folder structure.
    Dataset may be downloaded at `<https://www.openslr.org/17/>`_.
    """
    assert os.path.isdir(root_directory)
    df = pd.read_csv(csv_path, dtype=_df_types)
    df = df[df.duration > max_duration]
    df = df.sample(frac=1, random_state=0)
    df.loc[:, 'max_offset'] = (df.duration - max_duration) * sample_rate
    df.loc[:, 'max_offset'] = df['max_offset'].astype(int)
    df.loc[:, 'filepath'] = (
        root_directory + '/' + df.data_type + '/' + df.set_id + '/'
        + df.data_type + '-' + df.set_id + '-' + df.clip_id + '.wav'
    )
    df.loc[:, 'split'] = df.set_id
    df.split = df.split.replace({'free-sound': 'train', 'sound-bible': 'test'})
    # assert all(df.filepath.apply(os.path.isfile))
    return df


class MyDataset(torch.utils.data.IterableDataset):

    def __init__(
            self,
            speaker_ids: List[str],
            noise_subset: str = 'free-sound',
            utterance_duration: Optional[int] = 3,
            mixture_snr: Union[float, Tuple[float, float]] = (-5, 5)
    ):
        super().__init__()
        self.rng = np.random.default_rng(0)
        self.speaker_ids = speaker_ids
        if isinstance(mixture_snr, Tuple):
            self.mixture_snr_min = min(mixture_snr)
            self.mixture_snr_max = max(mixture_snr)
        else:
            self.mixture_snr_min = self.mixture_snr_max = mixture_snr
        self.df_s = librispeech.query(f'speaker_id in {speaker_ids}')
        self.df_n = musan.query(f'set_id == "{noise_subset}"')
        self.utterance_duration = utterance_duration

    def __iter__(self):
        return self

    def __next__(self):

        length = self.utterance_duration * sample_rate
        is_same_speaker = bool(round(self.rng.uniform()))

        if is_same_speaker:

            # pick speaker IDs
            sp_id_1 = self.rng.choice(self.speaker_ids)
            df = self.df_s.query(f'speaker_id == "{sp_id_1}"').sample(
                n=2, random_state=self.rng.bit_generator
            )
            sp_1 = df.iloc[0]
            sp_2 = df.iloc[1]

            # pick random offset
            sp_1_offset = self.rng.integers(0, sp_1.max_offset)
            sp_2_offset = self.rng.integers(0, sp_2.max_offset)

        else:

            # pick speaker IDs
            (sp_id_1,
            sp_id_2) = self.rng.choice(self.speaker_ids, size=2, replace=False)
            sp_1 = self.df_s.query(f'speaker_id == "{sp_id_1}"').sample(
                n=1, random_state=self.rng.bit_generator
            ).iloc[0]
            sp_2 = self.df_s.query(f'speaker_id == "{sp_id_2}"').sample(
                n=1, random_state=self.rng.bit_generator
            ).iloc[0]

            # pick random offset
            sp_1_offset = self.rng.integers(0, sp_1.max_offset)
            sp_2_offset = self.rng.integers(0, sp_2.max_offset)

        # read utterances
        (_, s_1) = wavfile.read(sp_1.filepath)
        s_1 = s_1[sp_1_offset:sp_1_offset+length]
        s_1 = s_1 / (EPS + s_1.std())

        (_, s_2) = wavfile.read(sp_2.filepath)
        s_2 = s_2[sp_2_offset:sp_2_offset+length]
        s_2 = s_2 / (EPS + s_2.std())

        # read noises
        df = self.df_n.sample(n=2, random_state=self.rng.bit_generator)
        no_1 = df.iloc[0]
        no_2 = df.iloc[1]
        no_1_offset = self.rng.integers(0, no_1.max_offset)
        no_2_offset = self.rng.integers(0, no_2.max_offset)

        (_, n_1) = wavfile.read(no_1.filepath)
        n_1 = n_1[no_1_offset:no_1_offset+length]
        n_1 = n_1 / (EPS + n_1.std())

        (_, n_2) = wavfile.read(no_2.filepath)
        n_2 = n_2[no_2_offset:no_2_offset+length]
        n_2 = n_2 / (EPS + n_2.std())

        # mix the signals
        snr = self.rng.uniform(self.mixture_snr_min, self.mixture_snr_max)
        x_1 = s_1 + (n_1 * 10**(-snr/20.))
        snr = self.rng.uniform(self.mixture_snr_min, self.mixture_snr_max)
        x_2 = s_2 + (n_2 * 10**(-snr/20.))

        # create output tuple
        scale_factor = EPS + max(abs(x_1).max(), abs(x_2).max())
        sample = (
            torch.Tensor(x_1) / scale_factor,
            torch.Tensor(x_2) / scale_factor,
            torch.Tensor([1. if is_same_speaker else 0.]),
        )

        return sample


librispeech = create_df_librispeech('/media/sdc1/librispeech_8khz/')
musan = create_df_musan('/media/sdc1/musan_8khz/')

speakers_vl = pd.read_csv('speakers/validation.csv', dtype=_df_types)
speakers_te = pd.read_csv('speakers/test.csv', dtype=_df_types)
speaker_ids_vl = set(speakers_vl.speaker_id)
speaker_ids_te = set(speakers_te.speaker_id)
speaker_ids_tr = set(librispeech.speaker_id) - speaker_ids_vl - speaker_ids_te
speaker_ids_vl = sorted(speaker_ids_vl)
speaker_ids_te = sorted(speaker_ids_te)
speaker_ids_tr = sorted(speaker_ids_tr)
