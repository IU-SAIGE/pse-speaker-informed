import os
import pathlib
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


_df_types = dict(
    channel=str, chapter_id=str, clip_id=str, data_type=str, duration=float,
    is_sparse=bool, set_id=str, speaker_id=str, utterance_id=str,
    freesound_id=str,
)

EPS = 1e-8
ROOT = os.path.dirname(os.path.realpath(__file__))
DEFAULT_UTTERANCE_DURATION: int = 5  # seconds
DEFAULT_MIXTURE_SNRS: Union[float, Tuple[float, float]] = (-5, 10) # dB
DEFAULT_SAMPLE_RATE: int = 8000  # Hz


def create_df_librispeech(
    root_directory: str,
    csv_path: str = os.path.join(ROOT, 'corpora', 'librispeech.csv'),
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    max_duration: int = DEFAULT_UTTERANCE_DURATION,
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
    df.loc[:, 'split'] = 'train'
    for speaker_id in df.speaker_id.unique():
        _mask = (df['speaker_id'] == speaker_id)
        _last_row = _mask[::-1].idxmax()
        df.loc[_last_row-20:_last_row-10, 'split'] = 'val'
        df.loc[_last_row-10:_last_row, 'split'] = 'test'
    df.loc[:, 'filepath'] = (
        root_directory + '/' + df.set_id + '/' + df.speaker_id + '/'
        + df.chapter_id + '/' + df.speaker_id + '-' + df.chapter_id
        + '-' + df.utterance_id + '.wav'
    )
    df = df.sample(frac=1, random_state=0)
    assert all(df.filepath.apply(os.path.isfile))
    return df


def create_df_musan(
    root_directory: str,
    csv_path: str = os.path.join(ROOT, 'corpora', 'musan.csv'),
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    max_duration: int = DEFAULT_UTTERANCE_DURATION,
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
    assert all(df.filepath.apply(os.path.isfile))
    return df


class DatasetSV(torch.utils.data.IterableDataset):

    def __init__(
            self,
            speaker_ids: List[str],
            speech_subset: str = 'train',
            noise_subset: str = 'train',
            mixture_snr: Union[float, Tuple[float, float]] = DEFAULT_MIXTURE_SNRS,
            sample_rate: int = DEFAULT_SAMPLE_RATE,
            utterance_duration: int = DEFAULT_UTTERANCE_DURATION,
    ):
        super().__init__()
        self.rng = np.random.default_rng(0)
        self.speaker_ids = speaker_ids
        if isinstance(mixture_snr, Tuple):
            self.mixture_snr_min = min(mixture_snr)
            self.mixture_snr_max = max(mixture_snr)
        else:
            self.mixture_snr_min = self.mixture_snr_max = mixture_snr
        self.df_s = librispeech.query(f'speaker_id in {speaker_ids}'
        	).query(f'split == "{speech_subset}"')
        self.df_n = musan.query(f'split == "{noise_subset}"')

        self.num_samples = int(sample_rate * utterance_duration)

    def __iter__(self):
        return self

    def __next__(self):

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
        s_1 = s_1[sp_1_offset:sp_1_offset+self.num_samples]
        s_1 = s_1 / (EPS + s_1.std())

        (_, s_2) = wavfile.read(sp_2.filepath)
        s_2 = s_2[sp_2_offset:sp_2_offset+self.num_samples]
        s_2 = s_2 / (EPS + s_2.std())

        # read noises
        df = self.df_n.sample(n=2, random_state=self.rng.bit_generator)
        no_1 = df.iloc[0]
        no_2 = df.iloc[1]
        no_1_offset = self.rng.integers(0, no_1.max_offset)
        no_2_offset = self.rng.integers(0, no_2.max_offset)

        (_, n_1) = wavfile.read(no_1.filepath)
        n_1 = n_1[no_1_offset:no_1_offset+self.num_samples]
        n_1 = n_1 / (EPS + n_1.std())

        (_, n_2) = wavfile.read(no_2.filepath)
        n_2 = n_2[no_2_offset:no_2_offset+self.num_samples]
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


class DatasetSE(torch.utils.data.IterableDataset):

    def __init__(
            self,
            speaker_ids: List[str],
            speech_subset: str = 'train',
            noise_subset: str = 'train',
            mixture_snr: Union[float, Tuple[float, float]] = DEFAULT_MIXTURE_SNRS,
            sample_rate: int = DEFAULT_SAMPLE_RATE,
            utterance_duration: int = DEFAULT_UTTERANCE_DURATION,
    ):
        super().__init__()
        self.rng = np.random.default_rng(0)
        (self.s_idx, self.n_idx) = (-1, -1)
        self.speaker_ids = speaker_ids
        if isinstance(mixture_snr, Tuple):
            self.mixture_snr_min = min(mixture_snr)
            self.mixture_snr_max = max(mixture_snr)
        else:
            self.mixture_snr_min = self.mixture_snr_max = mixture_snr
        self.df_s = librispeech.query(f'speaker_id in {speaker_ids}'
        	).query(f'split == "{speech_subset}"')
        self.df_n = musan.query(f'split == "{noise_subset}"')

        self.num_samples = int(sample_rate * utterance_duration)

    def __iter__(self):
        return self

    def __next__(self):

        # increment pointers
        self.s_idx = (self.s_idx + 1) % len(self.df_s)
        self.n_idx = (self.n_idx + 1) % len(self.df_n)

        offset_s = self.rng.integers(0, self.df_s.max_offset.iloc[self.s_idx])
        offset_n = self.rng.integers(0, self.df_n.max_offset.iloc[self.n_idx])

        # read speech file, offset and truncate, then normalize
        (_, s) = wavfile.read(self.df_s.filepath.iloc[self.s_idx])
        s = s[offset_s:offset_s+self.num_samples]
        s = s / s.std()

        # read noise file, offset and truncate, then normalize
        (_, n) = wavfile.read(self.df_n.filepath.iloc[self.n_idx])
        n = n[offset_n:offset_n+self.num_samples]
        n = n / n.std()

        # mix the signals
        snr = self.rng.uniform(self.mixture_snr_min, self.mixture_snr_max)
        x = s + (n * 10**(-snr/20.))

        # create output tuple
        scale_factor = abs(x).max()
        sample = (
            torch.Tensor(x) / scale_factor,
            torch.Tensor(s) / scale_factor,
            torch.Tensor(n) / scale_factor,
        )

        return sample


class DatasetClassification(torch.utils.data.IterableDataset):

    def __init__(
            self,
            speaker_ids: List[str],
            mapping_filepath: str,
            speech_subset: str = 'train',
            noise_subset: str = 'train',
            mixture_snr: Union[float, Tuple[float, float]] = DEFAULT_MIXTURE_SNRS,
            sample_rate: int = DEFAULT_SAMPLE_RATE,
            utterance_duration: int = DEFAULT_UTTERANCE_DURATION,
    ):
        super().__init__()
        self.rng = np.random.default_rng(0)
        (self.s_idx, self.n_idx) = (-1, -1)
        self.speaker_ids = speaker_ids
        if isinstance(mixture_snr, Tuple):
            self.mixture_snr_min = min(mixture_snr)
            self.mixture_snr_max = max(mixture_snr)
        else:
            self.mixture_snr_min = self.mixture_snr_max = mixture_snr
        self.df_s = librispeech.query(f'speaker_id in {speaker_ids}'
            ).query(f'split == "{speech_subset}"')
        self.df_n = musan.query(f'split == "{noise_subset}"')

        self.num_samples = int(sample_rate * utterance_duration)
        with open(mapping_filepath, 'r') as fp:
            self.mapping = yaml.safe_load(fp)
        self.num_clusters = len(list(self.mapping['specialists'].keys()))

    def __iter__(self):
        return self

    def __next__(self):

        # increment pointers
        self.s_idx = (self.s_idx + 1) % len(self.df_s)
        self.n_idx = (self.n_idx + 1) % len(self.df_n)

        offset_s = self.rng.integers(0, self.df_s.max_offset.iloc[self.s_idx])
        offset_n = self.rng.integers(0, self.df_n.max_offset.iloc[self.n_idx])

        # read speech file, offset and truncate, then normalize
        (_, s) = wavfile.read(self.df_s.filepath.iloc[self.s_idx])
        s = s[offset_s:offset_s+self.num_samples]
        s = s / s.std()

        # read noise file, offset and truncate, then normalize
        (_, n) = wavfile.read(self.df_n.filepath.iloc[self.n_idx])
        n = n[offset_n:offset_n+self.num_samples]
        n = n / n.std()

        # mix the signals
        snr = self.rng.uniform(self.mixture_snr_min, self.mixture_snr_max)
        x = s + (n * 10**(-snr/20.))

        # generate the one-hot vector based on which cluster the speaker is in
        y = torch.zeros(self.num_clusters)
        speaker_id = self.df_s.speaker_id.iloc[self.s_idx]
        y[self.mapping['speakers'][speaker_id]] = 1.

        # create output tuple
        scale_factor = abs(x).max()
        sample = (
            torch.Tensor(x) / scale_factor,
            y
        )

        return sample


librispeech = create_df_librispeech('/media/sdc1/librispeech_8khz/')
musan = create_df_musan('/media/sdc1/musan_8khz/')

speakers_tr = pd.read_csv(os.path.join(ROOT, 'speakers', 'train.csv'), dtype=_df_types)
speakers_vl = pd.read_csv(os.path.join(ROOT, 'speakers', 'validation.csv'), dtype=_df_types)
speakers_te = pd.read_csv(os.path.join(ROOT, 'speakers', 'test.csv'), dtype=_df_types)
speaker_ids_tr = sorted(speakers_tr.speaker_id)
speaker_ids_vl = sorted(speakers_vl.speaker_id)
speaker_ids_te = sorted(speakers_te.speaker_id)
