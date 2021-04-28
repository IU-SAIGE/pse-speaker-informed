from typing import List, Optional
import json
import numpy as np
import torch

from scipy.spatial.distance import cdist

FFT_SIZE: int = 1024
HOP_LENGTH: int = 256
WINDOW = torch.hann_window(FFT_SIZE)


def stft(waveform: torch.Tensor):
    """Calculates the Short-time Fourier transform (STFT)."""

    # perform the short-time Fourier transform
    spectrogram = torch.stft(
        waveform, FFT_SIZE, HOP_LENGTH, window=WINDOW.to(waveform.device)
    )

    # swap seq_len & feature_dim of the spectrogram (for RNN processing)
    spectrogram = spectrogram.permute(0, 2, 1, 3)

    # calculate the magnitude spectrogram
    magnitude_spectrogram = torch.sqrt(spectrogram[..., 0] ** 2 +
                                       spectrogram[..., 1] ** 2)

    return (spectrogram, magnitude_spectrogram)


def istft(spectrogram: torch.Tensor, mask: Optional[torch.Tensor] = None):
    """Calculates the inverse Short-time Fourier transform (ISTFT)."""

    # apply a time-frequency mask if provided
    if mask is not None:
        spectrogram[..., 0] *= mask
        spectrogram[..., 1] *= mask

    # swap seq_len & feature_dim of the spectrogram (undo RNN processing)
    spectrogram = spectrogram.permute(0, 2, 1, 3)

    # perform the inverse short-time Fourier transform
    waveform = torch.istft(
        spectrogram, FFT_SIZE, HOP_LENGTH, window=WINDOW.to(spectrogram.device)
    )

    return waveform


class NetSV(torch.nn.Module):

    def __init__(self, hidden_size, num_layers):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = torch.nn.GRU(
            input_size=int(FFT_SIZE // 2 + 1),
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True
        )


    def embedding(self, x):
        x = stft(x)[1]
        x = self.rnn(x)[0]
        return x[:, -1]

    def forward(self, x_1, x_2):
        feature_1 = self.embedding(x_1)
        feature_2 = self.embedding(x_2)
        is_same = torch.bmm(
            feature_1.unsqueeze(1),
            feature_2.unsqueeze(2)
        ).squeeze()
        return is_same


class NetSE(torch.nn.Module):

    def __init__(self, hidden_size: int, num_layers: int = 2):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # create a neural network which predicts a TF binary ratio mask
        self.encoder = torch.nn.GRU(
            input_size=int(FFT_SIZE // 2 + 1),
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True
        )
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(
                in_features=self.hidden_size,
                out_features=int(FFT_SIZE // 2 + 1)
            ),
            torch.nn.Sigmoid()
        )
        self.name = (self.__class__.__name__ +
                     f'_{hidden_size:04d}x{num_layers:02d}')

    def forward(self, waveform):
        # convert waveform to spectrogram
        (X, X_magnitude) = stft(waveform)

        # generate a time-frequency mask
        H = self.encoder(X_magnitude)[0]
        Y = self.decoder(H)
        Y = Y.reshape_as(X_magnitude)

        # convert masked spectrogram back to waveform
        denoised = istft(X, mask=Y)

        return denoised


class NetEnsemble(torch.nn.Module):

    def __init__(
            self,
            gating_hidden_size: int,
            gating_num_layers: int = 2,
            specialist_hidden_size: int = 256,
            specialist_num_layers: int = 2,
            num_specialists: int = 10,
    ):
        super().__init__()
        self.gating_hidden_size = gating_hidden_size
        self.gating_num_layers = gating_num_layers
        self.specialist_hidden_size = specialist_hidden_size
        self.specialist_num_layers = specialist_num_layers
        self.num_specialists = num_specialists

        self.gating_network = NetSV(
            self.gating_hidden_size,
            self.gating_num_layers
        )

        self.specialist_network = torch.nn.ModuleList([
            NetSE(specialist_hidden_size, specialist_num_layers)
            for _ in range(self.num_specialists)
        ])

        self.specialist_mapping = {}
        self.speaker_mapping = {}
        self.speaker_vectors = {}
        self.cluster_means = []

    def load_speaker_vectors(
            self,
            specialist_mapping_filepath: str,
            speaker_mapping_filepath: str,
            speaker_features_filepath: str,
    ):
        with open(speaker_mapping_filepath, 'r') as fp:
            self.speaker_mapping = json.load(fp)
        with open(specialist_mapping_filepath, 'r') as fp:
            self.specialist_mapping = json.load(fp)
        self.speaker_vectors = np.load(speaker_features_filepath,
                                       allow_pickle=True).item()
        self.cluster_means = np.array([
            np.mean([
                self.speaker_vectors[spkr_id]
                for spkr_id in self.specialist_mapping[str(k)]], axis=0)
            for k in range(self.num_specialists)
        ])
        return True

    def load_gating_network(
            self,
            checkpoint_filepath: List[str],
    ):
        sd = torch.load(checkpoint_filepath)
        self.gating_network.load_state_dict(sd, strict=True)
        return True

    def load_specialist_networks(
            self,
            checkpoint_filepaths: List[str],
    ):
        assert len(checkpoint_filepaths) == self.num_specialists
        for i in range(self.num_specialists):
            sd = torch.load(checkpoint_filepaths[i])
            self.specialist_network[i].load_state_dict(sd, strict=True)
        return True

    def forward(self, waveform: torch.Tensor):

        # convert waveform to spectrogram
        (X, X_magnitude) = stft(waveform)

        # generate gating prediction
        H = self.gating_network.rnn(X_magnitude)[0][:, -1]
        H = H.detach().cpu().numpy()
        S = cdist(H, self.cluster_means, 'euclidean').argmin(axis=1)

        result = []
        for i in range(len(S)):
            result.append(
                self.specialist_network[S[i]](waveform[i].unsqueeze(0))
            )
        return torch.stack(result)
