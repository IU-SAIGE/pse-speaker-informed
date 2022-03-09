from collections import OrderedDict
from typing import List, Optional
import yaml
import numpy as np
import torch


FFT_SIZE: int = 1024
HOP_LENGTH: int = 256
WINDOW = torch.hann_window(FFT_SIZE)


class Baseline(torch.nn.Module):

    def __init__(self, hidden_size: int, num_layers: int = 2):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
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
        initialize_weights(self)

    def estimate_mask(self, magnitude_spectrogram):
        hidden = many_to_many(self.encoder(magnitude_spectrogram))
        mask = self.decoder(hidden)
        mask = mask.reshape_as(magnitude_spectrogram)
        return mask

    def forward(self, waveform):
        (X, X_magnitude) = stft(waveform)
        Y = self.estimate_mask(X_magnitude)
        denoised = istft(X, mask=Y)
        return denoised


class Specialist(Baseline):

    def __init__(self, hidden_size: int, num_layers: int = 2):
        super().__init__(hidden_size, num_layers)


class Gating(torch.nn.Module):

    def __init__(self, hidden_size: int, num_layers: int = 2, num_clusters: int = 10):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = torch.nn.GRU(
            input_size=int(FFT_SIZE // 2 + 1),
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True
        )
        self.dnn = torch.nn.Linear(hidden_size, num_clusters)
        self.softmax = torch.nn.Softmax(dim=-1)
        initialize_weights(self)

    def embedding(self, magnitude_spectrogram):
        x = many_to_one(self.rnn(magnitude_spectrogram))
        return x

    def similarity(self, waveform_1, waveform_2, as_logits: bool = True):
        magspec_1 = stft(waveform_1)[1]
        magspec_2 = stft(waveform_2)[1]
        feature_1 = self.embedding(magspec_1)
        feature_2 = self.embedding(magspec_2)
        x = torch.bmm(
            feature_1.unsqueeze(1),
            feature_2.unsqueeze(2)
        ).squeeze()
        if not as_logits:
            x = self.softmax(x)
        return x

    def classify(self, features, alpha: float = 10, as_logits: bool = True):
        x = alpha * self.dnn(features)
        if not as_logits:
            x = self.softmax(x)
        return x

    def forward(self, waveform, as_logits: bool = True):
        x = stft(waveform)[1]
        x = self.embedding(x)
        x = self.classify(x, as_logits=as_logits)
        return x


class Ensemble(torch.nn.Module):

    def __init__(
            self,
            num_specialists: int = 10,
            specialist_hidden_size: int = 256,
            specialist_num_layers: int = 2,
            gating_hidden_size: int = 32,
            gating_num_layers: int = 2,
    ):
        super().__init__()
        self.gating_hidden_size = gating_hidden_size
        self.gating_num_layers = gating_num_layers
        self.specialist_hidden_size = specialist_hidden_size
        self.specialist_num_layers = specialist_num_layers
        self.num_specialists = num_specialists

        self.gating_network = Gating(
            self.gating_hidden_size,
            self.gating_num_layers,
            self.num_specialists
        )

        weights = torch.load('/media/sdb1/asivara/Research/2021_waspaa/weights/sv')
        weights = OrderedDict((k[4:] if 'rnn' in k else k, v) for k, v in weights.items())
        self.gating_network.rnn.load_state_dict(weights, strict=True)

        weights = torch.load('/media/sdb1/asivara/Research/2021_waspaa/weights/classifier/K={:02d}/checkpoint'.format(num_specialists))
        weights = OrderedDict((k[11:] if 'classifier' in k else k, v) for k, v in weights.items())
        self.gating_network.dnn.load_state_dict(weights, strict=True)

        self.specialist_network = torch.nn.ModuleList([])
        for i in range(self.num_specialists):
            s = Specialist(specialist_hidden_size, specialist_num_layers)
            s.load_state_dict(
                torch.load('/media/sdb1/asivara/Research/2021_waspaa/weights/specialists/se-hs={:04d}_K={:02d}_k={:02d}/checkpoint'.format(
                    specialist_hidden_size, num_specialists, i)),
                strict=True)
            self.specialist_network.append(s)

    def forward(self, waveform: torch.Tensor):

        p = self.gating_network(waveform, as_logits=False)
        p = p[..., None, None]

        # run each specialist network on all the inputs
        (X, X_magnitude) = stft(waveform)
        Y = torch.stack([
            self.specialist_network[k].estimate_mask(X_magnitude)
            for k in range(self.num_specialists)
        ], dim=1)

        # combine all the specialist inferences as weighted by the
        # gating network
        Y_hat = (p * Y).sum(dim=1)

        denoised = istft(X, mask=Y_hat)
        return denoised


def stft(waveform: torch.Tensor):
    """Calculates the Short-time Fourier transform (STFT)."""

    # perform the short-time Fourier transform
    spectrogram = torch.stft(
        waveform, FFT_SIZE, HOP_LENGTH, window=WINDOW.to(waveform.device),
        return_complex=False
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
        spectrogram, FFT_SIZE, HOP_LENGTH, window=WINDOW.to(spectrogram.device),
        return_complex=False
    )

    return waveform


def initialize_weights(network):
    for name, param in network.named_parameters():
        if 'bias' in name:
            torch.nn.init.constant_(param, 0.0)
        elif 'weight' in name:
            torch.nn.init.xavier_uniform_(param)


def many_to_one(rnn_output, batch_first=True):
    if batch_first:
        o = rnn_output[0][:, -1]
    else:
        o = rnn_output[0][-1]
    return o


def many_to_many(rnn_output):
    o = rnn_output[0]
    return o


def compare_models(sd_1, sd_2):
    models_differ = 0
    for key_item_1, key_item_2 in zip(sd_1.items(), sd_2.items()):
        if torch.equal(key_item_1[1], key_item_2[1]):
            pass
        else:
            models_differ += 1
    return (models_differ == 0)
