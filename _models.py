from typing import Optional
import torch

class NetSV(torch.nn.Module):

    fft_size: int = 1024
    hop_length: int = 256
    window = torch.hann_window(fft_size)

    def __init__(self, hidden_size, num_layers):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = torch.nn.GRU(
            input_size=int(self.fft_size // 2 + 1),
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True
        )

    def mag_stft(self, waveform: torch.Tensor):
        """Calculates the Short-time Fourier transform (STFT)."""

        # perform the short-time Fourier transform
        self.window = self.window.to(waveform.device)
        spectrogram = torch.stft(
            waveform, self.fft_size, self.hop_length, window=self.window
        )

        # swap seq_len & feature_dim of the spectrogram (for RNN processing)
        spectrogram = spectrogram.permute(0, 2, 1, 3)

        # calculate the magnitude spectrogram
        magnitude_spectrogram = torch.sqrt(spectrogram[..., 0] ** 2 +
                                           spectrogram[..., 1] ** 2)

        return magnitude_spectrogram


    def embedding(self, x):
        x = self.mag_stft(x)
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

    fft_size: int = 1024
    hop_length: int = 256
    window = torch.hann_window(fft_size)

    def __init__(self, hidden_size: int, num_layers: int = 2):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # create a neural network which predicts a TF binary ratio mask
        self.encoder = torch.nn.GRU(
            input_size=int(self.fft_size // 2 + 1),
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True
        )
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(
                in_features=self.hidden_size,
                out_features=int(self.fft_size // 2 + 1)
            ),
            torch.nn.Sigmoid()
        )
        self.name = (self.__class__.__name__ +
                     f'_{hidden_size:04d}x{num_layers:02d}')

    def stft(self, waveform: torch.Tensor):
        """Calculates the Short-time Fourier transform (STFT)."""

        # perform the short-time Fourier transform
        self.window = self.window.to(waveform.device)
        spectrogram = torch.stft(
            waveform, self.fft_size, self.hop_length, window=self.window
        )

        # swap seq_len & feature_dim of the spectrogram (for RNN processing)
        spectrogram = spectrogram.permute(0, 2, 1, 3)

        # calculate the magnitude spectrogram
        magnitude_spectrogram = torch.sqrt(spectrogram[..., 0] ** 2 +
                                           spectrogram[..., 1] ** 2)

        return spectrogram, magnitude_spectrogram

    def istft(self, spectrogram: torch.Tensor,
              mask: Optional[torch.Tensor] = None):
        """Calculates the inverse Short-time Fourier transform (ISTFT)."""

        # apply a time-frequency mask if provided
        if mask is not None:
            spectrogram[..., 0] *= mask
            spectrogram[..., 1] *= mask

        # swap seq_len & feature_dim of the spectrogram (undo RNN processing)
        spectrogram = spectrogram.permute(0, 2, 1, 3)

        # perform the inverse short-time Fourier transform
        waveform = torch.istft(
            spectrogram, self.fft_size, self.hop_length, window=self.window
        )

        return waveform

    def forward(self, waveform):
        # convert waveform to spectrogram
        (X, X_magnitude) = self.stft(waveform)

        # generate a time-frequency mask
        H = self.encoder(X_magnitude)[0]
        Y = self.decoder(H)
        Y = Y.reshape_as(X_magnitude)

        # convert masked spectrogram back to waveform
        denoised = self.istft(X, mask=Y)

        return denoised

