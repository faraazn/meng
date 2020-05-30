import torch
import torch.nn as nn
from torchaudio.transforms import MelSpectrogram, AmplitudeToDB

class ProcessMelSpectrogram(nn.Module):
    """
    Expects as input: [batch, samples] or [b, memory (or frameskip), s], or [b, m, f, s]
    Output: [batch, 1, n_mels, samples/hop_len]
    """
    def __init__(self, sr=44100, n_fft=2048, n_mels=256, win_len=512, hop_len=32):
        super(ProcessMelSpectrogram, self).__init__()
        # og spectrogram process: sr 11025, n_fft 1024, n_mels 256, win_len 256, hop_len 8
        # og output shape 256, 92
        # librosa default params: sr 22050, n_fft 2048, n_mels ?, win_len 2048, hop_len 512
        # music processing: 93 ms, speech processing: 23 ms (computed by 1/(sr/hop_len))
        self.mel_s = MelSpectrogram(
            sample_rate=sr, n_fft=n_fft, n_mels=n_mels, win_length=win_len, hop_length=hop_len)
        self.a_to_db = AmplitudeToDB(top_db=80)

    def forward(self, x):
        assert len(x.shape) in [2, 3, 4]
        if len(x.shape) != 2:
            x = torch.flatten(x, start_dim=1, end_dim=-1)
        # x shape [b, flat_samples]
        x = self.mel_s(x)  # shape [b, n_mels, flat_samples/hop_len]
        x = self.a_to_db(x)  # (max - min) range [0, 80]
        x = x - x.max(1, True)[0].max(1, True)[0]  # range [0, -80]
        x = x.unsqueeze(1) / -80  # range [0, 1], shape [b, 1, n_mels, flat_samples/hop_len]
        return x


class ProcessRGBVideo(nn.Module):
    """
    Expects as input: [batch, height, width, channels] or [b, memory (or frameskip), h, w, c], or [b, m, f, h, w, c]
    Output: [batch, channels, depth, height, width]
    """
    def forward(self, x):
        assert len(x.shape) in [4, 5, 6]
        # permute last 3 dims [height, width, channels] to [channels, height, width]
        p = list(range(len(x.shape)))
        p = p[:-3] + [p[-1]] + p[-3:-1]
        x = x.permute(p)
        if len(x.shape) == 6:
            # x shape [batch, memory, frameskip, channels, height, width]
            x = torch.flatten(x, start_dim=1, end_dim=2)  # [batch, depth, channels, height, width]
        if len(x.shape) == 5:
            # x shape [batch, depth, channels, height, width]
            x = x.transpose(1, 2)  # [batch, channels, depth, height, width]
            x = x.squeeze(2)  # if depth 1, [batch, channels, height, width]
        # x shape [batch, channels, height, width] or [batch, channels, depth, height, width]
        return x #/ 255
