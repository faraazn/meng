import torch.nn as nn
from torchaudio.transforms import MelSpectrogram, AmplitudeToDB

class ProcessMelSpectrogram(nn.Module):
    def __init__(self):
        super(ProcessMelSpectrogram, self).__init__()
        self.mel_s = MelSpectrogram(
            sample_rate=11025, n_fft=1024, n_mels=256, win_length=256, hop_length=8)
        self.a_to_db = AmplitudeToDB(top_db=80)

    def forward(self, x):
        # x shape [b, 735]
        x = self.mel_s(x)  # shape [b, 256, 92]
        x = self.a_to_db(x)  # (max - min) range [0, 80]
        x = x - x.max(1, True)[0].max(1, True)[0]  # range [0, -80]
        x = x.unsqueeze(1) / -80  # range [0, 1], shape [b, 1, 256, 92]
        return x
