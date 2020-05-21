import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchaudio.transforms import MelSpectrogram

from a2c_ppo_acktr.distributions import Bernoulli, Categorical, DiagGaussian
from a2c_ppo_acktr.utils import init


class Flatten(nn.Module):
    def forward(self, x):
        if not x.is_contiguous():
            x = x.contiguous()
        return x.view(x.size(0), -1)

class Policy(nn.Module):
    def __init__(self, obs_space, obs_module, action_space, base_kwargs=None):
        super(Policy, self).__init__()
        self.obs_space = obs_space
        self.obs_module = obs_module

        if base_kwargs is None:
            base_kwargs = {}

        # base takes all of the observations and produces a single feature vector
        self.base = NNBase2(obs_space, obs_module, **base_kwargs)
        
        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0))
        self.critic_linear = init_(nn.Linear(self.base.output_size, 1))
        
        if action_space.__class__.__name__ == "Discrete":
            num_outputs = action_space.n
            self.dist = Categorical(self.base.output_size, num_outputs)
        elif action_space.__class__.__name__ == "Box":
            num_outputs = action_space.shape[0]
            self.dist = DiagGaussian(self.base.output_size, num_outputs)
        elif action_space.__class__.__name__ == "MultiBinary":
            num_outputs = action_space.shape[0]
            self.dist = Bernoulli(self.base.output_size, num_outputs)
        else:
            raise NotImplementedError
    
    @property
    def is_recurrent(self):
        return self.base.is_recurrent

    @property
    def recurrent_hidden_state_size(self):
        """Size of rnn_hx."""
        return self.base.recurrent_hidden_state_size

    def forward(self, inputs, rnn_hxs, masks):
        raise NotImplementedError

    def act(self, obs, rnn_hxs, masks, deterministic=False):
        actor_features, rnn_hxs = self.base(obs, rnn_hxs, masks)
        value = self.critic_linear(actor_features)
        dist = self.dist(actor_features)

        if deterministic:
            action = dist.mode()
        else:
            action = dist.sample()

        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()

        return value, action, action_log_probs, rnn_hxs


    def get_value(self, inputs, rnn_hxs, masks):
        # inputs is a dict, 0: vid_obs, 1: aud_obs
        actor_features, _ = self.base(inputs, rnn_hxs, masks)
        value = self.critic_linear(actor_features)
        return value

    def evaluate_actions(self, inputs, rnn_hxs, masks, action):
        # inputs is a dict, 0: vid_obs, 1: aud_obs
        actor_features, rnn_hxs = self.base(inputs, rnn_hxs, masks)
        value = self.critic_linear(actor_features)
        dist = self.dist(actor_features)

        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()

        return value, action_log_probs, dist_entropy, rnn_hxs


class NNBase(nn.Module):
    def __init__(self, recurrent, recurrent_input_size, hidden_size):
        super(NNBase, self).__init__()

        self._hidden_size = hidden_size
        self._recurrent = recurrent

        if recurrent:
            self.gru = nn.GRU(recurrent_input_size, hidden_size)
            for name, param in self.gru.named_parameters():
                if 'bias' in name:
                    nn.init.constant_(param, 0)
                elif 'weight' in name:
                    nn.init.orthogonal_(param)

    @property
    def is_recurrent(self):
        return self._recurrent

    @property
    def recurrent_hidden_state_size(self):
        if self._recurrent:
            return self._hidden_size
        return 1

    @property
    def output_size(self):
        return self._hidden_size

    def _forward_gru(self, x, hxs, masks):
        if x.size(0) == hxs.size(0):
            x, hxs = self.gru(x.unsqueeze(0), (hxs * masks).unsqueeze(0))
            x = x.squeeze(0)
            hxs = hxs.squeeze(0)
        else:
            # x is a (T, N, -1) tensor that has been flatten to (T * N, -1)
            N = hxs.size(0)
            T = int(x.size(0) / N)

            # unflatten
            x = x.view(T, N, x.size(1))

            # Same deal with masks
            masks = masks.view(T, N)

            # Let's figure out which steps in the sequence have a zero for any agent
            # We will always assume t=0 has a zero in it as that makes the logic cleaner
            has_zeros = ((masks[1:] == 0.0) \
                            .any(dim=-1)
                            .nonzero()
                            .squeeze()
                            .cpu())

            # +1 to correct the masks[1:]
            if has_zeros.dim() == 0:
                # Deal with scalar
                has_zeros = [has_zeros.item() + 1]
            else:
                has_zeros = (has_zeros + 1).numpy().tolist()

            # add t=0 and t=T to the list
            has_zeros = [0] + has_zeros + [T]

            hxs = hxs.unsqueeze(0)
            outputs = []
            for i in range(len(has_zeros) - 1):
                # We can now process steps that don't have any zeros in masks together!
                # This is much faster
                start_idx = has_zeros[i]
                end_idx = has_zeros[i + 1]

                rnn_scores, hxs = self.gru(
                    x[start_idx:end_idx],
                    hxs * masks[start_idx].view(1, -1, 1))

                outputs.append(rnn_scores)

            # assert len(outputs) == T
            # x is a (T, N, -1) tensor
            x = torch.cat(outputs, dim=0)
            # flatten
            x = x.view(T * N, -1)
            hxs = hxs.squeeze(0)

        return x, hxs

class NNBase2(NNBase):
    def __init__(self, obs_space, obs_module, recurrent=False, hidden_size=512):
        super(NNBase2, self).__init__(recurrent, hidden_size, hidden_size)

        #self.output_size = 0
        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), nn.init.calculate_gain('relu'))

        for obs_name in sorted(obs_module.keys()):  # order of adding layers matters
            if obs_module[obs_name] == 'video-small':
                # My original video model - 942,080 parameters
                num_inputs = obs_space[obs_name].shape[0]  # should be 3, RGB
                module = nn.Sequential(  # video shape [3, 224, 320]
                    nn.BatchNorm2d(num_inputs),
                    init_(nn.Conv2d(num_inputs, 32, 8, stride=4)), nn.ReLU(),  # [32, 55, 79]
                    init_(nn.Conv2d(32, 64, 4, stride=4)), nn.ReLU(),  # [64, 13, 19]
                    init_(nn.Conv2d(64, 32, 3, stride=2)), nn.ReLU(), Flatten(),  # [32, 6, 9]
                    init_(nn.Linear(32*6*9, 512)), nn.ReLU())
                #self.output_size += 512
            elif obs_module[obs_name] == 'video-medium':
                # OpenAI Baselines small - 8,104,960 parameters
                num_inputs = obs_space[obs_name].shape[0]  # should be 3, RGB
                module = nn.Sequential(  # video shape [3, 224, 320]
                    nn.BatchNorm2d(num_inputs),
                    init_(nn.Conv2d(num_inputs, 16, 8, stride=4)), nn.ReLU(),  # [16, 55, 79]
                    init_(nn.Conv2d(16, 32, 4, stride=2)), nn.ReLU(),  # [32, 26, 38]
                    init_(nn.Linear(32*26*38, 512)), nn.ReLU())
                #self.output_size += 512
            elif obs_module[obs_name] == 'video-large':
                # OpenAI Baselines large - 28,387,328 parameters
                num_inputs = obs_space[obs_name].shape[0]  # should be 3, RGB
                module = nn.Sequential(  # video shape [3, 224, 320]
                    #nn.BatchNorm2d(num_inputs),
                    init_(nn.Conv2d(num_inputs, 32, 8, stride=4)), nn.ReLU(),  # [32, 55, 79]
                    #nn.BatchNorm2d(32), nn.ReLU(),
                    init_(nn.Conv2d(32, 64, 4, stride=2)), nn.ReLU(),  # [64, 26, 38]
                    #nn.BatchNorm2d(64), nn.ReLU(),
                    init_(nn.Conv2d(64, 64, 3, stride=1)), nn.ReLU(),  # [64, 24, 36]
                    #nn.BatchNorm2d(64), nn.ReLU(),
                    Flatten(), init_(nn.Linear(64*24*36, 512)), nn.ReLU())  # [512]
                #self.output_size += 512
            elif obs_module[obs_name] == 'audio-small':
                # My original audio model - 65,536 parameters
                module = nn.Sequential(  # audio shape [735,]
                    MelSpectrogram(sample_rate=22050, n_fft=512, hop_length=735//2+1),  # [128, 2]
                    Flatten(), nn.BatchNorm1d(256),  # [256]
                    init_(nn.Linear(128*2, 256)), nn.ReLU())  # [256]
                #self.output_size += 256
            else:
                raise NotImplementedError
            
            if obs_name == 'video':
                self.video_module = module
            elif obs_name == 'audio':
                self.audio_module = module
            else:
                raise NotImplementedError

        self.train()

    def forward(self, obs, rnn_hxs, masks):
        x = []
        for obs_name in sorted(obs.keys()):
            if obs_name == 'video':
                x.append(self.video_module(obs[obs_name] / 255.0))
            elif obs_name == 'audio':
                x.append(self.audio_module(obs[obs_name]))
            else:
                raise NotImplementedError
        x = torch.cat(x, dim=1)

        return x, rnn_hxs
