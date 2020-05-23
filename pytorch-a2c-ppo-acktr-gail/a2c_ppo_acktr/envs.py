import os

import gym
import numpy as np
import torch
from gym.spaces.box import Box

from .bench import Monitor
from .vec_env.vec_env import VecEnvWrapper
from .vec_env.dummy_vec_env import DummyVecEnv
from .vec_env.shmem_vec_env import ShmemVecEnv
from .vec_env.vec_normalize import VecNormalize as VecNormalize_

import retro
from .wrappers import SonicJointEnv, TimeLimit, AllowBacktracking, SonicMaxXSumRInfo, \
                      SonicDiscretizer, RewardScaler, StochasticFrameSkip, EnvAudio
from .core_wrapper import ObservationWrapper


def make_env(env_states, seed, rank, allow_early_resets, mode):
    def _thunk():
        env = SonicJointEnv(env_states)
        env = EnvAudio(env)
        env = SonicDiscretizer(env)
        env = AllowBacktracking(env)
        env = SonicMaxXSumRInfo(env)
        if mode == 'train':
            env = RewardScaler(env, scale=0.005)
        env = StochasticFrameSkip(env, 4, 0.25)
        env = TimeLimit(env, max_episode_steps=4500)

        env.seed(seed + rank)

        # wrap for PyTorch convolutions
        env = TransposeImage(env, op=[2, 0, 1])

        return env

    return _thunk


def make_vec_envs(env_states,
                  seed,
                  num_processes,
                  gamma,
                  device,
                  allow_early_resets,
                  mode,
                  num_frame_stack=None):
    assert mode in ['train', 'eval']
    envs = [
        make_env(env_states, seed, i, allow_early_resets, mode)
        for i in range(num_processes)
    ]

    if len(envs) > 1:
        envs = ShmemVecEnv(envs, context='forkserver')
    else:
        envs = DummyVecEnv(envs)

    envs = VecPyTorch(envs, device)
    return envs


# Checks whether done was caused my timit limits or not
class TimeLimitMask(gym.Wrapper):
    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        if done and self.env._max_episode_steps == self.env._elapsed_steps:
            info['bad_transition'] = True

        return obs, rew, done, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)


class TransposeObs(ObservationWrapper):
    def __init__(self, env=None):
        """
        Transpose observation space (base class)
        """
        super(TransposeObs, self).__init__(env)


class TransposeImage(TransposeObs):
    def __init__(self, env=None, op=[2, 0, 1]):
        """
        Transpose observation space for images
        """
        super(TransposeImage, self).__init__(env)
        assert len(op) == 3, "Error: Operation, " + str(op) + ", must be dim3"
        self.op = op
        # TODO: generalize to any 3 dim in obs dict?
        vid_obs_shape = self.observation_space['video'].shape
        self.observation_space.spaces['video'] = gym.spaces.Box(
            self.observation_space['video'].low[0, 0, 0],
            self.observation_space['video'].high[0, 0, 0], [
                vid_obs_shape[self.op[0]], vid_obs_shape[self.op[1]],
                vid_obs_shape[self.op[2]]
            ], dtype=self.observation_space['video'].dtype)

    def observation(self, ob):
        ob['video'] = ob['video'].transpose(self.op[0], self.op[1], self.op[2])
        return ob


class VecPyTorch(VecEnvWrapper):
    def __init__(self, venv, device):
        """Return only every `skip`-th frame"""
        super(VecPyTorch, self).__init__(venv)
        self.device = device
        # TODO: Fix data types

    def reset(self):
        obs = self.venv.reset()
        obs = {k: torch.from_numpy(obs[k]).float().to(self.device) for k in obs.keys()}
        return obs

    def step_async(self, actions):
        if isinstance(actions, torch.LongTensor):
            # Squeeze the dimension for discrete actions
            actions = actions.squeeze(1)
        actions = actions.cpu().numpy()
        self.venv.step_async(actions)

    def step_wait(self):
        obs, reward, done, info = self.venv.step_wait()
        obs = {k: torch.from_numpy(obs[k]).float().to(self.device) for k in obs.keys()}
        reward = torch.from_numpy(reward).unsqueeze(dim=1).float()
        return obs, reward, done, info


class VecNormalize(VecNormalize_):
    def __init__(self, *args, **kwargs):
        super(VecNormalize, self).__init__(*args, **kwargs)
        self.training = True

    def _obfilt(self, obs, update=True):
        if self.ob_rms:
            if self.training and update:
                self.ob_rms.update(obs)
            obs = np.clip((obs - self.ob_rms.mean) /
                          np.sqrt(self.ob_rms.var + self.epsilon),
                          -self.clipob, self.clipob)
            return obs
        else:
            return obs

    def train(self):
        self.training = True

    def eval(self):
        self.training = False


# Derived from
# https://github.com/openai/baselines/blob/master/baselines/common/vec_env/vec_frame_stack.py
class VecPyTorchFrameStack(VecEnvWrapper):
    def __init__(self, venv, nstack, device=None):
        self.venv = venv
        self.nstack = nstack

        wos = venv.observation_space  # wrapped ob space
        self.shape_dim0 = wos.shape[0]

        low = np.repeat(wos.low, self.nstack, axis=0)
        high = np.repeat(wos.high, self.nstack, axis=0)

        if device is None:
            device = torch.device('cpu')
        self.stacked_obs = torch.zeros((venv.num_envs, ) +
                                       low.shape).to(device)

        observation_space = gym.spaces.Box(
            low=low, high=high, dtype=venv.observation_space.dtype)
        VecEnvWrapper.__init__(self, venv, observation_space=observation_space)

    def step_wait(self):
        obs, rews, news, infos = self.venv.step_wait()
        self.stacked_obs[:, :-self.shape_dim0] = \
            self.stacked_obs[:, self.shape_dim0:].clone()
        for (i, new) in enumerate(news):
            if new:
                self.stacked_obs[i] = 0
        self.stacked_obs[:, -self.shape_dim0:] = obs
        return self.stacked_obs, rews, news, infos

    def reset(self):
        obs = self.venv.reset()
        if torch.backends.cudnn.deterministic:
            self.stacked_obs = torch.zeros(self.stacked_obs.shape)
        else:
            self.stacked_obs.zero_()
        self.stacked_obs[:, -self.shape_dim0:] = obs
        return self.stacked_obs

    def close(self):
        self.venv.close()
