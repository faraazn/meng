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
                      SonicDiscretizer, RewardScaler, StochasticFrameSkip, EnvAudio, ObsMemoryBuffer
from .core_wrapper import ObservationWrapper


def make_env(env_states, seed, rank, mode, args):
    def _thunk():
        env = SonicJointEnv(env_states)
        if args.use_audio:
            env = EnvAudio(env)
        env = SonicDiscretizer(env)
        env = AllowBacktracking(env)
        env = SonicMaxXSumRInfo(env)
        if mode == 'train':
            env = RewardScaler(env, scale=args.rew_scale)
        env = StochasticFrameSkip(env, args.fskip_num, args.fskip_prob, args.obs_keep_fskip)
        env = ObsMemoryBuffer(env, args.obs_mbuf)
        env = TimeLimit(env, max_episode_steps=args.max_episode_steps)

        env.seed(seed + rank)
        return env

    return _thunk


def make_vec_envs(env_states,
                  seed,
                  num_processes,
                  gamma,
                  device,
                  mode,
                  args):
    assert mode in ['train', 'eval']
    if num_processes % len(env_states) == 0:
        # one state per process
        envs = [
            make_env([env_states[i%len(env_states)]], seed, i, mode, args)
            for i in range(num_processes)
        ]
    else:
        # random sample new state on done
        envs = [
            make_env(env_states, seed, i, mode, args)
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
