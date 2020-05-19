import numpy as np
import gym
from .core_wrapper import Wrapper, ActionWrapper, RewardWrapper, ObservationWrapper
import retro
import random
import librosa
import torch


ZONE_ACT_2_LVL_MAX_X = {
    "GreenHillZone.Act1":  9568, "GreenHillZone.Act2":  8032,  "GreenHillZone.Act3":  10538,
    "MarbleZone.Act1":     6240, "MarbleZone.Act2":     6240,  "MarbleZone.Act3":     5920,
    "SpringYardZone.Act1": 9056, "SpringYardZone.Act2": 10592, "SpringYardZone.Act3": 11139,
    "LabyrinthZone.Act1":  6736, "LabyrinthZone.Act2":  4432,  "LabyrinthZone.Act3":  7364,
    "StarLightZone.Act1":  8288, "StarLightZone.Act2":  8288,  "StarLightZone.Act3":  8008,
    "ScrapBrainZone.Act1": 8800, "ScrapBrainZone.Act2": 7904
}

class SonicJointEnv(gym.Env):
    """
    An environment that samples a new sub-environment at
    every episode boundary.
    This can be used for joint-training.
    """

    def __init__(self, env_states):
        """
        Create a joint environment.
        Args:
          env_states: names of retro game states to use.
        """
        self.env_states = env_states
        self.env = None
        self.em = None
        self.env_idx = 0
        
        env = retro.make(
            game='SonicTheHedgehog-Genesis', state=self.env_states[0])
        self.action_space = env.action_space
        self.observation_space = env.observation_space
        env.close()

    def reset(self, **kwargs):
        if self.env is not None:
            self.em = None
            self.env.close()
        self.env_idx = random.randrange(len(self.env_states))
        env_state = self.env_states[self.env_idx]
        self.env = retro.make(
            game='SonicTheHedgehog-Genesis', state=env_state)
        self.em = self.env.em
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        info = info.copy()
        info['env_idx'] = self.env_idx
        env_state = self.env_states[self.env_idx]
        info['env_state'] = env_state
        info['lvl_max_x'] = ZONE_ACT_2_LVL_MAX_X[env_state]
        return obs, rew, done, info

    def render(self, mode='human'):
        if self.env is None:
            return
        return self.env.render(mode=mode)

    def seed(self, seed=None):
        if self.env is None:
            return
        return self.env.seed(seed=seed)


class TimeLimit(Wrapper):
    def __init__(self, env, max_episode_steps=None):
        super(TimeLimit, self).__init__(env)
        self._max_episode_steps = max_episode_steps
        self._elapsed_steps = 0

    def step(self, ac):
        observation, reward, done, info = self.env.step(ac)
        self._elapsed_steps += 1
        if self._elapsed_steps >= self._max_episode_steps:
            done = True
            info['TimeLimit.truncated'] = True
        return observation, reward, done, info

    def reset(self, **kwargs):
        self._elapsed_steps = 0
        return self.env.reset(**kwargs)
    

class ClipActionsWrapper(Wrapper):
    def step(self, action):
        action = np.nan_to_num(action)
        action = np.clip(action, self.action_space.low, self.action_space.high)
        return self.env.step(action)

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

class SonicDiscretizer(ActionWrapper):
    """
    Wrap a gym-retro environment and make it use discrete
    actions for the Sonic game.
    """
    def __init__(self, env):
        super(SonicDiscretizer, self).__init__(env)
        buttons = ["B", "A", "MODE", "START", "UP", "DOWN", "LEFT", "RIGHT", "C", "Y", "X", "Z"]
        actions = [['LEFT'], ['RIGHT'], ['LEFT', 'DOWN'], ['RIGHT', 'DOWN'], ['DOWN'],
                   ['DOWN', 'B'], ['B']]
        self._actions = []
        for action in actions:
            arr = np.array([False] * 12)
            for button in action:
                arr[buttons.index(button)] = True
            self._actions.append(arr)
        self.action_space = gym.spaces.Discrete(len(self._actions))

    def action(self, a): # pylint: disable=W0221
        a = a[0]  # TODO: is this a cuda thing?
        return self._actions[a].copy()


class RewardScaler(RewardWrapper):
    """
    Bring rewards to a reasonable scale for PPO.
    This is incredibly important and effects performance
    drastically.
    """
    def __init__(self, env, scale=0.01):
        super(RewardScaler, self).__init__(env)
        self.scale = scale

    def reward(self, reward):
        return reward * self.scale

class AllowBacktracking(Wrapper):
    """
    Use deltas in max(X) as the reward, rather than deltas
    in X. This way, agents are not discouraged too heavily
    from exploring backwards if there is no way to advance
    head-on in the level.
    """
    def __init__(self, env):
        super(AllowBacktracking, self).__init__(env)
        self._cur_x = 0
        self._max_x = 0

    def reset(self, **kwargs): # pylint: disable=E0202
        self._cur_x = 0
        self._max_x = 0
        return self.env.reset(**kwargs)

    def step(self, action): # pylint: disable=E0202
        obs, rew, done, info = self.env.step(action)
        self._cur_x += rew
        rew = max(0, self._cur_x - self._max_x)
        self._max_x = max(self._max_x, self._cur_x)
        return obs, rew, done, info

class SonicMaxXSumRInfo(Wrapper):
    """
    Horizontal reward sums linearly to a total of 9000. 
    """
    def __init__(self, env):
        super(SonicMaxXSumRInfo, self).__init__(env)
        self.max_x = 0
        self.sum_r = 0
        self.start_x = -1

    def reset(self, **kwargs): # pylint: disable=E0202
        self.max_x = 0
        self.sum_r = 0
        self.start_x = -1
        return self.env.reset(**kwargs)

    def step(self, action): # pylint: disable=E0202
        obs, rew, done, info = self.env.step(action)
        if self.start_x < 0:
            self.start_x = info['x']
            info['start_x'] = self.start_x
        self.max_x = max(info['x'], self.max_x)
        info['max_x'] = self.max_x
        self.sum_r += rew
        info['sum_r'] = self.sum_r

        return obs, rew, done, info

class StochasticFrameSkip(Wrapper):
    def __init__(self, env, n, stickprob):
        gym.Wrapper.__init__(self, env)
        self.n = n
        self.stickprob = stickprob
        self.curac = None
        self.rng = np.random.RandomState()
        self.supports_want_render = hasattr(env, "supports_want_render")

    def reset(self, **kwargs):
        self.curac = None
        return self.env.reset(**kwargs)

    def step(self, ac):
        done = False
        totrew = 0
        for i in range(self.n):
            # First step after reset, use action
            if self.curac is None:
                self.curac = ac
            # First substep, delay with probability=stickprob
            elif i==0:
                if self.rng.rand() > self.stickprob:
                    self.curac = ac
            # Second substep, new action definitely kicks in
            elif i==1:
                self.curac = ac
            if self.supports_want_render and i<self.n-1:
                ob, rew, done, info = self.env.step(self.curac, want_render=False)
            else:
                ob, rew, done, info = self.env.step(self.curac)
            totrew += rew
            if done: break
        return ob, totrew, done, info

    def seed(self, s):
        self.rng.seed(s)

class RewardScaler(RewardWrapper):
    """
    Bring rewards to a reasonable scale for PPO.
    This is incredibly important and effects performance
    drastically.
    """
    def __init__(self, env, scale=0.01):
        super(RewardScaler, self).__init__(env)
        self.scale = scale

    def reward(self, reward):
        return reward * self.scale

class EnvAudio(ObservationWrapper):
    """
    Adds environment audio to observation, creating obs tuple (image, audio).
    """
    def __init__(self, env):
        super(EnvAudio, self).__init__(env)
        audio_obs_space = gym.spaces.Box(-1.0, 1.0, shape=(735,), dtype=np.float64)
        self.observation_space = gym.spaces.Tuple((self.observation_space, audio_obs_space))

    def observation(self, observation):
        audio = self.em.get_audio()[:735,0] / 2**15
        return {0: observation, 1: audio}

class AudioFeaturizer(ObservationWrapper):
    """
    Processes audio observation to feature.
    """
    def __init__(self, env, feature_type='spectrogram'):
        super(AudioFeaturizer, self).__init__(env)
        self.feature_type = feature_type
        audio_feature_obs_space = gym.spaces.Box(float('-inf'), 0.0, shape=(128,2), dtype=np.float32)
        self.observation_space = gym.spaces.Tuple((self.observation_space[0], audio_feature_obs_space))

    def observation(self, observation):
        assert len(observation) == 2
        audio = observation[1]
        # the audio was playing at 60fps but now needs to be 15 fps?
        #audio_feats = librosa.feature.mfcc(y=audio, sr=22050, n_mfcc=40)
        mel_s = librosa.feature.melspectrogram(y=audio, sr=22050, hop_length=512)
        audio_feats = librosa.power_to_db(mel_s, ref=np.max)
        return {0: observation[0], 1: audio_feats}
        

