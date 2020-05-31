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
        self.observation_space = gym.spaces.Dict({
            'video': gym.spaces.Box(np.float32(0), np.float32(1), shape=(224, 320, 3), dtype=np.float32)
        })
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

        obs = {'video': np.float32(self.env.reset(**kwargs))/255}  # easier to work w float32
        return obs

    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        obs = {'video': np.float32(obs)/255}  # easier to work w float32
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
    """
    For each obs, returns [n, obs] if keep_frames[obs_name] else [1, obs]
    """
    def __init__(self, env, n, stickprob, keep_frames):
        gym.Wrapper.__init__(self, env)
        self.n = n
        self.stickprob = stickprob
        self.keep_frames = keep_frames
        self.cur_ac = None
        self.rng = np.random.RandomState()
        self.supports_want_render = hasattr(env, "supports_want_render")
        # expand observation space
        self.final_obs = {}
        for obs_name in self.observation_space.spaces.keys():
            cur_obs_space = self.observation_space.spaces[obs_name]
            assert type(cur_obs_space) == gym.spaces.Box
            low = np.unique(cur_obs_space.low)
            high = np.unique(cur_obs_space.high)
            assert len(low) == 1 and len(high) == 1
            new_space_n = self.n if self.keep_frames[obs_name] else 1
            new_space_shape = [new_space_n]+list(cur_obs_space.shape)
            self.observation_space.spaces[obs_name] = gym.spaces.Box(
                low[0], high[0], new_space_shape, cur_obs_space.dtype)
            self.final_obs[obs_name] = np.zeros(new_space_shape)

    def reset(self, **kwargs):
        self.cur_ac = None
        return self.env.reset(**kwargs)

    def step(self, ac):
        done = False
        total_rew = 0
        if self.cur_ac is None:
            # first step after reset, use action
            self.cur_ac = ac
        elif self.rng.rand() > self.stickprob:
            # first substep, delay with probability stickprob
            self.cur_ac = ac
        obs, rew, done, info = self.env.step(self.cur_ac)
        # initialize final obs
        final_obs = self.final_obs.copy()
        for obs_name in obs.keys():
            final_obs[obs_name][0] = obs[obs_name]

        self.cur_ac = ac
        for i in range(1, self.n):
            # second or more substep, use the given action for sure
            obs, rew, done, info = self.env.step(self.cur_ac)
            total_rew += rew
            for obs_name in obs.keys():
                new_obs_i = i if self.keep_frames[obs_name] else 0
                final_obs[obs_name][new_obs_i] = obs[obs_name]
            if done:
                break
        return final_obs, total_rew, done, info

    def seed(self, s):
        self.rng.seed(s)


class RewardScaler(RewardWrapper):
    """
    Bring rewards to a reasonable scale for PPO.
    This is incredibly important and effects performance
    drastically.
    """
    def __init__(self, env, scale):
        super(RewardScaler, self).__init__(env)
        self.scale = scale

    def reward(self, reward):
        return reward * self.scale


class EnvAudio(ObservationWrapper):
    """
    Adds environment audio to observation dict.
    """
    def __init__(self, env):
        super(EnvAudio, self).__init__(env)
        audio_obs_space = gym.spaces.Box(np.float32(-1), np.float32(1), shape=(735,), dtype=np.float32)
        self.observation_space.spaces['audio'] = audio_obs_space

    def observation(self, obs):
        audio = self.em.get_audio()[:735]  # should be 735 samples but sometimes receives 736
        audio = audio.mean(axis=1, dtype=np.float32) / 2**15  # convert to mono and float32
        obs['audio'] = audio
        return obs


class ObsMemoryBuffer(ObservationWrapper):
    """
    Extends current observation with the last n-1 observations.
    """
    def __init__(self, env, memory_len):
        super(ObsMemoryBuffer, self).__init__(env)
        self.obs_mem_buf = {}
        for obs_name in self.observation_space.spaces.keys():
            n = memory_len[obs_name]
            # expand observation space
            cur_obs_space = self.observation_space.spaces[obs_name]
            assert type(cur_obs_space) == gym.spaces.Box
            assert np.unique(cur_obs_space.low) == 1 and np.unique(cur_obs_space.high) == 1
            new_space_shape = [n]+list(cur_obs_space.shape)
            self.observation_space.spaces[obs_name] = gym.spaces.Box(
                cur_obs_space.low[0], cur_obs_space.high[0], new_space_shape, cur_obs_space.dtype)
            # initialize deque
            self.obs_mem_buf[obs_name] = collections.deque(max_len=n)
            # initialize memory buffer with 0 as default obs value
            for i in range(n):
                obs_sample = np.zeros(self.observation_space[obs_name].shape)
                self.obs_mem_buf[obs_name].append(obs_sample)

    def observation(self, obs):
        final_obs = {}
        for obs_name in obs.keys():
            self.obs_mem_buf.popleft()  # remove oldest obs
            self.obs_mem_buf.append(obs[obs_name])  # add current obs
            final_obs[obs_name] = np.concatenate(self.obs_mem_buf[obs_name])
        return final_obs



