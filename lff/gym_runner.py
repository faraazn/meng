import gym
import numpy as np

class GymRunner:
  def __init__(self, env_id, monitor_dir, seed, max_timesteps=100000):
    """
    Initialize a GymRunner object.

    Args:
      env_id: str = the type of OpenAI gym environment to be used
      monitor_dir: str = directory to save gym videos
      seed: int = random seed
      max_timesteps: int = max number of timesteps per episode

    Returns:
      None
    """
    self.max_timesteps = max_timesteps

    self.env = gym.make(env_id)
    self.env.seed(seed)
    if monitor_dir:
      self.env = gym.wrappers.Monitor(self.env, monitor_dir, force=True)

  def run(self, agent, num_episodes, do_train, record_file="", render=False):
    """
    Run the OpenAI gym environment to generate episode data.

    Args:
      agent: Agent = interface that selects actions given environment state
      num_episodes: int = number of episodes to generate
      do_train: bool = True if running inference in training mode
      record_file: str = location to write episode data; do not write if empty

    Returns:
      List of episode dictionaries.
    """
    if record_file:
      with open(record_file, 'w') as f:
        pass

    episodes = []
    for i_episode in range(num_episodes):
      state = self.env.reset()
      ep_info = {
        't_steps':  [],  # time step
        'states':   [],  # state
        'a_probs':  [],  # action probability
        'actions':  [],  # action
        'rewards':  [],  # reward
        'n_states': [],  # next state
        'dones':    []   # done
      }
      for t in range(self.max_timesteps):  # t < 200 for cartpole v0
        a_prob, action = agent.select_action(state, do_train)

        # perform the selected action
        next_state, reward, done, _ = self.env.step(action)
        ep_info['t_steps'].append(t)
        ep_info['states'].append(state)
        ep_info['a_probs'].append(a_prob)
        ep_info['actions'].append(action)
        ep_info['rewards'].append(reward)
        ep_info['n_states'].append(next_state)
        ep_info['dones'].append(done)

        # record step results
        if record_file:
          with open(record_file, 'a') as f:
            f.write(f"{t},{state},{a_probs},{action},{reward},{next_state},{done}\n")

        if render:
          self.env.render()

        state = next_state
        if done:
          episodes.append(ep_info)
          break

    self.env.close()
    return episodes


  def load_episodes(self, record_file):
    """
    Load episodes from a file and return.

    Args:
      record_file: str = location of written episode data.

    Returns:
      List of episode dictionaries.
    """
    episodes = []
    with open(record_file, 'r') as f:
      ep_info = {
        't_steps':  [],
        'states':   [],
        'a_probs':  [],
        'actions':  [],
        'rewards':  [],
        'n_states': [],
        'dones':    []
      }
      for line in f:
        t, state, a_probs, action, reward, next_state, done = line.strip().split(',')
        t = int(t)
        state = np.fromstring(state[1:-1], dtype=float, sep=' ')  # str format: [0 1 2 3]
        a_prob = float(a_prob)
        action = int(action)
        reward = float(reward)
        next_state = np.fromstring(next_state[1:-1], dtype=float, sep=' ')
        done = done == 'True'

        ep_info['t_steps'].append(t)
        ep_info['states'].append(state)
        ep_info['a_probs'].append(a_prob)
        ep_info['actions'].append(action)
        ep_info['rewards'].append(reward)
        ep_info['n_states'].append(next_state)
        ep_info['dones'].append(done)

        if done:
          episodes.append(ep_info)
          ep_info = {
            't_steps':  [],
            'states':   [],
            'a_probs':  [],
            'actions':  [],
            'rewards':  [],
            'n_states': [],
            'dones':    []
          }
    return episodes

  def get_num_actions(self):
    """
    Get the number of possible actions in the environment action space.

    Returns:
      Number of actions.
    """
    return self.env.action_space.n

  def get_reward_threshold(self):
    """
    Get the reward threshold before the task is considered solved.

    Returns:
      Float reward threshold.
    """
    return self.env.reward_threshold
