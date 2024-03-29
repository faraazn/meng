import numpy as np
import random
from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

eps = np.finfo(np.float32).eps.item()

class Agent(ABC):
  """
  Agent interface for interacting with GymRunner and main loop.
  """
  @abstractmethod
  def select_action(self, state, do_train=False):
    """
    Provide an action given the environment state.

    Args:
      state: np array = environment state
      do_train: bool = True if running inference in training mode

    Returns:
      Categorical int representing the action taken.
    """
    pass

  @abstractmethod
  def finish_episode(self, states, actions, a_probs, rewards, next_states, dones, i):
    """
    Update agent parameters from data generated by one episode.

    Args:
      states: List[np array] = episode states
      actions: List[int] = episode actions
      a_probs: List[float] = probabilities of episode actions
      rewards: List[float] = episode rewards
      next_states: List[np array] = episode next states
      dones: List[bool] = True if final state of an episode
      i: int = finish_episode iteration, used for logging

    Returns:
      None
    """
    pass


class Policy(nn.Module):
  def __init__(self, num_obs, num_actions, hidden_size, lr):
    """
    Initialize instance of Policy object.

    Args:
      num_obs: int = number of observations the agent makes
      num_actions: int = number of actions the agent can take
      hidden_size: int = hidden layer size
      lr: float = learning rate

    Returns:
      None
    """
    super(Policy, self).__init__()
    self.affine1 = nn.Linear(num_obs, hidden_size)
    self.affine2 = nn.Linear(hidden_size, num_actions)

    self.optimizer = optim.Adam(self.parameters(), lr=lr)

  def forward(self, x):
    """
    Forward pass through the neural network policy.

    Args:
      x: tensor = environment state

    Returns:
      Action probabilities given the environment state.
    """
    x = F.relu(self.affine1(x))
    distribution = F.softmax(self.affine2(x), dim=-1)
    return distribution


class ReinforceAgent(Agent):
  def __init__(self, args, writer, num_obs, num_actions):
    """
    Initialize instance of ReinforceAgent object.

    Args:
      args: ArgumentParser = command line arguments for the agent
      writer: SummaryWriter = tensorboard summary writer for logging data
      num_obs: int = number of observations the agent makes
      num_actions: int = number of actions the agent can take

    Returns:
      None
    """
    super(ReinforceAgent, self).__init__()
    self.writer = writer
    self.policy = Policy(num_obs, num_actions, args.hidden_size, args.lr)

    self.gamma = args.gamma
    self.use_ips = args.use_ips
    self.debug = args.debug

  def select_action(self, state, do_train=False):
    """
    Select an action given the environment state.

    Args:
      state: np array = environment state
      do_train: bool = True if running inference in training mode

    Returns:
      A tuple of the action taken probability, and action taken.
    """
    if do_train:
      self.policy.train()
    else:
      self.policy.eval()
    state = torch.from_numpy(state).float().unsqueeze(0)
    probs = self.policy(state)
    m = Categorical(probs)
    action = m.sample()
    return probs[0][action[0]].squeeze(0).item(), action.item()

  def finish_episode(self, states, actions, a_probs, rewards, next_states, dones, i):
    """
    Update value parameters from data generated by one episode.

    Args:
      states: List[np array] = episode states
      actions: List[int] = episode actions
      a_probs: List[float] = probabilities of episode actions
      rewards: List[float] = episode rewards
      next_states: List[np array] = episode next states
      dones: List[bool] = True if final state of an episode
      i: int = finish_episode iteration, used for logging

    Returns:
      None
    """
    self.policy.train()
    R = 0
    policy_loss = []
    returns = []
    for r in rewards[::-1]:
      R = r + self.gamma * R
      returns.insert(0, R)
    returns = torch.tensor(returns)
    returns = (returns - returns.mean()) / (returns.std() + eps)
    for state, action, a_prob, R in zip(states, actions, a_probs, returns):
      state = torch.from_numpy(state).float().unsqueeze(0)
      action = torch.tensor(action).unsqueeze(0)
      probs = self.policy(state)
      action_prob = probs[:,action] + eps
      log_prob = torch.log(action_prob)
      if self.use_ips:
        # use inverse propensity scoring
        log_prob *= action_prob.detach() / (a_prob + eps)
      policy_loss.append(-log_prob * R)

    self.policy.optimizer.zero_grad()
    policy_loss = torch.cat(policy_loss).sum()
    self.writer.add_scalar('Loss', policy_loss.item(), i)
    policy_loss.backward()
    self.policy.optimizer.step()


class ActionValue(nn.Module):
  def __init__(self, num_obs, num_actions, hidden_size, lr):
    """
    Initialize instance of ActionValue object.

    Args:
      num_obs: int = number of observations the agent makes
      num_actions: int = number of actions the agent can take
      hidden_size: int = hidden layer size
      lr: float = learning rate

    Returns:
      None
    """
    super(ActionValue, self).__init__()
    self.affine1 = nn.Linear(num_obs, hidden_size)
    self.affine2 = nn.Linear(hidden_size, num_actions)

    self.optimizer = optim.Adam(self.parameters(), lr=lr)

  def forward(self, x):
    """
    Forward pass to get a value given the state.

    Args:
      x: tensor = environment state

    Returns:
      Estimated value array of actions taken in this environment state.
    """
    x = F.relu(self.affine1(x))
    value = self.affine2(x)
    return value


class QLearningAgent(Agent):
  def __init__(self, args, writer, num_obs, num_actions):
    """
    Initialize instance of QLearningAgent object.

    Args:
      args: ArgumentParser = command line arguments for the agent
      writer: SummaryWriter = tensorboard summary writer for logging data
      num_obs: int = number of observations the agent makes
      num_actions: int = number of actions the agent can take

    Returns:
      None
    """
    super(QLearningAgent, self).__init__()
    self.writer = writer
    self.value = ActionValue(num_obs, num_actions, args.hidden_size, args.lr)
    self.num_actions = num_actions

    # hyperparameters
    self.gamma = args.gamma  # reward discount factor
    self.explore_rate = 0.1  # exploration rate
    self.debug = args.debug  # True if in debug mode

  def select_action(self, state, do_train=False):
    """
    Select an action given the environment state.

    Args:
      state: np array = environment state
      do_train: bool = True if running inference in training mode

    Returns:
      A tuple of the action taken probability, and action taken.
    """
    if do_train:
      self.value.train()
    else:
      self.value.eval()

    if do_train and np.random.rand() <= self.explore_rate:
      # explore random action
      return self.explore_rate/self.num_actions, random.randrange(self.num_actions)
    else:
      # select best action
      state = torch.from_numpy(state).float().unsqueeze(0)
      action_values = self.value(state).detach().numpy()
      return (1-self.explore_rate), np.argmax(action_values)

  def finish_episode(self, states, actions, a_probs, rewards, next_states, dones, i):
    """
    Update value parameters from data generated by one episode.

    Args:
      states: List[np array] = episode states
      actions: List[int] = episode actions
      a_probs: List[float] = probabilities of episode actions
      rewards: List[float] = episode rewards
      next_states: List[np array] = episode next states
      dones: List[bool] = True if final state of an episode
      i: int = finish_episode iteration, used for logging

    Returns:
      None
    """
    self.value.train()
    value_loss = []
    for state, action, reward, next_state, done in zip(
        states, actions, rewards, next_states, dones):
      if done:
        target = reward
      else:
        next_state = torch.from_numpy(next_state).float().unsqueeze(0)
        action_values = self.value(next_state)[0].detach().numpy()
        target = reward + self.gamma * np.amax(action_values)
      state = torch.from_numpy(state).float().unsqueeze(0)
      target_f = self.value(state).detach()
      target_f[0][action] = target
      prediction = self.value(state)
      loss = F.mse_loss(prediction, target_f).unsqueeze(0)
      value_loss.append(loss)

    self.value.optimizer.zero_grad()
    value_loss = torch.cat(value_loss).sum()
    self.writer.add_scalar('Loss', value_loss.item(), i)
    value_loss.backward()
    self.value.optimizer.step()


class RandomAgent(Agent):
  def __init__(self, args, writer, num_obs, num_actions):
    """
    Initialize instance of RandomAgent object.

    Args:
      args: ArgumentParser = command line arguments for the agent
      writer: SummaryWriter = tensorboard summary writer for logging data
      num_obs: int = number of observations the agent makes
      num_actions: int = number of actions the agent can take

    Returns:
      None
    """
    super(RandomAgent, self).__init__()
    self.num_actions = num_actions

  def select_action(self, state, do_train=False):
    """
    Select an action given the environment state.

    Args:
      state: np array = environment state
      do_train: bool = True if running inference in training mode

    Returns:
      A tuple of the action taken probability, and action taken.
    """
    return 1.0/self.num_actions, random.randint(0, self.num_actions-1)

  def finish_episode(self, states, actions, a_probs, rewards, next_states, dones, i):
    """
    Update value parameters from data generated by one episode.

    Args:
      states: List[np array] = episode states
      actions: List[int] = episode actions
      a_probs: List[float] = probabilities of episode actions
      rewards: List[float] = episode rewards
      next_states: List[np array] = episode next states
      dones: List[bool] = True if final state of an episode
      i: int = finish_episode iteration, used for logging

    Returns:
      None
    """
    pass


class FixedAgent(Agent):
  def __init__(self, args, writer, num_obs, num_actions):
    """
    Initialize instance of FixedAgent object.

    Args:
      args: ArgumentParser = command line arguments for the agent
      writer: SummaryWriter = tensorboard summary writer for logging data
      num_obs: int = number of observations the agent makes
      num_actions: int = number of actions the agent can take

    Returns:
      None
    """
    super(FixedAgent, self).__init__()
    self.fixed_action = 0

  def select_action(self, state, do_train=False):
    """
    Select an action given the environment state.

    Args:
      state: np array = environment state
      do_train: bool = True if running inference in training mode

    Returns:
      A tuple of the action taken probability, and action taken.
    """
    return 1.0, self.fixed_action

  def finish_episode(self, states, actions, a_probs, rewards, next_states, dones, i):
    """
    Update value parameters from data generated by one episode.

    Args:
      states: List[np array] = episode states
      actions: List[int] = episode actions
      a_probs: List[float] = probabilities of episode actions
      rewards: List[float] = episode rewards
      next_states: List[np array] = episode next states
      dones: List[bool] = True if final state of an episode
      i: int = finish_episode iteration, used for logging

    Returns:
      None
    """
    pass


class Actor(nn.Module):
  def __init__(self, num_obs, num_actions, hidden_size, lr):
    """
    Initialize instance of Actor object.

    Args:
      num_obs: int = number of observations the agent makes
      num_actions: int = number of actions the agent can take
      hidden_size: int = hidden layer size
      lr: float = learning rate

    Returns:
      None
    """
    super(Actor, self).__init__()
    self.linear1 = nn.Linear(num_obs, hidden_size)
    self.linear2 = nn.Linear(hidden_size, num_actions)

    self.optimizer = optim.Adam(self.parameters(), lr=lr)

  def forward(self, x):
    """
    Forward pass through the neural network policy.

    Args:
      x: tensor = environment state

    Returns:
      Action probabilities given the environment state.
    """
    x = F.relu(self.linear1(x))
    distribution = F.softmax(self.linear2(x), dim=-1)
    return distribution


class StateCritic(nn.Module):
  def __init__(self, num_obs, hidden_size, lr):
    """
    Initialize instance of StateCritic object.

    Args:
      num_obs: int = number of observations the agent makes
      hidden_size: int = hidden layer size
      lr: float = learning rate

    Returns:
      None
    """
    super(StateCritic, self).__init__()
    self.linear1 = nn.Linear(num_obs, hidden_size)
    self.linear2 = nn.Linear(hidden_size, 1)

    self.optimizer = optim.Adam(self.parameters(), lr=lr)

  def forward(self, x):
    """
    Forward pass through the neural network.

    Args:
      x: tensor = environment state

    Returns:
      Value given the environment state.
    """
    x = F.relu(self.linear1(x))
    value = self.linear2(x)
    return value


class ActionCritic(nn.Module):
  def __init__(self, num_obs, num_actions, hidden_size, lr):
    """
    Initialize instance of ActionCritic object.

    Args:
      num_obs: int = number of observations the agent makes
      num_actions: int = number of actions the agent can take
      hidden_size: int = hidden layer size
      lr: float = learning rate

    Returns:
      None
    """
    super(ActionCritic, self).__init__()
    self.linear1 = nn.Linear(num_obs, hidden_size)
    self.linear2 = nn.Linear(hidden_size, num_actions)

    self.optimizer = optim.Adam(self.parameters(), lr=lr)

  def forward(self, x):
    """
    Forward pass through the neural network.

    Args:
      x: tensor = environment state

    Returns:
      Action values given the environment state.
    """
    x = F.relu(self.linear1(x))
    value = self.linear2(x)
    return value


class ActorCriticAgent(Agent):
  def __init__(self, args, writer, num_obs, num_actions):
    """
    Initialize instance of ActorCriticAgent object.

    Args:
      args: ArgumentParser = command line arguments for the agent
      writer: SummaryWriter = tensorboard summary writer for logging data
      num_obs: int = number of observations the agent makes
      num_actions: int = number of actions the agent can take

    Returns:
      None
    """
    super(ActorCriticAgent, self).__init__()
    self.writer = writer
    self.actor = Actor(num_obs, num_actions, args.hidden_size, args.lr)
    self.critic = StateCritic(num_obs, args.hidden_size, args.lr)

    self.gamma = args.gamma
    self.use_ips = args.use_ips
    self.debug = args.debug

  def select_action(self, state, do_train=False):
    """
    Select an action given the environment state.

    Args:
      state: np array = environment state
      do_train: bool = True if running inference in training mode

    Returns:
      A tuple of the action taken probability, and action taken.
    """
    if do_train:
      self.actor.train()
    else:
      self.actor.eval()
    state = torch.from_numpy(state).float().unsqueeze(0)
    probs = self.actor(state)
    m = Categorical(probs)
    action = m.sample()
    return probs[0][action[0]].squeeze(0).item(), action.item()

  def finish_episode(self, states, actions, a_probs, rewards, next_states, dones, i):
    """
    Update value parameters from data generated by one episode.

    Args:
      states: List[np array] = episode states
      actions: List[int] = episode actions
      a_probs: List[float] = probabilities of episode actions
      rewards: List[float] = episode rewards
      next_states: List[np array] = episode next states
      dones: List[bool] = True if final state of an episode
      i: int = finish_episode iteration, used for logging

    Returns:
      None
    """
    self.actor.train()
    self.critic.train()
    R = 0
    actor_loss = []
    critic_loss = []
    returns = []
    for r in rewards[::-1]:
      R = r + self.gamma * R
      returns.insert(0, R)
    returns = torch.tensor(returns)
    returns = (returns - returns.mean()) / (returns.std() + eps)
    for state, action, a_prob, R in zip(states, actions, a_probs, returns):
      state = torch.from_numpy(state).float().unsqueeze(0)
      action = torch.tensor(action).unsqueeze(0)
      probs = self.actor(state)
      action_prob = probs[:,action] + eps
      log_prob = torch.log(action_prob)
      if self.use_ips:
        # use inverse propensity scoring
        log_prob *= action_prob.detach() / (a_prob + eps)
      value = self.critic(state)
      advantage = R - value
      actor_loss.append(-log_prob * advantage.detach())
      critic_loss.append(advantage.pow(2))

    actor_loss = torch.cat(actor_loss).sum()
    critic_loss = torch.cat(critic_loss).sum()
    total_loss = actor_loss + critic_loss
    self.writer.add_scalar('Actor Loss', actor_loss.item(), i)
    self.writer.add_scalar('Critic Loss', critic_loss.item(), i)
    self.writer.add_scalar('Loss', total_loss.item(), i)

    self.actor.optimizer.zero_grad()
    self.critic.optimizer.zero_grad()
    actor_loss.backward()
    critic_loss.backward()
    self.actor.optimizer.step()
