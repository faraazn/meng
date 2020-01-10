import argparse
import numpy as np
import random

import torch
from torch.utils.tensorboard import SummaryWriter

from gym_runner import GymRunner
from agents import ActorCriticAgent, ReinforceAgent, QLearningAgent, RandomAgent, FixedAgent

parser = argparse.ArgumentParser(description='Offline Agent Reinforcement Learning')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor (default: 0.99)')
parser.add_argument('--seed', type=int, default=543, metavar='N',
                    help='random seed (default: 543)')
parser.add_argument('--render', action='store_true',
                    help='render the environment')
parser.add_argument('--record', type=str, default="", metavar='F',
                    help='save records to this location')
parser.add_argument('--monitor_dir', type=str, default="", metavar='F',
                    help='save monitors to this directory')
parser.add_argument('--save_model', type=str, default="", metavar='F',
                    help='save weights to this location')
parser.add_argument('--num_episodes', type=int, default=200, metavar='N',
                    help='number of episodes to run per policy (default: 10)')
parser.add_argument('--num_agents', type=int, default=1, metavar='N',
                    help='number of agents to train (default: 1)')
parser.add_argument('--hidden_size', type=int, default=128, metavar='H',
                    help='size of hidden layer (default: 128)')
parser.add_argument('--use_ips', action='store_true',
                    help='use ips reweighting')
parser.add_argument('--debug', action='store_true',
                    help='run in debug mode')
parser.add_argument('--lr', type=float, default=1e-2, metavar='L',
                    help='learning rate (default: 1e-2)')
parser.add_argument('--agent', type=str, default='ActorCritic', metavar='A',
                    help='agent type (ActorCritic, Reinforce, QLearning, Random, Fixed)')
parser.add_argument('--online', action='store_true',
                    help='run the agent online rather than offline')
parser.add_argument('--env', type=str, default="CartPole-v0", metavar='E',
                    help='openAI gym environment (default: CartPole-v0)')
parser.add_argument('--render_end', action='store_true',
                    help='render 3 episodes of the final model')


def main():
  args = parser.parse_args()
  random.seed(args.seed)
  torch.manual_seed(args.seed)
  torch.cuda.manual_seed(args.seed)
  np.random.seed(args.seed)

  writer = SummaryWriter()
  gr = GymRunner(args.env, args.monitor_dir, args.seed)
  print(f"Reward threshold: {gr.env.spec.reward_threshold}")

  if args.agent.lower() == 'actorcritic':
    agent_class = ActorCriticAgent
  elif args.agent.lower() == 'reinforce':
    agent_class = ReinforceAgent
  elif args.agent.lower() == 'qlearning':
    agent_class = QLearningAgent
  elif args.agent.lower() == 'random':
    agent_class = RandomAgent
  elif args.agent.lower() == 'fixed':
    agent_class = FixedAgent
  else:
    raise ValueError('Invalid agent type was specified.')
  agent = agent_class(args, writer, gr.env.observation_space.shape[0],
                      gr.env.action_space.n)

  all_episodes = []
  all_rewards = []
  ep_count = 0
  for i in range(args.num_agents):
    if args.online:
      # use running average of past 100 episodes to evaluate model
      if len(all_rewards) == 0:
        # generate 100 episodes to evaluate model
        eval_episodes = gr.run(
          agent, 100, False, record_file=args.record, render=args.render)
        for ep_info in eval_episodes:
          all_rewards.append(sum(ep_info['rewards']))
      assert len(all_rewards) >= 100
      avg_rewards = sum(all_rewards[-100:]) / 100
    else:
      # generate 100 episodes to evaluate model
      eval_episodes = gr.run(
        agent, 100, False, record_file=args.record, render=args.render)
      sum_rewards = 0
      for ep_info in eval_episodes:
        sum_rewards += sum(ep_info['rewards'])
      avg_rewards = sum_rewards / 100

    writer.add_scalar('Reward', avg_rewards, i)
    if avg_rewards > gr.env.spec.reward_threshold:
      print("Solved! Average reward is now {}!".format(avg_rewards))
      break
    else:
      print("{}/{}\tAverage reward: {:.2f}".format(
        i+1, args.num_agents, avg_rewards))
 
    # generate episodes to train model
    train_episodes = gr.run(
      agent, args.num_episodes, True, record_file=args.record, render=args.render)

    if args.online:
      # run the agent online
      all_episodes = train_episodes
    else:
      # reinitialize the agent and run offline
      all_episodes.extend(train_episodes)
      random.shuffle(all_episodes)
      agent = agent_class(args, writer, gr.env.observation_space.shape[0],
                          gr.env.action_space.n)

    for ep_info in all_episodes:
      states = ep_info['states']
      a_probs = ep_info['a_probs']
      actions = ep_info['actions']
      rewards = ep_info['rewards']
      next_states = ep_info['n_states']
      dones = ep_info['dones']

      all_rewards.append(sum(rewards))
      agent.finish_episode(
        states, actions, a_probs, rewards, next_states, dones, ep_count)
      ep_count += 1

  if args.save_model:
    # save agent to file
    torch.save(agent, args.save_model)
  writer.close()

  if args.render_end:
    gr.run(agent, 3, False, record_file=args.record, render=True)

if __name__ == '__main__':
  main()
