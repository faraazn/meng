import copy
import glob
import os
import time
import random
from datetime import datetime
from collections import deque

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from a2c_ppo_acktr import algo, utils
from a2c_ppo_acktr.algo import gail
from a2c_ppo_acktr.arguments import get_args
from a2c_ppo_acktr.envs import make_vec_envs
from a2c_ppo_acktr.model import Policy
from a2c_ppo_acktr.storage import RolloutStorage
from evaluation import evaluate

from torch.utils.tensorboard import SummaryWriter

TRAIN_STATES = (
    'GreenHillZone.Act1',  'GreenHillZone.Act3', 'MarbleZone.Act1', 
    'MarbleZone.Act2',     'MarbleZone.Act3',    'SpringYardZone.Act2', 
    'SpringYardZone.Act3', 'LabyrinthZone.Act1', 'LabyrinthZone.Act2', 
    'LabyrinthZone.Act3',  'StarLightZone.Act1', 'StarLightZone.Act2',
    'ScrapBrainZone.Act2', #'ScrapBrainZone.Act3'
)

EVAL_STATES = (
    'GreenHillZone.Act1', 'MarbleZone.Act3', 'LabyrinthZone.Act1', 'StarLightZone.Act2'
)

TEST_STATES = (
    'GreenHillZone.Act2', 'SpringYardZone.Act1', 'StarLightZone.Act3', 'ScrapBrainZone.Act1'
)

def main():
    args = get_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    if args.cuda and torch.cuda.is_available() and args.cuda_deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    # format MM-DD_hh-mm-ss 
    run_name = str(datetime.now())[5:].replace(' ', '_').replace(':', '-').split('.')[0]
    writer_dir = f"runs/test/{run_name}/"
    
    vid_save_dir = f"runs/test/{run_name}/videos/"
    try:
        os.makedirs(vid_save_dir)
    except OSError:
        pass
    
    ckpt_save_dir = f"runs/test/{run_name}/ckpts/"
    try:
        os.makedirs(ckpt_save_dir)
    except OSError:
        pass

    log_dir = os.path.expanduser(args.log_dir)
    utils.cleanup_log_dir(log_dir)

    torch.set_num_threads(1)
    device = torch.device("cuda:0" if args.cuda else "cpu")

    envs = make_vec_envs(TRAIN_STATES, args.seed, args.num_processes,
                         args.gamma, args.log_dir, device, False)

    if args.load:
        actor_critic, env_step, episode_num = torch.load(f'trained_models/{args.load}')
        print(f"loaded model {args.load} at episode {episode_num}")
    else:
        actor_critic = Policy(
            envs.observation_space.shape,
            envs.action_space,
            base_kwargs={'recurrent': args.recurrent_policy})
        episode_num = 0
        env_step = 0
    actor_critic.to(device)
    
    if args.algo == 'a2c':
        agent = algo.A2C_ACKTR(
            actor_critic,
            args.value_loss_coef,
            args.entropy_coef,
            lr=args.lr,
            eps=args.eps,
            alpha=args.alpha,
            max_grad_norm=args.max_grad_norm)
    elif args.algo == 'ppo':
        agent = algo.PPO(
            actor_critic,
            args.clip_param,
            args.ppo_epoch,
            args.num_mini_batch,
            args.value_loss_coef,
            args.entropy_coef,
            lr=args.lr,
            eps=args.eps,
            max_grad_norm=args.max_grad_norm)
    elif args.algo == 'acktr':
        agent = algo.A2C_ACKTR(
            actor_critic, args.value_loss_coef, args.entropy_coef, acktr=True)

    if args.gail:
        assert len(envs.observation_space.shape) == 1
        discr = gail.Discriminator(
            envs.observation_space.shape[0] + envs.action_space.shape[0], 100,
            device)
        file_name = os.path.join(
            args.gail_experts_dir, "trajs_{}.pt".format(
                args.env_name.split('-')[0].lower()))
        
        expert_dataset = gail.ExpertDataset(
            file_name, num_trajectories=4, subsample_frequency=20)
        drop_last = len(expert_dataset) > args.gail_batch_size
        gail_train_loader = torch.utils.data.DataLoader(
            dataset=expert_dataset,
            batch_size=args.gail_batch_size,
            shuffle=True,
            drop_last=drop_last)

    writer = SummaryWriter(log_dir=writer_dir)

    rollouts = RolloutStorage(args.num_steps, args.num_processes,
                              envs.observation_space.shape, envs.action_space,
                              actor_critic.recurrent_hidden_state_size)

    obs = envs.reset()
    rollouts.obs[0].copy_(obs)
    rollouts.to(device)

    episode_rewards = deque(maxlen=10)

    args.num_env_steps = int(args.num_env_steps)
    num_updates = args.num_env_steps // args.num_steps // args.num_processes
    batch_size = args.num_steps * args.num_processes
    start = time.time()
    while env_step < args.num_env_steps:
        s = time.time()
        if args.use_linear_lr_decay:
            # decrease learning rate linearly
            utils.update_linear_schedule(
                agent.optimizer, j, num_updates,
                agent.optimizer.lr if args.algo == "acktr" else args.lr)
        
        for step in range(args.num_steps):
            # Sample actions
            with torch.no_grad():
                value, action, action_log_prob, recurrent_hidden_states = actor_critic.act(
                    rollouts.obs[step], rollouts.recurrent_hidden_states[step],
                    rollouts.masks[step])

            # Observe reward and next obs
            obs, reward, done, infos = envs.step(action)
            for info in infos:
                if 'max_x' in info.keys():
                    episode_rewards.append(info['max_x'])
                if 'episode' in info.keys():
                    writer.add_scalar('episode_max_x', info['max_x'], episode_num)
                    episode_num += 1

            # If done then clean the history of observations.
            masks = torch.FloatTensor(
                [[0.0] if done_ else [1.0] for done_ in done])
            bad_masks = torch.FloatTensor(
                [[0.0] if 'bad_transition' in info.keys() else [1.0]
                 for info in infos])
            rollouts.insert(obs, recurrent_hidden_states, action,
                            action_log_prob, value, reward, masks, bad_masks)
        with torch.no_grad():
            next_value = actor_critic.get_value(
                rollouts.obs[-1], rollouts.recurrent_hidden_states[-1],
                rollouts.masks[-1]).detach()

        if args.gail:
            assert False
            if j >= 10:
                envs.venv.eval()

            gail_epoch = args.gail_epoch
            if j < 10:
                gail_epoch = 100  # Warm up
            for _ in range(gail_epoch):
                discr.update(gail_train_loader, rollouts,
                             utils.get_vec_normalize(envs)._obfilt)

            for step in range(args.num_steps):
                rollouts.rewards[step] = discr.predict_reward(
                    rollouts.obs[step], rollouts.actions[step], args.gamma,
                    rollouts.masks[step])

        rollouts.compute_returns(next_value, args.use_gae, args.gamma,
                                 args.gae_lambda, args.use_proper_time_limits)
        value_loss, action_loss, dist_entropy = agent.update(rollouts)
        rollouts.after_update()

        env_step += batch_size
        
        # save for every interval-th episode or for the last epoch
        fps = batch_size / (time.time()-s)
        writer.add_scalar('fps', fps, env_step)
        writer.add_scalar('value_loss', value_loss / batch_size, env_step)
        writer.add_scalar('action_loss', action_loss / batch_size, env_step)
        writer.add_scalar('dist_entropy', dist_entropy / batch_size, env_step)
        writer.add_scalar('batch_max_x', max(episode_rewards), env_step)
        prev_env_step = max(0, env_step + 1 - batch_size)
        # TODO: fix this env_step+1 condition to actually trigger at last iteration
        if ((env_step+1)//args.save_interval > prev_env_step//args.save_interval or env_step+1 == args.num_env_steps):

            torch.save([
                actor_critic,
                env_step,
                episode_num,
            ], os.path.join(ckpt_save_dir, f"step{env_step}-ep{episode_num}.pt"))
            print(f"Saved model at step {env_step}. Running evaluation.")

            envs.close()
            del envs
            eval_score, e_dict = evaluate(EVAL_STATES, args.seed, device, actor_critic, 10000, env_step, writer, vid_save_dir)
            print(f"  Evaluation score: {eval_score}")
            print(f"    eval ep rewards: {e_dict}")
            writer.add_scalar('eval_score', eval_score, env_step)

            envs = make_vec_envs(TRAIN_STATES, args.seed, args.num_processes, args.gamma, args.log_dir, device, False)
            obs = envs.reset()
            rollouts.obs[0].copy_(obs)
        
        if (env_step+1)//args.log_interval > prev_env_step//args.log_interval and len(episode_rewards) > 1:
            end = time.time()
            print("Env step {} of {}: {:.1f}s, {:.1f}fps".format(
                env_step+1, args.num_env_steps, end-start, fps))
            print("  Last {} episodes: mean/med reward {:.1f}/{:.1f}, min/max reward {:.1f}/{:.1f}".format(
                len(episode_rewards), np.mean(episode_rewards),
                np.median(episode_rewards), np.min(episode_rewards),
                np.max(episode_rewards)))
            print("  dist_entropy {:.5f}, value_loss {:.9f}, action_loss {:.9f}".format(
                dist_entropy, value_loss, action_loss))
            start = time.time()

        
    writer.close()
if __name__ == "__main__":
    main()
