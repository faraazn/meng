import copy
import glob
import os
import time
import random
import glob
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
from a2c_ppo_acktr.envs import make_vec_envs
from a2c_ppo_acktr.model import Policy
from a2c_ppo_acktr.storage import RolloutStorage

from evaluation import evaluate

from torch.utils.tensorboard import SummaryWriter

ID_2_ZONE = {
    0: 'GreenHillZone', 1: 'MarbleZone', 2: 'SpringYardZone', 
    3: 'LabyrinthZone', 4: 'StarLightZone', 5: 'ScrapBrainZone'
}

screen_x_end = {}

def train(train_states, run_dir, args, num_env_steps, eval_env_steps, device, writer, writer_name, init_model=None):
    envs = make_vec_envs(train_states, args.seed, args.num_processes,
                         args.gamma, device, False, 'train')

    if init_model:
        actor_critic, env_step, episode_num, model_name = init_model
        print(f"  [load] Loaded model {model_name} at step {env_step}")
    else:
        actor_critic = Policy(
            envs.observation_space.shape,
            envs.action_space,
            base_kwargs={'recurrent': args.recurrent_policy})
        env_step = 0
        episode_num = 0
    actor_critic.to(device)
    
    run_name = run_dir.replace('/', '_')
    vid_save_dir = f"{run_dir}/videos/"
    try:
        os.makedirs(vid_save_dir)
    except OSError:
        pass
    ckpt_save_dir = f"{run_dir}/ckpts/"
    try:
        os.makedirs(ckpt_save_dir)
    except OSError:
        pass

    if args.algo == 'ppo':
        agent = algo.PPO(
            actor_critic, args.clip_param, args.ppo_epoch, args.num_mini_batch,
            args.value_loss_coef, args.entropy_coef, lr=args.lr, eps=args.eps,
            max_grad_norm=args.max_grad_norm)
    elif args.algo == 'rainbow':
        assert False, 'Rainbow not implemented yet.'
    elif args.algo == 'jerk':
        assert False, 'JERK not implemented yet.'
    else:
        assert False, f'Invalid algorithm provided.'

    rollouts = RolloutStorage(args.num_steps, args.num_processes,
                              envs.observation_space.shape, envs.action_space,
                              actor_critic.recurrent_hidden_state_size)

    obs = envs.reset()
    rollouts.obs[0].copy_(obs)
    rollouts.to(device)

    episode_rewards = deque(maxlen=10)

    num_updates = num_env_steps // args.num_steps // args.num_processes
    batch_size = args.num_steps * args.num_processes
    start = time.time()
    while env_step < num_env_steps:
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
            obs, reward, dones, infos = envs.step(action)
            for done, info in zip(dones, infos):
                env_state = f"{ID_2_ZONE[info['zone']]}.Act{info['act']+1}"
                if env_state not in screen_x_end and info['screen_x_end'] > 0:
                    screen_x_end[env_state] = info['screen_x_end']
                if 'max_x' in info.keys():
                    episode_rewards.append(info['max_x'])
                if done:
                    writer.add_scalar(f'train_episode_x/{env_state}', info['max_x'], env_step)
                    writer.add_scalar(f'train_episode_%/{env_state}', info['max_x']/screen_x_end[env_state]*100, env_step)
                    episode_num += 1

            # If done then clean the history of observations.
            masks = torch.FloatTensor(
                [[0.0] if done else [1.0] for done in dones])
            bad_masks = torch.FloatTensor(
                [[0.0] if 'bad_transition' in info.keys() else [1.0]
                 for info in infos])
            rollouts.insert(obs, recurrent_hidden_states, action,
                            action_log_prob, value, reward, masks, bad_masks)
        with torch.no_grad():
            next_value = actor_critic.get_value(
                rollouts.obs[-1], rollouts.recurrent_hidden_states[-1],
                rollouts.masks[-1]).detach()

        rollouts.compute_returns(next_value, args.use_gae, args.gamma,
                                 args.gae_lambda, args.use_proper_time_limits)
        value_loss, action_loss, dist_entropy = agent.update(rollouts)
        rollouts.after_update()

        env_step += batch_size
        
        # write training metrics for each batch
        fps = batch_size / (time.time()-s)
        writer.add_scalar(f'fps/{writer_name}', fps, env_step)
        writer.add_scalar(f'value_loss/{writer_name}', value_loss / batch_size, env_step)
        writer.add_scalar(f'action_loss/{writer_name}', action_loss / batch_size, env_step)
        writer.add_scalar(f'dist_entropy/{writer_name}', dist_entropy / batch_size, env_step)
        total_norm = 0
        for p in list(filter(lambda p: p.grad is not None, actor_critic.parameters())):
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
        total_norm = total_norm ** (1. / 2)
        writer.add_scalar(f'grad_norm/{writer_name}', total_norm, env_step)
        prev_env_step = max(0, env_step + 1 - batch_size)
        
        # print log to console
        if (env_step+1)//args.log_interval > prev_env_step//args.log_interval and len(episode_rewards) > 1:
            end = time.time()
            print("  [log] Env step {} of {}: {:.1f}s, {:.1f}fps".format(
                env_step+1, num_env_steps, end-start, fps))
            print("    Last {} episodes: mean/med reward {:.1f}/{:.1f}, min/max reward {:.1f}/{:.1f}".format(
                len(episode_rewards), np.mean(episode_rewards),
                np.median(episode_rewards), np.min(episode_rewards),
                np.max(episode_rewards)))
            print("    dist_entropy {:.5f}, value_loss {:.9f}, action_loss {:.9f}".format(
                dist_entropy, value_loss, action_loss))
            start = time.time()

        # save model to ckpt and run evaluation
        if ((env_step+1)//args.save_interval > prev_env_step//args.save_interval):

            torch.save([
                actor_critic,
                env_step,
                episode_num,
                run_name
            ], os.path.join(ckpt_save_dir, f"{run_name}-{env_step}.pt"))
            print(f"  [save] Saved model at step {env_step+1}.")

            envs.close()
            del envs  # close does not actually get rid of envs, need to del
            eval_score, e_dict = evaluate(train_states, args.seed, device, actor_critic, 10000, env_step, writer, vid_save_dir)
            print(f"  [eval] Evaluation score: {eval_score}")
            writer.add_scalar('eval_score', eval_score, env_step)

            envs = make_vec_envs(train_states, args.seed, args.num_processes, args.gamma, device, False, 'train')
            obs = envs.reset()
            rollouts.obs[0].copy_(obs)

    # final model save
    torch.save([
        actor_critic,
        env_step,
        episode_num,
        run_name
    ], os.path.join(ckpt_save_dir, f"{run_name}-{env_step}.pt"))
    print(f"  [save] Final model saved at step {env_step+1}.")

    # final model eval
    envs.close()
    del envs
    eval_score, eval_dict = evaluate(train_states, args.seed, device, actor_critic, eval_env_steps, env_step, writer, vid_save_dir)
    print(f"  [eval] Final model evaluation score: {eval_score}")

    return (actor_critic, env_step, episode_num, run_name), eval_score, eval_dict
