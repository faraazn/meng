import copy
import glob
import os
import time
import random
import glob
from datetime import datetime
from collections import deque
import psutil
#import nvidia_smi
#nvidia_smi.nvmlInit()
#handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0)

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


def train(train_states, run_dir, num_env_steps, eval_env_steps, writer, writer_name, args, init_model=None):
    envs = make_vec_envs(train_states, args.seed, args.num_processes, args.gamma, 'cpu', 'train', args)

    if init_model:
        actor_critic, env_step, model_name = init_model
        obs_space = actor_critic.obs_space
        obs_process = actor_critic.obs_process
        obs_module = actor_critic.obs_module
        print(f"  [load] Loaded model {model_name} at step {env_step}")
    else:
        obs_space = envs.observation_space
        actor_critic = Policy(
            obs_space,
            args.obs_process,
            args.obs_module,
            envs.action_space,
            base_kwargs={'recurrent': args.recurrent_policy})
        env_step = 0
    actor_critic.to(args.device)
    #print(actor_critic)

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
            args.value_loss_coef, args.entropy_coef, args.device, lr=args.lr, eps=args.eps,
            max_grad_norm=args.max_grad_norm)
    elif args.algo == 'a2c':
        agent = algo.A2C_ACKTR(
            actor_critic, args.value_loss_coef, args.entropy_coef, lr=args.lr,
            eps=args.eps, alpha=args.alpha, max_grad_norm=args.max_grad_norm,
            acktr=False)
    elif args.algo == 'acktr':
        agent = algo.A2C_ACKTR(
            actor_critic, args.value_loss_coef, args.entropy_coef, lr=args.lr,
            eps=args.eps, alpha=args.alpha, max_grad_norm=args.max_grad_norm,
            acktr=True)
    else:
        raise NotImplementedError

    rollouts = RolloutStorage(args.num_steps, args.num_processes,
                              envs.observation_space, envs.action_space,
                              actor_critic.recurrent_hidden_state_size)

    obs = envs.reset()
    actor_critic.eval()
    """
    try:
        writer.add_graph(actor_critic, obs)
    except ValueError:
        print("Unable to write model graph to tensorboard.")
    """
    actor_critic.train()
    
    for k in rollouts.obs.keys():
        rollouts.obs[k][0].copy_(obs[k][0])

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
                value, action, action_log_prob, recurrent_hidden_states, _ = actor_critic.act(
                    {k: rollouts.obs[k][step].float().to(args.device) for k in rollouts.obs.keys()},
                    rollouts.recurrent_hidden_states[step].to(args.device), rollouts.masks[step].to(args.device))
                value = value.cpu()
                action = action.cpu()
                action_log_prob = action_log_prob.cpu()
                recurrent_hidden_states = recurrent_hidden_states.cpu()
            # Observe reward and next obs
            obs, reward, dones, infos = envs.step(action)
            
            for done, info in zip(dones, infos):
                env_state = info['env_state'][1]
                if done:
                    writer.add_scalar(f'train_episode_x/{env_state}', info['max_x'], env_step)
                    writer.add_scalar(f'train_episode_%/{env_state}', info['max_x']/info['lvl_max_x']*100, env_step)
                    writer.add_scalar(f'train_episode_r/{env_state}', info['sum_r'], env_step)

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
                {k: rollouts.obs[k][-1].float().to(args.device) for k in rollouts.obs.keys()},
                rollouts.recurrent_hidden_states[-1].to(args.device), rollouts.masks[-1].to(args.device)).detach().cpu()


        rollouts.compute_returns(next_value, args.use_gae, args.gamma,
                                 args.gae_lambda, args.use_proper_time_limits)
        value_loss, action_loss, dist_entropy = agent.update(rollouts)
        rollouts.after_update()

        env_step += batch_size
        fps = batch_size / (time.time()-s)
        #res = nvidia_smi.nvmlDeviceGetUtilizationRates(handle)
        #writer.add_scalar(f'gpu_usage/{writer_name}', res.gpu, env_step)
        #writer.add_scalar(f'gpu_mem/{writer_name}', res.memory, env_step)
        total_norm = 0
        for p in list(filter(lambda p: p.grad is not None, actor_critic.parameters())):
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
        total_norm = total_norm ** (1. / 2)
        obs_norm = {}
        for obs_name in args.obs_keys:
            t_norm = 0
            if obs_name == 'video':
                md = actor_critic.base.video_module
            elif obs_name == 'audio':
                md = actor_critic.base.audio_module
            else:
                raise NotImplementedError
            for p in list(filter(lambda p: p.grad is not None, md.parameters())):
                param_norm = p.grad.data.norm(2)
                t_norm += param_norm.item() ** 2
            obs_norm[obs_name] = t_norm ** (1. / 2)
        
        prev_env_step = max(0, env_step + 1 - batch_size)
        # write training metrics for this batch, usually takes 0.003s
        if (env_step+1)//args.write_interval > prev_env_step//args.write_interval:
            writer.add_scalar(f'grad_norm/{writer_name}', total_norm, env_step)
            writer.add_scalar(f'fps/{writer_name}', fps, env_step)
            writer.add_scalar(f'value_loss/{writer_name}', value_loss / batch_size, env_step)
            writer.add_scalar(f'action_loss/{writer_name}', action_loss / batch_size, env_step)
            writer.add_scalar(f'dist_entropy/{writer_name}', dist_entropy / batch_size, env_step)
            writer.add_scalar(f'cpu_usage/{writer_name}', psutil.cpu_percent(), env_step)
            writer.add_scalar(f'cpu_mem/{writer_name}', psutil.virtual_memory()._asdict()['percent'], env_step)
            for obs_name in args.obs_keys:
                writer.add_scalar(f'grad_norm_{obs_name}/{writer_name}', obs_norm[obs_name], env_step)
        
        # print log to console
        if (env_step+1)//args.log_interval > prev_env_step//args.log_interval:
            end = time.time()
            print("  [log] Env step {} of {}: {:.1f}s, {:.1f}fps".format(
                env_step+1, num_env_steps, end-start, fps))
            if len(episode_rewards) > 0:
                print("    Last {} episodes: mean/med reward {:.1f}/{:.1f}, min/max reward {:.1f}/{:.1f}".format(
                    len(episode_rewards), np.mean(episode_rewards),
                    np.median(episode_rewards), np.min(episode_rewards),
                    np.max(episode_rewards)))
            print("    dist_entropy {:.5f}, value_loss {:.6f}, action_loss {:.6f}, grad_norm {:.6f}".format(
                dist_entropy, value_loss, action_loss, total_norm))
            start = time.time()

        # save model to ckpt
        if ((env_step+1)//args.save_interval > prev_env_step//args.save_interval):
            torch.save([
                actor_critic,
                env_step,
                run_name,
            ], os.path.join(ckpt_save_dir, f"{run_name}-{env_step}.pt"))
            print(f"  [save] Saved model at step {env_step+1}.")

        # save model to ckpt and run evaluation if eval_interval and not final iteration in training loop
        if ((env_step+1)//args.eval_interval > prev_env_step//args.eval_interval) and env_step < num_env_steps and eval_env_steps > 0:
            torch.save([
                actor_critic,
                env_step,
                run_name,
            ], os.path.join(ckpt_save_dir, f"{run_name}-{env_step}.pt"))
            print(f"  [save] Saved model at step {env_step+1}.")
            
            envs.close()
            del envs  # close does not actually get rid of envs, need to del
            actor_critic.eval()
            eval_score, e_dict = evaluate(
                train_states, actor_critic, eval_env_steps, env_step, writer,
                vid_save_dir, args.vid_tb_steps, args.vid_file_steps, args.obs_viz_layer, args)
            print(f"  [eval] Evaluation score: {eval_score}")
            writer.add_scalar('eval_score', eval_score, env_step)

            actor_critic.train()
            envs = make_vec_envs(train_states, args.seed, args.num_processes, args.gamma, 'cpu', 'train', args)
            obs = envs.reset()
            # TODO: does this work? do we need to increment env step or something? whydden_states insert at 0
            for k in rollouts.obs.keys():
                rollouts.obs[k][0].copy_(obs[k][0])

    # final model save
    final_model_path = os.path.join(ckpt_save_dir, f"{run_name}-{env_step}.pt")
    torch.save([
        actor_critic,
        env_step,
        run_name,
    ], final_model_path)
    print(f"  [save] Final model saved at step {env_step+1} to {final_model_path}")

    # final model eval
    envs.close()
    del envs
    eval_score = None
    eval_dict = None
    if eval_env_steps > 0:
        eval_score, eval_dict = evaluate(
            train_states, actor_critic, eval_env_steps, env_step, writer,
            vid_save_dir, args.vid_tb_steps, args.vid_file_steps, args.obs_viz_layer, args)
        print(f"  [eval] Final model evaluation score: {eval_score:.3f}")

    return (actor_critic, env_step, run_name), eval_score, eval_dict
