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
from a2c_ppo_acktr.arguments import get_args

from evaluation import evaluate
from train import train
from constants import ALL_STATES, TRAIN_STATES, EVAL_STATES, TEST_STATES

from torch.utils.tensorboard import SummaryWriter


def get_newest_ckpt(load_dir):
    try:
        _, _, ckpt_names = next(os.walk(load_dir))
    except StopIteration:
        return None

    newest_ckpt = None
    for ckpt_name in ckpt_names:
        ckpt = torch.load(os.path.join(load_dir, ckpt_name))
        if newest_ckpt is None or ckpt[1] > newest_ckpt[1]:  # index 1 is ckpt env_step
            newest_ckpt = ckpt
    return newest_ckpt

def main():
    args = get_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.set_num_threads(1)
    random.seed(args.seed)
    np.random.seed(args.seed)

    if args.cuda and torch.cuda.is_available() and args.cuda_deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        
    # format MM-DD_hh-mm-ss 
    run_name = str(datetime.now())[5:].replace(' ', '_').replace(':', '-').split('.')[0]

    # ppo-joint training
    writer = SummaryWriter(log_dir=f"runs/{run_name}/ppo-joint/")
    print(f"\n[train] Starting ppo-joint training")
    init_model = get_newest_ckpt(os.path.join(f"{args.load}", "ppo-joint/ckpts/"))
    joint_model, eval_score, _ = train(
        TRAIN_STATES, f"runs/{run_name}/ppo-joint", args.joint_train_steps, args.joint_eval_steps, writer, 'joint', args, init_model)
    writer.close()
    
    # fine tune ppo-joint on test set
    writer = SummaryWriter(log_dir=f"runs/{run_name}/ppo-joint-test/")
    for i, env_state in enumerate(TEST_STATES):
        print(f"\n[train] {i+1}/{len(TEST_STATES)}: Starting {env_state[0]} {env_state[1]} ppo-joint-test training")
        init_model = get_newest_ckpt(os.path.join(f"{args.load}", f"ppo-joint-test/{env_state[1]}/ckpts/"))
        if init_model is None:
            init_model = list(joint_model)
            print(f"  Fine tuning joint model at step {init_model[1]}")
            init_model[1] = 0  # reset step to 0
        model, eval_score, _ = train(
            [env_state], f"runs/{run_name}/ppo-joint-test/{env_state[1]}", args.solo_train_steps, args.solo_eval_steps, writer, env_state[1], args, init_model)
    writer.close()

    # fine tune ppo-joint on train set
    writer = SummaryWriter(log_dir=f"runs/{run_name}/ppo-joint-train/")
    for i, env_state in enumerate(TRAIN_STATES):
        print(f"\n[train] {i+1}/{len(TRAIN_STATES)}: Starting {env_state[0]} {env_state[1]} ppo-joint-train training")
        init_model = get_newest_ckpt(os.path.join(f"{args.load}", f"ppo-joint-train/{env_state[1]}/ckpts/"))
        if init_model is None:
            init_model = list(joint_model)
            print(f"  Fine tuning joint model at step {init_model[1]}")
            init_model[1] = 0  # reset step to 0
        model, eval_score, _ = train(
            [env_state], f"runs/{run_name}/ppo-joint-train/{env_state[1]}", args.solo_train_steps, args.solo_eval_steps, writer, env_state[1], args, init_model)
    writer.close()
    
    # ppo-solo training
    writer = SummaryWriter(log_dir=f"runs/{run_name}/ppo-solo/")
    for i, env_state in enumerate(ALL_STATES):
        print(f"\n[train] {i+1}/{len(ALL_STATES)}: Starting {env_state[0]} {env_state[1]} ppo-solo training")
        init_model = get_newest_ckpt(os.path.join(f"{args.load}", f"ppo-solo/{env_state[1]}/ckpts/"))
        model, eval_score, _ = train(
            [env_state], f"runs/{run_name}/ppo-solo/{env_state[1]}", args.solo_train_steps, args.solo_eval_steps, writer, env_state[1], args, init_model)
    writer.close()
    

if __name__ == "__main__":
    main()
