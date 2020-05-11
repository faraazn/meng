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

from torch.utils.tensorboard import SummaryWriter

ALL_STATES = (
    'GreenHillZone.Act1',  'GreenHillZone.Act2',  'GreenHillZone.Act3',
    'MarbleZone.Act1',     'MarbleZone.Act2',     'MarbleZone.Act3',    
    'SpringYardZone.Act1', 'SpringYardZone.Act2', 'SpringYardZone.Act3', 
    'LabyrinthZone.Act1',  'LabyrinthZone.Act2',  'LabyrinthZone.Act3',  
    'StarLightZone.Act1',  'StarLightZone.Act2',  'StarLightZone.Act3',
    'ScrapBrainZone.Act1', 'ScrapBrainZone.Act2', #'ScrapBrainZone.Act3'
)

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


def get_newest_ckpt_path(load_dir):
    try:
        _, _, ckpt_names = next(os.walk(load_dir))
    except StopIteration:
        return None

    newest_ckpt = None
    for ckpt_name in ckpt_names:
      ckpt = torch.load(os.path.join(load_dir, ckpt_name))
      if newest_ckpt is None or ckpt[2] > newest_ckpt[2]:  # index 2 is ckpt env_step
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
    device = torch.device("cuda:0" if args.cuda else "cpu")
        
    # format MM-DD_hh-mm-ss 
    run_name = str(datetime.now())[5:].replace(' ', '_').replace(':', '-').split('.')[0]

    solo_env_steps = 5e6
    eval_env_steps = 1e5
    writer = SummaryWriter(log_dir=f"runs/{run_name}/ppo/")
    for env_state in ALL_STATES:
        print(f"[train] Starting {env_state} ppo training")
        init_model = get_newest_ckpt_path(os.path.join(f"{args.load}", f"ppo/{env_state}/ckpts/"))
        model, eval_score, _ = train(
            [env_state], f"runs/{run_name}/ppo/{env_state}", args, solo_env_steps, eval_env_steps, device, writer, env_state, init_model)
        print(f"[train] {env_state} ppo score {eval_score}\n")
    writer.close()

    return
    joint_env_steps = 3e7
    writer = SummaryWriter(log_dir=f"runs/{run_name}/ppo-joint/")
    print(f"[train] Starting ppo-joint training")
    init_model = get_newest_ckpt_path(os.path.join(f"{args.load}", "ppo-joint/ckpts/"))
    joint_model, eval_score, _ = train(
        TRAIN_STATES, f"runs/{run_name}/ppo-joint", args, joint_env_steps, eval_env_steps, device, writer, 'joint', init_model)
    print(f"[train] ppo-joint score {eval_score}\n")
    writer.close()

    writer = SummaryWriter(log_dir=f"runs/{run_name}/ppo-joint-train/")
    for env_state in TRAIN_STATES:
        print(f"[train] Starting {env_state} ppo-joint-train training")
        init_model = get_newest_ckpt_path(os.path.join(f"{args.load}", f"ppo-joint-train/{env_state}/ckpts/"))
        if init_model is None:
            init_model = list(joint_model)
            init_model[1] = 0  # reset step to 0
        model, eval_score, _ = train(
            [env_state], f"runs/{run_name}/ppo-joint-train/{env_state}", args, solo_env_steps, eval_env_steps, device, writer, env_state, init_model)
        print(f"[train] {env_state} ppo-joint-train score {eval_score}\n")
    writer.close()

    writer = SummaryWriter(log_dir=f"runs/{run_name}/ppo-joint-test/")
    for env_state in TEST_STATES:
        print(f"[train] Starting {env_state} ppo-joint-test training")
        init_model = get_newest_ckpt_path(os.path.join(f"{args.load}", f"ppo-joint-test/{env_state}/ckpts/"))
        if init_model is None:
            init_model = list(joint_model)
            init_model[1] = 0  # reset step to 0
        model, eval_score, _ = train(
            [env_state], f"runs/{run_name}/ppo-joint-test/{env_state}", args, solo_env_steps, eval_env_steps, device, writer, env_state, init_model)
        print(f"[train] {env_state} ppo-joint-test score {eval_score}\n")
    writer.close()

if __name__ == "__main__":
    main()
