import argparse
import os
# workaround to unpickle olf model files
import sys

import numpy as np
import torch

from a2c_ppo_acktr.envs import VecPyTorch, make_vec_envs
from a2c_ppo_acktr.utils import get_render_func, get_vec_normalize

#from renderer import Renderer

sys.path.append('a2c_ppo_acktr')

parser = argparse.ArgumentParser(description='RL')
parser.add_argument(
    '--seed', type=int, default=1, help='random seed (default: 1)')
parser.add_argument(
    '--log-interval',
    type=int,
    default=10,
    help='log interval, one log per n updates (default: 10)')
parser.add_argument(
    '--env-name',
    default='sonic-2',#'SeaquestNoFrameskip-v4',
    help='environment to train on (default: PongNoFrameskip-v4)')
parser.add_argument(
    '--load-dir',
    default='./trained_models/',
    help='directory to save agent logs (default: ./trained_models/)')
parser.add_argument(
    '--non-det',
    action='store_true',
    default=False,
    help='whether to use a non-deterministic policy')
parser.add_argument(
    '--load',
    default='',
    help='name of model checkpoint to load')
args = parser.parse_args()

args.det = not args.non_det

env = make_vec_envs(
    args.env_name,
    args.seed + 1000,
    1,
    None,
    None,
    device='cpu',
    allow_early_resets=False)
    
# Get a render function
#renderer = Renderer()

# We need to use the same statistics for normalization as used in training
actor_critic, episode_num, env_step = torch.load(args.load, map_location=torch.device('cpu'))
print(f"loaded {args.load} at episode {episode_num}, env step {env_step}")

recurrent_hidden_states = torch.zeros(1, actor_critic.recurrent_hidden_state_size)
masks = torch.zeros(1, 1)

obs = env.reset()

while True:
    with torch.no_grad():
        value, action, _, recurrent_hidden_states = actor_critic.act(
            obs, recurrent_hidden_states, masks, deterministic=args.det)

    # Obser reward and next obs
    obs, reward, done, info = env.step(action)
    aud_frame = env.envs[0].em.get_audio()
    masks.fill_(0.0 if done else 1.0)

    vid_frame = obs[0].detach().numpy().astype(np.uint8)
    vid_frame = vid_frame.transpose((1, 2, 0))
    #renderer.render(vid_frame, aud_frame)

    if done:
        print(f"max_x {info[0]['max_x']}, stage len {info[0]['screen_x_end']}")
        break

#renderer.close()
