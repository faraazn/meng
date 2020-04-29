import argparse
import os
# workaround to unpickle olf model files
import sys

import numpy as np
import torch

from a2c_ppo_acktr.envs import VecPyTorch, make_vec_envs
from a2c_ppo_acktr.utils import get_render_func, get_vec_normalize

from renderer import Renderer

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
    
#envs = make_vec_envs(args.env_name, args.seed, args.num_processes,
#                         args.gamma, args.log_dir, device, False)

# Get a render function
render_func = None #get_render_func(env)
renderer = Renderer()

# We need to use the same statistics for normalization as used in training
actor_critic, ob_rms = \
            torch.load(os.path.join(args.load_dir, args.env_name + ".pt"))

vec_norm = get_vec_normalize(env)
if vec_norm is not None:
    vec_norm.eval()
    vec_norm.ob_rms = ob_rms

recurrent_hidden_states = torch.zeros(1,
                                      actor_critic.recurrent_hidden_state_size)
masks = torch.zeros(1, 1)

obs = env.reset()

while True:
    with torch.no_grad():
        #print(f"obs {obs.shape}")
        #print(f"ac {actor_critic.obs_shape}")
        value, action, _, recurrent_hidden_states = actor_critic.act(
            obs, recurrent_hidden_states, masks, deterministic=args.det)

    # Obser reward and next obs
    obs, reward, done, _ = env.step(action)
    #print(f"obs {obs.shape}")
    aud_frame = env.envs[0].em.get_audio()
    #print(f"aud {aud_frame.shape}")
    masks.fill_(0.0 if done else 1.0)

    vid_frame = obs[0].detach().numpy().astype(np.uint8)
    vid_frame = vid_frame.transpose((1, 2, 0))
    renderer.render(vid_frame, aud_frame)

    if done:
        break

renderer.close()
