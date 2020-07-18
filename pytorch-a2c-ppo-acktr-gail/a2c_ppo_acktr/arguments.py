import argparse
import json

import torch

def str_bool_dict(d):
    d = json.loads(d)
    for k in d.keys():
        if d[k].lower() == 'true':
            d[k] = True
        elif d[k].lower() == 'false':
            d[k] = False
        else:
            raise argparse.ArgumentTypeError(f"invalid bool value given {d[k]}. must be true or false.")
    return d

def str_uint_dict(d):
    d = json.loads(d)
    for k in d.keys():
        if int(d[k]) < 0:
            raise argparse.ArgumentTypeError(f"invalid uint value given {d[k]}. must be int >= 0.")
        d[k] = int(d[k])
    return d

def get_args():
    parser = argparse.ArgumentParser(description='RL')
    parser.add_argument(
        '--algo', default='ppo',
        type=str,
        help='algorithm to use: a2c | ppo | acktr')
    parser.add_argument(
        '--lr', type=float, default=7e-4, help='learning rate (default: 7e-4)')
    parser.add_argument(
        '--eps',
        type=float,
        default=1e-5,
        help='RMSprop optimizer epsilon (default: 1e-5)')
    parser.add_argument(
        '--alpha',
        type=float,
        default=0.99,
        help='RMSprop optimizer apha (default: 0.99)')
    parser.add_argument(
        '--gamma',
        type=float,
        default=0.99,
        help='discount factor for rewards (default: 0.99)')
    parser.add_argument(
        '--use-gae',
        action='store_true',
        default=True,
        help='use generalized advantage estimation')
    parser.add_argument(
        '--gae-lambda',
        type=float,
        default=0.95,
        help='gae lambda parameter (default: 0.95)')
    parser.add_argument(
        '--entropy-coef',
        type=float,
        default=0.001,
        help='entropy term coefficient (default: 0.01)')
    parser.add_argument(
        '--value-loss-coef',
        type=float,
        default=0.5,
        help='value loss coefficient (default: 0.5)')
    parser.add_argument(
        '--max-grad-norm',
        type=float,
        default=2048,
        help='max norm of gradients (default: 0.5)')
    parser.add_argument(
        '--seed', type=int, default=886739, help='random seed (default: 1)')
    parser.add_argument(
        '--cuda-deterministic',
        action='store_true',
        default=False,
        help="sets flags for determinism when using CUDA (potentially slow!)")
    parser.add_argument(
        '--num-processes',
        type=int,
        default=1,#48*3,
        help='how many training CPU processes to use (default: 16)')
    parser.add_argument(
        '--num-steps',
        type=int,
        default=8192,#512,
        help='number of forward steps in A2C (default: 5)')
    parser.add_argument(
        '--ppo-epoch',
        type=int,
        default=4,
        help='number of ppo epochs (default: 4)')
    parser.add_argument(
        '--num-mini-batch',
        type=int,
        default=16,#72,
        help='number of batches for ppo (default: 32)')
    parser.add_argument(
        '--clip-param',
        type=float,
        default=0.2,
        help='ppo clip parameter (default: 0.2)')
    parser.add_argument(
        '--log-interval',
        type=int,
        default=1e5,
        help='log interval, one log per n env steps (default: 1000)')
    parser.add_argument(
        '--write-interval',
        type=int,
        default=5e3,
        help='tensorboard write interval, one log per n env steps (default: 1000)')
    parser.add_argument(
        '--save-interval',
        type=int,
        default=1e6,
        help='save interval, one save per n env steps (default: 10000)')
    parser.add_argument(
        '--eval-interval',
        type=int,
        default=5e6,
        help='eval interval, one eval per n env steps (default: None)')
    parser.add_argument(
        '--no-cuda',
        action='store_true',
        default=False,
        help='disables CUDA training')
    parser.add_argument(
        '--use-proper-time-limits',
        action='store_true',
        default=False,
        help='compute returns taking into account time limits')
    parser.add_argument(
        '--recurrent-policy',
        action='store_true',
        default=False,
        help='use a recurrent policy')
    parser.add_argument(
        '--use-linear-lr-decay',
        action='store_true',
        default=False,
        help='use a linear schedule on the learning rate')
    parser.add_argument(
        '--load',
        default='runs/07-10_11-42-50_10m',
        help='start training a model from given checkpoint'
    )

    parser.add_argument(
        '--use-video',
        action='store_true',
        default=True,
        help='use video observation'
    )
    parser.add_argument(
        '--use-audio',
        action='store_true',
        default=True,
        help='use audio observation'
    )
    parser.add_argument(
        '--obs-keep-fskip',
        default='{"video": "false", "audio": "true"}',
        type=str,
        help='true if keeping skipped frames as part of observation'
    )
    parser.add_argument(
        '--obs-mbuf',
        default='{"video": 4, "audio": 4}',
        type=str,
        help='number of previous observations, plus the current one, to concatenate together. 1 gives only the current observation.'
    )
    parser.add_argument(
        '--obs-process',
        default='{"video": "grayscale", "audio": "mel_s"}',
        type=str,
        help='preprocessing method for each observation'
    )
    parser.add_argument(
        '--obs-module',
        default='{"video": "video-medium", "audio": "audio-medium"}',
        type=str,
        help='neural network architecture for processing each observation'
    )
    parser.add_argument(
        '--obs-viz-layer',
        default='{"video": 3, "audio": 3}',
        type=str,
        help='which cnn layer in neural network architecture to visualize using scorecam method'
    )
    parser.add_argument(
        '--vid-file-steps',
        default=0,
        type=int,
        help='how many frames of video to save to file during evaluation. 0 means no video saved.'
    )
    parser.add_argument(
        '--vid-tb-steps',
        default=0,
        type=int,
        help='how many frames of video to write to tensorboard during evaluation. 0 means no video written.'
    )
    parser.add_argument(
        '--rew-scale',
        default=0.005,
        type=float,
        help='reward scaling factor applied during training'
    )
    parser.add_argument(
        '--fskip-num',
        default=4,
        type=int,
        help='number of environment frames to skip between actions'
    )
    parser.add_argument(
        '--fskip-prob',
        default=0.25,
        type=float,
        help='probability of repeating previous action for 1 more frame, before continuing to skip fskip_num-1 frames with new action'
    )
    parser.add_argument(
        '--max-episode-steps',
        default=4500,
        type=int,
        help='max episode steps post frameskip until reset'
    )
    
    parser.add_argument(
        '--solo-train-steps',
        default=1e6,
        type=int,
        help='number of training steps for each ppo solo train'
    )
    parser.add_argument(
        '--solo-eval-steps',
        default=1e4,
        type=int,
        help='number of training steps for each ppo solo eval'
    )
    parser.add_argument(
        '--joint-train-steps',
        default=0,#3e7,
        type=int,
        help='number of training steps for joint ppo training'
    )
    parser.add_argument(
        '--joint-eval-steps',
        default=0,#1e4,
        type=int,
        help='number of training steps for each level in joint ppo training'
    )
    
    args = parser.parse_args()

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    args.device = 'cuda:0' if args.cuda else 'cpu'
    
    args.obs_keys = []
    if args.use_video:
        args.obs_keys.append('video')
    if args.use_audio:
        args.obs_keys.append('audio')
    assert args.obs_keys

    args.obs_keep_fskip = str_bool_dict(args.obs_keep_fskip)
    args.obs_mbuf = str_uint_dict(args.obs_mbuf)
    args.obs_process = json.loads(args.obs_process)
    args.obs_module = json.loads(args.obs_module)
    args.obs_viz_layer = str_uint_dict(args.obs_viz_layer)

    assert args.algo in ['a2c', 'ppo', 'acktr']
    if args.recurrent_policy:
        assert args.algo in ['a2c', 'ppo'], \
            'Recurrent policy is not implemented for ACKTR'

    print(f"args: {args}")
    return args
