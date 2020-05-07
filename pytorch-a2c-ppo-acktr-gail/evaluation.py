import numpy as np
import time
import torch
import moviepy

from a2c_ppo_acktr import utils
from a2c_ppo_acktr.envs import make_vec_envs

def write_eval_episode(writer, step, env_states, seed, device, actor_critic):
    start = time.time()
    env = make_vec_envs(
        env_states,
        seed + 1000,
        1,
        None,
        None,
        device=device,
        allow_early_resets=False)

    recurrent_hidden_states = torch.zeros(1, actor_critic.recurrent_hidden_state_size)
    masks = torch.zeros(1, 1)

    vid_frames = []
    aud_frames = []
    max_x = 0
    obs = env.reset()
    t = 0
    while t < 450:  # max 30 sec
        with torch.no_grad():
            value, action, _, recurrent_hidden_states = actor_critic.act(
                obs, recurrent_hidden_states, masks, deterministic=True)

        # Obser reward and next obs
        obs, reward, done, info = env.step(action)
        #aud_frame = env.envs[0].em.get_audio()[:,0]
        #aud_frames.append(aud_frame)
        masks.fill_(0.0 if done else 1.0)

        vid_frame = obs[0].detach().cpu().numpy().astype(np.uint8)
        vid_frames.append(np.expand_dims(vid_frame, axis=0))

        if done:
            max_x = max(max_x, info[0]['max_x'])
            break

        t += 1

    vid_frames = np.expand_dims(np.concatenate(vid_frames), axis=0)
    #aud_frames = np.expand_dims(np.concatenate(aud_frames) / 2**15, axis=0)
    print(f"  generated data {time.time()-start}s")
    start = time.time()

    writer.add_video('eval_ep_video', vid_frames, global_step=step, fps=60)
    #print(f"  wrote video {time.time()-start}s")
    #start = time.time()
    #writer.add_audio('eval_ep_audio', aud_frames)
    #print(f"  wrote audio {time.time()-start}s")
    #start = time.time()
    writer.add_scalar('eval_ep_t', t+1, step)
    writer.add_scalar('eval_ep_x', max_x, step)

    print(f"  wrote eval video at env_step {step}: t={t+1}, x={max_x}, {time.time()-start}s")


def evaluate(actor_critic, ob_rms, env_name, seed, num_processes, eval_log_dir,
             device):
    eval_envs = make_vec_envs(env_name, seed + num_processes, num_processes,
                              None, eval_log_dir, device, True)

    vec_norm = utils.get_vec_normalize(eval_envs)
    if vec_norm is not None:
        vec_norm.eval()
        vec_norm.ob_rms = ob_rms

    eval_episode_rewards = []

    obs = eval_envs.reset()
    eval_recurrent_hidden_states = torch.zeros(
        num_processes, actor_critic.recurrent_hidden_state_size, device=device)
    eval_masks = torch.zeros(num_processes, 1, device=device)

    while len(eval_episode_rewards) < 10:
        with torch.no_grad():
            _, action, _, eval_recurrent_hidden_states = actor_critic.act(
                obs,
                eval_recurrent_hidden_states,
                eval_masks,
                deterministic=True)

        # Obser reward and next obs
        obs, _, done, infos = eval_envs.step(action)

        eval_masks = torch.tensor(
            [[0.0] if done_ else [1.0] for done_ in done],
            dtype=torch.float32,
            device=device)

        for info in infos:
            if 'episode' in info.keys():
                eval_episode_rewards.append(info['episode']['r'])

    eval_envs.close()

    print(" Evaluation using {} episodes: mean reward {:.5f}\n".format(
        len(eval_episode_rewards), np.mean(eval_episode_rewards)))
