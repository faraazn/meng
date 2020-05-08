import numpy as np
import time
import torch
import cv2
import os

from a2c_ppo_acktr import utils
from a2c_ppo_acktr.envs import make_vec_envs


def evaluate(env_states, seed, device, actor_critic, eval_t, step, writer=None, vid_save_dir=None):
    # don't write videos that are too long
    MAX_WRITER = 450  # 30 sec with frame skip 4
    MAX_VID_SAVE = 4500  # 5 min with frame skip 4

    eval_ep_rewards = {}
    for env_state in env_states:
        start = time.time()
        eval_ep_rewards[env_state] = []
        
        if vid_save_dir:
            vid_width = 320
            vid_height = 224
            fps = 60  # record at 4x speed with frame skip 4
            vid_filepath = os.path.join(vid_save_dir, f"{env_state}-{step}.mp4")
            vid_record = cv2.VideoWriter(
                vid_filepath, cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), fps, (vid_width, vid_height))

        # TODO: specify evaluation mode during make envs so there's no reward scaler
        env = make_vec_envs(
            [env_state],
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
        ep_reward = 0
        while t < eval_t:
            with torch.no_grad():
                value, action, _, recurrent_hidden_states = actor_critic.act(
                    obs, recurrent_hidden_states, masks, deterministic=True)

            # Obser reward and next obs
            obs, reward, done, info = env.step(action)
            ep_reward += reward

            #aud_frame = env.envs[0].em.get_audio()[:,0]
            #aud_frames.append(aud_frame)
            masks.fill_(0.0 if done else 1.0)

            if writer and t < MAX_WRITER:
                vid_frame = obs[0].detach().cpu().numpy().astype(np.uint8)
                vid_frames.append(np.expand_dims(vid_frame, axis=0))

            if vid_save_dir and t < MAX_VID_SAVE:
                vid_frame = obs[0].detach().cpu().numpy().astype(np.uint8)
                vid_record.write(vid_frame)
                
            if done:
                r = ep_reward[0].detach().cpu().item()
                eval_ep_rewards[env_state].append(r)
                ep_reward = 0
                max_x = max(max_x, info[0]['max_x'])
                obs = env.reset()

            t += 1

        env.close()
        del env

        # if first episode did not run to completion, append its current reward
        if not eval_ep_rewards[env_state]:
            eval_ep_rewards[env_state].append(ep_reward)

        print(f"  generated eval data for {env_state}, max_x {max_x}: {time.time()-start}s")

        if writer:
            start = time.time()
            vid_frames = np.expand_dims(np.concatenate(vid_frames), axis=0)
            #aud_frames = np.expand_dims(np.concatenate(aud_frames) / 2**15, axis=0)
            writer.add_video(env_state, vid_frames, global_step=step, fps=60)
            #print(f"  wrote video {time.time()-start}s")
            #start = time.time()
            #writer.add_audio('eval_ep_audio', aud_frames)
            #print(f"  wrote audio {time.time()-start}s")
            #start = time.time()
            writer.add_scalar(f'eval_max_x/{env_state}', max_x, step)

            print(f"    wrote video to tensorboard: {time.time()-start}s")

        if vid_save_dir:
            start = time.time()
            vid_record.release()
            print(f"    wrote video to {vid_save_dir}: {time.time()-start}s")

    # compute evaluation metric
    score = np.mean([np.mean(eval_ep_rewards[env_state]) for env_state in env_states])
    return score, eval_ep_rewards
