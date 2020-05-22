import numpy as np
import time
import torch
import cv2
import os

from a2c_ppo_acktr import utils
from a2c_ppo_acktr.envs import make_vec_envs
from visualize.visualize import gen_eval_vid_frame

def evaluate(env_states, seed, device, actor_critic, eval_t, step, writer=None, vid_save_dir=None):
    # don't write videos that are too long
    MAX_WRITER = 450  # 30 sec with frame skip 4
    MAX_VID_SAVE = 1800  # 2 min with frame skip 4
    assert eval_t > 0

    actor_critic.eval()

    eval_dict = {'x': {}, '%': {}, 'r': {}}
    for env_state in env_states:
        start = time.time()
        eval_dict['x'][env_state] = []
        eval_dict['%'][env_state] = []
        eval_dict['r'][env_state] = []
        
        if vid_save_dir:
            vid_width = 320*3
            vid_height = 224+50
            fps = 15  # record at 1x speed with frame skip 4
            vid_filepath = os.path.join(vid_save_dir, f"{env_state}-{step}.webm")
            vid_record = cv2.VideoWriter(
                vid_filepath, cv2.VideoWriter_fourcc('v', 'p', '9', '0'), fps, (vid_width, vid_height))

        # TODO: specify evaluation mode during make envs so there's no reward scaler
        env = make_vec_envs(
            [env_state],
            seed + 1000,
            1,
            None,
            device=device,
            allow_early_resets=False,
            mode='eval')

        recurrent_hidden_states = torch.zeros(1, actor_critic.recurrent_hidden_state_size)
        masks = torch.zeros(1, 1)

        vid_frames = []
        aud_frames = []
        obs = env.reset()
        t = 0
        ep_reward = 0
        last_info = None
        screen_x_end = None
        info = None
        while t < eval_t:
            with torch.no_grad():
                value, action, log_probs, recurrent_hidden_states, logits = actor_critic.act(
                    obs, recurrent_hidden_states, masks, deterministic=True)
                
                if vid_save_dir and t < MAX_VID_SAVE:
                    x = info[0]['x'] if info else 0
                    max_x = info[0]['max_x'] if info else 0
                    pct = info[0]['max_x']/info[0]['lvl_max_x']*100 if info else 0
                    rew = info[0]['sum_r'] if info else 0
                    a = action.item()
                    tgt_layers = {'video': 5}
                    vid_frame = gen_eval_vid_frame(
                        actor_critic, env_state, x, max_x, pct, rew, t, a, logits, obs, tgt_layers)
                    vid_frame = vid_frame[:,:,::-1]  # format for cv2 writing
                    vid_record.write(vid_frame)


            # Obser reward and next obs
            obs, reward, done, info = env.step(action)
            ep_reward += reward
            last_info = info
            
            masks.fill_(0.0 if done else 1.0)

            if writer and t < MAX_WRITER:
                vid_frame = obs['video'][0].detach().cpu().numpy().astype(np.uint8)
                vid_frames.append(np.expand_dims(vid_frame, axis=0))
                
            if done[0]:
                r = ep_reward[0].detach().cpu().item()
                eval_dict['r'][env_state].append(r)
                ep_reward = 0

                eval_dict['x'][env_state].append(info[0]['max_x'])
                eval_dict['%'][env_state].append(info[0]['max_x']/info[0]['lvl_max_x'] * 100)
                obs = env.reset()

            t += 1

        env.close()
        del env

        # if first episode did not run to completion, append its current reward
        if not eval_dict['r'][env_state]:
            assert not eval_dict['x'][env_state] and not eval_dict['%'][env_state]
            r = ep_reward[0].detach().cpu().item()
            eval_dict['r'][env_state].append(r)
            eval_dict['x'][env_state].append(last_info[0]['max_x'])
            eval_dict['%'][env_state].append(last_info[0]['max_x']/info[0]['lvl_max_x'] * 100)

        print(f"    generated eval data for {env_state}: {time.time()-start}s")

        if writer:
            start = time.time()
            vid_frames = np.expand_dims(np.concatenate(vid_frames), axis=0)
            #aud_frames = np.expand_dims(np.concatenate(aud_frames) / 2**15, axis=0)
            writer.add_video(env_state, vid_frames, global_step=step, fps=60)  # 4x speed w frameskip 4
            #print(f"  wrote video {time.time()-start}s")
            #start = time.time()
            #writer.add_audio('eval_ep_audio', aud_frames)
            #print(f"  wrote audio {time.time()-start}s")
            #start = time.time()
            writer.add_scalar(f'eval_episode_x/{env_state}', np.mean(eval_dict['x'][env_state]), step)
            writer.add_scalar(f'eval_episode_%/{env_state}', np.mean(eval_dict['%'][env_state]), step)
            writer.add_scalar(f'eval_episode_r/{env_state}', np.mean(eval_dict['r'][env_state]), step)
            print(f"    wrote video to tensorboard")

        if vid_save_dir:
            start = time.time()
            vid_record.release()
            print(f"    wrote video to {vid_save_dir}")

    # compute evaluation metric
    score = np.mean([np.mean(eval_dict['r'][env_state]) for env_state in env_states])
    return score, eval_dict
