import numpy as np
import time
import torch
import cv2
import wave
import os
import subprocess

from a2c_ppo_acktr import utils
from a2c_ppo_acktr.envs import make_vec_envs
from visualize.visualize import gen_eval_vid_frame, gen_eval_vid_frame_small

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
        
        env = make_vec_envs(
            [env_state],
            seed + 1000,
            1,
            None,
            device=device,
            allow_early_resets=False,
            mode='eval')
        obs = env.reset()
        
        if vid_save_dir:
            vid_filepath = os.path.join(vid_save_dir, f"{env_state}-{step}.webm")
            if 'audio' in obs.keys():
                sr = 44100 #int(self.env.em.get_audio_rate())
                aud_filepath = "/tmp/temp.wav"
                aud_record = wave.open(aud_filepath,'w')
                aud_record.setnchannels(1)  # mono
                aud_record.setsampwidth(2)
                aud_record.setframerate(sr)
                # make vid file temporary if muxing with audio later
                vid_filepath = "/tmp/temp.webm"
            
            vid_frame = gen_eval_vid_frame(
                actor_critic, env_state, 0, 0, 0, 0, 0, 0, None, obs, {'video': 5, 'audio': 5})
            vid_height = vid_frame.shape[0]
            vid_width = vid_frame.shape[1]
            fps = 60/4  # record at 1x speed with frame skip 4
            vid_record = cv2.VideoWriter(
                vid_filepath, cv2.VideoWriter_fourcc(*'vp90'), fps, (vid_width, vid_height))
                
        recurrent_hidden_states = torch.zeros(1, actor_critic.recurrent_hidden_state_size)
        masks = torch.zeros(1, 1)

        vid_frames = []
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
                    tgt_layers = {'video': 5, 'audio': 5}
                    vid_frame = gen_eval_vid_frame(
                        actor_critic, env_state, x, max_x, pct, rew, t, a, logits, obs, tgt_layers)
                    assert vid_frame.shape[0] == vid_height and vid_frame.shape[1] == vid_width
                    vid_frame = vid_frame[:,:,::-1]  # format 'BGR' for cv2 writing
                    vid_record.write(vid_frame)

                    if 'audio' in obs.keys():
                        # obs['audio'] shape [1, 735] and dtype float32??
                        aud_frame = np.int16(obs['audio'].detach().cpu().numpy()[0]*2**15)
                        aud_record.writeframesraw(aud_frame)

            # Obser reward and next obs
            obs, reward, done, info = env.step(action)
            ep_reward += reward
            last_info = info
            
            masks.fill_(0.0 if done else 1.0)

            if writer and t < MAX_WRITER:
                vid_frame = gen_eval_vid_frame_small(actor_critic, obs)
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

        print(f"    generated eval data for {env_state}: {time.time()-start:.1f}s")

        if writer:
            start = time.time()
            vid_frames = np.expand_dims(np.concatenate(vid_frames), axis=0)
            writer.add_video(env_state, vid_frames, global_step=step, fps=60)  # 4x speed w frameskip 4
            writer.add_scalar(f'eval_episode_x/{env_state}', np.mean(eval_dict['x'][env_state]), step)
            writer.add_scalar(f'eval_episode_%/{env_state}', np.mean(eval_dict['%'][env_state]), step)
            writer.add_scalar(f'eval_episode_r/{env_state}', np.mean(eval_dict['r'][env_state]), step)
            print(f"    wrote video to tensorboard")

        if vid_save_dir:
            start = time.time()
            vid_record.release()

            if 'audio' in obs.keys():
                aud_record.close()
                # combine the audio and video into a new file
                final_vid_filepath = os.path.join(vid_save_dir, f"{env_state}-{step}.webm")
                process = subprocess.check_call(
                    ['ffmpeg', '-hide_banner', '-loglevel', 'warning', '-y', '-i', vid_filepath,
                     '-i', aud_filepath, '-c:v', 'copy', '-c:a', 'libopus', final_vid_filepath])
                vid_filepath = final_vid_filepath
            print(f"    wrote video to {vid_filepath}")

    # compute evaluation metric
    score = np.mean([np.mean(eval_dict['r'][env_state]) for env_state in env_states])
    return score, eval_dict
