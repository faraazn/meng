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

def evaluate(env_states, actor_critic, eval_t, step, writer, vid_save_dir, vid_tb_steps, vid_file_steps, viz_layer, args):
    assert eval_t > 0
    assert vid_tb_steps == 0 or writer is not None
    print(f"  [eval] Running {eval_t} evaluation steps")

    actor_critic.eval()

    eval_dict = {'x': {}, '%': {}, 'r': {}}
    for env_state in env_states:
        start = time.time()
        print(f"    {env_state[0]} {env_state[1]}")
        eval_dict['x'][env_state] = []
        eval_dict['%'][env_state] = []
        eval_dict['r'][env_state] = []
        
        # use args.device instead of 'cpu' because only generating data 1 process and 1 step at a time
        env = make_vec_envs([env_state], args.seed + 1000, 1, None, args.device, 'eval', args)
        obs = env.reset()

        if vid_file_steps > 0:
            vid_filepath = os.path.join(vid_save_dir, f"{env_state[1]}-{step}.webm")
            if args.use_audio:
                sr = 44100  #env.em.get_audio_rate()  # can't actually do it this way
                aud_filepath = "/tmp/temp.wav"
                aud_record = wave.open(aud_filepath,'w')
                aud_record.setnchannels(1)  # mono
                aud_record.setsampwidth(2)
                aud_record.setframerate(sr)
                # make vid file temporary if muxing with audio later
                vid_filepath = "/tmp/temp.webm"
            
            vid_frame = gen_eval_vid_frame(
                actor_critic, env_state[1], 0, 0, 0, 0, 0, 0, 0, None, obs, viz_layer)
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
                
            if (vid_file_steps > 0 or vid_tb_steps > 0) and t < max(vid_file_steps, vid_tb_steps):
                x = info[0]['x'] if info else 0
                max_x = info[0]['max_x'] if info else 0
                pct = info[0]['max_x']/info[0]['lvl_max_x']*100 if info else 0
                rew = info[0]['sum_r'] if info else 0
                v = action.item()
                a = action.item()
                vid_frame = gen_eval_vid_frame(
                    actor_critic, env_state[1], x, max_x, pct, rew, t, v, a, logits, obs, viz_layer)
            
            if vid_file_steps > 0 and t < min(vid_file_steps, eval_t - 1):
                vid_record.write(vid_frame[:,:,::-1])  # format 'BGR' for cv2 writing
                if args.use_audio:
                    # obs['audio'] shape [1, 735] and dtype float32
                    # or shape [1, memory (or frameskip), 735] or [1, m, f, 735]
                    # TODO: need to differentiate between mem and frameskip, mem needs to be removed
                    aud_frame = np.int16(obs['audio'].detach().cpu().numpy()[0][-1])
                    aud_record.writeframesraw(aud_frame)
            elif vid_file_steps > 0 and t == min(vid_file_steps, eval_t - 1):
                vid_record.release()
                if args.use_audio:
                    aud_record.close()
                    # combine the audio and video into a new file
                    final_vid_filepath = os.path.join(vid_save_dir, f"{env_state[1]}-{step}.webm")
                    process = subprocess.check_call(
                        ['ffmpeg', '-hide_banner', '-loglevel', 'warning', '-y', '-i', vid_filepath,
                        '-i', aud_filepath, '-c:v', 'copy', '-c:a', 'libopus', final_vid_filepath])
                    vid_filepath = final_vid_filepath
                print(f"      wrote video to {vid_filepath}")
            
            if vid_tb_steps > 0 and t < min(vid_tb_steps, eval_t - 1):
                vid_frames.append(np.expand_dims(vid_frame.transpose((2,0,1)), axis=0))
            elif vid_tb_steps > 0 and t == min(vid_tb_steps, eval_t - 1):
                vid_frames = np.expand_dims(np.concatenate(vid_frames), axis=0)
                writer.add_video(env_state[1], vid_frames, global_step=step, fps=60)
                del vid_frames
                print(f"      wrote video to tensorboard")

            # Obser reward and next obs
            obs, reward, done, info = env.step(action)
            ep_reward += reward
            last_info = info
            
            masks.fill_(0.0 if done else 1.0)

            if done[0]:
                r = ep_reward[0].detach().cpu().item()
                eval_dict['r'][env_state].append(r)
                ep_reward = 0

                eval_dict['x'][env_state].append(info[0]['max_x'])
                eval_dict['%'][env_state].append(info[0]['max_x']/info[0]['lvl_max_x'] * 100)
                
                if writer is not None:
                    writer.add_scalar(f'eval_episode_x/{env_state[1]}', eval_dict['x'][env_state][-1], step+t)
                    writer.add_scalar(f'eval_episode_%/{env_state[1]}', eval_dict['%'][env_state][-1], step+t)
                    writer.add_scalar(f'eval_episode_r/{env_state[1]}', eval_dict['r'][env_state][-1], step+t)
                obs = env.reset()

            t += 1

        env.close()
        del env
        del obs

        # if first episode did not run to completion, append its current reward
        if not eval_dict['r'][env_state]:
            assert not eval_dict['x'][env_state] and not eval_dict['%'][env_state]
            r = ep_reward[0].detach().cpu().item()
            eval_dict['r'][env_state].append(r)
            eval_dict['x'][env_state].append(last_info[0]['max_x'])
            eval_dict['%'][env_state].append(last_info[0]['max_x']/info[0]['lvl_max_x'] * 100)
            
            if writer is not None:
                writer.add_scalar(f'eval_episode_x/{env_state[1]}', eval_dict['x'][env_state][-1], step+t)
                writer.add_scalar(f'eval_episode_%/{env_state[1]}', eval_dict['%'][env_state][-1], step+t)
                writer.add_scalar(f'eval_episode_r/{env_state[1]}', eval_dict['r'][env_state][-1], step+t)
        
        if writer is not None:
            writer.add_scalar(f'eval_episode_x_avg/{env_state[1]}', np.mean(eval_dict['x'][env_state]), step)
            writer.add_scalar(f'eval_episode_%_avg/{env_state[1]}', np.mean(eval_dict['%'][env_state]), step)
            writer.add_scalar(f'eval_episode_r_avg/{env_state[1]}', np.mean(eval_dict['r'][env_state]), step)
            writer.add_scalar(f'eval_episode_x_std/{env_state[1]}', np.std(eval_dict['x'][env_state]), step)
            writer.add_scalar(f'eval_episode_%_std/{env_state[1]}', np.std(eval_dict['%'][env_state]), step)
            writer.add_scalar(f'eval_episode_r_std/{env_state[1]}', np.std(eval_dict['r'][env_state]), step)

        print(f"      score {np.mean(eval_dict['r'][env_state]):.3f}, stddev {np.std(eval_dict['r'][env_state]):.3f}, {time.time()-start:.1f}s")

    # compute evaluation metric
    score = np.mean([np.mean(eval_dict['r'][env_state]) for env_state in env_states])
    return score, eval_dict
