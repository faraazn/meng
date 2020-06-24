import torch
import numpy as np
import time

from visualize.scorecam import CamExtractor, ScoreCam
from visualize.misc_functions import apply_colormap_on_image

from preprocess import ProcessMelSpectrogram

from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw
from PIL import ImageEnhance

ACTIONS = ('L ', 'R ', 'LD', 'RD', 'D ', 'DB', 'B ')


def gen_eval_vid_frame_small(actor_critic, obs):
    actor_critic.eval()
    vid_process = actor_critic.base.video_process(obs['video'])[0]
    vid_process = np.uint8(vid_process.detach().cpu().numpy()*255)
    return vid_process


def gen_eval_vid_frame(actor_critic, env_state, x, max_x, pct, rew, t, value, action, logits, obs, viz_layers):
    actor_critic.eval()
    processed_obs = obs.copy()
    for obs_name in sorted(obs.keys()):
        if obs_name == 'video':
            processed_obs[obs_name] = actor_critic.base.video_process(obs[obs_name])
        elif obs_name == 'audio':
            processed_obs[obs_name] = actor_critic.base.audio_process(obs[obs_name])
        else:
            raise NotImplementedError
    score_cam = ScoreCam(actor_critic, viz_layers)
    cams = score_cam.generate_cam(processed_obs, action)

    frame_x = 1080#320*3  # this needs to be known max length along x axis to align heatmap etc
    frame_y = 100  # this increases with each obs since we append along y axis

    # generate frame info
    cat_im = Image.new('RGB', (frame_x, frame_y))
    logits = logits.detach().cpu().numpy()[0] if logits is not None else []
    logits_txt = '['+', '.join([f"{logit:.3f}" for logit in logits])[:-1]+']'
    draw = ImageDraw.Draw(cat_im)
    draw.text(
        (0, 4),
        f"""    {env_state} | t: {t}
            x: {x:07.2f} | max_x: {max_x:07.2f} | pct: {pct:05.2f}% | rew: {rew:07.2f} | val: {value:.4f}
            action: {ACTIONS[action]} | logits: {logits_txt}""",
        (255,255,255))

    for obs_name in sorted(obs.keys()):
        obs_im_array = np.uint8(processed_obs[obs_name][0].detach().cpu().numpy()*255).transpose((1,2,0))
        if obs_im_array.shape[2] == 3:
            obs_im = Image.fromarray(obs_im_array, 'RGB')
        elif obs_im_array.shape[2] == 1:
            obs_im = Image.fromarray(obs_im_array[:,:,0], 'P')
        elif obs_im_array.shape[2] == 4:
            obs_im = Image.fromarray(obs_im_array, 'RGBA')
        else:
            raise ValueError
        #aud_im = Image.fromarray(np.flip(obs_audio_mels[0][0], axis=0), 'L')
        
        hm, hm_on_im = apply_colormap_on_image(obs_im, cams[obs_name], 'hsv')
        hm = ImageEnhance.Brightness(hm).enhance(0.75)  # heatmap is usually too bright

        if obs_name == 'audio':
            # for some reason spectrogram ends up flipped
            obs_im = obs_im.transpose(Image.FLIP_TOP_BOTTOM)
            hm = hm.transpose(Image.FLIP_TOP_BOTTOM)
            hm_on_im = hm_on_im.transpose(Image.FLIP_TOP_BOTTOM)

        # add to previous image
        new_frame_y = frame_y + obs_im.size[1]
        new_cat_im = Image.new('RGB', (frame_x, new_frame_y))
        new_cat_im.paste(cat_im, (0,0))
        new_cat_im.paste(obs_im, (0, frame_y))  
        new_cat_im.paste(hm_on_im, (frame_x//3,frame_y))
        new_cat_im.paste(hm, (frame_x//3*2,frame_y))
        frame_y = new_frame_y
        cat_im = new_cat_im

    return np.uint8(cat_im)
