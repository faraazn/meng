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


def gen_eval_vid_frame(actor_critic, env_state, x, max_x, pct, rew, t, action, logits, obs, tgt_layers):
    actor_critic.eval()

    frame_y = 0

    score_cam = ScoreCam(actor_critic, tgt_layers)
    cams = score_cam.generate_cam(obs.copy(), action)
    cam = cams['video']

    obs_im_array = np.uint8(obs['video'][0].detach().cpu().numpy()).transpose((1,2,0))  # shape [224, 320, 3]
    obs_im = Image.fromarray(obs_im_array, 'RGB')  # size [320, 224]

    hm, hm_on_im = apply_colormap_on_image(obs_im, cam, 'hsv')
    hm = ImageEnhance.Brightness(hm).enhance(0.75)  # heatmap is usually too bright

    cat_im = Image.new('RGB', (320*3, 224))
    cat_im.paste(obs_im, (0,0))
    cat_im.paste(hm_on_im, (320,0))
    cat_im.paste(hm, (640,0))

    frame_y += 224

    if 'audio' in obs.keys():
        obs_audio_mels = actor_critic.base.audio_process(obs['audio'])
        obs_audio_mels = np.uint8(obs_audio_mels.detach().cpu().numpy()*255)  # scale to full uint8 image range
        # perform vertical flip before passing to image
        aud_im = Image.fromarray(np.flip(obs_audio_mels[0][0], axis=0), 'L')

        cat_aud_im = Image.new('RGB', (320*3, 256))
        cat_aud_im.paste(aud_im, (0,0))

        hm, hm_on_im = apply_colormap_on_image(aud_im, cams['audio'], 'hsv')
        hm = ImageEnhance.Brightness(hm).enhance(0.75)
        cat_aud_im.paste(hm_on_im, (320,0))
        cat_aud_im.paste(hm, (640,0))

        new_cat_im = Image.new('RGB', (320*3, frame_y+256))
        new_cat_im.paste(cat_im, (0,0))
        new_cat_im.paste(cat_aud_im, (0,frame_y))
        cat_im = new_cat_im
        frame_y += 256

    new_cat_im = Image.new('RGB', (320*3, frame_y+100))
    new_cat_im.paste(cat_im, (0,0))
    cat_im = new_cat_im

    logits = logits.detach().cpu().numpy()[0]
    logits_txt = '['+', '.join([f"{logit:.3f}" for logit in logits])[:-1]+']'
    draw = ImageDraw.Draw(cat_im)
    draw.text(
        (0, frame_y+4),
        f"""    {env_state} | t: {t}
            x: {x:07.2f} | max_x: {max_x:07.2f} | pct: {pct:05.2f}% | rew: {rew:07.2f}
            action: {ACTIONS[action]} | logits: {logits_txt}""",
        (255,255,255))
    return np.uint8(cat_im)
