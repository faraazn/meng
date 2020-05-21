import torch
import numpy as np
import time

from visualize.scorecam import CamExtractor, ScoreCam
from visualize.misc_functions import apply_colormap_on_image

from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw
from PIL import ImageEnhance

ACTIONS = ('L ', 'R ', 'LD', 'RD', 'D ', 'DB', 'B ')


def gen_eval_vid_frame(actor_critic, env_state, x, max_x, pct, rew, t, action, logits, obs, tgt_layers):
    actor_critic.eval()

    score_cam = ScoreCam(actor_critic, tgt_layers)
    cams = score_cam.generate_cam(obs, action)
    cam = cams['video']

    obs_array = np.uint8(obs['video'][0].detach().cpu().numpy()).transpose((1,2,0))  # shape [224, 320, 3]
    obs_im = Image.fromarray(obs_array, 'RGB')  # size [320, 224]

    hm, hm_on_im = apply_colormap_on_image(obs_im, cam, 'hsv')
    hm = ImageEnhance.Brightness(hm).enhance(0.75)  # heatmap is usually too bright

    cat_im = Image.new('RGB', (320*3, 224+50))
    cat_im.paste(obs_im, (0,0))
    cat_im.paste(hm_on_im, (320,0))
    cat_im.paste(hm, (640,0))

    logits = logits.detach().cpu().numpy()[0]
    logits_txt = [f"{logit:.2f}" for logit in logits]
    draw = ImageDraw.Draw(cat_im)
    draw.text(
        (0, 224+4),
        f"""    {env_state} | t: {t}
             x: {x:07.2f} | max_x: {max_x:07.2f} | pct: {pct:05.2f}% | rew: {rew:07.2f}
             action: {ACTIONS[action]} | logits: {logits_txt}""",
        (255,255,255))
    return np.uint8(cat_im)
