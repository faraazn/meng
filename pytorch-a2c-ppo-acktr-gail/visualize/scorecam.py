"""
Created on Wed Apr 29 16:11:20 2020
@author: Haofan Wang - github.com/haofanwang
"""
from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F

from visualize.misc_functions import get_example_params, save_class_activation_images


class CamExtractor():
    """
        Extracts cam features from the model
    """
    def __init__(self, actor_critic, obs_target):
        self.actor_critic = actor_critic
        self.obs_target = obs_target

    def forward_pass_on_convolutions(self, obs):
        """
            Does a forward pass on convolutions, hooks the function at given layer
        """
        conv_outputs = {}
        x = []
        for obs_name in obs.keys():
            if obs_name == 'video':
                module = self.actor_critic.base.video_module
                # TODO: make this nicer by including divide in module
                vid_x = obs['video'] / 255
                for m_pos, m in module._modules.items():
                    vid_x = m(vid_x)
                    if obs_name in self.obs_target.keys() and self.obs_target[obs_name] == int(m_pos):
                        conv_outputs[obs_name] = vid_x
                x.append(vid_x)
            elif obs_name == 'audio':
                module = self.actor_critic.base.audio_module
                aud_x = obs['audio']
                for m_pos, m in module._modules.items():
                    aud_x = m(aud_x)
                    if obs_name in self.obs_target.keys() and self.obs_target[obs_name] == m_pos:
                        conv_outputs[obs_name] = aud_x
                x.append(aud_x)
            else:
                raise NotImplementedError
        x = torch.cat(x, dim=1)
        return conv_outputs, x

    def forward_pass(self, x):
        """
            Does a full forward pass on the model
        """
        # Forward pass on the model, saving the conv outputs
        conv_outputs, x = self.forward_pass_on_convolutions(x)
        x = self.actor_critic.dist(x)
        return conv_outputs, x


class ScoreCam():
    """
        Produces class activation map
    """
    def __init__(self, actor_critic, tgt_layers):
        self.actor_critic = actor_critic
        self.actor_critic.eval()
        # Define extractor
        self.extractor = CamExtractor(self.actor_critic, tgt_layers)

    def generate_cam(self, obs, target_class):
        # Full forward pass
        # conv_output is the output of convolutions at specified layer
        # model_output is the final output of the model (1, 1000)
        conv_outputs, model_output = self.extractor.forward_pass(obs)
        cams = {}
        for obs_name in conv_outputs.keys():
            # Get convolution outputs
            target = conv_outputs[obs_name][0]
            # Create empty numpy array for cam
            cam = np.ones(target.shape[1:], dtype=np.float32)
            # Multiply each weight with its conv output and then, sum
            for i in range(len(target)):
                # Unsqueeze to 4D
                saliency_map = torch.unsqueeze(torch.unsqueeze(target[i, :, :],0),0)
                # Upsampling to input size
                saliency_map = F.interpolate(saliency_map, size=(224, 320), mode='bilinear', align_corners=False)
                if saliency_map.max() == saliency_map.min():
                    continue
                # Scale between 0-1
                norm_saliency_map = (saliency_map - saliency_map.min()) / (saliency_map.max() - saliency_map.min())
                # Get the target score
                new_obs = obs.copy()
                new_obs[obs_name] = obs[obs_name]*norm_saliency_map
                w = self.extractor.forward_pass(new_obs)[1]
                w = w.probs[0][target_class]
                cam += w.cpu().data.numpy() * target[i, :, :].cpu().data.numpy()
            cam = np.maximum(cam, 0)
            cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam))  # Normalize between 0-1
            cam = np.uint8(cam * 255)  # Scale between 0-255 to visualize
            cam = np.uint8(Image.fromarray(cam).resize((320, 224), Image.ANTIALIAS))/255
            cams[obs_name] = cam
        return cams
