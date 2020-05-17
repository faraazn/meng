"""
Created on Wed Apr 29 16:11:20 2020
@author: Haofan Wang - github.com/haofanwang
"""
from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F

from misc_functions import get_example_params, save_class_activation_images


class CamExtractor():
    """
        Extracts cam features from the model
    """
    def __init__(self, features, classifier, target_layer):
        self.features = features
        self.classifier = classifier
        self.target_layer = target_layer

    def forward_pass_on_convolutions(self, x):
        """
            Does a forward pass on convolutions, hooks the function at given layer
        """
        conv_output = None
        for module_pos, module in self.features._modules.items():
            x = module(x)  # Forward
            if int(module_pos) == self.target_layer:
                conv_output = x  # Save the convolution output on that layer
        return conv_output, x

    def forward_pass(self, x):
        """
            Does a full forward pass on the model
        """
        # Forward pass on the convolutions (ends in a flatten)
        conv_output, x = self.forward_pass_on_convolutions(x)
        # Forward pass on the classifier
        x = self.classifier(x)
        return conv_output, x


class ScoreCam():
    """
        Produces class activation map
    """
    def __init__(self, features, classifier, target_layer):
        self.features = features
        self.classifier = classifier
        self.features.eval()
        self.classifier.eval()
        # Define extractor
        self.extractor = CamExtractor(self.features, self.classifier, target_layer)

    def generate_cam(self, forward_image, target_class):
        # Full forward pass
        # conv_output is the output of convolutions at specified layer
        # model_output is the final output of the model (1, 1000)
        conv_output, model_output = self.extractor.forward_pass(forward_image)
        # Get convolution outputs
        target = conv_output[0]
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
            w = self.extractor.forward_pass(forward_image*norm_saliency_map)[1]
            w = w.probs[0][target_class]
            cam += w.cpu().data.numpy() * target[i, :, :].cpu().data.numpy()
        cam = np.maximum(cam, 0)
        cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam))  # Normalize between 0-1
        cam = np.uint8(cam * 255)  # Scale between 0-255 to visualize
        cam = np.uint8(Image.fromarray(cam).resize((320, 224), Image.ANTIALIAS))/255
        return cam