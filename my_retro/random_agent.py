import retro
import time
import numpy as np
import pyaudio
import ffmpeg
import wave
import struct
import math
import random
import cv2
import subprocess
from moviepy.editor import VideoFileClip
import pygame as pg
import threading
from tqdm import tqdm
import queue
from image_viewer import SimpleImageViewer
import collections
import matplotlib.pyplot as plt
from renderer import Renderer

class GymRunner:
    def __init__(self):
        # set up environment
        self.env = retro.make(
            game='SonicTheHedgehog-Genesis', state='LabyrinthZone.Act1')

        # initialize renderer
        self.renderer = Renderer()
    
    def run(self, num_steps):
        # initialize environment
        obs = self.env.reset()
        aud_frame = self.env.em.get_audio()
        
        start = time.time()
        for i in range(num_steps):
            s = time.time()
            # obs is the video frame
            #time.sleep(np.random.random()/5*i/1000+0.00)
            #time.sleep(0.005)
            
            obs, rew, done, info = self.env.step(self.env.action_space.sample())
            
            vid_frame = obs
            aud_frame = self.env.em.get_audio()
            if rew != 0:
                print(f"rew {rew}, x {info['x']}")
            self.renderer.render(vid_frame, aud_frame, smooth_audio=True)
            
            if done:
                break

        print(f"total time {time.time()-start}s")
        self.renderer.close()
        


def main(): 
    gr = GymRunner()
    gr.run(100000)


if __name__ == '__main__':
    main()
