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
import sounddevice as sd
from image_viewer import SimpleImageViewer

class GymRunner:
    def __init__(self):
        # set up environment
        self.env = retro.make(
            game='SonicTheHedgehog-Genesis', state='LabyrinthZone.Act1')

        # set up video recording
        self.vid_width = 320
        self.vid_height = 224
        self.fps = 60
        self.vid_filename = "recording.mp4"
        self.vid_record = cv2.VideoWriter(self.vid_filename,
                                          cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), 
                                          self.fps, (self.vid_width, self.vid_height))
        self.vid_frames = None

        # set up audio recording
        self.sr = int(self.env.em.get_audio_rate())
        assert self.sr == 44100
        self.aud_filename = "sound.wav"
        self.aud_record = wave.open(self.aud_filename,'w')
        self.aud_record.setnchannels(2) # stereo
        self.aud_record.setsampwidth(2)
        self.aud_record.setframerate(self.sr)
        self.aud_frames = None

        # set up video streaming
        self.viewer = SimpleImageViewer()

        # set up audio streaming
        self.pya = pyaudio.PyAudio()
        self.frames_per_buffer = 4096
        self.aud_stream = self.pya.open(format=pyaudio.paInt16, 
                                        channels=1, rate=self.sr, output=True,
                                        frames_per_buffer=self.frames_per_buffer)

        # other streaming vars
        self.last_stream = None
        self.last_vid_frame = None
        self.last_aud_frame = None
        # 16x slowdown -> 11760 samples/frame, 0.26 sec/frame
        # BERT inference time on optimized CPU: 0.05 sec
        self.max_slowdown = 16

    def clean_up(self):
        # close the audio and video files
        self.vid_record.release()
        self.aud_record.close()
        
        # stop audio streaming
        self.aud_stream.stop_stream()
        self.aud_stream.close()
        self.pya.terminate()
        
        # combine the audio and video into a new file
        process = subprocess.Popen(
            ['ffmpeg', '-y', '-i', self.vid_filename, '-i', self.aud_filename, '-c:v', 
            'copy', '-c:a', 'aac', '-strict', 'experimental', 'output.mp4'])
        process.communicate()


    def render(self, vid_frame, aud_frame, smooth_audio=True, save=True):
        available = self.aud_stream.get_write_available()
        print(f"available {available}")
        if smooth_audio:
            # play with 1 frame delay so we can resample audio
            if self.last_stream is None:
                # create the 1 frame delay
                self.last_vid_frame = vid_frame
                self.last_aud_frame = aud_frame
                self.last_stream = time.time()
            
            else:
                # play the last vid frame
                self.viewer.imshow(self.last_vid_frame)
                self.last_vid_frame = vid_frame
                
                # resample the last aud frame and play
                time_diff = time.time() - self.last_stream
                resample_factor = max(1 / self.max_slowdown, 1 / self.fps / time_diff)
                desired_x = np.arange(0, self.last_aud_frame.shape[0], resample_factor)
                current_x = np.arange(0, self.last_aud_frame.shape[0])
                current_y = self.last_aud_frame[:,0]
                res_aud_frame = np.interp(desired_x, current_x, current_y).astype(np.int16)
                self.aud_stream.write(res_aud_frame.tostring())
                self.last_aud_frame = aud_frame
                
                self.last_stream = time.time()
        else:
            # play with no resampling and risk choppy audio
            self.viewer.imshow(vid_frame)
            self.aud_stream.write(aud_frame[:,0].tostring())
        
        if save:
            # write data to file
            self.vid_record.write(vid_frame)
            self.aud_record.writeframesraw(aud_frame)

    
    def run(self, num_steps):
        # initialize environment
        obs = self.env.reset()
        self.vid_frames = [obs]
        samples = self.env.em.get_audio()
        self.aud_frames = [samples]

        start = time.time()
        for i in range(num_steps):
            # obs is the video frame
            s = time.time()
            #time.sleep(np.random.random()/10+0.05)
            time.sleep(0.01)
            
            obs, rew, done, info = self.env.step(self.env.action_space.sample())
            
            vid_frame = obs
            aud_frame = self.env.em.get_audio()
            self.render(vid_frame, aud_frame, smooth_audio=True)
            
            print(f"{1 / self.fps / (time.time()-s)}")
            if done:
                break

        print(f"total time {time.time()-start}s")
        self.clean_up()
        clip = VideoFileClip('output.mp4')
        clip.preview()


def main(): 
    gr = GymRunner()
    gr.run(1000)


if __name__ == '__main__':
    main()
