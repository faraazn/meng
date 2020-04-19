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
import collections
import matplotlib.pyplot as plt


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

        # set up audio recording
        self.sr = int(self.env.em.get_audio_rate())
        assert self.sr == 44100
        self.aud_filename = "sound.wav"
        self.aud_record = wave.open(self.aud_filename,'w')
        self.aud_record.setnchannels(2) # stereo
        self.aud_record.setsampwidth(2)
        self.aud_record.setframerate(self.sr)

        # set up video streaming
        self.viewer = SimpleImageViewer()

        # set up audio streaming
        self.pya = pyaudio.PyAudio()
        self.frames_per_buffer = 4096
        self.aud_stream = self.pya.open(format=pyaudio.paInt16, 
                                        channels=1, rate=self.sr, output=True,
                                        frames_per_buffer=self.frames_per_buffer)

        # other streaming vars
        # 16x slowdown -> 11760 samples/frame, 0.26 sec/frame
        # BERT inference time on optimized CPU: 0.05 sec
        self.max_slowdown = 16
        self.time_diffs_f = open("data/time_diffs_no_overwrite.txt", "w+")
        self.write_times_f = open("data/write_times_test.txt", "w+")
        self.viz_times_f = open("data/viz_times_test.txt", "w+")

        # set up frame buffer
        self.frame_buf = collections.deque()
        self.frame_buf_size = 10
        self.times = []
        self.avg_times = []
        self.avails = []


    def clean_up(self):
        # close the audio and video files
        self.vid_record.release()
        self.aud_record.close()
        
        # stop audio streaming
        self.aud_stream.stop_stream()
        self.aud_stream.close()
        self.pya.terminate()

        # close the time diff file
        self.time_diffs_f.close()
        self.write_times_f.close()
        self.viz_times_f.close()

        # combine the audio and video into a new file
        process = subprocess.Popen(
            ['ffmpeg', '-y', '-i', self.vid_filename, '-i', self.aud_filename, '-c:v', 
            'copy', '-c:a', 'aac', '-strict', 'experimental', 'output.mp4'])
        process.communicate()


    def render(self, vid_frame, aud_frame, smooth_audio=True, save=True):
        available = self.aud_stream.get_write_available()
        print(f"available {available}")
        if smooth_audio:
            
            if len(self.frame_buf) >= self.frame_buf_size:
                # take from the buffer
                v_f, a_f, t = self.frame_buf.popleft()

                # play video
                self.viewer.imshow(v_f)

                # resample the audio and play
                res_time = time.time()
                avg_time_diff = (res_time - t) / (self.frame_buf_size - 1)
                res_factor = max(1 / self.max_slowdown, 1 / self.fps / avg_time_diff)
                #print(f"  resample_factor {resample_factor}, avg_time_diff {avg_time_diff}s")
                desired_x = np.arange(0, a_f.shape[0], res_factor)[:available]
                current_x = np.arange(0, a_f.shape[0])
                current_y = a_f[:,0]
                res_a_f = np.interp(desired_x, current_x, current_y).astype(np.int16)
                self.aud_stream.write(res_a_f.tostring())
            
                # add to the buffer using resample time
                self.frame_buf.append((vid_frame, aud_frame, res_time))
                self.times.append(res_time - self.frame_buf[-2][2])
                self.avg_times.append(avg_time_diff)
                self.avails.append(available)
            else:
                # add to the buffer
                time.sleep(0.001)
                self.frame_buf.append((vid_frame, aud_frame, time.time()))
                
        else:
            # play with no resampling and risk choppy audio, cutting off samples
            self.viewer.imshow(vid_frame)
            self.aud_stream.write(aud_frame[:available,0].tostring())
        
        if save:
            # write data to file
            self.vid_record.write(vid_frame)
            self.aud_record.writeframesraw(aud_frame)

    
    def run(self, num_steps):
        # initialize environment
        obs = self.env.reset()
        aud_frame = self.env.em.get_audio()

        start = time.time()
        for i in range(num_steps):
            s = time.time()
            # obs is the video frame
            #time.sleep(np.random.random()/5*i/1000+0.00)
            time.sleep(0.005)
            
            obs, rew, done, info = self.env.step(self.env.action_space.sample())
            
            vid_frame = obs
            aud_frame = self.env.em.get_audio()
            self.render(vid_frame, aud_frame, smooth_audio=True)
            #print(f"{1 / self.fps / (time.time()-s)}")
            if done:
                #self.finish_render()
                break

        print(f"total time {time.time()-start}s")
        self.clean_up()
        
        plt.plot(self.times)
        plt.plot(self.avg_times)
        plt.show()
        plt.plot(self.avails)
        plt.show()

        #clip = VideoFileClip('output.mp4')
        #clip.preview()


def main(): 
    gr = GymRunner()
    gr.run(1000)


if __name__ == '__main__':
    main()
