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


class Renderer:
    def __init__(self):
        # set up video recording
        self.vid_width = 320
        self.vid_height = 224
        self.fps = 60
        self.vid_filename = "recording.mp4"
        self.vid_record = cv2.VideoWriter(self.vid_filename,
                                          cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), 
                                          self.fps, (self.vid_width, self.vid_height))

        # set up audio recording
        self.sr = 44100 #int(self.env.em.get_audio_rate())
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
        self.time_diffs_f = open("data/time_diffs_1.txt", "w+")
        self.avg_times_f = open("data/avg_times_1.txt", "w+")
        self.avails_f = open("data/avails_1.txt", "w+")

        # set up avg time window
        self.window_size = 10
        self.times = collections.deque()
        self.time_diffs = []
        self.avg_times = []
        self.avails = []


    def close(self):
        # close the audio and video files
        self.vid_record.release()
        self.aud_record.close()
        
        # stop audio streaming
        self.aud_stream.stop_stream()
        self.aud_stream.close()
        self.pya.terminate()

        # close the monitor files
        for td, avgt, avail in zip(self.time_diffs, self.avg_times, self.avails):
            self.time_diffs_f.write(f"{td}\n")
            self.avg_times_f.write(f"{avgt}\n")
            self.avails_f.write(f"{avail}\n")
        self.time_diffs_f.close()
        self.avg_times_f.close()
        self.avails_f.close()

        # combine the audio and video into a new file
        process = subprocess.Popen(
            ['ffmpeg', '-y', '-i', self.vid_filename, '-i', self.aud_filename, '-c:v', 
            'copy', '-c:a', 'aac', '-strict', 'experimental', 'output.mp4'])
        process.communicate()
        
        plt.plot(self.time_diffs)
        plt.plot(self.avg_times)
        plt.show()
        plt.plot(np.array(self.time_diffs) / max(self.time_diffs))
        plt.plot(np.array(self.avg_times) / max(self.avg_times))
        plt.plot(np.array(self.avails) / max(self.avails))
        plt.show()
        plt.plot(self.avails)
        plt.show()


    def render(self, vid_frame, aud_frame, smooth_audio=True, save=True):
        available = self.aud_stream.get_write_available()
        #print(f"available {available}")
        
        # play video
        self.viewer.imshow(vid_frame)

        # resample the audio and play
        res_time = time.time()
        if not self.times:
            # first iteration, estimate real time playback
            self.times.append(res_time - 1 / self.fps)
        if len(self.times) >= self.window_size:
            t = self.times.popleft()
            avg_time_diff = (res_time - t) / self.window_size
        else:
            t = self.times[0]
            avg_time_diff = (res_time - t) / (len(self.times)+1)

        res_factor = max(1 / self.max_slowdown, 1 / self.fps / avg_time_diff)
        #print(f"  resample_factor {resample_factor}, avg_time_diff {avg_time_diff}s")
        desired_x = np.arange(0, aud_frame.shape[0], res_factor)[:available]
        current_x = np.arange(0, aud_frame.shape[0])
        current_y = aud_frame[:,0]
        res_a_f = np.interp(desired_x, current_x, current_y).astype(np.int16)
        self.aud_stream.write(res_a_f.tostring())
    
        # add to the buffer using resample time
        self.times.append(res_time)
        self.time_diffs.append(res_time - self.times[-2])
        self.avg_times.append(avg_time_diff)
        self.avails.append(available)
            
        if False:
            # play with no resampling and risk choppy audio, cutting off samples
            self.viewer.imshow(vid_frame)
            self.aud_stream.write(aud_frame[:available,0].tostring())
        
        if save:
            # write data to file
            self.vid_record.write(vid_frame)
            self.aud_record.writeframesraw(aud_frame)
