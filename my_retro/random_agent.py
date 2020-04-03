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


class GymRunner:
    def __init__(self):
        # set up environment
        self.env = retro.make(
            game='SonicTheHedgehog-Genesis', state='LabyrinthZone.Act1')

        # set up audio streaming
        self.sr = int(self.env.em.get_audio_rate())
        assert self.sr == 44100
        self.pya = pyaudio.PyAudio()
        self.aud_stream = self.pya.open(format=pyaudio.paInt16, 
                                        channels=2, rate=self.sr, output=True,
                                        frames_per_buffer=1024)

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
        self.aud_filename = "sound.wav"
        self.aud_record = wave.open(self.aud_filename,'w')
        self.aud_record.setnchannels(2) # stereo
        self.aud_record.setsampwidth(2)
        self.aud_record.setframerate(self.sr)
        self.aud_frames = None

        # other streaming vars
        self.streaming = False
        self.frame_buffer = self.fps * 3  # min 3 seconds
        self.record_q = queue.Queue()
        self.play_q = queue.Queue()

        # set up pygame
        pg.mixer.pre_init(self.sr, -16, 2, 735)
        pg.init()
        self.screen = pg.display.set_mode((self.vid_width, self.vid_height))
        self.clock = pg.time.Clock()
        


    def play_thread(self):
        def play_video(vid_frame):
            a = pg.surfarray.make_surface(vid_frame.swapaxes(0, 1))
            self.screen.blit(a, (0, 0))
            pg.display.flip()
           
        def play_audio(aud_frame):
            available = self.aud_stream.get_write_available()
            self.aud_stream.write(aud_frame.tostring())
            print(f"available {available}")

        while True:
            if not self.play_q.empty():
                # there is data to be streamed
                vid_frame, aud_frame = self.play_q.get()
            
                s = time.time()
                play_video(vid_frame)
                print(f"vid {time.time()-s}")
                
                s = time.time()
                play_audio(aud_frame)
                print(f"aud {time.time()-s}")

            elif self.streaming:
                # available data has been streamed, need to regenerate buffer
                with tqdm(total=self.frame_buffer) as pbar:
                    while self.play_q.qsize() < self.frame_buffer and self.streaming:
                        pbar.update(self.play_q.qsize() - pbar.n)
                        time.sleep(0.05)
                    pbar.update(self.frame_buffer - pbar.n)

            else:
                # all data has been streamed and there is no more coming
                self.aud_stream.stop_stream()
                self.aud_stream.close()
                self.pya.terminate()
                break
       

    def record_thread(self):
        while True:
            if not self.record_q.empty():
                # there is data to be recorded
                vid_frame, aud_frame = self.record_q.get()
                # save video frames to file
                self.vid_record.write(vid_frame)
                # save audio frames to file
                self.aud_record.writeframesraw(aud_frame)

            elif self.streaming:
                # available data has been streamed, wait a little bit
                time.sleep(0.05)
            else:
                # all data has been streamed and there is no more coming
                
                # close the audio and video files
                self.vid_record.release()
                self.aud_record.close()
                
                # combine the audio and video into a new file
                process = subprocess.Popen(
                    ['ffmpeg', '-y', '-i', self.vid_filename, '-i', self.aud_filename, '-c:v', 
                    'copy', '-c:a', 'aac', '-strict', 'experimental', 'output.mp4'])
                process.communicate()
                break

    def run(self, num_steps):
        # initialize processing thread
        self.streaming = True
        p_thread = threading.Thread(target=self.play_thread)
        r_thread = threading.Thread(target=self.record_thread)
        
        # initialize environment
        obs = self.env.reset()
        self.vid_frames = [obs]
        samples = self.env.em.get_audio()
        self.aud_frames = [samples]

        start = time.time()
        for i in range(num_steps):
            if i == 50:
                p_thread.start()
                r_thread.start()

            # obs is the video frame
            obs, rew, done, info = self.env.step(self.env.action_space.sample())
            
            # add audio and video data to queues
            vid_frame = obs
            aud_frame = self.env.em.get_audio()
            self.record_q.put((vid_frame, aud_frame))
            self.play_q.put((vid_frame, aud_frame))
           
            time.sleep(0.005)
            if done:
                break

        # take a moment to sync with stream thread
        self.streaming = False
        print(f"done streaming {time.time()-start}s")
        r_thread.join()
        print(f"done recording {time.time()-start}s")
        p_thread.join()
        print(f"done playing {time.time()-start}s")

        clip = VideoFileClip('output.mp4')
        clip.preview()


def main(): 
    gr = GymRunner()
    gr.run(1000)


if __name__ == '__main__':
    main()
