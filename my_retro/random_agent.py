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
from threading import Thread

class GymRunner:
    def __init__(self):
        print("initializing -------\n\n\n\n\n")
        # set up environment
        self.env = retro.make(
            game='SonicTheHedgehog-Genesis', state='LabyrinthZone.Act1')

        # set up audio streaming
        self.sr = int(self.env.em.get_audio_rate())
        assert self.sr == 44100
        self.p_audio = pyaudio.PyAudio()
        self.aud_stream = self.p_audio.open(
                            format = pyaudio.paFloat32,
                            channels = 1,
                            rate = self.sr,
                            frames_per_buffer = 512,
                            output_device_index = None,
                            output = True,
                            input = False,
                            stream_callback=self.aud_stream_callback)

        # set up video recording
        self.vid_width = 320
        self.vid_height = 224
        self.vid_filename = "recording.mp4"
        self.vid_record = (
            ffmpeg
            .input('pipe:', format='rawvideo', pix_fmt='rgb24', s='{}x{}'.format(
                self.vid_width, self.vid_height))
            #.filter('fps', fps="60")
            .output(self.vid_filename, pix_fmt='yuv420p')
            .overwrite_output()
            .run_async(pipe_stdin=True)
        )
        self.vid_frames = None

        # set up audio recording
        self.aud_filename = "sound.wav"
        self.wav_record = wave.open(self.aud_filename,'w')
        self.wav_record.setnchannels(1) # mono
        self.wav_record.setsampwidth(2)
        self.wav_record.setframerate(self.sr)
        self.aud_samples = None
        self.aud_sample_i = 0

        # other streaming vars
        self.done_streaming = False


    def aud_stream_callback(self, input_data, sample_count, time_info, status):
        data = np.zeros(sample_count)
        slice_end = min(len(self.aud_samples), self.aud_sample_i+sample_count)
        slice_len = slice_end - self.aud_sample_i
        data[:slice_len] = self.aud_samples[self.aud_sample_i:slice_end] / 2**15
        self.aud_sample_i = slice_end

        return data, self.done_streaming

    def run(self, num_steps):
        # initialize environment and video / audio recording
        obs = self.env.reset()
        self.vid_frames = [obs]
        self.vid_record.stdin.write(
            obs
            .astype(np.uint8)
            .tobytes()
        )
        samples = self.env.em.get_audio()
        self.aud_samples = samples[:,0]
        self.aud_stream.start_stream()

        for i in range(num_steps):
            print(f"vid frame / audio sample [735]: {self.aud_sample_i/(i+1)}")
            obs, rew, done, info = self.env.step(self.env.action_space.sample())
            
            # record the video frame
            self.vid_frames.append(obs)
            self.vid_record.stdin.write(
                obs
                .astype(np.uint8)
                .tobytes()
            )
            # record the audio samples in mono
            samples = self.env.em.get_audio()
            self.aud_samples = np.concatenate([self.aud_samples, samples[:,0]], axis=0)

            self.env.render()
            if done:
                break
        
        self.done_streaming = True
        self.aud_stream.stop_stream()
        self.aud_stream.close()
        self.p_audio.terminate()

        print(f"aud samples {self.aud_samples.shape}")
        self.wav_record.writeframesraw(self.aud_samples)
        self.wav_record.close()
        
        self.vid_record.stdin.close()


def main(): 
    gr = GymRunner()
    gr.run(1000)


if __name__ == '__main__':
    main()
