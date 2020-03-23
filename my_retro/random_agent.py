import retro
import time
import numpy as np
import pyaudio

def main():
    print("generating frames")
    env = retro.make(game='SonicTheHedgehog-Genesis', state='LabyrinthZone.Act1')
    obs = env.reset()
    frames = []
    for _ in range(1000):
        frames.append(env.em.get_audio()[:,0])

        obs, rew, done, info = env.step(env.action_space.sample())
        fs = env.em.get_audio_rate()
        assert fs == 44100
        env.render()
        if done:
            obs = env.reset()

    frames = np.concatenate(frames, axis=0)
    print(frames.shape)

    print("playing audio")
    audio = pyaudio.PyAudio()
    stream = audio.open(format = pyaudio.paFloat32,
                        channels = 1,
                        rate = 44100,
                        frames_per_buffer = 2048,
                        output_device_index = None,
                        output = True,
                        input = False)

    start = time.time()
    frame_i = 0
    num_frames = stream.get_write_available()
    while frame_i+num_frames < len(frames):
        if num_frames > 0:
            data = frames[frame_i:frame_i+num_frames]
            stream.write(data.astype(np.float32).tostring())
            frame_i += num_frames
        num_frames = stream.get_write_available()
    print(frame_i)
    print(f"{time.time()-start}s")


if __name__ == '__main__':
  main()
