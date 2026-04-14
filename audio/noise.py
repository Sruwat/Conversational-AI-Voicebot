import noisereduce as nr
import numpy as np

def reduce_noise(audio, sr=16000, noise_sample=None):
    if audio is None or len(audio) == 0:
        return audio
    noise_clip = noise_sample
    if noise_clip is None or len(noise_clip) == 0:
        return audio
    reduced = nr.reduce_noise(y=audio, sr=sr, y_noise=noise_clip, prop_decrease=1.0)
    return reduced
