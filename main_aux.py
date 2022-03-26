import features
import data_sets
import modelos
import time
import soundfile
import numpy as np

sp1_00 = './audios/yeah.wav'
user01 = [sp1_00]

audio, sr = features.load_file(sp1_00)
emph = features.pre_emphasis(audio)
frames = features.framing(emph)
print(frames.shape)