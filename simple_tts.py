from transformers import VitsModel, AutoTokenizer
import torch
import scipy.io.wavfile
import numpy as np

# model = VitsModel.from_pretrained("facebook/mms-tts-urd-script_arabic")
# tokenizer = AutoTokenizer.from_pretrained("facebook/mms-tts-urd-script_arabic")
# text = "یہ اردو متن کو آواز میں تبدیل کرنے کا ایک نمونہ ہے۔"

# inputs = tokenizer(text, return_tensors="pt")

# with torch.no_grad():
#     output = model(**inputs).waveform

# output = output.cpu()


# data_np = output.numpy()
# data_np_squeezed = np.squeeze(data_np)
# scipy.io.wavfile.write("output.wav", rate=model.config.sampling_rate, data=data_np_squeezed)

import wave
from piper import PiperVoice

voice = PiperVoice.load("models/tts/piper-tts/en_US-lessac-medium.onnx")


# voice.config.sample_rate = 16000
with wave.open("test.wav", "wb") as wav_file:
    voice.synthesize_wav("what is photosynthesis?", wav_file)

    