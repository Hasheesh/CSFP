import numpy as np
import sounddevice as sd
from piper import PiperVoice
from transformers import VitsModel, AutoTokenizer
import torch
import gc
import scipy.io.wavfile
from model_registery import ModelRegistery
# from speech_recognizer import SpeechRecognizer
import psutil
import time
import os
    # Get virtual memory information
def proc_mem():
    p = psutil.Process(os.getpid())
    rss_mb = p.memory_info().rss / (1024**2)
    return f"Process RSS: {rss_mb:.1f} MiB"

class SpeechSynthesizer:

    def __init__(self, model_path):
        # self.model_path = model_path
        self.tokenizer = None
        self.voice = None
        self.model_path = model_path
        self.first_load = True
        self.active_lang = None
        
    def load(self, lang):

        if self.active_lang == lang:
            return

        self.unload()  # clear previous 


        if lang == 'en':
            print("this is mode path: " + self.model_path)
            self.voice = PiperVoice.load(self.model_path)
            

        elif lang == 'ur':
            # Load urdu tts
            self.voice = VitsModel.from_pretrained(self.model_path)
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)

        self.active_lang = lang
        print("[load]", lang, proc_mem())

    def process_input(self, text, lang):
        self.load(lang)

        # use piper tts

        if lang == 'en':
        # Stream audio directly to speakers
            print(self.voice.config.sample_rate)
            for chunk in self.voice.synthesize(text):
                int_data = np.frombuffer(chunk.audio_int16_bytes, dtype=np.int16)
                # stream.write(int_data)
                sd.play(int_data, samplerate=self.voice.config.sample_rate)
                print(self.voice.config.sample_rate)

                sd.wait()
        
        elif lang == 'ur':
            
            # text1 = "یہ اردو متن کو آواز میں تبدیل کرنے کا ایک نمونہ ہے۔"

            inputs = self.tokenizer(text, return_tensors="pt")

            with torch.no_grad():
                output = self.voice(**inputs).waveform

            output = output.cpu()

            data_np = output.numpy()
            data_np_squeezed = np.squeeze(data_np)
            # scipy.io.wavfile.write("output.wav", rate=self.model.config.sampling_rate, data=data_np_squeezed)
            sd.play(data_np_squeezed, samplerate=self.voice.config.sampling_rate)
            sd.wait()

        print("Audio playback completed!")
        print("Audio playback completed!")
        print("[after playback]", proc_mem())



    def unload(self):
        if self.active_lang is None and not (self.voice or self.tokenizer):
            return

        print(f"[unload] {self.active_lang} ...")


        del self.voice
        self.voice = None
        del self.tokenizer
        self.tokenizer = None

        self.active_lang = None


        # run GC; give OS a tick to reclaim
        gc.collect()
        time.sleep(0.1)

        print("[after unload]", proc_mem())
    

# print(proc_mem())  # before load
# synth = SpeechSynthesizer("models/tts/piper-tts-en/en_US-lessac-medium.onnx")
# synth.process_input("Hello...this is an Ai tutor's voice, Welcome!", "en")
# print(proc_mem())  # stays higher (model kept)
# synth.unload()
# print(proc_mem()) 

# synth1 = SpeechSynthesizer('models/tts/piper-tts-en/en_US-lessac-medium.onnx')
# synth1.process_input("Hello this is an AI tutor. Welcome!", 'en')
# print(proc_mem())  # stays higher (model kept)
# synth1.unload()
# print(proc_mem()) 
# synth2 = SpeechSynthesizer('models/tts/mss-tts-urd-script-arabic')
# synth2.process_input("یہ اردو متن کو آواز میں تبدیل کرنے کا ایک نمونہ ہے۔", 'ur')
# print(proc_mem())  # stays higher (model kept)
# synth2.unload()
# print(proc_mem()) 
     