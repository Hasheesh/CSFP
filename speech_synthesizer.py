import numpy as np
import sounddevice as sd
from piper import PiperVoice
from transformers import VitsModel, AutoTokenizer
import torch
import scipy.io.wavfile
from model_registery import ModelRegistery


class SpeechSynthesizer:

    def __init__(self, model_type,model_name):
        # self.model_path = model_path
        self.voice = None
        self.model_name = model_name
        self.model_type = model_type
        self.model_reg  = ModelRegistery()
        self.first_load = True
        
    def load(self):
        model_path = self.model_reg.get_model_path(self.model_type, self.model_name)
        print("this is mode path: " + self.model_path)
        # self.model_path = "/home/gulabo/Desktop/CSFP/models/tts/en_US-lessac-medium.onnx"
        self.voice = PiperVoice.load(model_path)
        
    def process_input(self, text):
        if self.first_load:
            self.load() 
        self.first_load = not self.first_load

        # lang = 'en'

        # if model == 'piper-tts':
        # # Stream audio directly to speakers
        print(self.voice.config.sample_rate)
        for chunk in self.voice.synthesize(text):
            int_data = np.frombuffer(chunk.audio_int16_bytes, dtype=np.int16)
            # stream.write(int_data)
            sd.play(int_data, samplerate=self.voice.config.sample_rate)
            sd.wait()
        
        # elif model == 'mms-tts-ur':
        #     model = VitsModel.from_pretrained("facebook/mms-tts-urd-script_arabic")
        #     tokenizer = AutoTokenizer.from_pretrained("facebook/mms-tts-urd-script_arabic")
        #     text = "یہ اردو متن کو آواز میں تبدیل کرنے کا ایک نمونہ ہے۔"

        #     inputs = tokenizer(text, return_tensors="pt")

        #     with torch.no_grad():
        #         output = model(**inputs).waveform

        #     output = output.cpu()


        #     data_np = output.numpy()
        #     data_np_squeezed = np.squeeze(data_np)
        #     scipy.io.wavfile.write("output.wav", rate=model.config.sampling_rate, data=data_np_squeezed)


        print("Audio playback completed!")




    

    
# # Input Urdu text

# # Process text and generate speech
# inputs = processor(text=text, return_tensors="pt")
# speech = model.generate(**inputs).cpu().numpy()

# # Save generated speech to a WAV file
# sf.write("urdu_speech.wav", speech.squeeze(), samplerate=model.config.sampling_rate)

# class TTSEvaluator:
#     def __init__(self):
#         self.engines = {
#             "piper": PiperEngine("en_US-lessac-medium.onnx"),
#             "coqui": CoquiEngine(),
#             "mimic": EducationalEngine()
#         }
    
#     def benchmark(self, text="The quick brown fox jumps over the lazy dog"):
#         results = {}
#         for name, engine in self.engines.items():
#             start = time.time()
#             audio = engine.synthesize(text)
#             latency = time.time() - start
#             quality = self.rate_quality(audio)
#             results[name] = {"latency": latency, "quality": quality}
#         return results
    
#     def rate_quality(self, audio):
#         # Implement MOS (Mean Opinion Score) evaluation
#         return random.uniform(4.0, 5.0)  # Actual implementation would use real metrics