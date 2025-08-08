
''' took the whisper transcribing usage from whisper example and documentation at https://huggingface.co/openai/whisper-tiny
    the rest of the code was all written by me '''
import soundfile as sf
import librosa
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from model_registery import ModelRegistery
# from speech_recognizer import SpeechRecognizer
import psutil
import time
import os
import gc
    # Get virtual memory information
def proc_mem():
    p = psutil.Process(os.getpid())
    rss_mb = p.memory_info().rss / (1024**2)
    return f"Process RSS: {rss_mb:.1f} MiB"


class SpeechRecognizer:

    def __init__(self, model_path):
        self.speech_rec = None 
        self.active_lang = None  
        self.processsor = None
        self.model_path = model_path
        self.first_load = True    
        self.active_lang = None
        
    def load(self):
        self.processor = WhisperProcessor.from_pretrained(self.model_path)
        self.speech_rec     = WhisperForConditionalGeneration.from_pretrained(self.model_path)
        print(f"Successfully loaded {self.model_path}")


    def process_input(self, audio_path, lang):
        # Read audio, resampling if needed
        if self.first_load:
            self.load() 
            self.first_load = not self.first_load

        self.active_lang = lang 

        audio, sr = sf.read(audio_path)

        if sr != self.processor.feature_extractor.sampling_rate:
            audio = librosa.resample(audio, orig_sr=sr, 
                                    target_sr=self.processor.feature_extractor.sampling_rate)

            sr = self.processor.feature_extractor.sampling_rate

        input = self.processor.feature_extractor(audio, sampling_rate=sr, 
                                            return_tensors="pt")
        input_features = input.input_features.to(self.speech_rec.device) 

        
        generated_ids = self.speech_rec.generate(
            input_features, 
            max_length=2000,
            num_beams=5,
            task="transcribe",
            language=lang,
            early_stopping=True
        )

        # 5. Decode to text
        transcription = self.processor.decode(generated_ids[0], skip_special_tokens=True)
        print(transcription)
        return transcription

    def unload(self):
        if self.active_lang is None and not (self.speech_rec or self.processor):
            return

        print(f"[unload] {self.active_lang} ...")

        del self.speech_rec
        self.speech_rec = None
        del self.processor
        self.processor = None

        # run GC; give OS a tick to reclaim
        gc.collect()
        time.sleep(0.1)

        print("[after unload]", proc_mem())

# print(proc_mem()) 
# synth2 = SpeechRecognizer('models/stt/whisper-small')
# synth2.process_input('output1.wav', 'ur')
# print(proc_mem())  # stays higher (model kept)
# synth2.unload()
# print(proc_mem()) 