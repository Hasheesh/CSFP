''' Instructions for regex were taken from https://labex.io/tutorials/python-how-to-apply-lambda-in-regex-substitution-420893
'''
import numpy as np
import sounddevice as sd
from piper import PiperVoice
from transformers import VitsModel, AutoTokenizer
import torch
import gc
from model_registery import ModelRegistery
import psutil
import time
import os
import re
from scipy.io import wavfile
from num2words import num2words


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
        
        # Text processing patterns and mappings for TTS
        self.MD_MARKERS = re.compile(r"(\*\*|__|\*|_)")
        self.BULLETS = re.compile(r"(?m)^\s*[*\-+•●▪︎◦·]\s+")
        self.EMOJI_PATTERN = re.compile("["
                                       "\U0001F300-\U0001FAFF"
                                       "\U00002700-\U000027BF"
                                       "\U00002600-\U000026FF"
                                       "\U0001F1E6-\U0001F1FF"
                                       "]+", re.UNICODE)
        self.NUM_PATTERN = re.compile(r"\b\d{1,9}(?:\.\d+)?\b")
        
        # Symbol mappings for English TTS
        self.SYMBOLS_MAP = {
            "&": " and ",
            "%": " percent ",
            "+": " plus ",
            "-": " minus ",
            "x": " times ",
            "÷": " divided by ",
            "=": " equal to ",
            "π": " pi ",
            "°": " degree ",
            "€": " euros ",
            "$": " dollars ",
            "£": " pounds ",
            "@": " at ",
            "#": " number ",
            "°C": " degrees Celsius ",
            "°F": " degrees Fahrenheit ",
        }
        
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

    def clean_text_for_english_tts(self, text: str) -> str:
        """Clean English text for TTS processing."""
        if not text:
            return ""
            
        # Remove emojis and markdown
        text = self.EMOJI_PATTERN.sub("", text)
        text = self.MD_MARKERS.sub("", text)
        text = self.BULLETS.sub("", text)
        
        # Convert symbols to words
        for symbol, word in self.SYMBOLS_MAP.items():
            text = text.replace(symbol, word)
        
        # Clean up extra whitespace
        text = re.sub(r"\s+", " ", text).strip()
        
        return text



    def convert_num_to_urdu_words(self, text: str, translator) -> str:
        """Convert numbers to Urdu words via English translation for better Urdu TTS."""
        if not text or not translator:
            return text
            
        numbers = self.NUM_PATTERN.findall(text)
        
        for number in numbers:
            if not number or not number.strip():
                continue
            
            if "." in number:
                # for decimal numbers like "3.14"
                whole, decimal = number.split(".", 1)
                if not whole or not decimal:
                    continue
                    
                whole_words = num2words(int(whole))
                decimal_digits = [d for d in decimal if d.isdigit()]
                if not decimal_digits:
                    continue
                decimal_words = " ".join(num2words(int(d)) for d in decimal_digits)
                english_words = f"{whole_words} point {decimal_words}"
            else:
                # for whole numbers like "123"
                english_words = num2words(int(number))
            

            urdu_words = translator.process_input(english_words, 'en')
            if urdu_words and urdu_words.strip():
                text = text.replace(number, urdu_words)
                    
        
        return text

    def process_input(self, text, lang, translator=None):
        self.load(lang)

        if lang == 'en':
            # Clean English text for TTS
            cleaned_text = self.clean_text_for_english_tts(text)
            
            print(self.voice.config.sample_rate)

            # Collect all audio chunks first
            audio_chunks = []
            for chunk in self.voice.synthesize(cleaned_text):
                # Convert the audio bytes to a numpy array
                int_data = np.frombuffer(chunk.audio_int16_bytes, dtype=np.int16)
                audio_chunks.append(int_data)
            
            # Concatenate all chunks into a single array
            if audio_chunks:
                full_audio = np.concatenate(audio_chunks)
                
                # Save to WAV file using scipy
                wavfile.write("tts_output.wav", self.voice.config.sample_rate, full_audio)
                print("Audio saved to tts_output.wav")
                
                return "tts_output.wav"  # Return the file path
            return None
        
        elif lang == 'ur':
            # Convert numbers to Urdu words for better TTS pronunciation
            if translator:
                processed_text = self.convert_num_to_urdu_words(text, translator)
            else:
                processed_text = text
            
            # Clean up extra whitespace
            processed_text = re.sub(r"\s+", " ", processed_text).strip()

            inputs = self.tokenizer(processed_text, return_tensors="pt")

            with torch.no_grad():
                output = self.voice(**inputs).waveform

            output = output.cpu()

            data_np = output.numpy()
            data_np_squeezed = np.squeeze(data_np)
            wavfile.write("tts_output.wav", rate=self.voice.config.sampling_rate, data=data_np_squeezed)
            print("Audio saved to tts_output.wav")
            return "tts_output.wav"  # Return the file path

        print("Audio processing completed!")
        print("[after processing]", proc_mem())
        return None  # Return None if no audio was generated



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
# synth = SpeechSynthesizer("models/tts/piper-tts-en-amy/en_US-amy-medium.onnx")
# synth.process_input("Hello...this is an Ai tutor's voice, Welcome!", "en")
# print(proc_mem())  # stays higher (model kept)
# synth.unload()
# print(proc_mem()) 

# synth1 = SpeechSynthesizer('models/tts/piper-tts-en-amy/en_US-amy-medium.onnx')
# synth1.process_input("Hello this is an AI tutor. Welcome!", 'en')
# print(proc_mem())  # stays higher (model kept)
# synth1.unload()
# # print(proc_mem()) 
# # synth2 = SpeechSynthesizer('models/tts/mss-tts-urd-script-arabic')
# # synth2.process_input("یہ اردو متن کو آواز میں تبدیل کرنے کا ایک نمونہ ہے۔", 'ur')
# # print(proc_mem())  # stays higher (model kept)
# # synth2.unload()
# # print(proc_mem()) 
     