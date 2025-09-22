""" speech_synthesizer.py

This file handles the speech synthesis functionality for the ai tutor.
It uses the piper and vits models to synthesize speech for english and urdu.

Instructions for regex were taken from https://labex.io/tutorials/python-how-to-apply-lambda-in-regex-substitution-420893

The parts of code not written by me are referenced from the following sources:
- Code to use facebook's mms-tts-urd-script-arabic (VITS) model from https://dataloop.ai/library/model/facebook_mms-tts-urd-script_arabic/
- Code to use piper tts for english from https://github.com/OHF-Voice/piper1-gpl/blob/main/docs/API_PYTHON.md
- Code to save wav file from vits model from https://huggingface.co/docs/transformers/model_doc/vits
- Code to remove emojis from https://stackoverflow.com/questions/33404752/removing-emojis-from-a-string-in-python/49146722#49146722
- Code to convert numbers to words from https://github.com/savoirfairelinux/num2words
"""
import numpy as np
from piper import PiperVoice
from transformers import VitsModel, AutoTokenizer
import torch
import re
from scipy.io import wavfile
from num2words import num2words
from config_loader import get_tts_patterns, get_tts_symbols_map, get_emoji_regex_pattern

class SpeechSynthesizer:
    """Synthesizes speech from text using piper for English and vits for Urdu."""
    
    def __init__(self, model_path):
        """Initializes the SpeechSynthesizer with the model path and configuration."""
        self.tokenizer = None
        self.voice = None
        self.model_path = model_path
        self.first_load = True
        self.active_lang = None
        
        # Load text processing patterns and mappings from config
        tts_patterns = get_tts_patterns()

        # Markdown markers: ** __ * _
        self.MD_MARKERS = re.compile(tts_patterns.get("md_markers"))

        # Bullet leaders at start of line
        self.BULLETS = re.compile(tts_patterns.get("bullets"))

        # Numbers like 123 or 20.5
        self.NUM_PATTERN = re.compile(
            tts_patterns.get("num_pattern")
        )

        
        emoji_pattern = get_emoji_regex_pattern() 
        self.EMOJI_PATTERN = re.compile(emoji_pattern)

        # Symbols map for TTS normalization
        self.SYMBOLS_MAP = get_tts_symbols_map()
        
    def load(self, lang):
        """Loads the appropriate TTS model based on the selected language."""
        if self.active_lang == lang:
            return

        if lang == 'en':
            self.voice = PiperVoice.load(self.model_path)
            
        elif lang == 'ur':
            # Load urdu tts
            self.voice = VitsModel.from_pretrained(self.model_path)
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)

        self.active_lang = lang

    def clean_text(self, text: str) -> str:
        """Clean English text for TTS processing."""
            
        # Remove emojis and markdown
        text = self.EMOJI_PATTERN.sub("", text)
        text = self.MD_MARKERS.sub("", text)
        text = self.BULLETS.sub("", text)
        
        # Convert symbols to words
        for symbol, word in self.SYMBOLS_MAP.items():
            text = text.replace(symbol, word)
        
        # remove newlines to handle paragraphing
        text = text.replace("\n", " ")
        # Clean up extra whitespace
        text = re.sub(r"\s+", " ", text).strip()
        
        return text

    def convert_num_to_urdu_words(self, text, translator):
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
        """Synthesizes speech from text and saves it to a WAV file.
           Optional translator is used for urdu number conversion for urdu text.
        """
        self.load(lang)

        if lang == 'en':
            # clean english text for TTS
            cleaned_text = self.clean_text(text)
            
            # collect all audio chunks first
            audio_chunks = []
            for chunk in self.voice.synthesize(cleaned_text):
                # convert the audio bytes to a numpy array
                int_data = np.frombuffer(chunk.audio_int16_bytes, dtype=np.int16)
                audio_chunks.append(int_data)
            
            # concatenate all chunks into a single array
            if audio_chunks:
                full_audio = np.concatenate(audio_chunks)
                
                # save to wav file using scipy
                wavfile.write("tts_output.wav", self.voice.config.sample_rate, full_audio)
                print("Audio saved to tts_output.wav")
                
                return "tts_output.wav"  # return the file path
            return None
        
        elif lang == 'ur':
            # convert numbers to Urdu words for better TTS pronunciation
            if translator:
                processed_text = self.convert_num_to_urdu_words(text, translator)
            else:
                processed_text = text
            
            cleaned_text = self.clean_text(processed_text)

            inputs = self.tokenizer(cleaned_text, return_tensors="pt")

            with torch.no_grad():
                output = self.voice(**inputs).waveform

            # move the output tensor to the cpu
            output = output.cpu()

            # convert tensor to numpy array and remove extra dimensions
            data_np = output.numpy()
            data_np_squeezed = np.squeeze(data_np)
            
            # save the audio to a wav file
            wavfile.write("tts_output.wav", rate=self.voice.config.sampling_rate, data=data_np_squeezed)
            print("Audio saved to tts_output.wav")
            return "tts_output.wav"  # return the file path

        print("Audio processing completed!")
        return None  # return None if no audio was generated 
     