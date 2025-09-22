"""
translator.py

This file handles the text translation functionality for the AI tutor.
It uses CTranslate2 to run QuickMT models (for ARM) and NLLB (for x86_64)
to translate text between English and Urdu.

The parts of code not written by me are referenced from the following sources:
- Code to use quickmt translator model from https://huggingface.co/quickmt/quickmt-en-ur
- Code to use ctranslate2 from the docs at https://opennmt.net/CTranslate2/python/overview.html
- The tokenizer for the ARM version is from Helsinki-NLP OPUS-MT models.
"""
import os
import json
from transformers import AutoTokenizer, MarianTokenizer
import ctranslate2
import re
from config_loader import get_emoji_regex_pattern, get_urdu_numbers


class Translator:
    """Translates text between English and Urdu using a CTranslate2 model."""

    def __init__(self, version, model_path, source_lang=None, target_lang=None):
        """Initializes the Translator with the model path and configuration."""
        self.version = version
        self.model_path = model_path
        self.source_lang = source_lang
        self.target_lang = target_lang
        
        self.translator = None
        self.tokenizer = None
        self.first_load = True

    def load(self):
        """Loads the translation model and tokenizer into memory."""
        tokenizer_path = self.model_path
        if self.version == 'arm':
            # For ARM, we use a compatible tokenizer from Helsinki-NLP, chosen
            # based on the translation direction.
            if self.source_lang == 'en' and self.target_lang == 'ur':
                tokenizer_path = "models/translation/opus-mt-en-ur-tokenizer"
            elif self.source_lang == 'ur' and self.target_lang == 'en':
                tokenizer_path = "models/translation/opus-mt-ur-en-tokenizer"
            self.translator = ctranslate2.Translator(self.model_path, device="cpu", compute_type="int8")
        else:  # x86_64
            self.translator = ctranslate2.Translator(self.model_path, device="cpu")
        
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    def process_input(self, text, lang):
        """Translates a string of text to the target language."""
        if self.first_load:
            self.load() 
            self.first_load = False

        cleaned_text = self._clean_text_for_translation(text)
        
        source_tokens = self.tokenizer.convert_ids_to_tokens(self.tokenizer.encode(cleaned_text))

        if self.version == 'arm':
            # QuickMT (OPUS-MT based) model processing, no language prefix needed
            results = self.translator.translate_batch([source_tokens])
            target_tokens = results[0].hypotheses[0]
            return self.tokenizer.decode(self.tokenizer.convert_tokens_to_ids(target_tokens))
        
        else:  # x86_64
            # NLLB model processing, requires a language prefix
            if lang == 'en': 
                target_prefix = ["urd_Arab"] 
            elif lang == 'ur': 
                target_prefix = ["eng_Latn"]
            else:
                target_prefix = None
                
            if target_prefix:
                results = self.translator.translate_batch([source_tokens], target_prefix=[target_prefix])
                # NLLB includes the target prefix in the output, so we skip it
                target_tokens = results[0].hypotheses[0][1:]
                return self.tokenizer.decode(self.tokenizer.convert_tokens_to_ids(target_tokens))
            return cleaned_text


    def _clean_text_for_translation(self, text):
        """Clean text by removing emojis and other characters that cause translation issues."""
        emoji_pattern_str = get_emoji_regex_pattern()
        if emoji_pattern_str:
            emoji_pattern = re.compile(emoji_pattern_str, flags=re.UNICODE)
            cleaned_text = emoji_pattern.sub('', text)
        else:
            cleaned_text = text
        
        return cleaned_text

    def convert_to_urdu_num(self, text):
        """Convert english numbers (0-9) to urdu numbers (۰-۹) for display."""
        if not text:
            return text
        num_map = get_urdu_numbers()
        for english, urdu in num_map.items():
            text = text.replace(english, urdu)
        return text


if __name__ == '__main__':
    from model_registry import ModelRegistry
    model_reg = ModelRegistry()

    print("--- Testing x86_64 version ---")
    
    # For x86_64, the same model is used for both directions
    model_path_x86 = model_reg.get_model_path('translation', 'nllb-200-600M-Q8')
    
    # English to Urdu
    translator_en_ur_x86 = Translator(version='x86_64', model_path=model_path_x86)
    en_text = "Hello, how are you?"
    translated_text_ur_x86 = translator_en_ur_x86.process_input(en_text, lang='en')
    print(f"English to Urdu (x86_64): '{en_text}' -> '{translated_text_ur_x86}'")
    
    # Urdu to English
    translator_ur_en_x86 = Translator(version='x86_64', model_path=model_path_x86)
    ur_text = "ہیلو، آپ کیسے ہیں؟"
    translated_text_en_x86 = translator_ur_en_x86.process_input(ur_text, lang='ur')
    print(f"Urdu to English (x86_64): '{ur_text}' -> '{translated_text_en_x86}'")
    print("-" * 20)

    print("\n--- Testing arm version ---")
    
    # English to Urdu
    model_path_en_ur_arm = model_reg.get_model_path('translation', 'quickmt-en-ur-Q8')
    translator_en_ur_arm = Translator(version='arm', model_path=model_path_en_ur_arm, source_lang='en', target_lang='ur')
    translated_text_ur_arm = translator_en_ur_arm.process_input(en_text, lang='en')
    print(f"English to Urdu (arm): '{en_text}' -> '{translated_text_ur_arm}'")

    # Urdu to English
    model_path_ur_en_arm = model_reg.get_model_path('translation', 'quickmt-ur-en-Q8')
    translator_ur_en_arm = Translator(version='arm', model_path=model_path_ur_en_arm, source_lang='ur', target_lang='en')
    translated_text_en_arm = translator_ur_en_arm.process_input(ur_text, lang='ur')
    print(f"Urdu to English (arm): '{ur_text}' -> '{translated_text_en_arm}'")
    print("-" * 20)