"""
translator.py

This file handles the text translation functionality for the AI tutor.
It uses the nllb-200-distilled-600M model to translate text between English and Urdu.

The parts of code not written by me are referenced from the following sources:
- Code to use nllb-200-distilled-600M model from https://opennmt.net/CTranslate2/guides/transformers.html#nllb
"""
from transformers import AutoTokenizer
import ctranslate2
import re
from config_loader import get_emoji_regex_pattern, get_urdu_numbers


class Translator:
    """Translates text between English and Urdu using a CTranslate2 model."""

    def __init__(self, model_path):
        """Initializes the Translator with the model path and configuration."""
        self.translator = None
        self.model_path = model_path
        self.first_load = True
        self.model = None
        self.tokenizer = None
        
    def load(self):
        """Loads the translation model and tokenizer into memory."""
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.translator = ctranslate2.Translator(self.model_path, device="cpu")

    def process_input(self, text, lang):
        """Translates a string of text to the target language."""
        # load the model only once when processing the first time
        if self.first_load:
            self.load() 
            self.first_load = not self.first_load

        # clean text by removing emojis and other problematic characters
        cleaned_text = self._clean_text_for_translation(text)
        
        # tokenize the input text for the model
        source_tokens = self.tokenizer.convert_ids_to_tokens(self.tokenizer.encode(cleaned_text))
        
        # set the target language prefix required by the nllb model
        if lang == 'en': 
            target_prefix = ["urd_Arab"] 
        elif lang == 'ur': 
            target_prefix = ["eng_Latn"]
            
        # translate the tokens using ctranslate2
        results = self.translator.translate_batch([source_tokens], target_prefix=[target_prefix])
        
        # decode the translated tokens back to a string
        target = results[0].hypotheses[0][1:]
        return self.tokenizer.decode(self.tokenizer.convert_tokens_to_ids(target))

    def _clean_text_for_translation(self, text):
        """Clean text by removing emojis and other characters that cause translation issues."""
        
        # get emoji pattern from config
        emoji_pattern_str = get_emoji_regex_pattern()
        if emoji_pattern_str:
            emoji_pattern = re.compile(emoji_pattern_str, flags=re.UNICODE)
            # remove emojis
            cleaned_text = emoji_pattern.sub('', text)
        else:
            cleaned_text = text
        
        return cleaned_text

    def convert_to_urdu_num(self, text):
        """Convert english numbers (0-9) to urdu numbers (۰-۹) for display."""
        if not text:
            return text
            
        # get number mapping from config
        num_map = get_urdu_numbers()
        
        # replace each english number with its urdu equivalent
        for english, urdu in num_map.items():
            text = text.replace(english, urdu)
        
        return text 
