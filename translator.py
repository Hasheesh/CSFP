from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import psutil
import ctranslate2
import time
import gc
import os
import urduhack
from bidi.algorithm import get_display
import re

# Get virtual memory information
def proc_mem():
    p = psutil.Process(os.getpid())
    rss_mb = p.memory_info().rss / (1024**2)
    return f"Process RSS: {rss_mb:.1f} MiB"



class Translator:

    def __init__(self, model_path):
        self.translator = None
        self.model_path = model_path
        self.first_load = True
        self.model = None
        self.tokenizer = None
    def load(self):
        
        # Load model and tokenizer from local directory
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.translator = ctranslate2.Translator(self.model_path, device="cpu")

    # code for translation taken from ctranslate2 documentation at https://opennmt.net/CTranslate2/guides/transformers.html#nllb
    def process_input(self, text, lang):
        if self.first_load:
            self.load() 
            self.first_load = not self.first_load

        source_tokens = self.tokenizer.convert_ids_to_tokens(self.tokenizer.encode(text))
        if lang == 'en': target_prefix = ["urd_Arab"] 
        elif lang == 'ur': target_prefix = ["eng_Latn"]
        results = self.translator.translate_batch([source_tokens], target_prefix=[target_prefix])
        target = results[0].hypotheses[0][1:]
        return self.tokenizer.decode(self.tokenizer.convert_tokens_to_ids(target))

    def unload(self):
        if not self.first_load and not (self.tokenizer or self.translator):
            return

        print(f"[unload] ...")

        del self.translator
        self.translator = None
        del self.model 
        self.model = None
        del self.tokenizer
        self.tokenizer = None

        # run GC; give OS a tick to reclaim
        gc.collect()
        time.sleep(0.1)

        print("[after unload]", proc_mem())

    def convert_to_urdu_num(self, text: str) -> str:
        """Convert english numbers (0-9) to urdu numbers (۰-۹) for display."""
        if not text:
            return text
            
        # Mapping of english numbers to urdu numbers
        num_map = {
            '0': '۰', '1': '۱', '2': '۲', '3': '۳', '4': '۴',
            '5': '۵', '6': '۶', '7': '۷', '8': '۸', '9': '۹'
        }
        
        # Replace each english number with its urdu equivalent
        for english, urdu in num_map.items():
            text = text.replace(english, urdu)
        
        return text

    def fix_urdu_text_display(self, text: str) -> str:
        """Fix Urdu text for proper RTL display."""
        if not text:
            return text
            
        # Normalize Urdu text
        normalized_text = urduhack.normalization.normalize(text)
        # Fix RTL display
        display_text = get_display(normalized_text)
        return display_text
    

# tr = Translator("models/translation/nllb-200-distilled-600M-Q8")
# # ten.process_input("Hello...this is an Ai tutor's voice, Welcome!")

# def fix_urdu_script(text):
#     n_text = urduhack.normalization.normalize(text)
#     n_text = get_display(n_text) # fix RTL
#     return n_text


# with open("trans.txt", "w") as f:
#     f.write(tr.process_input("How do plants make their own food?", "en"))
#     # f.write(tr.process_input("What is the difference between a solids, liquids and gases?", 'en'))
#     f.close()


# print(proc_mem())  # stays higher (model kept)

# tr = Translator('models/translation/nllb-200-distilled-600M-Q8')
# print(tr.process_input("پودے اپنا کھانا کیسے تیار کرتے ہیں ؟", 'ur'))

# print(proc_mem()) 
