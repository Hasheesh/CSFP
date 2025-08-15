from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import psutil
import time
import gc
import os
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
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_path)

        # Create translation pipeline
        self.translator = pipeline(
            "translation", 
            model=self.model, 
            tokenizer=self.tokenizer
        )

    def process_input(self, text):
        if self.first_load:
            self.load() 
            self.first_load = not self.first_load

        result = self.translator(text)
        print(result[0]['translation_text'])
        return result[0]['translation_text']

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
    

# print(proc_mem())  # before load
# ten = Translator("models/translation/opus-mt-en-ur")
# ten.process_input("Hello...this is an Ai tutor's voice, Welcome!")
# print(proc_mem())  # stays higher (model kept)

# tur = Translator('models/translation/opus-mt-ur-en')
# tur.process_input("یہ اردو متن کو آواز میں تبدیل کرنے کا ایک نمونہ ہے۔")

# print(proc_mem()) 
