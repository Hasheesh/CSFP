"""
test_translation.py

This file tests the translation models.

The code to import modules from absolute paths is referenced from https://www.geeksforgeeks.org/python/python-import-from-parent-directory/
"""
import os
import sys
import gc
import psutil
import ctranslate2
from transformers import AutoTokenizer, MarianMTModel
import time
import pandas as pd

# Add parent directory to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model_registry import ModelRegistry
from translator import Translator as CT2Translator


def proc_mem():
    """Gets and prints the max memory usage of the process."""
    # ensure garbage is collected before measuring memory
    gc.collect()
    # for linux
    process = psutil.Process()
    memory_info = process.memory_info()
    max_rss_mb = memory_info.rss / (1024 * 1024)
    print(f"Max memory used: {max_rss_mb:.1f} MiB")
    return max_rss_mb


class OpusCT2:
    """A simple translator class for Hugging Face Opus-MT models."""
    def __init__(self, model_path, tokenizer_path):
        self.translator = ctranslate2.Translator(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    
    def process_input(self, text, **kwargs):
        source = self.tokenizer.convert_ids_to_tokens(self.tokenizer.encode(text))
        results = self.translator.translate_batch([source])
        target = results[0].hypotheses[0]
        return self.tokenizer.decode(self.tokenizer.convert_tokens_to_ids(target))

class NLLBTranslator:
    """Implementation used for NLLB CTranslate2 translator in project but then the final model used for translation is quickmt."""
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
        self.translator = ctranslate2.Translator(
            self.model_path,
            device="cpu",
            # compute_type="int8",
            # inter_threads=4,
            # intra_threads=1,
        )

    def process_input(self, text, lang):
        """Translates a string of text to the target language."""
        # load the model only once when processing the first time
        if self.first_load:
            self.load() 
            self.first_load = not self.first_load

        # tokenize the input text for the model and dont waste time on special tokens
        source_tokens = self.tokenizer.convert_ids_to_tokens(self.tokenizer.encode(text))
        # set the target language prefix required by the nllb model
        if lang == 'en': 
            target_prefix = ["urd_Arab"] 
        elif lang == 'ur': 
            target_prefix = ["eng_Latn"]
            
        # translate the tokens using ctranslate2
        results = self.translator.translate_batch(
            [source_tokens],
            target_prefix=[target_prefix],
        )
        
        # decode the translated tokens back to a string
        target = results[0].hypotheses[0][1:]
        return self.tokenizer.decode(self.tokenizer.convert_tokens_to_ids(target))


class M2M100C2Translator:
    """A wrapper for the project's M2M100 CTranslate2 translator for testing."""
    def __init__(self, model_path):
        self.tokenizer = AutoTokenizer.from_pretrained("models/translation/m2m100-418-tokenizer")
        self.translator = ctranslate2.Translator(model_path, device="cpu")
    
    def process_input(self, text, lang):
        self.tokenizer.src_lang = "en" if lang == 'en' else "ur"
        target_lang = "ur" if lang == 'en' else "en"
        
        source_tokens = self.tokenizer.convert_ids_to_tokens(self.tokenizer.encode(text))
        target_prefix = [self.tokenizer.lang_code_to_token[target_lang]]

        results = self.translator.translate_batch([source_tokens], target_prefix=[target_prefix])
        target = results[0].hypotheses[0][1:]
        return self.tokenizer.decode(self.tokenizer.convert_tokens_to_ids(target))


class OpusMTTranslator:
    """A translator class for Hugging Face Opus-MT models using MarianMTModel."""
    def __init__(self, model_path):
        self.model = MarianMTModel.from_pretrained(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

    def process_input(self, text, **kwargs):
        batch = self.tokenizer([text], return_tensors="pt")
        generated_ids = self.model.generate(**batch)
        return self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]


if __name__ == "__main__":
    results = []
    
    test_text_en = ["how do plants make their food?",
                    "What is photosynthesis? Teach me the Pythagorean theorem with an example. How does the water cycle work?",
                    "What is photosynthesis? \n Teach me the Pythagorean theorem with an example.\n\n How does the water cycle work?"]


    test_text_ur = ["پودے اپنی خوراک کیسے بناتے ہیں",
                    "فوٹو سنتھیسس کیا ہے؟ مجھے ایک مثال کے ساتھ پائتھاگورین تھیوریم سکھائیں۔ پانی کا چکر کیسے کام کرتا ہے؟",
                    "فوٹو سنتھیسس کیا ہے؟ \n مجھے ایک مثال کے ساتھ پائتھاگورین تھیوریم سکھائیں۔ \n\n پانی کا چکر کیسے کام کرتا ہے؟"]

 

    model_reg = ModelRegistry()
    model_type = "translation"
    start_time = time.time()


    # QuickMT ur en
    model_name = "quickmt-ur-en"
    model_path = model_reg.get_model_path(model_type, model_name)
    tokenizer_path_ur_en = "models/translation/opus-mt-ur-en-tokenizer"
    quickmt_ur_en = OpusCT2(model_path, tokenizer_path_ur_en)
    start_time = time.time()
    for t in test_text_ur:
        response = quickmt_ur_en.process_input(t)
        t = t.replace("\n", "") 
        response = response.replace("\n", "")
        time_taken = time.time() - start_time
        mem_used = proc_mem()
        results.append({
            "model type": model_type,
            "model name": model_name,
            "time taken": f"{time_taken:.2f}",
            "Memory used (MiB)": f"{mem_used:.2f}",
            "input": t,
            "output": response.strip(),
        })

    # QuickMT en ur
    model_name = "quickmt-en-ur"
    model_path = model_reg.get_model_path(model_type, model_name)
    tokenizer_path_en_ur = "models/translation/opus-mt-en-ur-tokenizer"
    quickmt_en_ur = OpusCT2(model_path, tokenizer_path_en_ur)
    start_time = time.time()
    for t in test_text_en:
        response = quickmt_en_ur.process_input(t)
        t = t.replace("\n", "") 
        response = response.replace("\n", "")
        time_taken = time.time() - start_time
        mem_used = proc_mem()
        results.append({
            "model type": model_type,
            "model name": model_name,
            "time taken": f"{time_taken:.2f}",
            "Memory used (MiB)": f"{mem_used:.2f}",
            "input": t,
            "output": response.strip(),
        })


    # New Translator en ur (quantized)
    model_name = "quickmt-en-ur-Q8"
    model_path = model_reg.get_model_path(model_type, model_name)
    new_translator_en_ur = CT2Translator(version='arm', model_path=model_path, source_lang='en', target_lang='ur')
    start_time = time.time()
    for t in test_text_en:
        response = new_translator_en_ur.process_input(t, "en")
        t = t.replace("\n", "") 
        response = response.replace("\n", "")
        time_taken = time.time() - start_time
        mem_used = proc_mem()
        results.append({
            "model type": model_type,
            "model name": model_name,
            "time taken": f"{time_taken:.2f}",
            "Memory used (MiB)": f"{mem_used:.2f}",
            "input": t,
            "output": response.strip(),
        })

    # New Translator ur en (quantized)
    model_name = "quickmt-ur-en-Q8"
    model_path = model_reg.get_model_path(model_type, model_name)
    new_translator_ur_en = CT2Translator(version='arm', model_path=model_path, source_lang='ur', target_lang='en')
    start_time = time.time()
    for t in test_text_ur:
        response = new_translator_ur_en.process_input(t, "ur")
        t = t.replace("\n", "") 
        response = response.replace("\n", "")
        time_taken = time.time() - start_time
        mem_used = proc_mem()
        results.append({
            "model type": model_type,
            "model name": model_name,
            "time taken": f"{time_taken:.2f}",
            "Memory used (MiB)": f"{mem_used:.2f}",
            "input": t,
            "output": response.strip(),
        })

    # # NLLB 200 600M CTranslate2
    # model_name = "nllb-200-600M-ct2"
    # model_path = model_reg.get_model_path(model_type, model_name)
    # nllb200_ct2 = NLLBTranslator(model_path)
    # start_time = time.time()
    # for t in test_text_en:
    #     response = nllb200_ct2.process_input(t, "en")
    #     t = t.replace("\n", "") 
    #     response = response.replace("\n", "")
    #     time_taken = time.time() - start_time
    #     mem_used = proc_mem()
    #     results.append({
    #         "model type": model_type,
    #         "model name": model_name,
    #         "input": t,
    #         "output": response.strip(),
    #         "time taken": f"{time_taken:.2f}",
    #         "Memory used (MiB)": f"{mem_used:.2f}"
    #     })
    # start_time = time.time()
    # for t in test_text_ur:
    #     response = nllb200_ct2.process_input(t, "ur")
    #     t = t.replace("\n", "") 
    #     response = response.replace("\n", "")
    #     time_taken = time.time() - start_time
    #     mem_used = proc_mem()
    #     results.append({
    #         "model type": model_type,
    #         "model name": model_name,
    #         "input": t,
    #         "output": response.strip(),
    #         "time taken": f"{time_taken:.2f}",
    #         "Memory used (MiB)": f"{mem_used:.2f}"
    #     })


    # # NLLB 200 600M Q8 with ctranslate2
    # model_name = "nllb-200-600M-Q8"
    # model_path = model_reg.get_model_path(model_type, model_name)
    # nllb200_q8 = NLLBTranslator(model_path)
    # start_time = time.time()
    # for t in test_text_en:
    #     response = nllb200_q8.process_input(t, "en")
    #     t = t.replace("\n", "") 
    #     response = response.replace("\n", "")
    #     time_taken = time.time() - start_time
    #     mem_used = proc_mem()
    #     results.append({
    #         "model type": model_type,
    #         "model name": model_name,
    #         "time taken": f"{time_taken:.2f}",
    #         "Memory used (MiB)": f"{mem_used:.2f}",
    #         "input": t,
    #         "output": response.strip(),
    #     })
    # start_time = time.time()
    # for t in test_text_ur:
    #     response = nllb200_q8.process_input(t, "ur")
    #     t = t.replace("\n", "") 
    #     response = response.replace("\n", "")
    #     time_taken = time.time() - start_time
    #     mem_used = proc_mem()
    #     results.append({
    #         "model type": model_type,
    #         "model name": model_name,
    #         "time taken": f"{time_taken:.2f}",
    #         "Memory used (MiB)": f"{mem_used:.2f}",
    #         "input": t,
    #         "output": response.strip(),
    #     })


    # # M2M 100 418M CTranslate2
    # model_name = "m2m100-418M-ct2"
    # model_path = model_reg.get_model_path(model_type, model_name)
    # m2m100_ct2 = M2M100C2Translator(model_path)
    # start_time = time.time()
    # for t in test_text_en:
    #     response = m2m100_ct2.process_input(t, "en")
    #     t = t.replace("\n", "") 
    #     response = response.replace("\n", "")
    #     time_taken = time.time() - start_time
    #     mem_used = proc_mem()
    #     results.append({
    #         "model type": model_type,
    #         "model name": model_name,
    #         "time taken": f"{time_taken:.2f}",
    #         "Memory used (MiB)": f"{mem_used:.2f}",
    #         "input": t,
    #         "output": response.strip(),
    #     })
    # start_time = time.time()
    # for t in test_text_ur:
    #     response = m2m100_ct2.process_input(t, "ur")
    #     t = t.replace("\n", "") 
    #     response = response.replace("\n", "")
    #     time_taken = time.time() - start_time
    #     mem_used = proc_mem()
    #     results.append({
    #         "model type": model_type,
    #         "model name": model_name,
    #         "time taken": f"{time_taken:.2f}",
    #         "Memory used (MiB)": f"{mem_used:.2f}",
    #         "input": t,
    #         "output": response.strip(),
    #     })


    # # M2M 100 418M Q8 with ctranslate2
    # model_name = "m2m100-418M-Q8"
    # model_path = model_reg.get_model_path(model_type, model_name)
    # m2m100_q8 = M2M100C2Translator(model_path)
    # start_time = time.time()
    # for t in test_text_en:
    #     response = m2m100_q8.process_input(t, "en")
    #     t = t.replace("\n", "") 
    #     response = response.replace("\n", "")
    #     time_taken = time.time() - start_time
    #     mem_used = proc_mem()
    #     results.append({
    #         "model type": model_type,
    #         "model name": model_name,
    #         "time taken": f"{time_taken:.2f}",
    #         "Memory used (MiB)": f"{mem_used:.2f}",
    #         "input": t,   
    #         "output": response.strip(),
    #     })
    # start_time = time.time()
    # for t in test_text_ur:
    #     response = m2m100_q8.process_input(t, "ur")
    #     t = t.replace("\n", "") 
    #     response = response.replace("\n", "")
    #     time_taken = time.time() - start_time
    #     mem_used = proc_mem()
    #     results.append({
    #         "model type": model_type,
    #         "model name": model_name,
    #         "time taken": f"{time_taken:.2f}",
    #         "Memory used (MiB)": f"{mem_used:.2f}",
    #         "input": t,
    #         "output": response.strip(),
    #     })  


    # # Opus MT en ur
    # model_name = "opus-mt-en-ur"
    # model_path = model_reg.get_model_path(model_type, model_name)
    # opus_mt_en_ur = OpusMTTranslator(model_path)
    # start_time = time.time()
    # for t in test_text_en:
    #     response = opus_mt_en_ur.process_input(t)
    #     t = t.replace("\n", "") 
    #     response = response.replace("\n", "")
    #     time_taken = time.time() - start_time
    #     mem_used = proc_mem()
    #     results.append({
    #         "model type": model_type,   
    #         "model name": model_name,
    #         "time taken": f"{time_taken:.2f}",
    #         "Memory used (MiB)": f"{mem_used:.2f}",
    #         "input": t,
    #         "output": response.strip(),
    #     })


    # # Opus MT ur en
    # model_name = "opus-mt-ur-en"
    # model_path = model_reg.get_model_path(model_type, model_name)
    # opus_mt_ur_en = OpusMTTranslator(model_path)
    # start_time = time.time()
    # for t in test_text_ur:
    #     response = opus_mt_ur_en.process_input(t)
    #     t = t.replace("\n", "") 
    #     response = response.replace("\n", "")
    #     time_taken = time.time() - start_time
    #     mem_used = proc_mem()
    #     results.append({
    #         "model type": model_type,
    #         "model name": model_name,
    #         "time taken": f"{time_taken:.2f}",
    #         "Memory used (MiB)": f"{mem_used:.2f}",
    #         "input": t,
    #         "output": response.strip(),
    #     })


    # # Opus MT en ur ct2
    # model_name = "opus-mt-en-ur-ct2"
    # model_path = model_reg.get_model_path(model_type, model_name)
    # opus_mt_en_ur_ct2 = OpusCT2(model_path, model_path)
    # start_time = time.time()
    # for t in test_text_en:
    #     response = opus_mt_en_ur_ct2.process_input(t)
    #     t = t.replace("\n", "") 
    #     response = response.replace("\n", "")
    #     time_taken = time.time() - start_time
    #     mem_used = proc_mem()
    #     results.append({
    #         "model type": model_type,
    #         "model name": model_name,
    #         "time taken": f"{time_taken:.2f}",
    #         "Memory used (MiB)": f"{mem_used:.2f}",
    #         "input": t,
    #         "output": response.strip(),
    #     })

    # # Opus MT ur en ct2
    # model_name = "opus-mt-ur-en-ct2"
    # model_path = model_reg.get_model_path(model_type, model_name)
    # opus_mt_ur_en_ct2 = OpusCT2(model_path, model_path)
    # start_time = time.time()
    # for t in test_text_ur:
    #     response = opus_mt_ur_en_ct2.process_input(t)
    #     t = t.replace("\n", "") 
    #     response = response.replace("\n", "")
    #     time_taken = time.time() - start_time
    #     mem_used = proc_mem()
    #     results.append({
    #         "model type": model_type,
    #         "model name": model_name,
    #         "time taken": f"{time_taken:.2f}",
    #         "Memory used (MiB)": f"{mem_used:.2f}",   
    #         "input": t,
    #         "output": response.strip(),
    #     })


    # # Opus MT en ur Q8
    # model_name = "opus-mt-en-ur-Q8"
    # model_path = model_reg.get_model_path(model_type, model_name)
    # tokenizer_path_en_ur = "models/translation/opus-mt-en-ur-tokenizer"
    # opus_mt_en_ur_q8 = OpusCT2(model_path, tokenizer_path_en_ur)
    # start_time = time.time()
    # for t in test_text_en:
    #     response = opus_mt_en_ur_q8.process_input(t)
    #     t = t.replace("\n", "") 
    #     response = response.replace("\n", "")
    #     time_taken = time.time() - start_time
    #     mem_used = proc_mem()
    #     results.append({
    #         "model type": model_type,
    #         "model name": model_name,
    #         "time taken": f"{time_taken:.2f}",
    #         "Memory used (MiB)": f"{mem_used:.2f}",
    #         "input": t,
    #         "output": response.strip(),
    #     })

    # # Opus MT ur en Q8
    # model_name = "opus-mt-ur-en-Q8"
    # model_path = model_reg.get_model_path(model_type, model_name)
    # tokenizer_path_ur_en = "models/translation/opus-mt-ur-en-tokenizer"
    # opus_mt_ur_en_q8 = OpusCT2(model_path, tokenizer_path_ur_en)
    # start_time = time.time()
    # for t in test_text_ur:
    #     response = opus_mt_ur_en_q8.process_input(t)
    #     t = t.replace("\n", "") 
    #     response = response.replace("\n", "")
    #     time_taken = time.time() - start_time
    #     mem_used = proc_mem()
    #     results.append({
    #         "model type": model_type,
    #         "model name": model_name,
    #         "time taken": f"{time_taken:.2f}",
    #         "Memory used (MiB)": f"{mem_used:.2f}",
    #         "input": t,
    #         "output": response.strip(),   
    #     })

    df = pd.DataFrame(results)
    # write header only if file does not exist
    file_exists = os.path.isfile("model_tests/test_outputs/translator_stats.csv")
    df.to_csv("model_tests/test_outputs/translator_stats.csv", mode='a', header=not file_exists, index=False)
    print("\ndata saved to csv")
