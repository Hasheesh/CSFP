"""
test_tts.py

This file tests the Text-to-Speech models.

The code to import modules from absolute paths is referenced from https://www.geeksforgeeks.org/python/python-import-from-parent-directory/
"""
import os
import sys
import gc
import resource
import time
import pandas as pd
import numpy as np
from piper import PiperVoice
from transformers import VitsModel, AutoTokenizer
import torch
import re
from scipy.io import wavfile

# Add parent directory to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model_registry import ModelRegistry

def proc_mem():
    """Gets and prints the max memory usage of the process."""
    # ensure garbage is collected before measuring memory
    gc.collect()
    max_rss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    max_rss_mb = max_rss_kb / 1024
    print(f"Max memory used: {max_rss_mb:.1f} MiB")
    return max_rss_mb

class PiperTTS:
    """A simple TTS class for Piper models."""
    def __init__(self, model_path):
        self.voice = PiperVoice.load(model_path)
        
       
    
    def process_input(self, text, filename):
        """Synthesize speech from text and save to WAV file."""
        audio_chunks = []
        for chunk in self.voice.synthesize(text):
            int_data = np.frombuffer(chunk.audio_int16_bytes, dtype=np.int16)
            audio_chunks.append(int_data)
        
        if audio_chunks:
            full_audio = np.concatenate(audio_chunks)
            
            output_filename = f"model_tests/test_outputs/tts/piper_tts_{filename}.wav"
            wavfile.write(output_filename, self.voice.config.sample_rate, full_audio)
            print(f"Audio saved to {output_filename}")
            
            return output_filename
        return None

class MMSVitsTTS:
    """A simple TTS class for MMS VITS models."""
    def __init__(self, model_path):
        self.model = VitsModel.from_pretrained(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        
    
    def process_input(self, text, filename):
        """Synthesize speech from text and save to WAV file."""
        inputs = self.tokenizer(text, return_tensors="pt")
        
        with torch.no_grad():
            output = self.model(**inputs).waveform
        
        output = output.cpu()
        
        data_np = output.numpy()
        data_np_squeezed = np.squeeze(data_np)
        
        output_filename = f"model_tests/test_outputs/tts/mms_tts_{filename}.wav"
        wavfile.write(output_filename, rate=self.model.config.sampling_rate, data=data_np_squeezed)
        print(f"Audio saved to {output_filename}")
        return output_filename

if __name__ == "__main__":
    results = []
    
    en_text = "Hello, how are you today?"
    en_text_n = "The temperature is 25.5 degrees Celsius."
    en_text_l = "The equation is 2 + 2 = 4 and π is approximately 3.14. Today is a beautiful day."
    en_text_p = "This is a paragraph text to test how well the TTS system handles extended speech synthesis. \n\nIt includes multiple sentences and should demonstrate the quality of the voice synthesis."
    
    ur_text = "ہیلو، آپ کیسے ہیں؟"
    ur_text_n = "جواب ۴۲ ہے"
    ur_text_l = "یہ ایک طویل متن ہے۔ اس کے دو جملے ہیں۔"
    ur_text_p = "یہ ایک طویل متن ہے جو ٹی ٹی ایس سسٹم کی طویل تقریر کی ترکیب کو کس طرح سنبھالتا ہے اس کا امتحان لینے کے لیے ہے۔\n\n اس میں متعدد جملے شامل ہیں اور آواز کی ترکیب کا معیار ظاہر کرنا چاہیے۔"
    
    model_reg = ModelRegistry()
    model_type = "tts"
    
    # # ------------------------------------ Piper TTS English
    # model_name = "piper-tts-en-amy"
    # model_path = model_reg.get_model_path(model_type, model_name)
    # piper_amy = PiperTTS(model_path)
    
    # gc.collect()

    # start_time = time.time()
    # output_file = piper_amy.process_input(en_text, "en_text")
    # time_taken = time.time() - start_time
    # mem_used = proc_mem()
    # results.append({
    #     "model type": model_type,
    #     "model name": model_name,
    #     "time taken": f"{time_taken:.2f}",
    #     "Memory used (MiB)": f"{mem_used:.2f}",
    #     "input": en_text,    
    #     "output": output_file,
    # })
    
    # start_time = time.time()
    # output_file = piper_amy.process_input(en_text_n, "en_text_n")
    # time_taken = time.time() - start_time
    # mem_used = proc_mem() 
    # results.append({
    #     "model type": model_type,
    #     "model name": f"{model_name}",
    #     "time taken": f"{time_taken:.2f}",
    #     "Memory used (MiB)": f"{mem_used:.2f}",
    #     "input": en_text_n,
    #     "output": output_file,
    # })
    
    # start_time = time.time()
    # output_file = piper_amy.process_input(en_text_l, "en_text_l")
    # time_taken = time.time() - start_time   
    # mem_used = proc_mem()
    # results.append({
    #     "model type": model_type,
    #     "model name": f"{model_name}",
    #     "time taken": f"{time_taken:.2f}",
    #     "Memory used (MiB)": f"{mem_used:.2f}",
    #     "input": en_text_l,
    #     "output": output_file,
    # })
    
    # start_time = time.time()
    # output_file = piper_amy.process_input(en_text_p, "en_text_p")
    # time_taken = time.time() - start_time
    # mem_used = proc_mem()
    # results.append({
    #     "model type": model_type,
    #     "model name": f"{model_name}",
    #     "time taken": f"{time_taken:.2f}",
    #     "Memory used (MiB)": f"{mem_used:.2f}",
    #     "input": en_text_p.replace("\n", " ").strip(),
    #     "output": output_file,
    # })


    # # ------------------------------------ MMS TTS English
    # model_name = "mms-tts-en"
    # model_path = model_reg.get_model_path(model_type, model_name)
    # mms_en = MMSVitsTTS(model_path)
    
    # gc.collect()
    # start_time = time.time()
    # output_file = mms_en.process_input(en_text, "en_text")
    # time_taken = time.time() - start_time
    # mem_used = proc_mem()
    # results.append({
    #     "model type": model_type,
    #     "model name": model_name,
    #     "time taken": f"{time_taken:.2f}",
    #     "Memory used (MiB)": f"{mem_used:.2f}",
    #     "input": en_text,
    #     "output": output_file,

    # })
    
    # start_time = time.time()
    # output_file = mms_en.process_input(en_text_n, "en_text_n")
    # time_taken = time.time() - start_time
    # mem_used = proc_mem()
    # results.append({
    #     "model type": model_type,
    #     "model name": f"{model_name}",
    #     "time taken": f"{time_taken:.2f}",
    #     "Memory used (MiB)": f"{mem_used:.2f}",
    #     "input": en_text_n,
    #     "output": output_file,

    # })
    
    # start_time = time.time()
    # output_file = mms_en.process_input(en_text_l, "en_text_l")
    # time_taken = time.time() - start_time
    # mem_used = proc_mem()
    # results.append({
    #     "model type": model_type,
    #     "model name": f"{model_name}",
    #     "time taken": f"{time_taken:.2f}",
    #     "Memory used (MiB)": f"{mem_used:.2f}",
    #     "input": en_text_l,
    #     "output": output_file,

    # })

    # start_time = time.time()
    # output_file = mms_en.process_input(en_text_p, "en_text_p")
    # time_taken = time.time() - start_time
    # mem_used = proc_mem()
    # results.append({
    #     "model type": model_type,
    #     "model name": f"{model_name}",
    #     "time taken": f"{time_taken:.2f}",
    #     "Memory used (MiB)": f"{mem_used:.2f}",
    #     "input": en_text_p.replace("\n", " ").strip(),
    #     "output": output_file,
    # })
    
    # --------------------------------- MS TTS Urdu
    model_name = "mms-tts-ur"
    model_path = model_reg.get_model_path(model_type, model_name)
    mms_ur = MMSVitsTTS(model_path)
    
    gc.collect()
    start_time = time.time()
    output_file = mms_ur.process_input(ur_text, "ur_text")
    time_taken = time.time() - start_time
    mem_used = proc_mem()
    results.append({
        "model type": model_type,
        "model name": model_name,
        "time taken": f"{time_taken:.2f}",
        "Memory used (MiB)": f"{mem_used:.2f}",
        "input": ur_text,
        "output": output_file,
    })
    
    start_time = time.time()
    output_file = mms_ur.process_input(ur_text_n, "ur_text_n")
    time_taken = time.time() - start_time
    mem_used = proc_mem()
    results.append({
        "model type": model_type,
        "model name": f"{model_name}",
        "time taken": f"{time_taken:.2f}",
        "Memory used (MiB)": f"{mem_used:.2f}",
        "input": ur_text_n,
        "output": output_file,
    })
    
    start_time = time.time()
    output_file = mms_ur.process_input(ur_text_l, "ur_text_l")
    time_taken = time.time() - start_time
    mem_used = proc_mem()
    results.append({
        "model type": model_type,
        "model name": f"{model_name}",
        "time taken": f"{time_taken:.2f}",
        "Memory used (MiB)": f"{mem_used:.2f}",
        "input": ur_text_l,
        "output": output_file,
    })

    start_time = time.time()
    output_file = mms_ur.process_input(ur_text_p, "ur_text_p")
    time_taken = time.time() - start_time
    mem_used = proc_mem()
    results.append({
        "model type": model_type,
        "model name": f"{model_name}",
        "time taken": f"{time_taken:.2f}",
        "Memory used (MiB)": f"{mem_used:.2f}",
        "input": ur_text_p.replace("\n", " ").strip(),
        "output": output_file,
    })
    





    df = pd.DataFrame(results)
    file_exists = os.path.isfile("model_tests/test_outputs/tts_stats.csv")
    df.to_csv("model_tests/test_outputs/tts_stats.csv", mode='a', header=not file_exists, index=False)
    print("\nData saved to tts_stats.csv")
    