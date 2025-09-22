"""
test_stt.py

This file tests the Speech-to-Text models.

The code to import modules from absolute paths is referenced from https://www.geeksforgeeks.org/python/python-import-from-parent-directory/
"""
import sys
import os
import gc
import resource
import time
import pandas as pd
import soundfile as sf
import librosa
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from faster_whisper import WhisperModel

# Add parent directory to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model_registry import ModelRegistry

def proc_mem():
    """Gets and prints the max memory usage of the process."""
    # ensure garbage is collected before measuring memory
    gc.collect()
    # for linux
    max_rss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    max_rss_mb = max_rss_kb / 1024
    print(f"Max memory used: {max_rss_mb:.1f} MiB")
    return max_rss_mb

class WhisperSTT:
    def __init__(self, model_path):
        self.processor = WhisperProcessor.from_pretrained(model_path)
        self.model = WhisperForConditionalGeneration.from_pretrained(model_path)

    def process_input(self, wav_path, language):
        audio, sr = sf.read(wav_path)
        if sr != self.processor.feature_extractor.sampling_rate:
            audio = librosa.resample(audio, orig_sr=sr, 
                                     target_sr=self.processor.feature_extractor.sampling_rate)
            sr = self.processor.feature_extractor.sampling_rate

        inputs = self.processor.feature_extractor(audio, sampling_rate=sr, 
                                             return_tensors="pt")
        input_features = inputs.input_features.to(self.model.device)

        generated_ids = self.model.generate(
            input_features, 
            max_length=2000,
            num_beams=5,
            task="transcribe",
            language=language,
            early_stopping=True
        )
        
        transcription = self.processor.decode(generated_ids[0], skip_special_tokens=True)
        return transcription

class FasterWhisperSTT:
    def __init__(self, model_path):
        self.model = WhisperModel(
            model_path,
            device="cpu",
            compute_type="int8",
            cpu_threads=16
        )

    def process_input(self, audio_path, lang):
        segments, info = self.model.transcribe(
            audio_path,
            language=lang,
            task="transcribe",
            beam_size=1,
            vad_filter=True
        )
        text_chunks = [seg.text.strip() for seg in segments]
        transcription = " ".join(t for t in text_chunks if t)
        return transcription

if __name__ == "__main__":
    results = []
    
    model_reg = ModelRegistry()
    model_type = "stt"
    
#     # model_name = "whisper-small"
    # model_path = model_reg.get_model_path(model_type, model_name)
    # stt_model = WhisperSTT(model_path)
# 

    model_name = "faster-whisper-small"
    model_path = model_reg.get_model_path(model_type, model_name)
    stt_model = FasterWhisperSTT(model_path)

    gc.collect()
    
    start_time = time.time()

    # response = stt_model.process_input("en_test.wav", "en")
    response = stt_model.process_input("ur_test.wav", "ur")


    time_taken = time.time() - start_time
    
    mem_used = proc_mem()
    
    print(f"\nTranscription: {response}")

    # actual_audio = "This is a test of speech recognition"
    actual_audio = "یہ تقریر کی شناخت کا امتحان ہے۔"

    results.append({
        "model type": model_type,
        "model name": model_name,
        "time taken": f"{time_taken:.2f}",
        "Memory used (MiB)": f"{mem_used:.2f}",
        "input": actual_audio,
        "output": response.strip(),
    })

    df = pd.DataFrame(results)
    file_exists = os.path.isfile("model_tests/test_outputs/stt_stats.csv")
    df.to_csv("model_tests/test_outputs/stt_stats.csv", mode='a', header=not file_exists, index=False)
    print("\ndata saved to csv")
