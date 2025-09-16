import soundfile as sf
import numpy as np
import librosa
import psutil
import time
import os
import gc, ctypes
from faster_whisper import WhisperModel
    # Get virtual memory information
def proc_mem():
    p = psutil.Process(os.getpid())
    rss_mb = p.memory_info().rss / (1024**2)
    return f"Process RSS: {rss_mb:.1f} MiB"


class SpeechRecognizer:

    def __init__(self, model_path,
                 device: str = "cpu",
                 compute_type: str = "int8",   # "int8" or "int8_float16" are great on Pi; use "float32" if needed
                 num_threads: int = 16,
                 beam_size: int = 1,
                 vad_filter: bool = True):
        self.model_path = model_path
        self.device = device
        self.compute_type = compute_type
        self.num_threads = num_threads
        self.beam_size = beam_size
        self.vad_filter = vad_filter

        self.model = None
        self.first_load = True
        self.active_lang = None

    def load(self):

        self.model = WhisperModel(
            self.model_path,
            device=self.device,
            compute_type=self.compute_type,
            cpu_threads=self.num_threads
        )
        print(f"[faster-whisper] Successfully loaded {self.model_path} ({self.device}, {self.compute_type})")
        print("[after load]", proc_mem())

    


    def process_input(self, audio_path: str, lang: str):
        if self.first_load:
            self.load()
            self.first_load = False

        self.active_lang = lang


        # Transcribe
        # task="transcribe" keeps original language
        # language can be "ur", "en", etc.
        segments, info = self.model.transcribe(
            audio_path,
            language=lang,
            task="transcribe",
            beam_size=self.beam_size,
            vad_filter=self.vad_filter
        )

        text_chunks = [seg.text.strip() for seg in segments]
        transcription = " ".join(t for t in text_chunks if t)
        print(transcription)
        return transcription

    def unload(self):
        if self.model is None:
            return
        print(f"[faster-whisper unload] {self.active_lang} ...")
        del self.model
        self.model = None
        gc.collect()
        time.sleep(0.1)
        print("[after unload]", proc_mem())
        try:
            ctypes.CDLL("libc.so.6").malloc_trim(0)
        except Exception:
            pass

# if __name__ == "__main__":
#     print(proc_mem())
#     rec = SpeechRecognizer(
#         model_path="models/stt/faster-whisper-small",  # your local ct2 model dir
#         device="cpu",
#         compute_type="int8",     # try "int8_float16" or "float32" if you hit issues
#         num_threads=16,
#         beam_size=5,
#         vad_filter=True
#     )
#     out = rec.process_input("techno.wav", "ur")
#     print("TRANSCRIPT:", out)
#     print(proc_mem())  # model kept
#     # rec.unload()
#     print(proc_mem())
#     out = rec.process_input("question.wav", "en")
#     print("TRANSCRIPT:", out)
#     print(proc_mem())  # model kept
#     rec.unload()
#     print(proc_mem())
