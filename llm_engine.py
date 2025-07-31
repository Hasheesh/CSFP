import os
import time
from llama_cpp import Llama
import torch

class LLM:
    def __init__(self, model_path="/home/gulabo/Desktop/CSFP/models/llm/phi-4-mini/Phi-4-mini.gguf"):
        # Hardware optimization
        os.environ["OMP_NUM_THREADS"] = "16"
        # os.environ["GGML_OPENBLAS"] = "1"
        torch.set_num_threads(16)

        # Initialize model with proper configuration
        self.model = Llama(
            model_path=model_path,
            n_ctx=2048,
            n_threads=4,  # Match Pi 5's 4 cores
            n_gpu_layers=0,
            seed=42,
            use_mlock=True,  # Prevent swapping
            low_vram=True,    # Reduce memory overhead
            verbose=False,
            n_batch=128,      # Reduced for Pi stability
            main_gpu=0        # Explicitly set to CPU
        )

    def process_input(self, question: str) -> str:
        prompt = (
            "<|system|>You are a patient homework tutor. "
            "Explain concepts clearly for 12-year-olds.<|end|>\n"
            "<|user|>\n"
            f"{question}<|end|>\n"
            "<|assistant|>"
        )
        answer = ""
        start_time = time.time()
        stream = self.model.create_completion(
            prompt,
            max_tokens=256,
            temperature=0.7,
            top_p=0.9,
            stream=True,
            stop=["<|end|>"]
        )
        for chunk in stream:
            token = chunk["choices"][0]["text"]
            print(token, end="", flush=True)
            answer += token
        print(f"\n\nGenerated in {time.time()-start_time:.2f}s")
        return answer.strip()

