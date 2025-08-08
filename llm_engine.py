import os
import time
from llama_cpp import Llama
import torch

class LLM:
    def __init__(self, model_path):

        self.llm = None
        self.model_path = model_path
        self.first_load = True
        self.load() 

        
    def load(self):
        # Hardware optimization
        os.environ["OMP_NUM_THREADS"] = "16"
        # os.environ["GGML_OPENBLAS"] = "1"
        torch.set_num_threads(16)

        model_path = self.model_path
        # Initialize model with proper configuration
        self.llm = Llama(
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
        stream = self.llm.create_completion(
            prompt,
            max_tokens=256,
            temperature=0.7,
            top_p=0.9,
            stream=True,
            stop=["<|end|>"]
        )
        print(f"\nTutor: ")

        for chunk in stream:
            token = chunk["choices"][0]["text"]
            print(token, end="", flush=True)
            answer += token
        print(f"\n\nGenerated in {time.time()-start_time:.2f}s")
        return answer.strip()

# TODO
# def load_model(self, model_path: str) -> bool:
#     """Load a different model from the specified path."""
    
# def unload_model(self):
#     """Unload the current model to free memory."""
    
    
# def get_model_info(self) -> dict:
#     """Get information about the currently loaded model."""

# update_config()