import os
import time
from llama_cpp import Llama
import torch

class LLM:
    def __init__(self, model_path="/home/gulabo/Desktop/CSFP/models/llm/phi-4-mini/Phi-4-mini.gguf"):
        # Hardware optimization
        os.environ["OMP_NUM_THREADS"] = "4"
        os.environ["GGML_OPENBLAS"] = "1"
        torch.set_num_threads(4)

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

    def tutor(self, question: str) -> str:
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
        for chunk in stream:
            token = chunk["choices"][0]["text"]
            print(token, end="", flush=True)
            answer += token
        print(f"\n\nGenerated in {time.time()-start_time:.2f}s")
        return answer.strip()

    def interactive(self):
        print("Offline AI Tutor - 'q' to quit")
        while True:
            try:
                question = input("\nQuestion: ").strip()
                if question.lower() == 'q':
                    break
                if question:
                    full_answer = self.tutor(question)
                    print("-" * 40)
            except KeyboardInterrupt:
                print("\nExiting...")
                break
            except Exception as e:
                print(f"\nError: {str(e)}. Resetting...")


# main.py
from models import get_model_path
from llm_engine import LLM

class AITutor:
    def __init__(self, config):
        self.engines = {}
        self.load_engines(config)
        
    def load_engines(self, config):
        # Initialize all engines
        self.engines["llm"] = LLM(get_model_path("llm", config["llm_model"]))
        
    def process_input(self, input_data, input_type):
        if input_type == "audio":
            text = self.engines["stt"].process(input_data)
        elif input_type == "image":
            text = self.engines["ocr"].process(input_data)
        else:  # text
            text = input_data
            
        # Get LLM response
        response = self.engines["llm"].interactive(text)
            
        return response

# Example usage
if __name__ == "__main__":
    config = {
        "llm_model": "phi4-mini",
    }
    

