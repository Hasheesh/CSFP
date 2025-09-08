'''
took help for streaming llm output from https://github.com/abetlen/llama-cpp-python/discussions/319

'''

import os
import time
from llama_cpp import Llama
import torch

class LLM:
    def __init__(self, model_path):

        self.llm = None
        self.model_path = model_path
        self.first_load = True
        self.system_prompt = (
            "You are a patient tutor assistant. Explain concepts clearly for 12-year-olds. "
            "Use simple, direct sentences. Avoid complex metaphors, idioms, and emojis. "
            "Ask back questions as well, be a stohastic tutor. "
        )
        self.chat_history = []
        self.load() 

        
    def load(self):
        # Hardware optimization
        # os.environ["OMP_NUM_THREADS"] = "16"
        # os.environ["GGML_OPENBLAS"] = "1"
        torch.set_num_threads(16)

        model_path = self.model_path
        # Initialize model with proper configuration
        self.llm = Llama(
            model_path=model_path,
            n_ctx=2048,
            n_threads=16,  # Match Pi 5's 4 cores
            n_gpu_layers=0,
            seed=42,
            use_mlock=True,  # Prevent swapping
            # low_vram=True,    # Reduce memory overhead
            verbose=False,
            # n_batch=128,      # Reduced for Pi stability
            main_gpu=0        # Explicitly set to CPU
            
        )

    def process_input(self, text):
        # Build chat messages with history
        
        messages = self.chat_history.copy()
        if not messages:
            # First turn: fold the system guidance into the user's content
            first_user = (
                f"{self.system_prompt}\n\n"
                f"User question:\n{text}"
            )
            messages.append({"role": "user", "content": first_user})
        else:
            messages.append({"role": "user", "content": text})


        start_time = time.time()
        # stream = self.llm.create_chat_completion(
        #     messages=messages,
        #     temperature=0.7,
        #     top_p=0.9,
        #     max_tokens=256,
        #     stream=True,
        # )
        stream = self.llm.create_chat_completion(
            messages=messages,
            temperature=0.6,
            top_p=0.9,
            # max_tokens=256
            stream=True,
        )
       
        answer_parts = []
        role_seen = None

        for chunk in stream:
            choice = chunk["choices"][0]
            delta = choice.get("delta", {}) or {}
            # print(chunk)
            # role appears once at the start; don't print it
            if "role" in delta and role_seen is None:
                role_seen = delta["role"]
                continue

            piece = delta.get("content")
            if piece:
                print(piece, end="", flush=True)
                answer_parts.append(piece)

        print(f"\n\nGenerated in {time.time()-start_time:.2f}s")
        answer = "".join(answer_parts).strip()

    # save to history so short follow-ups like "yes" work
        self.chat_history.append({"role": "user", "content": text})
        self.chat_history.append({"role": "assistant", "content": answer})
        print()
        return answer

        


# from model_registery import ModelRegistery
# model_reg = ModelRegistery()
# llm_path = model_reg.get_model_path('llm', 'qwen-2.5-1.5b-math')

# llm_tutor = LLM(llm_path)
# llm_tutor.process_input("what is two plus two")

# while True:
#     text = input("You: ")
#     llm_tutor.process_input(text)
    