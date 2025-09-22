"""
test_llm.py

This file tests the LLM models.

The code to import modules from absolute paths is referenced from https://www.geeksforgeeks.org/python/python-import-from-parent-directory/
"""
import sys
import os
import gc
import resource
import time
import pandas as pd
from llama_cpp import Llama

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

class SimpleLLM:    
    def __init__(self, model_path, n_ctx=8192, n_batch=1024):
        self.model_path = model_path
        self.llm = None
        self.chat_history = []  # cli uses simple list to store history gui stores in db
        self.first_load = True
        self.system_prompt = "You are a patient and encouraging tutor to a grade 7 student. Use language and explanations appropriate for a grade 7 student. Always ask a follow-up question to check understanding."
        self.n_ctx = n_ctx
        self.n_batch = n_batch
        self.load()


    def load(self):
        
        self.llm = Llama(
            model_path=self.model_path,
            n_ctx=self.n_ctx,
            n_threads=16, # raspberry pi 5 has 4 cores
            n_gpu_layers=0, # disable gpu to avoid issues on pi
            n_batch=self.n_batch, # set a smaller batch size to avoid issues on pi
            seed=1, # set a seed to get consistent results
            # use_mlock=False, # disable mlock to avoid issues on pi
            # verbose=False, # disable ve/rbose output
            verbose=True, # enable verbose output
        )

    def process_input(self, text):
        """process input and return complete response for CLI
           build chat messages with history list"""
        messages = self.chat_history.copy()
        if not messages:
            # add the system prompt to the first user message with default grade and subject
            first_use = (
                f"{self.system_prompt}\n\n"
                f"User input:\n{text}"
            )

            messages.append({"role": "user", "content": first_use})
        else:
            messages.append({"role": "user", "content": text})

        start_time = time.time()
        stream = self.llm.create_chat_completion(
            messages=messages,
            temperature=0.6,
            top_p=0.9,
            stream=True,
        )
        

        answer_parts = []
        role_seen = None

        for chunk in stream:
            choice = chunk["choices"][0]
            delta = choice.get("delta")
            # role appears once at the start skip it
            # only print content
            if "role" in delta and role_seen is None:
                role_seen = delta["role"]
                continue

            piece = delta.get("content")
            if piece:
                print(piece, end="", flush=True)
                answer_parts.append(piece)

        print(f"\n\nTime taken: {time.time()-start_time:.2f}s")
        answer = "".join(answer_parts).strip()

        # save to history so short follow ups like yes/no work
        self.chat_history.append({"role": "user", "content": text})
        self.chat_history.append({"role": "assistant", "content": answer})
        print()
        return answer



if __name__ == "__main__":
    results = []
    simple_llm = None

    model_reg = ModelRegistry()
    # model_name = "gemma-2-2b"
    # model_name = "llama-3.2-3b-Q4"
    # model_name = "gemma-3-4b-Q4"
    # model_name = "qwen-3-4b-Q5"
    # model_name = "gemma-3-1b-Q4O"
    # model_name = "deepseek-qwen2.5-1.5b-Q4"
    # model_name = "gemma-3-1b-Q4"
    # model_name = "gemma-3-1b-Q40"
    # model_name = "gemma-3-1b-Q6"
    # model_name = "phi-4-mini"
    # model_name = "phi-3.1-mini"
    # model_name = "phi-3-mini"
    model_name = "qwen-2.5-3b-Q4"
    model_type = "llm"
    model_path = model_reg.get_model_path(model_type, model_name)


    prompt = "How do plants make their own food?"

    n_ctx = 2048
    n_batch = 512
    simple_llm = SimpleLLM(model_path, n_ctx, n_batch)
    print(f"\n\nSystem prompt: {simple_llm.system_prompt}")
    print("\nModel Output:\n\n")


    # ensure garbage is collected before measuring memory
    gc.collect()
    
    start_time = time.time()
    response = simple_llm.process_input(prompt)
    print(f"\n\nResponse: {response}")
    time_taken = time.time() - start_time
    
    mem_used = proc_mem()
    prompt = prompt.replace("\n", "")
    response = response.replace("\n", "")
    results.append({
        "model type": model_type,
        "model name": model_name,
        "context size": n_ctx,
        "batch size": n_batch,
        "time taken": f"{time_taken:.2f}",
        "Memory used (MiB)": f"{mem_used:.2f}",
        "input": prompt,
        "output": response.strip()
    })



    df = pd.DataFrame(results)
    # write header only if file does not exist
    file_exists = os.path.isfile("model_tests/test_outputs/llm_stats.csv")
    df.to_csv("model_tests/test_outputs/llm_stats.csv", mode='a', header=not file_exists, index=False)
    print("\ndata saved to csv")




