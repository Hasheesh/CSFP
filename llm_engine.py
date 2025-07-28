import os
import time
from llama_cpp import Llama
import torch

# Hardware optimization
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["GGML_OPENBLAS"] = "1"
torch.set_num_threads(4)

# Initialize model with proper configuration
llm = Llama(
    model_path=r"C:\Users\hasheesh\Desktop\CSFP\models\llm\gemma-3\google_gemma-3n-E2B-it-Q4_K_M.gguf",
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

def llm_tutor(question: str) -> str:
    prompt = (
        "<|system|>You are a patient homework tutor. "
        "Explain concepts clearly for 12-year-olds.<|end|>\n"
        "<|user|>\n"
        f"{question}<|end|>\n"
        "<|assistant|>"
    )
    answer = ""
    start_time = time.time()
    
    # Use create_completion instead of direct call
    stream = llm.create_completion(
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

if __name__ == "__main__":
    print("Offline AI Tutor - 'q' to quit")
    while True:
        try:
            question = input("\nQuestion: ").strip()
            if question.lower() == 'q':
                break
            if question:
                full_answer = llm_tutor(question)
                print("-" * 40)
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"\nError: {str(e)}. Resetting...")