'''
LLM engine that works for both cli and gui.
CLI was created first to test implementation and orchestration and then gui was created to make it more user friendly.
CLI is used for testing the models and gui is used for the final product.

Took help for streaming llm output from https://github.com/abetlen/llama-cpp-python/discussions/319
'''

import time
from llama_cpp import Llama
import torch


class LLMEngine:
    def __init__(self, model_path):
        self.llm = None
        self.model_path = model_path
        self.first_load = True
        # system prompt for the tutor assistant
        self.system_prompt = (
            "You are a patient tutor assistant. Explain concepts clearly for 12-year-olds. "
            "Use simple, direct sentences. Avoid complex metaphors, idioms, and emojis. "
            "Ask back questions as well, be a stohastic tutor. "
            "Always respond in clear English only. Do not mix languages or use non-English words. "
        )
        self.chat_history = []  # cli uses simple list to store history
        self.load() 

    def load(self):
        # load the llm model with optimized configuration for raspberry pi
        torch.set_num_threads(4)

        # initialize model with custom configuration
        self.llm = Llama(
            model_path=self.model_path,
            n_ctx=4096,        
            n_threads=4, # raspberry pi 5 has 4 cores
            n_gpu_layers=0, # disable gpu layers to avoid issues on pi
            n_batch=1024, # set a smaller batch size to avoid issues on pi
            seed=1, # set a seed to get consistent results
            use_mlock=False, # disable mlock to avoid issues on pi
            verbose=False, # disable verbose output
        )

    def build_messages(self, db_messages):
        # convert db rows into llama.cpp chat messages
        # this is used for the gradio gui
        # if the first message is from the user, it adds the system prompt 
        # into that first user message. gemma-2 llm uses the prompt role 
        # format [user][assistant]/[user][assistant]/.../...
        if not db_messages:
            return []
        
        msgs = [{"role": m["role"], "content": m["content"]} for m in db_messages]
        
        # check for first msg role
        if msgs and msgs[0]["role"] == "user":
            first = msgs[0]["content"]
            msgs[0] = {
                "role": "user",
                "content": f"{self.system_prompt}\n\nStudent's question:\n{first}"
            }
        return msgs

    def stream_reply(self, messages):
        # stream llm response token by token for gradio gui
        stream = self.llm.create_chat_completion(
            messages=messages,
            temperature=0.6,
            top_p=0.9,
            stream=True,
        )
       
        for chunk in stream:
            choice = chunk["choices"][0]
            delta = choice.get("delta", {}) or {}
            part = delta.get("content")
            if part:
                yield part

    def process_input(self, text):
        # process input and return complete response for cli
        # build chat messages with history
        messages = self.chat_history.copy()
        if not messages:
            # first turn: fold the system guidance into the user's content
            first_use = (
                f"{self.system_prompt}\n\n"
                f"User question:\n{text}"
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
            delta = choice.get("delta", {}) or {}
            
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
