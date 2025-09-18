'''
LLM engine that works for both cli and gui.
CLI was created first to test implementation and orchestration and then gui was created to make it more user friendly.
CLI is used for testing the models and gui is used for the final product.

The parts of code not written by me are referenced from the following sources:
- Took help for streaming llm output from https://github.com/abetlen/llama-cpp-python/discussions/319
- Streaming LLM code was modified from the gradio chatbot docs at https://www.gradio.app/guides/creating-a-custom-chatbot-with-blocks

'''

import time
import re
from llama_cpp import Llama
import torch
import gc
from config_loader import get_system_prompt

class LLMEngine:
    def __init__(self, model_path):
        self.llm = None
        self.model_path = model_path
        self.first_load = True
        # system prompt for the tutor assistant loaded from config
        self.system_prompt_template = get_system_prompt()
        # default grade and subject values
        self.grade = 5
        self.subject = "General"
        self.chat_history = []  # cli uses simple list to store history gui stores in db
        self.stop_generation = False
        self.load()

    def stop_stream(self):
        """Stops the current LLM generation stream by setting a flag."""
        self.stop_generation = True
            
    def switch_model(self, new_model_path):
        """Unload the current model and load a new one."""
        # if the model is already loaded, return
        if self.model_path == new_model_path and self.llm is not None:
            return  

        # unload the current model
        if self.llm is not None:
            del self.llm
            self.llm = None
            gc.collect()
        
        self.model_path = new_model_path
        self.load()

    def load(self):
        """Load the model with optimized configuration for raspberry pi"""
        torch.set_num_threads(4)
        
        self.llm = Llama(
            model_path=self.model_path,
            n_ctx=8192,
            n_threads=4, # raspberry pi 5 has 4 cores
            n_gpu_layers=0, # disable gpu to avoid issues on pi
            n_batch=1024, # set a smaller batch size to avoid issues on pi
            seed=1, # set a seed to get consistent results
            use_mlock=False, # disable mlock to avoid issues on pi
            verbose=False, # disable verbose output
            # verbose=True, # enable verbose output
        )

    def format_system_prompt(self, grade, subject):
        """Format the system prompt with grade and subject information.
           Deepseek R1 Qwen 1.5B model for math and reasoning works best with direct and simple prompts.
        """
        if subject == "Math":
            qwen_prompt = f"Use appropriate vocabulary for a grade {grade} student. Ask one simple follow-up question to check understanding. Explain: "
            return qwen_prompt
        # for other subjects, we use the system prompt template from config
        return self.system_prompt_template.format(grade=grade, subject=subject)

    def build_messages(self, db_messages, grade, subject):
        """
        Convert DB rows into llama.cpp chat messages.
        This is used for the gradio gui.
        If the first message is from the user, it adds the system prompt 
        into that first user message. gemma-2 and qwen-2.5-1.5b LLM use the prompt role 
        format [user][assistant]/[user][assistant]/.../...
        so we need to add the system prompt to the first user message.
        if the grade or subject changes, we add the system prompt to the first user message.
        if the conversation is trimmed, we add the system prompt to the first user message.
        """
        # if no messages, return empty list
        if not db_messages:
            return []

        # create a clean copy of messages removing any html used for display like in Urdu rtl
        msgs = []
        for m in db_messages:
            clean_content = re.sub(r'<[^>]+>', '', m["content"])
            msgs.append({"role": m["role"], "content": clean_content})
        
        # trim after every 10 messages to avoid context window issues
        original_length = len(msgs)
        is_trimmed = False
        if original_length > 12:
            msgs = msgs[-10:] 
            is_trimmed = True

        # check if we need to prepend the system prompt
        is_new_chat = (original_length == 1)
        grade_or_subject_changed = (grade != self.grade or subject != self.subject)
        
        prepend_prompt = is_new_chat or grade_or_subject_changed or is_trimmed
        
        if prepend_prompt:
            self.grade = grade
            self.subject = subject
            
            # ensure last message is from the user
            if msgs and msgs[-1]["role"] == "user":
                user_message = msgs[-1]
                original_content = user_message["content"]
                prompt_to_use = self.format_system_prompt(grade, subject)
                user_message["content"] = f"{prompt_to_use}\n\n{original_content}"

        return msgs

    def stream_reply(self, messages):
        """stream llm response token by token for gradio gui"""
        self.stop_generation = False  # reset stop flag
        start_time = time.time()
        # Start a streaming request to the language model with the chat history.
        # stream=True tells the model to send back the response piece by piece.
        stream = self.llm.create_chat_completion(
            messages=messages,
            temperature=0.6,
            top_p=0.9,
            stream=True,
        )

        # loop through each chunk of data sent from the model
        for chunk in stream:
            if self.stop_generation:
                print("\nLLM generation stopped by user.")
                break  # exit the loop if stop is requested

            # each chunk contains one choice, which is a dict with a delta key
            choice = chunk["choices"][0]
            # get the delta, which is the new part of the response in this chunk
            delta = choice.get("delta") or {}
            # extract the actual text content from the delta
            part = delta.get("content")
            # if there is content, yield sends this part
            # back to the ui without ending the function so it can
            # continue receiving the next parts of the stream
            if part:
                yield part
        # print the total time taken to generate the response
        if not self.stop_generation:
            print(f"\n\nGenerated in {time.time()-start_time:.2f}s")
        self.stop_generation = False # ensure flag is reset

    def process_input(self, text):
        """process input and return complete response for CLI
           build chat messages with history list"""
        messages = self.chat_history.copy()
        if not messages:
            # add the system prompt to the first user message with default grade and subject
            default_prompt = self.format_system_prompt(grade="5", subject="general")

            first_use = (
                f"{default_prompt}\n\n"
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
            delta = choice.get("delta") or {}
            # role appears once at the start skip it
            # only print content
            if "role" in delta and role_seen is None:
                role_seen = delta["role"]
                continue

            piece = delta.get("content")
            if piece:
                print(piece, end="", flush=True)
                answer_parts.append(piece)

        print(f"\n\nGenerated in {time.time()-start_time:.2f}s")
        answer = "".join(answer_parts).strip()

        # save to history so short follow ups like yes/no work
        self.chat_history.append({"role": "user", "content": text})
        self.chat_history.append({"role": "assistant", "content": answer})
        print()
        return answer
