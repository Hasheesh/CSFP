"""orchestrator.py

This file is the main orchestrator for the ai tutor.
It handles the logic for the model calls and database management.

The parts of code not written by me are referenced from the following sources:
- Code to read memory usage from https://stackoverflow.com/questions/938733/total-memory-used-by-python-process
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
import warnings
import logging
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=UserWarning)
logging.getLogger("tensorflow").setLevel(logging.ERROR)
import gradio as gr
import resource
import gc
import time
from PIL import Image
from model_registry import ModelRegistry
from llm_engine import LLMEngine
from vision_processor import VisionProcessor
from speech_recognizer import SpeechRecognizer
from translator import Translator
from speech_synthesizer import SpeechSynthesizer
from config_loader import get_status_message, get_user_profile

def proc_mem():
    """Gets and prints the max memory usage of the process."""
    # ensure garbage is collected before measuring memory
    gc.collect()
    # for linux
    max_rss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    max_rss_mb = max_rss_kb / 1024
    print(f"Max memory used: {max_rss_mb:.1f} MiB")


class Orchestrator:
    """Main orchestrator that handles all AI model calls and database management."""
    
    def __init__(self, version):
        """Initializes all the AI models and engines."""
        self.model_reg = ModelRegistry()
        self.version = version
        self.llm = None
        self.speech_synth_en = None
        self.speech_synth_ur = None
        self.speech_rec = None
        self.translator_en = None
        self.translator_ur = None
        self.vis_pro_en = None
        self.vis_pro_ur = None
        self.current_subject = None
        self.current_llm_model_name = None
        self.load_models()

    def load_models(self):
        """Loads all AI models based on the current version."""
        print(f"Loading models for version: {self.version}")
        if self.version == "x86_64":
            self.general_llm_model = "gemma-2-2b"
            self.math_llm_model = "deepseek-qwen2.5-1.5b-Q4"

        else:
            self.general_llm_model = "gemma-3-1b-Q40"
            self.math_llm_model = "deepseek-qwen2.5-1.5b-Q40"

        # get user profile to find initial subject
        user_profile = get_user_profile()
        initial_subject = user_profile.get("subject")
        self.current_subject = initial_subject

        # determine which LLM to load initially
        initial_llm_model = self.math_llm_model if initial_subject == "Math" else self.general_llm_model
        self.current_llm_model_name = initial_llm_model
        llm_path = self.model_reg.get_model_path("llm", initial_llm_model)
        self.llm = LLMEngine(llm_path)

        # load text to speech models for both languages
        self.tts_en_model_name = 'piper-tts-en-amy'
        tts_en_path = self.model_reg.get_model_path('tts', self.tts_en_model_name)
        self.speech_synth_en = SpeechSynthesizer(tts_en_path)
        self.tts_ur_model_name = 'mms-tts-ur'
        tts_ur_path = self.model_reg.get_model_path('tts', self.tts_ur_model_name)
        self.speech_synth_ur = SpeechSynthesizer(tts_ur_path)
        
        # load speech to text model
        self.stt_model_name = 'faster-whisper-small'
        stt_path = self.model_reg.get_model_path('stt', self.stt_model_name)
        self.speech_rec = SpeechRecognizer(stt_path)
        
        # load translation models (using quantized versions)
        if self.version == "x86_64":
            self.translator_en_model_name = 'nllb-200-600M-Q8'
            self.translator_ur_model_name = 'nllb-200-600M-Q8'
            translator_ur_en_path = self.model_reg.get_model_path('translation', self.translator_en_model_name)
            self.translator_en = Translator(self.version, translator_ur_en_path)
            translator_en_ur_path = self.model_reg.get_model_path('translation', self.translator_ur_model_name)
            self.translator_ur = Translator(self.version, translator_en_ur_path)
        else:
            self.translator_en_model_name = 'quickmt-ur-en-Q8'
            self.translator_ur_model_name = 'quickmt-en-ur-Q8'
            translator_ur_en_path = self.model_reg.get_model_path('translation', self.translator_en_model_name)
            self.translator_en = Translator(self.version, translator_ur_en_path, source_lang='ur', target_lang='en')
            translator_en_ur_path = self.model_reg.get_model_path('translation', self.translator_ur_model_name)
            self.translator_ur = Translator(self.version, translator_en_ur_path, source_lang='en', target_lang='ur')


        # load paddleocr models for english 
        ocr_en_det_path = self.model_reg.get_model_path('ocr', 'PP-OCRv5_mobile_det')
        ocr_en_rec_path = self.model_reg.get_model_path('ocr', 'en_PP-OCRv5_mobile_rec')
        ocr_en_det_model_name = 'PP-OCRv5_mobile_det'
        ocr_en_rec_model_name = 'en_PP-OCRv5_mobile_rec'

        # load paddle ocr models for urdu
        ocr_ur_det_path = self.model_reg.get_model_path('ocr', 'PP-OCRv3_mobile_det')
        ocr_ur_rec_path = self.model_reg.get_model_path('ocr', 'arabic_PP-OCRv3_mobile_rec')
        ocr_ur_det_model_name = 'PP-OCRv3_mobile_det'
        ocr_ur_rec_model_name = 'arabic_PP-OCRv3_mobile_rec'

        if self.version == "arm":
            self.vis_pro_en = VisionProcessor(self.version)
            self.vis_pro_ur = VisionProcessor(self.version)
        else:  
            self.vis_pro_en = VisionProcessor(self.version, ocr_en_det_path, ocr_en_rec_path, ocr_en_det_model_name, ocr_en_rec_model_name)
            self.vis_pro_ur = VisionProcessor(self.version, ocr_ur_det_path, ocr_ur_rec_path, ocr_ur_det_model_name, ocr_ur_rec_model_name)


    def switch_version(self, new_version):
        """Switches the version and reloads all models."""
        if self.version == new_version:
            return

        self.version = new_version
        
        # Clear existing models to free memory
        self.llm = None
        self.speech_synth_en = None
        self.speech_synth_ur = None
        self.speech_rec = None
        self.translator_en = None
        self.translator_ur = None
        self.vis_pro_en = None
        self.vis_pro_ur = None
        gc.collect()

        self.load_models()
      
 
    def stop_llm_generation(self):
        """Signals the LLM engine to stop the current generation stream."""
        if self.llm:
            self.llm.stop_stream()

    def switch_llm_if_needed(self, subject):
        """
        Switches the LLM based on the selected subject.
        - Math subject uses the 'deepseek-qwen2.5-1.5b' model.
        - Other subjects use the 'gemma-2-2b' model.
        """
        if self.current_subject == subject:
            return  # no change needed

        # determine if a model switch is necessary Math or Others
        is_new_subject_math = (subject == "Math")
        is_current_subject_math = (self.current_subject == "Math")

        if is_new_subject_math != is_current_subject_math:
            gr.Info(f"Subject changed to {subject}. Switching AI model...")

            target_model_name = self.math_llm_model if is_new_subject_math else self.general_llm_model
            self.current_llm_model_name = target_model_name

            target_model_path = self.model_reg.get_model_path("llm", target_model_name)
            self.llm.switch_model(target_model_path)

            gc.collect()
            gr.Info("AI model switched successfully.")

        self.current_subject = subject

    def ocr_image(self, image_path, lang="en"):
        """Extracts text from an image using OCR."""
        if not image_path:
            return ""
        gr.Info(get_status_message("extracting_text", lang))
        start_time = time.time()
        ocr_engine = "Tesseract" if self.version == "arm" else "PaddleOCR"
        print(f"Extracting text from image using {ocr_engine}...")
        
        if lang == 'en':
            extracted_text = self.vis_pro_en.process_input(image_path, lang)
            # save extracted text for debugging
            with open('ocr_en.txt', 'w') as f:
                f.write(extracted_text)
            end_time = time.time()
            print(f"Text extracted in {end_time - start_time:.2f} seconds.")
            return extracted_text.strip()
        elif lang == 'ur':
            extracted_text = self.vis_pro_ur.process_input(image_path, lang)
            with open('ocr_ur.txt', 'w') as f:
                f.write(extracted_text)
            end_time = time.time()
            print(f"Text extracted in {end_time - start_time:.2f} seconds.")
            return extracted_text.strip()
        else:
            return ""

    def transcribe_audio(self, audio_path, lang="en"):
        """Converts audio to text using speech recognition."""
        if not audio_path:
            return ""
        gr.Info(get_status_message("transcribing_audio", lang))
        start_time = time.time()
        print(f"Transcribing audio using {self.stt_model_name}...")
        transcription = self.speech_rec.process_input(audio_path, lang)

        # save transcription for debugging
        with open('transcription.txt', 'w') as f:
            f.write(transcription)

        end_time = time.time()
        print(f"Audio transcribed in {end_time - start_time:.2f} seconds.")
        return transcription.strip()

    def synthesize_speech(self, text, lang="en"):
        """Converts text to speech using TTS.
           Optional translator is used for urdu number conversion for urdu text.
        """
        if not text:
            return None

        gr.Info(get_status_message("synthesizing_speech", lang))
        start_time = time.time()

        if lang == 'en':
            print(f"Synthesizing speech using {self.tts_en_model_name}...")
            audio_file = self.speech_synth_en.process_input(text, lang)
            end_time = time.time()
            print(f"Speech synthesized in {end_time - start_time:.2f} seconds.")
            return audio_file  # return the file path
        elif lang == 'ur':
            print(f"Synthesizing speech using {self.tts_ur_model_name}...")
            audio_file = self.speech_synth_ur.process_input(text, lang, translator=self.translator_ur)
            end_time = time.time()
            print(f"Speech synthesized in {end_time - start_time:.2f} seconds.")
            return audio_file  # return the file path

    def translate_text(self, text, source_lang, target_lang):
        """Translates text between English and Urdu."""
        if not text or source_lang == target_lang:
            return text

        gr.Info(get_status_message("translating_text", target_lang))
        start_time = time.time()

        translator = None
        model_name = None
        if source_lang == 'ur' and target_lang == 'en':
            translator = self.translator_en
            model_name = self.translator_en_model_name
        elif source_lang == 'en' and target_lang == 'ur':
            translator = self.translator_ur
            model_name = self.translator_ur_model_name

        if translator is None:
            return text
            
        print(f"Translating text using {model_name}...")
        # split by paragraph or new line for better formatting
        if '\n\n' in text:
            paragraphs = text.split('\n\n')
            translated_paragraphs = []
            for paragraph in paragraphs:
                if paragraph.strip():
                    translated_chunk = translator.process_input(paragraph.strip(), source_lang)
                    translated_paragraphs.append(translated_chunk)
            result = "\n\n".join(translated_paragraphs)
        elif '\n' in text:
            lines = text.split('\n')
            translated_lines = []
            for line in lines:
                if line.strip():
                    translated_chunk = translator.process_input(line.strip(), source_lang)
                    translated_lines.append(translated_chunk)
            result = "\n".join(translated_lines)

        else:
            result = translator.process_input(text.strip(), source_lang)
        
        end_time = time.time()
        print(f"Text translated in {end_time - start_time:.2f} seconds.")
        # save translation for debugging
        with open('translated.txt', 'w', encoding='utf-8') as f:
            f.write(f"Original: {text}\n\nTranslated: {result}")
        
        return result

    def user_message(self, user_text, image_path, audio_path, session_id, lang, history, session_manager):
        """Handles user input, processes multimodal inputs, and updates the chat history."""
        # gather all multimodal inputs
        parts = []
        if user_text:
            parts.append(user_text.strip())

        # handle image input
        if image_path:
            img_text = self.ocr_image(image_path, lang)
            if img_text.strip():
                parts.append(f"{img_text.strip()}")

        # handle audio input
        if audio_path:
            aud_text = self.transcribe_audio(audio_path, lang)
            if aud_text.strip():
                parts.append(f"{aud_text.strip()}")

        combined_input = "\n\n".join(parts).strip()
        
        # llm_input is always english, display_text is for the UI
        llm_input = combined_input
        display_text = combined_input
        translated_content_for_db = None

        if lang == 'ur' and combined_input:
            # translate to english for the LLM
            llm_input = self.translate_text(combined_input, 'ur', 'en')
            # the UI will display the original Urdu
            display_text = combined_input
            # store the translated version for the database
            translated_content_for_db = f'<div dir="rtl" lang="ur">{display_text}</div>'

        # save to database
        session_manager.add_message_to_database(
            session_id, "user", llm_input, translated_content_for_db
        )

        # update the UI history 
        if translated_content_for_db:
            history.append({"role": "user", "content": translated_content_for_db})
        else:
            history.append({"role": "user", "content": display_text})

        # check if we need to refresh session list (title might have changed)
        sessions = session_manager.list_session_choices()
        return "", history, gr.update(value=None), gr.update(value=None), gr.update(choices=sessions, value=session_id)


    def llm_response(self, session_id, lang, grade, subject, speak, history, session_manager):
        """Generates a response from the LLM with streaming and handles post-processing."""
        self.switch_llm_if_needed(subject)

        # get history with original english content for the llm
        llm_history = session_manager.get_history_for_llm(session_id)
        
        # build llm messages from current history, but replace last user message with translated version
        # format system prompt using UI selections
        subject_for_prompt = subject
        grade_for_prompt = grade
        
        llm_messages = self.llm.build_messages(llm_history, grade_for_prompt, subject_for_prompt)

        # add empty assistant message to history
        history.append({"role": "assistant", "content": ""})
        gr.Info(get_status_message("generating_response", lang))
        # wait for the llm to finish streaming response, handling <think> tags
        part = ""
        state = "INITIAL"  # can be INITIAL, THINKING, STREAMING
        response_buffer = ""

        start_time = time.time()
        print(f"Generating LLM stream using {self.current_llm_model_name}...")
        for token in self.llm.stream_reply(llm_messages):
            # if we are already streaming, just append the token and update UI
            if state == "STREAMING":
                part += token
                history[-1]["content"] = part
                yield history, None
                continue

            # buffer tokens while we determine the state
            response_buffer += token

            if state == "INITIAL":
                if "<think>" in response_buffer:
                    state = "THINKING"
                    history[-1]["content"] = "thinking..."
                    yield history, None
                # if no <think> tag after a limit, assume normal streaming
                elif len(response_buffer) > 20:
                    state = "STREAMING"
                    part = response_buffer
                    history[-1]["content"] = part
                    yield history, None

            elif state == "THINKING":
                if "</think>" in response_buffer:
                    state = "STREAMING"
                    # extract content that comes after the </think> tag
                    part = response_buffer.split("</think>", 1)[1]
                    history[-1]["content"] = part
                    yield history, None
        
        end_time = time.time()
        print(f"LLM stream generated in {end_time - start_time:.2f} seconds.")
        # after the loop, handle any content that was buffered but not streamed
        # like very short responses that don't trigger thinking state
        if state == "INITIAL":
            part = response_buffer

        display_response = part
        translated_for_db = None  # will store the translated content for database

        if lang == 'ur':
            # translate english response to urdu for display
            translated_text = self.translate_text(display_response, 'en', 'ur')
            display_text = self.translator_ur.convert_to_urdu_num(translated_text)
            # wrap urdu text with rtl html attributes for display
            display_response = f'<div dir="rtl" lang="ur">{display_text}</div>'
            
            # store the wrapped translation for database
            translated_for_db = display_response
            
            # update the history with rtl-wrapped translated response
            history[-1]["content"] = display_response
            yield history, None

        # save messages to the database with translation if available
        session_manager.add_message_to_database(session_id, "assistant", part, translated_for_db)
        
        # generate tts if enabled
        audio_file = None
        if speak:
            if lang == 'ur':
                audio_file = self.synthesize_speech(translated_text, lang)
            else:
                audio_file = self.synthesize_speech(display_response, lang)
        
        # final yield with audio file
        proc_mem()
        gc.collect()
        yield history, audio_file

