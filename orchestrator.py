"""orchestrator.py

This file is the main orchestrator for the ai tutor.
It handles the logic for the model calls and database management.

The parts of code not written by me are referenced from the following sources:
- Code to read memory usage from https://stackoverflow.com/questions/938733/total-memory-used-by-python-process
"""
import os
import psutil
import gradio as gr

from model_registry import ModelRegistry
from llm_engine import LLMEngine
from vision_processor import VisionProcessor
from speech_recognizer import SpeechRecognizer
from translator import Translator
from speech_synthesizer import SpeechSynthesizer
from config_loader import get_status_message, get_user_profile

def proc_mem():
    """Returns the process's current memory usage as a formatted string."""
    p = psutil.Process(os.getpid())
    rss_mb = p.memory_info().rss / (1024**2)
    return f"Process RSS: {rss_mb:.1f} MiB"

class Orchestrator:
    """Main orchestrator that handles all AI model calls and database management."""
    
    def __init__(self):
        """Initializes all the AI models and engines."""
        self.model_reg = ModelRegistry()
        self.general_llm_model = "gemma-2-2b"
        self.math_llm_model = "deepseek-qwen2.5-1.5b"

        # get user profile to find initial subject
        user_profile = get_user_profile()
        initial_subject = user_profile.get("subject")
        self.current_subject = initial_subject

        # determine which LLM to load initially
        initial_llm_model = self.math_llm_model if initial_subject == "Math" else self.general_llm_model
        llm_path = self.model_reg.get_model_path("llm", initial_llm_model)
        self.llm = LLMEngine(llm_path)

        # load text to speech models for both languages
        tts_en_path = self.model_reg.get_model_path('tts', 'piper-tts-en-amy')
        self.speech_synth_en = SpeechSynthesizer(tts_en_path)
        tts_ur_path = self.model_reg.get_model_path('tts', 'mms-tts-ur')
        self.speech_synth_ur = SpeechSynthesizer(tts_ur_path)
        
        # load speech to text model
        stt_path = self.model_reg.get_model_path('stt', 'faster-whisper-small')
        self.speech_rec = SpeechRecognizer(stt_path)
        
        # load translation model
        translator_path = self.model_reg.get_model_path('translation', 'nllb-200-600M-Q8')
        self.translator = Translator(translator_path)

        # load ocr models for english - using server detection for better accuracy
        ocr_en_det_path = self.model_reg.get_model_path('ocr', 'PP-OCRv5_server_det')
        ocr_en_rec_path = self.model_reg.get_model_path('ocr', 'en_PP-OCRv5_mobile_rec')
        ocr_en_det_model_name = 'PP-OCRv5_server_det'
        ocr_en_rec_model_name = 'en_PP-OCRv5_mobile_rec'
        self.vis_pro_en = VisionProcessor(ocr_en_det_path, ocr_en_rec_path, ocr_en_det_model_name, ocr_en_rec_model_name)
        
        # load ocr models for urdu
        ocr_ur_det_path = self.model_reg.get_model_path('ocr', 'PP-OCRv3_mobile_det')
        ocr_ur_rec_path = self.model_reg.get_model_path('ocr', 'arabic_PP-OCRv3_mobile_rec')
        ocr_ur_det_model_name = 'PP-OCRv3_mobile_det'
        ocr_ur_rec_model_name = 'arabic_PP-OCRv3_mobile_rec'
        self.vis_pro_ur = VisionProcessor(ocr_ur_det_path, ocr_ur_rec_path, ocr_ur_det_model_name, ocr_ur_rec_model_name)

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

            target_model_path = self.model_reg.get_model_path("llm", target_model_name)
            self.llm.switch_model(target_model_path)

            gr.Info("AI model switched successfully.")

        self.current_subject = subject

    def ocr_image(self, image_path, lang="en"):
        """Extracts text from an image using OCR."""
        if not image_path:
            return ""
        gr.Info(get_status_message("extracting_text", lang))
        
        if lang == 'en':
            extracted_text = self.vis_pro_en.process_input(image_path, lang)
            # save extracted text for debugging
            with open('ocr_en.txt', 'w') as f:
                f.write(extracted_text)
            return extracted_text.strip()
        elif lang == 'ur':
            extracted_text = self.vis_pro_ur.process_input(image_path, lang)
            with open('ocr_ur.txt', 'w') as f:
                f.write(extracted_text)
            return extracted_text.strip()
        else:
            return ""

    def transcribe_audio(self, audio_path, lang="en"):
        """Converts audio to text using speech recognition."""
        if not audio_path:
            return ""
        gr.Info(get_status_message("transcribing_audio", lang))
        transcription = self.speech_rec.process_input(audio_path, lang)

        # save transcription for debugging
        with open('transcription.txt', 'w') as f:
            f.write(transcription)

        return transcription.strip()

    def synthesize_speech(self, text, lang="en"):
        """Converts text to speech using TTS.
           Optional translator is used for urdu number conversion for urdu text.
        """
        if not text:
            return None

        gr.Info(get_status_message("synthesizing_speech", lang))

        if lang == 'en':
            audio_file = self.speech_synth_en.process_input(text, lang)
            return audio_file  # return the file path
        elif lang == 'ur':
            audio_file = self.speech_synth_ur.process_input(text, lang, translator=self.translator)
            return audio_file  # return the file path

    def translate_text(self, text, source_lang, target_lang):
        """Translates text between English and Urdu."""
        if not text or source_lang == target_lang:
            return text

        gr.Info(get_status_message("translating_text", target_lang))
            
        # translate paragraph by paragraph for better quality
        if '\n\n' in text:
            paragraphs = text.split('\n\n')
            translated_paragraphs = []
            for paragraph in paragraphs:
                if paragraph.strip():
                    translated_chunk = self.translator.process_input(paragraph.strip(), source_lang)
                    translated_paragraphs.append(translated_chunk)
            result = "\n\n".join(translated_paragraphs)
        else:
            result = self.translator.process_input(text.strip(), source_lang)
        
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
        
        # after the loop, handle any content that was buffered but not streamed
        # like very short responses that don't trigger thinking state
        if state == "INITIAL":
            part = response_buffer

        display_response = part
        translated_for_db = None  # will store the translated content for database

        if lang == 'ur':
            # translate english response to urdu for display
            translated_text = self.translate_text(display_response, 'en', 'ur')
            display_text = self.translator.convert_to_urdu_num(translated_text)
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
        yield history, audio_file

