"""orchestrator.py

This file is the main orchestrator for the ai tutor.
It handles the logic for the model calls and database management.

The parts of code not written by me are referenced from the following sources:
- Code to remove emojis from https://stackoverflow.com/questions/33404752/removing-emojis-from-a-string-in-python/49146722#49146722
- Code to convert numbers to words from https://github.com/savoirfairelinux/num2words

"""

import os
import re
import gc
import time
import psutil
import gradio as gr
import urduhack
from bidi.algorithm import get_display
# num2words moved to speech_synthesizer.py

# Local modules
from model_registery import ModelRegistery
from chat_database import ChatDatabase
from llm_engine import LLMEngine
from vision_processor import VisionProcessor
from speech_recognizer import SpeechRecognizer
from translator import Translator
from speech_synthesizer import SpeechSynthesizer
# Session management moved to gradio_ui.py SessionController

# status messages for user feedback in both languages
STATUS_MESSAGES = {
    "en": {
        "extracting_text": "Extracting Text from Image",
        "transcribing_audio": "Transcribing Audio",
        "synthesizing_speech": "Synthesizing Speech",
        "translating_text": "Translating Text",
        "generating_response": "Generating Response",
    },
    "ur": {
        "extracting_text": "تصویر سے متن نکال رہے ہیں",
        "transcribing_audio": "آواز کو متن میں تبدیل کر رہے ہیں",
        "synthesizing_speech": "آواز بنانے میں",
        "translating_text": "متن کا ترجمہ کر رہے ہیں",
        "generating_response": "جواب تیار کر رہے ہیں",
    }
}

def get_status_message(key, lang="en"):
    # get the right status message based on language
    return STATUS_MESSAGES.get(lang, STATUS_MESSAGES["en"]).get(key, key)

# Moved text processing patterns to respective classes

# get memory usage info for debugging
def proc_mem():
    p = psutil.Process(os.getpid())
    rss_mb = p.memory_info().rss / (1024**2)
    return f"Process RSS: {rss_mb:.1f} MiB"

def fix_urdu_ocr(text):
    # normalize urdu text and fix display direction
    n_text = urduhack.normalization.normalize(text)
    n_text = get_display(n_text)
    return n_text




class Orchestrator:
    # main orchestrator that handles all ai model calls and database management
    
    def __init__(self):
        # initialize all the ai models we need
        model_reg = ModelRegistery()

        # load the llm model - using gemma-2-2b for raspberry pi
        llm_path = model_reg.get_model_path("llm", "gemma-2-2b")
        self.llm = LLMEngine(llm_path)

        # load text to speech models for both languages
        tts_en_path = model_reg.get_model_path('tts', 'piper-tts-en-amy')
        self.speech_synth_en = SpeechSynthesizer(tts_en_path)
        tts_ur_path = model_reg.get_model_path('tts', 'mms-tts-ur')
        self.speech_synth_ur = SpeechSynthesizer(tts_ur_path)
        
        # load speech to text model
        stt_path = model_reg.get_model_path('stt', 'faster-whisper-small')
        self.speech_rec = SpeechRecognizer(stt_path)
        
        # load translation model
        translator_path = model_reg.get_model_path('translation', 'nllb-200-600M-Q8')
        self.translator = Translator(translator_path)

        # load ocr models for english - using server detection for better accuracy
        ocr_en_det_path = model_reg.get_model_path('ocr', 'PP-OCRv5_server_det')
        ocr_en_rec_path = model_reg.get_model_path('ocr', 'en_PP-OCRv5_mobile_rec')
        ocr_en_det_model_name = 'PP-OCRv5_server_det'
        ocr_en_rec_model_name = 'en_PP-OCRv5_mobile_rec'
        self.vis_pro_en = VisionProcessor(ocr_en_det_path, ocr_en_rec_path, ocr_en_det_model_name, ocr_en_rec_model_name)
        
        # load ocr models for urdu
        ocr_ur_det_path = model_reg.get_model_path('ocr', 'PP-OCRv3_mobile_det')
        ocr_ur_rec_path = model_reg.get_model_path('ocr', 'arabic_PP-OCRv3_mobile_rec')
        ocr_ur_det_model_name = 'PP-OCRv3_mobile_det'
        ocr_ur_rec_model_name = 'arabic_PP-OCRv3_mobile_rec'
        self.vis_pro_ur = VisionProcessor(ocr_ur_det_path, ocr_ur_rec_path, ocr_ur_det_model_name, ocr_ur_rec_model_name)

    # Session management moved to gradio_ui.py SessionController

    # multimodal utility functions for handling different input types
    def ocr_image(self, image_path, lang="en"):
        # extract text from image using ocr
        if not image_path:
            return ""
        gr.Info(get_status_message("extracting_text", lang))
        try:
            if lang == 'en':
                extracted_text = self.vis_pro_en.process_input(image_path, lang)
                # save extracted text for debugging
                with open('ocr_en.txt', 'w') as f:
                    f.write(extracted_text)
                return extracted_text.strip()
            elif lang == 'ur':
                extracted_text = self.vis_pro_ur.process_input(image_path, lang)
                print(f"Extracted text: {extracted_text}")
                # fix urdu text display issues
                fixed_text = fix_urdu_ocr(extracted_text)
                with open('ocr_ur.txt', 'w') as f:
                    f.write(fixed_text)
                print(f"Fixed text: {fixed_text}")
                return fixed_text.strip()
            else:
                return ""
        except Exception as e:
            print(f"Error in OCR processing: {e}")
            return ""

    def transcribe_audio(self, audio_path, lang="en"):
        # convert audio to text using speech recognition
        if not audio_path:
            return ""
        gr.Info(get_status_message("transcribing_audio", lang))
        try:
            transcription = self.speech_rec.process_input(audio_path, lang)
            if lang == 'ur':
                # save transcription for debugging
                with open('transcription.txt', 'w') as f:
                    f.write(transcription)
            return transcription.strip()
        except Exception as e:
            print(f"Error in STT processing: {e}")
            return ""

    def synthesize_speech(self, text, lang="en"):
        # convert text to speech using tts
        if not text:
            return None
        gr.Info(get_status_message("synthesizing_speech", lang))
        try:
            if lang == 'en':
                audio_file = self.speech_synth_en.process_input(text, lang)
                return audio_file  # return the file path
            elif lang == 'ur':
                # pass translator for urdu number conversion
                audio_file = self.speech_synth_ur.process_input(text, lang, translator=self.translator)
                return audio_file  # return the file path
            return None
        except Exception as e:
            print(f"Error in TTS processing: {e}")
            return None

    def translate_text(self, text, source_lang, target_lang):
        # translate text between english and urdu
        if not text or source_lang == target_lang:
            return text
        gr.Info(get_status_message("translating_text", target_lang))
        try:
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
        except Exception as e:
            print(f"Error in translation: {e}")
            return text

    # get_history moved to gradio_ui.py SessionManager

    # main response functions for handling user input and generating responses
    def user_message(self, user_text, image_path, audio_path, session_id, lang, history, session_manager):
        # handle user input and add to chat history
        # session switching is now handled by the ui layer

        # gather all multimodal inputs
        parts = []
        if user_text:
            parts.append(user_text.strip())

        # handle image input - skip if it's a directory or none
        if image_path and not os.path.isdir(image_path):
            img_text = self.ocr_image(image_path, lang)
            if img_text.strip():
                parts.append(f"{img_text.strip()}")

        # handle audio input - skip if it's a directory or none
        if audio_path and not os.path.isdir(audio_path):
            aud_text = self.transcribe_audio(audio_path, lang)
            if aud_text.strip():
                parts.append(f"[Audio]\n{aud_text.strip()}")

        combined_prompt = "\n\n".join(parts).strip() or ""
        
        # keep original text for display, translate for llm processing
        display_text = combined_prompt
        llm_input = combined_prompt
        
        if lang == 'ur' and combined_prompt != "":
            # translate for llm but keep original for display
            llm_input = self.translate_text(combined_prompt, 'ur', 'en')
            # store original urdu text for display
            display_text = combined_prompt

        # add user message to database 
        session_manager.add_message_to_database(session_id, "user", display_text)

        # add user message to history with rtl formatting if needed
        if lang == 'ur' and display_text:
            # wrap urdu user input with rtl html attributes
            formatted_display_text = f'<div dir="rtl" lang="ur">{display_text}</div>'
            history.append({"role": "user", "content": formatted_display_text})
        else:
            history.append({"role": "user", "content": display_text})
        
        # store translated input for llm processing
        self._current_llm_input = llm_input
        
        # check if we need to refresh session list (title might have changed)
        sessions = session_manager.list_session_choices()
        return "", history, gr.update(value=None), gr.update(value=None), gr.update(choices=sessions, value=session_id)


    def llm_response(self, session_id, lang, speak, history, session_manager):
        # generate bot response with streaming
        # build llm messages from current history, but replace last user message with translated version
        llm_messages = self.llm.build_messages(history)

        # if we have translated input, replace the last user message
        if hasattr(self, '_current_llm_input') and self._current_llm_input:
            if llm_messages and llm_messages[-1]["role"] == "user":
                llm_messages[-1]["content"] = self._current_llm_input

        # add empty assistant message to history
        history.append({"role": "assistant", "content": ""})
        gr.Info(get_status_message("generating_response", lang))
        
        part = ""
        for token in self.llm.stream_reply(llm_messages):
            part += token
            # update the last message in history
            history[-1]["content"] = part
            yield history, None  # yield history and none for audio initially

        display_response = part

        if lang == 'ur':
            # translate english response to urdu for display
            translated_text = self.translate_text(display_response, 'en', 'ur')
            display_text = self.translator.convert_to_urdu_num(translated_text)
            # wrap urdu text with rtl html attributes for display
            display_response = f'<div dir="rtl" lang="ur">{display_text}</div>'
            
            # update the history with rtl-wrapped translated response
            history[-1]["content"] = display_response
            yield history, None

        # save assistant message to database (store original english)
        session_manager.add_message_to_database(session_id, "assistant", part)
        
        # generate tts if enabled
        audio_file = None
        if speak:
            if lang == 'ur':
                audio_file = self.synthesize_speech(translated_text, lang)
            else:
                audio_file = self.synthesize_speech(display_response, lang)
        
        # final yield with audio file
        yield history, audio_file

    # def unload(self):
    #     """Unload models to free memory."""
    #     print(f"[unload] ...")
        
    #     # Unload STT model
    #     if hasattr(self, 'speech_rec') and self.speech_rec:
    #         self.speech_rec.unload()
    #         del self.speech_rec
    #         self.speech_rec = None
        
    #     # Unload TTS models
    #     if hasattr(self, 'speech_synth_en') and self.speech_synth_en:
    #         self.speech_synth_en.unload()
    #         del self.speech_synth_en
    #         self.speech_synth_en = None
            
    #     if hasattr(self, 'speech_synth_ur') and self.speech_synth_ur:
    #         self.speech_synth_ur.unload()
    #         del self.speech_synth_ur
    #         self.speech_synth_ur = None
        
    #     # Unload translation model
    #     if hasattr(self, 'translator') and self.translator:
    #         self.translator.unload()
    #         del self.translator
    #         self.translator = None
        
    #     # Unload OCR models
    #     if hasattr(self, 'vis_pro_en') and self.vis_pro_en:
    #         self.vis_pro_en.unload()
    #         del self.vis_pro_en
    #         self.vis_pro_en = None
            
    #     if hasattr(self, 'vis_pro_ur') and self.vis_pro_ur:
    #         self.vis_pro_ur.unload()
    #         del self.vis_pro_ur
    #         self.vis_pro_ur = None

    #     # Run garbage collection
    #     gc.collect()
    #     time.sleep(0.2)
    #     print("[after unload]", proc_mem())