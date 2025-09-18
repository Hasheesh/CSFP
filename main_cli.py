"""
main_cli.py

This file provides a command-line interface for the AI tutor.
It's primarily used for testing and debugging the various models and engines.

"""
from llm_engine import LLMEngine
from vision_processor import VisionProcessor
from speech_synthesizer import SpeechSynthesizer
from speech_recognizer import SpeechRecognizer
from translator import Translator
from model_registry import ModelRegistry
import psutil
import os
from bidi.algorithm import get_display
import urduhack

def proc_mem():
    """Returns the process's current memory usage as a formatted string."""
    p = psutil.Process(os.getpid())
    rss_mb = p.memory_info().rss / (1024**2)
    return f"Process RSS: {rss_mb:.1f} MiB"


def fix_urdu_script(text):
    """Normalizes and corrects RTL display for Urdu text."""
    n_text = urduhack.normalization.normalize(text)
    n_text = get_display(n_text) # Apply RTL correction
    return n_text


class AITutor:
    """Main AI Tutor class that orchestrates all models for the CLI."""

    def __init__(self):
        """Initializes all the AI models and engines for the CLI."""
        model_reg = ModelRegistry()
        
        # set the desired LLM model here
        llm_path = model_reg.get_model_path('llm', 'llama-3.2-3b-q4')
        self.llm_tutor = LLMEngine(llm_path)

        tts_en_path = model_reg.get_model_path('tts', 'piper-tts-en-amy')
        self.speech_synth_en = SpeechSynthesizer(tts_en_path)
        tts_ur_path = model_reg.get_model_path('tts', 'mms-tts-ur')
        self.speech_synth_ur = SpeechSynthesizer(tts_ur_path)
        
        stt_path = model_reg.get_model_path('stt', 'faster-whisper-small')
        self.speech_rec = SpeechRecognizer(stt_path)
        
        translator_path = model_reg.get_model_path('translation', 'nllb-200-600M-Q8')
        self.translator = Translator(translator_path)

        ocr_en_det_path = model_reg.get_model_path('ocr', 'PP-OCRv5_mobile_det')
        ocr_en_rec_path = model_reg.get_model_path('ocr', 'en_PP-OCRv5_mobile_rec')
        ocr_en_det_model_name = 'PP-OCRv5_mobile_det'
        ocr_en_rec_model_name = 'en_PP-OCRv5_mobile_rec'
        self.vis_pro_en = VisionProcessor(ocr_en_det_path, ocr_en_rec_path, ocr_en_det_model_name, ocr_en_rec_model_name)
        
        ocr_ur_rec_path = model_reg.get_model_path('ocr', 'arabic_PP-OCRv3_mobile_rec')
        ocr_ur_rec_model_name = 'arabic_PP-OCRv3_mobile_rec'
        self.vis_pro_ur = VisionProcessor(ocr_en_det_path, ocr_ur_rec_path, ocr_en_det_model_name, ocr_ur_rec_model_name)
        

    def interactive_mode(self):
        """Runs the interactive command-line interface loop."""
        is_speech = True
        lang = "ur"

        print("\n=== AI Tutor CLI ===")
        print("  1. Text input")
        print("  2. Image input")
        print("  3. Speech input")
        print("  4. Select language")
        print("  5. Toggle speech output")
        print("  Enter 'quit' to Exit")
        
        while True:
            print(f"\n{proc_mem()} | Lang: {lang.upper()} | Speech: {'ON' if is_speech else 'OFF'}")
            try:
                option = input("Option: ").strip()
                    
                if option == "1":
                    question = input("Question: ").strip()
                    if not question:
                        continue

                    if lang == 'en':
                        print("\n" + "-"*50)
                        response = self.llm_tutor.process_input(question)
                        print(response)
                        print("-"*50)
                        if is_speech:
                            print("Synthesizing speech...")
                            self.speech_synth_en.process_input(response, lang)

                    elif lang == 'ur':
                        print("\nTranslating to English...")
                        t_question_en = self.translator.process_input(question, lang)
                        print(f"Translated: {t_question_en}")

                        print("\nGenerating response...")
                        response = self.llm_tutor.process_input(t_question_en)
                        
                        print("\nTranslating response to Urdu...")
                        paragraphs = response.split('\n\n')
                        translated_paragraphs = [
                            self.translator.process_input(p, 'en') for p in paragraphs if p.strip()
                        ]
                        t_response = "\n\n".join(translated_paragraphs)
                        
                        final_response = fix_urdu_script(t_response)
                        print(f"Final Response:\n{final_response}")

                        if is_speech:
                            print("\nSynthesizing speech...")
                            self.speech_synth_ur.process_input(t_response, lang)

                elif option == "2":
                    path = input("Image Path: ").strip()
                    if not path:
                        continue

                    if lang == 'en':
                        extracted_text = self.vis_pro_en.process_input(path, lang)
                        print(f"Extracted Text: {extracted_text}")
                        response = self.llm_tutor.process_input("Text from image: " + extracted_text) 
                        print(f"Response:\n{response}")
                        print("-"*50)
                        if is_speech:
                            self.speech_synth_en.process_input(response, lang)

                    elif lang == 'ur':
                        extracted_text = self.vis_pro_ur.process_input(path, lang)
                        n_text = fix_urdu_script(extracted_text)
                        print(f"Extracted Text (Urdu):\n{n_text}")
                        print("-"*50)

                elif option == "3":
                    path = input("Audio Path: ").strip()
                    if not path:
                        continue
                    
                    transcription = self.speech_rec.process_input(path, lang)
                    print(f"Transcription: {transcription}")
                    
                    response = self.llm_tutor.process_input("Text from audio: " + transcription) 
                    print(f"Response:\n{response}")
                    print("-"*50)

                elif option == "4":
                    selected_lang = input("Select language (ur/en): ").strip().lower()
                    if selected_lang in ["ur", "en"]:
                        lang = selected_lang
                    else:
                        print("Invalid language selected.")

                elif option == "5":
                    is_speech = not is_speech
                    print(f"Speech output {'enabled' if is_speech else 'disabled'}")    
                
                elif option.lower() == 'quit':
                    print("\nExiting...")
                    break

            except KeyboardInterrupt:
                print("\nExiting...")
                break
            except Exception as e:
                print(f"An error occurred: {e}")

def main():
    """Initializes and runs the AI Tutor CLI."""
    tutor = AITutor()
    tutor.interactive_mode()

if __name__ == "__main__":
    main()


     

