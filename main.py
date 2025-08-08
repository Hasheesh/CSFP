from gettext import translation
import os
import time

from transformers.models import FalconForSequenceClassification

# Import all engines
from llm_engine import LLM
from vision_processor import VisionProcessor
from speech_synthesizer import SpeechSynthesizer
from speech_recognizer import SpeechRecognizer
from translator import Translator
from model_registery import ModelRegistery
# Placeholder classes for engines that are not yet implemented
import psutil
import time
import os
import gc
    # Get virtual memory information
def proc_mem():
    p = psutil.Process(os.getpid())
    rss_mb = p.memory_info().rss / (1024**2)
    return f"Process RSS: {rss_mb:.1f} MiB"



class AITutor:
    """Main AI Tutor class that orchestrates all models."""
    
    def __init__(self):
        # Initialize all engines
        model_reg = ModelRegistery()
        # llm_path = model_reg.get_model_path('llm', 'phi-4-mini')
        llm_path = model_reg.get_model_path('llm', 'gemma-3-4b')

        self.llm_tutor = LLM(llm_path)
        tts_en_path = model_reg.get_model_path('tts', 'piper-tts-en')
        self.speech_synth_en = SpeechSynthesizer(tts_en_path)
        tts_ur_path = model_reg.get_model_path('tts', 'mms-tts-ur')
        self.speech_synth_ur = SpeechSynthesizer(tts_ur_path)
        stt_path = model_reg.get_model_path('stt', 'whisper-small')
        self.speech_rec= SpeechRecognizer(stt_path)
        translation_ur_en_path = model_reg.get_model_path('translation', 'opus-mt-ur-en')
        self.translator_en = Translator(translation_ur_en_path)
        translation_en_ur_path = model_reg.get_model_path('translation', 'opus-mt-en-ur')
        self.translator_ur = Translator(translation_en_ur_path)
        ocr_en_det_path = model_reg.get_model_path('ocr', 'PP-OCRv5_mobile_det')
        ocr_en_rec_path = model_reg.get_model_path('ocr', 'PP-OCRv5_mobile_rec')
        ocr_en_det_model_name = 'PP-OCRv5_mobile_det'
        ocr_en_rec_model_name = 'PP-OCRv5_mobile_rec'
        self.vis_pro_en = VisionProcessor(ocr_en_det_path, ocr_en_rec_path, ocr_en_det_model_name, ocr_en_rec_model_name)
        ocr_ur_rec_path = model_reg.get_model_path('ocr', 'arabic_PP-OCRv3_mobile_rec')
        ocr_ur_rec_model_name = 'arabic_PP-OCRv3_mobile_rec'
        self.vis_pro_ur = VisionProcessor(ocr_en_det_path, ocr_ur_rec_path, ocr_en_det_model_name, ocr_ur_rec_model_name)



    def interactive_mode(self):
        is_speech = FalconForSequenceClassification
        lang = "en"

        print("=== AI Tutor ===")
        print("  1. for text input")
        print("  2. for image input")
        print("  3. for speech input")
        print("  4. to select your language")
        print("  5. to toggle speech output")
        print("  Enter quit to Exit")
        
        while True:
            print(proc_mem())
            try:
                option = input("\nOption: ").strip()
                    
                if option == "1":
                    print("Enter your Question")
                    question = input("\nQuestion: ").strip()

                    if question:
                        if lang == 'en':
                            print("\n" + "-"*50)
                            response = self.llm_tutor.process_input(question)
                            print("-"*50)
                            if is_speech:
                                print("Waiting for answer before speech...")
                                print(response)
                                print("-"*50)
                                self.speech_synth_en.process_input(response, lang)

                        elif lang == 'ur':
                            # t_question_en = self.translator_en.process_input(question)
                            response = self.llm_tutor.process_input(question)
                            # translate response to urdu
                            t_response = self.translator_ur.process_input(response)
                            print(t_response)
                            if is_speech:
                                self.speech_synth_ur.process_input(t_response, lang)

                elif option == "2":
                    
                    path = input("Image Path: ").strip()
                    
                    if path:
                        if lang == 'en':
                            extracted_text = self.vis_pro_en.process_input(path, lang)
                            # response = self.llm_tutor.process_input(extracted_text) 
                            print("-"*50)

                        elif lang == 'ur':
                            extracted_text = self.vis_pro_ur.process_input(path, lang)
                            # response = self.llm_tutor.process_input(extracted_text) 
                            print("-"*50)

                elif option == "3":
                    # TODO take input from mic
                    path = input("Audio Path: ").strip()
                    
                    if path:
                        print("\n" + "-"*50)
                        transcription = self.speech_rec.process_input(path, lang)
                        print(f"Transcription: {transcription}")
                        # # Process transcription with LLM
                        # response = self.llm_tutor.process_input(transcription)
                        # print(f"LLM Response: {response}")
                         
                elif option == "4":
                    print("Please select your language: ")
                    lang = input("ur/en ? ")
                    
                    

                elif option == "5":
                   
                    is_speech = not is_speech

                    if is_speech:
                        print("Speech output enabled")
                    else:
                        print("Speech output disabled")    

                    print("-"*50)
                    print(proc_mem())



                elif option.lower() == 'quit':
                        print("\nExiting...")
                        break

            except KeyboardInterrupt:
                print("\nExiting...")
                break
            except Exception as e:
                print(f"Error: {str(e)}")

    def unload(self):
        

        print(f"[unload] ...")


        del self.speech_rec
        self.speech_rec = None
        


        # run GC; give OS a tick to reclaim
        gc.collect()
        time.sleep(0.2)

        print("[after unload]", proc_mem())

def main():
    """Main entry point for the AI Tutor."""
    print("=== AI Tutor System ===")
    print("Complete Pipeline: Input → STT/OCR → Translation → LLM → Translation → TTS")

    # Create AI Tutor instance
    tutor = AITutor()
    
    # Start interactive mode
    tutor.interactive_mode()



if __name__ == "__main__":
    print(proc_mem()) 
    main()


     