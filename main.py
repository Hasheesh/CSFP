# Import all engines
from llm_engine import LLM
from vision_processor import VisionProcessor
from speech_synthesizer import SpeechSynthesizer
from speech_recognizer import SpeechRecognizer
from translator import Translator
from model_registery import ModelRegistery
import psutil
import time
import os
import gc
import re
from bidi.algorithm import get_display
import urduhack


# Get memory information
def proc_mem():
    p = psutil.Process(os.getpid())
    rss_mb = p.memory_info().rss / (1024**2)
    return f"Process RSS: {rss_mb:.1f} MiB"


def fix_urdu_script(text):
    n_text = urduhack.normalization.normalize(text)
    n_text = get_display(n_text) # fix RTL
    return n_text

def clean_text_en(text):
    BULLETS = re.compile(r"^\s*([-*+•]|\.)\s+", re.MULTILINE)
    SYMBOLS_MAP = {
    "&": " and ",
    "%": " percent ",
    "+": " plus ",
    "€": " euros ",
    "$": " dollars ",
    "£": " pounds ",
    "@": " at ",
    "#": " number ",
    "°C": " degrees Celsius ",
    "°F": " degrees Fahrenheit ",
    }
    


    text = BULLETS.sub("", text)
    text = text.replace("**", "").replace("__", "").replace("*", "").replace("_", "")

    for s, w in SYMBOLS_MAP.items():
        text = text.replace(s, w)

    text = re.sub(r"\s+", " ", text)                # collapse spaces/newlines
    text = re.sub(r"[!?]{2,}", lambda m: m.group(0)[0], text)  # !!?? -> ! or ?
    text = re.sub(r"\.{3,}", " … ", text)           # change 3 dots to elipsis 
    return text.strip()



class AITutor:
    """Main AI Tutor class that orchestrates all models."""
    
    def __init__(self):
        # Initialize all engines
        model_reg = ModelRegistery()
        # llm_path = model_reg.get_model_path('llm', 'phi-4-mini')
        llm_path = model_reg.get_model_path('llm', 'gemma-2-2b')
        # llm_path = model_reg.get_model_path('llm', 'qwen-3-4b-q5')
        # llm_path = model_reg.get_model_path('llm', 'gemma-3-4b')
        # llm_path = model_reg.get_model_path('llm', 'gemma-3n-4b-q4')
        # llm_path = model_reg.get_model_path('llm', 'phi-3.1-mini')
        # llm_path = model_reg.get_model_path('llm', 'phi-3-mini')
        # llm_path = model_reg.get_model_path('llm', 'qwen-2.5-3b-q4')
        # llm_path = model_reg.get_model_path('llm', 'qwen-3-4b-q5')




        self.llm_tutor = LLM(llm_path)
        tts_en_path = model_reg.get_model_path('tts', 'piper-tts-en-amy')
        self.speech_synth_en = SpeechSynthesizer(tts_en_path)
        tts_ur_path = model_reg.get_model_path('tts', 'mms-tts-ur')
        self.speech_synth_ur = SpeechSynthesizer(tts_ur_path)
        
        stt_path = model_reg.get_model_path('stt', 'faster-whisper-small')
        self.speech_rec= SpeechRecognizer(stt_path)
        
        translator_path = model_reg.get_model_path('translation', 'nllb-200-600M-Q8')
        self.translator = Translator(translator_path)

        ocr_en_det_path = model_reg.get_model_path('ocr', 'PP-OCRv5_mobile_det')
        ocr_en_rec_path = model_reg.get_model_path('ocr', 'PP-OCRv5_mobile_rec')
        ocr_en_det_model_name = 'PP-OCRv5_mobile_det'
        ocr_en_rec_model_name = 'PP-OCRv5_mobile_rec'
        self.vis_pro_en = VisionProcessor(ocr_en_det_path, ocr_en_rec_path, ocr_en_det_model_name, ocr_en_rec_model_name)
        ocr_ur_rec_path = model_reg.get_model_path('ocr', 'arabic_PP-OCRv3_mobile_rec')
        ocr_ur_rec_model_name = 'arabic_PP-OCRv3_mobile_rec'
        self.vis_pro_ur = VisionProcessor(ocr_en_det_path, ocr_ur_rec_path, ocr_en_det_model_name, ocr_ur_rec_model_name)



    def interactive_mode(self):
        is_speech = True
        # lang = "en"
        lang = "ur"


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
                            print(response)

                            print("-"*50)
                            if is_speech:
                                print("Waiting for answer before speech...")
                                print(response)
                                print("-"*50)
                                self.speech_synth_en.process_input(response, lang)

                        elif lang == 'ur':
                            # question = fix_urdu_script(question)
                            t_question_en = self.translator.process_input("پودے اپنا کھانا کیسے تیار کرتے ہیں ؟", lang)
                            print(t_question_en)
                            response = self.llm_tutor.process_input(t_question_en)
                            # translate response to urdu
                            cleaned_response = clean_text_en(response)
                            # --- START OF FIX ---
                            print("\nTranslating response to Urdu paragraph by paragraph...")
                            
                            # Split the long response into paragraphs
                            paragraphs = response.split('\n\n')
                            
                            translated_paragraphs = []
                            for paragraph in paragraphs:
                                if paragraph.strip(): # Ensure the paragraph is not empty
                                    # Translate each paragraph individually
                                    translated_chunk = self.translator.process_input(paragraph, 'en')
                                    translated_paragraphs.append(translated_chunk)
                            
                            # Join the translated paragraphs back together
                            t_response = "\n\n".join(translated_paragraphs)
                            
                            # print("\n" + t_response)

                            with open("trans.txt", "w", encoding="utf-8") as f:
                                f.write(t_response)
                            # t_response = self.translator.process_input(response, 'en')
                            # print(cleaned_response)
                            # with open("trans.txt", "w") as f:
                            #     f.write(t_response)
                            #     f.close()
                            print("translated")
                            if is_speech:
                                self.speech_synth_ur.process_input(t_response, lang)

                elif option == "2":
                    
                    path = input("Image Path: ").strip()
                    
                    if path:
                        if lang == 'en':
                            extracted_text = self.vis_pro_en.process_input(path, lang)
                            print(extracted_text)
                            response = self.llm_tutor.process_input("Text extracted from image input:  " + extracted_text) 
                            print("-"*50)
                            if is_speech:
                                print("Waiting for answer before speech...")
                                print(response)
                                print("-"*50)
                                self.speech_synth_en.process_input(response, lang)

                        elif lang == 'ur':
                            extracted_text = self.vis_pro_ur.process_input(path, lang)
                            n_text = fix_urdu_script(extracted_text)
                            with open('urdu.txt', 'w') as f:
                                f.write(n_text + " \n")
                            print(n_text)
                            # response = self.llm_tutor.process_input("Text extracted from image input:  " + n_text) 
                            print("-"*50)

                elif option == "3":
                    # TODO take input from mic
                    path = input("Audio Path: ").strip()
                    
                    if path:
                        if lang == 'en':
                            transcription = self.speech_rec.process_input(path, lang)
                            print(f"Transcription: {transcription}")
                            response = self.llm_tutor.process_input("Text extracted from audio input:  " + transcription) 
                            print("-"*50)

                        elif lang == 'ur':
                            print("\n" + "-"*50)
                            transcription = self.speech_rec.process_input(path, lang)
                            print(f"Transcription: {transcription}")
                            transcription = urduhack.normalization.normalize(transcription)
                            # Process transcription with LLM
                            # response = self.llm_tutor.process_input(transcription)
                            # print(f"LLM Response: {response}")
                            with open ('urdu.txt', 'w') as f:
                                f.write(transcription)


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


     