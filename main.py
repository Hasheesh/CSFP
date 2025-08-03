import os
import time

# Import all engines
from llm_engine import LLM
from vision_processor import VisionProcessor
from speech_synthesizer import SpeechSynthesizer
from speech_recognizer import SpeechRecognizer
from translator import Translator
# Placeholder classes for engines that are not yet implemented








class AITutor:
    """Main AI Tutor class that orchestrates all models."""
    
    def __init__(self):
        # Initialize all engines
        self.llm_tutor = LLM('llm','phi-4-mini')
        self.speech_synth_en = SpeechSynthesizer('tts', 'piper-tts-en')
        self.speech_rec= SpeechRecognizer('stt', 'whisper-small')
        self.translator_en = Translator('translation', 'opus-mt-ur-en')
        self.translator_ur = Translator('translation', 'opus-mt-en-ur')
        self.vis_pro = VisionProcessor()

 

    

    def interactive_mode(self):
        is_speech = False
        lang = ""

        print("=== AI Tutor ===")
        print("  1. for text input")
        print("  2. for image input")
        print("  3. for speech input")
        print("  4. to select your language")
        print("  5. to enable speech output")
        print("  Enter quit to Exit")
        
        while True:
            try:
                option = input("\nOption: ").strip()
                    
                if option == "1":
                    print("Enter your Question")
                    question = input("\nQuestion: ").strip()

                    if question:
                        print("\n" + "-"*50)
                        print(f"\nTutor: ")
                        response = self.llm_tutor.process_input(question)
                        print("-"*50)

                elif option == "2":
                
                    path = input("Image Path: ").strip()
                    
                    if path:
                        extracted_text = self.vis_pro.process_input(path)
                        response = self.llm_tutor.process_input(extracted_text) 
                        print("-"*50)

                elif option == "3":
                    # TODO take input from mic
                    path = input("Audio Path: ").strip()
                    
                    if path:
                        print("\n" + "-"*50)
                        transcription = self.speech_rec.process_input(path)
                        print(f"Transcription: {transcription}")
                        
                        # Process transcription with LLM
                        response = self.llm_tutor.process_input(transcription)
                        print(f"LLM Response: {response}")
                         
                elif option == "4":
                    print("Please select your language: ")
                    lang = input("ur/en ? ")
                    
                    

                elif option == "5":
                    if response:
                        print("\n" + "-"*50)
                        is_speech = not is_speech

                        if is_speech:
                            print("Speech output enabled")
                        else:
                            print("Speech output disabled")    

                        print("-"*50)



                elif option.lower() == 'quit':
                        print("\nExiting...")
                        break

            except KeyboardInterrupt:
                print("\nExiting...")
                break
            except Exception as e:
                print(f"Error: {str(e)}")

def main():
    """Main entry point for the AI Tutor."""
    print("=== AI Tutor System ===")
    print("Complete Pipeline: Input → STT/OCR → Translation → LLM → Translation → TTS")

    # Create AI Tutor instance
    tutor = AITutor()
    
    # Start interactive mode
    tutor.interactive_mode()

if __name__ == "__main__":
    main()