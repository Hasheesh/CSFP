import os
import time

# Import all engines
from llm_engine import LLM
from models import get_model_path

# Placeholder classes for engines that are not yet implemented








class AITutor:
    """Main AI Tutor class that orchestrates all models."""
    
    def __init__(self):
        # Initialize all engines
        self.llm_tutor = LLM()

    

 

    

    def interactive_mode(self):
        print("=== AI Tutor ===")
        print("  1. for text input")
        print("  2. for image input")
        print("  3. for speech input")
        print("  Enter quit to Exit")
        
        while True:
            try:
                option = input("\nOption: ").strip()
                    
                if option == "1":
                    print("Enter your Question")
                    question = input("\nQuestion: ").strip()

                    if question:
                        print("\n" + "="*50)
                        response = self.process_text_input(question)
                        print("="*50)
                        print(f"Final Response: {response}")

                elif option == "2":
                    path = input("Image Path: ").strip()
                    
                    if path:
                        print("\n" + "="*50)
                        response = self.process_image_input(path)
                        print("="*50)
                        print(f"Final Response: {response}")

                elif option == "3":
                    path = input("Audio Path: ").strip()
                    
                    if path:
                        print("\n" + "="*50)
                        response = self.process_audio_input(path)
                        print("="*50)
                        print(f"Final Response: {response}")

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