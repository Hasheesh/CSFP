import os
import cv2
import json
import time
import gc
from paddleocr import PaddleOCR
from model_registery import ModelRegistery


class VisionProcessor:
    def __init__(self, det_model_path, rec_model_path, det_model_name, rec_model_name):
        self.ocr = None
        self.det_model_name = det_model_name
        self.rec_model_name = rec_model_name
        self.det_model_path = det_model_path
        self.rec_model_path = rec_model_path
        self.first_load = True
        self.active_lang = None
    def load(self, lang):
        """Initialize and load the PaddleOCR model with PP-OCRv5_mobile models"""
        if self.ocr is not None and self.active_lang == lang:
            return  # already correct model
        self.unload()

        # if lang == "en":
        self.ocr = PaddleOCR(
            lang="en",
            text_detection_model_name=self.det_model_name,
            text_detection_model_dir=self.det_model_path,
            text_recognition_model_name=self.rec_model_name,
            text_recognition_model_dir=self.rec_model_path,
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
        )

        self.active_lang = lang
    
    def process_input(self, image_path, lang):
        self.load(lang)
        

        """Process an image and extract text from it - no preprocessing"""
        result = self.ocr.predict(image_path)
        extracted_text = ""
        all_text = []
        for res in result:  
            # res.print()  
            # res.save_to_img("output")  
            # res.save_to_json("output")
            json_result = res.json
            res_data = json_result['res']
            for text in res_data['rec_texts']:
                extracted_text = extracted_text + " " + text
        print("EXTRACTED TEXT:  " + extracted_text)
        return extracted_text  
          
    def unload(self):
        """Free the currently loaded OCR (if any)."""
        if self.ocr is not None:
            self.ocr = None
        self.active_lang = None
        
        gc.collect()

        
# # Example usage and testing
# if __name__ == "__main__":
#     reader = VisionProcessor()
#     reader.process_input("ur.jpg", 'ur')
   


