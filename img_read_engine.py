import os
import cv2
import json
import time

from paddleocr import PaddleOCR

class ImageReader:
    def __init__(self):
        self.ocr = None
        self._load_ocr_model()
        
    def _load_ocr_model(self):
        """Initialize and load the PaddleOCR model with PP-OCRv5_mobile models"""
        self.ocr = PaddleOCR(
            text_detection_model_name="PP-OCRv5_mobile_det",
            text_detection_model_dir="./models/img_reader/PP-OCRv5_mobile_det_infer",
            text_recognition_model_name="PP-OCRv5_mobile_rec",
            text_recognition_model_dir="./models/img_reader/PP-OCRv5_mobile_rec_infer",
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
        )
    
    def process_input(self, image_path):
        """Process an image and extract text from it - no preprocessing"""
        result = self.ocr.predict(image_path)
        extracted_text = ""
        for res in result:  
            # res.print()  
            # res.save_to_img("output")  
            # res.save_to_json("output")
            json_result = res.json
            res_data = json_result['res']
            for text in res_data['rec_texts']:
                extracted_text = extracted_text + text
        return extracted_text        

# Example usage and testing
# if __name__ == "__main__":
#     reader = ImageReader()
#     reader.process_input("./img.png")
   


