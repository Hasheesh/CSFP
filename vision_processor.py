''' the code for fixing the urdu text order was taken from https://github.com/PaddlePaddle/PaddleOCR/discussions/14947
    the rest of the cide was written by me
'''
import os
import cv2
import json
import time
import gc
from paddleocr import PaddleOCR
from model_registery import ModelRegistery
from bidi.algorithm import get_display
import urduhack

def fix_urdu_ocr(text):
    n_text = urduhack.normalization.normalize(text)
    n_text = get_display(n_text)
    return n_text



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
        for res in result:  
            # res.print()  
            # res.save_to_img("output")  
            # res.save_to_json("output")
            json_result = res.json
            res_data = json_result['res']
            for text in res_data['rec_texts']:
                extracted_text = extracted_text + "\n " + text
        
        return extracted_text
          
    def unload(self):
        if self.ocr is not None:
            self.ocr = None
        self.active_lang = None
        gc.collect()

        
# # Example usage and testing
# if __name__ == "__main__":
#     model_reg = ModelRegistery()
#     ocr_en_det_path = model_reg.get_model_path('ocr', 'PP-OCRv5_mobile_det')
#     ocr_en_det_model_name = 'PP-OCRv5_mobile_det'
#     ocr_en_rec_path = model_reg.get_model_path('ocr', 'PP-OCRv5_mobile_rec')
#     ocr_en_rec_model_name = 'PP-OCRv5_mobile_rec'
#     vis_pro_en = VisionProcessor(ocr_en_det_path, ocr_en_rec_path, ocr_en_det_model_name, ocr_en_rec_model_name)
#     ocr_ur_det_path = model_reg.get_model_path('ocr', 'PP-OCRv3_mobile_det')
#     ocr_ur_det_name = 'PP-OCRv3_mobile_det'
#     ocr_ur_rec_path = model_reg.get_model_path('ocr', 'arabic_PP-OCRv3_mobile_rec')
#     ocr_ur_rec_model_name = 'arabic_PP-OCRv3_mobile_rec'
#     vis_pro_ur = VisionProcessor(ocr_ur_det_path, ocr_ur_rec_path, ocr_ur_det_name, ocr_ur_rec_model_name)
#     ocr_text = vis_pro_ur.process_input('ur.jpg','ur')
#     en_text = vis_pro_en.process_input('img.png', 'en')

    
#     nr_text = fix_urdu_ocr(ocr_text)
#     with open("urdu.txt", 'w') as f:
#         f.write(ocr_text + "\n")
#         f.write(nr_text + "\n")
