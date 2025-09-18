"""
vision_processor.py

This file handles the OCR functionality for the AI tutor.
It uses the PaddleOCR model to extract text from images.

The parts of code not written by me are referenced from the following sources:
- Code to use PaddleOCR model from https://github.com/PaddlePaddle/PaddleOCR
- Code to fix the urdu text order from https://github.com/PaddlePaddle/PaddleOCR/discussions/14947

The rest of the code was written by me and is based on the paddleocr documentation.
"""
from paddleocr import PaddleOCR
import urduhack

class VisionProcessor:
    """Extracts text from images using PaddleOCR."""

    def __init__(self, det_model_path, rec_model_path, det_model_name, rec_model_name):
        """Initializes the VisionProcessor with model paths and names."""
        self.ocr = None
        self.det_model_name = det_model_name
        self.rec_model_name = rec_model_name
        self.det_model_path = det_model_path
        self.rec_model_path = rec_model_path
        self.active_lang = None

    def load(self, lang):
        """Initializes and loads the PaddleOCR model based on the language."""
        if self.ocr is not None and self.active_lang == lang:
            return  # already correct model

        if lang == "en":
            self.ocr = PaddleOCR(
                text_detection_model_name=self.det_model_name,
                text_detection_model_dir=self.det_model_path,
                text_recognition_model_name=self.rec_model_name,
                text_recognition_model_dir=self.rec_model_path,
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=False,
            )
        elif lang == "ur":
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
        """Processes an image and extracts text from it."""
        self.load(lang)
        
        # get raw OCR prediction from the model
        result = self.ocr.predict(image_path)
        
        # extract and combine all text pieces from the result
        extracted_text = ""
        for res in result:  
            json_result = res.json
            res_data = json_result['res']
            for text in res_data['rec_texts']:
                extracted_text = extracted_text + "\n " + text

        # normalize Urdu text to fix script issues
        if lang == "ur":
            fixed_text = urduhack.normalization.normalize(extracted_text)
        else:
            fixed_text = extracted_text
        return fixed_text
