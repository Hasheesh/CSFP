"""
vision_processor.py

This file handles the OCR functionality for the AI tutor.
It uses Tesseract for ARM architectures (like Raspberry Pi) and PaddleOCR for x86_64 architectures.

- Tesseract (via pytesseract) is used for its lightweight footprint on ARM.
- PaddleOCR is used for its performance and accuracy on x86_64 systems.

The parts of code not written by me are referenced from the following sources:
- Basic code to use PaddleOCR from https://github.com/PaddlePaddle/PaddleOCR
- Basic code to use pytesseract from https://github.com/madmaze/pytesseract
- Code to fix Urdu OCR from https://github.com/urduhack/urduhack
"""
import os
import pytesseract
from PIL import Image
import urduhack
from bidi.algorithm import get_display
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
from paddleocr import PaddleOCR


def fix_urdu_ocr(text):
    """Normalizes and reshapes Urdu text for correct display."""
    n_text = urduhack.normalization.normalize(text)
    n_text = get_display(n_text)
    return n_text

class VisionProcessor:
    """Extracts text from images using Tesseract for ARM and PaddleOCR for x86_64."""

    def __init__(self, version, det_model_path=None, rec_model_path=None, det_model_name=None, rec_model_name=None):
        """Initializes the VisionProcessor. Tesseract doesn't need model paths."""
        self.version = version
        self.active_lang = None
        self.ocr = None

        if self.version == 'x86_64':
            self.det_model_path = det_model_path
            self.rec_model_path = rec_model_path
            self.det_model_name = det_model_name
            self.rec_model_name = rec_model_name


    def load(self, lang):
        """Loads the appropriate OCR engine based on the architecture and language."""
        if self.version == 'arm':
            self.active_lang = lang
        else:
            if self.ocr is not None and self.active_lang == lang:
                return
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
        """Processes an image and extracts text from it using Tesseract or PaddleOCR."""
        self.load(lang)
        
        if self.version == 'arm':
            if lang == 'ur':
                src_lang = 'urd'
            else:
                src_lang = 'eng'
            
            image = Image.open(image_path)
            extracted_text = pytesseract.image_to_string(image, lang=src_lang)
            return extracted_text.strip()
        else:
            result = self.ocr.predict(image_path)
            extracted_lines = []

            for res in result:
                res_data = res.json['res']
                texts = res_data['rec_texts']

                if lang == 'ur':
                    line_text = " ".join(reversed(texts))
                    line_text = fix_urdu_ocr(line_text)
                else:
                    line_text = " ".join(texts)

                extracted_lines.append(line_text)

            return "\n".join(extracted_lines).strip()
            
if __name__ == "__main__":

    # Example for ARM version (Tesseract)
    vis_pro_en_arm = VisionProcessor(version='arm')
    ocr_text_en_arm = vis_pro_en_arm.process_input('model_tests/test_data/en_ocr_test.png', 'en')
    print(f"English OCR Result (ARM): {ocr_text_en_arm}")

    vis_pro_ur_arm = VisionProcessor(version='arm')
    ocr_text_ur_arm = vis_pro_ur_arm.process_input('model_tests/test_data/ur_ocr_test.png', 'ur')
    print(f"Urdu OCR Result (ARM): {ocr_text_ur_arm}")
    print(f"\nAvailable Tesseract languages: {pytesseract.get_languages()}")

    # Example for x86_64 version (PaddleOCR)
    from model_registry import ModelRegistry
    model_reg = ModelRegistry()
    
    ocr_en_det_path = model_reg.get_model_path('ocr', 'PP-OCRv5_mobile_det')
    ocr_en_rec_path = model_reg.get_model_path('ocr', 'en_PP-OCRv5_mobile_rec')
    ocr_en_det_model_name = 'PP-OCRv5_mobile_det'
    ocr_en_rec_model_name = 'en_PP-OCRv5_mobile_rec'
    
    vis_pro_en_x86 = VisionProcessor('x86_64', ocr_en_det_path, ocr_en_rec_path, ocr_en_det_model_name, ocr_en_rec_model_name)
    ocr_text_en_x86 = vis_pro_en_x86.process_input('model_tests/test_data/en_ocr_test.png', 'en')
    print(f"\nEnglish OCR Result (x86_64): {ocr_text_en_x86}")

    ocr_ur_det_path = model_reg.get_model_path('ocr', 'PP-OCRv3_mobile_det')
    ocr_ur_rec_path = model_reg.get_model_path('ocr', 'arabic_PP-OCRv3_mobile_rec')
    ocr_ur_det_model_name = 'PP-OCRv3_mobile_det'
    ocr_ur_rec_model_name = 'arabic_PP-OCRv3_mobile_rec'

    vis_pro_ur_x86 = VisionProcessor('x86_64', ocr_ur_det_path, ocr_ur_rec_path, ocr_ur_det_model_name, ocr_ur_rec_model_name)
    ocr_text_ur_x86 = vis_pro_ur_x86.process_input('model_tests/test_data/ur_ocr_test.png', 'ur')
    print(f"Urdu OCR Result (x86_64): {ocr_text_ur_x86}")