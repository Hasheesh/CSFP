"""
test_ocr.py

This file tests the OCR models.
-tesseract ocr
-paddleocr

The code to import modules from absolute paths is referenced from https://www.geeksforgeeks.org/python/python-import-from-parent-directory/
"""
import os
import sys
import gc
import psutil
import time
import pandas as pd

# Add parent directory to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model_registry import ModelRegistry
from vision_processor import VisionProcessor as TOCRProcessor
from vision_processor_x86 import VisionProcessor as PPOCRProcessor


def proc_mem():
    """Gets and prints the max memory usage of the process."""
    gc.collect()
    process = psutil.Process()
    memory_info = process.memory_info()
    max_rss_mb = memory_info.rss / (1024 * 1024)
    print(f"Max memory used: {max_rss_mb:.1f} MiB")
    return max_rss_mb


if __name__ == "__main__":
    results = []
    
    model_reg = ModelRegistry()
    start_time = time.time()

    tesseract_processor = TOCRProcessor()
    
    # # English Tesseract
    # start_time = time.time()
    # ocr_text = tesseract_processor.process_input('en_ocr_test.png', 'en')
    # time_taken = time.time() - start_time
    # mem_used = proc_mem()
    # ocr_text = ocr_text.replace('\n', ' ')
    # results.append({
    #     "model type": "ocr",
    #     "model name": "tesseract-ocr-en",
    #     "time taken": f"{time_taken:.2f}",
    #     "Memory used (MiB)": f"{mem_used:.2f}",
    #     "input image": "en_ocr_test.png",
    #     "language": "en",
    #     "output": ocr_text.strip(),
    # })
    # print(mem_used)
    
    # # Urdu Tesseract
    # start_time = time.time()
    # ocr_text = tesseract_processor.process_input('ur_ocr_test.png', 'ur')
    # time_taken = time.time() - start_time
    # mem_used = proc_mem()
    # ocr_text = ocr_text.replace('\n', ' ')
    # results.append({
    #     "model type": "ocr",
    #     "model name": "tesseract-ocr-ur",
    #     "time taken": f"{time_taken:.2f}",
    #     "Memory used (MiB)": f"{mem_used:.2f}",
    #     "input image": "ur_ocr_test.png",
    #     "language": "ur",
    #     "output": ocr_text.strip(),
    # })
    # print(mem_used)

    
    # # English PaddleOCR
    # ocr_en_det_path = model_reg.get_model_path('ocr', 'PP-OCRv5_mobile_det')
    # ocr_en_det_model_name = 'PP-OCRv5_mobile_det'
    # ocr_en_rec_path = model_reg.get_model_path('ocr', 'en_PP-OCRv5_mobile_rec')
    # ocr_en_rec_model_name = 'en_PP-OCRv5_mobile_rec'
    
    # paddle_processor_en = PPOCRProcessor(
    #     ocr_en_det_path, ocr_en_rec_path, 
    #     ocr_en_det_model_name, ocr_en_rec_model_name
    # )
    # start_time = time.time()
    # ocr_text = paddle_processor_en.process_input('en_ocr_test.png', 'en')
    # time_taken = time.time() - start_time
    # mem_used = proc_mem()
    # ocr_text = ocr_text.replace('\n', ' ')
    # results.append({
    #     "model type": "ocr",
    #     "model name": "en_PP-OCRv5_mobile",
    #     "time taken": f"{time_taken:.2f}",
    #     "Memory used (MiB)": f"{mem_used:.2f}",
    #     "input image": "en_ocr_test.png",
    #     "language": "en",
    #     "output": ocr_text.strip(),
    # })
    # print(mem_used)

    # # Urdu PaddleOCR
    # ocr_ur_det_path = model_reg.get_model_path('ocr', 'PP-OCRv3_mobile_det')
    # ocr_ur_det_model_name = 'PP-OCRv3_mobile_det'
    # ocr_ur_rec_path = model_reg.get_model_path('ocr', 'arabic_PP-OCRv3_mobile_rec')
    # ocr_ur_rec_model_name = 'arabic_PP-OCRv3_mobile_rec'
    
    # paddle_processor_ur = PPOCRProcessor(
    #     ocr_ur_det_path, ocr_ur_rec_path,
    #     ocr_ur_det_model_name, ocr_ur_rec_model_name
    # )
    # start_time = time.time()
    # ocr_text = paddle_processor_ur.process_input('ur_ocr_test.png', 'ur')
    # time_taken = time.time() - start_time
    # mem_used = proc_mem()
    # ocr_text = ocr_text.replace('\n', ' ')
    # results.append({
    #     "model type": "ocr",
    #     "model name": "arabic_PP-OCRv3_mobile",
    #     "time taken": f"{time_taken:.2f}",
    #     "Memory used (MiB)": f"{mem_used:.2f}",
    #     "input image": "ur_ocr_test.png",
    #     "language": "ur",
    #     "output": ocr_text.strip(),
    # })
    # print(mem_used)

    
    df = pd.DataFrame(results)
    file_exists = os.path.isfile("model_tests/test_outputs/ocr_stats.csv")
    df.to_csv("model_tests/test_outputs/ocr_stats.csv", mode='a', header=not file_exists, index=False)
    print("\ndata saved to csv")