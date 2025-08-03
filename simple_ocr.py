from paddleocr import PaddleOCR  
import json
# ocr = PaddleOCR(
#     use_doc_orientation_classify=False, # Disables document orientation classification model via this parameter
#     use_doc_unwarping=False, # Disables text image rectification model via this parameter
#     use_textline_orientation=False, # Disables text line orientation classification model via this parameter
# )
# ocr = PaddleOCR(lang="en") # Uses English model by specifying language parameter
# ocr = PaddleOCR(ocr_version="PP-OCRv4") # Uses other PP-OCR versions via version parameter
# ocr = PaddleOCR(device="gpu") # Enables GPU acceleration for model inference via device parameter
ocr = PaddleOCR(
    text_detection_model_name="PP-OCRv5_mobile_det",
    text_detection_model_dir="./models/ocr/PP-OCRv5_mobile_det_infer",
    text_recognition_model_name="PP-OCRv5_mobile_rec",
    text_recognition_model_dir="./models/ocr/PP-OCRv5_mobile_rec_infer",
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False,
) # Switch to PP-OCRv5_mobile models
result = ocr.predict("./img.png")

for res in result:  
    # res.print()  
    # res.save_to_img("output")  
    # res.save_to_json("output")
    json_result = res.json
    res_data = json_result['res']
    for text in res_data['rec_texts']:
            print(text)
    