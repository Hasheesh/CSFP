MODEL_REGISTRY = {
    "llm": {
        "phi-4-mini": "models/llm/phi-4-mini/Phi-4-mini.gguf",
        "phi-3-mini": "models/llm/Phi-3-mini-4k-instruct-Q4_K_M.gguf",
        "phi-3.1-mini": "models/llm/Phi-3.1-mini-4k-instruct-Q4_K_M.gguf",
        "gemma-2-2b": "models/llm/gemma-2-2b/gemma-2-2b.gguf",
        "gemma-3-4b": "models/llm/gemma-3-4b/google_gemma-3-4b.gguf",
        "gemma-3n-4b": "models/llm/gemma-3n-4b/google_gemma-3n-E4B-it-Q4_K_M.gguf",
        "gemma-3n-4b-q4": "models/llm/google_gemma-3n-E2B-it-Q4_K_M.gguf",
        "qwen-2.5-1.5b-math": "models/llm/qwen-2.5-1.5b-math-it/Qwen2.5-Math-1.5B-Instruct-Q4_K_M.gguf",
        "qwen-3-4b-q5": "models/llm/Qwen_Qwen3-4B-Instruct-2507-Q5_K_M.gguf",
        "qwen-3-4b-q4": "models/llm/Qwen_Qwen3-4B-Instruct-2507-Q4_K_M.gguf",
        "qwen-2.5-3b-q4": "models/llm/Qwen2.5-3B-Instruct-Q4_K_M.gguf"


    },
    "stt": {
        "whisper-small": "models/stt/whisper-small",
        "faster-whisper-small": "models/stt/faster-whisper-small" # ctranslate2 based version of whisper-small
    },
    "tts": {
        "piper-tts-en-lessac": "models/tts/piper-tts-en-lessac/en_US-lessac-medium.onnx",
        "piper-tts-en-amy": "models/tts/piper-tts-en-amy/en_US-amy-medium.onnx",
        "mms-tts-ur": "models/tts/mms-tts-urd-script-arabic"
    },
    "ocr": {
        "PP-OCRv5_mobile_det": "models/ocr/PP-OCRv5_mobile_det_infer",
        "PP-OCRv5_mobile_rec": "models/ocr/PP-OCRv5_mobile_rec_infer",
        "PP-OCRv3_mobile_det": "models/ocr/PP-OCRv3_mobile_det_infer",
        "arabic_PP-OCRv3_mobile_rec": "models/ocr/arabic_PP-OCRv3_mobile_rec_infer"
    },
    "translation": {
        "opus-mt-en-ur": "models/translation/opus-mt-en-ur",
        "opus-mt-ur-en": "models/translation/opus-mt-ur-en",
        "ct2-opus-mt-en-ur": "models/translation/ct2-opus-mt-en-ur", # ctranslate2 format of opus-mt-en-ur
        "nllb-200-600M-Q8": "models/translation/nllb-200-distilled-600M-Q8", # nllb-200 ctranslate2 format with int8 quantization
        "nllb-200-600M-ct2": "models/translation/nllb-200-distilled-600M-ct2", # non quantized version of nllb 200 converted to run on ctranslate2
        "m2m100-418m": "models/translation/ct2fast-m2m100-418M",

    }
}

class ModelRegistery:
    def __init__(self):
        self.registry = MODEL_REGISTRY

    def get_model_path(self, model_type, model_name):
        return self.registry[model_type][model_name]        
    
    # def get_model_path(self, model_name):
    #     """
    #     Get model path by model name. For TTS models, assumes the model_name 
    #     corresponds to a TTS model in the registry.
    #     """
    #     if model_name in self.registry["tts"]:
    #         return self.registry["tts"][model_name]
    #     else:
    #         raise ValueError(f"Model '{model_name}' not found in TTS registry")
    
    # def get_model_path_by_type(self, model_type, model_name):
    #     """
    #     Get model path by model type and model name.
    #     """
    #     if model_type in self.registry and model_name in self.registry[model_type]:
    #         return self.registry[model_type][model_name]
    #     else:
    #         raise ValueError(f"Model '{model_name}' not found in {model_type} registry")

