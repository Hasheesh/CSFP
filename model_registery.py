MODEL_REGISTRY = {
    "llm": {
        "phi-4-mini": "models/llm/phi-4-mini/Phi-4-mini.gguf",
        "gemma-2-2b": "models/llm/gemma-2-2b",
        "gemma-3-4b": "models/llm/gemma-3-4b"
    },
    "stt": {
        "whisper-small": "models/stt/whisper-small"
    },
    "tts": {
        "piper-tts-en": "models/tts/piper-tts/en_US-lessac-medium.onnx",
        "mms-tts-ur": "models/tts/mss-tts-urd-script-arabic"
    },
    "ocr": {
        "PP-OCRv5_mobile_det": "models/ocr/PP-OCRv5_mobile_det_infer",
        "PP-OCRv5_mobile_rec": "models/ocr/PP-OCRv5_mobile_rec_infer",
        "trocr-handwritten": "models/ocr/trocr-handwritten/trocr-handwritten.gguf"
    },
    "translation": {
        "opus-en-ur": "models/translation/opus-mt-en-ur/opus-mt-en-ur.pt",
        "opus-ur-en": "models/translation/opus-mt-ur-en"
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

