MODEL_REGISTRY = {
    "llm": {
        "phi4-mini": "/home/gulabo/Desktop/CSFP/models/llm/phi-4-mini/Phi-4-mini.gguf",
    },
    "stt": {
        "whisper-small": "/home/gulabo/Desktop/CSFP/models/stt/whisper-small/whisper-small.gguf"
    },
    "tts": {
        "piper-english": "/home/gulabo/Desktop/CSFP/models/tts/piper/en",
        "piper-urdu": "/home/gulabo/Desktop/CSFP/models/tts/piper/ur"
    },
    "ocr": {
        "trocr-printed": "/home/gulabo/Desktop/CSFP/models/ocr/trocr-printed/trocr-printed.gguf",
        "trocr-handwritten": "/home/gulabo/Desktop/CSFP/models/ocr/trocr-handwritten/trocr-handwritten.gguf"
    },
    "translation": {
        "opus-en-ur": "/home/gulabo/Desktop/CSFP/models/translation/opus-mt-en-ur/opus-mt-en-ur.pt"
    }
}

def get_model_path(model_type, model_name):
    return MODEL_REGISTRY[model_type][model_name]