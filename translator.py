from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from model_registery import ModelRegistery

class Translator:

    def __init__(self, model_type, model_name):
        self.translator = None
        self.model_reg = ModelRegistery()
        self.model_type = model_type
        self.model_name = model_name
        self.first_load = True   


    def load(self):
        model_path = self.model_reg.get_model_path(self.model_type, self.model_name)
        # Load model and tokenizer from local directory
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_path)

        # Create translation pipeline
        self.translator = pipeline(
            "translation", 
            model=model, 
            tokenizer=tokenizer
        )

    def process_input(self, text):
        if self.first_load:
            self.load() 
        self.first_load = not self.first_load
        result = self.translator(text)
        return result[0]['translation_text']

