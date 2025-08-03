from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

# Load model and tokenizer from local directory
model_dir = "models/translation/opus-mt-en-ur"  # Path to downloaded files

tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForSeq2SeqLM.from_pretrained(model_dir)

# Create translation pipeline
translator = pipeline(
    "translation", 
    model=model, 
    tokenizer=tokenizer
)

# Translate Urdu to English
urdu_text = "میں اپنا ہوم ورک کر رہا ہوں۔"
# urdu_text = "میں اپنا ہوم ورک کر رہا ہوں۔"
eng_text = "I am doing my work."

result = translator(eng_text)
print(result[0]['translation_text'])  # Output: "I am doing my homework."
print(urdu_text)


