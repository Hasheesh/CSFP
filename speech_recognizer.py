
''' took the whisper transcribing usage from whisper example and documentation at https://huggingface.co/openai/whisper-tiny
    the rest of the code was all written by me '''
import soundfile as sf
import librosa
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from model_registery import ModelRegistery

class SpeechRecognizer:

    def __init__(self, model_type, model_name):
        self.speech_rec = None, 
        self.processsor = None
        self.model_reg = ModelRegistery()
        self.model_type = model_type
        self.model_name = model_name
        self.first_load = True       
        
    def load(self):
        model_path = self.model_reg.get_model_path(self.model_type, self.model_name)
        self.processor = WhisperProcessor.from_pretrained(model_path)
        self.speech_rec     = WhisperForConditionalGeneration.from_pretrained(model_path)
        print(f"Successfully loaded {self.model_name}")

    def process_input(self, audio_path):
        # Read audio, resampling if needed
        if self.first_load:
            self.load() 
        self.first_load = not self.first_load

        audio, sr = sf.read(audio_path)

        if sr != self.processor.feature_extractor.sampling_rate:
            audio = librosa.resample(audio, orig_sr=sr, 
                                    target_sr=self.processor.feature_extractor.sampling_rate)

            sr = self.processor.feature_extractor.sampling_rate

        # 3. Preprocess → log-mel spectrogram features
        input = self.processor.feature_extractor(audio, sampling_rate=sr, 
                                            return_tensors="pt")
        input_features = input.input_features.to(self.speech_rec.device) 

        # 4. Generate token IDs (greedy or with beam search)
        #    beam_size=5 for better quality; omit num_beams for greedy.
        generated_ids = self.speech_rec.generate(
            input_features, 
            max_length=2000,
            num_beams=5,
            task="transcribe",
            # language=lang,
            early_stopping=True
        )

        # 5. Decode to text
        transcription = self.processor.decode(generated_ids[0], skip_special_tokens=True)
        return transcription
