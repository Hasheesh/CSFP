
''' took the whisper transcribing usage from whisper example and documentation at https://huggingface.co/openai/whisper-tiny
    the rest of the code was all written by me '''
import soundfile as sf
import librosa
from transformers import WhisperProcessor, WhisperForConditionalGeneration


model_name = "models/stt/whisper-small"
processor = WhisperProcessor.from_pretrained(model_name)
model     = WhisperForConditionalGeneration.from_pretrained(model_name)

# 2. Read your WAV, resampling if needed
# wav_path = "test.wav"
wav_path = "question.wav"

audio, sr = sf.read(wav_path)                    # load audio
if sr != processor.feature_extractor.sampling_rate:
    audio = librosa.resample(audio, orig_sr=sr, 
                             target_sr=processor.feature_extractor.sampling_rate)
    sr = processor.feature_extractor.sampling_rate

# 3. Preprocess → log-mel spectrogram features
inputs = processor.feature_extractor(audio, sampling_rate=sr, 
                                     return_tensors="pt")
input_features = inputs.input_features.to(model.device)  # shape (1,seq_len,80)

# 4. Generate token IDs (greedy or with beam search)
#    beam_size=5 for better quality; omit num_beams for greedy.
generated_ids = model.generate(
    input_features, 
    max_length=2000,
    num_beams=5,
    task="transcribe",
    language="ur",
    early_stopping=True
)

# 5. Decode to text
transcription = processor.decode(generated_ids[0], skip_special_tokens=True)
print(transcription)



# import soundfile as sf
# import torch
# from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
# import argparse

# # load pretrained model
# processor = Wav2Vec2Processor.from_pretrained("addy88/wav2vec2-urdu-stt")
# model = Wav2Vec2ForCTC.from_pretrained("addy88/wav2vec2-urdu-stt")
# # load audio
# audio_input, sample_rate = sf.read('output.wav')
# # pad input values and return pt tensor
# input_values = processor(audio_input, sampling_rate=sample_rate, return_tensors="pt").input_values
# # INFERENCE
# # retrieve logits & take argmax
# logits = model(input_values).logits
# predicted_ids = torch.argmax(logits, dim=-1)
# # transcribe
# transcription = processor.decode(predicted_ids[0], skip_special_tokens=True)
# print(transcription)
