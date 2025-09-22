"""
speech_recognizer.py

This file handles the speech recognition functionality for the AI tutor.
It uses the faster-whisper model to transcribe audio to text.

The parts of code not written by me are referenced from the following sources:
- Code to use faster-whisper model from https://github.com/SYSTRAN/faster-whisper

"""
from faster_whisper import WhisperModel


class SpeechRecognizer:
    """Handles speech-to-text transcription using the faster-whisper model."""

    def __init__(self, model_path):
        """Initializes the SpeechRecognizer with the model path and configuration."""
        self.model_path = model_path
        self.device = "cpu"
        self.compute_type = "int8"
        self.num_threads = 4
        self.beam_size = 1
        self.vad_filter = True

        self.model = None
        self.first_load = True
        self.active_lang = None

    def load(self):
        """Loads the faster-whisper model into memory."""
        self.model = WhisperModel(
            self.model_path,
            device=self.device,
            compute_type=self.compute_type,
            cpu_threads=self.num_threads
        )

    def process_input(self, audio_path, lang):
        """Transcribes an audio file to text in the specified language."""
        if self.first_load:
            self.load()
            self.first_load = False

        self.active_lang = lang

        # Transcribe audio to text
        segments, info = self.model.transcribe(
            audio_path,
            language=lang,
            task="transcribe",
            beam_size=self.beam_size,
            vad_filter=self.vad_filter
        )

        text_chunks = [seg.text.strip() for seg in segments]
        transcription = " ".join(t for t in text_chunks if t)
        # print(transcription)
        return transcription
