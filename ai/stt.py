import logging
import os
import tempfile

import soundfile as sf


class SpeechToText:
    def __init__(self, provider="auto", model_name="base"):
        self.provider = provider
        self.model_name = model_name
        self.backend_name = None
        self.model = None
        self._load_backend()

    def _load_backend(self):
        if self.provider in {"auto", "faster_whisper"}:
            try:
                from faster_whisper import WhisperModel

                self.model = WhisperModel(self.model_name, device="cpu", compute_type="int8")
                self.backend_name = "faster_whisper"
                logging.info("[STT] Using faster-whisper backend.")
                return
            except Exception as exc:
                logging.warning("[STT] faster-whisper unavailable: %s", exc)

        if self.provider in {"auto", "whisper"}:
            try:
                import whisper

                self.model = whisper.load_model(self.model_name)
                self.backend_name = "whisper"
                logging.info("[STT] Using openai-whisper backend.")
                return
            except Exception as exc:
                logging.warning("[STT] whisper unavailable: %s", exc)

        self.backend_name = "none"
        logging.error("[STT] No speech-to-text backend is available.")

    def transcribe(self, audio, sample_rate=16000):
        if audio is None or len(audio) == 0:
            return ""
        if self.backend_name == "none":
            return ""

        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        text = ""
        try:
            sf.write(tmp.name, audio, sample_rate)
            tmp.close()

            if self.backend_name == "faster_whisper":
                segments, _ = self.model.transcribe(tmp.name, vad_filter=True, beam_size=1)
                text = " ".join(segment.text.strip() for segment in segments).strip()
            elif self.backend_name == "whisper":
                result = self.model.transcribe(tmp.name, fp16=False)
                text = (result or {}).get("text", "").strip()
        except Exception as exc:
            logging.error("[STT] Transcription failed: %s", exc)
            text = ""
        finally:
            try:
                os.remove(tmp.name)
            except Exception as exc:
                logging.warning("[STT] Failed to remove temp file: %s", exc)

        return " ".join(text.split()).strip()
