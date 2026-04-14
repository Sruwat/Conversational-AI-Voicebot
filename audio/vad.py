import collections
import logging
import queue
import time

import numpy as np


class VoiceActivityDetector:
    def __init__(self, provider="auto", energy_threshold=0.01, auto_calibrate=True, samplerate=16000):
        self.provider = provider
        self.energy_threshold = energy_threshold
        self.auto_calibrate = auto_calibrate
        self.samplerate = samplerate
        self.backend_name = "energy"
        self.silero_model = None
        self._load_backend()

    def _load_backend(self):
        if self.provider not in {"auto", "silero"}:
            return
        try:
            from silero_vad import get_speech_timestamps, load_silero_vad

            self.silero_model = load_silero_vad()
            self._get_speech_timestamps = get_speech_timestamps
            self.backend_name = "silero"
            logging.info("[VAD] Using Silero VAD backend.")
        except Exception as exc:
            logging.warning("[VAD] Silero VAD unavailable, using energy threshold: %s", exc)

    def calibrate(self, audio):
        if audio is None or len(audio) == 0:
            return
        baseline = float(np.sqrt(np.mean(np.square(audio)))) if len(audio) else self.energy_threshold
        self.energy_threshold = max(0.008, baseline * 2.2)
        logging.info("[VAD] Calibrated energy threshold to %.5f", self.energy_threshold)

    def is_speech(self, audio):
        if audio is None or len(audio) == 0:
            return False

        audio = np.asarray(audio, dtype=np.float32)
        if self.backend_name == "silero" and self.silero_model is not None:
            try:
                import torch

                timestamps = self._get_speech_timestamps(torch.from_numpy(audio), self.silero_model, sampling_rate=self.samplerate)
                return bool(timestamps)
            except Exception as exc:
                logging.warning("[VAD] Silero inference failed, falling back to energy VAD: %s", exc)

        energy = np.sqrt(np.mean(np.square(audio)))
        return bool(energy > self.energy_threshold)


class WakeWordDetector:
    def __init__(
        self,
        vad,
        wake_word="hey assistant",
        provider="auto",
        samplerate=16000,
        chunk_duration=0.5,
        buffer_chunks=6,
        model_path="",
    ):
        self.vad = vad
        self.wake_word = wake_word.lower().strip()
        self.provider = provider
        self.samplerate = samplerate
        self.chunk_duration = chunk_duration
        self.buffer_chunks = buffer_chunks
        self.buffer = collections.deque(maxlen=buffer_chunks)
        self.detector = None
        self.backend_name = "phrase"
        self.model_path = model_path
        self._load_backend()

    def _load_backend(self):
        if self.provider not in {"auto", "openwakeword"}:
            return
        try:
            from openwakeword.model import Model

            kwargs = {}
            if self.model_path:
                kwargs["wakeword_models"] = [self.model_path]
            kwargs["inference_framework"] = "onnx"
            self.detector = Model(**kwargs)
            self.backend_name = "openwakeword"
            logging.info("[WakeWord] Using openWakeWord backend.")
        except Exception as exc:
            logging.warning("[WakeWord] openWakeWord unavailable, using phrase detection: %s", exc)

    def listen_for_wake_word(self, stt, source_queue, timeout=60):
        start_time = time.time()
        self.buffer.clear()
        while time.time() - start_time < timeout:
            try:
                chunk = source_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            if not self.vad.is_speech(chunk):
                self.buffer.clear()
                continue

            if self.backend_name == "openwakeword" and self.detector is not None:
                try:
                    scores = self.detector.predict(chunk)
                    if scores and max(scores.values()) >= 0.5:
                        logging.info("[WakeWord] openWakeWord activation detected.")
                        return True
                except Exception as exc:
                    logging.warning("[WakeWord] openWakeWord inference failed: %s", exc)

            self.buffer.append(chunk)
            if len(self.buffer) < self.buffer_chunks:
                continue

            audio = np.concatenate(list(self.buffer))
            text = stt.transcribe(audio, sample_rate=self.samplerate)
            if text and self.wake_word in text.lower():
                logging.info("[WakeWord] Phrase wake word detected.")
                return True
        return False
