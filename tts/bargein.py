import logging
import queue
import threading
import time

import numpy as np


class BargeInMonitor:
    def __init__(
        self,
        vad,
        tts_speaker,
        source_queue,
        chunk_ms=100,
        min_interrupt_ms=200,
        ignore_playback_ms=300,
        speech_energy_multiplier=1.8,
    ):
        self.vad = vad
        self.tts_speaker = tts_speaker
        self.source_queue = source_queue
        self.chunk_ms = max(20, int(chunk_ms))
        self.min_interrupt_ms = max(100, int(min_interrupt_ms))
        self.ignore_playback_ms = max(0, int(ignore_playback_ms))
        self.speech_energy_multiplier = max(1.0, float(speech_energy_multiplier))
        self._thread = None
        self._stop_flag = threading.Event()
        self._barge_in_triggered = threading.Event()
        self._started_at = 0.0

    def start(self):
        self._stop_flag.clear()
        self._barge_in_triggered.clear()
        self._started_at = time.time()
        self._thread = threading.Thread(target=self._monitor, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop_flag.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1)

    def barge_in_callback(self):
        return self._barge_in_triggered.is_set()

    def _monitor(self):
        speech_ms = 0
        raised_threshold = self.vad.energy_threshold * self.speech_energy_multiplier
        while self.tts_speaker.is_speaking.is_set() and not self._stop_flag.is_set():
            try:
                chunk = self.source_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            if (time.time() - self._started_at) * 1000 < self.ignore_playback_ms:
                speech_ms = 0
                continue

            energy = float(np.sqrt(np.mean(np.square(chunk)))) if len(chunk) else 0.0
            vad_hit = self.vad.is_speech(chunk)
            energy_hit = energy >= raised_threshold

            if vad_hit and energy_hit:
                speech_ms += self.chunk_ms
            else:
                speech_ms = max(0, speech_ms - self.chunk_ms)

            if speech_ms >= self.min_interrupt_ms:
                logging.info(
                    "[BargeIn] User speech detected. energy=%.5f threshold=%.5f duration_ms=%s",
                    energy,
                    raised_threshold,
                    speech_ms,
                )
                self._barge_in_triggered.set()
                self.tts_speaker.stop()
                break

            time.sleep(0.01)
