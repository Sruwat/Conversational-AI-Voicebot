import base64
import queue
import threading
import time

import numpy as np


def pcm16_to_float32(payload):
    if not payload:
        return np.array([], dtype=np.float32)
    audio = np.frombuffer(payload, dtype=np.int16).astype(np.float32) / 32768.0
    return audio


class RealtimeConversation:
    def __init__(
        self,
        session,
        speaker,
        vad,
        chunk_ms=100,
        sample_rate=16000,
        silence_ms=600,
        partial_interval_ms=700,
    ):
        self.session = session
        self.speaker = speaker
        self.vad = vad
        self.chunk_ms = int(chunk_ms)
        self.sample_rate = sample_rate
        self.silence_ms = int(silence_ms)
        self.partial_interval_ms = int(partial_interval_ms)
        self.events = queue.Queue()
        self.stop_event = threading.Event()
        self._speech_frames = []
        self._silence_accumulator_ms = 0
        self._last_partial_at = 0.0
        self._response_thread = None
        self._turn_lock = threading.Lock()
        self._playback_started_at = 0.0
        self._interrupt_speech_ms = 0
        self._assistant_audio_active = False
        self._assistant_chunk_buffer = ""
        self._settings = {
            "silence_ms": self.silence_ms,
            "partial_interval_ms": self.partial_interval_ms,
            "interrupt_threshold_ms": 200,
            "interrupt_energy_multiplier": 1.8,
            "ignore_playback_ms": 300,
        }
        self._speech_detected_ms = 0
        self._max_utterance_ms = 8000
        self._speech_floor = 0.0015
        self._speech_energy_streak = 0
        self._speech_start_threshold_ms = 80

    def start(self):
        self.emit("status", state=self.session.controller.get_state().value)
        self.emit("ready", session_id=self.session.session_id)

    def stop(self):
        self.stop_event.set()

    def configure(self, settings):
        if "silence_ms" in settings:
            self.silence_ms = max(250, int(settings["silence_ms"]))
            self._settings["silence_ms"] = self.silence_ms
        if "partial_interval_ms" in settings:
            self.partial_interval_ms = max(300, int(settings["partial_interval_ms"]))
            self._settings["partial_interval_ms"] = self.partial_interval_ms
        if "interrupt_threshold_ms" in settings:
            self._settings["interrupt_threshold_ms"] = max(120, int(settings["interrupt_threshold_ms"]))
        if "interrupt_energy_multiplier" in settings:
            self._settings["interrupt_energy_multiplier"] = max(1.1, float(settings["interrupt_energy_multiplier"]))
        if "ignore_playback_ms" in settings:
            self._settings["ignore_playback_ms"] = max(0, int(settings["ignore_playback_ms"]))
        self.emit("settings", **self._settings)

    def mark_playback_finished(self):
        self._assistant_audio_active = False
        if self.session.controller.get_state().value != "thinking":
            self.session.controller.finish_speaking()
            self.emit("status", state=self.session.controller.get_state().value)

    def flush_current_turn(self):
        if not self._speech_frames:
            return
        utterance = np.concatenate(self._speech_frames).astype(np.float32)
        self._flush_utterance(utterance)

    def handle_audio_chunk(self, payload):
        audio = pcm16_to_float32(payload)
        if audio.size == 0 or self.stop_event.is_set():
            return

        self.session.touch()
        self._handle_possible_interrupt(audio)

        energy = float(np.sqrt(np.mean(np.square(audio)))) if len(audio) else 0.0
        vad_hit = self.vad.is_speech(audio)
        soft_threshold = max(self.vad.energy_threshold * 0.55, self._speech_floor)
        if energy >= soft_threshold:
            self._speech_energy_streak += self.chunk_ms
        else:
            self._speech_energy_streak = max(0, self._speech_energy_streak - self.chunk_ms)
        speech_hit = vad_hit or self._speech_energy_streak >= self._speech_start_threshold_ms

        if speech_hit:
            self._speech_frames.append(audio)
            self._silence_accumulator_ms = 0
            self._speech_detected_ms += self.chunk_ms
            self._maybe_emit_partial()
        elif self._speech_frames:
            self._speech_frames.append(audio)
            self._silence_accumulator_ms += self.chunk_ms
            if self._silence_accumulator_ms >= self.silence_ms:
                utterance = np.concatenate(self._speech_frames).astype(np.float32)
                self._flush_utterance(utterance)

        if self._speech_frames and self._speech_detected_ms >= self._max_utterance_ms:
            utterance = np.concatenate(self._speech_frames).astype(np.float32)
            self._flush_utterance(utterance)

    def _handle_possible_interrupt(self, audio):
        controller = self.session.controller
        if controller.get_state().value != "speaking":
            self._interrupt_speech_ms = 0
            return

        if (time.time() - self._playback_started_at) * 1000 < self._settings["ignore_playback_ms"]:
            self._interrupt_speech_ms = 0
            return

        energy = float(np.sqrt(np.mean(np.square(audio)))) if len(audio) else 0.0
        raised_threshold = self.vad.energy_threshold * self._settings["interrupt_energy_multiplier"]
        if self.vad.is_speech(audio) and energy >= raised_threshold:
            self._interrupt_speech_ms += self.chunk_ms
        else:
            self._interrupt_speech_ms = max(0, self._interrupt_speech_ms - self.chunk_ms)

        if self._interrupt_speech_ms >= self._settings["interrupt_threshold_ms"]:
            controller.interrupt()
            self._assistant_audio_active = False
            self.emit("interrupt", reason="user_speech")
            self.emit("status", state=controller.get_state().value)
            controller.reset_after_interrupt()
            self.emit("status", state=controller.get_state().value)
            self._interrupt_speech_ms = 0

    def _maybe_emit_partial(self):
        now = time.time()
        if (now - self._last_partial_at) * 1000 < self.partial_interval_ms:
            return
        if not self._speech_frames:
            return

        recent_audio = np.concatenate(self._speech_frames[-max(1, 2000 // self.chunk_ms) :]).astype(np.float32)
        transcript = self.session.stt.transcribe(recent_audio, sample_rate=self.sample_rate)
        if transcript:
            self.emit("partial_transcript", text=transcript)
        self._last_partial_at = now

    def _start_turn(self, utterance):
        if self._response_thread and self._response_thread.is_alive():
            return

        self._response_thread = threading.Thread(target=self._run_turn, args=(utterance,), daemon=True)
        self._response_thread.start()

    def _flush_utterance(self, utterance):
        self._speech_frames = []
        self._silence_accumulator_ms = 0
        self._last_partial_at = 0.0
        self._speech_detected_ms = 0
        self._speech_energy_streak = 0
        self._start_turn(utterance)

    def _run_turn(self, utterance):
        with self._turn_lock:
            self.emit("status", state="thinking")
            self._assistant_chunk_buffer = ""
            transcript = self.session.stt.transcribe(utterance, sample_rate=self.sample_rate)
            if not transcript:
                self.emit("status", state="listening")
                return

            self.emit("final_transcript", text=transcript)
            chunks = []
            audio_started = False
            for chunk in self.session.controller.stream_response(transcript, commit=True):
                chunks.append(chunk)
                self.emit("assistant_chunk", text=chunk)
                for sentence in self._consume_ready_sentences(chunk):
                    if not audio_started:
                        self.session.controller.mark_speaking()
                        self._assistant_audio_active = True
                        self._playback_started_at = time.time()
                        self.emit("status", state="speaking")
                        audio_started = True
                    self._emit_audio_chunk(sentence)

            final_response = " ".join(" ".join(chunks).split()).strip()
            if not final_response:
                self.emit("status", state="listening")
                return

            self.emit("assistant_message", text=final_response)
            remaining = self._flush_sentence_buffer()
            if remaining:
                if not audio_started:
                    self.session.controller.mark_speaking()
                    self._assistant_audio_active = True
                    self._playback_started_at = time.time()
                    self.emit("status", state="speaking")
                    audio_started = True
                self._emit_audio_chunk(remaining)

            self.emit("assistant_audio_end")

            if not audio_started:
                self.session.controller.finish_speaking()
                self.emit("status", state=self.session.controller.get_state().value)
                self.emit("assistant_audio_unavailable", text=final_response)

    def emit(self, event_type, **payload):
        payload["type"] = event_type
        self.events.put(payload)

    def _consume_ready_sentences(self, chunk):
        self._assistant_chunk_buffer = f"{self._assistant_chunk_buffer} {chunk}".strip()
        ready = []
        buffer = self._assistant_chunk_buffer
        last_cut = -1
        for index, char in enumerate(buffer):
            if char in ".!?":
                last_cut = index
        if last_cut >= 0:
            completed = buffer[: last_cut + 1].strip()
            self._assistant_chunk_buffer = buffer[last_cut + 1 :].strip()
            for sentence in [item.strip() for item in completed.splitlines() if item.strip()]:
                if sentence:
                    ready.append(sentence)
        return ready

    def _flush_sentence_buffer(self):
        remaining = self._assistant_chunk_buffer.strip()
        self._assistant_chunk_buffer = ""
        return remaining

    def _emit_audio_chunk(self, text):
        audio_bytes = self.speaker.synthesize_bytes(text)
        if not audio_bytes:
            return
        encoded = base64.b64encode(audio_bytes).decode("ascii")
        self.emit("assistant_audio_chunk", audio_b64=encoded, mime_type="audio/wav", text=text)
