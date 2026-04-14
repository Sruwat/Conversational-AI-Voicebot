import io
import logging
import threading
import time
import uuid

import numpy as np
import soundfile as sf

from conversation import ConversationController


class ChatSession:
    def __init__(self, llm, stt, metrics, session_id=None, max_recent_messages=16):
        self.session_id = session_id or str(uuid.uuid4())
        self.llm = llm
        self.stt = stt
        self.metrics = metrics
        self.controller = ConversationController(llm=llm, max_recent_messages=max_recent_messages)
        self.created_at = time.time()
        self.last_access = self.created_at
        self._lock = threading.Lock()
        self.min_browser_audio_seconds = 0.35

    def touch(self):
        self.last_access = time.time()

    def chat(self, text):
        with self._lock:
            self.touch()
            self.metrics.increment("chat_requests")
            started = time.time()
            response = self.controller.get_response(text)
            self.metrics.observe_ms("chat_latency_ms", (time.time() - started) * 1000)
            return {"response": response, "state": self.controller.get_state().value}

    def stream_chat(self, text):
        with self._lock:
            self.touch()
            self.metrics.increment("chat_requests")
            self.metrics.increment("llm_streams")
            started = time.time()
            for chunk in self.controller.stream_response(text):
                yield chunk
            self.controller.set_state("listening")
            self.metrics.observe_ms("chat_latency_ms", (time.time() - started) * 1000)

    def transcribe_audio(self, raw_audio_bytes):
        with io.BytesIO(raw_audio_bytes) as buffer:
            audio, sample_rate = sf.read(buffer, dtype="float32")
        if getattr(audio, "ndim", 1) > 1:
            audio = audio[:, 0]
        audio = np.asarray(audio, dtype=np.float32)
        duration_seconds = float(len(audio)) / float(sample_rate or 1)
        return audio, sample_rate, duration_seconds, self.stt.transcribe(audio, sample_rate=sample_rate)

    def chat_from_audio(self, raw_audio_bytes):
        with self._lock:
            self.touch()
            self.metrics.increment("audio_requests")
            started = time.time()
            try:
                audio, sample_rate, duration_seconds, transcript = self.transcribe_audio(raw_audio_bytes)
            except Exception as exc:
                logging.exception("[API] Failed to decode browser audio")
                self.metrics.observe_ms("audio_latency_ms", (time.time() - started) * 1000)
                return {
                    "ok": False,
                    "error": "Audio upload could not be decoded.",
                    "transcript": "",
                    "response": "",
                    "state": self.controller.get_state().value,
                }

            logging.info(
                "[API] Browser audio bytes=%s sample_rate=%s duration_s=%.2f stt_backend=%s",
                len(raw_audio_bytes),
                sample_rate,
                duration_seconds,
                self.stt.backend_name,
            )

            if duration_seconds < self.min_browser_audio_seconds:
                self.metrics.observe_ms("audio_latency_ms", (time.time() - started) * 1000)
                return {
                    "ok": False,
                    "error": "That clip was too short. Please speak for a little longer.",
                    "transcript": "",
                    "response": "",
                    "state": self.controller.get_state().value,
                }

            if not np.any(np.abs(audio) > 1e-4):
                self.metrics.observe_ms("audio_latency_ms", (time.time() - started) * 1000)
                return {
                    "ok": False,
                    "error": "No audible microphone signal was detected in that clip.",
                    "transcript": "",
                    "response": "",
                    "state": self.controller.get_state().value,
                }

            logging.info("[API] Browser transcript empty=%s", not bool(transcript))
            if not transcript:
                self.metrics.observe_ms("audio_latency_ms", (time.time() - started) * 1000)
                return {
                    "ok": False,
                    "error": "I heard audio, but speech recognition did not return text. Try speaking longer or closer to the mic.",
                    "transcript": "",
                    "response": "",
                    "state": self.controller.get_state().value,
                }

            response = self.controller.get_response(transcript)
            self.metrics.observe_ms("audio_latency_ms", (time.time() - started) * 1000)
            return {
                "ok": True,
                "transcript": transcript,
                "response": response,
                "state": self.controller.get_state().value,
                "error": "",
            }

    def reset(self):
        with self._lock:
            max_recent_messages = self.controller.memory.max_recent_messages
            self.controller = ConversationController(llm=self.llm, max_recent_messages=max_recent_messages)
            self.touch()
            return {"reset": True, "state": self.controller.get_state().value}


class SessionManager:
    def __init__(self, llm, stt, metrics, max_messages=12, max_long_term=6):
        self.llm = llm
        self.stt = stt
        self.metrics = metrics
        self.max_recent_messages = max(max_messages, 12)
        self._sessions = {}
        self._lock = threading.Lock()

    def create(self):
        session = ChatSession(
            llm=self.llm,
            stt=self.stt,
            metrics=self.metrics,
            max_recent_messages=self.max_recent_messages,
        )
        with self._lock:
            self._sessions[session.session_id] = session
        self.metrics.increment("sessions_created")
        return session

    def get(self, session_id):
        with self._lock:
            return self._sessions.get(session_id)

    def delete(self, session_id):
        with self._lock:
            removed = self._sessions.pop(session_id, None)
        if removed:
            self.metrics.increment("sessions_deleted")
        return removed

    def list_sessions(self):
        with self._lock:
            return [
                {
                    "session_id": session.session_id,
                    "created_at": session.created_at,
                    "last_access": session.last_access,
                    "state": session.controller.get_state().value,
                }
                for session in self._sessions.values()
            ]
