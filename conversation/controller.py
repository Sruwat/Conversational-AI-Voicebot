import queue
import threading
from enum import Enum

from ai.memory import ConversationMemory


class ConversationState(str, Enum):
    LISTENING = "listening"
    THINKING = "thinking"
    SPEAKING = "speaking"
    INTERRUPTED = "interrupted"


class ConversationController:
    def __init__(self, llm, memory=None, memory_file=None, max_recent_messages=16):
        self.llm = llm
        self.memory = memory or ConversationMemory(max_recent_messages=max_recent_messages)
        self.memory.summarizer = getattr(llm, "summarize_messages", None)
        self.memory_file = memory_file
        self._state = ConversationState.LISTENING
        self._state_lock = threading.Lock()
        self._assistant_partial = ""

    def set_state(self, state):
        with self._state_lock:
            self._state = ConversationState(state)

    def get_state(self):
        with self._state_lock:
            return self._state

    def is_interrupted(self):
        return self.get_state() == ConversationState.INTERRUPTED

    def start_user_turn(self, user_text):
        cleaned = " ".join((user_text or "").split()).strip()
        if not cleaned:
            return ""
        self.memory.add_message("user", cleaned)
        self._assistant_partial = ""
        self.set_state(ConversationState.THINKING)
        return cleaned

    def build_prompt_messages(self):
        return self.llm.build_messages(self.memory)

    def _complete_turn(self, cleaned):
        response = self.llm.get_response_from_messages(self.build_prompt_messages(), cleaned)
        finalized = self._finalize_response(response)
        self.memory.clear_interrupted_reply()
        self.memory.add_message("assistant", finalized)
        self._persist()
        self.set_state(ConversationState.LISTENING)
        return finalized

    def _stream_turn(self, cleaned, commit=True):
        raw_stream = self.llm.stream_response_from_messages(self.build_prompt_messages(), cleaned)
        collected = []
        for chunk in raw_stream:
            if self.is_interrupted():
                break
            normalized = " ".join((chunk or "").split()).strip()
            if not normalized:
                continue
            collected.append(normalized)
            self._assistant_partial = " ".join(collected).strip()
            yield normalized

        if commit and not self.is_interrupted():
            finalized = self._finalize_response(" ".join(collected))
            if finalized:
                self.memory.clear_interrupted_reply()
                self.memory.add_message("assistant", finalized)
                self._assistant_partial = finalized
                self._persist()

    def get_response(self, user_text):
        cleaned = self.start_user_turn(user_text)
        if not cleaned:
            self.set_state(ConversationState.LISTENING)
            return ""
        return self._complete_turn(cleaned)

    def stream_response(self, user_text, commit=True):
        cleaned = self.start_user_turn(user_text)
        if not cleaned:
            self.set_state(ConversationState.LISTENING)
            return
        for chunk in self._stream_turn(cleaned, commit=commit):
            yield chunk

    def mark_speaking(self):
        self.set_state(ConversationState.SPEAKING)

    def finish_speaking(self):
        if self.is_interrupted():
            self.set_state(ConversationState.LISTENING)
            return
        self.set_state(ConversationState.LISTENING)

    def interrupt(self):
        partial = " ".join(self._assistant_partial.split()).strip()
        if partial:
            self.memory.remember_interrupted_reply(partial)
            self._persist()
        self.set_state(ConversationState.INTERRUPTED)

    def reset_after_interrupt(self):
        self.set_state(ConversationState.LISTENING)

    def _finalize_response(self, response):
        return self.llm.normalize_voice_response(response)

    def commit_assistant_response(self, response):
        finalized = self._finalize_response(response)
        if finalized:
            self.memory.clear_interrupted_reply()
            self.memory.add_message("assistant", finalized)
            self._assistant_partial = finalized
            self._persist()
        return finalized

    def _persist(self):
        if self.memory_file:
            self.memory.save_to_file(self.memory_file)
