import logging
import time

from groq import Groq

from ai.fallback import RuleBasedResponder


class GroqLLM:
    def __init__(
        self,
        api_key,
        model,
        complex_model=None,
        max_tokens=160,
        temperature=0.3,
        max_retries=2,
        cooldown_seconds=20,
    ):
        self.api_key = api_key
        self.model = model
        self.complex_model = complex_model or model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.max_retries = max_retries
        self.cooldown_seconds = cooldown_seconds
        self.rule_based = RuleBasedResponder()
        self.failure_count = 0
        self.cooldown_until = 0
        self.system_prompt = (
            "You are a warm, human-like real-time voice assistant. "
            "Sound natural in spoken conversation, not like a chatbot. "
            "Use short, voice-friendly replies that are usually one or two sentences. "
            "Keep momentum, maintain context from memory, and ask one natural follow-up question when it helps. "
            "Use occasional gentle conversational openers such as 'Okay,' 'Got it,' or 'Let me think,' but do not overuse them. "
            "Avoid lists, markdown, long paragraphs, and robotic phrasing."
        )
        try:
            self.client = Groq(api_key=api_key) if api_key else None
        except Exception as exc:
            logging.error("[LLM] Groq client init error: %s", exc)
            self.client = None

    def _select_model(self, user_text):
        text = (user_text or "").strip()
        if len(text.split()) > 35 or any(token in text.lower() for token in ("explain", "compare", "steps", "detailed")):
            return self.complex_model
        return self.model

    def build_messages(self, memory):
        messages = [{"role": "system", "content": self.system_prompt}]
        messages.extend(memory.get_history(max_tokens=900))
        return messages

    def normalize_voice_response(self, response):
        cleaned = " ".join((response or "").split()).strip()
        if not cleaned:
            return ""

        sentences = []
        buffer = []
        for token in cleaned.replace("!", ".").replace("?", ".").split("."):
            item = token.strip()
            if item:
                buffer.append(item)
        sentences.extend(buffer)
        if len(sentences) > 2:
            cleaned = ". ".join(sentences[:2]).strip()
        else:
            cleaned = ". ".join(sentences).strip()

        if cleaned and cleaned[-1] not in ".!?":
            cleaned += "."

        openers = ("okay", "got it", "let me think", "alright", "sure")
        if not cleaned.lower().startswith(openers):
            first_word_count = len(cleaned.split())
            if first_word_count > 4:
                cleaned = f"Okay, {cleaned[0].lower() + cleaned[1:] if len(cleaned) > 1 else cleaned.lower()}"

        return cleaned

    def _stream_from_client(self, messages, user_text):
        model = self._select_model(user_text)
        for attempt in range(1, self.max_retries + 2):
            try:
                stream = self.client.chat.completions.create(
                    model=model,
                    messages=messages,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                    stream=True,
                )
                self.failure_count = 0
                raw_parts = []
                buffer = ""
                for event in stream:
                    delta = getattr(event.choices[0].delta, "content", "") if event.choices else ""
                    if delta:
                        raw_parts.append(delta)
                        buffer += delta
                        if any(buffer.rstrip().endswith(mark) for mark in (".", "!", "?", ",", ";", ":")) or len(buffer) >= 80:
                            chunk = " ".join(buffer.split()).strip()
                            if chunk:
                                yield chunk + (" " if chunk[-1].isalnum() else "")
                            buffer = ""
                final_chunk = " ".join(buffer.split()).strip()
                if final_chunk:
                    yield final_chunk
                if raw_parts:
                    return
            except Exception as exc:
                logging.error("[LLM] Streaming error during Groq request attempt %s: %s", attempt, exc)

        self.failure_count += 1
        if self.failure_count >= self.max_retries + 1:
            self.cooldown_until = time.time() + self.cooldown_seconds
            logging.warning("[LLM] Entering cooldown for %s seconds.", self.cooldown_seconds)
            self.failure_count = 0
        yield self.rule_based.reply_for_llm_failure(user_text)

    def summarize_messages(self, messages, existing_summary="", user_profile=None):
        if self.client is None:
            return ""

        transcript = []
        for message in messages[-10:]:
            role = message.get("role", "user")
            content = " ".join(str(message.get("content", "")).split()).strip()
            if content:
                transcript.append(f"{role}: {content}")
        if not transcript:
            return ""

        prompt = (
            "Summarize this older conversation context for a voice assistant. "
            "Keep it short, useful, and factual. "
            "Focus on user goals, preferences, commitments, and unresolved topics. "
            "Write plain text in 3 short sentences max."
        )
        messages_payload = [
            {"role": "system", "content": prompt},
            {
                "role": "user",
                "content": (
                    f"Existing summary: {existing_summary or 'none'}\n"
                    f"User profile: {user_profile or {}}\n"
                    "Conversation slice:\n" + "\n".join(transcript)
                ),
            },
        ]

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages_payload,
                max_tokens=120,
                temperature=0.2,
            )
            content = response.choices[0].message.content.strip()
            return self.normalize_voice_response(content)
        except Exception as exc:
            logging.warning("[LLM] Summary generation failed: %s", exc)
            return ""

    def stream_response_from_messages(self, messages, user_text):
        direct_rule = self.rule_based.reply_for_text(user_text)
        if direct_rule:
            yield self.normalize_voice_response(direct_rule)
            return

        if time.time() < self.cooldown_until or self.client is None:
            yield self.normalize_voice_response(self.rule_based.reply_for_llm_failure(user_text))
            return

        for chunk in self._stream_from_client(messages, user_text):
            yield chunk

    def get_response_from_messages(self, messages, user_text):
        direct_rule = self.rule_based.reply_for_text(user_text)
        if direct_rule:
            return self.normalize_voice_response(direct_rule)

        if time.time() < self.cooldown_until:
            logging.warning("[LLM] Circuit breaker active. Using fallback response.")
            return self.normalize_voice_response(self.rule_based.reply_for_llm_failure(user_text))

        if self.client is None:
            logging.warning("[LLM] Groq unavailable. Using rule-based fallback.")
            return self.normalize_voice_response(self.rule_based.reply_for_llm_failure(user_text))

        model = self._select_model(user_text)
        for attempt in range(1, self.max_retries + 2):
            try:
                response = self.client.chat.completions.create(
                    model=model,
                    messages=messages,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                )
                self.failure_count = 0
                content = response.choices[0].message.content.strip()
                return self.normalize_voice_response(content) or self.normalize_voice_response(
                    self.rule_based.reply_for_llm_failure(user_text)
                )
            except Exception as exc:
                logging.error("[LLM] Error during Groq request attempt %s: %s", attempt, exc)

        self.failure_count += 1
        if self.failure_count >= self.max_retries + 1:
            self.cooldown_until = time.time() + self.cooldown_seconds
            logging.warning("[LLM] Entering cooldown for %s seconds.", self.cooldown_seconds)
            self.failure_count = 0
        return self.normalize_voice_response(self.rule_based.reply_for_llm_failure(user_text))

    def stream_response(self, memory, user_text):
        for chunk in self.stream_response_from_messages(self.build_messages(memory), user_text):
            yield chunk

    def get_response(self, memory, user_text):
        return self.get_response_from_messages(self.build_messages(memory), user_text)
