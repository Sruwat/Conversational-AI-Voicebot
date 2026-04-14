import datetime as dt
import re


class RuleBasedResponder:
    def __init__(self):
        self._patterns = [
            (re.compile(r"\b(hi|hello|hey)\b", re.I), "Hello, how can I help you?"),
            (re.compile(r"\b(thank you|thanks)\b", re.I), "You're welcome."),
            (re.compile(r"\b(bye|goodbye|see you)\b", re.I), "Goodbye."),
            (re.compile(r"\bwhat('?s| is) your name\b", re.I), "I'm your voice assistant."),
        ]

    def reply_for_text(self, text):
        text = (text or "").strip()
        if not text:
            return ""

        lowered = text.lower()
        for pattern, response in self._patterns:
            if pattern.search(text):
                return response

        if "time" in lowered and any(token in lowered for token in ("what", "tell", "current")):
            return dt.datetime.now().strftime("The time is %I:%M %p.").lstrip("0")

        if "date" in lowered and any(token in lowered for token in ("what", "tell", "today")):
            return dt.datetime.now().strftime("Today is %B %d, %Y.")

        if "who made you" in lowered or "who built you" in lowered:
            return "I'm a local voice assistant connected to Groq with offline fallback rules."

        return ""

    def reply_for_llm_failure(self, text):
        fallback = self.reply_for_text(text)
        if fallback:
            return fallback
        return (
            "I'm having trouble reaching the language model right now. "
            "Please try again in a moment, or ask a simple command like the time or date."
        )
