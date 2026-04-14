import json
import logging
import os
import re
from typing import Dict, List


class ConversationMemory:
    def __init__(self, max_recent_messages=16):
        self.max_recent_messages = max_recent_messages
        self.recent_messages: List[Dict[str, str]] = []
        self.summary = ""
        self.summarizer = None
        self.user_profile = {
            "name": "",
            "preferences": [],
            "facts": [],
        }
        self.interrupted_reply = ""

    @property
    def messages(self):
        return self.recent_messages

    @messages.setter
    def messages(self, value):
        self.recent_messages = list(value or [])

    @property
    def long_term(self):
        return [self.summary] if self.summary else []

    @long_term.setter
    def long_term(self, value):
        if isinstance(value, list):
            self.summary = " ".join(str(item).strip() for item in value if str(item).strip()).strip()
        else:
            self.summary = str(value or "").strip()

    @property
    def facts(self):
        return self.user_profile

    @facts.setter
    def facts(self, value):
        if isinstance(value, dict):
            merged = {
                "name": str(value.get("name", "")).strip(),
                "preferences": list(value.get("preferences", [])),
                "facts": list(value.get("facts", [])),
            }
            if "preference" in value and value["preference"]:
                merged["preferences"].append(str(value["preference"]).strip())
            if "health" in value and value["health"]:
                merged["facts"].append(f"Health detail: {str(value['health']).strip()}")
            self.user_profile = self._dedupe_profile(merged)

    def add_message(self, role, content):
        cleaned = " ".join((content or "").split()).strip()
        if not cleaned:
            return
        self.recent_messages.append({"role": role, "content": cleaned})
        if role == "user":
            self._extract_user_profile(cleaned)
        self._maybe_rollup_history()

    def remember_interrupted_reply(self, text):
        self.interrupted_reply = " ".join((text or "").split()).strip()[:240]

    def clear_interrupted_reply(self):
        self.interrupted_reply = ""

    def _extract_user_profile(self, content):
        lowered = content.lower()

        name_match = re.search(r"\b(?:my name is|i am|i'm)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)", content)
        if name_match:
            self.user_profile["name"] = name_match.group(1).strip()

        preference_patterns = [
            r"\b(?:i like|i love|i prefer)\s+([A-Za-z0-9 ,'-]+)",
            r"\bmy favorite\s+([A-Za-z]+)\s+is\s+([A-Za-z0-9 ,'-]+)",
        ]
        for pattern in preference_patterns:
            match = re.search(pattern, content, re.I)
            if not match:
                continue
            preference = match.group(match.lastindex).strip(" .")
            if preference:
                self.user_profile["preferences"].append(preference)

        fact_patterns = [
            r"\bi work as\s+([A-Za-z0-9 ,'-]+)",
            r"\bi live in\s+([A-Za-z0-9 ,'-]+)",
            r"\bi am from\s+([A-Za-z0-9 ,'-]+)",
        ]
        for pattern in fact_patterns:
            match = re.search(pattern, content, re.I)
            if match:
                fact = match.group(1).strip(" .")
                if fact:
                    self.user_profile["facts"].append(fact)

        if "call me " in lowered:
            alias = content.lower().split("call me ", 1)[1].strip(" .")
            if alias:
                self.user_profile["name"] = alias.title()

        self.user_profile = self._dedupe_profile(self.user_profile)

    def _dedupe_profile(self, profile):
        preferences = []
        seen_preferences = set()
        for item in profile.get("preferences", []):
            cleaned = " ".join(str(item).split()).strip()
            key = cleaned.lower()
            if cleaned and key not in seen_preferences:
                seen_preferences.add(key)
                preferences.append(cleaned)

        facts = []
        seen_facts = set()
        for item in profile.get("facts", []):
            cleaned = " ".join(str(item).split()).strip()
            key = cleaned.lower()
            if cleaned and key not in seen_facts:
                seen_facts.add(key)
                facts.append(cleaned)

        return {
            "name": " ".join(str(profile.get("name", "")).split()).strip(),
            "preferences": preferences[-6:],
            "facts": facts[-8:],
        }

    def _maybe_rollup_history(self):
        if len(self.recent_messages) <= self.max_recent_messages:
            return

        overflow_count = len(self.recent_messages) - self.max_recent_messages
        overflow = self.recent_messages[:overflow_count]
        self.recent_messages = self.recent_messages[overflow_count:]
        delta_summary = self._build_summary_fragment(overflow)
        if callable(self.summarizer):
            try:
                llm_summary = self.summarizer(overflow, existing_summary=self.summary, user_profile=self.user_profile)
                if llm_summary:
                    delta_summary = llm_summary
            except Exception as exc:
                logging.warning("LLM summarization failed, falling back to heuristic summary: %s", exc)
        self.summary = self._merge_summaries(self.summary, delta_summary)

    def _build_summary_fragment(self, messages):
        user_points = []
        assistant_points = []
        for message in messages:
            text = message.get("content", "").strip()
            if not text:
                continue
            compressed = text[:140]
            if message.get("role") == "user":
                user_points.append(compressed)
            elif message.get("role") == "assistant":
                assistant_points.append(compressed)

        summary_parts = []
        if user_points:
            summary_parts.append("User discussed: " + " | ".join(user_points[-3:]))
        if assistant_points:
            summary_parts.append("Assistant covered: " + " | ".join(assistant_points[-3:]))
        return " ".join(summary_parts).strip()

    def _merge_summaries(self, existing, delta):
        parts = [part.strip() for part in (existing, delta) if part and part.strip()]
        merged = " ".join(parts).strip()
        return merged[-1200:]

    def get_history(self, max_tokens=800):
        prompt = []

        memory_context = self.build_memory_context()
        if memory_context:
            prompt.append({"role": "system", "content": memory_context})

        token_count = sum(len(item["content"].split()) for item in prompt)
        for message in self.recent_messages[-self.max_recent_messages :]:
            estimated = len(message["content"].split())
            if token_count + estimated > max_tokens:
                break
            prompt.append(message)
            token_count += estimated

        return prompt

    def build_memory_context(self):
        parts = []

        if self.summary:
            parts.append(f"Conversation summary: {self.summary}")

        profile_bits = []
        if self.user_profile.get("name"):
            profile_bits.append(f"name={self.user_profile['name']}")
        if self.user_profile.get("preferences"):
            profile_bits.append("preferences=" + ", ".join(self.user_profile["preferences"][-4:]))
        if self.user_profile.get("facts"):
            profile_bits.append("facts=" + "; ".join(self.user_profile["facts"][-4:]))
        if profile_bits:
            parts.append("User profile: " + " | ".join(profile_bits))

        if self.interrupted_reply:
            parts.append(
                "Earlier you were interrupted while saying: "
                f"{self.interrupted_reply}. Continue naturally if still relevant, without restarting from the top."
            )

        return "\n".join(parts).strip()

    def get_facts(self):
        return dict(self.user_profile)

    def get_long_term(self):
        return [self.summary] if self.summary else []

    def to_dict(self):
        return {
            "recent_messages": self.recent_messages[-self.max_recent_messages :],
            "summary": self.summary,
            "user_profile": self.user_profile,
            "interrupted_reply": self.interrupted_reply,
        }

    def save_to_file(self, path):
        try:
            with open(path, "w", encoding="utf-8") as handle:
                json.dump(self.to_dict(), handle, ensure_ascii=False, indent=2)
            logging.info("Saved memory to %s", path)
        except Exception as exc:
            logging.error("Failed to save memory: %s", exc)

    @classmethod
    def load_from_file(cls, path, max_recent_messages=16):
        memory = cls(max_recent_messages=max_recent_messages)
        if not os.path.exists(path):
            return memory

        try:
            with open(path, "r", encoding="utf-8") as handle:
                data = json.load(handle)
        except Exception as exc:
            logging.error("Failed to load memory: %s", exc)
            return memory

        if "recent_messages" in data:
            memory.recent_messages = list(data.get("recent_messages", []))[-max_recent_messages:]
            memory.summary = str(data.get("summary", "")).strip()
            memory.user_profile = memory._dedupe_profile(data.get("user_profile", {}))
            memory.interrupted_reply = str(data.get("interrupted_reply", "")).strip()
            return memory

        memory.recent_messages = list(data.get("messages", []))[-max_recent_messages:]
        legacy_long_term = data.get("long_term", [])
        if isinstance(legacy_long_term, list):
            fragments = []
            for item in legacy_long_term:
                if isinstance(item, dict):
                    fragments.extend(
                        [
                            str(item.get("user_goal", "")).strip(),
                            str(item.get("assistant_reply", "")).strip(),
                        ]
                    )
                else:
                    fragments.append(str(item).strip())
            memory.summary = " ".join(fragment for fragment in fragments if fragment).strip()
        memory.facts = data.get("facts", {})
        return memory
