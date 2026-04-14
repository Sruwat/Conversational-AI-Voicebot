import threading
import time


class MetricsRegistry:
    def __init__(self):
        self._lock = threading.Lock()
        self._counters = {
            "sessions_created": 0,
            "sessions_deleted": 0,
            "chat_requests": 0,
            "audio_requests": 0,
            "llm_streams": 0,
            "health_checks": 0,
        }
        self._timers = {
            "chat_latency_ms": [],
            "audio_latency_ms": [],
        }
        self._started_at = time.time()

    def increment(self, key, amount=1):
        with self._lock:
            self._counters[key] = self._counters.get(key, 0) + amount

    def observe_ms(self, key, value):
        with self._lock:
            values = self._timers.setdefault(key, [])
            values.append(round(float(value), 2))
            if len(values) > 100:
                del values[:-100]

    def snapshot(self):
        with self._lock:
            timer_summary = {}
            for key, values in self._timers.items():
                timer_summary[key] = {
                    "count": len(values),
                    "avg": round(sum(values) / len(values), 2) if values else 0,
                    "max": max(values) if values else 0,
                }
            return {
                "uptime_seconds": round(time.time() - self._started_at, 2),
                "counters": dict(self._counters),
                "timers": timer_summary,
            }
