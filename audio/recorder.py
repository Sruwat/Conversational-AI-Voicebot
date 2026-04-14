import logging
import queue
import threading
import time

import numpy as np
import sounddevice as sd


def list_input_devices():
    devices = []
    for device in sd.query_devices():
        if device.get("max_input_channels", 0) > 0:
            devices.append(device)
    return devices


def choose_input_device(preferred=None, sample_rate=16000):
    if preferred not in (None, ""):
        try:
            return int(preferred)
        except (TypeError, ValueError):
            return preferred

    devices = list_input_devices()
    ranked = []
    for device in devices:
        name = str(device.get("name", "")).lower()
        index = device.get("index")
        hostapi = int(device.get("hostapi", -1))
        max_channels = int(device.get("max_input_channels", 0))
        default_rate = float(device.get("default_samplerate", 0.0))

        score = 0
        if "microphone array" in name:
            score += 60
        elif "microphone" in name:
            score += 45
        elif "mic input" in name:
            score += 35

        if "realtek" in name:
            score += 30
        if "sst" in name:
            score += 15
        if "sound mapper" in name:
            score -= 40
        if "primary sound capture" in name:
            score -= 25
        if "stereo mix" in name:
            score -= 50
        if "headset" in name:
            score -= 10

        if abs(default_rate - sample_rate) < 1:
            score += 20
        elif abs(default_rate - 48000) < 1:
            score += 10

        score += min(max_channels, 4)
        score -= max(hostapi, 0)
        ranked.append((score, index, device))

    ranked.sort(key=lambda item: item[0], reverse=True)
    if ranked:
        best = ranked[0][2]
        logging.info(
            "[AudioBus] Auto-selected input device %s: %s (hostapi=%s, default_samplerate=%s)",
            best.get("index"),
            best.get("name"),
            best.get("hostapi"),
            best.get("default_samplerate"),
        )
        return best.get("index")
    return None


def probe_input_device(device_index, sample_rate=16000, duration_seconds=0.6, channels=1):
    frames = int(sample_rate * duration_seconds)
    try:
        audio = sd.rec(
            frames,
            samplerate=sample_rate,
            channels=channels,
            dtype="float32",
            device=device_index,
        )
        sd.wait()
        audio = np.asarray(audio, dtype=np.float32)
        if audio.ndim == 2:
            audio = np.mean(audio, axis=1)
        rms = float(np.sqrt(np.mean(np.square(audio)))) if len(audio) else 0.0
        peak = float(np.max(np.abs(audio))) if len(audio) else 0.0
        return {"ok": True, "rms": rms, "peak": peak}
    except Exception as exc:
        return {"ok": False, "error": str(exc), "rms": 0.0, "peak": 0.0}


def auto_select_working_input_device(sample_rate=16000):
    devices = list_input_devices()
    ranked = []
    for device in devices:
        name = str(device.get("name", "")).lower()
        index = device.get("index")
        base_score = 0
        if "microphone array" in name:
            base_score += 60
        elif "microphone" in name:
            base_score += 45
        elif "mic input" in name:
            base_score += 35
        if "realtek" in name:
            base_score += 30
        if "sst" in name:
            base_score += 15
        if "sound mapper" in name:
            base_score -= 40
        if "primary sound capture" in name:
            base_score -= 25
        if "stereo mix" in name or "pc speaker" in name:
            base_score -= 60
        ranked.append((base_score, device))

    ranked.sort(key=lambda item: item[0], reverse=True)
    top_candidates = [device for _, device in ranked[:5]]

    best_index = None
    best_score = -1.0
    for device in top_candidates:
        index = device.get("index")
        channels = min(max(int(device.get("max_input_channels", 1)), 1), 2)
        result = probe_input_device(index, sample_rate=sample_rate, channels=channels)
        if result["ok"]:
            logging.info(
                "[AudioProbe] device=%s name=%s rms=%.5f peak=%.5f",
                index,
                device.get("name"),
                result["rms"],
                result["peak"],
            )
            score = result["peak"] + result["rms"] * 10
            if score > best_score:
                best_score = score
                best_index = index
        else:
            logging.info(
                "[AudioProbe] device=%s name=%s failed=%s",
                index,
                device.get("name"),
                result.get("error", "unknown"),
            )

    if best_index is not None and best_score > 0.001:
        logging.info("[AudioProbe] Selected working microphone device %s after probing.", best_index)
        return best_index
    return choose_input_device(None, sample_rate=sample_rate)


class SharedAudioInput:
    def __init__(self, samplerate=16000, chunk_duration=0.25, channels=1, device=None):
        self.samplerate = samplerate
        self.chunk_duration = chunk_duration
        self.channels = channels
        self.device = choose_input_device(device, sample_rate=samplerate) if device not in (None, "") else auto_select_working_input_device(sample_rate=samplerate)
        self.chunk_samples = int(self.samplerate * self.chunk_duration)
        self._subscribers = {}
        self._lock = threading.Lock()
        self.stream = sd.InputStream(
            samplerate=self.samplerate,
            channels=self.channels,
            dtype="float32",
            blocksize=self.chunk_samples,
            device=self.device,
            callback=self.callback,
        )
        self.stream.start()

    def subscribe(self, name, maxsize=64):
        consumer = queue.Queue(maxsize=maxsize)
        with self._lock:
            self._subscribers[name] = consumer
        return consumer

    def unsubscribe(self, name):
        with self._lock:
            self._subscribers.pop(name, None)

    def callback(self, indata, frames, time_info, status):
        if status:
            logging.warning("[AudioBus] Stream status: %s", status)

        data = np.asarray(indata.copy(), dtype=np.float32)
        if data.ndim == 2:
            # Convert multi-channel input to mono instead of flattening channels into time.
            chunk = np.mean(data, axis=1).astype(np.float32)
        else:
            chunk = np.asarray(np.squeeze(data), dtype=np.float32).flatten()
        with self._lock:
            subscribers = list(self._subscribers.values())

        for consumer in subscribers:
            try:
                consumer.put_nowait(chunk)
            except queue.Full:
                try:
                    consumer.get_nowait()
                except queue.Empty:
                    pass
                try:
                    consumer.put_nowait(chunk)
                except queue.Full:
                    pass

    def stop(self):
        try:
            self.stream.stop()
            self.stream.close()
        except Exception:
            pass
