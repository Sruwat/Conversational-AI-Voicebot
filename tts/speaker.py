import logging
import os
import queue
import shutil
import subprocess
import tempfile
import threading
import time

import pyttsx3
import sounddevice as sd
import soundfile as sf


class TTSSpeaker:
    def __init__(self, engine="auto", voice="", piper_command="piper", piper_model_path="", piper_config_path=""):
        self.engine_name = engine
        self.voice = voice
        self.piper_command = piper_command
        self.piper_model_path = piper_model_path
        self.piper_config_path = piper_config_path
        self.is_speaking = threading.Event()
        self.stop_event = threading.Event()
        self._thread = None
        self._playback_stream = None
        self._process = None
        self.engine = None
        self._resolve_engine()

    def _resolve_engine(self):
        if self.engine_name == "auto":
            if self.piper_model_path and shutil.which(self.piper_command):
                self.engine_name = "piper"
            else:
                self.engine_name = "pyttsx3"

        if self.engine_name == "pyttsx3":
            try:
                self.engine = pyttsx3.init()
                if self.voice:
                    self.engine.setProperty("voice", self.voice)
            except Exception as exc:
                logging.error("[TTS] pyttsx3 init error: %s", exc)
                self.engine = None

    def speak(self, text, barge_in_callback=None, spoken_chunk_callback=None):
        self.stop()
        self.stop_event.clear()
        self.is_speaking.set()

        def run():
            try:
                if self.engine_name == "piper":
                    self._speak_with_piper(text, barge_in_callback, spoken_chunk_callback)
                else:
                    self._speak_with_pyttsx3(text, barge_in_callback, spoken_chunk_callback)
            finally:
                self.is_speaking.clear()

        self._thread = threading.Thread(target=run, daemon=True)
        self._thread.start()

    def speak_stream(self, chunk_queue, barge_in_callback=None, spoken_chunk_callback=None):
        self.stop()
        self.stop_event.clear()
        self.is_speaking.set()

        def run():
            try:
                while not self.stop_event.is_set():
                    try:
                        chunk = chunk_queue.get(timeout=0.2)
                    except queue.Empty:
                        continue

                    if chunk is None:
                        break
                    chunk = " ".join((chunk or "").split()).strip()
                    if not chunk:
                        continue

                    if spoken_chunk_callback:
                        spoken_chunk_callback(chunk)

                    if self.engine_name == "piper":
                        self._speak_with_piper(chunk, barge_in_callback, None)
                    else:
                        self._speak_with_pyttsx3(chunk, barge_in_callback, None)

                    if barge_in_callback and barge_in_callback():
                        break
            finally:
                self.is_speaking.clear()

        self._thread = threading.Thread(target=run, daemon=True)
        self._thread.start()

    def synthesize_bytes(self, text):
        tmp_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp_wav.close()
        try:
            success = self._synthesize_to_file(text, tmp_wav.name)
            if not success:
                return b""
            with open(tmp_wav.name, "rb") as handle:
                return handle.read()
        finally:
            if os.path.exists(tmp_wav.name):
                try:
                    os.remove(tmp_wav.name)
                except OSError:
                    pass

    def _speak_with_pyttsx3(self, text, barge_in_callback, spoken_chunk_callback=None):
        if self.engine is None:
            logging.error("[TTS] pyttsx3 engine unavailable.")
            return
        try:
            if spoken_chunk_callback:
                spoken_chunk_callback(text)
            self.engine.say(text)
            self.engine.startLoop(False)
            while self.engine.isBusy():
                if self.stop_event.is_set() or (barge_in_callback and barge_in_callback()):
                    self.engine.stop()
                    break
                self.engine.iterate()
                time.sleep(0.03)
            self.engine.endLoop()
        except Exception as exc:
            logging.error("[TTS] pyttsx3 playback failed: %s", exc)

    def _speak_with_piper(self, text, barge_in_callback, spoken_chunk_callback=None):
        tmp_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp_wav.close()
        try:
            if not self._synthesize_to_file(text, tmp_wav.name):
                self._speak_with_pyttsx3(text, barge_in_callback, spoken_chunk_callback)
                return

            if spoken_chunk_callback:
                spoken_chunk_callback(text)

            data, sample_rate = sf.read(tmp_wav.name, dtype="float32")
            if data.ndim > 1:
                data = data[:, 0]

            self._playback_stream = sd.OutputStream(samplerate=sample_rate, channels=1, dtype="float32")
            self._playback_stream.start()

            frame_size = 1024
            for start in range(0, len(data), frame_size):
                if self.stop_event.is_set() or (barge_in_callback and barge_in_callback()):
                    break
                self._playback_stream.write(data[start : start + frame_size])
            self._playback_stream.stop()
            self._playback_stream.close()
            self._playback_stream = None
        except Exception as exc:
            logging.error("[TTS] Piper playback failed, falling back to pyttsx3: %s", exc)
            self._speak_with_pyttsx3(text, barge_in_callback, spoken_chunk_callback)
        finally:
            if os.path.exists(tmp_wav.name):
                try:
                    os.remove(tmp_wav.name)
                except OSError:
                    pass

    def _synthesize_to_file(self, text, output_path):
        if self.engine_name == "piper":
            command = [self.piper_command, "--model", self.piper_model_path, "--output_file", output_path]
            if self.piper_config_path:
                command.extend(["--config", self.piper_config_path])
            try:
                self._process = subprocess.Popen(
                    command,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                )
                self._process.communicate(text, timeout=30)
                self._process = None
                return os.path.exists(output_path) and os.path.getsize(output_path) > 0
            except Exception as exc:
                logging.error("[TTS] Piper synthesis failed: %s", exc)
                return False

        if self.engine_name == "pyttsx3" and self.engine is not None:
            try:
                self.engine.save_to_file(text, output_path)
                self.engine.runAndWait()
                return os.path.exists(output_path) and os.path.getsize(output_path) > 0
            except Exception as exc:
                logging.error("[TTS] pyttsx3 synthesis failed: %s", exc)
        return False

    def stop(self):
        self.stop_event.set()
        self.is_speaking.clear()

        if self._process and self._process.poll() is None:
            self._process.kill()
            self._process = None

        if self._playback_stream is not None:
            try:
                self._playback_stream.abort()
                self._playback_stream.close()
            except Exception:
                pass
            self._playback_stream = None

        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=0.5)
