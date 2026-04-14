import logging
import queue
import signal
import sys
import threading
import time

from utils.bootstrap import ensure_user_site_packages

ensure_user_site_packages()

import numpy as np

from ai.llm import GroqLLM
from ai.memory import ConversationMemory
from ai.stt import SpeechToText
from audio.noise import reduce_noise
from audio.recorder import SharedAudioInput, list_input_devices
from audio.vad import VoiceActivityDetector, WakeWordDetector
from conversation import ConversationController, ConversationState
from tts.bargein import BargeInMonitor
from tts.speaker import TTSSpeaker
from utils.config import get_config


logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

shutdown_event = threading.Event()


def signal_handler(sig, frame):
    logging.info("Received exit signal. Shutting down.")
    shutdown_event.set()


signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


def drain_queue(target_queue):
    while True:
        try:
            target_queue.get_nowait()
        except queue.Empty:
            return


class TurnTracker:
    def __init__(self):
        self._lock = threading.Lock()
        self._latest_turn_id = 0

    def next_turn(self):
        with self._lock:
            self._latest_turn_id += 1
            return self._latest_turn_id

    def latest_turn(self):
        with self._lock:
            return self._latest_turn_id


class ActivityTracker:
    def __init__(self):
        self._lock = threading.Lock()
        self._last_activity = time.time()

    def touch(self):
        with self._lock:
            self._last_activity = time.time()

    def seconds_since_activity(self):
        with self._lock:
            return time.time() - self._last_activity


def format_level_bar(rms, width=20):
    normalized = min(1.0, max(0.0, rms * 40))
    filled = int(normalized * width)
    return "[" + ("#" * filled) + ("-" * (width - filled)) + "]"


def build_startup_greeting(name):
    hour = time.localtime().tm_hour
    if hour < 12:
        prefix = "Good morning"
    elif hour < 17:
        prefix = "Good afternoon"
    else:
        prefix = "Good evening"
    return f"{prefix}, {name}. I'm listening."


def wait_for_speech_completion(speaker, timeout_seconds=8.0, label="speech"):
    deadline = time.time() + timeout_seconds
    while speaker.is_speaking.is_set() and not shutdown_event.is_set():
        if time.time() >= deadline:
            logging.warning("[TTS] Timeout while waiting for %s to finish.", label)
            speaker.stop()
            break
        time.sleep(0.03)


def maybe_reduce_noise(audio, config, noise_sample=None):
    if not config["NOISE_REDUCTION"] or len(audio) == 0:
        return audio
    try:
        return reduce_noise(audio, sr=config["AUDIO_SAMPLE_RATE"], noise_sample=noise_sample)
    except Exception as exc:
        logging.warning("Noise reduction failed, using raw audio: %s", exc)
        return audio


def calibrate_noise(audio_queue, vad, duration_seconds=2):
    calibration_audio = []
    deadline = time.time() + duration_seconds
    while time.time() < deadline and not shutdown_event.is_set():
        try:
            chunk = audio_queue.get(timeout=0.5)
        except queue.Empty:
            continue
        if len(chunk):
            calibration_audio.append(chunk)
    if calibration_audio:
        noise_sample = np.concatenate(calibration_audio).astype(np.float32)
        vad.calibrate(noise_sample)
        return noise_sample
    return np.array([], dtype=np.float32)


def debug_audio_worker(config, vad, source_queue):
    report_interval = max(0.2, config["DEBUG_AUDIO_INTERVAL_MS"] / 1000)
    last_report = 0.0
    peak_rms = 0.0
    speech_hits = 0

    while not shutdown_event.is_set():
        try:
            chunk = source_queue.get(timeout=0.2)
        except queue.Empty:
            continue

        rms = float(np.sqrt(np.mean(np.square(chunk)))) if len(chunk) else 0.0
        peak_rms = max(peak_rms, rms)
        if vad.is_speech(chunk):
            speech_hits += 1

        now = time.time()
        if now - last_report >= report_interval:
            logging.info(
                "[DebugAudio] peak=%.5f speech=%s threshold=%.5f %s",
                peak_rms,
                "yes" if speech_hits > 0 else "no",
                vad.energy_threshold,
                format_level_bar(peak_rms),
            )
            last_report = now
            peak_rms = 0.0
            speech_hits = 0


def microphone_self_test(source_queue, sample_rate=16000, duration_seconds=2.0):
    deadline = time.time() + duration_seconds
    peak_rms = 0.0
    non_silent_chunks = 0

    while time.time() < deadline and not shutdown_event.is_set():
        try:
            chunk = source_queue.get(timeout=0.25)
        except queue.Empty:
            continue
        if len(chunk) == 0:
            continue
        rms = float(np.sqrt(np.mean(np.square(chunk))))
        peak_rms = max(peak_rms, rms)
        if rms > 0.002:
            non_silent_chunks += 1

    logging.info(
        "[MicTest] peak=%.5f non_silent_chunks=%s duration=%.1fs",
        peak_rms,
        non_silent_chunks,
        duration_seconds,
    )
    return peak_rms, non_silent_chunks


def wake_worker(config, detector, stt, source_queue, wake_event, active_event, activity_tracker):
    while not shutdown_event.is_set():
        if active_event.is_set():
            time.sleep(0.1)
            continue
        if detector.listen_for_wake_word(stt=stt, source_queue=source_queue, timeout=1.5):
            activity_tracker.touch()
            logging.info("[WakeWord] Wake event triggered.")
            wake_event.set()


def segment_worker(config, vad, source_queue, active_event, utterance_queue, activity_tracker):
    speech_frames = []
    silence_frames = 0
    chunk_duration_ms = config["CHUNK_MS"]
    max_frames = max(1, int((config["MAX_UTTERANCE_SECONDS"] * 1000) / chunk_duration_ms))
    silence_limit = max(1, int(config["END_OF_UTTERANCE_SILENCE_MS"] / chunk_duration_ms))

    while not shutdown_event.is_set():
        try:
            chunk = source_queue.get(timeout=0.2)
        except queue.Empty:
            continue

        if not active_event.is_set():
            speech_frames = []
            silence_frames = 0
            continue

        if vad.is_speech(chunk):
            speech_frames.append(chunk)
            silence_frames = 0
            activity_tracker.touch()
        elif speech_frames:
            speech_frames.append(chunk)
            silence_frames += 1
            if silence_frames >= silence_limit:
                utterance = np.concatenate(speech_frames).astype(np.float32)
                try:
                    utterance_queue.put_nowait(utterance)
                except queue.Full:
                    try:
                        utterance_queue.get_nowait()
                    except queue.Empty:
                        pass
                    utterance_queue.put_nowait(utterance)
                speech_frames = []
                silence_frames = 0

        if speech_frames and len(speech_frames) >= max_frames:
            utterance = np.concatenate(speech_frames).astype(np.float32)
            try:
                utterance_queue.put_nowait(utterance)
            except queue.Full:
                try:
                    utterance_queue.get_nowait()
                except queue.Empty:
                    pass
                utterance_queue.put_nowait(utterance)
            speech_frames = []
            silence_frames = 0


def stt_worker(config, stt, speaker, utterance_queue, transcript_queue, turn_tracker, activity_tracker, noise_sample):
    while not shutdown_event.is_set():
        try:
            audio = utterance_queue.get(timeout=0.2)
        except queue.Empty:
            continue

        clean_audio = maybe_reduce_noise(audio, config, noise_sample=noise_sample)
        user_text = stt.transcribe(clean_audio, sample_rate=config["AUDIO_SAMPLE_RATE"])
        if not user_text:
            logging.info("[STT] No text detected from utterance.")
            continue

        speaker.stop()
        turn_id = turn_tracker.next_turn()
        activity_tracker.touch()
        logging.info("[STT] Transcript: %s", user_text)
        try:
            transcript_queue.put_nowait((turn_id, user_text))
        except queue.Full:
            drain_queue(transcript_queue)
            transcript_queue.put_nowait((turn_id, user_text))


def llm_worker(controller, transcript_queue, response_queue, turn_tracker, activity_tracker):
    while not shutdown_event.is_set():
        try:
            turn_id, user_text = transcript_queue.get(timeout=0.2)
        except queue.Empty:
            continue

        logging.info("User said: %s", user_text)

        stream_queue = queue.Queue(maxsize=32)
        try:
            response_queue.put_nowait((turn_id, stream_queue))
        except queue.Full:
            drain_queue(response_queue)
            response_queue.put_nowait((turn_id, stream_queue))

        for chunk in controller.stream_response(user_text, commit=False):
            if turn_id != turn_tracker.latest_turn():
                logging.info("Discarding stale streamed response for turn %s.", turn_id)
                controller.interrupt()
                break
            activity_tracker.touch()
            try:
                stream_queue.put_nowait(chunk)
            except queue.Full:
                try:
                    stream_queue.get_nowait()
                except queue.Empty:
                    pass
                stream_queue.put_nowait(chunk)

        stream_queue.put(None)
        if controller.get_state() == ConversationState.INTERRUPTED:
            controller.reset_after_interrupt()
        activity_tracker.touch()


def tts_worker(config, controller, speaker, vad, barge_queue, response_queue, turn_tracker, activity_tracker):
    while not shutdown_event.is_set():
        try:
            turn_id, stream_queue = response_queue.get(timeout=0.2)
        except queue.Empty:
            continue

        if turn_id != turn_tracker.latest_turn():
            continue

        spoken_chunks = []

        def on_spoken_chunk(chunk):
            spoken_chunks.append(chunk)

        drain_queue(barge_queue)
        monitor = BargeInMonitor(
            vad=vad,
            tts_speaker=speaker,
            source_queue=barge_queue,
            chunk_ms=config["CHUNK_MS"],
            min_interrupt_ms=200,
            ignore_playback_ms=300,
            speech_energy_multiplier=1.8,
        )
        controller.mark_speaking()
        monitor.start()
        speaker.speak_stream(
            stream_queue,
            barge_in_callback=monitor.barge_in_callback,
            spoken_chunk_callback=on_spoken_chunk,
        )
        while speaker.is_speaking.is_set() and not shutdown_event.is_set():
            if monitor.barge_in_callback():
                controller.interrupt()
                logging.info("[State] %s", controller.get_state().value)
                break
            if turn_id != turn_tracker.latest_turn():
                controller.interrupt()
                speaker.stop()
                break
            time.sleep(0.03)
        monitor.stop()

        if controller.get_state() == ConversationState.INTERRUPTED:
            partial_response = " ".join(spoken_chunks).strip()
            if partial_response:
                controller.memory.remember_interrupted_reply(partial_response)
                controller.memory.save_to_file(controller.memory_file)
            controller.reset_after_interrupt()
        else:
            final_response = " ".join(spoken_chunks).strip()
            if final_response:
                controller.commit_assistant_response(final_response)
            controller.finish_speaking()
        activity_tracker.touch()


def main():
    config = get_config()
    memory = ConversationMemory.load_from_file(config["MEMORY_FILE"], max_recent_messages=16)
    turn_tracker = TurnTracker()
    activity_tracker = ActivityTracker()
    active_event = threading.Event()
    wake_event = threading.Event()

    vad = VoiceActivityDetector(
        provider=config["VAD_PROVIDER"],
        auto_calibrate=config["AUTO_CALIBRATE_NOISE"],
        samplerate=config["AUDIO_SAMPLE_RATE"],
    )
    stt = SpeechToText(provider=config["STT_PROVIDER"], model_name=config["STT_MODEL"])
    llm = GroqLLM(
        api_key=config["GROQ_API_KEY"],
        model=config["GROQ_MODEL"],
        complex_model=config["GROQ_COMPLEX_MODEL"],
        max_tokens=config["GROQ_MAX_TOKENS"],
        temperature=config["GROQ_TEMPERATURE"],
        max_retries=config["GROQ_MAX_RETRIES"],
        cooldown_seconds=config["GROQ_COOLDOWN_SECONDS"],
    )
    controller = ConversationController(llm=llm, memory=memory, memory_file=config["MEMORY_FILE"], max_recent_messages=16)
    speaker = TTSSpeaker(
        engine=config["TTS_PROVIDER"],
        voice=config["TTS_VOICE"],
        piper_command=config["PIPER_COMMAND"],
        piper_model_path=config["PIPER_MODEL_PATH"],
        piper_config_path=config["PIPER_CONFIG_PATH"],
    )
    for device in list_input_devices():
        logging.info(
            "[AudioDevice] input index=%s name=%s hostapi=%s rate=%s channels=%s",
            device.get("index"),
            device.get("name"),
            device.get("hostapi"),
            device.get("default_samplerate"),
            device.get("max_input_channels"),
        )
    audio_input = SharedAudioInput(
        samplerate=config["AUDIO_SAMPLE_RATE"],
        chunk_duration=config["CHUNK_MS"] / 1000,
        channels=config["AUDIO_CHANNELS"],
        device=config["MIC_DEVICE"],
    )

    calibration_queue = audio_input.subscribe("calibration", maxsize=32)
    wake_queue = audio_input.subscribe("wake", maxsize=64)
    segment_queue = audio_input.subscribe("segment", maxsize=128)
    barge_queue = audio_input.subscribe("barge", maxsize=128)
    debug_queue = audio_input.subscribe("debug", maxsize=64)
    mic_test_queue = audio_input.subscribe("mic_test", maxsize=64)
    utterance_queue = queue.Queue(maxsize=8)
    transcript_queue = queue.Queue(maxsize=8)
    response_queue = queue.Queue(maxsize=8)

    wake_detector = None
    if config["USE_WAKE_WORD"] and not config["BYPASS_WAKE_WORD"]:
        wake_detector = WakeWordDetector(
            vad=vad,
            wake_word=config["WAKE_WORD"],
            provider=config["WAKE_WORD_PROVIDER"],
            samplerate=config["AUDIO_SAMPLE_RATE"],
            chunk_duration=config["CHUNK_MS"] / 1000,
            model_path=config["WAKE_WORD_MODEL_PATH"],
        )
    else:
        active_event.set()
        controller.set_state(ConversationState.LISTENING)

    logging.info("Voice bot ready.")
    logging.info("STT=%s VAD=%s TTS=%s", stt.backend_name, vad.backend_name, speaker.engine_name)
    logging.info(
        "Mode=%s",
        "direct-listen" if config["BYPASS_WAKE_WORD"] or not config["USE_WAKE_WORD"] else "wake-word",
    )

    if config["AUTO_CALIBRATE_NOISE"]:
        logging.info("Calibrating background noise for 2 seconds.")
        noise_sample = calibrate_noise(calibration_queue, vad, duration_seconds=2)
    else:
        noise_sample = np.array([], dtype=np.float32)
    audio_input.unsubscribe("calibration")

    if active_event.is_set() and config["START_WITH_GREETING"]:
        greeting = build_startup_greeting(config["GREETING_NAME"])
        logging.info("[Startup] %s", greeting)
        speaker.speak(greeting)
        wait_for_speech_completion(speaker, timeout_seconds=8.0, label="startup greeting")
        logging.info("[MicTest] Speak now for 2 seconds so I can verify live capture.")
        microphone_self_test(mic_test_queue, sample_rate=config["AUDIO_SAMPLE_RATE"], duration_seconds=2.0)

    workers = [
        threading.Thread(
            target=segment_worker,
            args=(config, vad, segment_queue, active_event, utterance_queue, activity_tracker),
            daemon=True,
        ),
        threading.Thread(
            target=stt_worker,
            args=(config, stt, speaker, utterance_queue, transcript_queue, turn_tracker, activity_tracker, noise_sample),
            daemon=True,
        ),
        threading.Thread(
            target=llm_worker,
            args=(controller, transcript_queue, response_queue, turn_tracker, activity_tracker),
            daemon=True,
        ),
        threading.Thread(
            target=tts_worker,
            args=(config, controller, speaker, vad, barge_queue, response_queue, turn_tracker, activity_tracker),
            daemon=True,
        ),
    ]

    if config["DEBUG_AUDIO"]:
        workers.append(
            threading.Thread(
                target=debug_audio_worker,
                args=(config, vad, debug_queue),
                daemon=True,
            )
        )

    if wake_detector is not None:
        workers.append(
            threading.Thread(
                target=wake_worker,
                args=(config, wake_detector, stt, wake_queue, wake_event, active_event, activity_tracker),
                daemon=True,
            )
        )

    for worker in workers:
        worker.start()

    idle_timeout_seconds = config["IDLE_TIMEOUT_SECONDS"]

    try:
        while not shutdown_event.is_set():
            if not active_event.is_set():
                if wake_event.wait(timeout=0.2):
                    wake_event.clear()
                    active_event.set()
                    controller.set_state(ConversationState.LISTENING)
                    drain_queue(segment_queue)
                    drain_queue(barge_queue)
                    speaker.stop()
                    speaker.speak("Yes?")
                    while speaker.is_speaking.is_set() and not shutdown_event.is_set():
                        time.sleep(0.03)
                    logging.info("Wake word accepted. Conversation active.")
            else:
                if controller.get_state() == ConversationState.INTERRUPTED:
                    controller.reset_after_interrupt()
                if (
                    activity_tracker.seconds_since_activity() > idle_timeout_seconds
                    and config["USE_WAKE_WORD"]
                    and not config["BYPASS_WAKE_WORD"]
                ):
                    logging.info("Conversation timed out. Returning to wake-word mode.")
                    active_event.clear()
                    controller.set_state(ConversationState.LISTENING)
                    speaker.stop()
                    drain_queue(utterance_queue)
                    drain_queue(transcript_queue)
                    drain_queue(response_queue)
                time.sleep(0.1)
    finally:
        speaker.stop()
        audio_input.stop()
        controller.memory.save_to_file(config["MEMORY_FILE"])


if __name__ == "__main__":
    main()
