const chatLog = document.getElementById("chat-log");
const micButton = document.getElementById("mic-button");
const resetButton = document.getElementById("reset-button");
const applyButton = document.getElementById("apply-button");
const player = document.getElementById("player");
const statusPill = document.getElementById("status-pill");
const transportPill = document.getElementById("transport-pill");
const sessionLabel = document.getElementById("session-label");
const liveCaption = document.getElementById("live-caption");
const errorBanner = document.getElementById("error-banner");
const levelMeterFill = document.getElementById("level-meter-fill");
const deviceSelect = document.getElementById("device-select");
const silenceInput = document.getElementById("silence-input");
const partialInput = document.getElementById("partial-input");
const interruptInput = document.getElementById("interrupt-input");
const interruptLabel = document.getElementById("interrupt-label");
const transportHint = document.getElementById("transport-hint");

let audioContext = null;
let playbackContext = null;
let mediaStream = null;
let sourceNode = null;
let processorNode = null;
let monitorGainNode = null;
let websocket = null;
let sessionId = null;
let isStreaming = false;
let currentAssistantBubble = null;
let audioQueue = [];
let audioPlaying = false;
let userStoppedMic = false;
let currentPlaybackSource = null;
let browserSpeechFallbackTimer = null;
let lastAssistantText = "";
let inputSampleRate = 16000;
let localSpeechChunks = [];
let localSpeechActive = false;
let localSpeechMs = 0;
let localSilenceMs = 0;
let fallbackTurnTimer = null;
let fallbackInFlight = false;
let awaitingBackendTurn = false;
let browserConfig = {
  transport_mode: "turn",
  realtime_enabled: false,
};
let sessionRequest = null;

const stateLabels = {
  idle: "Idle",
  listening: "Listening",
  thinking: "Thinking",
  speaking: "Speaking",
  interrupted: "Interrupted",
  error: "Error",
};

function showError(message) {
  errorBanner.hidden = !message;
  errorBanner.textContent = message || "";
}

function isRealtimeEnabled() {
  return Boolean(browserConfig.realtime_enabled);
}

function applyTransportMode() {
  const isRealtime = isRealtimeEnabled();
  if (transportPill) {
    transportPill.textContent = isRealtime ? "Realtime Experimental" : "Turn Mode";
  }
  if (transportHint) {
    transportHint.textContent = isRealtime
      ? "Live websocket streaming is enabled for this browser session."
      : "Tap once to start listening. The browser will send each finished utterance as a voice turn.";
  }
  if (interruptInput) {
    interruptInput.disabled = !isRealtime;
  }
  if (interruptLabel) {
    interruptLabel.textContent = isRealtime ? "Interrupt Threshold (ms)" : "Interrupt Threshold (realtime only)";
  }
}

function setMicLevel(level) {
  if (!levelMeterFill) {
    return;
  }
  const normalized = Math.max(0, Math.min(1, level));
  levelMeterFill.style.width = `${Math.round(normalized * 100)}%`;
}

function clearFallbackTurnTimer() {
  if (fallbackTurnTimer) {
    window.clearTimeout(fallbackTurnTimer);
    fallbackTurnTimer = null;
  }
}

function downsampleBuffer(input, inputRate, outputRate) {
  if (!input || !input.length) {
    return new Int16Array(0);
  }
  if (!inputRate || inputRate <= outputRate) {
    const pcm = new Int16Array(input.length);
    for (let index = 0; index < input.length; index += 1) {
      const sample = Math.max(-1, Math.min(1, input[index]));
      pcm[index] = sample < 0 ? sample * 0x8000 : sample * 0x7fff;
    }
    return pcm;
  }

  const ratio = inputRate / outputRate;
  const outputLength = Math.max(1, Math.round(input.length / ratio));
  const output = new Int16Array(outputLength);
  let offsetResult = 0;
  let offsetBuffer = 0;

  while (offsetResult < outputLength) {
    const nextOffsetBuffer = Math.min(input.length, Math.round((offsetResult + 1) * ratio));
    let accumulator = 0;
    let count = 0;
    for (let index = offsetBuffer; index < nextOffsetBuffer; index += 1) {
      accumulator += input[index];
      count += 1;
    }
    const sample = count ? accumulator / count : input[Math.min(offsetBuffer, input.length - 1)];
    const clamped = Math.max(-1, Math.min(1, sample));
    output[offsetResult] = clamped < 0 ? clamped * 0x8000 : clamped * 0x7fff;
    offsetResult += 1;
    offsetBuffer = nextOffsetBuffer;
  }
  return output;
}

function clearSpeechFallbackTimer() {
  if (browserSpeechFallbackTimer) {
    window.clearTimeout(browserSpeechFallbackTimer);
    browserSpeechFallbackTimer = null;
  }
}

function pcmToFloat32(pcm) {
  const output = new Float32Array(pcm.length);
  for (let index = 0; index < pcm.length; index += 1) {
    output[index] = pcm[index] / 32768;
  }
  return output;
}

function mergeFloat32Chunks(chunks) {
  const totalLength = chunks.reduce((sum, chunk) => sum + chunk.length, 0);
  const merged = new Float32Array(totalLength);
  let offset = 0;
  for (const chunk of chunks) {
    merged.set(chunk, offset);
    offset += chunk.length;
  }
  return merged;
}

function encodeWav(samples, sampleRate = 16000) {
  const buffer = new ArrayBuffer(44 + samples.length * 2);
  const view = new DataView(buffer);
  const writeString = (offset, value) => {
    for (let index = 0; index < value.length; index += 1) {
      view.setUint8(offset + index, value.charCodeAt(index));
    }
  };

  writeString(0, "RIFF");
  view.setUint32(4, 36 + samples.length * 2, true);
  writeString(8, "WAVE");
  writeString(12, "fmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, 1, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true);
  writeString(36, "data");
  view.setUint32(40, samples.length * 2, true);

  let offset = 44;
  for (let index = 0; index < samples.length; index += 1) {
    const sample = Math.max(-1, Math.min(1, samples[index]));
    view.setInt16(offset, sample < 0 ? sample * 0x8000 : sample * 0x7fff, true);
    offset += 2;
  }
  return new Blob([buffer], { type: "audio/wav" });
}

async function ensurePlaybackContext() {
  if (!playbackContext) {
    playbackContext = new (window.AudioContext || window.webkitAudioContext)();
  }
  if (playbackContext.state === "suspended") {
    await playbackContext.resume();
  }
  return playbackContext;
}

function setStatus(state) {
  statusPill.dataset.state = state;
  statusPill.textContent = stateLabels[state] || state;
}

function persistChat() {
  const messages = Array.from(chatLog.querySelectorAll(".message")).map((node) => ({
    role: node.classList.contains("user") ? "user" : "assistant",
    text: node.querySelector("div")?.textContent || "",
  }));
  localStorage.setItem("voicebot_chat_log", JSON.stringify(messages));
}

async function playAssistantText(text) {
  const response = await fetch(`/sessions/${sessionId}/tts?text=${encodeURIComponent(text)}`);
  if (!response.ok || response.status === 204) {
    if ("speechSynthesis" in window && text) {
      const utterance = new SpeechSynthesisUtterance(text);
      window.speechSynthesis.cancel();
      window.speechSynthesis.speak(utterance);
      setStatus("speaking");
    }
    return;
  }

  const audioBlob = await response.blob();
  const arrayBuffer = await audioBlob.arrayBuffer();
  audioQueue = [arrayBuffer];
  void playNextAudioChunk();
}

async function loadBrowserConfig() {
  try {
    const response = await fetch("/browser-config", { cache: "no-store" });
    if (response.ok) {
      browserConfig = await response.json();
    }
  } catch (error) {
    console.error(error);
  }
  applyTransportMode();
}

function restoreChat() {
  const saved = localStorage.getItem("voicebot_chat_log");
  if (!saved) {
    return;
  }
  try {
    const messages = JSON.parse(saved);
    for (const message of messages) {
      addMessage(message.role, message.text);
    }
  } catch (error) {
    console.error(error);
  }
}

function addMessage(role, text) {
  const container = document.createElement("article");
  container.className = `message ${role}`;

  const title = document.createElement("strong");
  title.textContent = role === "user" ? "You" : "Assistant";

  const body = document.createElement("div");
  body.textContent = text;

  container.appendChild(title);
  container.appendChild(body);
  chatLog.appendChild(container);
  chatLog.scrollTop = chatLog.scrollHeight;
  persistChat();
  return body;
}

async function createSession() {
  if (sessionRequest) {
    return sessionRequest;
  }
  sessionRequest = (async () => {
    const response = await fetch("/sessions", { method: "POST" });
    const data = await response.json();
    sessionId = data.session_id;
    sessionLabel.textContent = `Session ${sessionId.slice(0, 8)}`;
    setStatus("idle");
    return sessionId;
  })();
  try {
    return await sessionRequest;
  } finally {
    sessionRequest = null;
  }
}

async function resetSession() {
  if (websocket && websocket.readyState <= WebSocket.OPEN) {
    websocket.close();
  }
  chatLog.innerHTML = "";
  localStorage.removeItem("voicebot_chat_log");
  liveCaption.textContent = "Partial transcript will appear here while you speak.";
  currentAssistantBubble = null;
  audioQueue = [];
  player.pause();
  player.removeAttribute("src");
  stopPlaybackQueue();
  clearFallbackTurnTimer();
  fallbackInFlight = false;
  awaitingBackendTurn = false;
  localSpeechChunks = [];
  localSpeechActive = false;
  localSpeechMs = 0;
  localSilenceMs = 0;
  setMicLevel(0);
  await cleanupLiveMic(false);
  await createSession();
  setStatus("idle");
}

async function loadDevices() {
  try {
    const devices = await navigator.mediaDevices.enumerateDevices();
    const inputs = devices.filter((device) => device.kind === "audioinput");
    deviceSelect.innerHTML = "";
    for (const device of inputs) {
      const option = document.createElement("option");
      option.value = device.deviceId;
      option.textContent = device.label || `Microphone ${deviceSelect.length + 1}`;
      deviceSelect.appendChild(option);
    }
  } catch (error) {
    console.error(error);
  }
}

function currentSettings() {
  return {
    silence_ms: Number(silenceInput.value || 600),
    partial_interval_ms: Number(partialInput.value || 700),
    interrupt_threshold_ms: Number(interruptInput.value || 200),
  };
}

function sendSettings() {
  if (isRealtimeEnabled() && websocket && websocket.readyState === WebSocket.OPEN) {
    websocket.send(JSON.stringify({ action: "configure", settings: currentSettings() }));
  }
}

function connectWebSocket() {
  if (!isRealtimeEnabled()) {
    return Promise.resolve();
  }
  if (!sessionId) {
    return Promise.reject(new Error("Missing session id"));
  }
  if (websocket && websocket.readyState <= 1) {
    return Promise.resolve();
  }

  return new Promise((resolve, reject) => {
    const protocol = window.location.protocol === "https:" ? "wss" : "ws";
    websocket = new WebSocket(`${protocol}://${window.location.host}/ws/${sessionId}`);
    websocket.binaryType = "arraybuffer";

    websocket.addEventListener("open", () => {
      showError("");
      sendSettings();
      resolve();
    }, { once: true });

    websocket.addEventListener("error", (event) => {
      reject(event);
    }, { once: true });

    websocket.addEventListener("message", async (event) => {
      const payload = JSON.parse(event.data);
      switch (payload.type) {
        case "ready":
          liveCaption.textContent = "Live connection ready.";
          break;
        case "realtime_disabled":
          showError("Realtime streaming is disabled for this browser. Using turn mode.");
          return;
        case "session_invalid":
          showError("Session expired. Click Start Listening again.");
          if (websocket) {
            websocket.close();
          }
          await createSession();
          return;
        case "status":
          setStatus(payload.state);
          if (payload.state === "listening") {
            currentAssistantBubble = null;
          }
          break;
        case "settings":
          if (payload.silence_ms) {
            silenceInput.value = payload.silence_ms;
          }
          if (payload.partial_interval_ms) {
            partialInput.value = payload.partial_interval_ms;
          }
          if (payload.interrupt_threshold_ms) {
            interruptInput.value = payload.interrupt_threshold_ms;
          }
          break;
        case "partial_transcript":
          if (fallbackInFlight) {
            break;
          }
          awaitingBackendTurn = false;
          clearFallbackTurnTimer();
          liveCaption.textContent = payload.text;
          stopPlaybackQueue();
          break;
        case "final_transcript":
          if (fallbackInFlight) {
            break;
          }
          awaitingBackendTurn = false;
          clearFallbackTurnTimer();
          liveCaption.textContent = "";
          addMessage("user", payload.text);
          currentAssistantBubble = null;
          break;
        case "assistant_chunk":
          if (fallbackInFlight) {
            break;
          }
          if (!currentAssistantBubble) {
            currentAssistantBubble = addMessage("assistant", payload.text);
          } else {
            currentAssistantBubble.textContent = `${currentAssistantBubble.textContent} ${payload.text}`.trim();
            persistChat();
          }
          chatLog.scrollTop = chatLog.scrollHeight;
          break;
        case "assistant_message":
          if (fallbackInFlight) {
            break;
          }
          awaitingBackendTurn = false;
          clearFallbackTurnTimer();
          lastAssistantText = payload.text || "";
          if (!currentAssistantBubble) {
            currentAssistantBubble = addMessage("assistant", payload.text);
          } else {
            currentAssistantBubble.textContent = payload.text;
            persistChat();
          }
          clearSpeechFallbackTimer();
          browserSpeechFallbackTimer = window.setTimeout(() => {
            if (!audioPlaying && lastAssistantText && "speechSynthesis" in window) {
              const utterance = new SpeechSynthesisUtterance(lastAssistantText);
              window.speechSynthesis.cancel();
              window.speechSynthesis.speak(utterance);
            }
          }, 700);
          break;
        case "assistant_audio_chunk":
          if (fallbackInFlight) {
            break;
          }
          clearSpeechFallbackTimer();
          enqueueAudio(payload.audio_b64, payload.mime_type);
          break;
        case "assistant_audio_end":
          break;
        case "assistant_audio_unavailable":
          setStatus("idle");
          showError("Assistant audio could not be generated for this turn.");
          break;
        case "interrupt":
          stopPlaybackQueue();
          setStatus("interrupted");
          break;
        default:
          break;
      }
    });

    websocket.addEventListener("close", () => {
      websocket = null;
      if (!isStreaming) {
        setStatus("idle");
      } else {
        showError("Live connection closed. Click Start Listening again.");
        setStatus("idle");
        isStreaming = false;
        micButton.classList.remove("recording");
        micButton.textContent = "Start Listening";
        void cleanupLiveMic(false);
      }
    });
  });
}

function enqueueAudio(audioB64, mimeType) {
  const binary = atob(audioB64);
  const bytes = new Uint8Array(binary.length);
  for (let index = 0; index < binary.length; index += 1) {
    bytes[index] = binary.charCodeAt(index);
  }
  audioQueue.push(bytes.buffer.slice(0));
  if (!audioPlaying) {
    void playNextAudioChunk();
  }
}

function stopPlaybackQueue() {
  audioQueue = [];
  audioPlaying = false;
  clearSpeechFallbackTimer();
  if ("speechSynthesis" in window) {
    window.speechSynthesis.cancel();
  }
  if (currentPlaybackSource) {
    try {
      currentPlaybackSource.stop();
    } catch (error) {
      console.error(error);
    }
    currentPlaybackSource.disconnect();
    currentPlaybackSource = null;
  }
  player.removeAttribute("src");
}

async function playNextAudioChunk() {
  if (!audioQueue.length) {
    audioPlaying = false;
    if (isRealtimeEnabled() && websocket && websocket.readyState === WebSocket.OPEN) {
      websocket.send(JSON.stringify({ action: "playback_finished" }));
    }
    return;
  }

  audioPlaying = true;
  try {
    const context = await ensurePlaybackContext();
    const chunk = audioQueue.shift();
    const decoded = await context.decodeAudioData(chunk.slice(0));
    const source = context.createBufferSource();
    source.buffer = decoded;
    source.connect(context.destination);
    currentPlaybackSource = source;
    setStatus("speaking");
    source.onended = () => {
      if (currentPlaybackSource === source) {
        currentPlaybackSource.disconnect();
        currentPlaybackSource = null;
      }
      void playNextAudioChunk();
    };
    source.start();
  } catch (error) {
    console.error(error);
    showError("Assistant audio playback failed. Trying next chunk.");
    audioPlaying = false;
    void playNextAudioChunk();
  }
}

async function cleanupLiveMic(updateUi = true) {
  if (processorNode) {
    processorNode.disconnect();
    processorNode.onaudioprocess = null;
    processorNode = null;
  }
  if (sourceNode) {
    sourceNode.disconnect();
    sourceNode = null;
  }
  if (monitorGainNode) {
    monitorGainNode.disconnect();
    monitorGainNode = null;
  }
  if (mediaStream) {
    mediaStream.getTracks().forEach((track) => track.stop());
    mediaStream = null;
  }
  if (audioContext) {
    await audioContext.close();
    audioContext = null;
  }
  if (updateUi) {
    isStreaming = false;
    micButton.classList.remove("recording");
    micButton.textContent = "Start Listening";
    setStatus("idle");
  }
  setMicLevel(0);
}

async function submitFallbackTurn(samples) {
  if (fallbackInFlight || !samples || !samples.length) {
    return;
  }
  fallbackInFlight = true;
  awaitingBackendTurn = false;
  clearFallbackTurnTimer();
  setStatus("thinking");
  liveCaption.textContent = "Processing your speech...";

  try {
    const formData = new FormData();
    formData.append("file", encodeWav(samples, 16000), "utterance.wav");
    const response = await fetch(`/sessions/${sessionId}/audio`, {
      method: "POST",
      body: formData,
    });
    if (!response.ok) {
      const errorPayload = await response.json().catch(() => ({}));
      throw new Error(errorPayload.error || `Audio upload failed: ${response.status}`);
    }
    const data = await response.json();
    if (!data.ok) {
      throw new Error(data.error || "Speech recognition failed.");
    }
    liveCaption.textContent = "";
    if (data.transcript) {
      addMessage("user", data.transcript);
    }
    currentAssistantBubble = null;
    if (data.response) {
      currentAssistantBubble = addMessage("assistant", data.response);
      await playAssistantText(data.response);
    } else {
      setStatus("idle");
    }
  } catch (error) {
    console.error(error);
    showError(error.message || "Speech reached the browser, but backend transcription failed.");
    setStatus("error");
  } finally {
    fallbackInFlight = false;
  }
}

function finalizeLocalSpeech(force = false) {
  if (!localSpeechChunks.length) {
    return;
  }
  const utterance = mergeFloat32Chunks(localSpeechChunks);
  localSpeechChunks = [];
  localSpeechActive = false;
  localSpeechMs = 0;
  localSilenceMs = 0;

  if (!utterance.length || (!force && utterance.length < 1600)) {
    return;
  }

  if (!isRealtimeEnabled()) {
    void submitFallbackTurn(utterance);
    return;
  }

  awaitingBackendTurn = true;
  clearFallbackTurnTimer();
  fallbackTurnTimer = window.setTimeout(() => {
    if (awaitingBackendTurn) {
      void submitFallbackTurn(utterance);
    }
  }, 900);
}

function handleLocalSpeechChunk(chunk, rms) {
  const chunkMs = (chunk.length / 16000) * 1000;
  const localThreshold = 0.004;
  const isSpeech = rms >= localThreshold;

  if (isSpeech) {
    localSpeechActive = true;
    localSilenceMs = 0;
    localSpeechMs += chunkMs;
    localSpeechChunks.push(chunk);
    liveCaption.textContent = "Hearing you...";
    return;
  }

  if (!localSpeechActive) {
    return;
  }

  localSpeechChunks.push(chunk);
  localSilenceMs += chunkMs;
  if (localSilenceMs >= Number(silenceInput.value || 600) || localSpeechMs >= 10000) {
    finalizeLocalSpeech(false);
  }
}

async function startLiveMic() {
  if (!sessionId) {
    await createSession();
  }
  userStoppedMic = false;
  clearFallbackTurnTimer();
  localSpeechChunks = [];
  localSpeechActive = false;
  localSpeechMs = 0;
  localSilenceMs = 0;
  awaitingBackendTurn = false;
  fallbackInFlight = false;

  mediaStream = await navigator.mediaDevices.getUserMedia({
    audio: {
      deviceId: deviceSelect.value ? { exact: deviceSelect.value } : undefined,
      channelCount: 1,
      noiseSuppression: true,
      echoCancellation: true,
      autoGainControl: true,
    },
  });

  await loadDevices();
  await connectWebSocket();

  audioContext = new (window.AudioContext || window.webkitAudioContext)();
  inputSampleRate = audioContext.sampleRate || 16000;
  await audioContext.resume();
  await ensurePlaybackContext();
  sourceNode = audioContext.createMediaStreamSource(mediaStream);
  processorNode = audioContext.createScriptProcessor(2048, 1, 1);
  monitorGainNode = audioContext.createGain();
  monitorGainNode.gain.value = 0;
  liveCaption.textContent = `Live mic active: ${deviceSelect.options[deviceSelect.selectedIndex]?.text || "default device"} (${Math.round(inputSampleRate)} Hz)`;

  const noMicTimer = window.setTimeout(() => {
    if (isStreaming && !localSpeechActive) {
      showError("No microphone input detected. Check your selected device.");
    }
  }, 3000);

  processorNode.onaudioprocess = (event) => {
    if (isRealtimeEnabled() && (!websocket || websocket.readyState !== WebSocket.OPEN)) {
      return;
    }
    const input = event.inputBuffer.getChannelData(0);
    let sumSquares = 0;
    for (let index = 0; index < input.length; index += 1) {
      sumSquares += input[index] * input[index];
    }
    const rms = Math.sqrt(sumSquares / Math.max(1, input.length));
    if (rms > 0.002) {
      window.clearTimeout(noMicTimer);
      showError("");
    }
    setMicLevel(Math.min(1, rms * 35));
    const pcm = downsampleBuffer(input, inputSampleRate, 16000);
    const floatChunk = pcmToFloat32(pcm);
    handleLocalSpeechChunk(floatChunk, rms);
    if (isRealtimeEnabled()) {
      websocket.send(pcm.buffer);
    }
  };

  sourceNode.connect(processorNode);
  processorNode.connect(monitorGainNode);
  monitorGainNode.connect(audioContext.destination);

  isStreaming = true;
  micButton.classList.add("recording");
  micButton.textContent = "Stop Listening";
  setStatus("listening");
}

async function stopLiveMic() {
  userStoppedMic = true;
  if (isRealtimeEnabled() && websocket && websocket.readyState === WebSocket.OPEN) {
    websocket.send(JSON.stringify({ action: "flush_turn" }));
  }
  finalizeLocalSpeech(true);
  await cleanupLiveMic(true);
}

micButton.addEventListener("click", async () => {
  try {
    showError("");
    if (!isStreaming) {
      await startLiveMic();
    } else {
      await stopLiveMic();
    }
  } catch (error) {
    console.error(error);
    showError("Microphone access failed. Check browser permissions and selected device.");
    setStatus("idle");
    sessionLabel.textContent = "Microphone error";
  }
});

resetButton.addEventListener("click", async () => {
  try {
    await resetSession();
  } catch (error) {
    console.error(error);
    showError("Could not reset the session.");
  }
});

applyButton.addEventListener("click", () => {
  sendSettings();
  liveCaption.textContent = "Updated listening settings applied.";
});

window.addEventListener("beforeunload", () => {
  if (isRealtimeEnabled() && websocket && websocket.readyState <= 1) {
    websocket.close();
  }
});

if (navigator.mediaDevices?.addEventListener) {
  navigator.mediaDevices.addEventListener("devicechange", () => {
    void loadDevices();
  });
}

(async () => {
  try {
    restoreChat();
    await loadBrowserConfig();
    await createSession();
    await navigator.mediaDevices.getUserMedia({ audio: true }).then((stream) => {
      stream.getTracks().forEach((track) => track.stop());
      liveCaption.textContent = "Microphone permission granted.";
    }).catch(() => {
      showError("Allow microphone access in the browser, then click Start Listening.");
    });
    await loadDevices();
  } catch (error) {
    console.error(error);
    sessionLabel.textContent = "Failed to create session";
    showError("Initial setup failed. Refresh the page and try again.");
  }
})();
