import asyncio
import json
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from ai.llm import GroqLLM
from ai.stt import SpeechToText
from audio.vad import VoiceActivityDetector
from service.metrics import MetricsRegistry
from service.realtime import RealtimeConversation
from service.session import SessionManager
from tts.speaker import TTSSpeaker
from utils.config import get_config


def create_app():
    from fastapi import FastAPI, File, HTTPException, Response, UploadFile, WebSocket, WebSocketDisconnect
    from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
    from fastapi.staticfiles import StaticFiles
    from pydantic import BaseModel

    config = get_config()
    metrics = MetricsRegistry()
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
    speaker = TTSSpeaker(
        engine=config["TTS_PROVIDER"],
        voice=config["TTS_VOICE"],
        piper_command=config["PIPER_COMMAND"],
        piper_model_path=config["PIPER_MODEL_PATH"],
        piper_config_path=config["PIPER_CONFIG_PATH"],
    )
    live_vad = VoiceActivityDetector(
        provider=config["VAD_PROVIDER"],
        auto_calibrate=config["AUTO_CALIBRATE_NOISE"],
        samplerate=config["AUDIO_SAMPLE_RATE"],
    )
    sessions = SessionManager(
        llm=llm,
        stt=stt,
        metrics=metrics,
        max_messages=12,
        max_long_term=config["MAX_LONG_TERM_SUMMARIES"],
    )
    static_dir = Path(config["BASE_DIR"]) / "web"

    class MessageIn(BaseModel):
        text: str

    @asynccontextmanager
    async def lifespan(app):
        app.state.metrics = metrics
        app.state.sessions = sessions
        app.state.config = config
        app.state.speaker = speaker
        yield

    app = FastAPI(title="Voice Bot Service", version="1.0.0", lifespan=lifespan)

    if static_dir.exists():
        app.mount("/static", StaticFiles(directory=static_dir), name="static")

    @app.get("/")
    async def home():
        index_path = static_dir / "index.html"
        if not index_path.exists():
            raise HTTPException(status_code=404, detail="Frontend not found")
        return FileResponse(index_path, headers={"Cache-Control": "no-store, no-cache, must-revalidate"})

    @app.get("/browser-config")
    async def browser_config():
        return {
            "transport_mode": "realtime" if config["ENABLE_BROWSER_REALTIME"] else "turn",
            "realtime_enabled": config["ENABLE_BROWSER_REALTIME"],
        }

    @app.get("/favicon.ico")
    async def favicon():
        return Response(status_code=204)

    @app.get("/health")
    async def health():
        metrics.increment("health_checks")
        return {
            "status": "ok",
            "stt_backend": stt.backend_name,
            "groq_model": config["GROQ_MODEL"],
            "wake_word": config["WAKE_WORD"],
            "session_count": len(sessions.list_sessions()),
        }

    @app.get("/metrics")
    async def get_metrics():
        return JSONResponse(metrics.snapshot())

    @app.post("/sessions")
    async def create_session():
        session = sessions.create()
        logging.info("[API] Created session %s", session.session_id)
        return {"session_id": session.session_id, "state": session.controller.get_state().value}

    @app.get("/sessions")
    async def list_sessions():
        return {"sessions": sessions.list_sessions()}

    @app.delete("/sessions/{session_id}")
    async def delete_session(session_id: str):
        if not sessions.delete(session_id):
            raise HTTPException(status_code=404, detail="Session not found")
        return {"deleted": True}

    @app.post("/sessions/{session_id}/reset")
    async def reset_session(session_id: str):
        session = sessions.get(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        return session.reset()

    @app.post("/sessions/{session_id}/message")
    async def message(session_id: str, payload: MessageIn):
        session = sessions.get(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        return session.chat(payload.text)

    @app.get("/sessions/{session_id}/stream")
    async def stream_message(session_id: str, text: str):
        session = sessions.get(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")

        def event_stream():
            for chunk in session.stream_chat(text):
                yield f"data: {json.dumps({'chunk': chunk})}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    @app.post("/sessions/{session_id}/audio")
    async def audio_message(session_id: str, file: UploadFile = File(...)):
        session = sessions.get(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        payload = await file.read()
        result = session.chat_from_audio(payload)
        status_code = 200 if result.get("ok", True) else 422
        return JSONResponse(result, status_code=status_code)

    @app.get("/sessions/{session_id}/tts")
    async def speak_text(session_id: str, text: str):
        session = sessions.get(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        audio_bytes = speaker.synthesize_bytes(text)
        if not audio_bytes:
            return Response(status_code=204)
        return Response(content=audio_bytes, media_type="audio/wav")

    @app.websocket("/ws/{session_id}")
    async def websocket_session(websocket: WebSocket, session_id: str):
        if not config["ENABLE_BROWSER_REALTIME"]:
            await websocket.accept()
            await websocket.send_json({"type": "realtime_disabled"})
            await websocket.close(code=1000)
            return

        session = sessions.get(session_id)
        if not session:
            await websocket.accept()
            await websocket.send_json({"type": "session_invalid", "session_id": session_id})
            await websocket.close(code=1008)
            return

        await websocket.accept()
        conversation = RealtimeConversation(
            session=session,
            speaker=speaker,
            vad=live_vad,
            chunk_ms=config["CHUNK_MS"],
            sample_rate=config["AUDIO_SAMPLE_RATE"],
            silence_ms=min(700, config["END_OF_UTTERANCE_SILENCE_MS"]),
            partial_interval_ms=700,
        )
        conversation.start()
        conversation.configure({})

        async def sender():
            try:
                while True:
                    event = await asyncio.to_thread(conversation.events.get)
                    await websocket.send_json(event)
            except asyncio.CancelledError:
                raise
            except Exception:
                return

        sender_task = asyncio.create_task(sender())
        try:
            while True:
                message = await websocket.receive()
                if message.get("type") == "websocket.disconnect":
                    break
                if "bytes" in message and message["bytes"] is not None:
                    conversation.handle_audio_chunk(message["bytes"])
                    continue
                if message.get("text"):
                    payload = json.loads(message["text"])
                    action = payload.get("action")
                    if action == "playback_finished":
                        conversation.mark_playback_finished()
                    elif action == "flush_turn":
                        conversation.flush_current_turn()
                    elif action == "reset":
                        session.reset()
                        await websocket.send_json({"type": "status", "state": "listening"})
                    elif action == "configure":
                        conversation.configure(payload.get("settings", {}))
        except WebSocketDisconnect:
            pass
        except RuntimeError:
            pass
        except Exception:
            conversation.stop()
            sender_task.cancel()
            raise
        finally:
            conversation.stop()
            sender_task.cancel()
            try:
                await sender_task
            except (asyncio.CancelledError, Exception):
                pass

    return app
