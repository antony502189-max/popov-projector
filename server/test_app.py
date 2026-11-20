import base64
from contextlib import contextmanager
import io
import sys
import threading
import time
import wave
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient
import numpy as np
import pytest

from server.app import create_app
from server.providers import audio_payload, Provider

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Исходники"))
from remote import Bridge
from service import Player


class FakeProvider:
    def __init__(self):
        self.calls = []
        self.broken = False

    def answer(self, question, history, language):
        self.calls.append((question, history, language))
        if self.broken:
            raise RuntimeError("Provider unavailable")
        return "A radio receiver detects radio waves."

    def speech(self, text, language, voice):
        self.calls.append((language, voice))
        return payload()

    def transcribe(self, raw, filename, language):
        self.calls.append((filename, language))
        return "How does radio work?"


def payload():
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(2); wav.setsampwidth(2); wav.setframerate(24000)
        signal = (np.sin(np.arange(4800)*.12)*12000).astype("<i2")
        wav.writeframes(np.repeat(signal, 2).tobytes())
    return audio_payload(buffer.getvalue())


@contextmanager
def running():
    provider = FakeProvider()
    app = create_app(":memory:", provider, "q"*32, "s"*32, "http://127.0.0.1:8000")
    with TestClient(app) as client:
        client.station = {"Authorization": "Bearer " + "s"*32}
        client.provider = provider
        yield client


def admission(client):
    return client.post("/station/poll", headers=client.station, json={"ready": True}).json()["url"].split("#join=")[1]


def start(client, code=None, identity="browser-identity-1", language="en", voice="nova"):
    result = client.post("/api/start", json={"admission": code or admission(client),
                         "client_id": identity, "language": language, "voice": voice})
    assert result.status_code == 200, result.text
    return {"Authorization": "Bearer " + result.json()["token"]}


def ask(client, user, key="question-1", text="How does radio work?"):
    return client.post("/api/question", headers=user, json={"text": text, "key": key})


def wait_job(client):
    deadline = time.monotonic()+4
    while time.monotonic() < deadline:
        response = client.post("/station/poll", headers=client.station, json={"ready": True}).json()
        if response.get("job"):
            return response["job"]
        time.sleep(.03)
    raise AssertionError("Job not delivered")


def test_full_question_speech_ack_and_history():
    with running() as c:
        user = start(c)
        assert ask(c, user).status_code == 200
        assert ask(c, user).status_code == 200
        assert ask(c, user, text="Different text").status_code == 409
        assert ask(c, user, key="question-2").status_code == 409
        job = wait_job(c)
        assert len(base64.b64decode(job["pcm"])) == 4410*2
        assert job["sample_rate"] == 22050
        for state in ("prepared", "started", "finished", "finished"):
            assert c.post("/station/jobs/"+job["id"], headers=c.station, json={"state": state}).status_code == 200
        assert c.post("/station/jobs/"+job["id"], headers=c.station, json={"state": "started"}).status_code == 409
        result = c.get("/api/session", headers=user).json()
        assert result["turns"][0]["state"] == "finished"
        assert result["language"] == "en"
        assert c.provider.calls[1] == ("en", "nova")
        assert ask(c, user, key="question-2", text="Why?").status_code == 200
        wait_job(c)
        assert c.provider.calls[2][1][0]["answer"] == job["text"]


def test_qr_rotation_revokes_sending_and_old_code_but_keeps_reading():
    with running() as c, patch("server.app.time.time", return_value=12001):
        code = admission(c)
        user = start(c, code)
        with patch("server.app.time.time", return_value=12121):
            fresh = admission(c)
            assert fresh != code
            assert c.post("/api/start", json={"admission": code, "client_id": "browser-identity-1"}).status_code == 401
            assert ask(c, user).status_code == 401
            assert c.get("/api/session", headers=user).json()["can_send"] is False
            assert c.post("/api/start", json={"admission": fresh, "client_id": "browser-identity-1"}).status_code == 200


def test_auth_forgery_origin_and_offline():
    with running() as c:
        assert c.post("/station/poll", json={"ready": True}).status_code == 401
        assert c.get("/api/session").status_code == 401
        assert c.post("/api/start", json={"admission": "forged", "client_id": "browser-identity-1"}).status_code == 401
        user = start(c)
        assert c.post("/api/question", headers={**user, "Origin": "https://attacker.example"}, json={"key": "question-1", "text": "Hello"}).status_code == 403
        c.post("/station/poll", headers=c.station, json={"ready": False})
        assert ask(c, user).status_code == 503


def test_start_is_idempotent_and_language_validated():
    with running() as c:
        code = admission(c)
        assert start(c, code) == start(c, code)
        assert len(c.app.state.store.rows("SELECT * FROM sessions")) == 1
        assert c.post("/api/start", json={"admission": code, "client_id": "browser-identity-1", "language": "xx"}).status_code == 422


def test_heartbeat_timeout_and_transcription():
    with running() as c:
        user = start(c, language="de")
        result = c.post("/api/transcribe", headers=user, files={"recording": ("question.webm", b"sample")})
        assert result.status_code == 200
        assert c.provider.calls[-1] == ("question.webm", "de")
        with patch("server.app.time.time", return_value=time.time()+11):
            assert c.get("/api/site").json()["online"] is False


def test_failed_provider_does_not_queue_or_speak():
    with running() as c:
        user = start(c)
        c.provider.broken = True
        ask(c, user)
        deadline = time.monotonic()+3
        while time.monotonic() < deadline:
            state = c.get("/api/session", headers=user).json()["turns"][0]["state"]
            if state == "failed":
                break
            time.sleep(.03)
        assert state == "failed"
        result = c.get("/api/session", headers=user).json()
        assert result["remaining"] == 3
        assert result["turns"][0]["error"]
        c.provider.broken = False
        assert ask(c, user, key="question-retry").status_code == 200
        wait_job(c)


def test_speech_failure_preserves_the_text_answer():
    with running() as c:
        user = start(c)
        def failed_speech(*args):
            raise RuntimeError("Speech unavailable")
        c.provider.speech = failed_speech
        ask(c, user)
        deadline = time.monotonic()+3
        while time.monotonic() < deadline:
            turn = c.get("/api/session", headers=user).json()["turns"][0]
            if turn["state"] == "failed":
                break
            time.sleep(.03)
        assert turn["state"] == "failed"
        assert turn["answer"] == "A radio receiver detects radio waves."


def test_station_uses_mobile_shell_without_exposing_secrets():
    with running() as c:
        shell = c.get("/popov.html").text
        assert shell == c.get("/").text
        assert 'name="gorod-station"' in shell
        assert "./design.css" in shell and "./app.js" in shell
        assert "'unsafe-eval'" not in c.get("/popov.html").headers["content-security-policy"]
        assert c.get("/popov.html?settings").status_code == 200
        assert c.get("/.env").status_code == 404
        assert c.get("/station.env").status_code == 404
        assert c.get("/assets/popov.webp").status_code == 200
        assert c.get("/popov.html").headers["referrer-policy"] == "no-referrer"


def test_bridge_real_player_protocol(monkeypatch):
    monkeypatch.setenv("POPOV_SERVER_URL", "http://127.0.0.1:8000")
    monkeypatch.setenv("POPOV_STATION_KEY", "s"*32)
    player = Player()
    messages = []
    def emit(message, pcm=b""):
        messages.append((message, pcm))
    with running() as c, patch("service.output", side_effect=emit):
        worker = threading.Thread(target=player.run, daemon=True)
        worker.start()
        deadline = time.monotonic()+2
        while not player.ready and time.monotonic() < deadline:
            time.sleep(.01)
        bridge = Bridge(player, lambda msg: emit(msg))
        def post(path, data):
            result = c.post(path, headers=c.station, json=data)
            result.raise_for_status()
            return result.json()
        bridge.post = post
        bridge.tick()
        user = start(c)
        ask(c, user)
        wait_job(c)
        bridge.tick()
        bridge.tick()  # duplicate delivery never adds a second local task
        deadline = time.monotonic()+2
        while not any(m[0]["type"] == "play" for m in messages) and time.monotonic() < deadline:
            time.sleep(.01)
        plays = [(m, pcm) for m, pcm in messages if m["type"] == "play"]
        assert len(plays) == 1
        msg, pcm = plays[0]
        assert pcm and msg["fps"] == 50 and msg["last"] is True
        for state in ("started", "finished"):
            player.handle({"command": state, "id": msg["id"]}, internal=True)
            bridge.tick()
        assert c.get("/api/session", headers=user).json()["turns"][0]["state"] == "finished"
        player.exit = True
        with player.lock:
            player.lock.notify_all()
        worker.join(timeout=1)


def test_restart_fails_ambiguous_jobs(tmp_path):
    path = tmp_path / "state.sqlite3"
    app = create_app(path, FakeProvider(), "q"*32, "s"*32, "http://localhost:8000")
    app.state.store.db.execute("INSERT INTO jobs(id,state,created,updated) VALUES('old','started',1,1)")
    app.state.store.db.close()
    resumed = create_app(path, FakeProvider(), "q"*32, "s"*32, "http://localhost:8000")
    assert resumed.state.store.rows("SELECT state FROM jobs")[0]["state"] == "failed"


def test_audio_alignment_and_bounds():
    data = payload()
    track = data["track"]
    assert track["seconds"] == .2
    assert track["cues"][0][0] == 0
    assert track["cues"][-1][1] == pytest.approx(.2)
    assert len(track["energy"]) == len(track["accent"])
    assert all(0 <= e <= 1 for e in track["energy"])
    with pytest.raises((wave.Error, EOFError)):
        audio_payload(b"invalid")


def test_unicode_admission_rejected_without_server_error():
    with running() as c:
        assert c.post("/api/start", json={"admission": "подделка", "client_id": "browser-identity-1"}).status_code == 401


def test_strict_venue_network_mode(monkeypatch):
    monkeypatch.setenv("VISITOR_NETWORKS", "203.0.113.0/24")
    with running() as c:
        code = admission(c)
        assert c.post("/api/start", json={"admission": code, "client_id": "browser-identity-1"}).status_code == 403
        # Network restriction applies to visitor writes, not authenticated Mac polling.
        assert c.post("/station/poll", headers=c.station, json={"ready": True}).status_code == 200


def test_global_queue_limit_cannot_be_bypassed_with_new_sessions():
    with running() as c:
        users = [start(c, identity="browser-identity-"+str(i)) for i in range(7)]
        for user in users[:6]:
            assert ask(c, user).status_code == 200
        assert ask(c, users[6]).status_code == 429


def test_change_language_between_questions_and_keep_busy_answer_language():
    with running() as c:
        user = start(c, language="en")
        first = c.post("/api/question", headers=user, json={"key": "question-1", "text": "What is radio?", "language": "de"})
        assert first.status_code == 200
        job = wait_job(c)
        assert c.provider.calls[0][2] == "de"
        rejected = c.post("/api/question", headers=user, json={"key": "question-2", "text": "Why?", "language": "fr"})
        assert rejected.status_code == 409
        assert c.get("/api/session", headers=user).json()["language"] == "de"
        c.post("/station/jobs/"+job["id"], headers=c.station, json={"state": "finished"})
        second = c.post("/api/question", headers=user, json={"key": "question-2", "text": "Why?", "language": "fr"})
        assert second.status_code == 200
        wait_job(c)
        assert c.provider.calls[2][2] == "fr"


def test_transcription_uses_selected_language_and_rejects_unknown_language():
    with running() as c:
        user = start(c, language="ru")
        result = c.post("/api/transcribe", headers=user, data={"language": "en"}, files={"recording": ("voice.wav", b"sample")})
        assert result.status_code == 200
        assert c.provider.calls[-1] == ("voice.wav", "en")
        assert c.post("/api/transcribe", headers=user, data={"language": "xx"}, files={"recording": ("voice.wav", b"sample")}).status_code == 422
