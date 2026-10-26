"""Voice Security Tester backend.

Two independent checks, both must pass:
  1. Speaker identity : SpeechBrain ECAPA-TDNN embeddings + cosine similarity vs enrolled profile.
  2. Spoken password  : faster-whisper ASR -> normalised text -> compared to enrolled phrase hash.
"""
import difflib
import hashlib
import hmac
import json
import logging
import os
import re
import tempfile
import threading
import unicodedata

import librosa
import numpy as np
import torch
from faster_whisper import WhisperModel
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

BASE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE, "data")
os.makedirs(DATA_DIR, exist_ok=True)
PROFILE_PATH = os.path.join(DATA_DIR, "profile.json")
KEY_PATH = os.path.join(DATA_DIR, "secret.key")
USED_PATH = os.path.join(DATA_DIR, "used_audio.txt")

SR = 16000
MIN_SAMPLES = 3
MAX_SAMPLES = 10
MIN_SPEECH_SEC = 1.0
MAX_SEC = 15.0
# Cosine-similarity threshold. NOT a probability. Calibrate with your own genuine/impostor recordings.
THRESHOLD = float(os.environ.get("VOICE_THRESHOLD", "0.50"))
ENROLL_CONSISTENCY = 0.35  # new enrollment sample must be at least this close to earlier ones
ASR_MODEL = os.environ.get("ASR_MODEL", "small")  # multilingual; use base.en for English-only phrases
_lang = os.environ.get("ASR_LANGUAGE", "en").strip().lower()
ASR_LANG = None if _lang in ("", "auto") else _lang

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("voice")

# index.html sits in the project root (no static/ folder). static_folder=None stops Flask from
# exposing a /static route; "/" below serves exactly one file (index.html) and nothing else,
# so data/secret.key and data/profile.json are never reachable over HTTP.
app = Flask(__name__, static_folder=None)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024
CORS(app, resources={r"/api/*": {"origins": [
    r"http://127.0.0.1:5001", r"http://localhost:5001",
    r"http://127.0.0.1:5500", r"http://localhost:5500"]}})

lock = threading.Lock()
_classifier = None
_asr = None


# ---------------------------------------------------------------- models
def load_models():
    global _classifier, _asr
    try:  # SpeechBrain >= 1.0
        from speechbrain.inference.speaker import EncoderClassifier
    except ImportError:  # SpeechBrain 0.5.x
        from speechbrain.inference.speaker import EncoderClassifier
    src = "speechbrain/spkrec-ecapa-voxceleb"
    save = os.path.join(BASE, "models", "ecapa")
    kwargs = dict(source=src, savedir=save, run_opts={"device": "cpu"})
    try:  # Windows without symlink rights needs COPY
        from speechbrain.utils.fetching import LocalStrategy
        _classifier = EncoderClassifier.from_hparams(local_strategy=LocalStrategy.COPY, **kwargs)
    except (ImportError, TypeError):
        _classifier = EncoderClassifier.from_hparams(**kwargs)
    _asr = WhisperModel(ASR_MODEL, device="cpu", compute_type="int8")
    log.info("Models loaded (ECAPA-TDNN + whisper %s)", ASR_MODEL)


class AudioError(Exception):
    pass


# ---------------------------------------------------------------- helpers
def normalize(text):
    text = unicodedata.normalize("NFKC", text or "").lower()
    # keep letters, combining marks (Indic matras) and digits; drop punctuation/symbols
    text = "".join(c if (unicodedata.category(c)[0] in "LMN" or c.isspace()) else " " if c in "-_" else ""
                   for c in text)
    return re.sub(r"\s+", " ", text).strip()


def secret():
    if not os.path.exists(KEY_PATH):
        with open(KEY_PATH, "wb") as f:
            f.write(os.urandom(32))
    with open(KEY_PATH, "rb") as f:
        return f.read()


def phrase_hash(norm, salt):
    return hashlib.pbkdf2_hmac("sha256", norm.encode(), salt, 200_000).hex()


def sign(body):
    return hmac.new(secret(), json.dumps(body, sort_keys=True).encode(), "sha256").hexdigest()


def load_profile():
    """Returns profile dict or None. Raises on tampering/corruption (never silently passes)."""
    if not os.path.exists(PROFILE_PATH):
        return None
    with open(PROFILE_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    mac = data.pop("hmac", "")
    if not hmac.compare_digest(mac, sign(data)):
        raise RuntimeError("Profile file failed integrity check. Reset and re-enroll.")
    return data


def save_profile(p):
    body = dict(p)
    body.pop("hmac", None)
    body["hmac"] = sign(body)
    tmp = PROFILE_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(body, f)
    os.replace(tmp, PROFILE_PATH)


def used_hashes():
    if not os.path.exists(USED_PATH):
        return set()
    with open(USED_PATH) as f:
        return {l.strip() for l in f if l.strip()}


def read_upload():
    """Save upload to temp file, load as 16 kHz mono float32, validate, cleanup. Returns (audio, sha256)."""
    f = request.files.get("audio")
    if f is None:
        raise AudioError("No audio file provided.")
    raw = f.read()
    if len(raw) < 1000:
        raise AudioError("Audio is empty or too short.")
    digest = hashlib.sha256(raw).hexdigest()
    suffix = os.path.splitext(f.filename or "")[1] or ".wav"
    fd, path = tempfile.mkstemp(suffix=suffix)
    try:
        with os.fdopen(fd, "wb") as t:
            t.write(raw)
        try:
            y, _ = librosa.load(path, sr=SR, mono=True)
        except Exception:
            raise AudioError("Could not decode audio. Use the page's recorder (16 kHz WAV).")
    finally:
        if os.path.exists(path):
            os.remove(path)
    if len(y) / SR > MAX_SEC:
        raise AudioError(f"Recording longer than {int(MAX_SEC)} seconds.")
    if y.size == 0 or float(np.max(np.abs(y))) < 0.01:
        raise AudioError("Recording is silent. Check your microphone.")
    trimmed, _ = librosa.effects.trim(y, top_db=30)
    if len(trimmed) / SR < MIN_SPEECH_SEC:
        raise AudioError("Speech too short or unclear. Say the full phrase clearly.")
    trimmed = (trimmed / (np.max(np.abs(trimmed)) + 1e-9)).astype(np.float32)
    return trimmed, digest


def embed(y):
    with torch.no_grad():
        e = _classifier.encode_batch(torch.from_numpy(y).unsqueeze(0)).squeeze().cpu().numpy()
    return e / (np.linalg.norm(e) + 1e-10)


def transcribe(y):
    segs, _ = _asr.transcribe(y, language=ASR_LANG, beam_size=5, condition_on_previous_text=False)
    return " ".join(s.text for s in segs).strip()


def centroid(embs):
    c = np.mean(np.array(embs), axis=0)
    return c / (np.linalg.norm(c) + 1e-10)


def err(msg, code=400, status="ERROR"):
    return jsonify({"success": False, "status": status, "error": msg, "message": msg}), code


# ---------------------------------------------------------------- routes
@app.route("/")
def index():
    # Lets you open http://127.0.0.1:5001/ directly (same origin as the API: no CORS, no Live Server).
    if os.path.exists(os.path.join(BASE, "index.html")):
        return send_from_directory(BASE, "index.html")
    if os.path.exists(os.path.join(BASE, "static", "index.html")):
        return send_from_directory(os.path.join(BASE, "static"), "index.html")
    return err("index.html not found next to app.py.", 404, "NO_FRONTEND")


@app.route("/api/status")
def status():
    try:
        p = load_profile()
    except Exception as e:
        return err(str(e), 500)
    return jsonify({"samples_count": len(p["embeddings"]) if p else 0, "required": MIN_SAMPLES,
                    "name": p["name"] if p else None, "threshold": THRESHOLD})


@app.route("/api/save-sample", methods=["POST"])
def save_sample():
    name = (request.form.get("name") or "").strip()[:60]
    password = request.form.get("password") or ""
    norm_pw = normalize(password)
    if not name:
        return err("Enter a name.")
    if len(norm_pw) < 3:
        return err("Enter a password phrase (at least 3 characters).")
    try:
        y, digest = read_upload()
    except AudioError as e:
        return err(str(e), 400, "BAD_AUDIO")
    try:
        with lock:
            p = load_profile()
            if p and len(p["embeddings"]) >= MAX_SAMPLES:
                return err(f"Maximum {MAX_SAMPLES} samples. Reset to start over.")
            if p and not hmac.compare_digest(p["phrase_hash"], phrase_hash(norm_pw, bytes.fromhex(p["salt"]))):
                return err("Password phrase differs from the enrolled one. Reset to change it.")
            heard = normalize(transcribe(y))
            if heard != norm_pw:
                return err(f'Heard "{heard or "nothing"}". Say the exact phrase, then try again.', 422, "WRONG_PASSWORD")
            e = embed(y)
            if p and len(p["embeddings"]) >= 2:
                sim = float(np.dot(centroid(p["embeddings"]), e))
                if sim < ENROLL_CONSISTENCY:
                    return err("This sample does not sound like the earlier samples. Same speaker only.", 422)
            if not p:
                salt = os.urandom(16)
                p = {"name": name, "salt": salt.hex(), "phrase_hash": phrase_hash(norm_pw, salt), "embeddings": []}
            p["name"] = name
            p["embeddings"].append(e.tolist())
            save_profile(p)
            with open(USED_PATH, "a") as f:
                f.write(digest + "\n")
            n = len(p["embeddings"])
    except Exception as ex:
        log.exception("enroll failed")
        return err(f"Enrollment error: {ex}", 500)
    return jsonify({"status": "success", "samples_count": n, "required": MIN_SAMPLES})


@app.route("/api/reset", methods=["POST"])
def reset():
    """Deletes enrolled voiceprints + phrase hash (profile.json) and the replay list (used_audio.txt).
    secret.key is deliberately kept: it only signs profile.json, and a new profile will be signed with it."""
    try:
        with lock:
            for pth in (PROFILE_PATH, USED_PATH):
                if os.path.exists(pth):
                    os.remove(pth)
            remaining = os.path.exists(PROFILE_PATH)
    except OSError as ex:
        log.exception("reset failed")
        return err(f"Could not delete profile: {ex}", 500)
    if remaining:
        return err("Profile file still exists after reset.", 500)
    return jsonify({"success": True, "status": "reset success", "samples_count": 0})


@app.route("/api/verify", methods=["POST"])
def verify():
    password = request.form.get("password") or ""
    norm_pw = normalize(password)
    try:
        p = load_profile()
    except Exception as e:
        return err(str(e), 500)
    if not p or len(p["embeddings"]) < MIN_SAMPLES:
        return err(f"Enroll at least {MIN_SAMPLES} samples first.", 400, "NOT_ENROLLED")
    try:
        y, digest = read_upload()
    except AudioError as e:
        return err(str(e), 400, "BAD_AUDIO")
    try:
        with lock:
            if digest in used_hashes():
                return err("This exact recording was already used. Record a fresh sample (replay protection).",
                           400, "REPLAY")
            with open(USED_PATH, "a") as f:
                f.write(digest + "\n")
        # Check 1: speaker identity
        score = float(np.dot(centroid(p["embeddings"]), embed(y)))
        speaker_ok = score >= THRESHOLD
        # Check 2: password phrase (ASR) -- the typed phrase must also be the enrolled phrase
        heard_raw = transcribe(y)
        heard = normalize(heard_raw)
        salt = bytes.fromhex(p["salt"])
        # The SPOKEN phrase (ASR text) must match the enrolled phrase hash. Typing it is not needed.
        phrase_ok = bool(heard) and hmac.compare_digest(p["phrase_hash"], phrase_hash(heard, salt))
        if norm_pw and not hmac.compare_digest(p["phrase_hash"], phrase_hash(norm_pw, salt)):
            phrase_ok = False  # if a typed phrase is supplied it must also be the enrolled one
    except Exception as ex:
        log.exception("verify failed")
        return err(f"Verification error: {ex}", 500)

    if speaker_ok and phrase_ok:
        st, msg = "SUCCESS", "Registered speaker and correct password."
    elif speaker_ok:
        st, msg = "WRONG_PASSWORD", "Wrong password phrase."
    elif phrase_ok:
        st, msg = "OTHER_SPEAKER", "Another person is speaking."
    else:
        st, msg = "REJECTED", "Speaker not recognised and wrong phrase."
    return jsonify({
        "success": bool(speaker_ok and phrase_ok), "status": st, "message": msg, "name": p["name"],
        "speakerMatch": speaker_ok, "phraseMatch": phrase_ok,
        "speakerScore": round(score, 4), "threshold": THRESHOLD,
        "voicePct": int(round(max(0.0, score) * 100)),  # raw cosine x100, NOT a probability
        "passPct": 100 if phrase_ok else 0,
        "recognizedText": heard_raw})


if __name__ == "__main__":
    load_models()
    app.run(host="127.0.0.1", port=5001, debug=False)