"""SystemOne-compatible HTTP API over the openjev JSONL encoder."""

from __future__ import annotations

import argparse
import base64
import hashlib
import hmac
import json
import math
import secrets
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from http.cookies import SimpleCookie
from typing import Any, TypedDict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

from openjev import OpenJevCrossEncoder, laya_payload, laya_answers

ROOT = Path(__file__).resolve().parent
HF_REPO = "AlexWortega/openjev"
LATEST_ID = "jev-latest"
KEV_LATEST = "kev-latest"
MAX_BODY = 16 * 1024 * 1024
MAX_CHOICE = 255
MAX_ENTRY_DEPTH = 32
MAX_IMAGE_BYTES = 12 * 1024 * 1024
IMAGE_SUFFIX = {
    "image/jpeg": ".jpg",
    "image/jpg": ".jpg",
    "image/webp": ".webp",
    "image/gif": ".gif",
    "image/png": ".png",
}

CATALOG = {
    "laya": {
        "family": "laya",
        "hub": "convaiinnovations/laya",
        "f16": "laya-f16.gguf",
        "q4": "laya-Q4_K_M.gguf",
        "head": "laya.head.gguf",
    },
    "openjev_0.8b": {
        "family": "openjev",
        "hf": "qwen3.5-0.8b-nli-v2s-long",
        "f16": "openjev-0.8b-f16.gguf",
        "q4": "openjev-0.8b-Q4_K_M.gguf",
        "mmproj": "openjev-0.8b-mmproj-f16.gguf",
    },
    "openjev_4b": {
        "family": "openjev",
        "hf": "qwen3.5-4b-nli-v2",
        "f16": "openjev-4b-f16.gguf",
        "q4": "openjev-4b-Q4_K_M.gguf",
        "mmproj": "openjev-4b-mmproj-f16.gguf",
    },
    "openjev_35b": {
        "family": "openjev",
        "hf": "qwen3.5-35b-a3b-nli",
        "f16": "openjev-35b-f16.gguf",
        "q4": "openjev-35b-Q4_K_M.gguf",
        "mmproj": "openjev-35b-mmproj-f16.gguf",
    },
    "kev_0.5b": {
        "family": "kev",
        "hub": "jaredpalmer/kev-0.5b",
        "f16": "kev-0.5b-f16.gguf",
        "q4": "kev-0.5b-Q4_K_M.gguf",
        "head": "kev-0.5b.head.bin",
    },
    "kev_0.8b": {
        "family": "kev",
        "hub": "jaredpalmer/kev-0.8b",
        "f16": "kev-0.8b-f16.gguf",
        "q4": "kev-0.8b-Q4_K_M.gguf",
        "head": "kev-0.8b.head.bin",
    },
    "kev_4b": {
        "family": "kev",
        "hub": "jaredpalmer/kev-4b",
        "f16": "kev-4b-f16.gguf",
        "q4": "kev-4b-Q4_K_M.gguf",
        "head": "kev-4b.head.bin",
    },
    "kev_9b": {
        "family": "kev",
        "hub": "jaredpalmer/kev-9b",
        "f16": "kev-9b-f16.gguf",
        "q4": "kev-9b-Q4_K_M.gguf",
        "head": "kev-9b.head.bin",
    },
}

ALIASES = {
    "laya-latest": "laya",
    LATEST_ID: "openjev_4b",
    KEV_LATEST: "kev_4b",
    "openjev-0.8b": "openjev_0.8b",
    "openjev-4b": "openjev_4b",
    "openjev-35b": "openjev_35b",
    "jev-0.8b": "openjev_0.8b",
    "jev-4b": "openjev_4b",
    "jev-35b": "openjev_35b",
    "kev-0.5b": "kev_0.5b",
    "kev-0.8b": "kev_0.8b",
    "kev-4b": "kev_4b",
    "kev-9b": "kev_9b",
}


class RequestError(ValueError):
    def __init__(self, message, status=400):
        super().__init__(message)
        self.status = status


PBKDF2_ITERS = 200000
SESSION_TTL = 86400
SAMPLE_KEEP = 300
ADMIN_FILE = ROOT / "openjev-admin.json"


def _pbkdf2(password, salt):
    return hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, PBKDF2_ITERS)


def hash_password(password, salt=None):
    salt = salt or secrets.token_bytes(16)
    return salt, _pbkdf2(password, salt)


def verify_password(password, salt, digest):
    return hmac.compare_digest(_pbkdf2(password, salt), digest)


class AdminAccount(TypedDict):
    username: str
    salt: str
    hash: str
    iters: int


class ApiKey(TypedDict):
    id: str
    name: str
    prefix: str
    hash: str
    created: int
    requests: int
    input_tokens: int
    output_tokens: int
    elapsed_ms: float
    last_used: int | None


class UsageTotals(TypedDict):
    requests: int
    input_tokens: int
    output_tokens: int
    elapsed_ms: float


class AdminData(TypedDict):
    admin: AdminAccount | None
    secret: str
    keys: list[ApiKey]
    samples: list[Any]
    totals: UsageTotals


class AdminStore:
    def __init__(self, path):
        self.path = Path(path)
        self._lock = threading.Lock()
        self._data: AdminData = {
            "admin": None,
            "secret": secrets.token_hex(32),
            "keys": [],
            "samples": [],
            "totals": {"requests": 0, "input_tokens": 0, "output_tokens": 0, "elapsed_ms": 0.0},
        }
        self._load()

    def _load(self):
        if not self.path.is_file():
            return
        raw = json.loads(self.path.read_text(encoding="utf-8"))
        if isinstance(raw, dict):
            self._data.update(raw)
            self._data.setdefault("keys", [])
            self._data.setdefault("samples", [])
            self._data.setdefault("totals", {"requests": 0, "input_tokens": 0, "output_tokens": 0, "elapsed_ms": 0.0})
            self._data.setdefault("secret", secrets.token_hex(32))

    def _save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(self._data), encoding="utf-8")
        tmp.replace(self.path)

    def needs_setup(self):
        return self._data.get("admin") is None

    def setup(self, username, password):
        username = (username or "").strip()
        if not username or len(username) > 64 or any(ch in username for ch in " \t\r\n:"):
            raise RequestError("username must be one token, 1 to 64 characters")
        if not isinstance(password, str) or len(password) < 8:
            raise RequestError("password must be at least 8 characters")
        with self._lock:
            if self._data.get("admin"):
                raise RequestError("admin already exists", 409)
            salt, digest = hash_password(password)
            self._data["admin"] = {
                "username": username,
                "salt": salt.hex(),
                "hash": digest.hex(),
                "iters": PBKDF2_ITERS,
            }
            self._save()
            return self._issue_session(username)

    def login(self, username, password):
        with self._lock:
            admin = self._data.get("admin")
            if not admin:
                raise RequestError("admin is not set up", 409)
            salt = bytes.fromhex(admin["salt"])
            digest = bytes.fromhex(admin["hash"])
            user_ok = hmac.compare_digest(admin["username"], (username or "").strip())
            pass_ok = isinstance(password, str) and verify_password(password, salt, digest)
            if not (user_ok and pass_ok):
                raise RequestError("invalid username or password", 401)
            return self._issue_session(admin["username"])

    def _issue_session(self, username):
        exp = int(time.time()) + SESSION_TTL
        payload = f"{username}:{exp}".encode("utf-8")
        sig = hmac.new(bytes.fromhex(self._data["secret"]), payload, hashlib.sha256).hexdigest()
        return base64.urlsafe_b64encode(payload).decode("ascii").rstrip("=") + "." + sig

    def session_user(self, token):
        if not token or "." not in token:
            return None
        blob, sig = token.rsplit(".", 1)
        pad = "=" * ((4 - len(blob) % 4) % 4)
        try:
            payload = base64.urlsafe_b64decode(blob + pad)
        except Exception:
            return None
        expect = hmac.new(bytes.fromhex(self._data["secret"]), payload, hashlib.sha256).hexdigest()
        if not hmac.compare_digest(expect, sig):
            return None
        try:
            username, exp_s = payload.decode("utf-8").rsplit(":", 1)
            exp = int(exp_s)
        except ValueError:
            return None
        if exp < int(time.time()):
            return None
        admin = self._data.get("admin")
        if not admin or not hmac.compare_digest(admin["username"], username):
            return None
        return username

    def create_key(self, name=""):
        name = (name or "").strip() or "key"
        if len(name) > 64:
            raise RequestError("key name is too long")
        raw = "oj_" + secrets.token_urlsafe(24)
        record: ApiKey = {
            "id": "k_" + secrets.token_hex(8),
            "name": name,
            "prefix": raw[:10],
            "hash": hashlib.sha256(raw.encode()).hexdigest(),
            "created": int(time.time()),
            "requests": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "elapsed_ms": 0.0,
            "last_used": None,
        }
        with self._lock:
            self._data["keys"].append(record)
            self._save()
        return {"id": record["id"], "name": name, "prefix": record["prefix"], "key": raw}

    def add_known_key(self, raw, name="cli"):
        digest = hashlib.sha256(raw.encode()).hexdigest()
        with self._lock:
            for key in self._data["keys"]:
                if key["hash"] == digest:
                    return
            self._data["keys"].append({
                "id": "k_" + secrets.token_hex(8),
                "name": name,
                "prefix": raw[:10],
                "hash": digest,
                "created": int(time.time()),
                "requests": 0,
                "input_tokens": 0,
                "output_tokens": 0,
                "elapsed_ms": 0.0,
                "last_used": None,
            })
            self._save()

    def revoke_key(self, key_id):
        with self._lock:
            before = len(self._data["keys"])
            self._data["keys"] = [key for key in self._data["keys"] if key["id"] != key_id]
            if len(self._data["keys"]) == before:
                raise RequestError("unknown key", 404)
            self._save()

    def find_key(self, bearer):
        if not bearer or not bearer.startswith("Bearer "):
            return None
        raw = bearer[7:].strip()
        digest = hashlib.sha256(raw.encode()).hexdigest()
        with self._lock:
            for key in self._data["keys"]:
                if hmac.compare_digest(key["hash"], digest):
                    return key
        return None

    def record(self, key_id, input_tokens, output_tokens, elapsed_ms):
        now = time.time()
        with self._lock:
            self._data["totals"]["requests"] += 1
            self._data["totals"]["input_tokens"] += int(input_tokens)
            self._data["totals"]["output_tokens"] += int(output_tokens)
            self._data["totals"]["elapsed_ms"] += float(elapsed_ms)
            for key in self._data["keys"]:
                if key["id"] == key_id:
                    key["requests"] += 1
                    key["input_tokens"] += int(input_tokens)
                    key["output_tokens"] += int(output_tokens)
                    key["elapsed_ms"] += float(elapsed_ms)
                    key["last_used"] = int(now)
                    break
            samples = self._data["samples"]
            samples.append([now, int(input_tokens), float(elapsed_ms), key_id])
            if len(samples) > SAMPLE_KEEP:
                del samples[: len(samples) - SAMPLE_KEEP]
            self._save()

    def _throughput(self, window=60.0):
        now = time.time()
        cutoff = now - window
        tokens = 0
        ms = 0.0
        reqs = 0
        for stamp, tok, elapsed, _key in self._data.get("samples") or []:
            if stamp >= cutoff:
                tokens += int(tok)
                ms += float(elapsed)
                reqs += 1
        return {
            "window_s": window,
            "requests": reqs,
            "req_per_s": reqs / window,
            "input_tokens": tokens,
            "tokens_per_s": tokens / window,
            "eval_tok_per_s": (1000.0 * tokens / ms) if ms else 0.0,
        }

    def public_keys(self):
        with self._lock:
            keys = []
            for key in self._data["keys"]:
                keys.append({
                    "id": key["id"],
                    "name": key["name"],
                    "prefix": key["prefix"],
                    "created": key["created"],
                    "requests": key["requests"],
                    "input_tokens": key["input_tokens"],
                    "output_tokens": key["output_tokens"],
                    "last_used": key["last_used"],
                    "eval_tok_per_s": (1000.0 * key["input_tokens"] / key["elapsed_ms"]) if key["elapsed_ms"] else 0.0,
                })
            totals = dict(self._data["totals"])
            throughput = self._throughput()
        return {"keys": keys, "totals": totals, "throughput": throughput}


def flatten(entry, depth=0):
    if depth > MAX_ENTRY_DEPTH:
        raise RequestError("entry nesting is too deep")
    if isinstance(entry, str):
        return entry.strip()
    if entry is None:
        return ""
    if isinstance(entry, bool):
        return "true" if entry else "false"
    if isinstance(entry, (int, float)):
        return str(entry)
    if isinstance(entry, list):
        parts = [flatten(item, depth + 1) for item in entry]
        return "\n".join(part for part in parts if part)
    if isinstance(entry, dict):
        parts = []
        for key, value in entry.items():
            if value is None:
                continue
            text = flatten(value, depth + 1)
            parts.append(f"{key}: {text}" if text else str(key))
        return "\n".join(parts)
    raise RequestError("entry values must be strings, arrays, or objects")


def _mime_suffix(mime):
    return IMAGE_SUFFIX.get((mime or "").split(";")[0].strip().lower(), ".png")


def _write_image(directory, blob, suffix):
    if not blob:
        raise RequestError("image payload is empty")
    if len(blob) > MAX_IMAGE_BYTES:
        raise RequestError("image is too large")
    path = Path(directory) / f"input{suffix}"
    path.write_bytes(blob)
    return str(path)


def _decode_data_url(text):
    if not isinstance(text, str) or not text.startswith("data:image/"):
        return None
    header, _, blob = text.partition(",")
    if not blob:
        raise RequestError("image data URL is missing data")
    mime = header[5:].split(";")[0]
    try:
        return base64.b64decode(blob, validate=False), _mime_suffix(mime)
    except Exception as exc:
        raise RequestError("image data URL is not valid base64") from exc


IMAGE_OBJECT_KEYS = {
    "type", "data", "media_type", "mime", "url", "image", "image_url", "source",
}


MAX_PATH_CHARS = 1024


def _existing_file(value):
    if not isinstance(value, str) or len(value) > MAX_PATH_CHARS or "\n" in value:
        return None
    try:
        path = Path(value)
        if path.is_file():
            return str(path.resolve())
    except OSError:
        return None
    return None


def _decode_b64_image(text, mime=None):
    if not isinstance(text, str) or len(text) < 8:
        return None
    try:
        blob = base64.b64decode(text, validate=False)
    except Exception:
        return None
    if not blob:
        return None
    if mime:
        return blob, _mime_suffix(mime)
    if blob[:3] == b"\xff\xd8\xff":
        return blob, ".jpg"
    if blob[:8] == b"\x89PNG\r\n\x1a\n":
        return blob, ".png"
    if blob[:4] == b"RIFF" and blob[8:12] == b"WEBP":
        return blob, ".webp"
    return blob, ".png"


def _image_from_mapping(value, directory):
    if not isinstance(value, dict):
        return None
    nested = value.get("source") if isinstance(value.get("source"), dict) else value
    if isinstance(value.get("image_url"), dict):
        nested = {**nested, **value["image_url"]}
    mime = nested.get("media_type") or nested.get("mime") or value.get("media_type") or value.get("mime")
    typed = value.get("type") in {"image", "image_url"} or (
        isinstance(mime, str) and mime.startswith("image/")
    )
    if not typed and (set(value.keys()) - IMAGE_OBJECT_KEYS):
        return None
    if not typed and "data" not in nested and "image" not in value and "url" not in nested:
        return None
    for candidate in (
        nested.get("url"),
        nested.get("data"),
        value.get("data"),
        value.get("url"),
        value.get("image"),
        nested.get("image"),
    ):
        if candidate is None or isinstance(candidate, dict):
            continue
        path = materialize_image(candidate, directory)
        if path:
            return path
        decoded = _decode_b64_image(candidate, mime if isinstance(mime, str) else None)
        if decoded:
            return _write_image(directory, decoded[0], decoded[1])
    if typed:
        raise RequestError("image object is missing data or a file path")
    return None


def materialize_image(value, directory):
    if value is None:
        return None
    if isinstance(value, str):
        decoded = _decode_data_url(value)
        if decoded:
            return _write_image(directory, decoded[0], decoded[1])
        existing = _existing_file(value)
        if existing:
            return existing
        if value.startswith("data:"):
            raise RequestError("only data:image/... URLs are supported")
        return None
    if isinstance(value, dict):
        return _image_from_mapping(value, directory)
    return None


def peel_images(entry, directory, found):
    path = materialize_image(entry, directory)
    if path:
        if found:
            raise RequestError("only one image is supported per request")
        found.append(path)
        return None
    if isinstance(entry, list):
        return [peel_images(item, directory, found) for item in entry]
    if isinstance(entry, dict):
        out = {}
        for key, value in entry.items():
            peeled = peel_images(value, directory, found)
            if peeled is not None:
                out[key] = peeled
        return out
    return entry


def attach_image(payload, image):
    if image:
        payload = dict(payload)
        payload["image"] = image
    return payload


def premise(state, instructions):
    left, right = flatten(state), flatten(instructions)
    if left and right:
        return left + "\n" + right
    if left or right:
        return left or right
    raise RequestError("state and instructions produced empty text")


def softmax(values):
    maximum = max(values)
    weights = [math.exp(value - maximum) for value in values]
    total = sum(weights)
    return [weight / total for weight in weights]


def entailment_logits(response):
    return [row["logits"][1] for row in response["results"]]


def binary_truth(row):
    contradiction, entailment = row["logits"][0], row["logits"][1]
    return softmax([contradiction, entailment])[1]


def confidence(probabilities):
    return max(probabilities.values()) if probabilities else 0.0


def prepare_noul(state, question):
    criteria = question.get("criteria") or {}
    if not isinstance(criteria, dict):
        raise RequestError("noul criteria must be an object")
    true_entry = criteria.get("true")
    false_entry = criteria.get("false")
    if true_entry is not None and false_entry is not None:
        return {
            "premise": premise(state, question["instructions"]),
            "hypotheses": [flatten(true_entry), flatten(false_entry)],
        }, {"type": "noul", "binary": False}
    hypothesis = flatten(true_entry) if true_entry is not None else flatten(question["instructions"])
    return {"pairs": [[flatten(state), hypothesis]]}, {"type": "noul", "binary": True}


def prepare_choice(state, question):
    criteria = question.get("criteria")
    if not isinstance(criteria, dict) or not criteria:
        raise RequestError("choice criteria must be a non-empty object")
    if len(criteria) > MAX_CHOICE:
        raise RequestError(f"choice supports at most {MAX_CHOICE} options")
    keys, options = [], []
    for key, value in criteria.items():
        if value is None:
            continue
        keys.append(str(key))
        options.append(flatten(value))
    if not keys:
        raise RequestError("choice requires at least one non-null option")
    return {
        "question": premise(state, question["instructions"]),
        "options": options,
    }, {"type": "choice", "keys": keys}


def prepare_score(state, question):
    criteria = question.get("criteria")
    if not isinstance(criteria, list) or not 2 <= len(criteria) <= 10:
        raise RequestError("score criteria must be an array of 2 to 10 levels")
    levels = [flatten(item) for item in criteria]
    if any(not level for level in levels):
        raise RequestError("score levels must flatten to non-empty text")
    return {
        "premise": premise(state, question["instructions"]),
        "hypotheses": levels,
    }, {"type": "score", "criteria": criteria}


def decode_question(meta, response):
    kind = meta["type"]
    if kind == "noul":
        value = binary_truth(response["results"][0]) if meta["binary"] else softmax(entailment_logits(response))[0]
        return {"type": "noul", "noul": value}
    if kind == "choice":
        keys = meta["keys"]
        probabilities = dict(zip(keys, softmax(entailment_logits(response))))
        choice = max(keys, key=lambda key: (probabilities[key], -keys.index(key)))
        return {"type": "choice", "choice": choice, "confidence": confidence(probabilities),
                "probabilities": probabilities}
    criteria = meta["criteria"]
    weights = softmax(entailment_logits(response))
    keys = [str(i) for i in range(len(criteria))]
    probabilities = dict(zip(keys, weights))
    return {
        "type": "score",
        "score": sum(i * weight for i, weight in enumerate(weights)),
        "confidence": confidence(probabilities),
        "legend": {key: criteria[int(key)] for key in keys},
        "probabilities": probabilities,
    }


def prepare_question(state, question):
    if not isinstance(question, dict) or "type" not in question or "instructions" not in question:
        raise RequestError("each question needs type and instructions")
    kind = question["type"]
    if kind == "noul":
        return prepare_noul(state, question)
    if kind == "choice":
        return prepare_choice(state, question)
    if kind == "score":
        return prepare_score(state, question)
    raise RequestError(f"unsupported question type: {kind}")


def evaluate_question(jev, state, question, image=None):
    payload, meta = prepare_question(state, question)
    response = jev.request(attach_image(payload, image))
    return decode_question(meta, response), response


def question_pairs(payload):
    if "pairs" in payload:
        return payload["pairs"]
    if "options" in payload:
        return [[payload["question"], "The correct answer is: " + option] for option in payload["options"]]
    return [[payload["premise"], hypothesis] for hypothesis in payload["hypotheses"]]


def known_model_ids():
    return [LATEST_ID, KEV_LATEST, *CATALOG]


def canonical_id(name):
    if not isinstance(name, str) or not name:
        raise RequestError("missing model", 404)
    if name in CATALOG:
        return name
    if name in ALIASES:
        return ALIASES[name]
    raise RequestError(f"unknown model: {name!r}; known ids: {', '.join(known_model_ids())}", 404)


def render_kev(value, indent=0):
    pad = " " * indent
    if value is None:
        return ""
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, (str, int, float)):
        return str(value)
    if isinstance(value, list):
        return "\n".join(f"{pad}- {render_kev(item, indent + 1).lstrip()}" for item in value)
    if isinstance(value, dict):
        parts = []
        for key, item in value.items():
            text = render_kev(item, indent + 1)
            if isinstance(item, (dict, list)):
                parts.append(f"{pad}{key}:\n{text}" if text else f"{pad}{key}:")
            else:
                parts.append(f"{pad}{key}: {text}" if text else f"{pad}{key}:")
        return "\n".join(parts)
    raise RequestError("entry values must be strings, arrays, or objects")


def option_text(name, desc):
    if desc is None or desc == "":
        return str(name)
    return f"{name}: {render_kev(desc)}"


def choice_confidence(probabilities):
    n = len(probabilities)
    return 1.0 if n <= 1 else (max(probabilities) - 1.0 / n) / (1.0 - 1.0 / n)


def score_confidence(probabilities):
    n = len(probabilities)
    mode = max(range(n), key=lambda i: probabilities[i])
    return 1.0 - sum(p * abs(i - mode) for i, p in enumerate(probabilities)) / (n - 1)


def kev_payload(state, questions):
    rec, meta = [], []
    for qid, question in questions.items():
        if not isinstance(question, dict) or "type" not in question or "instructions" not in question:
            raise RequestError("each question needs type and instructions")
        instr = render_kev(question["instructions"])
        kind = question["type"]
        if kind == "noul":
            criteria = question.get("criteria") or {}
            if criteria and not isinstance(criteria, dict):
                raise RequestError("noul criteria must be an object")
            options = [option_text("no", criteria.get("false")), option_text("yes", criteria.get("true"))]
            meta.append({"id": str(qid), "type": "noul"})
        elif kind == "choice":
            criteria = question.get("criteria")
            if not isinstance(criteria, dict) or not criteria:
                raise RequestError("choice criteria must be a non-empty object")
            keys, options = [], []
            for key, value in criteria.items():
                if value is None:
                    continue
                keys.append(str(key))
                options.append(option_text(key, value))
            if not keys:
                raise RequestError("choice requires at least one non-null option")
            if len(keys) > MAX_CHOICE:
                raise RequestError(f"choice supports at most {MAX_CHOICE} options")
            meta.append({"id": str(qid), "type": "choice", "keys": keys})
        elif kind == "score":
            criteria = question.get("criteria")
            if not isinstance(criteria, list) or not 2 <= len(criteria) <= 10:
                raise RequestError("score criteria must be an array of 2 to 10 levels")
            options = [render_kev(item) for item in criteria]
            if any(not level for level in options):
                raise RequestError("score levels must flatten to non-empty text")
            meta.append({
                "id": str(qid), "type": "score",
                "legend": {str(i): criteria[i] for i in range(len(criteria))},
            })
        else:
            raise RequestError(f"unsupported question type: {kind}")
        rec.append({"instr": instr, "options": options})
    return {"state": render_kev(state), "questions": rec}, meta


def kev_answers(probabilities, meta):
    answers = {}
    for probs, item in zip(probabilities, meta):
        if item["type"] == "noul":
            answers[item["id"]] = {"type": "noul", "noul": float(probs[1])}
        elif item["type"] == "choice":
            keys = item["keys"]
            dist = dict(zip(keys, [float(p) for p in probs]))
            choice = max(keys, key=lambda key: (dist[key], -keys.index(key)))
            answers[item["id"]] = {
                "type": "choice", "choice": choice,
                "confidence": choice_confidence(list(dist.values())),
                "probabilities": dist,
            }
        else:
            weights = [float(p) for p in probs]
            keys = [str(i) for i in range(len(weights))]
            answers[item["id"]] = {
                "type": "score",
                "score": sum(i * p for i, p in enumerate(weights)),
                "confidence": score_confidence(weights),
                "legend": item["legend"],
                "probabilities": dict(zip(keys, weights)),
            }
    return answers


def handle_request(get_encoder, payload):
    if not isinstance(payload, dict):
        raise RequestError("request must be an object")
    model = payload.get("model")
    cid = canonical_id(model)
    jev = get_encoder(model)
    if "state" not in payload:
        raise RequestError("missing state")
    questions = payload.get("questions")
    if not isinstance(questions, dict) or not questions:
        raise RequestError("questions must be a non-empty object")
    with tempfile.TemporaryDirectory(prefix="openjev-img-") as tmp:
        found = []
        state = peel_images(payload["state"], tmp, found)
        if state is None:
            state = ""
        top_image = materialize_image(payload.get("image"), tmp)
        if top_image:
            if found:
                raise RequestError("only one image is supported per request")
            found.append(top_image)
        image = found[0] if found else None
        if CATALOG[cid]["family"] == "laya":
            if image:
                raise RequestError("Laya does not score images")
            rec, meta = laya_payload(payload["state"], questions)
            response = jev.request(rec)
            return {"model": model, "answers": laya_answers(response, meta),
                    "usage": {"input_tokens": int(response["evaluated_tokens"]), "output_tokens": 0}}
        if CATALOG[cid]["family"] == "kev":
            if image:
                raise RequestError("kev does not score images; use an openjev model")
            rec, meta = kev_payload(state, questions)
            response = jev.request(rec)
            rows = response.get("results") or []
            if len(rows) != len(meta):
                raise RequestError("kev returned the wrong number of questions", 500)
            return {
                "model": model,
                "answers": kev_answers([row["probabilities"] for row in rows], meta),
                "usage": {"input_tokens": int(response.get("evaluated_tokens") or 0), "output_tokens": 0},
            }
        answers = {}
        if image and len(questions) > 1:
            pairs, groups = [], []
            for qid, question in questions.items():
                rec, meta = prepare_question(state, question)
                group = question_pairs(rec)
                groups.append((str(qid), meta, len(group)))
                pairs.extend(group)
            response = jev.request(attach_image({"pairs": pairs}, image))
            rows = response.get("results") or []
            if len(rows) != len(pairs):
                raise RequestError("encoder returned the wrong number of pairs", 500)
            offset = 0
            for qid, meta, count in groups:
                answers[qid] = decode_question(meta, {"results": rows[offset:offset + count]})
                offset += count
            return {"model": model, "answers": answers,
                    "usage": {"input_tokens": int(response.get("evaluated_tokens") or 0), "output_tokens": 0}}
        input_tokens = 0
        for qid, question in questions.items():
            answer, response = evaluate_question(jev, state, question, image=image)
            answers[str(qid)] = answer
            input_tokens += int(response.get("evaluated_tokens") or 0)
        return {
            "model": model,
            "answers": answers,
            "usage": {"input_tokens": input_tokens, "output_tokens": 0},
        }


def run(command, cwd=None):
    print("+ " + " ".join(command), flush=True)
    subprocess.check_call(command, cwd=cwd)


def download_hf(subfolder, dest_parent):
    dest_parent = Path(dest_parent)
    dest_parent.mkdir(parents=True, exist_ok=True)
    try:
        from huggingface_hub import snapshot_download
        snapshot_download(HF_REPO, allow_patterns=[subfolder + "/*"], local_dir=str(dest_parent))
        return
    except Exception as exc:
        print(f"openjev: huggingface_hub snapshot failed ({exc}); trying hf cli", flush=True)
    hf = shutil.which("hf") or str(ROOT / ".venv" / "bin" / "hf")
    run([hf, "download", HF_REPO, "--include", subfolder + "/*", "--local-dir", str(dest_parent)],
        cwd=str(ROOT))


class ModelHub:
    def __init__(self, root, *, binary, gpu_layers, threads, quantize_bin, inference_timeout=110.0):
        self.root = Path(root)
        self.models_dir = self.root / "models"
        self.hf_dir = self.models_dir / "openjev-hf"
        self.binary = binary
        self.gpu_layers = gpu_layers
        self.threads = threads
        self.quantize_bin = quantize_bin
        self.inference_timeout = inference_timeout
        self._lock = threading.Lock()
        self._loaded = {}
        self._jobs = {}

    def status(self):
        with self._lock:
            loaded = set(self._loaded)
            jobs = {key: dict(value) for key, value in self._jobs.items()}
        rows = []
        for name, spec in CATALOG.items():
            path = self.gguf_path(spec)
            job = jobs.get(name)
            rows.append({
                "id": name,
                "family": spec["family"],
                "source": spec.get("hub") or (HF_REPO + "/" + spec.get("hf", "")),
                "ready": path is not None,
                "file": path.name if path else None,
                "loaded": name in loaded,
                "job": None if job is None else {"state": job["state"], "error": job.get("error")},
            })
        return {
            "models": rows,
            "aliases": {LATEST_ID: "openjev_4b", KEV_LATEST: "kev_4b", "laya-latest": "laya"},
            "ready": self.ready_ids(),
            "loaded": sorted(loaded),
        }

    def start_install(self, name):
        cid = canonical_id(name)
        with self._lock:
            if self.gguf_path(CATALOG[cid]):
                return {"id": cid, "state": "ready"}
            job = self._jobs.get(cid)
            if job and job["state"] == "running":
                return {"id": cid, "state": "running"}
            self._jobs[cid] = {"state": "running", "error": None}
        thread = threading.Thread(target=self._install_job, args=(cid,), daemon=True)
        thread.start()
        return {"id": cid, "state": "running"}

    def _install_job(self, name):
        try:
            self.install([name])
            with self._lock:
                self._jobs[name] = {"state": "ready", "error": None}
        except Exception as exc:
            with self._lock:
                self._jobs[name] = {"state": "error", "error": str(exc)}
            print(f"openjev: install {name} failed: {exc}", flush=True)

    def gguf_path(self, spec):
        q4 = self.models_dir / spec["q4"]
        f16 = self.models_dir / spec["f16"]
        path = q4 if q4.is_file() else f16 if f16.is_file() else None
        if path is None:
            return None
        if spec.get("head") and not (self.models_dir / spec["head"]).is_file():
            return None
        return path

    def mmproj_path(self, spec):
        name = spec.get("mmproj")
        if not name:
            return None
        path = self.models_dir / name
        return path if path.is_file() else None

    def ready_ids(self):
        ready = []
        if self.gguf_path(CATALOG["openjev_4b"]):
            ready.append(LATEST_ID)
        if self.gguf_path(CATALOG["kev_4b"]):
            ready.append(KEV_LATEST)
        if self.gguf_path(CATALOG["laya"]):
            ready.append("laya-latest")
        ready += [name for name, spec in CATALOG.items() if self.gguf_path(spec)]
        return ready

    def install(self, names):
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.hf_dir.mkdir(parents=True, exist_ok=True)
        for name in names:
            cid = ALIASES.get(name, name)
            if cid not in CATALOG:
                raise RequestError(f"unknown model: {name!r}", 404)
            self._install_one(cid)

    def _install_one(self, name):
        spec = CATALOG[name]
        existing = self.gguf_path(spec)
        if existing:
            print(f"openjev: {name} already installed ({existing.name})", flush=True)
            return
        if spec.get("family") == "laya":
            run([sys.executable, str(self.root / "tools" / "openjev" / "convert_laya.py"),
                 "--hub", spec["hub"], "--out-dir", str(self.models_dir),
                 "--f16-name", spec["f16"], "--head-name", spec["head"]], cwd=str(self.root))
            if not self.gguf_path(spec):
                raise RequestError("install did not produce a complete Laya model", 500)
            return
        if spec.get("family") == "kev":
            convert = [
                sys.executable, str(self.root / "tools" / "openjev" / "convert_kev.py"),
                "--hub", spec["hub"], "--out-dir", str(self.models_dir),
                "--f16-name", spec["f16"], "--q4-name", spec["q4"], "--head-name", spec["head"],
            ]
            if self.quantize_bin:
                convert += ["--quantize", self.quantize_bin]
            run(convert, cwd=str(self.root))
            if not self.gguf_path(spec):
                raise RequestError(f"install did not produce a GGUF for {name}", 500)
            print(f"openjev: {name} ready ({self.gguf_path(spec).name})", flush=True)
            return
        hf_src = self.hf_dir / spec["hf"]
        if not (hf_src / "config.json").is_file():
            download_hf(spec["hf"], self.hf_dir)
        if not (hf_src / "config.json").is_file():
            raise RequestError(f"failed to download {spec['hf']} from {HF_REPO}", 500)
        f16 = self.models_dir / spec["f16"]
        convert = [sys.executable, str(self.root / "convert_hf_to_gguf.py"), str(hf_src), "--outtype", "f16"]
        run(convert + ["--outfile", str(f16)], cwd=str(self.root))
        mmproj = self.models_dir / spec["mmproj"]
        if not mmproj.is_file():
            try:
                run(convert + ["--mmproj", "--outfile", str(mmproj)], cwd=str(self.root))
            except subprocess.CalledProcessError as exc:
                print(f"openjev: mmproj convert skipped for {name}: {exc}", flush=True)
        q4 = self.models_dir / spec["q4"]
        if self.quantize_bin and Path(self.quantize_bin).is_file() and f16.is_file() and not q4.is_file():
            run([self.quantize_bin, str(f16), str(q4), "Q4_K_M"], cwd=str(self.root))
        if not self.gguf_path(spec):
            raise RequestError(f"install did not produce a GGUF for {name}", 500)
        print(f"openjev: {name} ready ({self.gguf_path(spec).name})", flush=True)

    def encoder(self, name):
        cid = canonical_id(name)
        with self._lock:
            if cid in self._loaded:
                if self._loaded[cid].is_alive():
                    return self._loaded[cid]
                self._loaded.pop(cid).close()
            spec = CATALOG[cid]
            path = self.gguf_path(spec)
            if path is None:
                raise RequestError(
                    f"model {cid} is not installed; open / and install it from settings",
                    404,
                )
            mmproj = self.mmproj_path(spec)
            ctx_size = 8192 if mmproj else 4096
            print(f"openjev: loading {cid} from {path.name} (ctx {ctx_size})", flush=True)
            self._loaded[cid] = OpenJevCrossEncoder(
                path, binary=self.binary, mmproj=mmproj, ctx_size=ctx_size,
                gpu_layers=self.gpu_layers, threads=self.threads,
                inference_timeout=self.inference_timeout,
            )
            return self._loaded[cid]

    def close(self):
        with self._lock:
            for jev in self._loaded.values():
                jev.close()
            self._loaded.clear()


ADMIN_PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>openjev admin</title>
<style>
body { font-family: ui-sans-serif, system-ui, sans-serif; margin: 24px; max-width: 980px; color: #111; }
h1 { font-size: 20px; margin: 0 0 8px; }
h2 { font-size: 16px; margin: 28px 0 8px; }
p, td, th, label, button { font-size: 14px; }
.muted { color: #555; }
table { border-collapse: collapse; width: 100%; margin-top: 8px; }
th, td { text-align: left; padding: 8px 10px; border-bottom: 1px solid #ddd; vertical-align: top; }
.row { display: flex; gap: 12px; align-items: center; flex-wrap: wrap; margin: 12px 0; }
input { padding: 6px 8px; }
button { cursor: pointer; }
.ok { color: #0a7; } .bad { color: #c00; } .run { color: #a60; }
.cards { display: flex; gap: 12px; flex-wrap: wrap; }
.card { border: 1px solid #ddd; padding: 10px 12px; min-width: 140px; }
.card b { display: block; font-size: 18px; }
.hide { display: none; }
code { word-break: break-all; }
</style>
</head>
<body>
<h1>openjev admin</h1>
<p class="muted" id="blurb"></p>
<section id="gate" class="hide">
  <form id="gate-form" class="row">
    <label>username <input name="username" autocomplete="username" required></label>
    <label>password <input name="password" type="password" autocomplete="new-password" required></label>
    <button type="submit" id="gate-btn">continue</button>
  </form>
</section>
<section id="dash" class="hide">
  <div class="row"><span id="who"></span><button id="logout">log out</button></div>
  <div class="cards" id="cards"></div>
  <h2>API keys</h2>
  <form id="key-form" class="row">
    <label>name <input name="name" placeholder="prod"></label>
    <button type="submit">create key</button>
  </form>
  <p id="newkey" class="ok"></p>
  <table><thead><tr><th>name</th><th>prefix</th><th>requests</th><th>input tokens</th><th>eval tok/s</th><th></th></tr></thead><tbody id="keys"></tbody></table>
  <h2>models</h2>
  <p class="muted" id="note"></p>
  <table><thead><tr><th>id</th><th>family</th><th>disk</th><th>loaded</th><th></th></tr></thead><tbody id="rows"></tbody></table>
</section>
<script>
function $(id) { return document.getElementById(id); }
function cls(state) {
  if (state === 'ready' || state === true) return 'ok';
  if (state === 'error' || state === false) return 'bad';
  if (state === 'running') return 'run';
  return '';
}
async function api(path, opt) {
  const r = await fetch(path, Object.assign({ credentials: 'same-origin' }, opt || {}));
  const data = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(data.error || r.statusText);
  return data;
}
function show(id) {
  $('gate').classList.toggle('hide', id !== 'gate');
  $('dash').classList.toggle('hide', id !== 'dash');
}
async function boot() {
  const s = await api('/v1/admin/state');
  if (s.setup) {
    $('blurb').textContent = 'Create the admin username and password. Password is stored as PBKDF2-HMAC-SHA256 with a random salt.';
    $('gate-btn').textContent = 'create admin';
    $('gate-form').onsubmit = async (e) => {
      e.preventDefault();
      const f = new FormData(e.target);
      try {
        await api('/v1/admin/setup', { method: 'POST', headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ username: f.get('username'), password: f.get('password') }) });
        location.reload();
      } catch (err) { alert(err.message); }
    };
    show('gate');
    return;
  }
  if (!s.auth) {
    $('blurb').textContent = 'Log in to install models, mint API keys, and read usage.';
    $('gate-btn').textContent = 'log in';
    $('gate-form').querySelector('[name=password]').autocomplete = 'current-password';
    $('gate-form').onsubmit = async (e) => {
      e.preventDefault();
      const f = new FormData(e.target);
      try {
        await api('/v1/admin/login', { method: 'POST', headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ username: f.get('username'), password: f.get('password') }) });
        location.reload();
      } catch (err) { alert(err.message); }
    };
    show('gate');
    return;
  }
  $('blurb').textContent = 'Bearer API keys call POST /v1/systemone. Install does not run until you click it.';
  $('who').textContent = 'signed in as ' + s.username;
  $('logout').onclick = async () => { await api('/v1/admin/logout', { method: 'POST' }); location.reload(); };
  $('key-form').onsubmit = async (e) => {
    e.preventDefault();
    const f = new FormData(e.target);
    try {
      const created = await api('/v1/admin/keys', { method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ name: f.get('name') }) });
      $('newkey').textContent = 'copy now (shown once): ' + created.key;
      refresh();
    } catch (err) { alert(err.message); }
  };
  show('dash');
  refresh();
  setInterval(refresh, 2000);
}
async function refresh() {
  const [admin, models] = await Promise.all([api('/v1/admin/dashboard'), api('/v1/settings')]);
  const t = admin.throughput || {};
  const tot = admin.totals || {};
  $('cards').innerHTML =
    card('requests', tot.requests) + card('input tokens', tot.input_tokens) +
    card('req/s (60s)', (t.req_per_s || 0).toFixed(2)) +
    card('tok/s wall (60s)', (t.tokens_per_s || 0).toFixed(1)) +
    card('eval tok/s (60s)', (t.eval_tok_per_s || 0).toFixed(1));
  const keys = $('keys'); keys.innerHTML = '';
  for (const k of admin.keys) {
    const tr = document.createElement('tr');
    tr.innerHTML = '<td>' + k.name + '</td><td><code>' + k.prefix + '...</code></td><td>' + k.requests +
      '</td><td>' + k.input_tokens + '</td><td>' + (k.eval_tok_per_s || 0).toFixed(1) + '</td>';
    const td = document.createElement('td');
    const b = document.createElement('button');
    b.textContent = 'revoke';
    b.onclick = async () => { await api('/v1/admin/keys/revoke', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ id: k.id }) }); refresh(); };
    td.appendChild(b); tr.appendChild(td); keys.appendChild(tr);
  }
  $('note').textContent = 'ready: ' + ((models.ready || []).join(', ') || 'none');
  const body = $('rows'); body.innerHTML = '';
  for (const m of models.models) {
    const tr = document.createElement('tr');
    const job = m.job ? m.job.state + (m.job.error ? ' ' + m.job.error : '') : '';
    const disk = m.ready ? (m.file || 'yes') : (job || 'missing');
    tr.innerHTML = '<td><code>' + m.id + '</code><div class="muted">' + m.source + '</div></td><td>' + m.family +
      '</td><td class="' + cls(m.ready ? 'ready' : (m.job && m.job.state)) + '">' + disk +
      '</td><td class="' + cls(m.loaded) + '">' + (m.loaded ? 'yes' : 'no') + '</td>';
    const td = document.createElement('td');
    const b = document.createElement('button');
    b.textContent = m.ready ? 'installed' : (m.job && m.job.state === 'running' ? 'installing...' : 'install');
    b.disabled = !!(m.ready || (m.job && m.job.state === 'running'));
    b.onclick = () => install(m.id);
    td.appendChild(b); tr.appendChild(td); body.appendChild(tr);
  }
}
function card(label, value) { return '<div class="card"><b>' + value + '</b><span class="muted">' + label + '</span></div>'; }
async function install(id) {
  try { await api('/v1/settings/install', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ model: id }) }); refresh(); }
  catch (err) { alert(err.message); }
}
boot();
</script>
</body>
</html>
"""


def make_handler(hub, store):
    inference_slot = threading.BoundedSemaphore(1)
    jobs = {}
    jobs_lock = threading.Lock()
    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="openjev-inference")

    def run_inference(key, payload):
        started = time.monotonic()
        try:
            with jobs_lock:
                jobs[key]["status"] = "running"
            if not inference_slot.acquire(blocking=False):
                raise RequestError("inference busy; wait for the active request to finish", 429)
            try:
                result = handle_request(hub.encoder, payload)
                usage = result.get("usage") or {}
                return result, usage, (time.monotonic() - started) * 1000.0
            finally:
                inference_slot.release()
        except Exception as exc:
            with jobs_lock:
                jobs[key].update(status="error", error=str(exc))
            return None

    def finish_job(future, job_id, api_key):
        try:
            result = future.result()
            if result is None:
                return
            payload, usage, elapsed = result
            store.record(api_key, usage.get("input_tokens") or 0, usage.get("output_tokens") or 0, elapsed)
            with jobs_lock:
                jobs[job_id].update(status="done", result=payload)
        except Exception as exc:
            with jobs_lock:
                jobs[job_id].update(status="error", error=str(exc))

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        inference_executor = executor

        def log_message(self, format, *args):
            sys.stderr.write("%s - %s\n" % (self.address_string(), format % args))

        def _send(self, status, payload, content_type="application/json; charset=utf-8", cookie=None, clear_cookie=False):
            if isinstance(payload, str):
                body = payload.encode("utf-8")
            else:
                body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Connection", "close")
            if cookie:
                self.send_header("Set-Cookie", "openjev_session=" + cookie + "; HttpOnly; SameSite=Lax; Path=/; Max-Age=86400")
            if clear_cookie:
                self.send_header("Set-Cookie", "openjev_session=; HttpOnly; SameSite=Lax; Path=/; Max-Age=0")
            self.end_headers()
            self.wfile.write(body)

        def _session(self):
            jar = SimpleCookie(self.headers.get("Cookie", ""))
            morsel = jar.get("openjev_session")
            return store.session_user(morsel.value if morsel else "")

        def _require_admin(self):
            user = self._session()
            if not user:
                raise RequestError("unauthorized", 401)
            return user

        def _read_json(self):
            length = self.headers.get("Content-Length")
            try:
                n = int(length or "0")
            except ValueError:
                raise RequestError("invalid Content-Length")
            if n < 1 or n > MAX_BODY:
                raise RequestError("request body is missing or too large")
            try:
                return json.loads(self.rfile.read(n).decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                raise RequestError("invalid JSON")

        def do_GET(self):
            path = urlparse(self.path).path
            if path in ("/", "/settings"):
                self._send(200, ADMIN_PAGE, "text/html; charset=utf-8")
                return
            if path == "/health":
                self._send(200, {"ok": True, "models": hub.ready_ids(), "loaded": sorted(hub._loaded),
                                 "setup": store.needs_setup()})
                return
            if path == "/v1/admin/state":
                if store.needs_setup():
                    self._send(200, {"setup": True, "auth": False})
                    return
                user = self._session()
                self._send(200, {"setup": False, "auth": bool(user), "username": user})
                return
            if path == "/v1/admin/dashboard":
                try:
                    self._require_admin()
                    self._send(200, store.public_keys())
                except RequestError as exc:
                    self._send(exc.status, {"error": str(exc)})
                return
            if path == "/v1/settings":
                try:
                    self._require_admin()
                    self._send(200, hub.status())
                except RequestError as exc:
                    self._send(exc.status, {"error": str(exc)})
                return
            if path == "/v1/models":
                self._send(200, {"data": [{"id": name, "owned_by": "openjev"} for name in known_model_ids()]})
                return
            if path.startswith("/v1/jobs/"):
                job_id = path.removeprefix("/v1/jobs/")
                with jobs_lock:
                    job = jobs.get(job_id)
                    if job is None:
                        self._send(404, {"error": "job not found"})
                    else:
                        response = {key: value for key, value in job.items() if key != "future"}
                        self._send(200, response)
                return
            self._send(404, {"error": "not found"})

        def do_POST(self):
            path = urlparse(self.path).path
            try:
                if path == "/v1/admin/setup":
                    payload = self._read_json()
                    token = store.setup(payload.get("username"), payload.get("password"))
                    self._send(200, {"ok": True}, cookie=token)
                    return
                if path == "/v1/admin/login":
                    payload = self._read_json()
                    token = store.login(payload.get("username"), payload.get("password"))
                    self._send(200, {"ok": True}, cookie=token)
                    return
                if path == "/v1/admin/logout":
                    self._send(200, {"ok": True}, clear_cookie=True)
                    return
                if path == "/v1/admin/keys":
                    self._require_admin()
                    payload = self._read_json()
                    if not isinstance(payload, dict):
                        raise RequestError("request must be an object")
                    self._send(200, store.create_key(payload.get("name")))
                    return
                if path == "/v1/admin/keys/revoke":
                    self._require_admin()
                    payload = self._read_json()
                    store.revoke_key((payload or {}).get("id"))
                    self._send(200, {"ok": True})
                    return
                if path == "/v1/settings/install":
                    self._require_admin()
                    payload = self._read_json()
                    if not isinstance(payload, dict):
                        raise RequestError("request must be an object")
                    self._send(200, hub.start_install(payload.get("model")))
                    return
                if path != "/v1/systemone":
                    self._send(404, {"error": "not found"})
                    return
                key = store.find_key(self.headers.get("Authorization", ""))
                if not key:
                    self._send(401, {"error": "unauthorized"})
                    return
                payload = self._read_json()
                if urlparse(self.path).query == "async=1":
                    job_id = uuid.uuid4().hex
                    with jobs_lock:
                        jobs[job_id] = {"id": job_id, "status": "queued", "created": time.time()}
                    future = executor.submit(run_inference, job_id, payload)
                    with jobs_lock:
                        jobs[job_id]["future"] = future
                    future.add_done_callback(lambda completed, jid=job_id, kid=key["id"]:
                                             finish_job(completed, jid, kid))
                    self._send(202, {"id": job_id, "status": "queued", "poll": "/v1/jobs/" + job_id})
                    return
                if not inference_slot.acquire(blocking=False):
                    raise RequestError("inference busy; wait for the active request to finish", 429)
                try:
                    started = time.monotonic()
                    result = handle_request(hub.encoder, payload)
                    usage = result.get("usage") or {}
                    store.record(key["id"], usage.get("input_tokens") or 0, usage.get("output_tokens") or 0,
                                 (time.monotonic() - started) * 1000.0)
                    self._send(200, result)
                finally:
                    inference_slot.release()
            except (BrokenPipeError, ConnectionResetError):
                self.close_connection = True
            except TimeoutError as exc:
                self._send(504, {"error": str(exc)})
            except RequestError as exc:
                self._send(exc.status, {"error": str(exc)})
            except ValueError as exc:
                self._send(400, {"error": str(exc)})
            except Exception as exc:
                self._send(500, {"error": str(exc)})

    return Handler


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary")
    parser.add_argument("--api-key", help="optional seed API key stored as a hash (named cli in the panel)")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--gpu-layers", type=int, default=99)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--inference-timeout", type=float, default=110.0,
                        help="Kill stalled model requests after this many seconds (default 110)")
    parser.add_argument("--admin-file", default=str(ADMIN_FILE))
    args = parser.parse_args()
    if not math.isfinite(args.inference_timeout) or args.inference_timeout <= 0:
        parser.error("--inference-timeout must be positive and finite")
    binary = args.binary or str(ROOT / "build" / "bin" / "openjev")
    store = AdminStore(args.admin_file)
    if args.api_key:
        store.add_known_key(args.api_key, "cli")
    hub = ModelHub(ROOT, binary=binary, gpu_layers=args.gpu_layers, threads=args.threads,
                   quantize_bin=str(ROOT / "build" / "bin" / "llama-quantize"),
                   inference_timeout=args.inference_timeout)
    handler = make_handler(hub, store)
    try:
        server = ThreadingHTTPServer((args.host, args.port), handler)
        print(f"openjev admin http://{args.host}:{args.port}/", flush=True)
        print(f"openjev systemone http://{args.host}:{args.port}/v1/systemone", flush=True)
        if store.needs_setup():
            print("openjev: create the admin account in the browser", flush=True)
        print("models: " + ", ".join(hub.ready_ids() or ["none"]), flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            server.server_close()
            handler.inference_executor.shutdown(wait=True, cancel_futures=True)
    finally:
        hub.close()


if __name__ == "__main__":
    main()
