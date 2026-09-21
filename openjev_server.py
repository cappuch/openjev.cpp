"""SystemOne-compatible HTTP API over the openjev JSONL encoder."""

from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

from openjev import OpenJevCrossEncoder

ROOT = Path(__file__).resolve().parent
HF_REPO = "AlexWortega/openjev"
LATEST_ID = "jev-latest"
MAX_BODY = 2 * 1024 * 1024
MAX_CHOICE = 255
MAX_ENTRY_DEPTH = 32

CATALOG = {
    "openjev_0.8b": {
        "hf": "qwen3.5-0.8b-nli-v2s-long",
        "f16": "openjev-0.8b-f16.gguf",
        "q4": "openjev-0.8b-Q4_K_M.gguf",
        "mmproj": "openjev-0.8b-mmproj-f16.gguf",
        "auto": True,
    },
    "openjev_4b": {
        "hf": "qwen3.5-4b-nli-v2",
        "f16": "openjev-4b-f16.gguf",
        "q4": "openjev-4b-Q4_K_M.gguf",
        "mmproj": "openjev-4b-mmproj-f16.gguf",
        "auto": True,
    },
    "openjev_35b": {
        "hf": "qwen3.5-35b-a3b-nli",
        "f16": "openjev-35b-f16.gguf",
        "q4": "openjev-35b-Q4_K_M.gguf",
        "mmproj": "openjev-35b-mmproj-f16.gguf",
        "auto": False,
    },
}

ALIASES = {
    LATEST_ID: "openjev_4b",
    "openjev-0.8b": "openjev_0.8b",
    "openjev-4b": "openjev_4b",
    "openjev-35b": "openjev_35b",
    "jev-0.8b": "openjev_0.8b",
    "jev-4b": "openjev_4b",
    "jev-35b": "openjev_35b",
}


class RequestError(ValueError):
    def __init__(self, message, status=400):
        super().__init__(message)
        self.status = status


def flatten(entry, depth=0):
    if depth > MAX_ENTRY_DEPTH:
        raise RequestError("entry nesting is too deep")
    if isinstance(entry, str):
        return entry.strip()
    if isinstance(entry, bool) or entry is None:
        raise RequestError("entry values must be strings, arrays, or objects")
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


def evaluate_noul(jev, state, question):
    criteria = question.get("criteria") or {}
    if not isinstance(criteria, dict):
        raise RequestError("noul criteria must be an object")
    true_entry = criteria.get("true")
    false_entry = criteria.get("false")
    if true_entry is not None and false_entry is not None:
        response = jev.request({
            "premise": premise(state, question["instructions"]),
            "hypotheses": [flatten(true_entry), flatten(false_entry)],
        })
        true_p, _false_p = softmax(entailment_logits(response))
        return {"type": "noul", "noul": true_p}, response
    hypothesis = flatten(true_entry) if true_entry is not None else flatten(question["instructions"])
    response = jev.request({"pairs": [[flatten(state), hypothesis]]})
    return {"type": "noul", "noul": binary_truth(response["results"][0])}, response


def evaluate_choice(jev, state, question):
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
    response = jev.request({
        "question": premise(state, question["instructions"]),
        "options": options,
    })
    probabilities = dict(zip(keys, softmax(entailment_logits(response))))
    choice = max(keys, key=lambda key: (probabilities[key], -keys.index(key)))
    return {
        "type": "choice",
        "choice": choice,
        "confidence": confidence(probabilities),
        "probabilities": probabilities,
    }, response


def evaluate_score(jev, state, question):
    criteria = question.get("criteria")
    if not isinstance(criteria, list) or not 2 <= len(criteria) <= 10:
        raise RequestError("score criteria must be an array of 2 to 10 levels")
    levels = [flatten(item) for item in criteria]
    if any(not level for level in levels):
        raise RequestError("score levels must flatten to non-empty text")
    response = jev.request({
        "premise": premise(state, question["instructions"]),
        "hypotheses": levels,
    })
    weights = softmax(entailment_logits(response))
    keys = [str(i) for i in range(len(levels))]
    probabilities = dict(zip(keys, weights))
    return {
        "type": "score",
        "score": sum(i * weight for i, weight in enumerate(weights)),
        "confidence": confidence(probabilities),
        "legend": {key: criteria[int(key)] for key in keys},
        "probabilities": probabilities,
    }, response


def evaluate_question(jev, state, question):
    if not isinstance(question, dict) or "type" not in question or "instructions" not in question:
        raise RequestError("each question needs type and instructions")
    kind = question["type"]
    if kind == "noul":
        return evaluate_noul(jev, state, question)
    if kind == "choice":
        return evaluate_choice(jev, state, question)
    if kind == "score":
        return evaluate_score(jev, state, question)
    raise RequestError(f"unsupported question type: {kind}")


def known_model_ids():
    return [LATEST_ID, *CATALOG]


def canonical_id(name):
    if not isinstance(name, str) or not name:
        raise RequestError("missing model", 404)
    if name in CATALOG:
        return name
    if name in ALIASES:
        return ALIASES[name]
    raise RequestError(f"unknown model: {name!r}; known ids: {', '.join(known_model_ids())}", 404)


def handle_request(get_encoder, payload):
    if not isinstance(payload, dict):
        raise RequestError("request must be an object")
    model = payload.get("model")
    jev = get_encoder(model)
    if "state" not in payload:
        raise RequestError("missing state")
    questions = payload.get("questions")
    if not isinstance(questions, dict) or not questions:
        raise RequestError("questions must be a non-empty object")
    answers = {}
    input_tokens = 0
    for qid, question in questions.items():
        answer, response = evaluate_question(jev, payload["state"], question)
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
    def __init__(self, root, *, binary, gpu_layers, threads, quantize_bin):
        self.root = Path(root)
        self.models_dir = self.root / "models"
        self.hf_dir = self.models_dir / "openjev-hf"
        self.binary = binary
        self.gpu_layers = gpu_layers
        self.threads = threads
        self.quantize_bin = quantize_bin
        self._lock = threading.Lock()
        self._loaded = {}

    def gguf_path(self, spec):
        q4 = self.models_dir / spec["q4"]
        f16 = self.models_dir / spec["f16"]
        if q4.is_file():
            return q4
        if f16.is_file():
            return f16
        return None

    def mmproj_path(self, spec):
        path = self.models_dir / spec["mmproj"]
        return path if path.is_file() else None

    def ready_ids(self):
        ready = []
        if self.gguf_path(CATALOG["openjev_4b"]):
            ready.append(LATEST_ID)
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
                return self._loaded[cid]
            spec = CATALOG[cid]
            path = self.gguf_path(spec)
            if path is None:
                raise RequestError(
                    f"model {cid} is not installed; start the server once for 0.8B/4B, or pass --install-35b",
                    404,
                )
            mmproj = self.mmproj_path(spec)
            print(f"openjev: loading {cid} from {path.name}", flush=True)
            self._loaded[cid] = OpenJevCrossEncoder(
                path, binary=self.binary, mmproj=mmproj,
                gpu_layers=self.gpu_layers, threads=self.threads,
            )
            return self._loaded[cid]

    def close(self):
        with self._lock:
            for jev in self._loaded.values():
                jev.close()
            self._loaded.clear()


def make_handler(hub, api_key):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, format, *args):
            sys.stderr.write("%s - %s\n" % (self.address_string(), format % args))

        def _send(self, status, payload):
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Connection", "close")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            path = urlparse(self.path).path
            if path == "/health":
                self._send(200, {"ok": True, "models": hub.ready_ids(), "loaded": sorted(hub._loaded)})
                return
            if path == "/v1/models":
                self._send(200, {"data": [{"id": name, "owned_by": "openjev"} for name in known_model_ids()]})
                return
            self._send(404, {"error": "not found"})

        def do_POST(self):
            if urlparse(self.path).path != "/v1/systemone":
                self._send(404, {"error": "not found"})
                return
            auth = self.headers.get("Authorization", "")
            if auth != "Bearer " + api_key:
                self._send(401, {"error": "unauthorized"})
                return
            length = self.headers.get("Content-Length")
            try:
                n = int(length or "0")
            except ValueError:
                self._send(400, {"error": "invalid Content-Length"})
                return
            if n < 1 or n > MAX_BODY:
                self._send(400, {"error": "request body is missing or too large"})
                return
            try:
                payload = json.loads(self.rfile.read(n).decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                self._send(400, {"error": "invalid JSON"})
                return
            try:
                self._send(200, handle_request(hub.encoder, payload))
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
    parser.add_argument("--api-key", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--gpu-layers", type=int, default=99)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--no-install", action="store_true")
    parser.add_argument("--install-35b", action="store_true")
    args = parser.parse_args()
    binary = args.binary or str(ROOT / "build" / "bin" / "openjev")
    hub = ModelHub(ROOT, binary=binary, gpu_layers=args.gpu_layers, threads=args.threads,
                   quantize_bin=str(ROOT / "build" / "bin" / "llama-quantize"))
    try:
        if not args.no_install:
            names = [name for name, spec in CATALOG.items() if spec["auto"]]
            if args.install_35b:
                names.append("openjev_35b")
            hub.install(names)
        server = ThreadingHTTPServer((args.host, args.port), make_handler(hub, args.api_key))
        print(f"openjev systemone on http://{args.host}:{args.port}/v1/systemone", flush=True)
        print("models: " + ", ".join(hub.ready_ids()), flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            server.server_close()
    finally:
        hub.close()


if __name__ == "__main__":
    main()
