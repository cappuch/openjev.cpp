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
KEV_LATEST = "kev-latest"
MAX_BODY = 2 * 1024 * 1024
MAX_CHOICE = 255
MAX_ENTRY_DEPTH = 32

CATALOG = {
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
    if CATALOG[cid]["family"] == "kev":
        rec, meta = kev_payload(payload["state"], questions)
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
            "aliases": {LATEST_ID: "openjev_4b", KEV_LATEST: "kev_4b"},
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
        if spec.get("family") == "kev" and not (self.models_dir / spec["head"]).is_file():
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
                return self._loaded[cid]
            spec = CATALOG[cid]
            path = self.gguf_path(spec)
            if path is None:
                raise RequestError(
                    f"model {cid} is not installed; open / and install it from settings",
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


SETTINGS_PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>openjev settings</title>
<style>
body { font-family: ui-sans-serif, system-ui, sans-serif; margin: 24px; max-width: 920px; color: #111; }
h1 { font-size: 20px; margin: 0 0 8px; }
p, td, th, label { font-size: 14px; }
.muted { color: #555; }
table { border-collapse: collapse; width: 100%; margin-top: 16px; }
th, td { text-align: left; padding: 8px 10px; border-bottom: 1px solid #ddd; vertical-align: top; }
button { cursor: pointer; }
.row { display: flex; gap: 12px; align-items: center; flex-wrap: wrap; margin: 12px 0; }
input[type=password], input[type=text] { min-width: 220px; padding: 6px 8px; }
.ok { color: #0a7; }
.bad { color: #c00; }
.run { color: #a60; }
</style>
</head>
<body>
<h1>openjev settings</h1>
<p class="muted">Nothing downloads until you press Install. SystemOne stays at <code>POST /v1/systemone</code>.</p>
<div class="row">
<label>API key <input id="key" type="password" autocomplete="off"></label>
<span id="note" class="muted"></span>
</div>
<table>
<thead><tr><th>id</th><th>family</th><th>disk</th><th>loaded</th><th></th></tr></thead>
<tbody id="rows"></tbody>
</table>
<script>
const keyEl = document.getElementById('key');
keyEl.value = localStorage.getItem('openjev-api-key') || '';
keyEl.addEventListener('change', () => localStorage.setItem('openjev-api-key', keyEl.value));
function cls(state) {
  if (state === 'ready' || state === true) return 'ok';
  if (state === 'error' || state === false) return 'bad';
  if (state === 'running') return 'run';
  return '';
}
async function refresh() {
  const r = await fetch('/v1/settings');
  const data = await r.json();
  document.getElementById('note').textContent = 'ready: ' + (data.ready.join(', ') || 'none');
  const body = document.getElementById('rows');
  body.innerHTML = '';
  for (const m of data.models) {
    const tr = document.createElement('tr');
    const job = m.job ? m.job.state + (m.job.error ? ' ' + m.job.error : '') : '';
    const disk = m.ready ? (m.file || 'yes') : (job || 'missing');
    tr.innerHTML = '<td><code>' + m.id + '</code><div class="muted">' + m.source + '</div></td>' +
      '<td>' + m.family + '</td>' +
      '<td class="' + cls(m.ready ? 'ready' : (m.job && m.job.state)) + '">' + disk + '</td>' +
      '<td class="' + cls(m.loaded) + '">' + (m.loaded ? 'yes' : 'no') + '</td>';
    const td = document.createElement('td');
    const b = document.createElement('button');
    b.textContent = m.ready ? 'installed' : (m.job && m.job.state === 'running' ? 'installing...' : 'install');
    b.disabled = !!(m.ready || (m.job && m.job.state === 'running'));
    b.onclick = () => install(m.id);
    td.appendChild(b);
    tr.appendChild(td);
    body.appendChild(tr);
  }
}
async function install(id) {
  const key = keyEl.value;
  if (!key) { alert('set the API key first'); return; }
  localStorage.setItem('openjev-api-key', key);
  const r = await fetch('/v1/settings/install', {
    method: 'POST',
    headers: { 'Authorization': 'Bearer ' + key, 'Content-Type': 'application/json' },
    body: JSON.stringify({ model: id }),
  });
  const data = await r.json();
  if (!r.ok) { alert(data.error || r.statusText); return; }
  refresh();
}
refresh();
setInterval(refresh, 2000);
</script>
</body>
</html>
"""


def make_handler(hub, api_key):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, format, *args):
            sys.stderr.write("%s - %s\n" % (self.address_string(), format % args))

        def _send(self, status, payload, content_type="application/json; charset=utf-8"):
            if isinstance(payload, str):
                body = payload.encode("utf-8")
            else:
                body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Connection", "close")
            self.end_headers()
            self.wfile.write(body)

        def _auth_ok(self):
            return self.headers.get("Authorization", "") == "Bearer " + api_key

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
                self._send(200, SETTINGS_PAGE, "text/html; charset=utf-8")
                return
            if path == "/health":
                self._send(200, {"ok": True, "models": hub.ready_ids(), "loaded": sorted(hub._loaded)})
                return
            if path == "/v1/settings":
                self._send(200, hub.status())
                return
            if path == "/v1/models":
                self._send(200, {"data": [{"id": name, "owned_by": "openjev"} for name in known_model_ids()]})
                return
            self._send(404, {"error": "not found"})

        def do_POST(self):
            path = urlparse(self.path).path
            if path == "/v1/settings/install":
                if not self._auth_ok():
                    self._send(401, {"error": "unauthorized"})
                    return
                try:
                    payload = self._read_json()
                    if not isinstance(payload, dict):
                        raise RequestError("request must be an object")
                    self._send(200, hub.start_install(payload.get("model")))
                except RequestError as exc:
                    self._send(exc.status, {"error": str(exc)})
                except Exception as exc:
                    self._send(500, {"error": str(exc)})
                return
            if path != "/v1/systemone":
                self._send(404, {"error": "not found"})
                return
            if not self._auth_ok():
                self._send(401, {"error": "unauthorized"})
                return
            try:
                self._send(200, handle_request(hub.encoder, self._read_json()))
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
    args = parser.parse_args()
    binary = args.binary or str(ROOT / "build" / "bin" / "openjev")
    hub = ModelHub(ROOT, binary=binary, gpu_layers=args.gpu_layers, threads=args.threads,
                   quantize_bin=str(ROOT / "build" / "bin" / "llama-quantize"))
    try:
        server = ThreadingHTTPServer((args.host, args.port), make_handler(hub, args.api_key))
        print(f"openjev settings http://{args.host}:{args.port}/", flush=True)
        print(f"openjev systemone http://{args.host}:{args.port}/v1/systemone", flush=True)
        print("models: " + ", ".join(hub.ready_ids() or ["none"]), flush=True)
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
