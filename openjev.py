"""Persistent Python client for the openjev.cpp executable (standard library only)."""

from __future__ import annotations

import json
import math
import subprocess
import threading
from pathlib import Path


def laya_payload(state, questions):
    """Render the option markers' text in the checkpoint's training format."""
    if not isinstance(questions, dict) or not questions:
        raise ValueError("questions must be a non-empty object")
    rows, meta = [], []
    for qid, question in questions.items():
        if not isinstance(question, dict) or "instructions" not in question:
            raise ValueError("each Laya question requires instructions")
        kind, criteria = question.get("type"), question.get("criteria")
        instr = question["instructions"]
        if not isinstance(instr, str):
            instr = json.dumps(instr)
        item = {"id": str(qid), "type": kind}
        if kind == "choice":
            if isinstance(criteria, list) and all(isinstance(c, str) for c in criteria):
                criteria = dict.fromkeys(criteria)
            if not isinstance(criteria, dict) or any(v is not None and not isinstance(v, str) for v in criteria.values()):
                raise ValueError("Laya choice criteria must be an object of strings/nulls or an array of strings")
            item["keys"] = list(criteria)
            options = [k if not v else f"{k}: {v}" for k, v in criteria.items()]
        elif kind == "score":
            if not isinstance(criteria, list) or not all(isinstance(c, str) for c in criteria):
                raise ValueError("Laya score criteria must be an array of strings")
            options = [f"level {i}: {c}" for i, c in enumerate(criteria)]
            item["legend"] = {str(i): c for i, c in enumerate(criteria)}
        elif kind == "noul":
            criteria = {} if criteria is None else criteria
            if not isinstance(criteria, dict) or any(v is not None and not isinstance(v, str) for v in criteria.values()):
                raise ValueError("Laya noul criteria must be an object of strings/nulls")
            options = ["false: " + (criteria.get("false") or "no, the statement does not hold"),
                       "true: " + (criteria.get("true") or "yes, the statement holds")]
        else:
            raise ValueError(f"unsupported Laya question type: {kind}")
        if not 2 <= len(options) <= 255:
            raise ValueError("Laya questions require 2 to 255 options")
        item["count"] = len(options)
        rows.append({"type": kind, "instr": instr, "options": options})
        meta.append(item)
    text = state if isinstance(state, str) else json.dumps(state, ensure_ascii=False)
    return {"state": text, "questions": rows}, meta


def laya_answers(response, meta):
    rows = response.get("results", [])
    if len(rows) != len(meta):
        raise ValueError("Laya returned the wrong number of questions")
    answers = {}
    for row, item in zip(rows, meta):
        p = row["probabilities"]
        if len(p) != item["count"] or any(not math.isfinite(v) or v < 0 for v in p):
            raise ValueError("invalid Laya probabilities")
        answer = {"type": item["type"], "rl_agent": {"act_probability": row["act_probability"]}}
        if item["type"] == "noul":
            answer["noul"] = round(p[1], 4)
        else:
            answer["confidence"] = round(1 + sum(v * math.log(max(v, 1e-12)) for v in p) / math.log(len(p)), 4)
            if item["type"] == "choice":
                keys = item["keys"]
                answer["choice"] = keys[max(range(len(p)), key=p.__getitem__)]
            else:
                keys = list(item["legend"])
                answer["legend"] = item["legend"]
                answer["score"] = round(sum(i * v for i, v in enumerate(p)), 4)
            answer["probabilities"] = {k: round(v, 4) for k, v in zip(keys, p)}
        answers[item["id"]] = answer
    return answers


class OpenJevCrossEncoder:
    """Load a GGUF once and score requests over JSONL. Use as a context manager."""

    def __init__(self, model, *, binary=None, mmproj=None, ctx_size=4096,
                 batch_size=512, threads=4, gpu_layers=99, prefix_cache=True, latents=False,
                 inference_timeout=110.0):
        if not math.isfinite(inference_timeout) or inference_timeout <= 0:
            raise ValueError("inference_timeout must be positive")
        self.inference_timeout = inference_timeout
        binary = binary or Path(__file__).resolve().parent / "build" / "bin" / "openjev"
        command = [str(binary), "-m", str(model), "-c", str(ctx_size), "-b", str(batch_size),
                   "-t", str(threads), "-ngl", str(gpu_layers)]
        if mmproj is not None:
            command += ["--mmproj", str(mmproj)]
        if not prefix_cache:
            command += ["--no-prefix-cache"]
        if latents:
            command += ["--latents"]
        self._latents = latents
        self._lock = threading.Lock()
        self._process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                         text=True, encoding="utf-8", bufsize=1)

    def request(self, payload):
        """Return logits, probabilities, labels and timing/token counters."""
        with self._lock:
            if self._process.poll() is not None:
                raise RuntimeError(f"openjev exited with status {self._process.returncode}; see stderr")
            expired = threading.Event()
            def expire():
                expired.set()
                self._process.kill()
            watchdog = threading.Timer(self.inference_timeout, expire)
            watchdog.daemon = True
            watchdog.start()
            try:
                try:
                    self._process.stdin.write(json.dumps(payload, ensure_ascii=False) + "\n")
                    self._process.stdin.flush()
                    line = self._process.stdout.readline()
                except BrokenPipeError as exc:
                    if not expired.is_set():
                        raise RuntimeError("openjev closed its input; see stderr") from exc
            finally:
                watchdog.cancel()
                watchdog.join()
            if expired.is_set():
                self._process.wait()
                raise TimeoutError(f"inference exceeded {self.inference_timeout:g}s; encoder stopped")
            if not line:
                raise RuntimeError("openjev exited without a response; see stderr")
            result = json.loads(line)
            if "error" in result:
                raise ValueError(result["error"])
            return result

    def is_alive(self):
        return self._process.poll() is None

    def _rows(self, payload, *, latent=False):
        if latent != self._latents:
            raise ValueError("create the encoder with latents=True for latent methods, False for classification")
        key = "latent" if latent else "probabilities"
        return [row[key] for row in self.request(payload)["results"]]

    def predict(self, pairs, *, image=None):
        payload = {"pairs": list(pairs)}
        if image is not None:
            payload["image"] = str(image)
        return self._rows(payload)

    def system_one(self, state, questions):
        """Answer typed questions with a Laya GGUF."""
        payload, meta = laya_payload(state, questions)
        response = self.request(payload)
        return {"model": "laya", "answers": laya_answers(response, meta),
                "usage": {"input_tokens": response["evaluated_tokens"], "output_tokens": 0}}

    def predict_hypotheses(self, premise, hypotheses, *, image=None):
        payload = {"premise": premise, "hypotheses": list(hypotheses)}
        if image is not None:
            payload["image"] = str(image)
        return self._rows(payload)

    def rerank(self, question, options):
        return self.request({"question": question, "options": list(options)})["index"]

    def grade(self, question, reference, candidate):
        return self.request({"question": question, "reference": reference, "candidate": candidate})["label"]

    def latents(self, pairs):
        return self._rows({"pairs": list(pairs)}, latent=True)

    def latents_hypotheses(self, premise, hypotheses):
        return self._rows({"premise": premise, "hypotheses": list(hypotheses)}, latent=True)

    def close(self):
        with self._lock:
            if self._process.stdin and not self._process.stdin.closed:
                try:
                    self._process.stdin.close()
                except BrokenPipeError:
                    pass
            try:
                self._process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self._process.terminate()
                try:
                    self._process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self._process.kill()
                    self._process.wait()
            self._process.stdout.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
