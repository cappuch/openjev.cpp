"""Persistent Python client for the openjev.cpp executable (standard library only)."""

from __future__ import annotations

import json
import math
import subprocess
import threading
from pathlib import Path


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
