"""Schema and routing checks for the SystemOne server. No checkpoint is loaded."""

import unittest
from unittest.mock import Mock

from openjev_server import RequestError, canonical_id, flatten, handle_request, softmax


def _http_connection(server):
    import http.client
    host, port = server.server_address[:2]
    return http.client.HTTPConnection(str(host), int(port), timeout=3)


class FlattenTests(unittest.TestCase):
    def test_nested_entry(self):
        text = flatten({"hp": 3, "inv": ["wood", "stone"], "note": "craft pickaxe"})
        self.assertIn("hp: 3", text)
        self.assertIn("wood", text)

    def test_bool_and_null(self):
        self.assertEqual(flatten(True), "true")
        self.assertEqual(flatten(False), "false")
        self.assertEqual(flatten({"flag": False, "skip": None}), "flag: false")


class HandleTests(unittest.TestCase):
    def setUp(self):
        self.jev = Mock()

        def fake_request(payload):
            n = 1
            if "options" in payload:
                n = len(payload["options"])
            elif "hypotheses" in payload:
                n = len(payload["hypotheses"])
            elif "pairs" in payload:
                n = len(payload["pairs"])
            logits = [[0.0, float(i), 0.0] for i in range(n)]
            if n >= 2:
                logits[1] = [0.0, 5.0, 0.0]
            return {"results": [{"logits": row} for row in logits], "evaluated_tokens": 10 * n}

        self.jev.request.side_effect = fake_request

        def get_encoder(name):
            if name in {"jev-latest", "openjev_4b", "openjev_0.8b"}:
                return self.jev
            raise RequestError(f"unknown model: {name!r}", 404)

        self.get_encoder = get_encoder

    def test_choice_keeps_ids_and_skips_null(self):
        out = handle_request(self.get_encoder, {
            "state": "The door is red.",
            "model": "jev-latest",
            "questions": {
                "foo": {
                    "type": "choice",
                    "instructions": "What colour is the door?",
                    "criteria": {"red": "red", "blue": "blue", "skip": None},
                }
            },
        })
        self.assertEqual(set(out["answers"]), {"foo"})
        self.assertEqual(out["answers"]["foo"]["type"], "choice")
        self.assertEqual(out["answers"]["foo"]["choice"], "blue")
        self.assertAlmostEqual(sum(out["answers"]["foo"]["probabilities"].values()), 1.0, places=6)
        self.assertNotIn("skip", out["answers"]["foo"]["probabilities"])
        self.assertEqual(out["usage"]["output_tokens"], 0)
        question = self.jev.request.call_args.args[0]["question"]
        self.assertNotIn("foo", question)

    def test_score_range(self):
        out = handle_request(self.get_encoder, {
            "state": "ok",
            "model": "jev-latest",
            "questions": {
                "bar": {
                    "type": "score",
                    "instructions": "quality",
                    "criteria": ["bad", "ok", "good"],
                }
            },
        })
        answer = out["answers"]["bar"]
        self.assertEqual(answer["type"], "score")
        self.assertGreaterEqual(answer["score"], 0)
        self.assertLessEqual(answer["score"], 2)
        self.assertEqual(set(answer["legend"]), {"0", "1", "2"})

    def test_unknown_model(self):
        with self.assertRaises(RequestError) as ctx:
            handle_request(self.get_encoder, {
                "state": "x", "model": "other", "questions": {"a": {"type": "noul", "instructions": "y"}},
            })
        self.assertEqual(ctx.exception.status, 404)

    def test_aliases(self):
        self.assertEqual(canonical_id("jev-latest"), "openjev_4b")
        self.assertEqual(canonical_id("openjev_0.8b"), "openjev_0.8b")
        self.assertEqual(canonical_id("kev-latest"), "kev_4b")
        self.assertEqual(canonical_id("kev-0.8b"), "kev_0.8b")
        self.assertEqual(canonical_id("laya-latest"), "laya")

    def test_laya_routing_preserves_reference_format(self):
        import json
        encoder = Mock()
        encoder.request.return_value = {"results": [
            {"probabilities": [0.2, 0.8], "act_probability": 0.9},
            {"probabilities": [0.1, 0.2, 0.7], "act_probability": 0.8},
            {"probabilities": [0.3, 0.7], "act_probability": 0.6},
        ], "evaluated_tokens": 42}
        state = {"body": "caf\u00e9", "ok": True}
        result = handle_request(lambda _: encoder, {
            "model": "laya", "state": state, "questions": {
                "c": {"type": "choice", "instructions": "pick", "criteria": {"z": None, "a": "alpha"}},
                "s": {"type": "score", "instructions": "rate", "criteria": ["low", "mid", "high"]},
                "n": {"type": "noul", "instructions": "true?"},
            },
        })
        payload = encoder.request.call_args.args[0]
        self.assertEqual(payload["state"], json.dumps(state, ensure_ascii=False))
        self.assertEqual(payload["questions"][0]["options"], ["z", "a: alpha"])
        self.assertEqual(payload["questions"][1]["options"], ["level 0: low", "level 1: mid", "level 2: high"])
        self.assertEqual(payload["questions"][2]["options"][0], "false: no, the statement does not hold")
        self.assertEqual(result["answers"]["c"]["choice"], "a")
        self.assertEqual(result["answers"]["s"]["score"], 1.6)
        self.assertEqual(result["answers"]["n"]["noul"], 0.7)
        self.assertEqual(result["answers"]["c"]["rl_agent"]["act_probability"], 0.9)
        self.assertEqual(result["usage"]["input_tokens"], 42)

    def test_laya_requires_companion(self):
        import tempfile
        from pathlib import Path
        from openjev_server import CATALOG, ModelHub
        with tempfile.TemporaryDirectory() as root:
            hub = ModelHub(root, binary="openjev", gpu_layers=0, threads=1, quantize_bin=None)
            hub.models_dir.mkdir()
            (hub.models_dir / "laya-f16.gguf").touch()
            self.assertIsNone(hub.gguf_path(CATALOG["laya"]))
            (hub.models_dir / "laya.head.gguf").touch()
            self.assertEqual(hub.gguf_path(CATALOG["laya"]), Path(root) / "models/laya-f16.gguf")

    def test_laya_invalid_schema(self):
        from openjev import laya_payload
        for question in [
            {"type": "choice", "instructions": "x", "criteria": ["only"]},
            {"type": "score", "instructions": "x", "criteria": "low"},
            {"type": "noul", "instructions": "x", "criteria": []},
            {"type": "unknown", "instructions": "x"},
        ]:
            with self.assertRaises(ValueError):
                laya_payload("state", {"q": question})

    def test_kev_choice_uses_pointer_options(self):
        def get_encoder(name):
            canonical_id(name)
            jev = Mock()

            def fake_request(payload):
                self.assertIn("questions", payload)
                self.assertEqual(payload["questions"][0]["options"][0], "red: red")
                return {
                    "results": [{"probabilities": [0.2, 0.8]}],
                    "evaluated_tokens": 9,
                }

            jev.request.side_effect = fake_request
            return jev

        out = handle_request(get_encoder, {
            "state": "The door is red.",
            "model": "kev_4b",
            "questions": {
                "foo": {
                    "type": "choice",
                    "instructions": "What colour is the door?",
                    "criteria": {"red": "red", "blue": "blue"},
                }
            },
        })
        self.assertEqual(out["answers"]["foo"]["choice"], "blue")
        self.assertAlmostEqual(out["answers"]["foo"]["probabilities"]["blue"], 0.8)
        self.assertEqual(out["usage"]["input_tokens"], 9)

    def test_kev_noul_is_yes_mass(self):
        def get_encoder(name):
            canonical_id(name)
            jev = Mock()
            jev.request.return_value = {
                "results": [{"probabilities": [0.25, 0.75]}],
                "evaluated_tokens": 4,
            }
            return jev

        out = handle_request(get_encoder, {
            "state": "ok",
            "model": "kev-latest",
            "questions": {"q": {"type": "noul", "instructions": "Is it ok?"}},
        })
        self.assertAlmostEqual(out["answers"]["q"]["noul"], 0.75)

    def test_softmax(self):
        weights = softmax([0.0, 0.0])
        self.assertAlmostEqual(weights[0], 0.5)

    def test_image_in_state_is_forwarded(self):
        import base64
        from pathlib import Path

        seen = {}

        def fake_request(payload):
            seen["image"] = payload.get("image")
            n = len(payload.get("options") or payload.get("hypotheses") or payload.get("pairs") or [0])
            return {"results": [{"logits": [0.0, 1.0, 0.0]} for _ in range(n)], "evaluated_tokens": 3}

        self.jev.request.side_effect = fake_request
        out = handle_request(self.get_encoder, {
            "state": {
                "goal": "click Today",
                "screenshot": {
                    "type": "image",
                    "media_type": "image/png",
                    "data": base64.b64encode(b"\x89PNG\r\n\x1a\n").decode(),
                },
            },
            "model": "openjev_0.8b",
            "questions": {
                "colour": {
                    "type": "choice",
                    "instructions": "What should we click?",
                    "criteria": {"today": "Today", "none": "nothing"},
                }
            },
        })
        self.assertEqual(out["answers"]["colour"]["choice"], "today")
        self.assertTrue(seen["image"])
        self.assertTrue(Path(seen["image"]).name.startswith("input"))

    def test_jpeg_base64_is_not_treated_as_a_path(self):
        import base64
        from pathlib import Path

        seen = {}
        jpeg = b"\xff\xd8\xff" + b"\x00" * 8000
        payload_b64 = base64.b64encode(jpeg).decode()
        self.assertGreater(len(payload_b64), 1024)
        self.assertTrue(payload_b64.startswith("/9j/"))

        def fake_request(payload):
            seen["image"] = payload.get("image")
            path = Path(payload["image"])
            self.assertTrue(path.is_file())
            self.assertLess(len(str(path)), 1024)
            return {"results": [{"logits": [0.0, 1.0, 0.0]}], "evaluated_tokens": 3}

        self.jev.request.side_effect = fake_request
        handle_request(self.get_encoder, {
            "state": {
                "screenshot": {
                    "type": "image",
                    "media_type": "image/jpeg",
                    "data": payload_b64,
                },
            },
            "model": "openjev_0.8b",
            "questions": {"q": {"type": "noul", "instructions": "Is there a window?"}},
        })
        self.assertTrue(seen["image"].endswith(".jpg"))

    def test_catalog_has_no_auto_install_flag(self):
        from openjev_server import CATALOG
        for spec in CATALOG.values():
            self.assertNotIn("auto", spec)

    def test_image_batch_matches_separate_questions(self):
        import base64
        import hashlib
        from openjev_server import evaluate_question, question_pairs

        def score(payload):
            pairs = question_pairs(payload)
            rows = [{"logits": [0.2, hashlib.sha256(repr(pair).encode()).digest()[0] / 40, -0.1]}
                    for pair in pairs]
            return {"results": rows, "evaluated_tokens": len(rows) * 10}

        self.jev.request.side_effect = score
        questions = {
            "done": {"type": "noul", "instructions": "Is it done?"},
            "risk": {"type": "noul", "instructions": "Is it sensitive?",
                     "criteria": {"true": "yes", "false": "no"}},
            "target": {"type": "choice", "instructions": "Which target?",
                       "criteria": {"a": "search", "b": "tab", "skip": None}},
            "score": {"type": "score", "instructions": "Quality?", "criteria": ["bad", "good"]},
        }
        expected = {qid: evaluate_question(self.jev, "screen", question)[0]
                    for qid, question in questions.items()}
        self.jev.reset_mock()
        out = handle_request(self.get_encoder, {
            "model": "openjev_0.8b", "state": "screen", "questions": questions,
            "image": {"type": "image", "media_type": "image/png",
                      "data": base64.b64encode(b"\x89PNG\r\n\x1a\n").decode()},
        })
        self.jev.request.assert_called_once()
        self.assertEqual(out["answers"], expected)
        self.assertEqual(out["usage"]["input_tokens"], 70)


class InferenceLifecycleTests(unittest.TestCase):
    def test_async_job_returns_before_inference_finishes(self):
        import json
        import threading
        import time
        from http.server import ThreadingHTTPServer
        from unittest.mock import patch
        from openjev_server import make_handler
        entered, release = threading.Event(), threading.Event()
        store, hub = Mock(), Mock()
        store.find_key.return_value = {"id": "test"}
        store.needs_setup.return_value = False
        hub.ready_ids.return_value = []
        hub._loaded = {}
        def infer(*args):
            entered.set(); release.wait(3)
            return {"answers": {}, "usage": {}}
        server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(hub, store))
        runner = threading.Thread(target=server.serve_forever); runner.start()
        def request(path, body="{}"):
            conn = _http_connection(server)
            conn.request("POST", path, body=body, headers={"Authorization": "Bearer key", "Content-Type": "application/json"})
            result = conn.getresponse(); data = json.loads(result.read()); conn.close()
            return result.status, data
        try:
            with patch("openjev_server.handle_request", side_effect=infer):
                status, job = request("/v1/systemone?async=1")
                self.assertEqual(status, 202)
                self.assertIn(job["status"], {"queued", "running"})
                self.assertTrue(entered.wait(2))
                release.set()
                for _ in range(30):
                    conn = _http_connection(server)
                    conn.request("GET", "/v1/jobs/" + job["id"])
                    result = conn.getresponse(); data = json.loads(result.read()); conn.close()
                    if data["status"] == "done": break
                    time.sleep(.02)
                self.assertEqual(data["status"], "done")
        finally:
            release.set(); server.shutdown(); server.server_close(); runner.join()
    def test_watchdog_stops_hung_process(self):
        import subprocess
        import sys
        import threading
        from openjev import OpenJevCrossEncoder
        encoder = OpenJevCrossEncoder.__new__(OpenJevCrossEncoder)
        encoder.inference_timeout = 0.1
        encoder._lock = threading.Lock()
        encoder._process = subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep(30)"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True,
        )
        try:
            with self.assertRaises(TimeoutError):
                encoder.request({"pairs": [["a", "b"]]})
            self.assertFalse(encoder.is_alive())
        finally:
            encoder.close()

    def test_busy_requests_fail_fast_and_health_stays_responsive(self):
        import json
        import threading
        from http.server import ThreadingHTTPServer
        from unittest.mock import patch
        from openjev_server import make_handler
        entered, release = threading.Event(), threading.Event()
        store, hub = Mock(), Mock()
        store.find_key.return_value = {"id": "test"}
        hub.ready_ids.return_value = []
        hub._loaded = {}
        store.needs_setup.return_value = False
        def infer(*args):
            entered.set()
            if not release.wait(5):
                raise RuntimeError("test did not release inference")
            return {"answers": {}, "usage": {}}
        server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(hub, store))
        runner = threading.Thread(target=server.serve_forever)
        runner.start()
        def request(path, method="POST"):
            conn = _http_connection(server)
            try:
                conn.request(method, path, body="{}" if method == "POST" else None)
                res = conn.getresponse()
                return res.status, json.loads(res.read())
            finally:
                conn.close()
        first_result = []
        first = threading.Thread(target=lambda: first_result.append(request("/v1/systemone")))
        try:
            with patch("openjev_server.handle_request", side_effect=infer):
                first.start()
                self.assertTrue(entered.wait(2))
                self.assertEqual(request("/v1/systemone")[0], 429)
                self.assertEqual(request("/health", "GET")[0], 200)
                release.set()
                first.join(3)
                self.assertEqual(first_result[0][0], 200)
                self.assertEqual(request("/v1/systemone")[0], 200)
        finally:
            release.set()
            first.join(3)
            server.shutdown()
            server.server_close()
            runner.join()


class AdminStoreTests(unittest.TestCase):
    def setUp(self):
        import tempfile
        from openjev_server import AdminStore
        self.dir = tempfile.TemporaryDirectory()
        self.store = AdminStore(self.dir.name + "/admin.json")

    def tearDown(self):
        self.dir.cleanup()

    def test_setup_login_and_keys(self):
        token = self.store.setup("boss", "correct-horse")
        self.assertEqual(self.store.session_user(token), "boss")
        with self.assertRaises(Exception):
            self.store.setup("boss", "correct-horse")
        again = self.store.login("boss", "correct-horse")
        self.assertEqual(self.store.session_user(again), "boss")
        created = self.store.create_key("prod")
        self.assertTrue(created["key"].startswith("oj_"))
        found = self.store.find_key("Bearer " + created["key"])
        self.assertEqual(found["id"], created["id"])
        self.store.record(created["id"], 10, 0, 25.0)
        snap = self.store.public_keys()
        self.assertEqual(snap["totals"]["requests"], 1)
        self.assertEqual(snap["keys"][0]["input_tokens"], 10)
        self.assertGreater(snap["throughput"]["eval_tok_per_s"], 0)

    def test_bad_password(self):
        from openjev_server import RequestError
        self.store.setup("boss", "correct-horse")
        with self.assertRaises(RequestError):
            self.store.login("boss", "wrong-wrong")


if __name__ == "__main__":
    unittest.main()
