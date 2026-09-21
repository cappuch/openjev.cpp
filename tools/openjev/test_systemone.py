"""Schema and routing checks for the SystemOne server. No checkpoint is loaded."""

import unittest
from unittest.mock import Mock

from openjev_server import RequestError, canonical_id, flatten, handle_request, softmax


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
