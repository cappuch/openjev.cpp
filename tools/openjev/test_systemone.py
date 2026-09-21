"""Schema and routing checks for the SystemOne server. No checkpoint is loaded."""

import unittest
from unittest.mock import Mock

from openjev_server import RequestError, canonical_id, flatten, handle_request, softmax


class FlattenTests(unittest.TestCase):
    def test_nested_entry(self):
        text = flatten({"hp": 3, "inv": ["wood", "stone"], "note": "craft pickaxe"})
        self.assertIn("hp: 3", text)
        self.assertIn("wood", text)

    def test_rejects_bool(self):
        with self.assertRaises(RequestError):
            flatten(True)


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

    def test_softmax(self):
        weights = softmax([0.0, 0.0])
        self.assertAlmostEqual(weights[0], 0.5)


if __name__ == "__main__":
    unittest.main()
