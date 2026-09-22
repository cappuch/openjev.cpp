"""Real-model regression checks. No checkpoint is downloaded by this script."""

import argparse
import math
from pathlib import Path
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from openjev import OpenJevCrossEncoder


def test_laya(args):
    import json
    from openjev import laya_payload
    state = {"body": "The door is red. Please refund the duplicate charge today.", "note": "caf\u00e9 [MASK]"}
    questions = {
        "color": {"type": "choice", "instructions": "What color is the door?", "criteria": ["red", "blue"]},
        "department": {"type": "choice", "instructions": "Which department should handle this?",
                       "criteria": {"billing": "refunds", "support": "technical problems", "other": None}},
        "urgency": {"type": "score", "instructions": "How urgent is this?", "criteria": ["low", "medium", "high"]},
        "refund": {"type": "noul", "instructions": "Does the sender request a refund?"},
        "many": {"type": "choice", "instructions": {"task": "Choose the color [MASK]"},
                 "criteria": ["red", "blue", "green", "black", "white", "yellow", "orange"]},
        "bucket11": {"type": "choice", "instructions": "Which number is two?", "criteria": [str(i) for i in range(12)]},
    }
    long_question = {"q": {"type": "choice", "instructions": "word " * 250,
                           "criteria": {"red": "red " * 100, "blue": "blue " * 100, "other": "other " * 100,
                                        "four": "four " * 100, "five": "five " * 100}}}
    cases = [(state, questions), ("The door is red. " * 200, long_question)]
    observed = []
    with OpenJevCrossEncoder(args.model, binary=args.binary, gpu_layers=args.gpu_layers) as jev:
        for state, qs in cases:
            payload, _ = laya_payload(state, qs)
            raw = jev.request(payload)
            observed.append(raw)
            assert all(abs(sum(r["probabilities"]) - 1) < 1e-6 for r in raw["results"])
        payload, _ = laya_payload(*cases[0])
        again = jev.request(payload)
        compare([r["logits"] for r in again["results"]], [r["logits"] for r in observed[0]["results"]], 1e-5)
        reversed_payload = {**payload, "questions": list(reversed(payload["questions"]))}
        reversed_rows = jev.request(reversed_payload)["results"]
        compare([r["logits"] for r in reversed_rows], [r["logits"] for r in reversed(observed[0]["results"])], 1e-5)
        for bad in [{"state": "x", "questions": []},
                    {"state": "x", "questions": [{"type": "bad", "options": ["a", "b"], "instr": "x"}]},
                    {"state": "x", "questions": [{"type": "choice", "options": ["a"], "instr": "x"}]},
                    {"state": "x", "questions": [{"type": "choice", "options": ["a"] * 255, "instr": "x"}]}]:
            try:
                jev.request(bad)
            except ValueError:
                pass
            else:
                raise AssertionError("accepted invalid Laya request")
        answers = jev.system_one(*cases[0])["answers"]
        assert answers["color"]["choice"] == "red"
        assert answers["refund"]["noul"] > 0.5
        assert 0 <= answers["urgency"]["score"] <= 2
    print("PASS Laya typed answers, long-input truncation, repeated/reordered questions and error recovery")
    if not args.hf_model:
        return
    import torch
    from safetensors.torch import load_file
    from transformers import AutoTokenizer
    sys.path.insert(0, args.hf_model)
    from rl_common import build_model, build_sequence, collate_items, QTYPES, temp_bucket
    torch.set_num_threads(4)
    source = Path(args.hf_model)
    cfg = json.loads((source / "rl_agent_config.json").read_text())
    model = build_model(cfg, encoder_dir=str(source / "encoder"))
    model.load_state_dict(load_file(source / "model.safetensors"), strict=True)
    model.encoder.config.reference_compile = False
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(source / "tokenizer")
    max_logit, max_prob, max_act = 0.0, 0.0, 0.0
    for (state, qs), actual in zip(cases, observed):
        items = []
        for question in qs.values():
            criteria = question.get("criteria")
            if question["type"] == "choice" and isinstance(criteria, list):
                criteria = dict.fromkeys(criteria)
            instr = question["instructions"]
            q = {"t": question["type"], "ins": instr if isinstance(instr, str) else json.dumps(instr), "crit": criteria}
            ids, markers = build_sequence(tokenizer, state, q, cfg["max_len"], cfg["head_max_len"])
            items.append({"ids": ids, "markers": markers, "qtype": QTYPES[q["t"]], "target": [0.] * len(markers),
                          "label": -1, "episode": 0, "ep_step": 0, "ep_len": 1})
        b = collate_items([items], tokenizer.pad_token_id)
        assert actual["evaluated_tokens"] == b["n_tokens"], "tokenization differs from reference"
        with torch.inference_mode():
            logits, act = model(b["input_ids"], b["attention_mask"], b["marker_pos"], b["marker_mask"], b["qtype"])
        for i, (item, row) in enumerate(zip(items, actual["results"])):
            k = len(item["markers"])
            expected = logits[i, :k]
            max_logit = max(max_logit, max(abs(a - b) for a, b in zip(row["logits"], expected.tolist())))
            temp = cfg.get("temperature_by_options", {}).get(temp_bucket(item["qtype"], k), cfg["temperature"][item["qtype"]])
            p = torch.softmax(expected / temp, -1).tolist()
            max_prob = max(max_prob, max(abs(a - b) for a, b in zip(row["probabilities"], p)))
            max_act = max(max_act, abs(row["act_probability"] - torch.softmax(act[i], -1)[0].item()))
    print(f"Laya reference errors: logits={max_logit:.6f}, probabilities={max_prob:.6f}, act={max_act:.6f}")
    assert max_logit < args.tolerance
    assert max_prob < 0.01 and max_act < 0.01
    print("PASS Laya PyTorch parity (all temperature buckets and 512-token input)")


def compare(a, b, tolerance):
    assert len(a) == len(b)
    error = max(abs(x - y) for row_a, row_b in zip(a, b) for x, y in zip(row_a, row_b))
    assert error < tolerance, f"maximum error {error} exceeds {tolerance}"
    return error


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--binary")
    parser.add_argument("--hf-model")
    parser.add_argument("--hf-dtype", choices=["float32", "bfloat16", "float16"], default="float32")
    parser.add_argument("--mmproj")
    parser.add_argument("--gpu-layers", type=int, default=99)
    parser.add_argument("--tolerance", type=float, default=0.06)
    parser.add_argument("--laya", action="store_true", help="test Laya; --hf-model points to its reference checkpoint/code")
    args = parser.parse_args()
    if args.laya:
        test_laya(args)
        return
    kwargs = dict(binary=args.binary, gpu_layers=args.gpu_layers, batch_size=64)
    pairs = [("A man is playing a guitar.", "Someone is making music."),
             ("A man is playing a guitar.", "Nobody is making music."),
             ("A man is playing a guitar.", "The man is wearing a blue shirt.")]
    with OpenJevCrossEncoder(args.model, **kwargs) as jev:
        raw = jev.request({"pairs": pairs})
        baseline = [r["logits"] for r in raw["results"]]
        probabilities = [r["probabilities"] for r in raw["results"]]
        assert [r["label"] for r in raw["results"]] == ["entailment", "contradiction", "neutral"]
        assert all(abs(sum(row) - 1) < 1e-6 for row in probabilities)
        premise = "The room has a red door and a blue window. " * 20
        hypotheses = ["The door is red.", "The door is green.", "There is a dog.", "There is a dog."]
        independent = jev.request({"pairs": [[premise, h] for h in hypotheses]})
        cached = jev.request({"premise": premise, "hypotheses": hypotheses})
        assert cached["prefix_tokens"] > 64
        assert cached["evaluated_tokens"] < independent["evaluated_tokens"]
        cache_error = compare([r["logits"] for r in independent["results"]],
                              [r["logits"] for r in cached["results"]], 0.03)
        reverse = jev.predict_hypotheses(premise, list(reversed(hypotheses)))
        compare(reverse, [r["probabilities"] for r in reversed(cached["results"])], 0.01)
        identical = jev.predict_hypotheses(pairs[0][0], [pairs[0][1]] * 3)
        compare(identical, [probabilities[0]] * 3, 0.01)
        for n in [1, 2]:
            short = jev.request({"premise": pairs[0][0], "hypotheses": [p[1] for p in pairs[:n]]})
            assert short["prefix_tokens"] == 0
        for bad in [{"pairs": []}, {"premise": "x", "hypotheses": []}, {"pairs": [["x"]]},
                    {"pairs": [["word " * 5000, "x"]]}, {"pairs": [["x", "y"]], "options": []}]:
            try:
                jev.request(bad)
            except ValueError:
                pass
            else:
                raise AssertionError(f"accepted invalid request: {str(bad)[:100]}")
        compare(jev.predict(pairs), probabilities, 0.001)
        assert jev.rerank("Which gas do plants absorb during photosynthesis?", ["oxygen", "carbon dioxide", "nitrogen"]) == 1
        assert jev.grade("What is 2 + 2?", "4", "4") == "entailment"
        print(f"PASS classification, reranking, grading, recovery; prefix max logit error {cache_error:.6f}")
        print(f"Prefix evaluated tokens: {independent['evaluated_tokens']} -> {cached['evaluated_tokens']}")
        print(f"Prefix elapsed ms: {independent['elapsed_ms']:.1f} -> {cached['elapsed_ms']:.1f}")
    with OpenJevCrossEncoder(args.model, latents=True, **kwargs) as jev:
        latent = jev.latents(pairs)
        shared_latent = jev.latents_hypotheses(pairs[0][0], [p[1] for p in pairs])
        compare(latent, shared_latent, 0.03)
        assert len(latent[0]) > 3 and all(math.isfinite(x) for row in latent for x in row)
        print(f"PASS latents ({len(latent[0])} dimensions)")
    image_logits = None
    if args.mmproj:
        from PIL import Image
        with tempfile.TemporaryDirectory(prefix="openjev-image-") as directory:
            image_path = Path(directory) / "red.png"
            Image.new("RGB", (128, 128), (255, 0, 0)).save(image_path)
            with OpenJevCrossEncoder(args.model, mmproj=args.mmproj, **kwargs) as jev:
                response = jev.request({"image": str(image_path), "premise": "A picture:",
                                        "hypotheses": ["The image is red."]})
                image_logits = response["results"][0]["logits"]
                assert all(math.isfinite(x) for x in image_logits)
                assert response["prefix_tokens"] == 0
            print("PASS image inference")
    if args.hf_model:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        torch.set_num_threads(4)
        tokenizer = AutoTokenizer.from_pretrained(args.hf_model)
        model = AutoModelForSequenceClassification.from_pretrained(args.hf_model, dtype=getattr(torch, args.hf_dtype)).eval()
        texts = [model.config.nli_template.format(premise=p, hypothesis=h) for p, h in pairs]
        with torch.inference_mode():
            inputs = tokenizer(texts, padding=True, return_tensors="pt")
            expected = model(**inputs).logits.tolist()
        error = compare(baseline, expected, args.tolerance)
        assert [max(range(3), key=row.__getitem__) for row in baseline] == [max(range(3), key=row.__getitem__) for row in expected]
        print(f"PASS Transformers parity: max logit error {error:.6f}")
        if image_logits is not None:
            from transformers import AutoProcessor
            from PIL import Image
            processor = AutoProcessor.from_pretrained(args.hf_model)
            premise = "<|vision_start|><|image_pad|><|vision_end|>\nA picture:"
            text = model.config.nli_template.format(premise=premise, hypothesis="The image is red.")
            inputs = processor(text=[text], images=[Image.new("RGB", (128, 128), (255, 0, 0))], return_tensors="pt")
            with torch.inference_mode():
                expected = model(**inputs).logits.tolist()
            error = compare([image_logits], expected, args.tolerance)
            print(f"PASS Transformers image parity: max logit error {error:.6f}")


if __name__ == "__main__":
    main()
