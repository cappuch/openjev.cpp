"""Real-model regression checks. No checkpoint is downloaded by this script."""

import argparse
import math
from pathlib import Path
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from openjev import OpenJevCrossEncoder


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
    args = parser.parse_args()
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
