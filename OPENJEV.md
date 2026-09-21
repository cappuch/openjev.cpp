# openjev.cpp

A local MIT-licensed fork of [llama.cpp](https://github.com/ggml-org/llama.cpp) for [AlexWortega/openjev](https://huggingface.co/AlexWortega/openjev), based on upstream commit `6f41ac59e0a49a00483a316a22ada6b04edd2950`.

## What changes

- Convert dense `Qwen3_5ForSequenceClassification` and MoE `Qwen3_5MoeForSequenceClassification` checkpoints to GGUF, including the trained `score.weight` head.
- Pool the last token with causal attention and return raw logits in `[contradiction, entailment, neutral]` order. The CLI applies stable softmax once, without embedding normalization or chat templates.
- Skip the vocabulary projection and its output buffer. Preserve the small classifier head in F32 during conversion and quantization.
- For three or more text hypotheses, tokenize complete pairs, find their common token prefix, prefill it once, and copy both full-attention KV and recurrent state into a separate working sequence. Suffixes run sequentially with bounded device memory, independent of the number of options. `--no-prefix-cache` provides an independent-pair baseline.
- Keep a loaded model alive across JSONL requests. The Python client exposes `predict`, `predict_hypotheses`, `rerank`, `grade`, `latents`, and `latents_hypotheses`.
- Convert the vision projector separately and evaluate image premises through upstream mtmd and MRoPE. Images use independent pair evaluation.

This is a classification executable; it does not generate text. Existing upstream tools and library names remain available. There is no llama-server `/classify` route. Task-specific latent MLP heads are not ported. A small Python process serves the SystemOne JSON API at `POST /v1/systemone`.

## Build

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -DLLAMA_BUILD_SERVER=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_TESTS=OFF
cmake --build build --target openjev llama-quantize -j
```

Metal is selected automatically on macOS. For NVIDIA builds add `-DGGML_CUDA=ON`; for a CPU-only build use `-DGGML_METAL=OFF` on macOS. Run with `-ngl 0` for CPU inference.

## Download and convert

Use Python 3.12 or newer and the upstream conversion dependencies:

```sh
uv venv --python 3.12
uv pip install --python .venv/bin/python torch transformers safetensors numpy sentencepiece protobuf mistral-common
.venv/bin/hf download AlexWortega/openjev \
  --include 'qwen3.5-4b-nli-v2/*' --local-dir models/openjev-hf
.venv/bin/python convert_hf_to_gguf.py models/openjev-hf/qwen3.5-4b-nli-v2 \
  --outtype f16 --outfile models/openjev-4b-f16.gguf
build/bin/llama-quantize models/openjev-4b-f16.gguf models/openjev-4b-Q4_K_M.gguf Q4_K_M
```

For the smaller checkpoint, replace the subfolder with `qwen3.5-0.8b-nli-v2s-long`. The same converter registrations cover the original 4B and 35B-A3B checkpoints; the 35B model needs substantially more memory. Validate quantized models on your own task: retaining the head in F32 does not eliminate backbone quantization error.

## Python

Run from this repository, or put it on `PYTHONPATH`. The client itself uses only the Python standard library and returns lists.

```python
from openjev import OpenJevCrossEncoder

with OpenJevCrossEncoder("models/openjev-4b-Q4_K_M.gguf") as jev:
    probabilities = jev.predict([
        ("A man is playing a guitar.", "Someone is making music.")
    ])
    index = jev.rerank("Which gas do plants absorb during photosynthesis?",
                       ["oxygen", "carbon dioxide", "nitrogen"])
    scores = jev.predict_hypotheses("The door is red.",
                                   ["The door is red.", "The door is blue.", "The door is open."])
    label = jev.grade("What is 2 + 2?", "4", "4")
```

`rerank` wraps each option with `The correct answer is: ` and chooses the greatest entailment probability, retaining the first option on a tie. `grade` uses the reference formatting from the original wrapper. Calls are serialized per Python instance. Use a context manager or call `close()` to release the child process.

## HTTP (SystemOne)

```sh
.venv/bin/python openjev_server.py --api-key secret --port 8080
```

Open `http://127.0.0.1:8080/` for settings. Launch does not download weights. Install a catalog id from that page (Bearer key required). Install runs in the background; the table polls `/v1/settings`. Prefer an existing `Q4_K_M` file, otherwise F16. Models load on the first `/v1/systemone` request that names them.

Kev checkpoints are LoRA adapters plus a pointer head on a Qwen base (`jaredpalmer/kev-*`). Install merges the adapter, converts the backbone, and writes a `*.head.bin` sidecar. The engine scores options with that pointer (state prefix + per-question branches), not 3-way NLI.

IDs: `jev-latest` (openjev 4B), `kev-latest` (kev 4B), `openjev_0.8b`, `openjev_4b`, `openjev_35b`, `kev_0.5b`, `kev_0.8b`, `kev_4b`, `kev_9b`. `GET /v1/models` lists them. `GET /health` reports which GGUFs are on disk and which processes are loaded.

```sh
curl -sS http://127.0.0.1:8080/v1/systemone \
  -H 'Authorization: Bearer secret' -H 'Content-Type: application/json' \
  -d '{"state":"The door is red.","model":"openjev_0.8b","questions":{"colour":{"type":"choice","instructions":"What colour is the door?","criteria":{"red":"red","blue":"blue"}}}}'
```

`POST /v1/systemone` accepts `state`, `model`, and `questions` keyed by caller IDs. Those keys are echoed in `answers` and are not sent to the model. Question types:

- `noul`: truth score in `[0, 1]`. On openjev, optional `criteria.true` / `criteria.false` are two hypotheses; otherwise the instructions are scored against the state. On kev, options are `no`/`yes` (with those criteria texts when present) and `noul` is p(yes).
- `choice`: up to 255 options. Openjev softmaxes entailment logits with the rerank wrapper `The correct answer is: `. Kev softmaxes pointer logits over `name` / `name: desc`. `choice` is the argmax, first option on a tie. Null criteria are skipped.
- `score`: 2-10 ordered levels. `score` is the expected level index; `legend` keeps the original level text.

`usage.input_tokens` is evaluated tokens; `usage.output_tokens` is 0. `GET /health` has no auth.


For latents, open an encoder with `latents=True` and call `latents(pairs)` or `latents_hypotheses(premise, hypotheses)`. These are the unnormalized final-token hidden states before the classifier head. Classification and latent output use separate contexts.

## JSONL protocol

```sh
echo '{"premise":"A man is playing a guitar.","hypotheses":["Someone is making music.","Nobody is making music.","The man is wearing blue."]}' | \
  build/bin/openjev -m models/openjev-4b-Q4_K_M.gguf
```

One JSON object is returned for every input line. Model diagnostics go to stderr. Each result has `logits`, `probabilities`, and `label`; the response includes `labels`, `prefix_tokens`, `evaluated_tokens`, and `elapsed_ms`. Errors are returned as `{"error":"..."}` and the process accepts the next request. Startup failures exit nonzero.

Other request forms:

```json
{"pairs":[["A bird flies.","An animal moves."]]}
{"question":"Which gas do plants absorb?","options":["oxygen","carbon dioxide","nitrogen"]}
{"question":"What is 2 + 2?","reference":"4","candidate":"4"}
```

Reranking adds `index`; grading adds `label`. With `--latents`, each result contains `latent` instead of classification scores. The default maximum length is 4096 tokens per pair. Overlong inputs are rejected, not truncated. Device context capacity is reserved for the prefix and one independent working sequence; host tokenization memory scales with request size. Split very large input lists into smaller calls.

## Images

```sh
.venv/bin/python convert_hf_to_gguf.py models/openjev-hf/qwen3.5-4b-nli-v2 \
  --mmproj --outtype f16 --outfile models/openjev-4b-mmproj-f16.gguf
echo '{"image":"scene.png","premise":"A photograph of a scene:","hypotheses":["There is a dog."]}' | \
  build/bin/openjev -m models/openjev-4b-Q4_K_M.gguf --mmproj models/openjev-4b-mmproj-f16.gguf
```

The image is inserted at the start of the premise by default. To control placement, put mtmd's media marker (`<__media__>`) in the premise. One image is supported per request and reused for all its pairs. Use `image="scene.png"` with the Python prediction methods. Do not manually insert repeated `<|image_pad|>` tokens; mtmd determines the image token count.

## Validation

```sh
.venv/bin/python tools/openjev/test_openjev.py \
  --model models/openjev-0.8b-f16.gguf \
  --hf-model models/openjev-hf/qwen3.5-0.8b-nli-v2s-long
```

This checks actual model predictions, final-token latents, cached versus independent inference across multiple prefill batches, duplicate/reordered hypotheses, invalid-input recovery, reranking, grading, and optional raw-logit parity with Transformers. It downloads nothing. The default parity tolerance is for F16; use an explicitly chosen tolerance for quantized backbones.

## License and provenance

The original [MIT license](LICENSE) and upstream history are retained. Openjev model weights have their own model-card MIT license. Changes are on the local `openjev` branch; no GitHub fork or remote publication is created by building this tree.
