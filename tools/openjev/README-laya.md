# Laya

The English root checkpoint of `convaiinnovations/laya` is supported by the `openjev` executable, Python client and `/v1/systemone` API. It uses the existing ModernBERT encoder and a companion GGUF containing the two decision layers, option scorer, question-type embeddings and act/escalate head.

Reference: [model and inference code](https://huggingface.co/convaiinnovations/laya/tree/main).

## Build and convert

Use the Python environment with the repository's conversion dependencies (`torch`, `transformers`, `safetensors`, `huggingface_hub` and `gguf-py`).

```sh
cmake --build build --target openjev -j
python tools/openjev/convert_laya.py --out-dir models
```

Conversion downloads only the English checkpoint and produces `models/laya-f16.gguf` and `models/laya.head.gguf`. For an existing download, pass `--model-dir /path/to/laya`. Keep both GGUFs in the same directory. Conversion refuses to overwrite existing output files.

The server model catalog also offers `laya` (alias `laya-latest`); its install action runs this converter. The encoder uses the configured GPU layers. The decision head uses the GPU when GPU offload is enabled, with CPU fallback for unsupported operations. With `-ngl 0`, it uses the available CPU accelerator (such as BLAS) and CPU backends with `--threads`. Its weights stay resident and its scheduler reuses allocation buffers between requests. Questions are packed without padding into batches of up to eight questions and 1,024 total tokens (or `--ctx-size` if smaller). Each batch uses one encoder call and shares the head's projections and feed-forward operations. Attention stays isolated per question. Larger requests are split automatically, with output order preserved. The JSONL response reports `batches`, `encoder_ms` and `head_ms` for profiling.

## Python

```python
from openjev import OpenJevCrossEncoder

with OpenJevCrossEncoder("models/laya-f16.gguf") as model:
    result = model.system_one(
        {"body": "We were charged twice. Please refund the duplicate."},
        {
            "department": {
                "type": "choice",
                "instructions": "Which department should handle this?",
                "criteria": {"billing": "payments and refunds", "technical": "bugs and outages"},
            },
            "refund": {"type": "noul", "instructions": "Does the sender request a refund?"},
            "urgency": {
                "type": "score",
                "instructions": "How urgent is this?",
                "criteria": ["low", "medium", "high"],
            },
        },
    )
    print(result["answers"])
```

For HTTP, send the same `state` and `questions` with `"model": "laya"` to `/v1/systemone` using the server's existing authentication.

The low-level JSONL interface accepts a serialized state string and a questions array. Each question has `type`, `instr`, and `options`; options must already use the reference format. Prefer `system_one()` or HTTP for typed questions: these format choice descriptions, `level N:` score options, and `false:`/`true:` options automatically.

Answers use the reference temperatures for question type and option count, entropy confidence, expected score, and `rl_agent.act_probability`. Null choice descriptions retain the option, as in Laya's reference API. The 512-token limit, 192-token question budget and option truncation follow the checkpoint. At least two options are required; requests whose option markers do not fit are rejected. Images, multilingual checkpoints, typed-decisions checkpoints and automatic language routing are not supported.

## Verification

```sh
PYTHONPATH=. python -m unittest discover -s tools/openjev -p test_systemone.py
python tools/openjev/test_openjev.py --laya --model models/laya-f16.gguf \
    --hf-model /path/to/laya --gpu-layers 0
```

The reference directory must contain the published `rl_common.py` as well as the checkpoint. The test compares raw logits, calibrated probabilities, act probabilities and token counts, including long inputs and all choice temperature buckets. Repeat with `--gpu-layers 99` to check GPU encoder inference.
