#!/usr/bin/env python3
"""Merge a Kev LoRA + pointer head and convert to GGUF + KEVHEAD1 sidecar."""

from __future__ import annotations

import argparse
import struct
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def write_head(path, state, temperature):
    qw = state["q.weight"].detach().float().cpu().contiguous()
    kw = state["k.weight"].detach().float().cpu().contiguous()
    qb = state["q.bias"].detach().float().cpu().contiguous()
    kb = state["k.bias"].detach().float().cpu().contiguous()
    dp, d = int(qw.shape[0]), int(qw.shape[1])
    if tuple(kw.shape) != (dp, d) or qb.numel() != dp or kb.numel() != dp:
        raise SystemExit("pointer head shapes do not match")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as out:
        out.write(b"KEVHEAD1")
        out.write(struct.pack("<IIf", d, dp, float(temperature)))
        out.write(qw.numpy().tobytes())
        out.write(qb.numpy().tobytes())
        out.write(kw.numpy().tobytes())
        out.write(kb.numpy().tobytes())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hub", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--f16-name", required=True)
    parser.add_argument("--q4-name", required=True)
    parser.add_argument("--head-name", required=True)
    parser.add_argument("--quantize")
    parser.add_argument("--hf-cache", type=Path)
    args = parser.parse_args()

    import torch
    from huggingface_hub import snapshot_download
    from peft import PeftModel  # ty: ignore[unresolved-import]
    from transformers import AutoModelForCausalLM, AutoTokenizer

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    cache = args.hf_cache or (out_dir / "kev-hf")
    adapter = Path(snapshot_download(args.hub, local_dir=str(cache / args.hub.replace("/", "--"))))
    try:
        meta = torch.load(adapter / "head.pt", map_location="cpu", weights_only=False)
    except TypeError:
        meta = torch.load(adapter / "head.pt", map_location="cpu")
    write_head(out_dir / args.head_name, meta["head"], meta.get("temperature", 1.0))
    f16 = out_dir / args.f16_name
    q4 = out_dir / args.q4_name
    if f16.is_file() or q4.is_file():
        print(f"openjev: kev GGUF already present, wrote {args.head_name}", flush=True)
        return

    base = meta["base"]
    revision = meta.get("base_revision")
    print(f"openjev: downloading base {base}", flush=True)
    kwargs = {"revision": revision} if revision else {}
    tok = AutoTokenizer.from_pretrained(base, **kwargs)
    model = AutoModelForCausalLM.from_pretrained(base, dtype=torch.float16, **kwargs)
    model = PeftModel.from_pretrained(model, str(adapter))
    merged = model.merge_and_unload()
    merged_dir = cache / (args.hub.replace("/", "--") + "-merged")
    if merged_dir.exists():
        import shutil
        shutil.rmtree(merged_dir)
    merged_dir.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(merged_dir)
    tok.save_pretrained(merged_dir)
    del merged, model
    convert = [
        sys.executable, str(ROOT / "convert_hf_to_gguf.py"), str(merged_dir),
        "--outtype", "f16", "--outfile", str(f16),
    ]
    print("+ " + " ".join(convert), flush=True)
    import subprocess
    subprocess.check_call(convert, cwd=str(ROOT))
    if args.quantize and Path(args.quantize).is_file() and f16.is_file() and not q4.is_file():
        quant = [args.quantize, str(f16), str(q4), "Q4_K_M"]
        print("+ " + " ".join(quant), flush=True)
        subprocess.check_call(quant, cwd=str(ROOT))


if __name__ == "__main__":
    main()
