#!/usr/bin/env python3
"""Convert Laya's English checkpoint to a ModernBERT GGUF and a decision-head GGUF."""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "gguf-py"))

import gguf
from safetensors import safe_open

from conversion.bert import ModernBertModel


class LayaEncoder(ModernBertModel):
    model_arch = gguf.MODEL_ARCH.MODERN_BERT

    @classmethod
    def filter_tensors(cls, item):
        name, gen = item
        if not name.startswith("encoder."):
            return None
        return super().filter_tensors((name[len("encoder."):], gen))

    def set_vocab(self):
        source = self.dir_model
        try:
            self.dir_model = source / "tokenizer"
            super().set_vocab()
        finally:
            self.dir_model = source

    def set_gguf_parameters(self):
        super().set_gguf_parameters()
        self.gguf_writer.add_string("laya.head_file", self.head_name)


def convert(source, out_dir, encoder_name="laya-f16.gguf", head_name="laya.head.gguf"):
    source, out_dir = Path(source), Path(out_dir)
    cfg = json.loads((source / "rl_agent_config.json").read_text())
    hparams = json.loads((source / "encoder" / "config.json").read_text())
    if (cfg.get("encoder") != "answerdotai/ModernBERT-large" or
            hparams.get("model_type") != "modernbert" or cfg.get("head_layers") != 2):
        raise ValueError("only the English ModernBERT-large Laya checkpoint is supported")
    if not 1 <= cfg["head_max_len"] < cfg["max_len"] <= hparams["max_position_embeddings"]:
        raise ValueError("invalid Laya token budgets")
    temperatures = cfg.get("temperature", [1.0] * 3)
    if len(temperatures) != 3 or any(not math.isfinite(t) or t <= 0 for t in
                                   [*temperatures, *cfg.get("temperature_by_options", {}).values()]):
        raise ValueError("temperatures must be finite and positive")
    if len(cfg["act_costs"]) != 1:
        raise ValueError("expected act/escalate head")
    for name in (encoder_name, head_name):
        if Path(name).name != name:
            raise ValueError("output names must be filenames")
    if encoder_name == head_name:
        raise ValueError("encoder and head filenames must differ")
    out_dir.mkdir(parents=True, exist_ok=True)
    encoder_path, head_path = out_dir / encoder_name, out_dir / head_name
    if encoder_path.exists() or head_path.exists():
        raise FileExistsError("output exists; use a new output directory or filenames")
    encoder_tmp, head_tmp = encoder_path.with_suffix(".gguf.tmp"), head_path.with_suffix(".gguf.tmp")
    model = LayaEncoder(source, gguf.LlamaFileType.MOSTLY_F16, encoder_tmp, hparams=hparams)
    model.head_name = head_name
    writer = gguf.GGUFWriter(head_tmp, "laya")
    writer.add_string("laya.config", json.dumps(cfg))
    writer.add_uint32("laya.embedding_length", hparams["hidden_size"])
    try:
        with safe_open(source / "model.safetensors", framework="pt", device="cpu") as tensors:
            for name in tensors.keys():
                if not name.startswith("encoder."):
                    writer.add_tensor(name, tensors.get_tensor(name).float().numpy())
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file()
        writer.close()
        model.write()
        head_tmp.replace(head_path)
        encoder_tmp.replace(encoder_path)
    finally:
        writer.close()
        encoder_tmp.unlink(missing_ok=True)
        head_tmp.unlink(missing_ok=True)
    return encoder_path, head_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hub", default="convaiinnovations/laya")
    parser.add_argument("--model-dir", type=Path, help="use a local checkpoint instead of downloading")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--f16-name", default="laya-f16.gguf")
    parser.add_argument("--head-name", default="laya.head.gguf")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    source = args.model_dir
    if source is None:
        from huggingface_hub import snapshot_download
        source = Path(snapshot_download(args.hub, allow_patterns=[
            "model.safetensors", "rl_agent_config.json", "encoder/config.json", "tokenizer/*",
        ]))
    for path in convert(source, args.out_dir, args.f16_name, args.head_name):
        print(path)


if __name__ == "__main__":
    main()
