from __future__ import annotations

import argparse
import time
from pathlib import Path
import sys

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from model import GeneformerForMaskedLM
from Geneformer_tokenizer import GeneformerTokenizer


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def save_tokenized(batch, path):
    torch.save(
        {
            "input_ids": batch["input_ids"].cpu(),
            "attention_mask": batch["attention_mask"].cpu(),
            "length": batch["length"].cpu(),
        },
        path,
    )


def load_tokenized(path):
    batch = torch.load(path, map_location="cpu")
    return {
        "input_ids": batch["input_ids"].long(),
        "attention_mask": batch["attention_mask"].long(),
        "length": batch["length"].long(),
    }


def resolve_return_layer(model, layer):
    num_layers = model.config.num_hidden_layers

    if layer < 0:
        return num_layers + 1 + layer

    return layer


def pool_hidden(hidden, input_ids, attention_mask, pool="mean_pool", special_token=False):
    if pool == "cls":
        return hidden[:, 0]

    if pool == "all":
        return hidden

    if pool != "mean_pool":
        raise ValueError(pool)

    mask = attention_mask.clone()

    if special_token:
        mask[:, 0] = 0

        lengths = attention_mask.sum(dim=1)
        batch_idx = torch.arange(input_ids.size(0), device=input_ids.device)
        eos_pos = lengths - 1
        valid = lengths > 1
        mask[batch_idx[valid], eos_pos[valid]] = 0

    mask = mask.unsqueeze(-1).to(hidden.dtype)
    denom = mask.sum(dim=1).clamp(min=1.0)

    return (hidden * mask).sum(dim=1) / denom


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--input", required=True, help="Input .h5ad or tokenized .pt file.")
    parser.add_argument("--output", required=True, help="Path to save .npy embeddings.")
    parser.add_argument("--mode", default="tokenized", choices=["raw", "tokenized"])
    parser.add_argument("--save-tokenized", default=None)

    parser.add_argument(
        "--model",
        default="Geneformer-V1",
        choices=[
            "Geneformer-V1",
            "Geneformer-V2-104M",
            "Geneformer-V2-104M_CLcancer",
            "Geneformer-V2-316M",
        ],
    )
    parser.add_argument("--batch_size", "--batch-size", type=int, default=64)
    parser.add_argument("--layer", type=int, default=-2)
    parser.add_argument("--pool", default="mean_pool", choices=["mean_pool", "cls", "all"])

    parser.add_argument("--no_amp", action="store_true")
    parser.add_argument("--no_compile", action="store_true")
    parser.add_argument(
        "--compile-mode",
        default="reduce-overhead",
        choices=[
            "default",
            "reduce-overhead",
            "max-autotune",
            "max-autotune-no-cudagraphs",
        ],
    )

    args = parser.parse_args()

    use_amp = not args.no_amp
    compile_model = not args.no_compile

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.backends.mps.is_available():
        device = torch.device("mps")

    tokenizer = GeneformerTokenizer.from_pretrained(args.model)

    if args.mode == "raw":
        print("Tokenizing raw h5ad...")
        encoded = tokenizer.encode_h5ad(args.input)

        if args.save_tokenized is not None:
            save_tokenized(encoded, args.save_tokenized)
            print(f"saved tokenized input to {args.save_tokenized}")

    else:
        print("Loading tokenized input...")
        encoded = load_tokenized(args.input)

    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]

    dataset = TensorDataset(input_ids, attention_mask)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    print(f"Loading nano-Geneformer: {args.model}")

    model = GeneformerForMaskedLM.from_pretrained(
        args.model,
        compile_model=False,
    ).to(device)

    model.eval()

    if compile_model and device.type == "cuda":
        model = model.optimize_for_inference(
            compile_model=True,
            compile_mode=args.compile_mode,
        )

    return_layer = resolve_return_layer(model, args.layer)

    print(
        f"device={device}, "
        f"use_amp={use_amp}, "
        f"compile_model={compile_model}, "
        f"compile_mode={args.compile_mode}, "
        f"mode={args.mode}, "
        f"batch_size={args.batch_size}, "
        f"pool={args.pool}, "
        f"layer={args.layer}"
    )

    print("Warmup...")
    with torch.inference_mode(), torch.amp.autocast(
        device_type=device.type,
        enabled=(use_amp and device.type == "cuda"),
    ):
        for i, (batch_ids, batch_mask) in enumerate(loader):
            batch_ids = batch_ids.to(device, non_blocking=True)
            batch_mask = batch_mask.to(device, non_blocking=True)

            outputs = model(
                input_ids=batch_ids,
                attention_mask=batch_mask,
                output_hidden_states=False,
                return_layer=return_layer,
            )

            hidden = outputs.selected_hidden_state
            if hidden is None:
                hidden = outputs.last_hidden_state

            _ = pool_hidden(
                hidden,
                batch_ids,
                batch_mask,
                pool=args.pool,
                special_token=tokenizer.special_token,
            )

            if i >= 10:
                break

    embeddings = []
    load_times = []
    model_times = []

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    sync(device)
    t_load = time.perf_counter()
    t_total = time.perf_counter()

    with torch.inference_mode(), torch.amp.autocast(
        device_type=device.type,
        enabled=(use_amp and device.type == "cuda"),
    ):
        for batch_ids, batch_mask in loader:
            load_times.append(time.perf_counter() - t_load)

            batch_ids = batch_ids.to(device, non_blocking=True)
            batch_mask = batch_mask.to(device, non_blocking=True)

            sync(device)
            t_model = time.perf_counter()

            outputs = model(
                input_ids=batch_ids,
                attention_mask=batch_mask,
                output_hidden_states=False,
                return_layer=return_layer,
            )

            hidden = outputs.selected_hidden_state
            if hidden is None:
                hidden = outputs.last_hidden_state

            emb = pool_hidden(
                hidden,
                batch_ids,
                batch_mask,
                pool=args.pool,
                special_token=tokenizer.special_token,
            )

            sync(device)
            model_times.append(time.perf_counter() - t_model)

            embeddings.append(emb.cpu())
            t_load = time.perf_counter()

    emb = torch.cat(embeddings, dim=0)
    total_time = time.perf_counter() - t_total

    print(f"Average data loading time per batch: {np.mean(load_times):.4f} seconds")
    print(f"Average model inference time per batch: {np.mean(model_times):.4f} seconds")
    print(
        f"Total time for {emb.shape[0]} cells: "
        f"{total_time:.2f} sec, "
        f"{total_time / emb.shape[0] * 1000:.3f} ms/cell"
    )
    print(f"Throughput: {emb.shape[0] / total_time:.2f} cells/s")

    if device.type == "cuda":
        peak_mem = torch.cuda.max_memory_allocated() / 1024**3
        print(f"Peak GPU memory: {peak_mem:.3f} GB")

    np.save(args.output, emb.numpy())
    print(f"Saved {tuple(emb.shape)} to {args.output}")


if __name__ == "__main__":
    main()