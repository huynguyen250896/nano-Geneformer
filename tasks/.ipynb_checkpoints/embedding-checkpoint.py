from __future__ import annotations

import argparse
import numpy as np
import torch

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from model import GeneformerForMaskedLM
from Geneformer_tokenizer import GeneformerTokenizer


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


def main(argv=None):
    p = argparse.ArgumentParser()

    p.add_argument("--input", required=True, help="Input .h5ad or tokenized .pt file.")
    p.add_argument("--output", required=True, help="Path to save .npy embeddings.")
    p.add_argument("--mode", default="raw", choices=["raw", "tokenized"])
    p.add_argument("--save-tokenized", default=None)

    p.add_argument(
        "--model",
        default="Geneformer-V1",
        choices=[
            "Geneformer-V1",
            "Geneformer-V2-104M",
            "Geneformer-V2-104M_CLcancer",
            "Geneformer-V2-316M",
        ],
    )
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument(
        "--layer",
        type=int,
        default=-2,
        help="-1 = last layer, -2 = second-to-last layer.",
    )
    p.add_argument(
        "--pool",
        default="mean_pool",
        choices=["mean_pool", "cls", "all"],
    )
    p.add_argument(
        "--no_amp",
        action="store_true",
        help="Disable automatic mixed precision on CUDA.",
    )
    p.add_argument("--no_compile", action="store_true")
    p.add_argument(
        "--compile-mode",
        default="reduce-overhead",
        choices=[
            "default",
            "reduce-overhead",
            "max-autotune",
            "max-autotune-no-cudagraphs",
        ],
    )

    args = p.parse_args(argv)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = not args.no_amp
    compile_model = not args.no_compile

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

    print(f"Loading nano-Geneformer: {args.model}")

    model = GeneformerForMaskedLM.from_pretrained(
        args.model,
        compile_model=False,
    ).to(device)
    
    model.eval()
    
    if compile_model:
        model = model.optimize_for_inference(
            compile_model=True,
            compile_mode=args.compile_mode,
        )

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

    embs = model.encode(
        encoded["input_ids"],
        attention_mask=encoded["attention_mask"],
        batch_size=args.batch_size,
        layer=args.layer,
        cell_emb_style=args.pool,
        special_token=tokenizer.special_token,
        use_amp=use_amp,
    )

    embs = embs.numpy()

    np.save(args.output, embs)

    print(f"embeddings: {embs.shape}")
    print(f"saved to {args.output}")


if __name__ == "__main__":
    main()