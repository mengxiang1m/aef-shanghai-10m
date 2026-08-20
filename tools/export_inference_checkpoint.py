from __future__ import annotations

import argparse
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Remove optimizer-only state from a trained checkpoint for inference"
    )
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    checkpoint = torch.load(args.input, map_location="cpu", weights_only=False)
    required = ("kind", "epoch", "model_state", "config", "stats")
    missing = [name for name in required if name not in checkpoint]
    if missing:
        raise KeyError(f"Checkpoint is missing required keys: {missing}")
    inference_checkpoint = {
        name: checkpoint[name]
        for name in ("kind", "epoch", "model_state", "config", "stats", "metrics")
        if name in checkpoint
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(inference_checkpoint, args.output)
    size_mib = args.output.stat().st_size / 1024**2
    print(f"wrote {args.output} ({size_mib:.2f} MiB)")


if __name__ == "__main__":
    main()
