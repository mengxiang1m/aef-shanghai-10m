from __future__ import annotations

import argparse
from pathlib import Path

import torch

from aef.config import load_config
from aef.model import AEFLite


def main() -> None:
    parser = argparse.ArgumentParser(description="Export an initialized AEF-Lite checkpoint")
    parser.add_argument("--config", default="configs/shanghai_10m.yaml")
    parser.add_argument("--output", default="artifacts/aef_lite_init.pt")
    args = parser.parse_args()
    torch.manual_seed(0)
    config = load_config(args.config)
    model = AEFLite(config)
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    clean_config = {key: value for key, value in config.items() if not key.startswith("_")}
    torch.save(
        {"kind": "initialized", "epoch": 0, "model_state": model.state_dict(), "config": clean_config, "stats": None},
        output,
    )
    parameters = sum(parameter.numel() for parameter in model.parameters())
    print(f"wrote {output} ({parameters:,} parameters; initialized, not trained)")


if __name__ == "__main__":
    main()

