from __future__ import annotations

import argparse
import json
from dataclasses import fields

import torch

from aef.config import load_config
from aef.losses import consistency_loss, reconstruction_loss
from aef.temporal_model import AEFTemporal, TemporalModelConfig


def main() -> None:
    parser = argparse.ArgumentParser(description="Full-shape AEFTemporal GPU memory smoke test")
    parser.add_argument("--config", default="configs/shanghai_stp_temporal.yaml")
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--patch-size", type=int, default=128)
    parser.add_argument("--consistency", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    allowed = {field.name for field in fields(TemporalModelConfig)}
    model_config = TemporalModelConfig(
        **{key: value for key, value in config["temporal_model"].items() if key in allowed}
    )
    target_specs = config["temporal_targets"]
    device = torch.device(args.device)
    model = AEFTemporal(
        {"s1": 3, "s2": 4},
        {name: int(spec["channels"]) for name, spec in target_specs.items()},
        model_config,
    ).to(device)
    batch, size = args.batch_size, args.patch_size
    sources = {
        "s1": torch.randn(batch, 12, 3, size, size, device=device),
        "s2": torch.randn(batch, 3, 4, size, size, device=device),
    }
    times = {
        "s1": torch.linspace(0, 335, 12, device=device).expand(batch, -1),
        "s2": torch.tensor([60, 213, 305], device=device).expand(batch, -1),
    }
    valid = {
        "s1": torch.ones(batch, 12, size, size, dtype=torch.bool, device=device),
        "s2": torch.ones(batch, 3, size, size, dtype=torch.bool, device=device),
    }
    valid["s1"][:, 5] = False
    valid["s2"][:, 1] = False
    period = torch.tensor([[0.0, 366.0]], device=device).expand(batch, -1)
    target_times = {
        "s1": torch.full((batch,), 152.0, device=device),
        "s2": torch.full((batch,), 213.0, device=device),
        "building": torch.full((batch,), 183.0, device=device),
        "lst": torch.full((batch,), 183.0, device=device),
        "dem": torch.full((batch,), 183.0, device=device),
        "ndvi": torch.full((batch,), 183.0, device=device),
    }
    targets = {
        name: torch.randn(batch, int(spec["channels"]), size, size, device=device)
        for name, spec in target_specs.items()
    }
    targets["building"] = (targets["building"] > 0).float()
    masks = {name: torch.ones_like(value, dtype=torch.bool) for name, value in targets.items()}
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    optimizer.zero_grad(set_to_none=True)
    torch.cuda.reset_peak_memory_stats(device)
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        output = model(sources, times, valid, period, target_times)
        loss, parts = reconstruction_loss(output["predictions"], targets, masks, target_specs)
        if args.consistency:
            student_valid = {name: value.clone() for name, value in valid.items()}
            student_sources = {name: value.clone() for name, value in sources.items()}
            student_valid["s1"][:, 2] = False
            student_valid["s2"][:, 0] = False
            student_sources["s1"][:, 2] = 0
            student_sources["s2"][:, 0] = 0
            student = model.encode(student_sources, times, student_valid, period)
            loss = loss + 0.05 * consistency_loss(output["embedding"], student)
    loss.backward()
    optimizer.step()
    result = {
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "batch_size": batch,
        "patch_size": size,
        "embedding_shape": list(output["embedding"].shape),
        "loss": float(loss.detach()),
        "consistency_branch": args.consistency,
        "parts": {name: float(value.detach()) for name, value in parts.items()},
        "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 2**30,
        "peak_reserved_gib": torch.cuda.max_memory_reserved(device) / 2**30,
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
