"""Checkpoint loading and saving without environment-specific paths."""

from __future__ import annotations

import pickle
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from torch import nn

DEFAULT_CLUT_ARCHITECTURE = "20+05+20"


def clut_architecture(model: nn.Module) -> str:
    """Return the compact ``bases+spatial_rank+width_rank`` model descriptor."""
    clut = getattr(model, "CLUTs", None)
    if clut is None or not all(hasattr(clut, name) for name in ("num", "s", "w")):
        raise TypeError("The model does not expose FedPAIE CLUT architecture attributes.")
    return f"{int(clut.num):02d}+{int(clut.s):02d}+{int(clut.w):02d}"


def _load_checkpoint(
    checkpoint_path: str | Path,
    device: torch.device | str,
    *,
    allow_legacy_pickle: bool,
) -> Any:
    """Load tensor-only checkpoints by default; opt in for trusted legacy modules."""
    path = Path(checkpoint_path)
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except pickle.UnpicklingError as error:
        if not allow_legacy_pickle:
            raise RuntimeError(
                "This checkpoint is not in the portable tensor-only format. "
                "Convert it first, or explicitly allow legacy pickle loading only "
                "when the file comes from a trusted source."
            ) from error
        return torch.load(path, map_location=device, weights_only=False)


def extract_state_dict(checkpoint: Any) -> Mapping[str, torch.Tensor]:
    """Extract a state dictionary from common PyTorch checkpoint layouts."""
    if isinstance(checkpoint, nn.Module):
        return checkpoint.state_dict()
    if not isinstance(checkpoint, Mapping):
        raise TypeError(f"Unsupported checkpoint type: {type(checkpoint).__name__}")

    for key in ("model_state_dict", "state_dict", "model"):
        value = checkpoint.get(key)
        if isinstance(value, Mapping):
            return value

    if checkpoint and all(isinstance(value, torch.Tensor) for value in checkpoint.values()):
        return checkpoint
    raise KeyError("No model state dictionary was found in the checkpoint.")


def load_state_dict(
    model: nn.Module,
    checkpoint_path: str | Path,
    device: torch.device | str,
    *,
    strict: bool = True,
    allow_legacy_pickle: bool = False,
) -> nn.Module:
    """Load a checkpoint into an existing model."""
    checkpoint = _load_checkpoint(
        checkpoint_path,
        device,
        allow_legacy_pickle=allow_legacy_pickle,
    )
    model.load_state_dict(extract_state_dict(checkpoint), strict=strict)
    return model


def load_clut_model(
    checkpoint_path: str | Path,
    device: torch.device | str,
    *,
    architecture: str | None = None,
    allow_legacy_pickle: bool = False,
) -> nn.Module:
    """Load either a serialized CLUT model or a CLUT state dictionary."""
    from models.clut_net import CLUTNet

    checkpoint = _load_checkpoint(
        checkpoint_path,
        device,
        allow_legacy_pickle=allow_legacy_pickle,
    )
    if isinstance(checkpoint, nn.Module):
        model = checkpoint
    else:
        checkpoint_architecture = checkpoint.get("architecture")
        resolved_architecture = (
            architecture
            or (str(checkpoint_architecture) if checkpoint_architecture else None)
            or DEFAULT_CLUT_ARCHITECTURE
        )
        model = CLUTNet(resolved_architecture)
        model.load_state_dict(extract_state_dict(checkpoint))
    return model.to(device)


def save_checkpoint(
    model: nn.Module,
    path: str | Path,
    **metadata: Any,
) -> Path:
    """Save a portable state-dictionary checkpoint and optional metadata."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {"model_state_dict": model.state_dict(), **metadata}
    torch.save(payload, destination)
    return destination
