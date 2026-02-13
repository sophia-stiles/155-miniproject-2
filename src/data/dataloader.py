from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import torch
from jax.dlpack import from_dlpack
from omegaconf import DictConfig

from data.base import VideoTextDataSample

# ==================== Preprocessing ====================

def preprocess_video_text_batch(
    batch: tuple[Any, Any],
    tube_masking_cfg: DictConfig | None = None,
    rng: jax.Array | None = None,
) -> tuple[jax.Array, list[str]]:
    """Convert batch to JAX arrays and optionally apply tube masking.

    Handles both torch.Tensor (zero-copy via DLPack) and np.ndarray inputs.
    """
    video_input, text_input = batch

    if isinstance(video_input, torch.Tensor):
        video_input = from_dlpack(video_input)
    elif isinstance(video_input, np.ndarray):
        video_input = jnp.asarray(video_input)
    return video_input, text_input

def collate_video_text(batch: list[VideoTextDataSample]) -> tuple[torch.Tensor, list[str]]:
    """PyTorch collate_fn: stack videos into a tensor and collect captions."""
    video_input = torch.stack([sample.video for sample in batch], dim=0).contiguous()
    text_input = [sample.caption for sample in batch]
    return video_input, text_input