"""Importable utilities for video-text embedding inference.

Provides main functions:
    - get_video_embeddings(video_folder, annotation_folder, npz_filepath)
    - get_text_embeddings(annotation_folder, npz_filepath)
    - plot_similarity_matrix(video_data, text_data)
    - plot_tsne(video_data, text_data, bg_npz_vid_filepath, bg_npz_text_filepath, bg_alpha, bg_s, perplexity)
    - build_tsne_baseline(...) / overlay_tsne_points(...)

Quick start::

    from src.utils.code import (
        load_model,
        get_video_embeddings,
        get_text_embeddings,
        plot_similarity_matrix,
        plot_tsne,
    )

    # 1. Load model (with optional fine-tuned checkpoint)
    load_model(checkpoint_path="params.msgpack")

    # 2. Compute embeddings — returns (embeddings, labels) tuples
    vid_data = get_video_embeddings("videos/", "annotations/", "vid.npz")
    txt_data = get_text_embeddings("annotations/", "txt.npz")

    # 3. Visualize — accepts the tuples directly
    plot_similarity_matrix(vid_data, txt_data)
    plot_tsne(vid_data, txt_data)
    plot_tsne(vid_data, txt_data, bg_npz_vid_filepath="bg_vid.npz",
              bg_npz_text_filepath="bg_txt.npz")
"""

from __future__ import annotations

import difflib
import json
from collections import Counter
from typing import Any
import colorsys
import hashlib
import pickle
import random
import re
import sys
import textwrap
from pathlib import Path

import jax
import jax.numpy as jnp
from jax.dlpack import from_dlpack
import numpy as np

# ---------------------------------------------------------------------------
# Ensure the ``src/`` directory is importable so that internal modules
# (``data.*``, ``model.*``) resolve correctly.
# ---------------------------------------------------------------------------
_SRC_DIR = Path(__file__).resolve().parents[1]  # …/src
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

_MODEL_CONFIG = _SRC_DIR / "config" / "gemma3_270M.yaml"

# ---------------------------------------------------------------------------
# Lazy model singleton
# ---------------------------------------------------------------------------
_model = None
_params = None


def _patch_tf_webp() -> None:
    """Patch TensorFlow for ``decode_webp`` if missing (needed by VideoPrism)."""
    try:
        import tensorflow.python.ops.image_ops_impl as _img_ops

        if not hasattr(_img_ops, "decode_webp"):
            try:
                import tensorflow_io as tfio

                _img_ops.decode_webp = tfio.image.decode_webp
            except ImportError:
                pass
    except Exception:
        pass


def load_model(checkpoint_path: str | Path | None = None) -> None:
    """Load the alignment model and optionally restore a fine-tuned checkpoint.

    This function is idempotent — calling it again after the model is already
    loaded is a no-op.  If you need to switch checkpoints, restart the Python
    process.

    Args:
        checkpoint_path: Path to a ``params.msgpack`` checkpoint file
            (e.g. downloaded from Google Drive with :func:`download_checkpoint`).
            If *None*, the pretrained model weights are used as-is.
    """
    global _model, _params

    if _model is not None:
        return  # already loaded

    _patch_tf_webp()

    from hydra.utils import instantiate
    from omegaconf import OmegaConf

    print(f"JAX devices: {jax.devices()}")

    cfg = OmegaConf.load(str(_MODEL_CONFIG))
    print(f"Loading model from config: {_MODEL_CONFIG}")

    _model = instantiate(cfg.model, _convert_="all")
    _params = _model.params

    if checkpoint_path is not None:
        from flax import serialization

        ckpt_path = Path(checkpoint_path).expanduser().resolve()
        size_mb = ckpt_path.stat().st_size / 1e6
        print(f"Loading checkpoint: {ckpt_path} ({size_mb:.1f} MB)")

        with open(ckpt_path, "rb") as f:
            raw = f.read()

        # Try standard flax deserialization first; fall back to a streaming
        # msgpack unpacker with raised size limits for large (>1 GB) files.
        try:
            loaded = serialization.from_bytes(_params, raw)
        except (ValueError, Exception) as exc:
            print(f"Standard from_bytes failed ({exc}), retrying with raised msgpack limits …")
            import io
            import msgpack
            from flax.serialization import _msgpack_ext_unpack, from_state_dict

            buf_len = len(raw) + 1024
            unpacker = msgpack.Unpacker(
                io.BytesIO(raw),
                ext_hook=_msgpack_ext_unpack,
                raw=False,
                max_buffer_size=buf_len,
                max_bin_len=buf_len,
                max_str_len=buf_len,
                max_array_len=buf_len,
                max_map_len=buf_len,
                max_ext_len=buf_len,
            )
            state_dict = next(unpacker)
            loaded = from_state_dict(_params, state_dict)
            del state_dict

        # Free the raw bytes and intermediate copies to reclaim ~2–4 GB
        del raw
        _params = jax.tree_util.tree_map(jnp.asarray, loaded)
        del loaded
        import gc as _gc
        _gc.collect()
    else:
        print("Using pretrained weights (no checkpoint)")

    # Free the pretrained adapter weights on the model object — they are
    # superseded by _params and no longer needed for inference.
    _model.adapter_params = {}

    # Store params in bfloat16 permanently to halve memory.  Inference
    # quality is unaffected; results are cast back to float32 after the
    # forward pass.
    _params = to_bfloat16(_params)

    import gc as _gc
    _gc.collect()

    print(f"Model class: {type(_model).__name__}")
    print(f"Top-level param keys: {list(_params.keys())}")


def _get_model_and_params():
    """Return the loaded (model, params), loading defaults if necessary."""
    if _model is None:
        load_model()
    return _model, _params


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------

def to_bfloat16(x):
    """Convert a JAX array or pytree of arrays to bfloat16.

    Only floating-point leaves are cast; integer and other dtypes are left
    unchanged.  Works on single arrays, nested dicts/lists (pytrees), or any
    structure ``jax.tree_util`` can traverse.
    """
    def _cast(leaf):
        if hasattr(leaf, "dtype") and jnp.issubdtype(leaf.dtype, jnp.floating):
            return leaf.astype(jnp.bfloat16)
        return leaf
    return jax.tree_util.tree_map(_cast, x)


def l2_normalize(x: jax.Array, eps: float = 1e-8) -> jax.Array:
    """L2-normalize along the last axis."""
    return x / (jnp.linalg.norm(x, axis=-1, keepdims=True) + eps)


def _parse_timecode_to_seconds(value: str | float | int) -> float | None:
    """Parse a timestamp into seconds.

    Supports:
      - ``HH:MM:SS(.mmm)``, e.g. ``0:05:46.400``
      - ``MM:SS(.mmm)``
      - numeric values already in seconds
    """
    if isinstance(value, (int, float)):
        return float(value)
    if not isinstance(value, str):
        return None

    text = value.strip()
    if not text:
        return None

    # Fast path: already numeric text in seconds.
    try:
        return float(text)
    except ValueError:
        pass

    parts = text.split(":")
    try:
        if len(parts) == 3:
            hours = int(parts[0])
            minutes = int(parts[1])
            seconds = float(parts[2])
            return hours * 3600.0 + minutes * 60.0 + seconds
        if len(parts) == 2:
            minutes = int(parts[0])
            seconds = float(parts[1])
            return minutes * 60.0 + seconds
    except ValueError:
        return None
    return None


def _extract_annotation_intervals(
    data: dict,
    segment_key: str = "video_descriptions",
) -> list[tuple[float, float]]:
    """Extract sorted ``(start_s, end_s)`` intervals from annotation JSON."""
    raw_segments = data.get(segment_key)
    if not isinstance(raw_segments, list):
        return []

    intervals: list[tuple[float, float]] = []
    for seg in raw_segments:
        if not isinstance(seg, dict):
            continue
        start_s = _parse_timecode_to_seconds(seg.get("t0"))
        end_s = _parse_timecode_to_seconds(seg.get("t1"))
        if start_s is None or end_s is None:
            continue
        if end_s <= start_s:
            continue
        intervals.append((float(start_s), float(end_s)))

    intervals.sort(key=lambda x: (x[0], x[1]))
    return intervals


def _extract_annotation_segment_texts(
    data: dict,
    segment_key: str = "video_descriptions",
    segment_text_key: str = "text",
) -> list[tuple[float, float, str]]:
    """Extract sorted ``(start_s, end_s, text)`` items from annotation JSON."""
    raw_segments = data.get(segment_key)
    if not isinstance(raw_segments, list):
        return []

    items: list[tuple[float, float, str]] = []
    for seg in raw_segments:
        if not isinstance(seg, dict):
            continue
        start_s = _parse_timecode_to_seconds(seg.get("t0"))
        end_s = _parse_timecode_to_seconds(seg.get("t1"))
        text = seg.get(segment_text_key)
        if start_s is None or end_s is None:
            continue
        if end_s <= start_s:
            continue
        if not isinstance(text, str):
            continue
        text = text.strip()
        if not text:
            continue
        items.append((float(start_s), float(end_s), text))

    items.sort(key=lambda x: (x[0], x[1]))
    return items


def _format_seconds_hhmmss(seconds: float) -> str:
    """Format float seconds as ``HH:MM:SS.mmm``."""
    if seconds < 0:
        seconds = 0.0
    hours = int(seconds // 3600)
    rem = seconds - hours * 3600
    minutes = int(rem // 60)
    secs = rem - minutes * 60
    return f"{hours:02d}:{minutes:02d}:{secs:06.3f}"


def _temporal_iou(
    start_a: float,
    end_a: float,
    start_b: float,
    end_b: float,
) -> float:
    """Intersection-over-union of 1D intervals."""
    inter = max(0.0, min(end_a, end_b) - max(start_a, start_b))
    if inter <= 0.0:
        return 0.0
    union = max(end_a, end_b) - min(start_a, start_b)
    if union <= 0.0:
        return 0.0
    return inter / union


def _compact_text_diff(
    before_text: str,
    after_text: str,
    max_lines: int = 10,
) -> list[str]:
    """Build a compact unified diff for two text snippets."""
    before_lines = textwrap.wrap(before_text, width=100) or [""]
    after_lines = textwrap.wrap(after_text, width=100) or [""]
    diff_lines = list(
        difflib.unified_diff(
            before_lines,
            after_lines,
            fromfile="before",
            tofile="after",
            lineterm="",
            n=1,
        )
    )
    if len(diff_lines) > max_lines:
        hidden = len(diff_lines) - max_lines
        return diff_lines[:max_lines] + [f"... ({hidden} more diff lines hidden)"]
    return diff_lines


def diff_video_descriptions(
    before_annotations_file: str | Path,
    after_annotations_file: str | Path,
    *,
    segment_key: str = "video_descriptions",
    text_key: str = "text",
    time_precision_decimals: int = 3,
    boundary_iou_threshold: float = 0.30,
    max_boundary_examples: int = 8,
    max_text_diff_examples: int = 10,
    max_text_diff_lines: int = 10,
) -> dict[str, Any]:
    """Compare ``video_descriptions`` between two annotation files.

    The report focuses on:
      - segment count before vs after
      - how time intervals changed (unchanged, added, removed, likely boundary edits)
      - content diffs for segments whose time interval stayed the same

    Args:
        before_annotations_file: Path to the original annotation JSON.
        after_annotations_file: Path to the updated annotation JSON.
        segment_key: Key that stores list-like time segments.
        text_key: Content key inside each segment (typically ``"text"``).
        time_precision_decimals: Rounding precision used to define "same" interval.
        boundary_iou_threshold: IoU threshold for likely boundary edit pairing.
        max_boundary_examples: Max number of boundary-change examples printed.
        max_text_diff_examples: Max number of same-interval content diffs printed.
        max_text_diff_lines: Max unified-diff lines per content change.

    Returns:
        Dictionary with summary counts and detailed examples.
    """
    before_path = Path(before_annotations_file).expanduser().resolve()
    after_path = Path(after_annotations_file).expanduser().resolve()

    with open(before_path) as f:
        before_data = json.load(f)
    with open(after_path) as f:
        after_data = json.load(f)

    raw_before = before_data.get(segment_key)
    raw_after = after_data.get(segment_key)
    if not isinstance(raw_before, list):
        raise ValueError(f"'{segment_key}' is not a list in {before_path}")
    if not isinstance(raw_after, list):
        raise ValueError(f"'{segment_key}' is not a list in {after_path}")

    def _collect_segments(raw_segments: list[Any]) -> list[dict[str, Any]]:
        segments: list[dict[str, Any]] = []
        for idx, seg in enumerate(raw_segments):
            if not isinstance(seg, dict):
                continue
            start_s = _parse_timecode_to_seconds(seg.get("t0"))
            end_s = _parse_timecode_to_seconds(seg.get("t1"))
            if start_s is None or end_s is None or end_s <= start_s:
                continue
            text_value = seg.get(text_key)
            text = text_value if isinstance(text_value, str) else ""
            key = (
                round(float(start_s), time_precision_decimals),
                round(float(end_s), time_precision_decimals),
            )
            segments.append(
                {
                    "idx": idx,
                    "start_s": float(start_s),
                    "end_s": float(end_s),
                    "time_key": key,
                    "text": text.strip(),
                }
            )
        segments.sort(key=lambda x: (x["start_s"], x["end_s"], x["idx"]))
        return segments

    before_segments = _collect_segments(raw_before)
    after_segments = _collect_segments(raw_after)

    before_by_key: dict[tuple[float, float], list[dict[str, Any]]] = {}
    after_by_key: dict[tuple[float, float], list[dict[str, Any]]] = {}
    for seg in before_segments:
        before_by_key.setdefault(seg["time_key"], []).append(seg)
    for seg in after_segments:
        after_by_key.setdefault(seg["time_key"], []).append(seg)

    same_interval_pairs: list[tuple[dict[str, Any], dict[str, Any]]] = []
    before_unmatched: list[dict[str, Any]] = []
    after_unmatched: list[dict[str, Any]] = []
    all_time_keys = sorted(set(before_by_key) | set(after_by_key))
    for key in all_time_keys:
        before_list = before_by_key.get(key, [])
        after_list = after_by_key.get(key, [])
        shared = min(len(before_list), len(after_list))
        for i in range(shared):
            same_interval_pairs.append((before_list[i], after_list[i]))
        before_unmatched.extend(before_list[shared:])
        after_unmatched.extend(after_list[shared:])

    # Pair likely boundary edits among unmatched segments using best IoU.
    boundary_pairs: list[tuple[dict[str, Any], dict[str, Any], float]] = []
    used_after_idx: set[int] = set()
    for bseg in before_unmatched:
        best_idx = None
        best_iou = 0.0
        for a_idx, aseg in enumerate(after_unmatched):
            if a_idx in used_after_idx:
                continue
            iou = _temporal_iou(
                bseg["start_s"],
                bseg["end_s"],
                aseg["start_s"],
                aseg["end_s"],
            )
            if iou > best_iou:
                best_iou = iou
                best_idx = a_idx
        if best_idx is not None and best_iou >= boundary_iou_threshold:
            used_after_idx.add(best_idx)
            boundary_pairs.append((bseg, after_unmatched[best_idx], best_iou))

    boundary_before_ids = {id(b) for b, _a, _iou in boundary_pairs}
    boundary_after_ids = {id(a) for _b, a, _iou in boundary_pairs}
    removed_segments = [s for s in before_unmatched if id(s) not in boundary_before_ids]
    added_segments = [s for s in after_unmatched if id(s) not in boundary_after_ids]

    changed_content_on_same_times: list[dict[str, Any]] = []
    for bseg, aseg in same_interval_pairs:
        if bseg["text"] != aseg["text"]:
            changed_content_on_same_times.append(
                {
                    "start_s": bseg["start_s"],
                    "end_s": bseg["end_s"],
                    "before_text": bseg["text"],
                    "after_text": aseg["text"],
                    "diff_lines": _compact_text_diff(
                        bseg["text"],
                        aseg["text"],
                        max_lines=max_text_diff_lines,
                    ),
                }
            )

    before_total_duration = sum(s["end_s"] - s["start_s"] for s in before_segments)
    after_total_duration = sum(s["end_s"] - s["start_s"] for s in after_segments)
    before_counts = Counter(seg["time_key"] for seg in before_segments)
    after_counts = Counter(seg["time_key"] for seg in after_segments)
    same_time_count = sum((before_counts & after_counts).values())

    report = {
        "before_file": str(before_path),
        "after_file": str(after_path),
        "segment_key": segment_key,
        "num_segments_before": len(before_segments),
        "num_segments_after": len(after_segments),
        "same_time_count": same_time_count,
        "removed_time_count": len(removed_segments),
        "added_time_count": len(added_segments),
        "likely_boundary_edit_count": len(boundary_pairs),
        "changed_text_same_time_count": len(changed_content_on_same_times),
        "before_total_duration_s": before_total_duration,
        "after_total_duration_s": after_total_duration,
        "boundary_examples": [
            {
                "before_t0": b["start_s"],
                "before_t1": b["end_s"],
                "after_t0": a["start_s"],
                "after_t1": a["end_s"],
                "delta_start_s": a["start_s"] - b["start_s"],
                "delta_end_s": a["end_s"] - b["end_s"],
                "iou": iou,
            }
            for b, a, iou in boundary_pairs[:max_boundary_examples]
        ],
        "content_diff_examples": changed_content_on_same_times[:max_text_diff_examples],
    }

    print("")
    print("=== video_descriptions diff report ===")
    print(f"Before file: {before_path}")
    print(f"After file : {after_path}")
    print(f"Segments   : before={len(before_segments)} | after={len(after_segments)}")
    print(
        "Durations  : "
        f"before={before_total_duration:.3f}s | after={after_total_duration:.3f}s"
    )
    print(
        "Time match : "
        f"same={same_time_count}, "
        f"boundary_edits={len(boundary_pairs)}, "
        f"removed={len(removed_segments)}, "
        f"added={len(added_segments)}"
    )
    print(f"Text edits : same-time segments with changed text={len(changed_content_on_same_times)}")

    if boundary_pairs:
        print("")
        print("Likely boundary edits (old -> new):")
        for bseg, aseg, iou in boundary_pairs[:max_boundary_examples]:
            print(
                "  "
                f"{_format_seconds_hhmmss(bseg['start_s'])}-{_format_seconds_hhmmss(bseg['end_s'])}"
                " -> "
                f"{_format_seconds_hhmmss(aseg['start_s'])}-{_format_seconds_hhmmss(aseg['end_s'])}"
                f" | d_start={aseg['start_s'] - bseg['start_s']:+.3f}s"
                f", d_end={aseg['end_s'] - bseg['end_s']:+.3f}s"
                f", IoU={iou:.2f}"
            )
        if len(boundary_pairs) > max_boundary_examples:
            print(f"  ... {len(boundary_pairs) - max_boundary_examples} more not shown")

    if changed_content_on_same_times:
        print("")
        print("Text diffs for unchanged time segments:")
        for item in changed_content_on_same_times[:max_text_diff_examples]:
            print(
                "  "
                f"[{_format_seconds_hhmmss(item['start_s'])}-{_format_seconds_hhmmss(item['end_s'])}]"
            )
            for line in item["diff_lines"]:
                print(f"    {line}")
        if len(changed_content_on_same_times) > max_text_diff_examples:
            print(
                "  "
                f"... {len(changed_content_on_same_times) - max_text_diff_examples} more"
                " text diffs not shown"
            )

    return report


# ---------------------------------------------------------------------------
# Embedding functions
# ---------------------------------------------------------------------------

def get_video_embeddings(
    video_folder: str | Path,
    annotation_folder: str | Path | None = None,
    npz_filepath: str | Path | None = None,
    *,
    segment_key: str = "video_descriptions",
    min_interval_segments: int = 2,
    segment_progress_every: int = 100,
    model=None,
    params=None,
    num_frames: int = 16,
    resolution: tuple[int, int] = (288, 288),
    seed: int = 42,
    batch_size: int = 1,
) -> tuple[jax.Array, list[str]]:
    """Compute L2-normalized video embeddings for every ``.mp4`` in a folder.

    Files are discovered as ``video_folder/*.mp4`` and sorted by filename stem
    so that the ordering is consistent with :func:`get_text_embeddings` when
    both folders contain files with the same names.

    If a matching annotation file ``<stem>.json`` exists, the function extracts
    time segments from ``segment_key`` and decodes interval clips by calling
    :func:`data.utils.decode` with ``frame_sampling_method="interval"``.
    Overlapping segments are supported and processed independently.

    If no annotation exists, segment extraction fails, or the number of valid
    segments is below ``min_interval_segments``, the function falls back to
    full-video decoding (single embedding for that video).

    Args:
        video_folder: Path to a directory containing ``.mp4`` video files.
        annotation_folder: Directory with ``.json`` annotations. If *None*,
            defaults to ``video_folder``.
        segment_key: JSON key that contains a list of segments with ``t0``/``t1``.
        min_interval_segments: Minimum number of valid intervals required to use
            interval decoding; otherwise full-video fallback is used.
        segment_progress_every: Print interval progress every N decoded segment
            attempts (in addition to first/last segment per video).
        model: Pre-loaded model object.  If *None*, uses the global singleton.
        params: Pre-loaded model params.  Must be provided together with *model*.
        num_frames: Number of frames to sample per video.
        resolution: Spatial resolution ``(H, W)`` for decoded frames.
        seed: Random seed for frame sampling.
        batch_size: Videos per forward-pass batch (default 1 to save RAM).

    Returns:
        ``(embeddings, labels)`` — only successfully decoded videos are included.
    """
    import gc

    import torch

    from data.utils import decode

    if model is None or params is None:
        model, params = _get_model_and_params()

    video_folder = Path(video_folder)
    annotation_folder = Path(annotation_folder) if annotation_folder is not None else video_folder
    video_files = sorted(video_folder.glob("*.mp4"), key=lambda p: p.stem)
    if not video_files:
        raise FileNotFoundError(f"No .mp4 files found in {video_folder}")

    print(f"Found {len(video_files)} videos in {video_folder}")
    print(f"Using annotations from: {annotation_folder}")

    rng = random.Random(seed)

    @jax.jit
    def _jit_video_forward(params, video_input):
        return model.get_adapted_video_embeddings(params, video_input, train=False)

    def _flush_batch(
        pending_tensors: list[torch.Tensor],
        pending_labels: list[str],
        emb_list: list[jax.Array],
        labels: list[str],
    ) -> None:
        """Run one forward pass on pending tensors and append outputs."""
        if not pending_tensors:
            return
        jax_frames = [from_dlpack(t.contiguous()) for t in pending_tensors]
        del pending_tensors[:]
        gc.collect()

        video_input = jnp.stack(jax_frames, axis=0)
        del jax_frames
        gc.collect()

        v_emb = _jit_video_forward(params, video_input)
        emb_list.append(np.asarray(l2_normalize(v_emb.astype(jnp.float32))))
        labels.extend(pending_labels)
        del pending_labels[:]

        del video_input, v_emb
        gc.collect()

    # Build a quick plan first so progress can include total segment count.
    decode_plan: list[tuple[Path, list[tuple[float, float]], bool]] = []
    total_interval_segments = 0
    for vf in video_files:
        ann_path = annotation_folder / f"{vf.stem}.json"
        intervals: list[tuple[float, float]] = []
        if ann_path.exists():
            try:
                with open(ann_path) as f:
                    ann_data = json.load(f)
                intervals = _extract_annotation_intervals(ann_data, segment_key=segment_key)
            except Exception:
                intervals = []
        use_intervals = len(intervals) >= min_interval_segments
        if use_intervals:
            total_interval_segments += len(intervals)
        decode_plan.append((vf, intervals, use_intervals))

    print(
        f"Planned interval decoding: {total_interval_segments} segments "
        f"(fallback videos: {sum(1 for _vf, _itv, use_itv in decode_plan if not use_itv)})"
    )
    if segment_progress_every < 1:
        segment_progress_every = 1

    emb_list: list[jax.Array] = []
    labels: list[str] = []
    pending_tensors: list[torch.Tensor] = []
    pending_labels: list[str] = []
    global_segment_done = 0
    global_segment_ok = 0

    for idx, (vf, intervals, use_intervals) in enumerate(decode_plan, start=1):
        print(f"  Processing video {idx} / {len(video_files)}: {vf.name} …")
        ann_path = annotation_folder / f"{vf.stem}.json"
        if not ann_path.exists():
            print(f"    NOTE: no annotation for {vf.name} at {ann_path.name} — full-video fallback")
        elif not use_intervals and len(intervals) == 0:
            # Distinguish parse failure / missing key from explicit short list.
            try:
                with open(ann_path) as f:
                    ann_data = json.load(f)
                if segment_key not in ann_data:
                    print(f"    NOTE: key '{segment_key}' not found in {ann_path.name} — full-video fallback")
            except Exception as exc:
                print(f"    WARNING: failed to parse {ann_path.name}: {exc} — full-video fallback")

        if use_intervals:
            print(f"    Using {len(intervals)} interval segments from '{segment_key}'")
            for seg_idx, (start_s, end_s) in enumerate(intervals):
                global_segment_done += 1
                if (
                    seg_idx == 0
                    or seg_idx + 1 == len(intervals)
                    or global_segment_done % segment_progress_every == 0
                ):
                    print(
                        "    Segment progress: "
                        f"video {seg_idx + 1}/{len(intervals)} | "
                        f"global {global_segment_done}/{total_interval_segments} "
                        f"(ok={global_segment_ok})"
                    )
                try:
                    video_tensor, _meta = decode(
                        str(vf),
                        num_frames,
                        resolution,
                        decode_method="pyav",
                        resize_method="center_crop_resize",
                        frame_sampling_method="interval",
                        output_range="unit",
                        dtype=torch.bfloat16,
                        rng=rng,
                        interval=(start_s, end_s),
                    )
                    if video_tensor.shape[0] == 0:
                        print(
                            f"    WARNING: {vf.name} segment {seg_idx} [{start_s:.3f}, {end_s:.3f}] "
                            "decoded to 0 frames — skipping"
                        )
                        continue
                    pending_tensors.append(video_tensor)
                    pending_labels.append(f"{vf.stem}#seg{seg_idx:03d}")
                    global_segment_ok += 1
                except Exception as exc:
                    print(f"    WARNING: failed interval decode for {vf.name} seg {seg_idx}: {exc} — skipping")
                    continue

                if len(pending_tensors) >= batch_size:
                    _flush_batch(pending_tensors, pending_labels, emb_list, labels)
        else:
            if ann_path.exists():
                print(
                    f"    NOTE: only {len(intervals)} valid segments (< {min_interval_segments}); "
                    "using full-video fallback"
                )
            try:
                video_tensor, _meta = decode(
                    str(vf),
                    num_frames,
                    resolution,
                    decode_method="pyav",
                    resize_method="center_crop_resize",
                    frame_sampling_method="max_stride",
                    output_range="unit",
                    dtype=torch.bfloat16,
                    rng=rng,
                )
                if video_tensor.shape[0] == 0:
                    print(f"    WARNING: {vf.name} decoded to 0 frames — skipping")
                    continue
                pending_tensors.append(video_tensor)
                pending_labels.append(vf.stem)
            except Exception as exc:
                print(f"    WARNING: failed to decode {vf.name}: {exc} — skipping")
                continue

            if len(pending_tensors) >= batch_size:
                _flush_batch(pending_tensors, pending_labels, emb_list, labels)

    _flush_batch(pending_tensors, pending_labels, emb_list, labels)

    if not emb_list:
        raise RuntimeError(
            f"All videos in {video_folder} failed to decode. "
            "Check that they are valid .mp4 files with enough frames."
        )

    print(f"  Successfully embedded {len(labels)} / {len(video_files)} videos")
    if total_interval_segments > 0:
        print(
            "  Interval decode totals: "
            f"attempted={global_segment_done}, successful={global_segment_ok}, "
            f"failed={global_segment_done - global_segment_ok}"
        )

    embeddings = jnp.asarray(np.concatenate(emb_list, axis=0))

    if npz_filepath is not None:
        _save_path = Path(npz_filepath).expanduser().resolve()
        _save_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            _save_path,
            video_embeddings=np.asarray(embeddings, dtype=np.float32),
            video_labels=np.asarray(labels, dtype=str),
        )
        print(f"  Saved video embeddings ({len(labels)} segments) to {_save_path}")

    return embeddings, labels


def _extract_caption(data: dict, caption_key: str) -> str | None:
    """Extract a plain-text caption from a JSON annotation dict.

    Handles two formats:
      1. Simple:  ``{"summary": "A dog catches a ball."}``
      2. Nested:  ``{"summary": "[{\\"summary_multimodal\\": \\"...\\" }]"}``
         where the value is a **stringified** JSON array.  In that case the
         function parses the inner JSON and looks for ``summary_multimodal``,
         then ``summary_video_only``, then ``summary``, then falls back to
         the first string value it finds.

    Returns *None* if no caption could be extracted.
    """
    raw = data.get(caption_key)
    if raw is None:
        return None

    # If the value is already a clean string (not JSON-encoded), use it.
    if isinstance(raw, str):
        stripped = raw.strip()
        # Detect stringified JSON (starts with '[' or '{')
        if stripped and stripped[0] in ("[", "{"):
            try:
                inner = json.loads(stripped)
                # Unwrap single-element list
                if isinstance(inner, list) and len(inner) > 0:
                    inner = inner[0]
                if isinstance(inner, dict):
                    # Prefer multimodal > video_only > summary > first string
                    for key in ("summary_multimodal", "summary_video_only", "summary"):
                        if key in inner and isinstance(inner[key], str):
                            return inner[key]
                    # Fallback: grab the first string value
                    for v in inner.values():
                        if isinstance(v, str) and len(v) > 20:
                            return v
                # If inner is itself a string, use it
                if isinstance(inner, str):
                    return inner
            except (json.JSONDecodeError, TypeError):
                pass  # not valid JSON — treat as plain text
        return raw

    return None


def get_text_embeddings(
    text_folder: str | Path,
    npz_filepath: str | Path | None = None,
    *,
    model=None,
    params=None,
    caption_key: str = "summary",
    segment_level: bool = True,
    segment_key: str = "video_descriptions",
    segment_text_key: str = "text",
    min_interval_segments: int = 2,
    fallback_caption_key: str = "summary_multimodal",
    batch_size: int = 32,
) -> tuple[jax.Array, list[str]]:
    """Compute L2-normalized text embeddings from ``.json`` caption files.

    Each JSON file should have a top-level key (default ``"summary"``) whose
    value is either a plain caption string **or** a stringified JSON array
    containing nested summary fields (``summary_multimodal``, etc.).  The
    function auto-detects the format and extracts the best available caption.

    Args:
        text_folder: Path to a directory containing ``.json`` caption files.
        model: Pre-loaded model object.  If *None*, uses the global singleton.
        params: Pre-loaded model params.  Must be provided together with *model*.
        caption_key: The JSON key that holds the caption text.
        segment_level: If *True*, extract text per segment from ``segment_key``
            and emit labels like ``<stem>#seg000``.
        segment_key: JSON key containing segment objects with ``t0``/``t1`` and
            segment text.
        segment_text_key: Key inside each segment object to embed as text.
        min_interval_segments: Minimum number of valid segments required to use
            segment-level text; otherwise fallback to single caption text.
        fallback_caption_key: Caption key to use for fallback when segment-level
            extraction is unavailable or too short.
        batch_size: Texts per forward-pass batch.

    Returns:
        ``(embeddings, labels)`` where *embeddings* is a JAX array of shape
        ``(N, D)`` and *labels* is a list of filename stems (without extension).
    """
    if model is None or params is None:
        model, params = _get_model_and_params()

    text_folder = Path(text_folder)
    json_files = sorted(text_folder.glob("*.json"), key=lambda p: p.stem)
    if not json_files:
        raise FileNotFoundError(f"No .json files found in {text_folder}")

    labels: list[str] = []
    texts: list[str] = []
    for jf in json_files:
        with open(jf) as f:
            data = json.load(f)
        if segment_level:
            segment_items = _extract_annotation_segment_texts(
                data,
                segment_key=segment_key,
                segment_text_key=segment_text_key,
            )
            if len(segment_items) >= min_interval_segments:
                print(
                    f"  {jf.stem}: using {len(segment_items)} segment texts "
                    f"from '{segment_key}.{segment_text_key}'"
                )
                for seg_idx, (_start_s, _end_s, seg_text) in enumerate(segment_items):
                    labels.append(f"{jf.stem}#seg{seg_idx:03d}")
                    texts.append(seg_text)
                continue

            print(
                f"  {jf.stem}: only {len(segment_items)} valid segments "
                f"(< {min_interval_segments}) — using fallback caption"
            )

        caption = _extract_caption(data, fallback_caption_key)
        if caption is None:
            caption = _extract_caption(data, caption_key)
        if caption is None:
            print(f"  Warning: {jf.name} — could not extract caption, skipping")
            continue
        labels.append(jf.stem)
        texts.append(caption)
        preview = caption[:80].replace("\n", " ")
        print(f"  {jf.stem}: \"{preview}…\"")

    if not texts:
        raise ValueError(
            f"No captions found — none of the JSON files in {text_folder} "
            f"contain a usable caption under key '{caption_key}'"
        )

    print(f"Found {len(texts)} captions in {text_folder}")

    import gc

    emb_list: list[jax.Array] = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]
        print(f"  Processing texts {i + 1}–{i + len(batch_texts)} / {len(texts)} …")
        tokenized = model.tokenize(batch_texts)
        tokenized = to_bfloat16(tokenized)
        t_emb = model.get_adapted_text_embeddings(params, tokenized, train=False)
        # Move to numpy immediately so the JAX device buffer is freed.
        emb_list.append(np.asarray(l2_normalize(t_emb.astype(jnp.float32))))
        del tokenized, t_emb
        gc.collect()

    embeddings = jnp.asarray(np.concatenate(emb_list, axis=0))

    if npz_filepath is not None:
        _save_path = Path(npz_filepath).expanduser().resolve()
        _save_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            _save_path,
            text_embeddings=np.asarray(embeddings, dtype=np.float32),
            text_labels=np.asarray(labels, dtype=str),
        )
        print(f"  Saved text embeddings ({len(labels)} segments) to {_save_path}")

    return embeddings, labels


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def plot_similarity_matrix(
    video_embeddings,
    text_embeddings,
    *,
    labels: list[str] | None = None,
    show: bool = True,
):
    """Plot the cosine-similarity matrix between text and video embeddings.

    Accepts either raw arrays of shape ``(N, D)`` **or** the ``(embeddings,
    labels)`` tuples returned by :func:`get_video_embeddings` /
    :func:`get_text_embeddings`.

    Args:
        video_embeddings: Array ``(N, D)`` or ``(array, labels)`` tuple.
        text_embeddings: Array ``(M, D)`` or ``(array, labels)`` tuple.
        labels: Optional tick labels.  Auto-detected from tuples if not given.
        show: If *True* (default), display the plot immediately.

    Returns:
        The matplotlib ``Figure`` when ``show=False``; otherwise ``None``.
    """
    import matplotlib.pyplot as plt

    # Unpack (embeddings, labels) tuples from get_video/text_embeddings
    if isinstance(video_embeddings, (tuple, list)) and len(video_embeddings) == 2:
        video_embeddings, _vid_labels = video_embeddings
        if labels is None:
            labels = _vid_labels
    if isinstance(text_embeddings, (tuple, list)) and len(text_embeddings) == 2:
        text_embeddings, _txt_labels = text_embeddings
        if labels is None:
            labels = _txt_labels

    video_np = np.asarray(video_embeddings)
    text_np = np.asarray(text_embeddings)
    sim_matrix = text_np @ video_np.T
    n_text, n_video = sim_matrix.shape

    if labels is None:
        labels = [f"sample_{i}" for i in range(max(n_text, n_video))]

    fig, ax = plt.subplots(
        figsize=(max(5, n_video * 0.8), max(4, n_text * 0.7))
    )
    im = ax.imshow(sim_matrix, cmap="viridis", aspect="auto")
    ax.set_xticks(range(n_video))
    ax.set_yticks(range(n_text))
    ax.set_xticklabels(labels[:n_video], rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(labels[:n_text], fontsize=8)
    ax.set_xlabel("Video")
    ax.set_ylabel("Text")
    ax.set_title("Text\u2013Video Cosine Similarity")

    # Annotate cells
    for i in range(n_text):
        for j in range(n_video):
            color = "white" if sim_matrix[i, j] < sim_matrix.mean() else "black"
            ax.text(
                j, i, f"{sim_matrix[i, j]:.2f}",
                ha="center", va="center", fontsize=7, color=color,
            )

    plt.colorbar(im, label="cosine similarity")
    plt.tight_layout()
    if show:
        plt.show()

    # Retrieval statistics
    n = min(n_text, n_video)
    diag_mean = float(np.mean(np.diag(sim_matrix[:n, :n])))
    mask = ~np.eye(n, dtype=bool)
    off_diag_mean = float(np.mean(sim_matrix[:n, :n][mask])) if n > 1 else float("nan")
    print(f"Diagonal mean:     {diag_mean:.4f}")
    print(f"Off-diagonal mean: {off_diag_mean:.4f}")
    print(f"Gap:               {diag_mean - off_diag_mean:.4f}")

    return None if show else fig


def align_embeddings_by_labels(
    video_embeddings: jax.Array | np.ndarray,
    video_labels: list[str],
    text_embeddings: jax.Array | np.ndarray,
    text_labels: list[str],
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Align video/text embeddings by shared labels in video-label order.

    This is useful before plotting when a few items were skipped on one side
    (for example, failed segment decode on video side).
    """
    video_np = np.asarray(video_embeddings)
    text_np = np.asarray(text_embeddings)
    if video_np.shape[0] != len(video_labels):
        raise ValueError("video_embeddings row count must match len(video_labels)")
    if text_np.shape[0] != len(text_labels):
        raise ValueError("text_embeddings row count must match len(text_labels)")

    text_index = {lab: i for i, lab in enumerate(text_labels)}
    keep_video_indices: list[int] = []
    keep_text_indices: list[int] = []
    keep_labels: list[str] = []

    for i, lab in enumerate(video_labels):
        j = text_index.get(lab)
        if j is None:
            continue
        keep_video_indices.append(i)
        keep_text_indices.append(j)
        keep_labels.append(lab)

    if not keep_labels:
        raise ValueError("No shared labels found between video_labels and text_labels")

    dropped_v = len(video_labels) - len(keep_video_indices)
    dropped_t = len(text_labels) - len(keep_text_indices)
    print(
        f"Aligned by labels: kept={len(keep_labels)}, "
        f"dropped_video={dropped_v}, dropped_text={dropped_t}"
    )
    return (
        video_np[np.asarray(keep_video_indices)],
        text_np[np.asarray(keep_text_indices)],
        keep_labels,
    )


def _unpack_embedding_data(data):
    """Accept ``(embeddings, labels)`` tuple or a bare embeddings array.

    Returns ``(np_array_2d, labels_list)``.
    """
    if isinstance(data, (tuple, list)) and len(data) == 2:
        emb, labels = data
        emb_np = np.asarray(emb)
        if emb_np.ndim == 2 and isinstance(labels, list):
            return emb_np.astype(np.float32, copy=False), list(labels)
    arr = np.asarray(data).astype(np.float32, copy=False)
    return arr, [f"item_{i}" for i in range(arr.shape[0])]


def plot_tsne(
    video_data,
    text_data,
    *,
    bg_npz_vid_filepath: str | Path | None = None,
    bg_npz_text_filepath: str | Path | None = None,
    bg_alpha: float = 0.30,
    bg_s: float = 15,
    seed: int = 42,
    perplexity: float | None = None,
    axis_padding: float = 0.10,
    figsize: tuple[float, float] = (12, 10),
    show: bool = True,
    save_path: str | Path | None = None,
):
    """Plot a joint t-SNE of video and text embeddings.

    Supports two modes:

    **Simple mode** (no background npz files):
        All video-text pairs are plotted with distinct per-video colours.
        Video segments as circles, text segments as triangles, with dashed
        lines connecting matched pairs.

    **Background mode** (with ``bg_npz_vid_filepath`` & ``bg_npz_text_filepath``):
        Background points (loaded from the npz files) are drawn in muted grey.
        Foreground points (from ``video_data`` / ``text_data``) are drawn in
        opaque, per-video colours.  Both layers use circles for video and
        triangles for text, with dashed lines for matched pairs.

    Args:
        video_data: Return value from :func:`get_video_embeddings` — either
            an ``(embeddings, labels)`` tuple or a raw array ``(N, D)``.
        text_data: Return value from :func:`get_text_embeddings` — either
            an ``(embeddings, labels)`` tuple or a raw array ``(N, D)``.
        bg_npz_vid_filepath: ``.npz`` file with keys ``video_embeddings`` and
            ``video_labels`` (produced by ``get_video_embeddings(..., npz_filepath=...)``).
        bg_npz_text_filepath: ``.npz`` file with keys ``text_embeddings`` and
            ``text_labels`` (produced by ``get_text_embeddings(..., npz_filepath=...)``).
        bg_alpha: Opacity for background points (default 0.30).
        bg_s: Marker size for background points (default 15).
        seed: Random seed for t-SNE.
        perplexity: t-SNE perplexity; auto-chosen if *None*.
        axis_padding: Extra padding as fraction of data range.
        figsize: Figure size.
        show: Display the plot immediately.
        save_path: If given, save the figure to this path.

    Returns:
        The matplotlib ``Figure`` when ``show=False``; otherwise ``None``.
    """
    import matplotlib.lines as mlines

    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE

    # ------------------------------------------------------------------
    # Unpack foreground data
    # ------------------------------------------------------------------
    fg_vid_emb, fg_vid_labels = _unpack_embedding_data(video_data)
    fg_txt_emb, fg_txt_labels = _unpack_embedding_data(text_data)

    # ------------------------------------------------------------------
    # Optionally load background data from npz files
    # ------------------------------------------------------------------
    has_background = (
        bg_npz_vid_filepath is not None and bg_npz_text_filepath is not None
    )
    bg_vid_emb = bg_vid_labels = bg_txt_emb = bg_txt_labels = None

    if has_background:
        bg_vid_data = np.load(bg_npz_vid_filepath, allow_pickle=True)
        bg_vid_emb = bg_vid_data["video_embeddings"].astype(np.float32)
        bg_vid_labels = [str(x) for x in bg_vid_data["video_labels"]]

        bg_txt_data = np.load(bg_npz_text_filepath, allow_pickle=True)
        bg_txt_emb = bg_txt_data["text_embeddings"].astype(np.float32)
        bg_txt_labels = [str(x) for x in bg_txt_data["text_labels"]]

        print(
            f"Background: {len(bg_vid_labels)} video + "
            f"{len(bg_txt_labels)} text segments"
        )

    # ------------------------------------------------------------------
    # Combine all embeddings for a single joint t-SNE
    # ------------------------------------------------------------------
    parts: list[np.ndarray] = []
    if has_background:
        parts.append(bg_vid_emb)
        parts.append(bg_txt_emb)
    parts.append(fg_vid_emb)
    parts.append(fg_txt_emb)

    all_emb = np.concatenate(parts, axis=0)

    if perplexity is None:
        perplexity = min(30, max(2, len(all_emb) // 3))

    print(f"Running t-SNE on {len(all_emb)} points (perplexity={perplexity}) …")
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        random_state=seed,
        init="pca",
        learning_rate="auto",
    )
    all_coords = tsne.fit_transform(all_emb)

    # Split coordinates back
    idx = 0
    bg_vid_coords = bg_txt_coords = None
    if has_background:
        n_bgv = len(bg_vid_labels)
        bg_vid_coords = all_coords[idx : idx + n_bgv]
        idx += n_bgv
        n_bgt = len(bg_txt_labels)
        bg_txt_coords = all_coords[idx : idx + n_bgt]
        idx += n_bgt

    n_fgv = fg_vid_emb.shape[0]
    fg_vid_coords = all_coords[idx : idx + n_fgv]
    idx += n_fgv
    n_fgt = fg_txt_emb.shape[0]
    fg_txt_coords = all_coords[idx : idx + n_fgt]

    # ------------------------------------------------------------------
    # Build figure
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=figsize)

    # ---- Layer 1: Background (muted grey) ----------------------------
    if has_background:
        bg_color = "lightgray"
        # Using bg_alpha and bg_s from parameters

        # Background video (circles)
        ax.scatter(
            bg_vid_coords[:, 0], bg_vid_coords[:, 1],
            c=bg_color, alpha=bg_alpha, s=bg_s, marker="o",
            edgecolors="none", zorder=1,
        )
        # Background text (triangles)
        ax.scatter(
            bg_txt_coords[:, 0], bg_txt_coords[:, 1],
            c=bg_color, alpha=bg_alpha, s=bg_s, marker="^",
            edgecolors="none", zorder=1,
        )

        # Background connecting lines
        bg_txt_lookup = {lab: i for i, lab in enumerate(bg_txt_labels)}
        for i, lab in enumerate(bg_vid_labels):
            j = bg_txt_lookup.get(lab)
            if j is not None:
                ax.plot(
                    [bg_vid_coords[i, 0], bg_txt_coords[j, 0]],
                    [bg_vid_coords[i, 1], bg_txt_coords[j, 1]],
                    color=bg_color, alpha=0.15, linestyle=":",
                    linewidth=0.5, zorder=0,
                )

    # ---- Layer 2: Foreground (opaque, per-video colours) -------------
    fg_vid_video_ids = [_strip_segment_suffix(lab) for lab in fg_vid_labels]
    fg_txt_video_ids = [_strip_segment_suffix(lab) for lab in fg_txt_labels]
    unique_video_ids = sorted(set(fg_vid_video_ids + fg_txt_video_ids))

    cmap = plt.cm.get_cmap("tab10", max(len(unique_video_ids), 1))
    vid_to_color = {vid: cmap(i) for i, vid in enumerate(unique_video_ids)}

    fg_alpha = 0.90
    fg_s_vid = 100
    fg_s_txt = 120

    for vid_id in unique_video_ids:
        color = vid_to_color[vid_id]

        # Video segments (circles)
        v_idx = [i for i, x in enumerate(fg_vid_video_ids) if x == vid_id]
        if v_idx:
            ax.scatter(
                fg_vid_coords[v_idx, 0], fg_vid_coords[v_idx, 1],
                color=color, marker="o", s=fg_s_vid, alpha=fg_alpha,
                edgecolors="white", linewidth=1.5,
                label=f"{vid_id} (video)", zorder=3,
            )

        # Text segments (triangles)
        t_idx = [i for i, x in enumerate(fg_txt_video_ids) if x == vid_id]
        if t_idx:
            ax.scatter(
                fg_txt_coords[t_idx, 0], fg_txt_coords[t_idx, 1],
                color=color, marker="^", s=fg_s_txt, alpha=fg_alpha,
                edgecolors="black", linewidth=1.0,
                label=f"{vid_id} (text)", zorder=3,
            )

    # Foreground connecting lines (match by label)
    fg_txt_lookup = {lab: i for i, lab in enumerate(fg_txt_labels)}
    for i, lab in enumerate(fg_vid_labels):
        j = fg_txt_lookup.get(lab)
        if j is not None:
            vid_id = _strip_segment_suffix(lab)
            color = vid_to_color[vid_id]
            ax.plot(
                [fg_vid_coords[i, 0], fg_txt_coords[j, 0]],
                [fg_vid_coords[i, 1], fg_txt_coords[j, 1]],
                color=color, alpha=0.6, linestyle="--",
                linewidth=1.5, zorder=2,
            )

    # ---- Axis padding ------------------------------------------------
    x_min, x_max = float(np.min(all_coords[:, 0])), float(np.max(all_coords[:, 0]))
    y_min, y_max = float(np.min(all_coords[:, 1])), float(np.max(all_coords[:, 1]))
    x_range = max(x_max - x_min, 1e-8)
    y_range = max(y_max - y_min, 1e-8)
    x_pad = x_range * max(axis_padding, 0.0)
    y_pad = y_range * max(axis_padding, 0.0)
    ax.set_xlim(x_min - x_pad, x_max + x_pad)
    ax.set_ylim(y_min - y_pad, y_max + y_pad)

    # ---- Legend ------------------------------------------------------
    handles = []
    if has_background:
        handles.append(mlines.Line2D(
            [], [], color="lightgray", marker="o", linestyle="None",
            markersize=6, label="Background",
        ))
    handles.append(mlines.Line2D(
        [], [], color="gray", marker="o", linestyle="None",
        markersize=8, label="Video segment",
    ))
    handles.append(mlines.Line2D(
        [], [], color="gray", marker="^", linestyle="None",
        markersize=8, label="Text segment",
    ))
    handles.append(mlines.Line2D(
        [], [], color="gray", linestyle="--", linewidth=1.5,
        label="Video–text pair",
    ))
    for vid_id in unique_video_ids:
        handles.append(mlines.Line2D(
            [], [], color=vid_to_color[vid_id], marker="s",
            linestyle="None", markersize=8, label=vid_id,
        ))

    ax.legend(handles=handles, bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    ax.set_title("t-SNE: Video (●) & Text (▲) Embeddings")
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Plot saved to {save_path}")

    if show:
        plt.show()

    return None if show else fig


def _to_numpy_2d(x: jax.Array | np.ndarray, *, name: str) -> np.ndarray:
    """Convert input to a 2D numpy array."""
    arr = np.asarray(x)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be 2D with shape (N, D); got shape {arr.shape}")
    return arr


def _strip_segment_suffix(label: str) -> str:
    """Map labels like '<video>#seg003' to '<video>'."""
    return re.sub(r"#seg\d+$", "", str(label))


def _stable_color_for_video_id(video_id: str) -> tuple[float, float, float]:
    """Generate a deterministic RGB color for a video id."""
    digest = hashlib.md5(video_id.encode("utf-8")).hexdigest()
    hue = int(digest[:8], 16) / float(16**8 - 1)
    # Keep saturation/value fixed for readability and contrast.
    return colorsys.hsv_to_rgb(hue, 0.65, 0.90)


def save_embeddings_npz(
    output_path: str | Path,
    video_embeddings: jax.Array | np.ndarray,
    video_labels: list[str],
    *,
    text_embeddings: jax.Array | np.ndarray | None = None,
    text_labels: list[str] | None = None,
) -> Path:
    """Save embeddings + labels into a compressed .npz file.

    Saved keys:
      - video_embeddings (N_video, D)
      - video_labels (N_video,)
      - text_embeddings (N_text, D), optional
      - text_labels (N_text,), optional
    """
    output = Path(output_path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    video_np = _to_numpy_2d(video_embeddings, name="video_embeddings")
    if len(video_labels) != video_np.shape[0]:
        raise ValueError("len(video_labels) must match video_embeddings rows")

    payload: dict[str, np.ndarray] = {
        "video_embeddings": video_np.astype(np.float32, copy=False),
        "video_labels": np.asarray(video_labels, dtype=str),
    }

    if text_embeddings is not None:
        text_np = _to_numpy_2d(text_embeddings, name="text_embeddings")
        payload["text_embeddings"] = text_np.astype(np.float32, copy=False)
        if text_labels is None:
            text_labels = [f"text_{i}" for i in range(text_np.shape[0])]
        if len(text_labels) != text_np.shape[0]:
            raise ValueError("len(text_labels) must match text_embeddings rows")
        payload["text_labels"] = np.asarray(text_labels, dtype=str)
    elif text_labels is not None:
        raise ValueError("text_labels was provided but text_embeddings is None")

    np.savez_compressed(output, **payload)
    return output


def load_embeddings_npz(npz_path: str | Path) -> dict[str, np.ndarray | list[str] | None]:
    """Load embedding arrays from .npz with tolerant key aliases."""
    path = Path(npz_path).expanduser().resolve()
    with np.load(path, allow_pickle=True) as data:
        keys = set(data.files)

        def _pick(candidates: list[str], required: bool) -> np.ndarray | None:
            for key in candidates:
                if key in keys:
                    return np.asarray(data[key])
            if required:
                raise KeyError(f"Missing required key. Tried {candidates}, found {sorted(keys)}")
            return None

        video_embeddings = _pick(["video_embeddings", "video_embeds", "video_emb"], required=True)
        video_labels = _pick(["video_labels", "labels", "video_ids"], required=True)
        text_embeddings = _pick(["text_embeddings", "text_embeds", "text_emb"], required=False)
        text_labels = _pick(["text_labels", "caption_labels"], required=False)

    video_np = _to_numpy_2d(video_embeddings, name="video_embeddings")
    video_labels_list = [str(x) for x in np.asarray(video_labels).tolist()]
    if len(video_labels_list) != video_np.shape[0]:
        raise ValueError("video_labels length does not match video_embeddings rows")

    text_np: np.ndarray | None = None
    text_labels_list: list[str] | None = None
    if text_embeddings is not None:
        text_np = _to_numpy_2d(text_embeddings, name="text_embeddings")
        if text_labels is None:
            text_labels_list = [f"text_{i}" for i in range(text_np.shape[0])]
        else:
            text_labels_list = [str(x) for x in np.asarray(text_labels).tolist()]
            if len(text_labels_list) != text_np.shape[0]:
                raise ValueError("text_labels length does not match text_embeddings rows")

    return {
        "video_embeddings": video_np.astype(np.float32, copy=False),
        "video_labels": video_labels_list,
        "text_embeddings": None if text_np is None else text_np.astype(np.float32, copy=False),
        "text_labels": text_labels_list,
    }


def build_tsne_baseline(
    video_embeddings: jax.Array | np.ndarray,
    video_labels: list[str],
    *,
    text_embeddings: jax.Array | np.ndarray | None = None,
    text_labels: list[str] | None = None,
    perplexity: float = 30.0,
    metric: str = "cosine",
    random_state: int = 42,
    standardize: bool = True,
) -> dict:
    """Fit an openTSNE baseline that can later transform new points.

    Returns a pickle-serializable session dict containing:
      - baseline_coords: (N, 2) baseline t-SNE coordinates
      - labels/modality/video_ids for baseline points
      - tsne_model: fitted openTSNE embedding object (supports transform)
      - scaler: fitted StandardScaler or None
    """
    try:
        from openTSNE import TSNE
    except ImportError as exc:
        raise ImportError(
            "openTSNE is required for incremental t-SNE transforms. "
            "Install with: pip install openTSNE"
        ) from exc

    video_np = _to_numpy_2d(video_embeddings, name="video_embeddings")
    if len(video_labels) != video_np.shape[0]:
        raise ValueError("len(video_labels) must match video_embeddings rows")

    labels: list[str] = [str(x) for x in video_labels]
    modalities: list[str] = ["video"] * len(labels)
    all_parts: list[np.ndarray] = [video_np]

    if text_embeddings is not None:
        text_np = _to_numpy_2d(text_embeddings, name="text_embeddings")
        if text_labels is None:
            text_labels = [f"text_{i}" for i in range(text_np.shape[0])]
        if len(text_labels) != text_np.shape[0]:
            raise ValueError("len(text_labels) must match text_embeddings rows")
        all_parts.append(text_np)
        labels.extend([str(x) for x in text_labels])
        modalities.extend(["text"] * text_np.shape[0])

    all_emb = np.concatenate(all_parts, axis=0).astype(np.float32, copy=False)

    scaler = None
    all_emb_proc = all_emb
    if standardize:
        from sklearn.preprocessing import StandardScaler

        scaler = StandardScaler()
        all_emb_proc = scaler.fit_transform(all_emb).astype(np.float32, copy=False)

    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        metric=metric,
        random_state=random_state,
        initialization="pca",
    )
    embedding = tsne.fit(all_emb_proc)
    coords = np.asarray(embedding, dtype=np.float32)
    video_ids = [_strip_segment_suffix(lab) for lab in labels]

    return {
        "baseline_coords": coords,
        "labels": labels,
        "modalities": modalities,
        "video_ids": video_ids,
        "tsne_model": embedding,
        "scaler": scaler,
        "config": {
            "perplexity": perplexity,
            "metric": metric,
            "random_state": random_state,
            "standardize": standardize,
        },
    }


def build_tsne_baseline_from_npz(
    npz_path: str | Path,
    *,
    perplexity: float = 30.0,
    metric: str = "cosine",
    random_state: int = 42,
    standardize: bool = True,
) -> dict:
    """Load embeddings from .npz and fit an incremental t-SNE baseline."""
    loaded = load_embeddings_npz(npz_path)
    return build_tsne_baseline(
        loaded["video_embeddings"],
        loaded["video_labels"],
        text_embeddings=loaded["text_embeddings"],
        text_labels=loaded["text_labels"],
        perplexity=perplexity,
        metric=metric,
        random_state=random_state,
        standardize=standardize,
    )


def save_tsne_session(session: dict, output_path: str | Path) -> Path:
    """Persist a t-SNE baseline session to disk with pickle."""
    output = Path(output_path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "wb") as f:
        pickle.dump(session, f)
    return output


def load_tsne_session(session_path: str | Path) -> dict:
    """Load a previously saved t-SNE baseline session."""
    path = Path(session_path).expanduser().resolve()
    with open(path, "rb") as f:
        session = pickle.load(f)
    return session


def plot_tsne_session(
    session: dict,
    *,
    figsize: tuple[float, float] = (11, 8),
    title: str = "Baseline t-SNE: Video/Text Embeddings",
    alpha: float = 0.85,
    s: float = 35,
    show: bool = True,
    connect_pairs: bool = True,
):
    """Plot baseline points from a fitted t-SNE session."""
    import matplotlib.pyplot as plt

    coords = np.asarray(session["baseline_coords"])
    labels = list(session["labels"])
    modalities = list(session["modalities"])
    video_ids = list(session["video_ids"])

    fig, ax = plt.subplots(figsize=figsize)

    # Draw connections first so they appear behind the points.
    if connect_pairs:
        label_to_indices = {}
        for i, lab in enumerate(labels):
            if lab not in label_to_indices:
                label_to_indices[lab] = {}
            label_to_indices[lab][modalities[i]] = i

        for lab, mods in label_to_indices.items():
            if "video" in mods and "text" in mods:
                v_xy = coords[mods["video"]]
                t_xy = coords[mods["text"]]
                ax.plot(
                    [v_xy[0], t_xy[0]],
                    [v_xy[1], t_xy[1]],
                    color="gray",
                    alpha=0.2,
                    linestyle="--",
                    linewidth=0.6,
                    zorder=0,
                )

    unique_video_ids = sorted(set(video_ids))
    for vid in unique_video_ids:
        idx = [i for i, x in enumerate(video_ids) if x == vid]
        if not idx:
            continue
        color = _stable_color_for_video_id(vid)
        idx_video = [i for i in idx if modalities[i] == "video"]
        idx_text = [i for i in idx if modalities[i] == "text"]
        if idx_video:
            xy = coords[np.asarray(idx_video)]
            ax.scatter(
                xy[:, 0],
                xy[:, 1],
                c=[color],
                s=s,
                alpha=alpha,
                marker="o",
                label=vid,
                edgecolors="k",
                linewidths=0.3,
            )
        if idx_text:
            xy = coords[np.asarray(idx_text)]
            ax.scatter(xy[:, 0], xy[:, 1], c=[color], s=s + 5, alpha=alpha, marker="x")

    ax.set_title(title)
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")

    # Keep legend compact for many videos.
    if len(unique_video_ids) <= 25:
        ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    plt.tight_layout()
    if show:
        plt.show()
    return None if show else (fig, ax)


def overlay_tsne_points(
    session: dict,
    new_embeddings: jax.Array | np.ndarray,
    new_labels: list[str],
    *,
    modality: str = "video",
    ax=None,
    marker: str | None = None,
    s: float = 70,
    alpha: float = 0.95,
    color: Any | None = None,
    connect_to_baseline: bool = True,
) -> np.ndarray:
    """Transform and overlay new points onto an existing baseline t-SNE axes.

    Notes:
      - Requires a baseline session fitted by build_tsne_baseline.
      - `new_labels` can be segment labels (e.g. '<video>#seg004'); points from
        the same base video id share a deterministic color (unless overridden).
    """
    import matplotlib.pyplot as plt

    new_np = _to_numpy_2d(new_embeddings, name="new_embeddings").astype(np.float32, copy=False)
    if len(new_labels) != new_np.shape[0]:
        raise ValueError("len(new_labels) must match new_embeddings rows")

    scaler = session.get("scaler")
    tsne_model = session.get("tsne_model")
    if tsne_model is None:
        raise ValueError("Session is missing `tsne_model`; cannot transform new points.")

    new_proc = scaler.transform(new_np) if scaler is not None else new_np
    new_xy = np.asarray(tsne_model.transform(new_proc), dtype=np.float32)
    new_video_ids = [_strip_segment_suffix(lab) for lab in new_labels]

    if ax is None:
        _fig, ax = plt.subplots(figsize=(8, 6))

    # 1. Draw connections to baseline points with matching labels (opposite modality)
    if connect_to_baseline:
        other_mod = "text" if modality == "video" else "video"
        b_coords = session.get("baseline_coords")
        b_labels = session.get("labels", [])
        b_modalities = session.get("modalities", [])

        # Build lookup for baseline points of the other modality
        baseline_lookup = {}
        if b_coords is not None:
            for i, (lab, mod) in enumerate(zip(b_labels, b_modalities)):
                if mod == other_mod:
                    baseline_lookup[lab] = b_coords[i]

        for i, lab in enumerate(new_labels):
            if lab in baseline_lookup:
                v1 = new_xy[i]
                v2 = baseline_lookup[lab]
                ax.plot(
                    [v1[0], v2[0]],
                    [v1[1], v2[1]],
                    color="gray",
                    alpha=0.4,
                    linestyle=":",
                    linewidth=1.2,
                    zorder=0,
                )

    # 2. Plot the new points
    point_marker = marker if marker is not None else ("^" if modality == "video" else "P")
    for vid in sorted(set(new_video_ids)):
        idx = [i for i, x in enumerate(new_video_ids) if x == vid]
        xy = new_xy[np.asarray(idx)]
        # Use provided color if available, otherwise stable video color
        plot_color = color if color is not None else _stable_color_for_video_id(vid)
        ax.scatter(
            xy[:, 0],
            xy[:, 1],
            c=[plot_color],
            s=s,
            alpha=alpha,
            marker=point_marker,
            edgecolors="k",
            linewidths=1.5,  # Thicker border for overlaid points
        )

    return new_xy


# ---------------------------------------------------------------------------
# Checkpoint download helper
# ---------------------------------------------------------------------------

def download_checkpoint(
    gdrive_url: str,
    output_path: str | Path = "params.msgpack",
) -> Path:
    """Download a model checkpoint from Google Drive.

    Uses ``gdown`` (included in project dependencies).

    Args:
        gdrive_url: A Google Drive sharing URL.
        output_path: Local path to save the file.

    Returns:
        Path to the downloaded checkpoint.
    """
    import gdown

    out = Path(output_path)
    if out.exists():
        print(f"Checkpoint already exists at {out}, skipping download.")
        return out

    print(f"Downloading checkpoint to {out} ...")
    gdown.download(gdrive_url, str(out), fuzzy=True)
    print("Download complete.")
    return out

def plot_assignment_tsne(
    background_npz: str | Path,
    student_video_emb: np.ndarray,
    student_video_labels: list[str],
    student_text_emb: np.ndarray,
    student_text_labels: list[str],
    perplexity: int = 30,
    save_path: str | None = None
):
    """
    Plots a t-SNE of student data overlaid on a pre-computed background.
    
    Visual Encoding:
    - Background points: Small, Light Gray, Transparent.
    - Student Video segments: Circles (o), Colored by Video ID.
    - Student Text segments: Triangles (^), Colored by Video ID.
    - Connections: A line connects a Text segment to its corresponding Video segment.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE
    import matplotlib.lines as mlines

    # 1. Load Background Data
    print(f"Loading background data from {background_npz}...")
    bg_data = np.load(background_npz)
    bg_emb = bg_data['embeddings']
    # We don't strictly need background labels for the plot, just the points
    
    # 2. Combine All Data (Background + Student Video + Student Text)
    # We must run t-SNE on the combined set so they share the same space.
    combined_emb = np.vstack([bg_emb, student_video_emb, student_text_emb])
    
    # Track indices to split them back apart later
    n_bg = len(bg_emb)
    n_vid = len(student_video_emb)
    n_txt = len(student_text_emb)
    
    print(f"Running t-SNE on {len(combined_emb)} total points...")
    tsne = TSNE(n_components=2, init='pca', learning_rate='auto', perplexity=perplexity, random_state=42)
    all_coords = tsne.fit_transform(combined_emb)
    
    # 3. Split Coordinates
    bg_coords = all_coords[:n_bg]
    vid_coords = all_coords[n_bg : n_bg + n_vid]
    txt_coords = all_coords[n_bg + n_vid :]
    
    # 4. Setup Plot
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # --- Layer 1: Background (Context) ---
    ax.scatter(
        bg_coords[:, 0], bg_coords[:, 1],
        c='lightgray', alpha=0.3, s=10, 
        label='Background Data', zorder=1
    )
    
    # --- Layer 2: Student Data (Focus) ---
    # Helper to extract "root" video name from "video_name#seg001"
    def get_root(label):
        return label.split('#')[0]

    # Identify unique videos to assign colors
    unique_videos = list(set([get_root(l) for l in student_video_labels]))
    colormap = plt.cm.get_cmap('tab10') # distinct colors
    
    # Create a lookup for video coordinates: "video_name#seg001" -> (x, y)
    video_lookup = {label: coords for label, coords in zip(student_video_labels, vid_coords)}

    print("Plotting student data...")
    
    # Plot per video ID so they get the same color
    for i, video_id in enumerate(unique_videos):
        color = colormap(i)
        
        # A. Plot Video Segments (Circles)
        # Find indices for this video
        v_indices = [idx for idx, label in enumerate(student_video_labels) if get_root(label) == video_id]
        if v_indices:
            ax.scatter(
                vid_coords[v_indices, 0], vid_coords[v_indices, 1],
                color=color, marker='o', s=100, edgecolors='white', linewidth=1.5,
                label=f"{video_id} (Video)", zorder=3
            )

        # B. Plot Text Segments (Triangles)
        t_indices = [idx for idx, label in enumerate(student_text_labels) if get_root(label) == video_id]
        if t_indices:
            ax.scatter(
                txt_coords[t_indices, 0], txt_coords[t_indices, 1],
                color=color, marker='^', s=120, edgecolors='black', linewidth=1.0,
                label=f"{video_id} (Text)", zorder=3
            )
            
            # C. Draw Connecting Lines
            # For every text label, try to find the EXACT matching video segment label
            for t_idx in t_indices:
                t_label = student_text_labels[t_idx]
                t_xy = txt_coords[t_idx]
                
                # Check if we have the corresponding video segment in our lookup
                if t_label in video_lookup:
                    v_xy = video_lookup[t_label]
                    
                    # Draw line
                    ax.plot(
                        [t_xy[0], v_xy[0]], [t_xy[1], v_xy[1]],
                        color=color, alpha=0.6, linestyle='--', linewidth=1.5, zorder=2
                    )

    # 5. Polish
    ax.set_title("Student Assignment: Video Segments vs. Text Annotations", fontsize=14)
    ax.axis('off') # Hide axis numbers for cleaner look
    
    # Custom Legend
    # We want one entry per Video ID, plus symbols for Video/Text
    handles = []
    # Background
    handles.append(mlines.Line2D([], [], color='lightgray', marker='o', linestyle='None', label='Background'))
    # Shapes
    handles.append(mlines.Line2D([], [], color='black', marker='o', linestyle='None', label='Video Segment'))
    handles.append(mlines.Line2D([], [], color='black', marker='^', linestyle='None', label='Text Annotation'))
    
    ax.legend(handles=handles, loc='best')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Plot saved to {save_path}")
    
    plt.show()

def create_background_dataset(
    video_folder: str | Path,
    annotation_folder: str | Path,
    output_path: str | Path = "background.npz",
    batch_size: int = 16,
    force_segment_level: bool = True
) -> Path:
    """
    Generates a consolidated .npz file containing embeddings for all videos 
    and annotations in the provided folders. 
    
    This file is intended to be distributed to students as the 'background' 
    or 'context' for their t-SNE plots.

    Args:
        video_folder: Path to folder containing .mp4 files.
        annotation_folder: Path to folder containing .json files.
        output_path: Where to save the resulting .npz file.
        batch_size: Batch size for inference (adjust based on GPU memory).
        force_segment_level: If True, forces the model to extract segments 
            (min_interval_segments=1) to ensure the background distribution 
            matches the students' segment-based work.

    Returns:
        Path to the saved .npz file.
    """
    import numpy as np
    import os

    # Ensure model is loaded
    if _model is None:
        load_model()

    print(f"--- Generating Background Dataset ---")
    print(f"Video Source: {video_folder}")
    print(f"Annotation Source: {annotation_folder}")

    # 1. Compute Video Embeddings
    # We set min_interval_segments=1 if force_segment_level is True.
    # This prevents the model from falling back to "whole video" embeddings,
    # ensuring the background points represent specific moments in time.
    min_segments = 1 if force_segment_level else 2
    
    print(f"\n[1/3] Computing Video Embeddings (min_segments={min_segments})...")
    bg_vid_emb, _ = get_video_embeddings(
        video_folder,
        annotation_folder=annotation_folder,
        min_interval_segments=min_segments,
        batch_size=batch_size
    )

    # 2. Compute Text Embeddings
    print(f"\n[2/3] Computing Text Embeddings...")
    bg_txt_emb, _ = get_text_embeddings(
        annotation_folder,
        segment_level=force_segment_level,
        segment_key="video_descriptions",
        segment_text_key="text",
        batch_size=batch_size * 2 # Text is cheaper, can double batch
    )

    # 3. Combine and Save
    print(f"\n[3/3] Saving to {output_path}...")
    
    # Concatenate video and text embeddings into one large matrix
    # Shape: (N_video + N_text, D)
    full_bg_embeddings = np.concatenate([bg_vid_emb, bg_txt_emb], axis=0)
    
    # Ensure output directory exists
    out = Path(output_path).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    
    np.savez_compressed(
        out, 
        embeddings=full_bg_embeddings
    )
    
    print(f"Success! Saved {len(full_bg_embeddings)} embeddings.")
    print(f"Stats: {len(bg_vid_emb)} video segments, {len(bg_txt_emb)} text segments.")
    
    return out