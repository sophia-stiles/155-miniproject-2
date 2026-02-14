"""Importable utilities for video-text embedding inference.

Provides four main functions:
    - get_video_embeddings(video_folder)
    - get_text_embeddings(text_folder)
    - plot_similarity_matrix(video_embeddings, text_embeddings)
    - plot_tsne(video_embeddings, text_embeddings)

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

    # 2. Compute embeddings — point at your folders
    video_emb, labels = get_video_embeddings("path/to/videos/")
    text_emb,  _      = get_text_embeddings("path/to/captions/")

    # 3. Visualize
    plot_similarity_matrix(video_emb, text_emb, labels=labels)
    plot_tsne(video_emb, text_emb, labels=labels)
"""

from __future__ import annotations

import json
import random
import sys
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


# ---------------------------------------------------------------------------
# Embedding functions
# ---------------------------------------------------------------------------

def get_video_embeddings(
    video_folder: str | Path,
    *,
    annotation_folder: str | Path | None = None,
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
    return jnp.asarray(np.concatenate(emb_list, axis=0)), labels


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
    *,
    model=None,
    params=None,
    caption_key: str = "summary",
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
        caption = _extract_caption(data, caption_key)
        if caption is None:
            print(f"  Warning: {jf.name} — could not extract caption, skipping")
            continue
        labels.append(jf.stem)
        texts.append(caption)
        # Show a preview so the user can verify the right text was extracted
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

    return jnp.asarray(np.concatenate(emb_list, axis=0)), labels


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def plot_similarity_matrix(
    video_embeddings: jax.Array | np.ndarray,
    text_embeddings: jax.Array | np.ndarray,
    *,
    labels: list[str] | None = None,
    show: bool = True,
):
    """Plot the cosine-similarity matrix between text and video embeddings.

    Args:
        video_embeddings: Array of shape ``(N, D)``.
        text_embeddings: Array of shape ``(M, D)`` (typically ``M == N``).
        labels: Optional tick labels for each sample.

    Returns:
        The matplotlib ``Figure`` when ``show=False``; otherwise ``None``.
    """
    import matplotlib.pyplot as plt

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


def plot_tsne(
    video_embeddings: jax.Array | np.ndarray,
    text_embeddings: jax.Array | np.ndarray,
    *,
    labels: list[str] | None = None,
    seed: int = 42,
    perplexity: float | None = None,
    show: bool = True,
):
    """Plot a joint t-SNE of video and text embeddings.

    Video embeddings are shown as circles, text as crosses.
    Matched pairs (same index) are connected by dashed lines.

    Args:
        video_embeddings: Array of shape ``(N, D)``.
        text_embeddings: Array of shape ``(N, D)``.
        labels: Optional label per sample (length ``N``).
        seed: Random seed for t-SNE.
        perplexity: t-SNE perplexity; auto-chosen if *None*.

    Returns:
        The matplotlib ``Figure`` when ``show=False``; otherwise ``None``.
    """
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE

    video_np = np.asarray(video_embeddings)
    text_np = np.asarray(text_embeddings)
    n = video_np.shape[0]

    if labels is None:
        labels = [f"sample_{i}" for i in range(n)]
    assert len(labels) == n, f"Expected {n} labels, got {len(labels)}"

    all_emb = np.concatenate([video_np, text_np], axis=0)  # (2N, D)
    modality = ["video"] * n + ["text"] * n
    all_labels = list(labels) + list(labels)

    if perplexity is None:
        perplexity = min(5, max(2, len(all_emb) - 1))

    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        random_state=seed,
        init="pca",
        learning_rate="auto",
    )
    coords = tsne.fit_transform(all_emb)

    fig, ax = plt.subplots(figsize=(8, 6))
    unique_labels = sorted(set(labels))
    cmap = plt.cm.get_cmap("tab10", len(unique_labels))
    label_to_color = {lab: cmap(i) for i, lab in enumerate(unique_labels)}

    for i in range(len(all_emb)):
        color = label_to_color[all_labels[i]]
        marker = "o" if modality[i] == "video" else "x"
        ax.scatter(
            coords[i, 0], coords[i, 1],
            c=[color], marker=marker, s=80,
            edgecolors="k" if marker == "o" else "none", linewidths=0.5,
        )

    # Lines connecting matched video-text pairs
    for i in range(n):
        ax.plot(
            [coords[i, 0], coords[n + i, 0]],
            [coords[i, 1], coords[n + i, 1]],
            color=label_to_color[labels[i]], alpha=0.3, linestyle="--", linewidth=0.8,
        )

    # Legend for labels
    for lab in unique_labels:
        ax.scatter([], [], c=[label_to_color[lab]], marker="s", label=lab)
    # Legend for modality
    ax.scatter([], [], c="gray", marker="o", label="video", edgecolors="k", linewidths=0.5)
    ax.scatter([], [], c="gray", marker="x", label="text")

    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    ax.set_title("t-SNE: Video (o) & Text (x) Embeddings")
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    plt.tight_layout()
    if show:
        plt.show()

    return None if show else fig


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
