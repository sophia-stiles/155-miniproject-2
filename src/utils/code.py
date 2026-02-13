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
        print(f"Loading checkpoint: {ckpt_path}")
        with open(ckpt_path, "rb") as f:
            loaded = serialization.from_bytes(_params, f.read())
        _params = jax.tree_util.tree_map(jnp.asarray, loaded)
    else:
        print("Using pretrained weights (no checkpoint)")

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

def l2_normalize(x: jax.Array, eps: float = 1e-8) -> jax.Array:
    """L2-normalize along the last axis."""
    return x / (jnp.linalg.norm(x, axis=-1, keepdims=True) + eps)


# ---------------------------------------------------------------------------
# Embedding functions
# ---------------------------------------------------------------------------

def get_video_embeddings(
    video_folder: str | Path,
    *,
    num_frames: int = 16,
    resolution: tuple[int, int] = (288, 288),
    seed: int = 42,
    batch_size: int = 8,
) -> tuple[jax.Array, list[str]]:
    """Compute L2-normalized video embeddings for every ``.mp4`` in a folder.

    Files are discovered as ``video_folder/*.mp4`` and sorted by filename stem
    so that the ordering is consistent with :func:`get_text_embeddings` when
    both folders contain files with the same names.

    Args:
        video_folder: Path to a directory containing ``.mp4`` video files.
        num_frames: Number of frames to sample per video.
        resolution: Spatial resolution ``(H, W)`` for decoded frames.
        seed: Random seed for frame sampling.
        batch_size: Videos per forward-pass batch.

    Returns:
        ``(embeddings, labels)`` where *embeddings* is a JAX array of shape
        ``(N, D)`` and *labels* is a list of filename stems (without extension).
    """
    import torch

    from data.utils import decode

    model, params = _get_model_and_params()

    video_folder = Path(video_folder)
    video_files = sorted(video_folder.glob("*.mp4"), key=lambda p: p.stem)
    if not video_files:
        raise FileNotFoundError(f"No .mp4 files found in {video_folder}")

    labels = [p.stem for p in video_files]
    print(f"Found {len(video_files)} videos in {video_folder}")

    rng = random.Random(seed)

    # Decode all videos into tensors
    video_tensors: list[torch.Tensor] = []
    for vf in video_files:
        video_tensor, _meta = decode(
            str(vf),
            num_frames,
            resolution,
            decode_method="decord",
            resize_method="center_crop_resize",
            frame_sampling_method="max_stride",
            output_range="unit",
            dtype=torch.float32,
            rng=rng,
        )
        video_tensors.append(video_tensor)

    # Forward in batches
    emb_list: list[jax.Array] = []
    for i in range(0, len(video_tensors), batch_size):
        batch = torch.stack(video_tensors[i : i + batch_size], dim=0)
        video_input = jnp.asarray(batch.numpy())
        v_emb = model.get_adapted_video_embeddings(params, video_input, train=False)
        emb_list.append(l2_normalize(v_emb.astype(jnp.float32)))

    return jnp.concatenate(emb_list, axis=0), labels


def get_text_embeddings(
    text_folder: str | Path,
    *,
    caption_key: str = "summary",
    batch_size: int = 32,
) -> tuple[jax.Array, list[str]]:
    """Compute L2-normalized text embeddings from ``.json`` caption files.

    Each JSON file should have a top-level key (default ``"summary"``) whose
    value is the caption string.  Files are discovered as
    ``text_folder/*.json`` and sorted by filename stem so the ordering matches
    :func:`get_video_embeddings` when both folders share the same names.

    Args:
        text_folder: Path to a directory containing ``.json`` caption files.
        caption_key: The JSON key that holds the caption text.
        batch_size: Texts per forward-pass batch.

    Returns:
        ``(embeddings, labels)`` where *embeddings* is a JAX array of shape
        ``(N, D)`` and *labels* is a list of filename stems (without extension).
    """
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
        if caption_key not in data:
            print(f"  Warning: {jf.name} has no '{caption_key}' key, skipping")
            continue
        labels.append(jf.stem)
        texts.append(data[caption_key])

    if not texts:
        raise ValueError(
            f"No captions found — none of the JSON files in {text_folder} "
            f"contain the key '{caption_key}'"
        )

    print(f"Found {len(texts)} captions in {text_folder}")

    emb_list: list[jax.Array] = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]
        tokenized = model.tokenize(batch_texts)
        t_emb = model.get_adapted_text_embeddings(params, tokenized, train=False)
        emb_list.append(l2_normalize(t_emb.astype(jnp.float32)))

    return jnp.concatenate(emb_list, axis=0), labels


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def plot_similarity_matrix(
    video_embeddings: jax.Array | np.ndarray,
    text_embeddings: jax.Array | np.ndarray,
    *,
    labels: list[str] | None = None,
):
    """Plot the cosine-similarity matrix between text and video embeddings.

    Args:
        video_embeddings: Array of shape ``(N, D)``.
        text_embeddings: Array of shape ``(M, D)`` (typically ``M == N``).
        labels: Optional tick labels for each sample.

    Returns:
        The matplotlib ``Figure``.
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
    plt.show()

    # Retrieval statistics
    n = min(n_text, n_video)
    diag_mean = float(np.mean(np.diag(sim_matrix[:n, :n])))
    mask = ~np.eye(n, dtype=bool)
    off_diag_mean = float(np.mean(sim_matrix[:n, :n][mask])) if n > 1 else float("nan")
    print(f"Diagonal mean:     {diag_mean:.4f}")
    print(f"Off-diagonal mean: {off_diag_mean:.4f}")
    print(f"Gap:               {diag_mean - off_diag_mean:.4f}")

    return fig


def plot_tsne(
    video_embeddings: jax.Array | np.ndarray,
    text_embeddings: jax.Array | np.ndarray,
    *,
    labels: list[str] | None = None,
    seed: int = 42,
    perplexity: float | None = None,
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
        The matplotlib ``Figure``.
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
    plt.show()

    return fig


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
