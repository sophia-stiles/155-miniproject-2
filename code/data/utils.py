from __future__ import annotations

import random
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import jaxtyping as jt
import numpy as np
import torch
from einops import rearrange
from torch.nn import functional as F

if TYPE_CHECKING:
    from data.base import DecodeMethod, FrameSamplingMethod, OutputRange, ResizeMethod


# ==================== Temporal Resampling ====================


def select_video_indices(
    total_frames: int,
    num_frames: int,
    method: FrameSamplingMethod = "max_stride",
    rng: random.Random | None = None,
) -> list[int]:
    """Sample video indices from a video.

    Args:
        total_frames: Total number of frames in the video.
        num_frames: Number of frames to sample.
        method: Frame sampling method.
            - uniform (should be the default method for training):
                randomly choose a stride and start index, then sample indices with the stride.
            - max_stride (should be the default method for inference):
                sample with the maximum stride (with interpolation) to cover most time range.
            - contiguous_random:
                randomly choose a start index, then sample indices with a single stride.
        rng: Random number generator (default: uses global random seed).

    Returns:
        List of frame indices (length = num_frames)
    """
    if rng is None:
        rng = random.Random(42)

    # 0. sanity check
    if total_frames <= 0 or num_frames <= 0:
        raise ValueError("total_frames and num_frames must be > 0")

    # 1. fewer frames: uniformly sample indices with rounded spacing
    if total_frames <= num_frames:
        indices = np.linspace(0, total_frames - 1, num_frames)
        return [int(round(x)) for x in indices]

    # 2. more frames: choose stride and start based on method
    if method == "random":
        # constraint: 1 + (num_frames - 1) * stride <= total_frames
        max_stride = (total_frames - 1) // (num_frames - 1)
        stride = rng.randrange(1, max_stride + 1)
        # constraint: start_index + (num_frames - 1) * stride < total_frames
        max_start = total_frames - (num_frames - 1) * stride - 1
        start = rng.randrange(0, max_start + 1)
        return [start + i * stride for i in range(num_frames)]
    elif method == "contiguous_random":
        stride = 1
        start = rng.randrange(0, total_frames - num_frames + 1)
        return [start + i * stride for i in range(num_frames)]
    elif method == "max_stride":
        indices = np.linspace(0, total_frames - 1, num_frames)
        return [int(round(x)) for x in indices]

    raise ValueError(f"unknown method: {method}")


def select_video_timestamps(
    start_time_s: float,
    end_time_s: float,
    num_frames: int,
) -> list[float]:
    """Sample uniformly-spaced timestamps in seconds within a time interval [start_time_s, end_time_s].

    Args:
        start_time_s: Interval start time in seconds.
        end_time_s: Interval end time in seconds.
        num_frames: Number of timestamps to sample within the interval.

    Returns:
        List of timestamps (length = num_frames).
    """
    if num_frames <= 0:
        raise ValueError(f"num_frames must be > 0, got {num_frames}")

    start_time_s = float(start_time_s)
    end_time_s = float(end_time_s)
    duration = end_time_s - start_time_s
    if duration <= 0:
        return []

    timestamps = np.linspace(start_time_s, end_time_s, num_frames).tolist()
    return timestamps


# ==================== Spatial Resizing ====================


def resize_video(
    video: jt.Real[torch.Tensor, "*b t h w c"],
    new_h: int,
    new_w: int,
) -> jt.Real[torch.Tensor, "*b t {new_h} {new_w} c"]:
    """Resize video tensor using bilinear interpolation.

    Args:
        video: Input video tensor.
        new_h: Target height.
        new_w: Target width.

    Returns:
        resized_video: Resized video tensor.
    """
    video = rearrange(video, "... t h w c -> ... t c h w")
    video = F.interpolate(video, size=(new_h, new_w), mode="bilinear", align_corners=False)
    resized_video = rearrange(video, "... t c h w -> ... t h w c")
    return resized_video


def center_crop_video(
    video: jt.Real[torch.Tensor, "*b t h w c"],
    crop_h: int,
    crop_w: int,
) -> jt.Real[torch.Tensor, "*b t {crop_h} {crop_w} c"]:
    """Apply center crop to video tensor.

    Args:
        video: Input video tensor.
        crop_h: Target crop height.
        crop_w: Target crop width.

    Returns:
        center_cropped_video: Center-cropped video tensor.

    Raises:
        ValueError: If crop dimensions are larger than video dimensions.
        ValueError: If crop dimensions are not positive.
    """
    h, w = video.shape[-3], video.shape[-2]
    if crop_h > h or crop_w > w:
        raise ValueError(f"Crop dimensions ({crop_h}, {crop_w}) cannot be larger than video dimensions ({h}, {w})")
    if crop_h <= 0 or crop_w <= 0:
        raise ValueError(f"Crop dimensions must be positive, got ({crop_h}, {crop_w})")

    top = max((h - crop_h) // 2, 0)
    left = max((w - crop_w) // 2, 0)
    center_cropped_video = video[..., top : top + crop_h, left : left + crop_w, :]
    return center_cropped_video


def random_crop_video(
    video: jt.Real[torch.Tensor, "*b t h w c"],
    crop_h: int,
    crop_w: int,
) -> jt.Real[torch.Tensor, "*b t {crop_h} {crop_w} c"]:
    """Apply random crop to video tensor.

    Args:
        video: Input video tensor.
        crop_h: Target crop height.
        crop_w: Target crop width.

    Returns:
        randomly_cropped_video: Randomly cropped video tensor.

    Raises:
        ValueError: If crop dimensions are larger than video dimensions.
        ValueError: If crop dimensions are not positive.
    """
    h, w = video.shape[-3], video.shape[-2]
    if crop_h > h or crop_w > w:
        raise ValueError(f"Crop dimensions ({crop_h}, {crop_w}) cannot be larger than video dimensions ({h}, {w})")
    if crop_h <= 0 or crop_w <= 0:
        raise ValueError(f"Crop dimensions must be positive, got ({crop_h}, {crop_w})")

    top = random.randint(0, max(0, h - crop_h))
    left = random.randint(0, max(0, w - crop_w))
    randomly_cropped_video = video[..., top : top + crop_h, left : left + crop_w, :]
    return randomly_cropped_video


def center_crop_resize_video(
    video: jt.Real[torch.Tensor, "*b t h w c"],
    new_h: int,
    new_w: int,
) -> jt.Real[torch.Tensor, "*b t {new_h} {new_w} c"]:
    """Apply center crop and resize method to video tensor.

    This method first resizes the video so that the short side matches the target resolution,
    then applies a center crop to get the exact target resolution.

    Args:
        video: Input video tensor.
        new_h: Target height.
        new_w: Target width.

    Returns:
        resized_and_center_cropped_video: Resized and center-cropped video tensor.
    """
    # 1. resize the short side to the target resolution
    h, w = video.shape[-3], video.shape[-2]
    scale = max(new_h / h, new_w / w)
    h = int(round(h * scale))
    w = int(round(w * scale))
    resized_video = resize_video(video, h, w)
    # 2. center crop the video to the target resolution
    resized_and_center_cropped_video = center_crop_video(resized_video, new_h, new_w)
    return resized_and_center_cropped_video


def randcrop_resize_video(
    video: jt.Real[torch.Tensor, "*b t h w c"],
    new_h: int,
    new_w: int,
) -> jt.Real[torch.Tensor, "*b t {new_h} {new_w} c"]:
    """Apply random crop and resize method to video tensor.

    This method first resizes the video so that the short side matches the target resolution,
    then applies a random crop to get the exact target resolution.

    Args:
        video: Input video tensor.
        new_h: Target height.
        new_w: Target width.

    Returns:
        resized_and_randomly_cropped_video: Resized and randomly cropped video tensor.
    """
    # 1. resize the short side to the target resolution
    h, w = video.shape[-3], video.shape[-2]
    scale = max(new_h / h, new_w / w)
    h = int(round(h * scale))
    w = int(round(w * scale))
    resized_video = resize_video(video, h, w)
    # 2. randomly crop the video to the target resolution
    resized_and_randomly_cropped_video = random_crop_video(resized_video, new_h, new_w)
    return resized_and_randomly_cropped_video


def pad_video(
    video: jt.Real[torch.Tensor, "*b t h w c"],
    pad_top: int,
    pad_bottom: int,
    pad_left: int,
    pad_right: int,
    value: float = -1.0,
) -> jt.Real[torch.Tensor, "*b t h+{pad_top}+{pad_bottom} w+{pad_left}+{pad_right} c"]:
    """Apply padding to video tensor.

    Args:
        video: Input video tensor.
        pad_top: Padding size on top.
        pad_bottom: Padding size on bottom.
        pad_left: Padding size on left.
        pad_right: Padding size on right.
        value: Padding value.

    Returns:
        padded_video: Padded video tensor.
    """
    video = rearrange(video, "... t h w c -> ... t c h w")
    video = F.pad(video, (pad_left, pad_right, pad_top, pad_bottom), mode="constant", value=value)
    padded_video = rearrange(video, "... t c h w -> ... t h w c")
    return padded_video


def pad_resize_video(
    video: jt.Real[torch.Tensor, "*b t h w c"],
    new_h: int,
    new_w: int,
    patch_h: int = 8,
    patch_w: int = 8,
    pad_value: float = -1.0,
) -> tuple[jt.Real[torch.Tensor, "*b t {new_h} {new_w} c"], jt.Bool[torch.Tensor, "h_num_patches w_num_patches"]]:
    """Apply padding-based resize method to get a resized video compatible with the patch size.

    For now only support square resolution (new_h == new_w = R).
    This method:
        1. Resizes the video so the long side equals R
        2. Ensures the short side is divisible by patch size (may require additional resize+crop)
        3. Pads the short side to R with patch-aligned padding
        4. Returns the resized and padded video tensor

    Args:
        video: Input video tensor.
        new_h: Target height.
        new_w: Target width.
        patch_h: Height of each patch (must divide new_h).
        patch_w: Width of each patch (must divide new_w).
        pad_value: Value used for padding (default: -1.0).

    Returns:
        resized_and_padded_video: Resized and padded video tensor.
        patch_mask: Patch mask of shape [new_h // patch_h, new_w // patch_w]
            - 1: valid content
            - 0: padding.

    Raises:
        NotImplementedError: If new_h != new_w. Only support square resolution for now.
        ValueError: If new_h or new_w is not divisible by patch_h or patch_w.
    """
    if new_h != new_w:
        raise NotImplementedError(f"Currently only support new_h == new_w, received new_h: {new_h}, new_w: {new_w}")
    if new_h % patch_h != 0 or new_w % patch_w != 0:
        raise ValueError(
            f"new_h: {new_h} and new_w: {new_w} must be divisible by patch_h: {patch_h} and patch_w: {patch_w}"
        )
    # 0. For clarity, set R = new_h = new_w (already checked before)
    R = new_h
    H, W, _ = video.shape[-3:]

    # 1: Determine the short side and compute initial scale
    scale = R / (H if H >= W else W)
    h = int(round(H * scale))
    w = int(round(W * scale))

    # 2: Check if the short side is divisible by the patch size
    short = w if H >= W else h
    patch_short = patch_w if H >= W else patch_h
    if short % patch_short == 0:
        # Short side is already divisible by patch size, just resize
        resized_video = resize_video(video, h, w)
    else:
        # Need to resize to make short side divisible by patch size, then crop
        short_new = int(np.ceil(short / patch_short) * patch_short)
        scale_new = short_new / short
        h = int(round(h * scale_new))
        w = int(round(w * scale_new))
        resized_video = resize_video(video, h, w)
        crop_h, crop_w = min(R, h), min(R, w)
        resized_video = center_crop_video(resized_video, crop_h, crop_w)
        h, w = crop_h, crop_w

    # 3: Pad the video to target resolution R with patch-aligned padding
    if H >= W:
        # Video is taller: pad width to R with w_patch alignment
        pad_total = R - w
        pad_left = (pad_total // 2 // patch_w) * patch_w
        pad_right = pad_total - pad_left
        pad_top = pad_bottom = 0
    else:
        # Video is wider: pad height to R with h_patch alignment
        pad_total = R - h
        pad_top = (pad_total // 2 // patch_h) * patch_h
        pad_bottom = pad_total - pad_top
        pad_left = pad_right = 0
    resized_and_padded_video = pad_video(resized_video, pad_top, pad_bottom, pad_left, pad_right, pad_value)

    # 4: Create patch mask indicating which patches contain valid content (1) vs padding (0)
    Hp = R // patch_h
    Wp = R // patch_w
    patch_mask = torch.zeros((Hp, Wp), dtype=torch.bool, device=video.device)

    top_p = pad_top // patch_h
    left_p = pad_left // patch_w
    h_p = h // patch_h
    w_p = w // patch_w
    patch_mask[top_p : top_p + h_p, left_p : left_p + w_p] = True
    return resized_and_padded_video, patch_mask


# ==================== Decoding ====================


def convert_video_output_format(
    video: jt.UInt[torch.Tensor, "*b t h w c"],
    output_range: OutputRange = "unit",
    dtype: str | torch.dtype = torch.float32,
) -> jt.Float[torch.Tensor, "*b t h w c"]:
    """Convert video tensor to the desired output format.

    Args:
        video: Input video tensor.
        output_range: Output range ("unit" or "symmetric").
            - unit: values in range [0.0, 1.0]
            - symmetric: values in range [-1.0, 1.0].
        dtype: Output dtype (default: torch.float32).

    Returns:
        converted_video: Converted video tensor.

    Raises:
        ValueError: If output_range is not recognized.
    """
    if output_range == "unit":
        converted_video = (video.to(torch.float32) / 255.0).to(dtype)
    elif output_range == "symmetric":
        converted_video = (video.to(torch.float32) / 127.5 - 1.0).to(dtype)
    else:
        raise ValueError(f"unknown output_range: {output_range}")
    return converted_video


def decode_frames_with_pyav(
    video_path: Path | str,
    num_frames: int,
    frame_sampling_method: FrameSamplingMethod = "max_stride",
    rng: random.Random | None = None,
) -> jt.UInt[torch.Tensor, "t h w c"]:
    """Decode video using PyAV.

    In case of exceptions, we do not raise, and instead return an empty tensor because the following code will handle
    the exception when receiving the empty video tensor.

    Args:
        video_path: Path to the video file.
        num_frames: Number of frames to sample.
        frame_sampling_method: Sampling method for frame indices.
        rng: Random number generator.

    Returns:
        decoded_video: Decoded video tensor in uint8 format.

    Raises:
        ImportError: If PyAV is not available.
    """
    import av  # noqa: PLC0415

    try:
        container = av.open(str(video_path))
        stream = container.streams.video[0]
        frames = [frame.to_ndarray(format="rgb24") for frame in container.decode(stream)]
        container.close()
    except Exception:
        return torch.empty((0, 0, 0, 0), dtype=torch.uint8)

    if len(frames) == 0:
        return torch.empty((0, 0, 0, 0), dtype=torch.uint8)

    indices = select_video_indices(len(frames), num_frames, method=frame_sampling_method, rng=rng)
    decoded_video = torch.from_numpy(np.stack([frames[i] for i in indices], axis=0))
    return decoded_video


def decode_frames_with_torchcodec(
    video_path: Path | str,
    num_frames: int,
    frame_sampling_method: FrameSamplingMethod = "max_stride",
    rng: random.Random | None = None,
    device: str | torch.device | None = None,
) -> jt.UInt[torch.Tensor, "t h w c"]:
    """Decode video using torchcodec.

    Args:
        video_path: Path to the video file.
        num_frames: Number of frames to sample.
        frame_sampling_method: Sampling method for frame indices.
        rng: Random number generator.
        device: Device to decode on (default: CPU).

    Returns:
        decoded_video: Decoded video tensor in uint8 format.

    Raises:
        ImportError: If torchcodec is not available.
    """
    from torchcodec.decoders import VideoDecoder, set_cuda_backend  # noqa: PLC0415

    if device is None:
        device = "cpu"
    device = torch.device(device)

    try:
        if device.type == "cuda":
            with set_cuda_backend("beta"):
                decoder = VideoDecoder(str(video_path), device=device, dimension_order="NHWC")
        else:
            decoder = VideoDecoder(str(video_path), device=device, dimension_order="NHWC")
        total_frames = int(decoder.metadata.num_frames or 0)
        indices = select_video_indices(total_frames, num_frames, method=frame_sampling_method, rng=rng)
        frame_batch = decoder.get_frames_at(indices)  # dimension_order="NHWC" returns (t, h, w, c) already
        video = frame_batch.data
    except Exception:
        return torch.empty((0, 0, 0, 0), dtype=torch.uint8)
    return video


def decode_frames_with_decord(
    video_path: Path | str,
    num_frames: int,
    frame_sampling_method: FrameSamplingMethod = "max_stride",
    rng: random.Random | None = None,
) -> jt.UInt[torch.Tensor, "t h w c"]:
    """Decode video using decord.

    Args:
        video_path: Path to the video file.
        num_frames: Number of frames to sample.
        frame_sampling_method: Sampling method for frame indices.
        rng: Random number generator.

    Returns:
        decoded_video: Decoded video tensor in uint8 format.
    """
    import decord  # noqa: PLC0415

    try:
        vr = decord.VideoReader(str(video_path), ctx=decord.cpu(0))
        total_frames = len(vr)
        if total_frames == 0:
            return torch.empty((0, 0, 0, 0), dtype=torch.uint8)
        indices = select_video_indices(total_frames, num_frames, method=frame_sampling_method, rng=rng)
        decoded_video = torch.from_numpy(vr.get_batch(indices).asnumpy())
    except Exception:
        return torch.empty((0, 0, 0, 0), dtype=torch.uint8)

    return decoded_video


def decode(
    video_path: Path | str,
    num_frames: int = 16,
    resolution: Sequence[int] = (288, 288),
    decode_method: DecodeMethod = "decord",
    resize_method: ResizeMethod = "center_crop_resize",
    frame_sampling_method: FrameSamplingMethod = "max_stride",
    output_range: OutputRange = "unit",
    dtype: str | torch.dtype = torch.float32,
    rng: random.Random | None = None,
    interval: Sequence[float] | None = None,
) -> tuple[jt.Real[torch.Tensor, "t h w c"], dict[str, Any]]:
    """Decode a video, optionally within a temporal interval.

    If ``interval`` is provided, frames are sampled uniformly in time from that interval. Otherwise, frame indices are
    sampled using ``frame_sampling_method``.

    Args:
        video_path: Path to the video file.
        num_frames: Number of frames to sample.
        resolution: Target resolution as (H, W) tuple.
        decode_method: Decoding method to use.
        resize_method: Resizing method to use.
        frame_sampling_method: Frame sampling method to use. If "interval" is provided, then the `interval` parameter
            must not be None.
        output_range: Output range to use.
        dtype: Output dtype to use.
        rng: Random number generator to use for frame index sampling.
        interval: Optional [start_time_s, end_time_s] to sample frames by time.

    Returns:
        video: Decoded video tensor.
        meta: Metadata dictionary.
            The metadata dictionary contains the following keys (if applicable):
                - "padding_mask": Patch mask for the padding resize method (Bool tensor).
                - "video_path": The path to the video file.
                - "interval": The interval of the video (if "interval" decoding method is used).
                - "interval_pts": The actual interval of the decoded video (if "interval" decoding method is used).

    Raises:
        ValueError: If decode_method is not recognized.
        ValueError: If resize_method is not recognized.
    """
    if rng is None:
        rng = random.Random(42)
    if frame_sampling_method == "interval" and interval is None:
        raise ValueError("frame_sampling_method='interval' requires a non-None interval")

    # case 1: interval-based decoding
    if interval is not None:
        if len(interval) != 2:
            raise ValueError(f"interval must have length 2, got {len(interval)}")
        start_time_s = float(interval[0])
        end_time_s = float(interval[1])
        if decode_method == "pyav":
            video, interval_pts = decode_interval_with_pyav(video_path, start_time_s, end_time_s, num_frames)
        elif decode_method == "torchcodec":
            video, interval_pts = decode_interval_with_torchcodec(video_path, start_time_s, end_time_s, num_frames)
        elif decode_method == "decord":
            video, interval_pts = decode_interval_with_decord(video_path, start_time_s, end_time_s, num_frames)
        else:
            raise ValueError(f"unknown decode method: {decode_method}")

    # case 2: frame-based decoding
    else:
        interval_pts = None
        if decode_method == "pyav":
            video = decode_frames_with_pyav(video_path, num_frames, frame_sampling_method, rng)
        elif decode_method == "torchcodec":
            video = decode_frames_with_torchcodec(video_path, num_frames, frame_sampling_method, rng)
        elif decode_method == "decord":
            video = decode_frames_with_decord(video_path, num_frames, frame_sampling_method, rng)
        else:
            raise ValueError(f"unknown decode method: {decode_method}")

    # return early if decoded video is empty
    if video.numel() == 0:
        return video, {}

    video = convert_video_output_format(video, output_range, dtype)

    meta: dict[str, Any] = {}
    if resize_method == "center_crop_resize":
        video = center_crop_resize_video(video, resolution[0], resolution[1])
    elif resize_method == "randcrop_resize":
        video = randcrop_resize_video(video, resolution[0], resolution[1])
    elif resize_method == "padding_resize":
        video, patch_mask = pad_resize_video(video, resolution[0], resolution[1])
        meta["padding_mask"] = patch_mask
    else:
        raise ValueError(f"unknown resize method: {resize_method}")

    # fill metadata
    # "interval" means the requested interval as a label
    # "interval_pts" means the actual interval of the decoded video
    meta["video_path"] = str(video_path)
    if interval is not None:
        meta["interval"] = [float(interval[0]), float(interval[1])]
        if interval_pts is not None:
            meta["interval_pts"] = interval_pts

    return video, meta


def decode_interval_with_pyav(
    video_path: Path | str,
    start_time_s: float,
    end_time_s: float,
    num_frames: int,
) -> tuple[jt.UInt[torch.Tensor, "t h w c"], tuple[float, float] | None]:
    """Decode a video clip using PyAV and sample frames by time.

    Some special cases of the return values:
    * If there are no frames in the requested interval, we return an empty tensor and None for the interval.
    * If errors occur, we return an empty tensor and None for the interval.
    * If there are less than num_frames frames in the interval, we repeat the last frame.
    """
    import av  # noqa: PLC0415

    if end_time_s <= start_time_s:
        return torch.empty((0, 0, 0, 0), dtype=torch.uint8), None

    container = None
    try:
        container = av.open(str(video_path))
        stream = container.streams.video[0]

        start_time_s = max(0.0, float(start_time_s))
        end_time_s = float(end_time_s)

        # stream.time_base is a Fraction object the corresponds to the unit of time the stream is expressed in
        # the units are (sec / tick), which is the OPPOSITE of how other time bases work
        if stream.time_base is not None:
            seek_offset = int(start_time_s / stream.time_base)  # the offset in units of ticks
            container.seek(seek_offset, stream=stream)
        else:
            # av.time_base is an integer hardcoded constant in (ticks / sec), so we multiply instead
            container.seek(int(start_time_s * av.time_base))  # the offset in units of ticks

        timestamps = select_video_timestamps(
            start_time_s,
            end_time_s,
            num_frames,
        )
        if len(timestamps) == 0:
            return torch.empty((0, 0, 0, 0), dtype=torch.uint8), None

        frames = []
        frame_ends = []
        frame_starts = []

        # loop through the frames in the stream
        for frame in container.decode(stream):
            if frame.time is None:
                continue
            frame_start = float(frame.time)
            assert frame.time_base is not None

            # frame.duration is in units of ticks, so we multiply by the time base to get the duration in seconds
            frame_end = frame_start + (float(frame.duration * frame.time_base) if frame.duration else 0.0)
            if frame_end <= start_time_s:  # skip frames before the start time
                continue
            if frame_start >= end_time_s:  # stop if we've reached the end time
                break
            frame_nd = frame.to_ndarray(format="rgb24")
            frames.append(frame_nd)
            frame_starts.append(frame_start)
            frame_ends.append(frame_end)
    except Exception:
        return torch.empty((0, 0, 0, 0), dtype=torch.uint8), None
    finally:
        if container is not None:
            container.close()

    if len(frames) == 0:
        return torch.empty((0, 0, 0, 0), dtype=torch.uint8), None

    # map each timestamp to the first decoded frame whose end-time is strictly greater than it
    frame_ends_arr = np.asarray(frame_ends, dtype=np.float64)
    if frame_ends_arr.size > 1 and not np.all(frame_ends_arr[1:] >= frame_ends_arr[:-1]):
        frame_ends_arr = np.maximum.accumulate(frame_ends_arr)
    ts_array = np.asarray(timestamps, dtype=frame_ends_arr.dtype)
    indices = np.searchsorted(frame_ends_arr, ts_array, side="right")
    indices = np.clip(indices, 0, len(frame_ends_arr) - 1).astype(np.int64)
    interval_pts = None
    if indices.size > 0:
        interval_pts = (float(frame_starts[indices[0]]), float(frame_starts[indices[-1]]))
    selected_frames = [frames[i] for i in indices.tolist()]
    return torch.from_numpy(np.stack(selected_frames, axis=0)), interval_pts


def decode_interval_with_torchcodec(
    video_path: Path | str,
    start_time_s: float,
    end_time_s: float,
    num_frames: int,
    device: torch.device | None = None,
) -> tuple[jt.UInt[torch.Tensor, "t h w c"], tuple[float, float] | None]:
    """Decode a video clip using torchcodec and sample frames from the decoded range."""
    from torchcodec.decoders import VideoDecoder, set_cuda_backend  # noqa: PLC0415

    if end_time_s <= start_time_s:
        return torch.empty((0, 0, 0, 0), dtype=torch.uint8), None

    if device is None:
        device = torch.device("cpu")

    try:
        if device.type == "cuda":
            with set_cuda_backend("beta"):
                decoder = VideoDecoder(str(video_path), device=device, dimension_order="NHWC")
        else:
            decoder = VideoDecoder(str(video_path), device=device, dimension_order="NHWC")

        begin_stream = float(decoder.metadata.begin_stream_seconds or 0.0)
        end_stream = decoder.metadata.end_stream_seconds

        start_s = max(float(start_time_s), begin_stream)
        end_s = float(end_time_s)
        if end_stream is not None:
            end_stream = float(end_stream)
            end_s = min(end_s, end_stream)

            # torchcodec rejects stop_seconds equal to end_stream_seconds due to strict checks
            # we nudge it slightly to the left to avoid this issue
            if not end_s < end_stream:
                end_s = end_stream - (1e-3 * max(1.0, abs(end_stream)))

        if end_s <= start_s:
            return torch.empty((0, 0, 0, 0), dtype=torch.uint8), None

        frame_batch = decoder.get_frames_played_in_range(start_s, end_s)
        video = frame_batch.data  # returned in (t, h, w, c) order bc of dimension_order="NHWC"
        pts_seconds = frame_batch.pts_seconds
        total_frames = video.shape[0]
        if total_frames == 0:
            return torch.empty((0, 0, 0, 0), dtype=torch.uint8), None

        # after getting all the frames, we select the largest stride to cover the range, including endpoints
        indices = select_video_indices(total_frames, num_frames, method="max_stride")
        video = video[indices]
    except Exception:
        return torch.empty((0, 0, 0, 0), dtype=torch.uint8), None

    interval_pts = None
    if pts_seconds is not None and int(pts_seconds.numel()) > 0:
        pts_selected = pts_seconds[indices]
        if int(pts_selected.numel()) > 0:
            interval_pts = (float(pts_selected[0].item()), float(pts_selected[-1].item()))
    return video, interval_pts


def decode_interval_with_decord(
    video_path: Path | str,
    start_time_s: float,
    end_time_s: float,
    num_frames: int,
) -> tuple[jt.UInt[torch.Tensor, "t h w c"], tuple[float, float] | None]:
    """Decode a video clip using decord and sample frames by time."""
    import decord  # noqa: PLC0415

    if end_time_s <= start_time_s:
        return torch.empty((0, 0, 0, 0), dtype=torch.uint8), None

    try:
        vr = decord.VideoReader(str(video_path), ctx=decord.cpu(0))
        total_frames = len(vr)
        if total_frames == 0:
            return torch.empty((0, 0, 0, 0), dtype=torch.uint8), None

        # get every single timestamp of every single frame in the video
        frame_ts = vr.get_frame_timestamp(np.arange(total_frames))
        start_times = frame_ts[:, 0]
        end_times = frame_ts[:, 1]
        end_stream = float(end_times[-1])

        start_s = max(float(start_time_s), float(start_times[0]))
        end_s = min(float(end_time_s), end_stream)
        if end_s <= start_s:
            return torch.empty((0, 0, 0, 0), dtype=torch.uint8), None

        timestamps = select_video_timestamps(
            start_s,
            end_s,
            num_frames,
        )
        if len(timestamps) == 0:
            return torch.empty((0, 0, 0, 0), dtype=torch.uint8), None

        # find the index of the frame that is closest to each of the requested timestamps
        ts_array = np.asarray(timestamps, dtype=start_times.dtype)
        indices = np.searchsorted(start_times, ts_array, side="right") - 1
        indices = np.clip(indices, 0, total_frames - 1)
        decoded_video = torch.from_numpy(vr.get_batch(indices.tolist()).asnumpy())
    except Exception:
        return torch.empty((0, 0, 0, 0), dtype=torch.uint8), None

    interval_pts = None
    if indices.size > 0:
        interval_pts = (float(start_times[indices[0]]), float(start_times[indices[-1]]))
    return decoded_video, interval_pts
