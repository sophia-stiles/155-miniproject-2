import random
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal, Protocol, runtime_checkable

import jaxtyping as jt
import torch
from torch.utils.data import Dataset

from data.utils import decode

DecodeMethod = Literal["pyav", "torchcodec", "decord"]
ResizeMethod = Literal["center_crop_resize", "randcrop_resize", "padding_resize"]
FrameSamplingMethod = Literal["random", "contiguous_random", "max_stride", "interval"]
OutputRange = Literal["unit", "symmetric"]


@runtime_checkable
class VideoDataSample(Protocol):
    """A minimum video data sample should contain a video tensor and a metadata dictionary.

    This is a protocol used to enforce API-compatibility for all video data samples.

    Fields:
        video: The video tensor.
        meta: The metadata dictionary.
    """

    video: jt.Float[torch.Tensor, "*b t h w c"]
    meta: dict[str, Any]


@dataclass(slots=True, kw_only=True)
class VideoTextDataSample:
    """A single video-text sample returned by VideoTextDataset and subclasses.

    Fields:
        video: Decoded video tensor.
        caption: Text caption for the video.
        meta: Decode metadata (e.g. indices, paths).
    """

    video: jt.Float[torch.Tensor, "*b t h w c"]
    caption: str
    meta: dict[str, Any]


class GeneralVideoDataset(Dataset[Any], ABC):
    """Abstract base for video datasets with lazy manifest loading.

    Subclasses must implement __getitem__ and _get_manifest_data.
    Manifest data is loaded on first access to manifest_data, so subclasses
    can set instance attributes (e.g. root, csv_path) in __init__ before
    it runs. Supports configurable decode/resize/frame sampling, output range/dtype,
    and optional max_num_videos cap.
    """

    def __init__(
        self,
        num_frames: int = 16,
        resolution: Sequence[int] = (288, 288),
        decode_method: DecodeMethod = "decord",
        resize_method: ResizeMethod = "center_crop_resize",
        frame_sampling_method: FrameSamplingMethod = "max_stride",
        output_range: OutputRange = "unit",
        dtype: str | torch.dtype = torch.float32,
        seed: int = 42,
        max_num_videos: int | None = None,
    ) -> None:
        """Initialize the GeneralVideoDataset.

        Args:
            num_frames: The number of frames to sample from the video.
            resolution: The resolution of the videos.
            decode_method: The method to decode the videos.
            resize_method: The method to resize the videos.
            frame_sampling_method: The method to sample the frames from the video.
            output_range: The range of the output videos.
            dtype: The dtype of the output videos.
            seed: The seed for the random number generator.
            max_num_videos: The maximum number of videos to use from the dataset.
        """
        self.num_frames = num_frames
        self.resolution = resolution
        self.decode_method: DecodeMethod = decode_method
        self.resize_method: ResizeMethod = resize_method
        self.frame_sampling_method: FrameSamplingMethod = frame_sampling_method
        self.output_range: OutputRange = output_range
        self.dtype = getattr(torch, dtype) if isinstance(dtype, str) else dtype
        self.seed = seed
        self.rng = random.Random(seed)
        self.max_num_videos = max_num_videos

        # Manifest data will be loaded lazily when first accessed
        # This allows subclasses to set their own variables before _get_manifest_data() is called
        self._manifest_data: list[Any] | None = None

    @property
    def manifest_data(self) -> list[Any]:
        """Lazy-load manifest data. Loads on first access to allow subclasses to set their variables first.

        Returns:
            manifest_data: A list of the information required to construct the tuples of information in the dataset.

            For example, this could be a list of tuples of (video_path, text_caption), or just a list of video paths
            for video-only datasets.
        """
        if self._manifest_data is None:
            self._manifest_data = self._get_manifest_data()
            if self.max_num_videos is not None and len(self._manifest_data) > self.max_num_videos:
                self._manifest_data = self._manifest_data[: self.max_num_videos]
        return self._manifest_data

    @abstractmethod
    def __getitem__(self, idx: int) -> VideoDataSample:
        """Get the video data sample at the given index."""

    @abstractmethod
    def _get_manifest_data(self) -> list[Any]:
        """Load and return the manifest data for the dataset.

        This method is called lazily when manifest_data is first accessed,
        allowing subclasses to set their own instance variables (e.g., root, csv_path)
        in __init__ before this method is called.

        Returns:
            List of manifest entries. The format depends on the dataset type:
            - For GeneralVideoDataset: list of (video_path, caption) tuples
            - For other datasets: format defined by the subclass
        """

    def __len__(self) -> int:
        """Return the number of videos in the dataset."""
        return len(self.manifest_data)

    def __repr__(self) -> str:
        """Return a compact multi-line representation of the dataset."""
        name = self.__class__.__name__
        n = len(self.manifest_data)
        h, w = self.resolution[0], self.resolution[1]
        dtype_str = str(self.dtype).replace("torch.", "")
        return (
            f"{name}(\n"
            f"  n_videos={n:,}, num_frames={self.num_frames}, resolution={h}x{w},\n"
            f"  decode={self.decode_method!r}, resize={self.resize_method!r}, "
            f"  sampling={self.frame_sampling_method!r},\n"
            f"  output_range={self.output_range!r}, dtype={dtype_str!r}, seed={self.seed}\n"
            ")"
        )


class VideoTextDataset(GeneralVideoDataset):
    """Base for video-text datasets: manifest entries are (video_path, caption).

    Implements __getitem__ by reading (video_path, caption) from manifest_data, decoding the video with decode,
    and returning a VideoTextDataSample.
    Subclasses only need to implement _get_manifest_data to provide the list of (video_path, caption) pairs.
    """

    def __getitem__(self, idx: int) -> VideoTextDataSample:
        """Get the video and caption at the given index."""
        video_path, caption = self.manifest_data[idx]
        video, meta = decode(
            str(video_path),
            self.num_frames,
            self.resolution,
            decode_method=self.decode_method,
            resize_method=self.resize_method,
            frame_sampling_method=self.frame_sampling_method,
            output_range=self.output_range,
            dtype=self.dtype,
            rng=self.rng,
        )
        return VideoTextDataSample(video=video, caption=caption, meta=meta)
