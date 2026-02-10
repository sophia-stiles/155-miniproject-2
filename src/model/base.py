from abc import ABC, abstractmethod
from typing import Generic, TypeVar

import jax
import jaxtyping as jt
from flax import struct

from model.adapter import BaseAdapter, make_dummy_input


@struct.dataclass
class Tokenized:
    """A standardized tokenized input."""


TokenizedT = TypeVar("TokenizedT", bound=Tokenized)


class AbstractAlignmentModel(ABC, Generic[TokenizedT]):
    """An abstract alignment model with Hydra-friendly adapter init helpers."""

    def __init__(
        self,
        text_adapter: BaseAdapter | None = None,
        video_adapter: BaseAdapter | None = None,
        adapter_init_rng: jax.Array | None = None,
    ) -> None:
        """Initialize the alignment model."""
        self.text_adapter = text_adapter
        self.video_adapter = video_adapter
        self._adapter_init_rng = adapter_init_rng if adapter_init_rng is not None else jax.random.PRNGKey(0)

        adapter_params = {}
        rng_text, rng_video = jax.random.split(self._adapter_init_rng, 2)

        if self.text_adapter is not None:
            dummy_text_input = make_dummy_input(self.text_adapter)
            text_adapter_params = self.text_adapter.init(rng_text, dummy_text_input)["params"]
            adapter_params["text_adapter"] = text_adapter_params

        if self.video_adapter is not None:
            dummy_video_input = make_dummy_input(self.video_adapter)
            video_adapter_params = self.video_adapter.init(rng_video, dummy_video_input)["params"]
            adapter_params["video_adapter"] = video_adapter_params

        self.adapter_params: jt.PyTree[jax.Array] = adapter_params

    @property
    def params(self) -> jt.PyTree[jax.Array]:
        """Get the parameters for the model."""
        return self.adapter_params

    def get_video_adapter_params(self, params: jt.PyTree[jax.Array]) -> jt.PyTree[jax.Array]:
        """Get the parameters for the video adapter."""
        return params["video_adapter"]

    def get_text_adapter_params(self, params: jt.PyTree[jax.Array]) -> jt.PyTree[jax.Array]:
        """Get the parameters for the text adapter."""
        return params["text_adapter"]

    @abstractmethod
    def tokenize(self, texts: list[str]) -> TokenizedT:
        """Tokenize a list of texts."""

    @abstractmethod
    def get_text_embeddings(
        self,
        params: jt.PyTree[jax.Array],
        tokenized_query: TokenizedT,
        rng: jt.Float[jax.Array, "2"] | None = None,
        train: bool = False,
    ) -> jt.Float[jax.Array, "B D"]:
        """Get text embeddings."""

    def get_adapted_text_embeddings(
        self,
        params: jt.PyTree[jax.Array],
        tokenized_query: TokenizedT,
        rng: jt.Float[jax.Array, "2"] | None = None,
        train: bool = False,
    ) -> jt.Float[jax.Array, "B D"]:
        """Get adapted text embeddings."""
        text_embeddings = self.get_text_embeddings(params, tokenized_query, rng, train)
        if self.text_adapter is None:
            return text_embeddings
        text_embeddings = self.text_adapter.apply(
            {"params": self.get_text_adapter_params(params)},
            text_embeddings,
        )
        return text_embeddings  # type: ignore

    @abstractmethod
    def get_video_embeddings(
        self,
        params: jt.PyTree[jax.Array],
        video: jt.Float[jax.Array, "B T H W C"],
        rng: jt.Float[jax.Array, "2"] | None = None,
        train: bool = False,
    ) -> jt.Float[jax.Array, "B N D"]:
        """Get video embeddings."""

    def get_adapted_video_embeddings(
        self,
        params: jt.PyTree[jax.Array],
        video: jt.Float[jax.Array, "B T H W C"],
        rng: jt.Float[jax.Array, "2"] | None = None,
        train: bool = False,
    ) -> jt.Float[jax.Array, "B D"]:
        """Get adapted video embeddings."""
        video_embeddings = self.get_video_embeddings(params, video, rng, train)
        if self.video_adapter is None:
            return video_embeddings
        video_embeddings = self.video_adapter.apply(
            {"params": self.get_video_adapter_params(params)},
            video_embeddings,
        )
        return video_embeddings  # type: ignore
