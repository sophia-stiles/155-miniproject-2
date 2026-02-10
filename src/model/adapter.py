from abc import ABC, abstractmethod
from typing import Any, Optional

import jax
import jax.numpy as jnp
import jaxtyping as jt
from flax import linen as nn


class BaseAdapter(nn.Module, ABC):
    """Base class for all adapters."""

    input_dim: int
    output_dim: int

    @abstractmethod
    def __call__(self, x: Any) -> Any:
        """Call the adapter."""
        raise NotImplementedError


class AttentionPoolingAdapter(BaseAdapter):
    """Pool token features with learned attention, then project to output_dim.

    Input shape: [B, N, D_in]
    Output shape: [B, D_out]
    """

    @nn.compact
    def __call__(self, x: jt.Float[jax.Array, "B N D"]) -> jt.Float[jax.Array, "B D"]:
        """Call the attention pooling adapter."""
        if x.ndim != 3:
            raise ValueError(f"AttentionPoolingAdapter expects 3D input [B, N, D], got {x.shape}")
        if x.shape[-1] != self.input_dim:
            raise ValueError(f"Last dim mismatch: expected {self.input_dim}, got {x.shape[-1]}")

        x = x.astype(jnp.float32)

        query_bias = self.param(
            "query_bias",
            nn.initializers.normal(stddev=0.02),
            (1, 1, self.input_dim),
        )
        attn_logits = nn.Dense(
            features=1,
            use_bias=True,
            kernel_init=nn.initializers.xavier_uniform(),
            name="attn_proj",
        )(x + query_bias).squeeze(-1)
        attn_weights = jax.nn.softmax(attn_logits, axis=1)

        pooled = jnp.einsum("bn,bnd->bd", attn_weights, x)
        return nn.Dense(
            features=self.output_dim,
            use_bias=True,
            kernel_init=nn.initializers.xavier_uniform(),
            name="out_proj",
        )(pooled)


class MLPAdapter(BaseAdapter):
    """Two-layer MLP adapter.

    Input shape: [..., D_in]
    Output shape: [..., D_out]
    """

    hidden_dim: Optional[int] = None
    activation: str = "gelu"

    def _act(self, x: jax.Array) -> jax.Array:
        if self.activation == "relu":
            return nn.relu(x)
        if self.activation == "silu":
            return nn.silu(x)
        return nn.gelu(x)

    @nn.compact
    def __call__(self, x: jt.Float[jax.Array, "B D"]) -> jt.Float[jax.Array, "B D"]:
        """Call the MLP adapter."""
        if x.shape[-1] != self.input_dim:
            raise ValueError(f"Last dim mismatch: expected {self.input_dim}, got {x.shape[-1]}")

        x = x.astype(jnp.float32)
        hidden_dim = self.hidden_dim or max(self.input_dim, self.output_dim)

        h = nn.Dense(
            features=hidden_dim,
            use_bias=True,
            kernel_init=nn.initializers.xavier_uniform(),
            name="fc1",
        )(x)
        h = self._act(h)
        return nn.Dense(
            features=self.output_dim,
            use_bias=True,
            kernel_init=nn.initializers.xavier_uniform(),
            name="fc2",
        )(h)


def make_dummy_input(
    adapter: BaseAdapter,
    num_tokens: int = 8,
    batch_size: int = 2,
    dtype: jnp.dtype = jnp.float32,
) -> jax.Array:
    """Make a dummy input for the adapter."""
    if isinstance(adapter, AttentionPoolingAdapter):
        return jnp.zeros((batch_size, num_tokens, adapter.input_dim), dtype=dtype)
    return jnp.zeros((batch_size, adapter.input_dim), dtype=dtype)
