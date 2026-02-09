import jax
import jax.numpy as jnp
import jaxtyping as jt
import numpy as np
from flax import struct
from gemma import gm, peft
from gemma.gm.ckpts import CheckpointPath
from videoprism import models as vp

from model.adapter import BaseAdapter
from model.base import AbstractAlignmentModel, Tokenized


@struct.dataclass
class Gemma3TokenizedQuery(Tokenized):
    """A Gemma 3 tokenized query."""

    token_ids: jt.Int[jax.Array, "*B L"]
    paddings: jt.Int[jax.Array, "*B L"]


GEMMA_CHECKPOINT_PATHS = {
    "Gemma3_270M": CheckpointPath.GEMMA3_270M_IT,
    "Gemma3_1B": CheckpointPath.GEMMA3_1B_IT,
    "Gemma3_4B": CheckpointPath.GEMMA3_4B_IT,
    "Gemma3_12B": CheckpointPath.GEMMA3_12B_IT,
    "Gemma3_27B": CheckpointPath.GEMMA3_27B_IT,
    "Gemma3n_E2B": CheckpointPath.GEMMA3N_E2B_IT,
    "Gemma3n_E4B": CheckpointPath.GEMMA3N_E4B_IT,
}


MODEL_CLASSES = {
    "Gemma3_270M": gm.nn.Gemma3_270M,
    "Gemma3_1B": gm.nn.Gemma3_1B,
    "Gemma3_4B": gm.nn.Gemma3_4B,
    "Gemma3_12B": gm.nn.Gemma3_12B,
    "Gemma3_27B": gm.nn.Gemma3_27B,
    "Gemma3n_E2B": gm.nn.Gemma3n_E2B,
    "Gemma3n_E4B": gm.nn.Gemma3n_E4B,
}


class Gemma3AlignmentModel(AbstractAlignmentModel[Gemma3TokenizedQuery]):
    """Alignment model with VideoPrism video encoder and JAX Gemma 3 text encoder."""

    def __init__(
        self,
        model_name: str = "videoprism_lvt_public_v1_base",
        gemma_model_class: str = "Gemma3_1B",
        gemma_text_only: bool = True,
        lora_rank: int | None = None,
        lora_verbose: bool = False,
        use_bfloat16: bool = True,
        max_length: int = 64,
        add_bos: bool = True,
        text_adapter: BaseAdapter | None = None,
        video_adapter: BaseAdapter | None = None,
    ) -> None:
        """Initialize the Gemma3 + VideoPrism alignment model.

        Args:
            model_name: VideoPrism model name for video embeddings.
            gemma_model_class: Gemma model class name under `gm.nn` (e.g. `Gemma3_1B`).
                - Gemma3_270M
                - Gemma3_1B
                - Gemma3_4B
                - Gemma3_12B
                - Gemma3_27B
                - Gemma3n_E2B
                - Gemma3n_E4B.
            gemma_checkpoint: Checkpoint enum name under `gm.ckpts.CheckpointPath` or an explicit checkpoint path.
                - GEMMA3_270M_IT
                - GEMMA3_1B_IT
                - GEMMA3_4B_IT
                - GEMMA3_12B_IT
                - GEMMA3_27B_IT
                - GEMMA3n_E2B_IT
                - GEMMA3n_E4B_IT.
            gemma_text_only: Whether to instantiate Gemma in text-only mode.
            lora_rank: Optional LoRA rank. If set (>0), wraps Gemma with `gm.nn.LoRA`.
            lora_verbose: Whether to print LoRA replacement diagnostics.
            use_bfloat16: Whether to use bfloat16 forward pass.
            max_length: Maximum token sequence length.
            add_bos: Whether to add BOS token. `None` defaults to `True` for Gemma.
            text_adapter: Optional text adapter.
            video_adapter: Optional video adapter.
        """
        super().__init__(text_adapter=text_adapter, video_adapter=video_adapter)

        # Video branch (VideoPrism).
        fprop_dtype = "bfloat16" if use_bfloat16 else None
        self._dtype = jnp.bfloat16 if use_bfloat16 else jnp.float32
        self._flax_model = vp.get_model(model_name, fprop_dtype=fprop_dtype)
        self._vp_params = vp.load_pretrained_weights(model_name)

        # Text branch (Gemma 3, JAX).
        self._max_length = max_length
        self._add_bos = add_bos

        # Get model
        gemma_model_ctor = MODEL_CLASSES[gemma_model_class]
        try:
            base_model = gemma_model_ctor(text_only=gemma_text_only, dtype=self._dtype)
        except TypeError:
            base_model = gemma_model_ctor(dtype=self._dtype)

        # Wrap with LoRA if requested
        if lora_rank is not None and lora_rank > 0:
            self._gemma_model = gm.nn.LoRA(
                rank=int(lora_rank),
                model=base_model,
                dtype=self._dtype,
                verbose=lora_verbose,
            )
        else:
            self._gemma_model = base_model

        # Get tokenizer
        self._gemma_tokenizer = gm.text.Gemma3Tokenizer()
        self._pad_id = int(self._gemma_tokenizer.special_tokens.PAD)

        # Initialize params with static token shape for JAX compilation consistency.
        init_rng = jax.random.PRNGKey(42)
        dummy_tokens = jnp.zeros((1, self._max_length), dtype=jnp.int32)
        gemma_params = self._gemma_model.init(
            init_rng,
            tokens=dummy_tokens,
            return_last_only=False,
            return_hidden_states=True,
        )["params"]

        # Load checkpoint
        checkpoint_path = GEMMA_CHECKPOINT_PATHS[gemma_model_class]
        if lora_rank is not None and lora_rank > 0:
            base_params, lora_params = peft.split_params(gemma_params)  # type: ignore[arg-type]
            restored_base = gm.ckpts.load_params(
                checkpoint_path,
                params=base_params,
                text_only=gemma_text_only,
            )
            self._gemma_params = peft.merge_params(restored_base, lora_params)  # type: ignore[arg-type]
        else:
            self._gemma_params = gm.ckpts.load_params(
                checkpoint_path,
                params=gemma_params,
                text_only=gemma_text_only,
            )

        self._gemma_hidden_size = int(self._gemma_model.config.embed_dim)
        self._full_params: jt.PyTree[jax.Array] = self.adapter_params | {
            "videoprism": self._vp_params,
            "gemma": self._gemma_params,
        }

    @property
    def params(self) -> jt.PyTree[jax.Array]:
        """Get model parameters."""
        return self._full_params

    def tokenize(self, texts: list[str]) -> Gemma3TokenizedQuery:
        """Tokenize texts using Gemma 3 tokenizer."""
        token_rows: list[list[int]] = []
        mask_rows: list[list[int]] = []

        for text in texts:
            token_ids = self._gemma_tokenizer.encode(text, add_bos=self._add_bos, add_eos=True)
            token_ids = token_ids[: self._max_length]
            valid = len(token_ids)

            if valid < self._max_length:
                token_ids = token_ids + [self._pad_id] * (self._max_length - valid)

            mask = [1] * valid + [0] * (self._max_length - valid)
            token_rows.append(token_ids)
            mask_rows.append(mask)

        token_ids_arr = jnp.asarray(np.asarray(token_rows, dtype=np.int32), dtype=jnp.int32)
        paddings_arr = jnp.asarray(np.asarray(mask_rows, dtype=np.int32), dtype=jnp.int32)
        return Gemma3TokenizedQuery(token_ids=token_ids_arr, paddings=paddings_arr)

    def get_text_embeddings(
        self,
        params: jt.PyTree[jax.Array],
        tokenized_query: Gemma3TokenizedQuery,
        rng: jt.Float[jax.Array, "2"] | None = None,
        train: bool = False,
    ) -> jt.Float[jax.Array, "B D"]:
        """Get text embeddings from Gemma hidden states."""
        del rng, train

        hidden_states: jt.Float[jax.Array, "B L D"] = self._gemma_model.apply(
            {"params": params["gemma"]},
            tokens=tokenized_query.token_ids,
            return_last_only=False,
            return_hidden_states=True,
        ).hidden_states  # type: ignore[attr-defined]

        valid_mask = tokenized_query.paddings.astype(jnp.int32)
        lengths = jnp.maximum(valid_mask.sum(axis=1), 1)
        last_idx = lengths - 1
        gather_idx = jnp.broadcast_to(last_idx[:, None, None], (hidden_states.shape[0], 1, hidden_states.shape[-1]))
        last_hidden = jnp.take_along_axis(hidden_states, gather_idx, axis=1)
        return jnp.squeeze(last_hidden, axis=1).astype(jnp.float32)

    def get_video_embeddings(
        self,
        params: jt.PyTree[jax.Array],
        video: jt.Float[jax.Array, "B T H W C"],
        rng: jt.Float[jax.Array, "2"] | None = None,
        train: bool = False,
    ) -> jt.Float[jax.Array, "B N D"]:
        """Get video embeddings from VideoPrism."""
        rngs = {"dropout": rng} if rng is not None else None

        result = self._flax_model.apply(
            params["videoprism"],
            video,
            None,
            None,
            train=train,
            rngs=rngs,
        )
        return result[0]  # type: ignore
