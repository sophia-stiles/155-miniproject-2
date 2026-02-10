from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import jaxtyping as jt
import numpy as np
from jax import tree_util
from jax.experimental import multihost_utils
from jax.sharding import Mesh, NamedSharding, PartitionSpec
from omegaconf import DictConfig
from orbax import checkpoint as ocp


def load_params(
    checkpoint_dir: str | Path,
    step: int | None = None,
) -> tuple[jt.PyTree[np.ndarray], int]:
    """Load only model params from a training checkpoint (for inference).

    This is a standalone utility that does not require a ``Mesh`` or
    distributed setup.  It opens the Orbax checkpoint directory, restores
    the requested (or latest) step, and returns ``params`` as a numpy
    pytree — ready to be passed to ``jax.device_put`` or
    ``put_replicated_tree`` by the caller.

    Args:
        checkpoint_dir: Checkpoint root directory
            (e.g. ``logs/project/run_name``).
        step: Specific step to load.  ``None`` loads the latest step.

    Returns:
        ``(params, step)`` where *params* is a pytree of ``np.ndarray``
        and *step* is the training step that was restored.

    Raises:
        FileNotFoundError: If the directory or checkpoint does not exist.
        KeyError: If the checkpoint has no ``params`` field.
    """
    ckpt_dir = Path(checkpoint_dir).expanduser().resolve()
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {ckpt_dir}")

    checkpointer = ocp.Checkpointer(ocp.PyTreeCheckpointHandler(use_ocdbt=True))
    mgr = ocp.CheckpointManager(
        str(ckpt_dir),
        checkpointer,
        options=ocp.CheckpointManagerOptions(create=False),
    )
    try:
        mgr.reload()
        resolved_step = int(step) if step is not None else mgr.latest_step()
        if resolved_step is None:
            raise FileNotFoundError(f"No checkpoint found under: {ckpt_dir}")
        # Restore without an abstract template — Orbax infers the structure.
        ckpt = mgr.restore(int(resolved_step), args=ocp.args.PyTreeRestore())
    finally:
        mgr.close()

    if "params" not in ckpt:
        raise KeyError(f"Checkpoint at step {resolved_step} has no 'params' field (keys: {list(ckpt.keys())})")

    # Convert all leaves to host numpy arrays.
    params = tree_util.tree_map(lambda x: np.asarray(jax.device_get(x)), ckpt["params"])
    atlas_logger.info("Loaded params from %s (step %d)", ckpt_dir, resolved_step)
    return params, int(resolved_step)


class CheckpointManager:
    """Process-0-only checkpoint manager for distributed JAX training.

    In multi-host setups without shared storage, standard Orbax multi-process
    checkpointing (OCDBT) requires all hosts to write to a shared filesystem.
    This class avoids that requirement: only process 0 creates the Orbax
    ``CheckpointManager`` and performs disk I/O.  All other processes
    participate only in barrier synchronization and receive restored state
    via ``broadcast_one_to_all``.
    """

    def __init__(
        self,
        cfg: DictConfig,
        mesh: Mesh,
        replicated_sharding: NamedSharding,
        num_local_devices: int,
        distributed_initialized: bool,
    ) -> None:
        """Initialize the checkpoint manager."""
        self._mesh = mesh
        self._replicated_sharding = replicated_sharding
        self._num_local_devices = num_local_devices
        self._distributed_initialized = distributed_initialized

        self._process_index = jax.process_index()
        self._is_main = self._process_index == 0

        ckpt_cfg = cfg.checkpoint
        self._save_every: int | None = ckpt_cfg.save_every_n_steps
        self._max_to_keep: int = ckpt_cfg.max_to_keep
        self._save_on_exit_cfg: bool = ckpt_cfg.save_on_exit
        self._resume_cfg: bool = ckpt_cfg.resume

        self._ckpt_dir = (Path(ckpt_cfg.dir) / cfg.manager.project / cfg.manager.name).resolve()

        # Only process 0 creates the Orbax manager (if needed).
        self._mgr: ocp.CheckpointManager | None = None
        need_mgr = self._save_every is not None or self._resume_cfg
        if self._is_main and need_mgr:
            self._ckpt_dir.mkdir(parents=True, exist_ok=True)
            checkpointer = ocp.Checkpointer(ocp.PyTreeCheckpointHandler(use_ocdbt=True))
            # In distributed mode, tell Orbax only process 0 participates in
            # checkpoint barriers.  Without this, Orbax expects ALL processes
            # to call its internal sync_global_devices.
            # Note: active_processes is incompatible with create=True, but we
            # already mkdir ourselves above so create=False is fine.
            if self._distributed_initialized:
                mp_opts = ocp.options.MultiprocessingOptions(
                    primary_host=0,
                    active_processes={0},
                )
                create = False
            else:
                mp_opts = ocp.options.MultiprocessingOptions(primary_host=0)
                create = True

            options = ocp.CheckpointManagerOptions(
                max_to_keep=self._max_to_keep,
                create=create,
                save_interval_steps=self._save_every or 1,
                multiprocessing_options=mp_opts,
            )
            self._mgr = ocp.CheckpointManager(str(self._ckpt_dir), checkpointer, options=options)
            atlas_logger.info(
                "Checkpoint: process-0-only save to %s (every %s steps, keep %d)",
                self._ckpt_dir,
                self._save_every,
                self._max_to_keep,
            )
        elif not need_mgr:
            atlas_logger.info("Checkpointing disabled (save_every_n_steps=None, resume=False)")

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def enabled(self) -> bool:
        """Whether periodic checkpoint saving is active."""
        return self._save_every is not None

    @property
    def should_resume(self) -> bool:
        """Whether resume-from-checkpoint is requested."""
        return self._resume_cfg

    @property
    def save_on_exit(self) -> bool:
        """Whether a final checkpoint should be saved on training exit."""
        return self._save_on_exit_cfg and self.enabled

    def should_save(self, step: int) -> bool:
        """Whether a periodic checkpoint should be saved at *step*."""
        if not self.enabled or self._save_every is None:
            return False
        return step > 0 and step % self._save_every == 0

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------

    def save(
        self,
        step: int,
        params: jt.PyTree[jax.Array],
        opt_state: jt.PyTree[jax.Array],
        rngs: jt.Shaped[jax.Array, "num_devices 2"],
        force: bool = False,
    ) -> None:
        """Save a checkpoint.  All processes must call this (barrier sync).

        Only process 0 writes to disk.  Other processes wait at barriers.
        """
        # Pre-save barrier: ensure all processes have finished the current step.
        self._sync("ckpt_save_pre", step)

        if self._is_main and self._mgr is not None:
            ckpt = self._make_ckpt_dict(step, params, opt_state, rngs)
            try:
                self._mgr.save(step, args=ocp.args.PyTreeSave(ckpt), force=force)
                if force:
                    # Block until async write completes (used for final checkpoint).
                    self._mgr.wait_until_finished()
                atlas_logger.info("Saved checkpoint at step %d", step)
            except Exception as e:
                atlas_logger.warning("Checkpoint save failed at step %d: %s", step, e)

        # Post-save barrier: all processes wait for process 0 to finish writing.
        self._sync("ckpt_save_post", step)

    # ------------------------------------------------------------------
    # Restore
    # ------------------------------------------------------------------

    def restore(
        self,
        params_ref: jt.PyTree[jax.Array],
        opt_state_ref: jt.PyTree[jax.Array],
    ) -> tuple[jt.PyTree[jax.Array], jt.PyTree[jax.Array], jax.Array, int] | None:
        """Restore the latest checkpoint.

        Process 0 loads from disk; all processes receive via broadcast.

        Args:
            params_ref: Reference parameter PyTree (used for shape/dtype inference
                on Orbax restore and as zero-filled placeholder on non-main hosts).
            opt_state_ref: Reference optimizer-state PyTree (same purpose).

        Returns:
            ``(params, opt_state, rngs, step)`` with params/opt_state replicated
            and rngs data-sharded, or ``None`` if no checkpoint was found.
        """
        self._sync("ckpt_restore_pre")

        # --- 1. Process 0 probes for an existing checkpoint ---------------
        has_ckpt = False
        latest_step = -1
        if self._is_main and self._mgr is not None:
            # Refresh metadata in case another run wrote checkpoints after init.
            self._mgr.reload()
            latest = self._mgr.latest_step()
            if latest is not None:
                has_ckpt = True
                latest_step = int(latest)

        # Broadcast [has_ckpt, step] from process 0 so all processes agree.
        meta = np.array([int(has_ckpt), latest_step], dtype=np.int32)
        meta = multihost_utils.broadcast_one_to_all(meta)
        if int(meta[0]) == 0:
            if self._is_main:
                atlas_logger.info("No checkpoint found for resume")
            self._sync("ckpt_restore_post")
            return None

        restored_step = int(meta[1])

        # --- 2. Build host-numpy pytrees on every process -----------------
        # All processes convert their reference state to numpy arrays of
        # identical shapes.  Process 0 will overwrite these with real data.
        params_host = self._to_host_numpy(params_ref)
        opt_state_host = self._to_host_numpy(opt_state_ref)

        if self._is_main and self._mgr is not None:
            # Use numpy reference trees as the Orbax "abstract" template so
            # Orbax knows the expected dtypes / shapes for deserialization.
            abstract = {
                "step": 0,
                "params": params_host,
                "opt_state": opt_state_host,
                "rngs": np.zeros((2,), dtype=np.uint32),
            }
            ckpt = self._mgr.restore(restored_step, args=ocp.args.PyTreeRestore(item=abstract))
            params_host = jax.device_get(ckpt["params"])
            opt_state_host = jax.device_get(ckpt["opt_state"])
            rng_key = np.asarray(jax.device_get(ckpt["rngs"]), dtype=np.uint32).ravel()
        else:
            # Non-main processes: placeholder (overwritten by broadcast below).
            rng_key = np.zeros((2,), dtype=np.uint32)

        # --- 3. Broadcast from process 0 to all --------------------------
        # broadcast_one_to_all sends process 0's pytree leaves to every host.
        params_host = multihost_utils.broadcast_one_to_all(params_host)
        opt_state_host = multihost_utils.broadcast_one_to_all(opt_state_host)
        rng_key = multihost_utils.broadcast_one_to_all(rng_key)

        # --- 4. Re-shard on each process ----------------------------------
        # Convert numpy leaves back to jax arrays (put_replicated_tree
        # expects jax.Array leaves due to jaxtyping annotation).
        params_jax = tree_util.tree_map(jnp.asarray, params_host)
        opt_state_jax = tree_util.tree_map(jnp.asarray, opt_state_host)

        # Place params & opt_state as fully-replicated global arrays.
        params = put_replicated_tree(params_jax, self._replicated_sharding)
        opt_state = put_replicated_tree(opt_state_jax, self._replicated_sharding)

        # Reconstruct per-device RNG keys from the single saved key (shape must be (2,)).
        rng_key = jnp.asarray(rng_key).ravel()
        rngs_local = jax.random.split(rng_key, self._num_local_devices)
        rngs = host_local_to_global_array(rngs_local, self._mesh, PartitionSpec("data"))

        atlas_logger.info(
            "Restored checkpoint at step %d (process %d)",
            restored_step,
            self._process_index,
        )
        self._sync("ckpt_restore_post")
        return params, opt_state, rngs, restored_step

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Release Orbax resources.  Safe to call from any process."""
        if self._mgr is not None:
            try:
                self._mgr.close()
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _sync(self, tag: str, step: int | None = None) -> None:
        """Barrier-sync all hosts when distributed mode is active."""
        label = f"{tag}_{step}" if step is not None else tag
        if self._distributed_initialized:
            multihost_utils.sync_global_devices(label)

    @staticmethod
    def _to_host_numpy(tree: jt.PyTree[Any]) -> jt.PyTree[np.ndarray]:
        """Convert a (possibly sharded) JAX pytree to host-local numpy arrays.

        For replicated arrays we extract shard 0 (all shards are identical).
        """

        def _leaf(x: Any) -> Any:
            if isinstance(x, jax.Array):
                # Replicated arrays: every shard holds the same data, take shard 0.
                if hasattr(x, "addressable_shards") and x.addressable_shards:
                    return np.asarray(jax.device_get(x.addressable_shards[0].data))
                return np.asarray(jax.device_get(x))
            if isinstance(x, jnp.ndarray):
                return np.asarray(x)
            return x

        return tree_util.tree_map(_leaf, tree)

    def _make_ckpt_dict(
        self,
        step: int,
        params: jt.PyTree[jax.Array],
        opt_state: jt.PyTree[jax.Array],
        rngs: jt.Shaped[jax.Array, "num_devices 2"],
    ) -> dict[str, Any]:
        """Build the serializable checkpoint dict on process 0.

        Params and opt_state are replicated so one shard suffices.
        RNG: save a single key (shard 0) for backward compatibility;
        on restore we ``jax.random.split`` it back to per-device keys.
        """
        params_host = self._to_host_numpy(params)
        opt_state_host = self._to_host_numpy(opt_state)
        # Take the first device's RNG key (all derived from the same global key).
        # Shard data may have shape (1, 2) due to data-sharding; flatten to (2,).
        rng_key = np.asarray(jax.device_get(rngs.addressable_shards[0].data)).ravel()
        return {
            "step": int(step),
            "params": params_host,
            "opt_state": opt_state_host,
            "rngs": rng_key,
        }
