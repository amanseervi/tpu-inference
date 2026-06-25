"""Optimized Gated MLP (GeGLU) Pallas kernel for TPU.

Fuses the gate/up projections, the GELU activation, and the down projection
into a single pipelined TPU kernel, completely eliminating intermediate HBM traffic.
"""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.experimental.shard_map import shard_map
from jax.sharding import PartitionSpec as P


def inner_mlp_kernel(
    x_tile,
    w_gu_tile,
    wd_tile,
    y_tile,
    y_scratch,
    *,
    b_seq: int,
    b_inter: int,
    hidden_size: int,
    num_inter: int,
):
    i_i = pl.program_id(0)

    def _compute(is_first: bool, is_last: bool):
        # 1. Fetch & Upcast weights
        w_gu_sram = w_gu_tile[...].astype(x_tile.dtype)

        # 2. Matmul 1 (x @ W_in)
        hu_sram = jnp.matmul(x_tile[...], w_gu_sram, preferred_element_type=jnp.float32)

        # Split and Activate (GELU gated activation)
        h_sram = hu_sram[:, :b_inter]
        u_sram = hu_sram[:, b_inter:]
        a_tile = jax.nn.gelu(h_sram, approximate=True) * u_sram
        a_tile = a_tile.astype(x_tile.dtype)

        # 3. Matmul 2 (a @ W_out)
        wd_sram = wd_tile[...].astype(x_tile.dtype)
        y_current_sram = jnp.matmul(a_tile, wd_sram, preferred_element_type=jnp.float32)

        # 4. Accumulate
        acc = y_current_sram if is_first else y_scratch[...] + y_current_sram

        if is_last:
            y_tile[...] = acc.astype(y_tile.dtype)
        else:
            y_scratch[...] = acc

    # Define matmul wrapper scopes for XLA compiler profiling
    @jax.named_scope("compute_first_last")
    def compute_first_last():
        _compute(True, True)

    @jax.named_scope("compute_first")
    def compute_first():
        _compute(True, False)

    @jax.named_scope("compute")
    def compute():
        _compute(False, False)

    @jax.named_scope("compute_last")
    def compute_last():
        _compute(False, True)

    is_first = i_i == 0
    is_last = i_i == (num_inter - 1)

    # Explicit control flow eliminates @pl.when overhead
    jax.lax.cond(
        is_first,
        lambda: jax.lax.cond(is_last, compute_first_last, compute_first),
        lambda: jax.lax.cond(is_last, compute_last, compute),
    )


def mlp_kernel_main(x_hbm, w_gu_hbm, wd_hbm, y_hbm, y_scratch, *, b_seq, b_inter, hidden_size):
    """Entry point for Pallas grid. Wires up HBM references to the pipeline."""
    seq_idx = pl.program_id(0)
    num_inter = w_gu_hbm.shape[1] // (2 * b_inter)

    # 1. Block specs mapping the inner pipeline loop (i_i) to HBM arrays
    x_spec = pl.BlockSpec((b_seq, hidden_size), lambda i_i: (seq_idx, 0))
    y_spec = pl.BlockSpec((b_seq, hidden_size), lambda i_i: (seq_idx, 0))

    # 2. Triple buffering for weights to hide HBM latency
    w_gu_spec = pl.BlockSpec(
        (hidden_size, 2 * b_inter),
        lambda i_i: (0, i_i),
        pipeline_mode=pl.Buffered(buffer_count=3),
    )
    wd_spec = pl.BlockSpec(
        (b_inter, hidden_size),
        lambda i_i: (i_i, 0),
        pipeline_mode=pl.Buffered(buffer_count=3),
    )

    # 3. Emit the pipeline over the intermediate dimension
    pipeline_fn = pltpu.emit_pipeline(
        functools.partial(
            inner_mlp_kernel,
            b_seq=b_seq,
            b_inter=b_inter,
            hidden_size=hidden_size,
            num_inter=num_inter,
        ),
        grid=(num_inter,),
        in_specs=(x_spec, w_gu_spec, wd_spec),
        out_specs=y_spec,
    )

    # Execute the pipeline, passing our scratchpad forward
    pipeline_fn(x_hbm, w_gu_hbm, wd_hbm, y_hbm, scratches=[y_scratch])


@functools.partial(jax.jit, static_argnums=(3, 4, 5))
def apply_fused_mlp_sharded_v1(
    x: jax.Array,
    w_gu: jax.Array,
    wd: jax.Array,
    mesh: jax.sharding.Mesh,
    b_seq: int = 64,
    b_inter: int = 128,
) -> jax.Array:
    in_specs = (
        P(None, None),  # x
        P(None, "model"),  # w_gu
        P("model", None),  # wd
    )
    out_specs = P(None, None)

    @functools.partial(
        shard_map,
        mesh=mesh,
        in_specs=in_specs,
        out_specs=out_specs,
        check_rep=False,
    )
    def local_fused_mlp(x_loc, w_gu_loc, wd_loc):
        seq_len, hidden_size = x_loc.shape

        # 1D outer grid (parallelizing sequence length across TPU cores)
        grid = (seq_len // b_seq,)

        # Pass full tensors to the kernel main as HBM references
        pallas_in_specs = (
            pl.BlockSpec(memory_space=pltpu.HBM),
            pl.BlockSpec(memory_space=pltpu.HBM),
            pl.BlockSpec(memory_space=pltpu.HBM),
        )
        pallas_out_specs = pl.BlockSpec(memory_space=pltpu.HBM)

        y_loc = pl.pallas_call(
            functools.partial(
                mlp_kernel_main,
                b_seq=b_seq,
                b_inter=b_inter,
                hidden_size=hidden_size,
            ),
            out_shape=jax.ShapeDtypeStruct((seq_len, hidden_size), x_loc.dtype),
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=0,
                grid=grid,
                in_specs=pallas_in_specs,
                out_specs=pallas_out_specs,
                scratch_shapes=[pltpu.VMEM((b_seq, hidden_size), jnp.float32)],
            ),
            compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel",)),
        )(x_loc, w_gu_loc, wd_loc)

        return jax.lax.psum(y_loc, axis_name="model")

    return local_fused_mlp(x, w_gu, wd)


# Fused MLP v2 kernel definition with loop-in-kernel to keep weights in SRAM
def fused_mlp_v2_kernel(
    x_ref,
    w_gu_ref,
    wd_ref,
    y_ref,                 # Output 0 (from out_specs)
    x_scratch_ref,         # Output 1 (from out_specs)
    sem_ref,               # Output 2 (from out_specs)
    *,
    b_seq: int,
    b_inter: int,
    local_inter_size: int,
    hidden_size: int,
    num_seq_blocks: int,
    num_inter_blocks: int,
):
    def loop_seq(i, _):
        start_idx = i * b_seq
        
        # 1. DMA Copy from HBM (x_ref slice) to VMEM (x_scratch_ref)
        x_hbm_slice = x_ref.at[pl.ds(start_idx, b_seq), :]
        dma_read = pltpu.make_async_copy(x_hbm_slice, x_scratch_ref, sem_ref)
        dma_read.start()
        dma_read.wait()
        
        # Load from VMEM to registers
        x_tile = x_scratch_ref[...]
        
        # Initialize y_tile accumulator in registers to 0
        y_tile = jnp.zeros((b_seq, hidden_size), dtype=x_ref.dtype)

        # 2. Inner loop over intermediate dimension (unroll=1 prevents unrolling)
        def loop_inter(j, carry_y):
            # Load w_gu slice dynamically from VMEM reference to registers
            w_gu_sram_slice = w_gu_ref[:, pl.ds(j * 2 * b_inter, 2 * b_inter)][...]
            
            # Compute Matmul 1: (b_seq, hidden_size) @ (hidden_size, 2 * b_inter) -> (b_seq, 2 * b_inter)
            hu = jnp.matmul(x_tile, w_gu_sram_slice, preferred_element_type=jnp.float32)
            
            # Split and Activate
            h = hu[:, :b_inter]
            u = hu[:, b_inter:]
            a = jax.nn.gelu(h, approximate=True) * u
            a = a.astype(x_ref.dtype)
            
            # Load wd slice dynamically from VMEM reference to registers
            wd_sram_slice = wd_ref[pl.ds(j * b_inter, b_inter), :][...]
            
            # Compute Matmul 2: (b_seq, b_inter) @ (b_inter, hidden_size) -> (b_seq, hidden_size)
            y_tile_contrib = jnp.matmul(a, wd_sram_slice, preferred_element_type=jnp.float32)
            
            # Accumulate
            return carry_y + y_tile_contrib.astype(carry_y.dtype)

        y_tile_final = jax.lax.fori_loop(0, num_inter_blocks, loop_inter, y_tile, unroll=1)
        
        # 3. Store output tile back to HBM via VMEM and DMA
        x_scratch_ref[...] = y_tile_final
        
        y_hbm_slice = y_ref.at[pl.ds(start_idx, b_seq), :]
        dma_write = pltpu.make_async_copy(x_scratch_ref, y_hbm_slice, sem_ref)
        dma_write.start()
        dma_write.wait()
        
        return None

    jax.lax.fori_loop(0, num_seq_blocks, loop_seq, None, unroll=1)


@functools.partial(jax.jit, static_argnums=(3, 4, 5))
def apply_fused_mlp_sharded_v2(
    x: jax.Array,
    w_gu: jax.Array,
    wd: jax.Array,
    mesh: jax.sharding.Mesh,
    b_seq: int = 128,
    b_inter: int = 128,
) -> jax.Array:
    in_specs = (
        P(None, None),  # x
        P(None, "model"),  # w_gu
        P("model", None),  # wd
    )

    @functools.partial(
        shard_map,
        mesh=mesh,
        in_specs=in_specs,
        out_specs=P(None, None),
        check_rep=False,
    )
    def local_fused_mlp_v2(x_loc, w_gu_loc, wd_loc):
        seq_len, hidden_size = x_loc.shape
        local_inter_size = wd_loc.shape[0]
        
        assert seq_len % b_seq == 0, f"seq_len ({seq_len}) must be multiple of b_seq ({b_seq})"
        num_seq_blocks = seq_len // b_seq
        num_inter_blocks = local_inter_size // b_inter

        # Block specs:
        x_spec = pl.BlockSpec(block_shape=(seq_len, hidden_size), index_map=lambda *args: (0, 0), memory_space=pltpu.HBM)
        y_spec = pl.BlockSpec(block_shape=(seq_len, hidden_size), index_map=lambda *args: (0, 0), memory_space=pltpu.HBM)
        
        w_gu_spec = pl.BlockSpec(block_shape=(hidden_size, w_gu_loc.shape[1]), index_map=lambda *args: (0, 0), memory_space=pltpu.VMEM)
        wd_spec = pl.BlockSpec(block_shape=(local_inter_size, hidden_size), index_map=lambda *args: (0, 0), memory_space=pltpu.VMEM)

        # Workspaces declared in out_shape
        out_shape = (
            jax.ShapeDtypeStruct((seq_len, hidden_size), x_loc.dtype), # Actual output y
            jax.ShapeDtypeStruct((b_seq, hidden_size), x_loc.dtype), # x_scratch in VMEM
            pltpu.SemaphoreType.DMA(()), # Semaphore
        )
        
        # Workspaces specifications in out_specs
        x_scratch_spec = pl.BlockSpec(block_shape=(b_seq, hidden_size), index_map=lambda *args: (0, 0), memory_space=pltpu.VMEM)
        sem_spec = pl.BlockSpec(block_shape=(), index_map=lambda *args: (), memory_space=pltpu.SEMAPHORE)
        
        out_specs = (y_spec, x_scratch_spec, sem_spec)

        y_loc, _, _ = pl.pallas_call(
            functools.partial(
                fused_mlp_v2_kernel,
                b_seq=b_seq,
                b_inter=b_inter,
                local_inter_size=local_inter_size,
                hidden_size=hidden_size,
                num_seq_blocks=num_seq_blocks,
                num_inter_blocks=num_inter_blocks,
            ),
            out_shape=out_shape,
            in_specs=(x_spec, w_gu_spec, wd_spec),
            out_specs=out_specs,
            grid=(1,),
            compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel",)),
        )(x_loc, w_gu_loc, wd_loc)

        return jax.lax.psum(y_loc, axis_name="model")

    return local_fused_mlp_v2(x, w_gu, wd)


def apply_fused_mlp_with_padding(
    x: jax.Array,
    w_gu: jax.Array,
    wd: jax.Array,
    mesh: jax.sharding.Mesh,
    b_inter: int = 128,
) -> jax.Array:
    """Hybrid Fused MLP selector.

    For large sequence lengths (prefill), runs a fast unfused merged-projection
    fallback using standard matmuls. For small sequence lengths (decode),
    runs the optimized fused MLP v2 kernel with padded sequence blocks
    to avoid recompilation.
    """
    seq_len, hidden_size = x.shape

    # 1. Prefill Fallback: seq_len > 128 uses fast sharded matmuls (compute-bound)
    if seq_len > 128:
        in_specs = (P(None, None), P(None, "model"), P("model", None))
        @functools.partial(
            shard_map,
            mesh=mesh,
            in_specs=in_specs,
            out_specs=P(None, None),
            check_rep=False,
        )
        def local_unfused(x_loc, w_gu_loc, wd_loc):
            gate_up = jnp.matmul(x_loc, w_gu_loc)
            gate, up = jnp.split(gate_up, 2, axis=-1)
            a = jax.nn.gelu(gate, approximate=True) * up
            y = jnp.matmul(a, wd_loc)
            return jax.lax.psum(y, axis_name="model")
        return local_unfused(x, w_gu, wd)

    # 2. Decode Mode: Pad token count to 64 or 128 to bound compilation shapes
    target_b_seq = 64 if seq_len <= 64 else 128
    rem = seq_len % target_b_seq

    if rem == 0:
        return apply_fused_mlp_sharded_v2(x, w_gu, wd, mesh, b_seq=target_b_seq, b_inter=b_inter)

    pad_len = target_b_seq - rem
    x_padded = jnp.pad(x, ((0, pad_len), (0, 0)), mode="constant")
    out_padded = apply_fused_mlp_sharded_v2(x_padded, w_gu, wd, mesh, b_seq=target_b_seq, b_inter=b_inter)
    return out_padded[:seq_len, :]

