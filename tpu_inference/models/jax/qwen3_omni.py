# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import re
from typing import List, Optional, Tuple, Iterable, Any

import jax
import torch
import jax.numpy as jnp
from flax import nnx
from jax.sharding import Mesh
from vllm.config import VllmConfig

from tpu_inference.layers.common.attention_metadata import AttentionMetadata
from tpu_inference.layers.jax import JaxModule
from tpu_inference.layers.jax.pp_utils import PPMissingLayer, make_layers
from tpu_inference.models.jax.jax_intermediate_tensor import JaxIntermediateTensors
from tpu_inference.models.jax.utils.weight_utils import (LoadableWithIterator,
                                                      _load_and_shard_weight,
                                                      check_all_loaded,
                                                      get_default_maps,
                                                      assign_and_shard_param,
                                                      load_hf_weights)

from tpu_inference.models.jax.qwen3_vl_moe import (
    Qwen3VLMoeTextModel,
    Qwen3VLMoeDecoderLayer,
    _VllmConfigAdapter,
)
from tpu_inference.models.jax.qwen3_vl import Qwen3VLForConditionalGeneration, Qwen3VLVisionTransformer
from tpu_inference.models.jax.utils.multi_modal_utils import (
    merge_multimodal_embeddings,
)
from tpu_inference.distributed.jax_parallel_state import get_pp_group
from tpu_inference.layers.jax.embed import JaxEmbed
from tpu_inference.layers.jax.norm import JaxRmsNorm
from tpu_inference.layers.jax.linear import JaxEinsum
from tpu_inference.layers.jax.pp_utils import PPMissingLayer, make_layers
from tpu_inference.models.jax.utils.weight_utils import LoadableWithIterator

init_fn = nnx.initializers.uniform()


def dump_all_tpu_memory(tag=""):
    import jax
    mem_strs = []
    for d in jax.devices():
        stats = d.memory_stats()
        used = stats.get('bytes_in_use', 0) / (1024**2)
        mem_strs.append(f"D{d.id}:{used:.0f}M")
    print(f"[MEM-ALL] {tag} | {' '.join(mem_strs)}")


import numpy as np
def _host_print_stats(name: str, t_np: np.ndarray):
    """Executes on the CPU/Host with standard numpy arrays for clean logging."""
    shape_str = f"[{', '.join(map(str, t_np.shape))}]"
    print(f"{name:<20} | Shape: {shape_str:<15} | Mean: {float(np.mean(t_np)):>8.4f} | Std: {float(np.std(t_np)):>8.4f} | Max: {float(np.max(t_np)):>8.4f} | Min: {float(np.min(t_np)):>8.4f} | NaN: {bool(np.isnan(t_np).any())}")

def jax_print_stats(name: str, t: jax.Array):
    """Safe to call inside jitted/traced functions."""
    jax.debug.callback(_host_print_stats, name, t)



#-------------------------------------------------------------------
#                       Audio Related parts go here
#-------------------------------------------------------------------

class SinusoidsPositionEmbedding(nnx.Module):
    def __init__(self, length: int, channels: int, max_timescale: int = 10000):
        super().__init__()
        if channels % 2 != 0:
            raise ValueError("SinusoidsPositionEmbedding needs even channels input")

        self.length = length
        self.channels = channels
        self.max_timescale = max_timescale

    def __call__(self, seqlen: int) -> jax.Array:
        log_timescale_increment = jnp.log(self.max_timescale) / (self.channels // 2 - 1)
        inv_timescales = jnp.exp(-log_timescale_increment * jnp.arange(self.channels // 2, dtype=jnp.float32))
        scaled_time = jnp.arange(self.length, dtype=jnp.float32)[:, jnp.newaxis] * inv_timescales[jnp.newaxis, :]
        positional_embedding = jnp.concatenate([jnp.sin(scaled_time), jnp.cos(scaled_time)], axis=1)
        
        return positional_embedding[:seqlen, :]


class Qwen3OmniMoeAudioAttention(nnx.Module):
    def __init__(self, config, dtype: jnp.dtype, rngs: nnx.Rngs, prefix: str = ""):
        self.embed_dim = config.d_model
        self.num_heads = config.encoder_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.scaling = self.head_dim ** -0.5

        # Column Parallel
        self.qkv = nnx.Linear(
            self.embed_dim, 
            self.embed_dim * 3, 
            use_bias=True, 
            param_dtype=dtype,
            kernel_init=nnx.with_partitioning(init_fn, (None, "model")),
            bias_init=nnx.with_partitioning(init_fn, ("model", )),
            rngs=rngs,
        )
        # Row Parallel
        self.out_proj = nnx.Linear(
            self.embed_dim, 
            self.embed_dim, 
            use_bias=True, 
            param_dtype=dtype,
            kernel_init=nnx.with_partitioning(init_fn, ("model", None)),
            bias_init=nnx.with_partitioning(init_fn, (None, )),
            rngs=rngs,
        )

    def __call__(self, hidden_states: jax.Array, attention_mask: Optional[jax.Array] = None, debug: bool = False) -> jax.Array:
        B, S, _ = hidden_states.shape
        qkv = self.qkv(hidden_states)
        
        if debug: jax_print_stats("   -> [Attn] 1. QKV Out", qkv)
        
        q, k, v = jnp.split(qkv, 3, axis=-1)

        q = q.reshape((B, S, self.num_heads, self.head_dim))
        k = k.reshape((B, S, self.num_heads, self.head_dim))
        v = v.reshape((B, S, self.num_heads, self.head_dim))

        # Standard Attention
        attn_weights = jnp.einsum('bqhd,bkhd->bhqk', q, k) * self.scaling
        if debug: jax_print_stats("   -> [Attn] 2. Logits (Pre-Mask)", attn_weights)
        
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
            
        attn_weights = jax.nn.softmax(attn_weights, axis=-1)
        if debug: jax_print_stats("   -> [Attn] 3. Probs (Softmax)", attn_weights)

        attn_output = jnp.einsum('bhqk,bkhd->bqhd', attn_weights, v)
        attn_output = attn_output.reshape((B, S, self.embed_dim))

        # ---> ADD THESE PRINTS HERE <---
        if debug: jax_print_stats("   -> [Attn] 2b. Post-Einsum", attn_output)
        
        attn_output = attn_output.reshape((B, S, self.embed_dim))
        
        if debug: jax_print_stats("   -> [Attn] 2c. Pre-Proj", attn_output)

        out = self.out_proj(attn_output)
        if debug: jax_print_stats("   -> [Attn] 3. Out Proj", out)
        

        return out


class Qwen3OmniMoeAudioEncoderLayer(nnx.Module):
    def __init__(self, config, dtype: jnp.dtype, rngs: nnx.Rngs, prefix: str = ""):
        self.embed_dim = config.d_model
        self.self_attn = Qwen3OmniMoeAudioAttention(config, dtype=dtype, rngs=rngs, prefix=f"{prefix}.self_attn")
        self.self_attn_layer_norm = nnx.LayerNorm(
            self.embed_dim, 
            epsilon=1e-5, 
            dtype=dtype, 
            rngs=rngs,
            scale_init=nnx.with_partitioning(init_fn, (None,)),
            bias_init=nnx.with_partitioning(init_fn, (None,)),
        )
        
        # Column Parallel
        self.fc1 = nnx.Linear(
            self.embed_dim, 
            config.encoder_ffn_dim, 
            use_bias=True, 
            param_dtype=dtype,
            kernel_init=nnx.with_partitioning(init_fn, (None, "model")),
            bias_init=nnx.with_partitioning(init_fn, ("model", )),
            rngs=rngs,
        )
        # Row Parallel
        self.fc2 = nnx.Linear(
            config.encoder_ffn_dim, 
            self.embed_dim, 
            use_bias=True, 
            param_dtype=dtype,
            kernel_init=nnx.with_partitioning(init_fn, ("model", None)),
            bias_init=nnx.with_partitioning(init_fn, (None, )),
            rngs=rngs,
        )
        self.final_layer_norm = nnx.LayerNorm(
            self.embed_dim, 
            epsilon=1e-5, 
            dtype=dtype, 
            rngs=rngs,
            scale_init=nnx.with_partitioning(init_fn, (None,)),
            bias_init=nnx.with_partitioning(init_fn, (None,)),
        )
        
        self.activation_fn = jax.nn.gelu

    def __call__(self, hidden_states: jax.Array, attention_mask: Optional[jax.Array] = None, debug: bool = False) -> jax.Array:
        residual = hidden_states
        
        hidden_states = self.self_attn_layer_norm(hidden_states)
        if debug: jax_print_stats("-> [L0] LN1 Out", hidden_states)
        
        hidden_states = self.self_attn(hidden_states, attention_mask, debug=debug)
        hidden_states = residual + hidden_states
        if debug: jax_print_stats("-> [L0] Post-Attn Res", hidden_states)

        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        if debug: jax_print_stats("-> [L0] LN2 Out", hidden_states)
        
        hidden_states = self.fc1(hidden_states)
        if debug: jax_print_stats("-> [L0] FC1 Out", hidden_states)
        
        hidden_states = self.activation_fn(hidden_states)
        if debug: jax_print_stats("-> [L0] Activation Out", hidden_states)
        
        hidden_states = self.fc2(hidden_states)
        if debug: jax_print_stats("-> [L0] FC2 Out", hidden_states)
        
        hidden_states = residual + hidden_states
        if debug: jax_print_stats("-> [L0] Final Out", hidden_states)

        return hidden_states

class Qwen3OmniMoeAudioEncoder(nnx.Module):
    def __init__(self, config, dtype: jnp.dtype, rngs: nnx.Rngs, prefix: str = ""):
        self.config = config
        embed_dim = config.d_model
        
        self.positional_embedding = SinusoidsPositionEmbedding(config.max_source_positions, embed_dim)

        self.conv2d1 = nnx.Conv(1, config.downsample_hidden_size, kernel_size=(3, 3), strides=(2, 2), padding=((1, 1), (1, 1)), param_dtype=dtype, rngs=rngs)
        self.conv2d2 = nnx.Conv(config.downsample_hidden_size, config.downsample_hidden_size, kernel_size=(3, 3), strides=(2, 2), padding=((1, 1), (1, 1)), param_dtype=dtype, rngs=rngs)
        self.conv2d3 = nnx.Conv(config.downsample_hidden_size, config.downsample_hidden_size, kernel_size=(3, 3), strides=(2, 2), padding=((1, 1), (1, 1)), param_dtype=dtype, rngs=rngs)

        conv_out_dim = config.downsample_hidden_size * ((((config.num_mel_bins + 1) // 2 + 1) // 2 + 1) // 2)
        self.conv_out = nnx.Linear(conv_out_dim, config.d_model, use_bias=False, param_dtype=dtype, rngs=rngs)

        self.layers = nnx.List([
            Qwen3OmniMoeAudioEncoderLayer(config, dtype=dtype, rngs=rngs, prefix=f"{prefix}.layers.{i}") 
            for i in range(config.encoder_layers)
        ])

        self.ln_post = nnx.LayerNorm(
            config.d_model, 
            epsilon=1e-5, 
            dtype=dtype, 
            rngs=rngs,
            scale_init=nnx.with_partitioning(init_fn, (None,)),
            bias_init=nnx.with_partitioning(init_fn, (None,)),
        )
        self.proj1 = nnx.Linear(config.d_model, config.d_model, use_bias=True, param_dtype=dtype, rngs=rngs)
        self.act = jax.nn.gelu
        self.proj2 = nnx.Linear(config.d_model, config.output_dim, use_bias=True, param_dtype=dtype, rngs=rngs)

    # def __call__(self, input_features: jax.Array) -> jax.Array:
    #     if input_features.ndim == 4 and input_features.shape[1] == 1:
    #         x = jnp.transpose(input_features, (0, 2, 3, 1))
    #     elif input_features.ndim == 3:
    #         x = jnp.expand_dims(input_features, axis=-1)
    #     elif input_features.ndim == 2:
    #         # Handle unbatched 2D inputs (F, T) -> (1, F, T, 1)
    #         x = jnp.expand_dims(input_features, axis=(0, -1))
    #     else:
    #         x = input_features
        
    #     jax.debug.print("DEBUG 0 [INPUT]: mean={mean}, std={std}", mean=x.mean(), std=x.std())

    #     x = jax.nn.gelu(self.conv2d1(x))
    #     x = jax.nn.gelu(self.conv2d2(x))
    #     x = jax.nn.gelu(self.conv2d3(x))

    #     B, F, T, C = x.shape
    #     x = jnp.transpose(x, (0, 2, 3, 1)) 
    #     x = x.reshape((B, T, C * F))

    #     x = self.conv_out(x)

    #     jax.debug.print("DEBUG 1 [POST-CONV]: mean={mean}, std={std}", mean=x.mean(), std=x.std())

    #     pos_emb = self.positional_embedding(x.shape[1])
    #     x = x + jnp.expand_dims(pos_emb, 0).astype(x.dtype)

    #     jax.debug.print("DEBUG 2 [POST-POSEMB]: mean={mean}, std={std}", mean=x.mean(), std=x.std())

    #     for i, layer in enumerate(self.layers):
    #         x = layer(x)
    #         if i == 0: # Just checking the first layer
    #             jax.debug.print("DEBUG 3 [POST-LAYER-0]: mean={mean}, std={std}", mean=x.mean(), std=x.std())

    #     x = self.ln_post(x)
    #     x = self.proj1(x)
    #     x = self.act(x)
    #     x = self.proj2(x)

    #     jax.debug.print("DEBUG 4 [FINAL-AUDIO-OUT]: mean={mean}, std={std}", mean=x.mean(), std=x.std())

    #     return x

    def __call__(self, input_features: jax.Array) -> jax.Array:
        # 1. Normalize input to [Freq, Time]
        x = input_features
        if x.ndim == 4:
            x = x[0, 0] if x.shape[1] == 1 else x[0, :, :, 0]
        elif x.ndim == 3:
            x = x[0]
            
        # Formatting Header
        jax.debug.print("\n" + "="*80)
        jax.debug.print("AUDIO ENCODER FULL TRACE DEBUG (JAX)")
        jax.debug.print("="*80)

        jax_print_stats("0. Input Features", x)

        F, T = x.shape
        chunk_size = 100
        
        # 2. Calculate chunking mathematically
        num_chunks = (T + chunk_size - 1) // chunk_size
        pad_len = num_chunks * chunk_size - T
        
        # 3. Pad the Time dimension with zeros
        x_padded = jnp.pad(x, ((0, 0), (0, pad_len)))
        
        # 4. Reshape into independent chunks: [Freq, num_chunks, 100]
        x_chunks = x_padded.reshape((F, num_chunks, chunk_size))
        
        # 5. Transpose for Conv2D: [num_chunks, Freq, 100, 1]
        x_chunks = jnp.transpose(x_chunks, (1, 0, 2))
        x_chunks = jnp.expand_dims(x_chunks, axis=-1)
        
        jax_print_stats("1. Pre-CNN Padded", x_chunks)

        # 6. Apply CNN independently to all chunks
        x_conv = jax.nn.gelu(self.conv2d1(x_chunks))
        x_conv = jax.nn.gelu(self.conv2d2(x_conv))
        x_conv = jax.nn.gelu(self.conv2d3(x_conv))
        
        # 7. Extract the dimensions: [num_chunks, F_out, T_out, C_out]
        N, F_out, T_out, C_out = x_conv.shape
        
        # 8. Transpose to match PyTorch permute(0, 3, 1, 2) -> [num_chunks, Time, Freq, Channels]
        x_conv = jnp.transpose(x_conv, (0, 2, 3, 1))
        
        # 9. Flatten Freq and Channels
        x_conv = x_conv.reshape((N, T_out, C_out * F_out))
        x_conv = self.conv_out(x_conv) # Shape: [num_chunks, T_out, Latent_Dim]
        
        jax_print_stats("2. Post-CNN", x_conv)

        # 10. Add Positional Embeddings PER CHUNK
        pos_emb = self.positional_embedding(T_out)
        pos_emb_expanded = jnp.expand_dims(pos_emb, 0).astype(x_conv.dtype)
        
        jax_print_stats("3. Pos Embeddings", pos_emb_expanded)
        
        x_conv = x_conv + pos_emb_expanded
        jax_print_stats("4. Post Pos-Embed", x_conv)

        # 11. Flatten the chunks into a single sequence
        x_flat = x_conv.reshape((N * T_out, -1))
        
        # 12. Slice only the valid tokens
        total_valid = int(_get_feat_extract_output_lengths(jnp.array(T)))
        x_valid = x_flat[:total_valid, :] 
        
        jax_print_stats("5. Hidden (Masked)", x_valid)

        # 13. Add the Batch dimension back for the Transformer Layers
        x_valid = jnp.expand_dims(x_valid, 0) # Shape: [1, Seq, Dim]
        
        # 14. Apply the rest of the tower
        for i, layer in enumerate(self.layers):
            x_valid = layer(x_valid, debug=(i == 0))
            if i == 0:
                jax_print_stats("6. Post Layer 0", x_valid)
                
        jax_print_stats("7. Post All Layers", x_valid)
            
        x_valid = self.ln_post(x_valid)
        jax_print_stats("8. Post LayerNorm", x_valid)

        x_valid = self.proj1(x_valid)
        jax_print_stats("9. Post Proj1", x_valid)

        x_valid = self.act(x_valid)
        x_valid = self.proj2(x_valid)
        
        jax_print_stats("10. FINAL OUTPUT", x_valid)
        jax.debug.print("="*80 + "\n")
        
        return x_valid

#-------------------------------------------------------------------
#                    Audio Encoder Ends
#-------------------------------------------------------------------





class Qwen3OmniTextModel(Qwen3VLMoeTextModel):
    def __init__(self, vllm_config, rng, mesh):
        adapted = _VllmConfigAdapter(vllm_config)
        model_config = adapted.model_config
        hf_config = model_config.hf_config
        vocab_size = model_config.get_vocab_size()
        dtype = model_config.dtype
        rms_norm_eps = hf_config.rms_norm_eps
        hidden_size = hf_config.hidden_size
        prefix = "thinker.model"

        self.is_first_rank = get_pp_group().is_first_rank
        self.is_last_rank = get_pp_group().is_last_rank

        if self.is_first_rank or (hf_config.tie_word_embeddings and self.is_last_rank):
            self.embed_tokens = JaxEmbed(
                num_embeddings=vocab_size,
                features=hidden_size,
                param_dtype=dtype,
                embedding_init=nnx.with_partitioning(init_fn, ("model", None)),
                rngs=rng,
                quant_config=adapted.quant_config,
                prefix=prefix + ".embed_tokens",
            )
        else:
            self.embed_tokens = PPMissingLayer()

        self.start_layer, self.end_layer, self.layers = make_layers(
            hf_config.num_hidden_layers,
            lambda layer_index: Qwen3VLMoeDecoderLayer(
                config=hf_config,
                dtype=dtype,
                rng=rng,
                mesh=mesh,
                kv_cache_dtype=adapted.cache_config.cache_dtype,
                quant_config=adapted.quant_config,
                layer_idx=layer_index,
                vllm_config=adapted,
                prefix=f"{prefix}.layers.{layer_index}",
            ))

        if self.is_last_rank:
            self.norm = JaxRmsNorm(
                hidden_size,
                epsilon=rms_norm_eps,
                param_dtype=dtype,
                scale_init=nnx.with_partitioning(init_fn, (None, )),
                rngs=rng,
                quant_config=adapted.quant_config,
                prefix=prefix + ".final_layernorm",
            )
        else:
            self.norm = PPMissingLayer()


class Qwen3OmniThinkerWrapper(JaxModule, LoadableWithIterator):
    def __init__(self, vllm_config, rng_key, mesh):
        # 1. The MRoPE-aware Text Backbone
        self.model = Qwen3OmniTextModel(vllm_config=vllm_config, rng=nnx.Rngs(rng_key), mesh=mesh)
        
        # 2. The Vision Tower
        config = vllm_config.model_config.hf_config
        text_config = getattr(config, "text_config", config)
        
        self.visual = Qwen3VLVisionTransformer(
            vllm_config=vllm_config,
            rngs=nnx.Rngs(rng_key),
            mesh=mesh,
            norm_eps=getattr(text_config, "rms_norm_eps", 1e-6),
            # Note: qwen3_vl might not take a prefix argument directly, 
            # but it will automatically nest under 'visual' since it's a class attribute here.
        )
        
        # 3. The Audio Tower
        if hasattr(config, "audio_config"):
            self.audio_tower = Qwen3OmniMoeAudioEncoder(
                config=config.audio_config,
                dtype=vllm_config.model_config.dtype,
                rngs=nnx.Rngs(rng_key),
                prefix="thinker.audio_tower",
            )
            
        # 4. The Language Head
        if not config.tie_word_embeddings:
            vocab_size = vllm_config.model_config.get_vocab_size()
            hidden_size = text_config.hidden_size
            self.lm_head = JaxEinsum(
                einsum_str="TD,DV->TV",
                kernel_shape=(hidden_size, vocab_size),
                dtype=vllm_config.model_config.dtype,
                rngs=nnx.Rngs(rng_key),
                quant_config=vllm_config.quant_config,
                prefix="thinker.lm_head",
            )
        else:
            self.lm_head = PPMissingLayer()
            
    def __call__(
        self, 
        kv_caches: List[jax.Array], 
        input_ids: Optional[jax.Array], 
        attention_metadata: AttentionMetadata, 
        inputs_embeds: Optional[jax.Array] = None, 
        visual_pos_mask: Optional[jax.Array] = None, 
        deepstack_visual_embeds: Optional[List[jax.Array]] = None
    ):
       
        return self.model(
            kv_caches=kv_caches,
            input_ids=input_ids,
            attention_metadata=attention_metadata,
            inputs_embeds=inputs_embeds,
            visual_pos_mask=visual_pos_mask,
            deepstack_visual_embeds=deepstack_visual_embeds
        )
        
    def compute_logits(self, hidden_states):
        if hasattr(self, 'lm_head') and not isinstance(self.lm_head, PPMissingLayer):
            return self.lm_head(hidden_states)
        return self.model.embed_tokens.decode(hidden_states)


class _Qwen3OmniModelConfigAdapter:
    def __init__(self, hf_config):
        self._hf_config = hf_config
        self._thinker_config = getattr(hf_config, "thinker_config", hf_config)
        self._text_config = getattr(self._thinker_config, "text_config", self._thinker_config)

    def __getattr__(self, name):
        if name == "vision_config":
            return getattr(self._thinker_config, "vision_config")
        if name == "audio_config":
            return getattr(self._thinker_config, "audio_config")
        
        if self._text_config is not None:
            try:
                return getattr(self._text_config, name)
            except AttributeError:
                pass
            
        return getattr(self._hf_config, name)


class _Qwen3OmniVllmModelConfigAdapter:
    def __init__(self, model_config):
        self._model_config = model_config
        self._hf_config_adapter = _Qwen3OmniModelConfigAdapter(
            model_config.hf_config)

    @property
    def hf_config(self):
        return self._hf_config_adapter

    @property
    def hf_text_config(self):
        return self._hf_config_adapter

    def __getattr__(self, name):
        return getattr(self._model_config, name)


class _Qwen3OmniVllmConfigAdapter:
    def __init__(self, vllm_config: VllmConfig):
        self.model_config = _Qwen3OmniVllmModelConfigAdapter(
            vllm_config.model_config)
        self.cache_config = vllm_config.cache_config
        self.quant_config = vllm_config.quant_config


def _get_feat_extract_output_lengths(input_lengths: jax.Array):
    input_lengths_leave = input_lengths % 100
    feat_lengths = (input_lengths_leave - 1) // 2 + 1
    output_lengths = (
        ((feat_lengths - 1) // 2 + 1 - 1) // 2 + 1 + (input_lengths // 100) * 13
    )
    return output_lengths

class Qwen3OmniMoeForConditionalGeneration(Qwen3VLForConditionalGeneration,JaxModule,LoadableWithIterator):
    def __init__(self, vllm_config: VllmConfig, rng: jax.Array, mesh: Mesh):
        self.vllm_config = vllm_config
        self.mesh = mesh

        # Wrap config to expose text_config for the language model
        adapted_vllm_config = _Qwen3OmniVllmConfigAdapter(vllm_config)

        # Rename to thinker to match PyTorch design directly!
        from tpu_inference.models.jax.qwen3_moe import Qwen3MoeForCausalLM
        self.thinker = Qwen3OmniThinkerWrapper(
            vllm_config=adapted_vllm_config,
            rng_key=rng,
            mesh=mesh,
        )

        self.config = adapted_vllm_config.model_config.hf_config
        self.visual = self.thinker.visual

        config = vllm_config.model_config.hf_config
        self.image_token_id = getattr(config, "image_token_id", 151655)
        self.video_token_id = getattr(config, "video_token_id", 151656)
        self.audio_token_id = getattr(config, "audio_token_id", 151675)
        self.vision_start_token_id = getattr(config, "vision_start_token_id", 151652)
        # Fix this later
        self.spatial_merge_size = getattr(config, "spatial_merge_size", 2)

    def __call__(
        self,
        kv_caches: List[jax.Array],
        input_ids: jax.Array,
        attention_metadata: AttentionMetadata,
        inputs_embeds: Optional[jax.Array] = None,
        *args, # Safely trap all of vLLM's extra positional arguments here
        **kwargs,
    ) -> Tuple[List[jax.Array], jax.Array | JaxIntermediateTensors, List[jax.Array]]:
        
        #Check this
        if (getattr(attention_metadata, "input_positions", None) is not None
                and attention_metadata.input_positions.ndim == 2
                and attention_metadata.input_positions.shape[0] == 3):
            attention_metadata.input_positions = attention_metadata.input_positions[0]

        visual_pos_mask = kwargs.get("visual_pos_mask", None)
        deepstack_visual_embeds = kwargs.get("deepstack_visual_embeds", None)

        out_kv_caches, hidden_states = self.thinker(
            kv_caches=kv_caches,
            input_ids=input_ids if inputs_embeds is None else None,
            attention_metadata=attention_metadata,
            inputs_embeds=inputs_embeds,
            visual_pos_mask=visual_pos_mask,
            deepstack_visual_embeds=deepstack_visual_embeds,
        )

        return out_kv_caches, hidden_states, []

    def compute_logits(self, hidden_states: jax.Array) -> jax.Array:
        return self.thinker.compute_logits(hidden_states)

    def get_input_embeddings(
        self,
        input_ids: jax.Array,
        multimodal_embeddings: Optional[jax.Array],
    ) -> jax.Array:
        inputs_embeds = self.thinker.model.embed_tokens(input_ids)
        
        jax.debug.print("DEBUG 5 [INPUT_IDS SHAPE]: {s}", s=input_ids.shape)

        if multimodal_embeddings is not None and len(multimodal_embeddings) != 0:
            # Check the shape of the audio features we are trying to insert
            jax.debug.print("DEBUG 6 [MM FEAT 0 SHAPE]: {s}", s=multimodal_embeddings[0].shape)
            
            # Count how many audio tokens are actually in the prompt
            audio_tok_count = jnp.sum(input_ids == self.audio_token_id)
            jax.debug.print("DEBUG 7 [AUDIO TOKENS IN PROMPT]: expected_id={id}, count={c}", id=self.audio_token_id, c=audio_tok_count)
            
            # Also check for image/video tokens just in case it got misclassified
            img_tok_count = jnp.sum(input_ids == self.image_token_id)
            vid_tok_count = jnp.sum(input_ids == self.video_token_id)
            jax.debug.print("DEBUG 8 [OTHER TOKENS]: img={i}, vid={v}", i=img_tok_count, v=vid_tok_count)

            inputs_embeds = merge_multimodal_embeddings(
                input_ids,
                inputs_embeds,
                multimodal_embeddings,
                [self.image_token_id, self.video_token_id, self.audio_token_id],
            )

        return inputs_embeds

    def embed_input_ids(
        self,
        input_ids: jax.Array,
        multimodal_embeddings: Optional[jax.Array] = None,
        *args,
        **kwargs,
    ) -> jax.Array:
        return self.get_input_embeddings(input_ids, multimodal_embeddings)

    def _parse_and_validate_multimodal_inputs(self, **kwargs: object) -> dict:
        from tpu_inference.models.jax.utils.multi_modal_utils import normalize_mm_grid_thw
        mm_input_by_modality = {}
        for input_key in kwargs:
            if input_key in ("pixel_values", "pixel_values_videos", "image_embeds") and "image" not in mm_input_by_modality:
                image_grid_thw = kwargs.get("image_grid_thw", None)
                if not image_grid_thw:
                    image_grid_thw = kwargs.get("video_grid_thw", None)
                image_grid_thw = normalize_mm_grid_thw(image_grid_thw)
                
                mm_input_by_modality["image"] = self._parse_and_validate_image_input(
                    image_grid_thw, **kwargs)
                    
            if input_key == "input_audio_features" and "audio" not in mm_input_by_modality:
                mm_input_by_modality["audio"] = {
                    "input_audio_features": kwargs.get("input_audio_features"),
                    "audio_feature_lengths": kwargs.get("audio_feature_lengths"),
                }
        return mm_input_by_modality

    def _process_audio_input(self, audio_input: dict) -> Tuple[jax.Array, ...]:
        input_audio_features = audio_input.get("input_audio_features")
        audio_feature_lengths = audio_input.get("audio_feature_lengths")
        if input_audio_features is None:
            return ()
            
        print(f"DEBUG AUDIO INPUT SHAPE: {input_audio_features.shape}")
        audio_embeds = self.thinker.audio_tower(input_audio_features)
        audio_output_lengths = _get_feat_extract_output_lengths(audio_feature_lengths)
        
        B = audio_embeds.shape[0]
        audio_splits = []
        for i in range(B):
            length = int(audio_output_lengths[i]) if hasattr(audio_output_lengths[i], "__int__") else audio_output_lengths[i]
            audio_splits.append(audio_embeds[i, :length])
            
        return tuple(audio_splits)

    def embed_multimodal(self, image_grid_thw=None, **kwargs) -> dict:
        mm_input_by_modality = self._parse_and_validate_multimodal_inputs(**kwargs)
        if not mm_input_by_modality:
            return {}

        multimodal_embeddings: tuple[jax.Array, ...] = ()
        deepstack_outputs = None
        
        for modality in mm_input_by_modality:
            multimodal_input = mm_input_by_modality[modality]
            if modality == "image":
                jax.debug.print("here reached")
                image_splits, deepstack_by_item = self._process_image_input(multimodal_input)
                multimodal_embeddings += image_splits
                if deepstack_by_item is not None:
                    if deepstack_outputs is None:
                        deepstack_outputs = []
                    deepstack_outputs.extend(deepstack_by_item)
            elif modality == "audio":
                audio_splits = self._process_audio_input(multimodal_input)
                multimodal_embeddings += audio_splits

        return {"embeds": multimodal_embeddings, "deepstack": deepstack_outputs}

    
    def get_multimodal_fns(self):
        return {
            "get_mrope_input_positions_fn": self.get_mrope_input_positions,
        }

    def get_mrope_input_positions(
        self,
        input_tokens: List[int],
        hf_config=None,
        image_grid_thw=None,
        video_grid_thw=None,
        second_per_grid_ts=None,
        context_len: int = 0,
        seq_len: Optional[int] = None,
        audio_feature_lengths=None,
        use_audio_in_video: bool = False,
    ) -> Tuple[jax.Array, int]:
        from tpu_inference.models.jax.qwen3_vl import build_mrope_input_positions

        if hf_config is None:
            hf_config = self.vllm_config.model_config.hf_config

        llm_positions, mrope_position_delta = build_mrope_input_positions(
            input_tokens=input_tokens,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            image_token_id=getattr(hf_config, "image_token_id", 151655),
            video_token_id=getattr(hf_config, "video_token_id", 151656),
            vision_start_token_id=getattr(hf_config, "vision_start_token_id", 151657),
            spatial_merge_size=getattr(getattr(hf_config, "vision_config", None),
                                       "spatial_merge_size", 2),
        )

        llm_positions = llm_positions[:, context_len:seq_len]
        return llm_positions, mrope_position_delta


    def _buffer_and_load_audio_qkv(
        self,
        name: str,
        tensor: torch.Tensor,
        qkv_buffer: dict,
        params: nnx.State,
        shardings: Any,
        metadata_map: Any,
    ):
        from tpu_inference.models.jax.utils.weight_utils import _load_and_shard_weight, ensure_cpu_jax_array
        
        parts = name.split(".")
        layer_idx = parts[3]
        proj_type = parts[5].split("_")[0] # 'q', 'k', 'v'
        weight_type = parts[6] # 'weight' or 'bias'
        
        key = (layer_idx, weight_type)
        if key not in qkv_buffer:
            qkv_buffer[key] = {}
        qkv_buffer[key][proj_type] = tensor
        
        if len(qkv_buffer[key]) == 3:
            q = qkv_buffer[key]['q']
            k = qkv_buffer[key]['k']
            v = qkv_buffer[key]['v']
            
            if weight_type == "weight":
                concatenated = torch.cat([q, k, v], dim=0)
                new_name = f"thinker.audio_tower.layers.{layer_idx}.self_attn.qkv.weight"
            else:
                concatenated = torch.cat([q, k, v], dim=0)
                new_name = f"thinker.audio_tower.layers.{layer_idx}.self_attn.qkv.bias"
                
            _load_and_shard_weight(
                vllm_config=self.vllm_config,
                params=params,
                shardings=shardings,
                metadata_map=metadata_map,
                mesh=self.mesh,
                hf_key=new_name,
                hf_weight=ensure_cpu_jax_array(concatenated),
                keep_hf_weight_suffix_when_match=[],
                pp_missing_layers=[], 
            )

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]) -> set[str]:
        from tpu_inference.models.jax.utils.weight_utils import (
            get_default_maps, _load_and_shard_weight, ensure_cpu_jax_array
        )

        # 1. Map Omni HF keys directly to Qwen3 VL JAX PyTree paths.
        mappings = {
            "thinker.visual.patch_embed.proj": "thinker.visual.patch_embed.proj.kernel",
            "thinker.visual.patch_embed.proj.bias": "thinker.visual.patch_embed.proj.bias",
            "thinker.visual.pos_embed": "thinker.visual.pos_embed.embedding",
            "thinker.visual.blocks.*.attn.qkv": "thinker.visual.blocks.*.attn.qkv_proj.kernel",
            "thinker.visual.blocks.*.attn.qkv.bias": "thinker.visual.blocks.*.attn.qkv_proj.bias",
            "thinker.visual.blocks.*.attn.proj": "thinker.visual.blocks.*.attn.proj.kernel",
            "thinker.visual.blocks.*.attn.proj.bias": "thinker.visual.blocks.*.attn.proj.bias",
            "thinker.visual.blocks.*.mlp.linear_fc1": "thinker.visual.blocks.*.mlp.fc1.kernel",
            "thinker.visual.blocks.*.mlp.linear_fc1.bias": "thinker.visual.blocks.*.mlp.fc1.bias",
            "thinker.visual.blocks.*.mlp.linear_fc2": "thinker.visual.blocks.*.mlp.fc2.kernel",
            "thinker.visual.blocks.*.mlp.linear_fc2.bias": "thinker.visual.blocks.*.mlp.fc2.bias",
            "thinker.visual.blocks.*.norm1": "thinker.visual.blocks.*.norm1.scale",
            "thinker.visual.blocks.*.norm1.bias": "thinker.visual.blocks.*.norm1.bias",
            "thinker.visual.blocks.*.norm2": "thinker.visual.blocks.*.norm2.scale",
            "thinker.visual.blocks.*.norm2.bias": "thinker.visual.blocks.*.norm2.bias",

            # --- OMNI MERGER MAPPINGS ---
            "thinker.visual.merger.ln_q": "thinker.visual.merger.norm.scale",
            "thinker.visual.merger.ln_q.bias": "thinker.visual.merger.norm.bias",
            "thinker.visual.merger.mlp.0": "thinker.visual.merger.linear_fc1.kernel",
            "thinker.visual.merger.mlp.0.bias": "thinker.visual.merger.linear_fc1.bias",
            "thinker.visual.merger.mlp.2": "thinker.visual.merger.linear_fc2.kernel",
            "thinker.visual.merger.mlp.2.bias": "thinker.visual.merger.linear_fc2.bias",

            # --- AUDIO TOWER MAPPINGS ---
            "thinker.audio_tower.conv2d1": "thinker.audio_tower.conv2d1.kernel",
            "thinker.audio_tower.conv2d1.bias": "thinker.audio_tower.conv2d1.bias",
            "thinker.audio_tower.conv2d2": "thinker.audio_tower.conv2d2.kernel",
            "thinker.audio_tower.conv2d2.bias": "thinker.audio_tower.conv2d2.bias",
            "thinker.audio_tower.conv2d3": "thinker.audio_tower.conv2d3.kernel",
            "thinker.audio_tower.conv2d3.bias": "thinker.audio_tower.conv2d3.bias",
            "thinker.audio_tower.conv_out": "thinker.audio_tower.conv_out.kernel",
            
            "thinker.audio_tower.layers.*.self_attn.qkv": "thinker.audio_tower.layers.*.self_attn.qkv.kernel",
            "thinker.audio_tower.layers.*.self_attn.qkv.bias": "thinker.audio_tower.layers.*.self_attn.qkv.bias",
            "thinker.audio_tower.layers.*.self_attn.out_proj": "thinker.audio_tower.layers.*.self_attn.out_proj.kernel",
            "thinker.audio_tower.layers.*.self_attn.out_proj.bias": "thinker.audio_tower.layers.*.self_attn.out_proj.bias",
            
            "thinker.audio_tower.layers.*.fc1": "thinker.audio_tower.layers.*.fc1.kernel",
            "thinker.audio_tower.layers.*.fc1.bias": "thinker.audio_tower.layers.*.fc1.bias",
            "thinker.audio_tower.layers.*.fc2": "thinker.audio_tower.layers.*.fc2.kernel",
            "thinker.audio_tower.layers.*.fc2.bias": "thinker.audio_tower.layers.*.fc2.bias",
            
            "thinker.audio_tower.layers.*.self_attn_layer_norm": "thinker.audio_tower.layers.*.self_attn_layer_norm.scale",
            "thinker.audio_tower.layers.*.self_attn_layer_norm.bias": "thinker.audio_tower.layers.*.self_attn_layer_norm.bias",
            "thinker.audio_tower.layers.*.final_layer_norm": "thinker.audio_tower.layers.*.final_layer_norm.scale",
            "thinker.audio_tower.layers.*.final_layer_norm.bias": "thinker.audio_tower.layers.*.final_layer_norm.bias",
            
            "thinker.audio_tower.ln_post": "thinker.audio_tower.ln_post.scale",
            "thinker.audio_tower.ln_post.bias": "thinker.audio_tower.ln_post.bias",
            "thinker.audio_tower.proj1": "thinker.audio_tower.proj1.kernel",
            "thinker.audio_tower.proj1.bias": "thinker.audio_tower.proj1.bias",
            "thinker.audio_tower.proj2": "thinker.audio_tower.proj2.kernel",
            "thinker.audio_tower.proj2.bias": "thinker.audio_tower.proj2.bias",
        }

        # 2. Dynamically add Deepstack mappings
        vision_config = self.vllm_config.model_config.hf_config.thinker_config.vision_config
        deepstack_indexes = getattr(vision_config, "deepstack_visual_indexes", [8, 16, 24])
        for i in range(len(deepstack_indexes)):
            # Omni HF calls it 'merger_list', JAX calls it 'deepstack_merger_list'
            mappings[f"thinker.visual.merger_list.{i}.ln_q"] = f"thinker.visual.deepstack_merger_list.{i}.norm.scale"
            mappings[f"thinker.visual.merger_list.{i}.ln_q.bias"] = f"thinker.visual.deepstack_merger_list.{i}.norm.bias"
            mappings[f"thinker.visual.merger_list.{i}.mlp.0"] = f"thinker.visual.deepstack_merger_list.{i}.linear_fc1.kernel"
            mappings[f"thinker.visual.merger_list.{i}.mlp.0.bias"] = f"thinker.visual.deepstack_merger_list.{i}.linear_fc1.bias"
            mappings[f"thinker.visual.merger_list.{i}.mlp.2"] = f"thinker.visual.deepstack_merger_list.{i}.linear_fc2.kernel"
            mappings[f"thinker.visual.merger_list.{i}.mlp.2.bias"] = f"thinker.visual.deepstack_merger_list.{i}.linear_fc2.bias"

        # 3. Generate default maps, and manually inject transposes for the new MLP keys
        adapted_model_config = _Qwen3OmniVllmModelConfigAdapter(self.vllm_config.model_config)
        metadata_map = get_default_maps(adapted_model_config, self.mesh, mappings)
        
        # Explicitly tell the loader to transpose (1, 0) for these new Omni dense layers
        metadata_map.transpose_map["mlp.0"] = (1, 0)
        metadata_map.transpose_map["mlp.2"] = (1, 0)
        
        audio_transposes = {
            "thinker.audio_tower.layers.*.self_attn.qkv": (1, 0),
            # "thinker.audio_tower.layers.*.self_attn.out_proj": (1, 0),
            "self_attn.out_proj": (1, 0),
            "thinker.audio_tower.conv2d1": (2, 3, 1, 0),
            "thinker.audio_tower.conv2d2": (2, 3, 1, 0),
            "thinker.audio_tower.conv2d3": (2, 3, 1, 0),
            "thinker.audio_tower.conv_out": (1, 0),
            "thinker.audio_tower.proj1": (1, 0),
            "thinker.audio_tower.proj2": (1, 0),
        }
        metadata_map.transpose_map = {**audio_transposes, **metadata_map.transpose_map}
        # 4. Fetch the NNX state and shardings
        params = nnx.state(self)
        try:
            shardings = nnx.get_named_sharding(params, self.mesh)
        except TypeError:
            shardings = params

        # 5. The Interceptor logic
        qkv_buffer = {}
        
        def intercept_and_filter(weights_iterator):
            for name, tensor in weights_iterator:
                if name.startswith("thinker.audio_tower.layers.") and (".self_attn.q_proj." in name or ".self_attn.k_proj." in name or ".self_attn.v_proj." in name):
                    self._buffer_and_load_audio_qkv(
                        name=name,
                        tensor=tensor,
                        qkv_buffer=qkv_buffer,
                        params=params,
                        shardings=shardings,
                        metadata_map=metadata_map,
                    )
                elif name.startswith("thinker.visual.") or name.startswith("thinker.audio_tower."):
                    _load_and_shard_weight(
                        vllm_config=self.vllm_config,
                        params=params,
                        shardings=shardings,
                        metadata_map=metadata_map,
                        mesh=self.mesh,
                        hf_key=name,
                        hf_weight=ensure_cpu_jax_array(tensor),
                        keep_hf_weight_suffix_when_match=[],
                        pp_missing_layers=[], 
                    )
                # Pass everything else (like thinker.model.*) to the MoE text loader
                elif name.startswith("thinker.model") or name == "thinker.lm_head.weight":
                    yield name, tensor

        # 6. Pass the *remaining* text/MoE weights down to the automated loader
        filtered_weights = intercept_and_filter(weights)
        
        # Note: We must execute the delegated loader so the generator actually runs!
        # result = super().load_weights(filtered_weights)
        result = LoadableWithIterator.load_weights(self, filtered_weights)
        
        visual_and_audio_params = {
            path: value for path, value in params.items() 
            if "visual" in path or "audio_tower" in path
        }
        
        # Apply only the intercepted weights back to the JAX PyTree
        nnx.update(self, nnx.State(visual_and_audio_params))
        
        return result