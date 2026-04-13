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
from typing import List, Optional, Tuple, Iterable

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
        
        # 2. The Language Head
        config = vllm_config.model_config.hf_config
        text_config = getattr(config, "text_config", config)
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
        self._text_config = getattr(getattr(hf_config, "thinker_config", None),
                                      "text_config", None)

    def __getattr__(self, name):
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


class Qwen3OmniMoeForConditionalGeneration(JaxModule,LoadableWithIterator):
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

    def __call__(
        self,
        kv_caches: List[jax.Array],
        input_ids: jax.Array,
        attention_metadata: AttentionMetadata,
        inputs_embeds: Optional[jax.Array] = None,
        *args, # Safely trap all of vLLM's extra positional arguments here
        **kwargs,
    ) -> Tuple[List[jax.Array], jax.Array | JaxIntermediateTensors, List[jax.Array]]:
        
        if (getattr(attention_metadata, "input_positions", None) is not None
                and attention_metadata.input_positions.ndim == 2
                and attention_metadata.input_positions.shape[0] == 3):
            attention_metadata.input_positions = attention_metadata.input_positions[0]

        visual_pos_mask = kwargs.get("visual_pos_mask", None)
        deepstack_visual_embeds = kwargs.get("deepstack_visual_embeds", None)

        out_kv_caches, hidden_states = self.thinker(
            kv_caches=kv_caches,
            input_ids=input_ids,
            attention_metadata=attention_metadata,
            inputs_embeds=inputs_embeds,
            visual_pos_mask=visual_pos_mask,
            deepstack_visual_embeds=deepstack_visual_embeds,
        )

        return out_kv_caches, hidden_states, []

    def compute_logits(self, hidden_states: jax.Array) -> jax.Array:
        return self.thinker.compute_logits(hidden_states)

    def embed_input_ids(
        self,
        input_ids: jax.Array,
        multimodal_embeddings: Optional[jax.Array] = None,
        *args,
        **kwargs,
    ) -> jax.Array:
        # For text-only, we just use the embed_tokens of the language model.
        # When we add multimodal, we will merge embeddings here.
        # To allow precompilation for text-only inference, we ignore multimodal_embeddings for now.
        # When we implement multimodal support, we will merge embeddings here.
        
        # In Qwen3MoeForCausalLM, self.thinker is Qwen3MoeModel which has self.embed_tokens
        return self.thinker.model.embed_tokens(input_ids)

    
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


    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]) -> set[str]:
        """
        Intercept the weights iterator to strip out multi-modal components 
        (talker, code2wav, visual, audio_tower) until they are implemented.
        """
        def text_only_filter(weights_iterator):
            for name, tensor in weights_iterator:
                # Only yield weights that belong to the language backbone and LM head
                if name.startswith("thinker.model.") or name.startswith("thinker.lm_head."):
                    yield name, tensor
                else:
                    # Explicitly dropping: 'talker.*', 'code2wav.*', 'thinker.visual.*', 'thinker.audio_tower.*'
                    # You can add a debug print here if you want to see exactly what is being dropped:
                    # print(f"Skipping unimplemented weight: {name}")
                    pass
        
        # Wrap the original iterator with our filter
        filtered_weights = text_only_filter(weights)
        
        # Pass the filtered iterable up to LoadableWithIterator
        return super().load_weights(filtered_weights)