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
from tpu_inference.models.jax.qwen3_vl import Qwen3VLVisionTransformer
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
        
        # 3. The Language Head
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
            # Omni HF calls it ln_q, mlp.0, mlp.2 -> JAX calls it norm, linear_fc1, linear_fc2
            "thinker.visual.merger.ln_q": "thinker.visual.merger.norm.scale",
            "thinker.visual.merger.ln_q.bias": "thinker.visual.merger.norm.bias",
            "thinker.visual.merger.mlp.0": "thinker.visual.merger.linear_fc1.kernel",
            "thinker.visual.merger.mlp.0.bias": "thinker.visual.merger.linear_fc1.bias",
            "thinker.visual.merger.mlp.2": "thinker.visual.merger.linear_fc2.kernel",
            "thinker.visual.merger.mlp.2.bias": "thinker.visual.merger.linear_fc2.bias",
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
        # 4. Fetch the NNX state and shardings
        params = nnx.state(self)
        try:
            shardings = nnx.get_named_sharding(params, self.mesh)
        except TypeError:
            shardings = params

        # 5. The Interceptor logic
        def intercept_and_filter(weights_iterator):
            for name, tensor in weights_iterator:
                # Intercept vision tower weights
                if name.startswith("thinker.visual."):
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
        result = super().load_weights(filtered_weights)
        
        visual_params_only = {
            path: value for path, value in params.items() 
            if "visual" in path  # Only grab the weights we intercepted
        }
        
        # Apply only the visual weights back to the JAX PyTree
        nnx.update(self, nnx.State(visual_params_only))
        
        return result