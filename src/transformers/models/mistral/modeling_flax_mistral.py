# coding=utf-8
# Copyright 2023 Mistral AI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
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
""" Flax Mistral model."""
import math
from typing import Any, List, Optional, Tuple, Union

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import torch
from flax.core.frozen_dict import FrozenDict, freeze, unfreeze
from flax.linen import combine_masks, make_causal_mask
from flax.linen.attention import dot_product_attention_weights
from flax.linen.initializers import ones
from flax.traverse_util import flatten_dict, unflatten_dict
from jax import lax

from ...modeling_flax_outputs import (FlaxBaseModelOutputWithPast,
                                      FlaxCausalLMOutputWithCrossAttentions,
                                      FlaxSequenceClassifierOutput)
from ...modeling_flax_utils import (ACT2FN, FlaxPreTrainedModel,
                                    append_call_sample_docstring, logging)
from ...utils import (add_start_docstrings,
                      add_start_docstrings_to_model_forward)
from .configuration_mistral import MistralConfig

logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "MistralConfig"
_CHECKPOINT_FOR_DOC = "mistralai/Mistral-7B-v0.1"

MISTRAL_START_DOCSTRING = r"""

    This model inherits from [`FlaxPreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a Flax Linen
    [flax.nn.Module](https://flax.readthedocs.io/en/latest/_autosummary/flax.nn.module.html) subclass. Use it as a
    regular Flax Module and refer to the Flax documentation for all matter related to general usage and behavior.

    Finally, this model supports inherent JAX features such as:

    - [Just-In-Time (JIT) compilation](https://jax.readthedocs.io/en/latest/jax.html#just-in-time-compilation-jit)
    - [Automatic Differentiation](https://jax.readthedocs.io/en/latest/jax.html#automatic-differentiation)
    - [Vectorization](https://jax.readthedocs.io/en/latest/jax.html#vectorization-vmap)
    - [Parallelization](https://jax.readthedocs.io/en/latest/jax.html#parallelization-pmap)

    Parameters:
        config ([`MistralConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~FlaxPreTrainedModel.from_pretrained`] method to load the model weights.
        dtype (`jax.numpy.dtype`, *optional*, defaults to `jax.numpy.float32`):
            The data type of the computation. Can be one of `jax.numpy.float32`, `jax.numpy.float16`, or
            `jax.numpy.bfloat16`.

            This can be used to enable mixed-precision training or half-precision inference on GPUs or TPUs. If
            specified all the computation will be performed with the given `dtype`.

            **Note that this only specifies the dtype of the computation and does not influence the dtype of model
            parameters.**

            If you wish to change the dtype of the model parameters, see [`~FlaxPreTrainedModel.to_fp16`] and
            [`~FlaxPreTrainedModel.to_bf16`].
"""

MISTRAL_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`numpy.ndarray` of shape `(batch_size, input_ids_length)`):
            Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
            it.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        attention_mask (`numpy.ndarray` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            If `past_key_values` is used, optionally only the last `decoder_input_ids` have to be input (see
            `past_key_values`).

            If you want to change padding behavior, you should read [`modeling_opt._prepare_decoder_attention_mask`]
            and modify to your needs. See diagram 1 in [the paper](https://arxiv.org/abs/1910.13461) for more
            information on the default strategy.

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.
        position_ids (`numpy.ndarray` of shape `(batch_size, sequence_length)`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
            config.n_positions - 1]`.

            [What are position IDs?](../glossary#position-ids)
        past_key_values (`Dict[str, np.ndarray]`, *optional*, returned by `init_cache` or when passing previous `past_key_values`):
            Dictionary of pre-computed hidden-states (key and values in the attention blocks) that can be used for fast
            auto-regressive decoding. Pre-computed key and value hidden-states are of shape *[batch_size, max_length]*.
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return jnp.concatenate((-x2, x1), dim=-1)


# Copied from transformers.models.gpt_neox.modeling_gpt_neox.apply_rotary_pos_emb
def apply_rotary_pos_emb(q, k, cos, sin, position_ids):
    cos = cos[position_ids].unsqueeze(1)  # [seq_len, dim] -> [batch_size, 1, seq_len, head_dim]
    sin = sin[position_ids].unsqueeze(1)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class FlaxMistralRMSNorm(nn.Module):
    hidden_size: int
    eps: float = 1e-6
    dtype: jnp.dtype = jnp.float32
    weight_dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, hidden_states: jnp.ndarray) -> jnp.ndarray:
        weight = self.param("weight", ones, self.hidden_size, self.weight_dtype)
        variance = (hidden_states**2).mean(-1, keepdims=True)
        hidden_states = hidden_states * 1 / jnp.sqrt(variance + self.eps)
        return weight * hidden_states


# class FlaxMistralRotaryEmbedding(nn.Module):
#     dim: int
#     max_position_embeddings: int = 2048
#     base: int = 10000
#     max_seq_len: int = 4096
#     dtype: jnp.dtype = jnp.float32

#     def setup(self):
#         self.inv_freq = 1 / (10000 ** (jnp.arange(0, self.dim, 2) / self.dim))
#         t = jnp.arange(0, self.max_seq_len, dtype=self.dtype)
#         freqs = jnp.einsum("i,j->ij", t, self.inv_freq)
#         emb = jnp.concatenate((freqs, freqs), axis=1)
#         self.cos_cached = jnp.cos(emb)
#         self.sin_cached = jnp.sin(emb)

#     @nn.compact
#     def __call__(self, x: jnp.ndarray, seq_len: int) -> Tuple[jnp.ndarray, jnp.ndarray]:
#         return (self.cos_cached[:seq_len], self.sin_cached[:seq_len])

class FlaxMistralRotaryEmbedding(nn.Module):
    config: MistralConfig
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        head_dim = self.config.hidden_size // self.config.num_attention_heads
        self.sincos = create_sinusoidal_positions(self.config.max_position_embeddings, head_dim)

    def __call__(self, key, query, position_ids):
        sincos = self.sincos[position_ids]
        sin_pos, cos_pos = jnp.split(sincos, 2, axis=-1)

        key = apply_rotary_pos_emb(key, sin_pos, cos_pos)
        query = apply_rotary_pos_emb(query, sin_pos, cos_pos)

        key = jnp.asarray(key, dtype=self.dtype)
        query = jnp.asarray(query, dtype=self.dtype)

        return key, query

class FlaxMistralMLP(nn.Module):

    config: MistralConfig
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.hidden_size = self.config.hidden_size
        self.intermediate_size = self.config.intermediate_size
        self.gate_proj = nn.Dense(self.intermediate_size, use_bias=False, dtype=self.dtype)
        self.up_proj = nn.Dense(self.intermediate_size, use_bias=False, dtype=self.dtype)
        self.down_proj = nn.Dense(self.hidden_size, use_bias=False, dtype=self.dtype)
        self.act_fn = ACT2FN[self.config.hidden_act]

    def __call__(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj


def flax_rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return jnp.concatenate((-x2, x1), axis=-1)


# Copied from transformers.models.gpt_neox.modeling_gpt_neox.apply_rotary_pos_emb
def flax_apply_rotary_pos_emb(q, k, cos, sin, position_ids):
    cos = jnp.expand_dims(cos[position_ids], 1)  # [seq_len, dim] -> [batch_size, 1, seq_len, head_dim]
    sin = jnp.expand_dims(sin[position_ids], 1)
    q_embed = (q * cos) + (flax_rotate_half(q) * sin)
    k_embed = (k * cos) + (flax_rotate_half(k) * sin)
    return q_embed, k_embed

def create_sinusoidal_positions(num_pos, dim):
    inv_freq = 1.0 / (10000 ** (np.arange(0, dim, 2) / dim))
    freqs = np.einsum("i , j -> i j", np.arange(num_pos), inv_freq).astype("float32")

    emb = np.concatenate((freqs, freqs), axis=-1)
    out = np.concatenate((np.sin(emb)[:, None, :], np.cos(emb)[:, None, :]), axis=-1)
    return jnp.array(out[:, :, :num_pos])


def rotate_half(tensor):
    """Rotates half the hidden dims of the input."""
    rotate_half_tensor = jnp.concatenate(
        (-tensor[..., tensor.shape[-1] // 2 :], tensor[..., : tensor.shape[-1] // 2]), axis=-1
    )
    return rotate_half_tensor


def apply_rotary_pos_emb(tensor, sin_pos, cos_pos):
    return (tensor * cos_pos) + (rotate_half(tensor) * sin_pos)



def flax_repeat_kv(hidden_states: jnp.ndarray, n_rep: int) -> jnp.ndarray:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, slen, num_key_value_heads, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = jnp.repeat(hidden_states[:, :, None, :, :], n_rep, axis=3)
    new_size = (batch, slen, num_key_value_heads * n_rep,  head_dim)
    return jax.lax.reshape(hidden_states, new_size)

def _flax_make_sliding_window_causal_mask(
    input_ids_shape: torch.Size,
    dtype: jnp.dtype,
    past_key_values_length: int = 0,
    sliding_window: int = 4096,
):
    """
    Make causal mask used for sliding window attention
    """
    bsz, tgt_len = input_ids_shape

    tensor = jnp.full(
        (tgt_len, tgt_len),
        fill_value=1,
    )
    mask = jnp.tril(tensor, k=0)
    # make the mask banded to account for sliding window
    mask = jnp.triu(mask, k=-sliding_window)
    mask = jnp.log(mask)
    if past_key_values_length > 0:
        mask = jnp.concatenate([jnp.zeros((tgt_len, past_key_values_length), dtype=dtype), mask], axis=-1)
    return jnp.repeat(mask[None, None], bsz, axis=0)

# Copied from transformers.models.bart.modeling_bart._make_causal_mask
def _flax_make_causal_mask(input_ids_shape: Tuple[int], dtype: jnp.dtype, past_key_values_length: int = 0):
    """
    Make causal mask used for bi-directional self-attention.
    """
    bsz, tgt_len = input_ids_shape
    mask = jnp.full((tgt_len, tgt_len), jnp.finfo(dtype).min)
    causal_mask = jnp.triu(jnp.ones((tgt_len, tgt_len)), k=1)
    mask = mask * causal_mask
    mask = mask[None, None, :, :]
    mask = jnp.repeat(mask, bsz, axis=0)

    if past_key_values_length > 0:  # TODO: Update when past_key_values_length is not 0
        mask = jnp.concatenate([jnp.zeros((tgt_len, past_key_values_length), dtype=dtype), mask], axis=-1)
        mask = jnp.repeat(mask, tgt_len + past_key_values_length, axis=1)
    return mask


def _flax_expand_mask(mask: jnp.ndarray, dtype: jnp.dtype, tgt_len: Optional[int] = None):
    """
    Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
    """
    bsz, src_len = mask.shape
    tgt_len = tgt_len if tgt_len is not None else src_len

    expanded_mask = mask[:, None, None, :]
    expanded_mask = jnp.repeat(expanded_mask, tgt_len, axis=2)

    inverted_mask = 1.0 - expanded_mask
    inverted_mask = inverted_mask * jnp.full(inverted_mask.shape, jnp.finfo(dtype).min)

    return inverted_mask



def _prepare_decoder_attention_mask(attention_mask, input_shape, inputs_embeds, past_key_values_length, sliding_window
    ):
    # create causal mask
    # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
    combined_attention_mask = None
    if input_shape[-1] > 1:
        combined_attention_mask = _flax_make_sliding_window_causal_mask(
            input_shape,
            inputs_embeds.dtype,
            past_key_values_length=past_key_values_length,
            sliding_window=sliding_window,
        )

    if attention_mask is not None:
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        expanded_attn_mask = _flax_expand_mask(attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1])
        combined_attention_mask = (
            expanded_attn_mask if combined_attention_mask is None else expanded_attn_mask + combined_attention_mask
        )

    return combined_attention_mask


class FlaxMistralAttention(nn.Module):
    config: MistralConfig
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        config = self.config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.attention_softmax_in_fp32 = self.dtype is not jnp.float32
        self.rope_theta = config.rope_theta
        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )
        self.q_proj = nn.Dense(self.num_heads * self.head_dim, use_bias=False, dtype=self.dtype)
        self.k_proj = nn.Dense(
            self.num_key_value_heads * self.head_dim, use_bias=False, dtype=self.dtype
        )
        self.v_proj = nn.Dense(
            self.num_key_value_heads * self.head_dim, use_bias=False, dtype=self.dtype
        )
        self.o_proj = nn.Dense(self.hidden_size, use_bias=False, dtype=self.dtype)
        self.causal_mask =jnp.triu(make_causal_mask(jnp.ones((1, config.max_position_embeddings), dtype="bool"), dtype="bool"),
                                    k=-config.sliding_window)
        self.rotary_emb = FlaxMistralRotaryEmbedding(config, dtype=self.dtype)
        
    def _split_heads(self, hidden_states, num_heads):
        return hidden_states.reshape(hidden_states.shape[:2] + (num_heads, self.head_dim))

    def _merge_heads(self, hidden_states):
        return hidden_states.reshape(hidden_states.shape[:2] + (self.hidden_size,))
        
    @nn.compact
    # Copied from transformers.models.gpt_neo.modeling_flax_gpt_neo.FlaxGPTNeoSelfAttention._concatenate_to_cache
    def _concatenate_to_cache(self, key, value, query, attention_mask):
        """
        This function takes projected key, value states from a single input token and concatenates the states to cached
        states from previous steps. This function is slighly adapted from the official Flax repository:
        https://github.com/google/flax/blob/491ce18759622506588784b4fca0e4bf05f8c8cd/flax/linen/attention.py#L252
        """
        # detect if we're initializing by absence of existing cache data.
        is_initialized = self.has_variable("cache", "cached_key")
        cached_key = self.variable("cache", "cached_key", jnp.zeros, key.shape, key.dtype)
        cached_value = self.variable("cache", "cached_value", jnp.zeros, value.shape, value.dtype)
        cache_index = self.variable("cache", "cache_index", lambda: jnp.array(0, dtype=jnp.int32))

        if is_initialized:
            *batch_dims, max_length, num_heads, depth_per_head = cached_key.value.shape
            # update key, value caches with our new 1d spatial slices
            cur_index = cache_index.value
            indices = (0,) * len(batch_dims) + (cur_index, 0, 0)
            key = lax.dynamic_update_slice(cached_key.value, key, indices)
            value = lax.dynamic_update_slice(cached_value.value, value, indices)
            cached_key.value = key
            cached_value.value = value
            num_updated_cache_vectors = query.shape[1]
            cache_index.value = cache_index.value + num_updated_cache_vectors
            # causal mask for cached decoder self-attention: our single query position should only attend to those key positions that have already been generated and cached, not the remaining zero elements.
            pad_mask = jnp.broadcast_to(
                jnp.arange(max_length) < cur_index + num_updated_cache_vectors,
                tuple(batch_dims) + (1, num_updated_cache_vectors, max_length),
            )
            attention_mask = combine_masks(pad_mask, attention_mask)
        return key, value, attention_mask

    def __call__(
        self,
        hidden_states: jnp.ndarray,
        attention_mask: Optional[jnp.ndarray] = None,
        position_ids: Optional[jnp.ndarray] = None,
        past_key_value: Optional[Tuple[jnp.ndarray]] = None,
        output_attentions: bool = False,
        init_cache: bool = False,
        padding_mask: Optional[jnp.ndarray] = None,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        bsz, q_len, _ = hidden_states.shape
        
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        
        # query_shape = (bsz, q_len, self.num_heads, self.head_dim)
        # kv_shape = (bsz, q_len, self.num_key_value_heads, self.head_dim)
        query_states = self._split_heads(query_states, self.num_heads)
        key_states = self._split_heads(key_states, self.num_key_value_heads)
        value_states = self._split_heads(value_states, self.num_key_value_heads)
        
        key_states, query_states = self.rotary_emb(key_states, query_states, position_ids)
        # query_states = jax.lax.reshape(query_states, query_shape).transpose(0, 2, 1, 3)
        # key_states = jax.lax.reshape(key_states, kv_shape).transpose(0, 2, 1, 3)
        # value_states = jax.lax.reshape(value_states, kv_shape).transpose(0, 2, 1, 3)

        # kv_seq_len = key_states.shape[1]

        # cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        # query_states, key_states = flax_apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)
        query_length, key_length = query_states.shape[1], key_states.shape[1]
        if self.has_variable("cache", "cached_key"):
            mask_shift = self.variables["cache"]["cache_index"]
            max_decoder_length = self.variables["cache"]["cached_key"].shape[1]
            causal_mask = lax.dynamic_slice(
                self.causal_mask, (0, 0, mask_shift, 0), (1, 1, query_length, max_decoder_length)
            )
        else:
            causal_mask = self.causal_mask[:, :, :query_length, :key_length]



        batch_size = hidden_states.shape[0]
        causal_mask = jnp.broadcast_to(causal_mask, (batch_size,) + causal_mask.shape[1:])
        attention_mask = jnp.broadcast_to(jnp.expand_dims(attention_mask, axis=(-3, -2)), causal_mask.shape)
        attention_mask = combine_masks(attention_mask, causal_mask)
        
        if self.has_variable("cache", "cached_key") or init_cache:
            key_states, value_states, attention_mask = self._concatenate_to_cache(key_states, value_states, query_states, attention_mask)
            
        key_states = flax_repeat_kv(key_states, self.num_key_value_groups)
        value_states = flax_repeat_kv(value_states, self.num_key_value_groups)
            
        attention_bias = lax.select(
            attention_mask > 0,
            jnp.full(attention_mask.shape, 0.0).astype(self.dtype),
            jnp.full(attention_mask.shape, jnp.finfo(self.dtype).min).astype(self.dtype),
        )

        # usual dot product attention
        attention_dtype = jnp.float32 if self.attention_softmax_in_fp32 else self.dtype
        attn_weights = dot_product_attention_weights(
            query_states,
            key_states,
            bias=attention_bias,
            deterministic=True,
            dtype=attention_dtype,
        )

        if self.attention_softmax_in_fp32:
            attn_weights = attn_weights.astype(self.dtype)

        # attn_weights = query_states @ key_states.transpose(0, 1, 3, 2) / math.sqrt(self.head_dim)

        # if attn_weights.shape != (bsz, self.num_heads, q_len, kv_seq_len):
        #     raise ValueError(
        #         f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
        #         f" {attn_weights.size()}"
        #     )
        # if attention_mask is not None:
        #     if attention_mask.shape != (bsz, 1, q_len, kv_seq_len):
        #         raise ValueError(
        #             f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.shape}"
        #         )
        
        attn_output = jnp.einsum("...hqk,...khd->...qhd", attn_weights, value_states)
        attn_output = self._merge_heads(attn_output)
        attn_output = self.o_proj(attn_output)

        outputs = (attn_output, attn_weights) if output_attentions else (attn_output,)
        return outputs


class FlaxMistralDecoderLayer(nn.Module):

    config: MistralConfig
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        config = self.config
        self.hidden_size = config.hidden_size
        self.self_attn = FlaxMistralAttention(config=config, dtype=self.dtype)
        self.mlp = FlaxMistralMLP(config, dtype=self.dtype)
        self.input_layernorm = FlaxMistralRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps, dtype=self.dtype
        )
        self.post_attention_layernorm = FlaxMistralRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps, dtype=self.dtype
        )

    def __call__(
        self,
        hidden_states: jnp.ndarray,
        attention_mask: Optional[jnp.ndarray] = None,
        position_ids: Optional[jnp.ndarray] = None,
        past_key_value: Optional[Tuple[jnp.ndarray]] = None,
        output_attentions: Optional[bool] = False,
        init_cache: Optional[bool] = False,
        padding_mask: Optional[jnp.ndarray] = None,
    ) -> Tuple[jnp.ndarray, Optional[Tuple[jnp.ndarray, jnp.ndarray]]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            init_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
        """

        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        outputs = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            init_cache=init_cache,
            padding_mask=padding_mask,
        )
        # residual connection
        attn_output = outputs[0]
        hidden_states = residual + attn_output

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states


        return (hidden_states,) + outputs[1:]


class FlaxMistralPreTrainedModel(FlaxPreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = MistralConfig
    base_model_prefix = "model"
    module_class: nn.Module = None

    def __init__(
        self,
        config: MistralConfig,
        input_shape: Tuple = (1, 5),
        seed: int = 0,
        dtype: jnp.dtype = jnp.float32,
        _do_init: bool = True,
        **kwargs,
    ):
        module = self.module_class(
            config=config,
            dtype=dtype,
            **kwargs,
        )
        super().__init__(config, module, input_shape=input_shape, seed=seed, dtype=dtype, _do_init=_do_init)

    def init_weights(self, rng: jax.random.PRNGKey, input_shape: Tuple, params: FrozenDict = None) -> FrozenDict:
        # init input tensors
        input_ids = jnp.zeros(input_shape, dtype="i4")
        attention_mask = jnp.ones_like(input_ids)
        position_ids = jnp.broadcast_to(jnp.arange(jnp.atleast_2d(input_ids).shape[-1]), input_shape)
        params_rng, dropout_rng = jax.random.split(rng)
        rngs = {"params": params_rng, "dropout": dropout_rng}

        module_init_outputs = self.module.init(rngs, input_ids, attention_mask, position_ids, return_dict=False)

        random_params = module_init_outputs["params"]

        if params is not None:
            random_params = flatten_dict(unfreeze(random_params))
            params = flatten_dict(unfreeze(params))
            for missing_key in self._missing_keys:
                params[missing_key] = random_params[missing_key]
            self._missing_keys = set()
            return freeze(unflatten_dict(params))
        else:
            return freeze(random_params)

    # Copied from transformers.models.bart.modeling_flax_bart.FlaxBartDecoderPreTrainedModel.init_cache
    def init_cache(self, batch_size, max_length):
        r"""
        Args:
            batch_size (`int`):
                batch_size used for fast auto-regressive decoding. Defines the batch size of the initialized cache.
            max_length (`int`):
                maximum possible length for auto-regressive decoding. Defines the sequence length of the initialized
                cache.
        """
        # init input variables to retrieve cache
        input_ids = jnp.ones((batch_size, max_length), dtype="i4")
        attention_mask = jnp.ones_like(input_ids, dtype="i4")
        position_ids = jnp.broadcast_to(jnp.arange(jnp.atleast_2d(input_ids).shape[-1]), input_ids.shape)

        init_variables = self.module.init(
            jax.random.PRNGKey(0), input_ids, attention_mask, position_ids, return_dict=True, init_cache=True
        )
        return unfreeze(init_variables["cache"])

    # @add_start_docstrings_to_model_forward(BERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    def __call__(
        self,
        input_ids,
        attention_mask=None,
        position_ids=None,
        init_cache: bool = False,
        params: dict = None,
        train: bool = False,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        past_key_values: dict = None,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.return_dict

        if position_ids is None:
            position_ids = jnp.broadcast_to(jnp.arange(jnp.atleast_2d(input_ids).shape[-1]), input_ids.shape)

        if attention_mask is None:
            attention_mask = jnp.ones_like(input_ids)

        # Handle any PRNG if needed
        rngs = {}
        inputs = {"params": params or self.params}
        
        if past_key_values:
            inputs["cache"] = past_key_values
            mutable = ["cache"]
        else:
            mutable = False


        outputs = self.module.apply(
            inputs,
            jnp.array(input_ids, dtype="i4"),
            jnp.array(attention_mask, dtype="i4"),
            position_ids=jnp.array(position_ids, dtype="i4"),
            init_cache=init_cache,
            past_key_values=past_key_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            rngs=rngs,
            mutable=mutable,
        )
        
        if past_key_values is not None and return_dict:
            outputs, past_key_values = outputs
            outputs["past_key_values"] = unfreeze(past_key_values["cache"])
            return outputs
        elif past_key_values is not None and not return_dict:
            outputs, past_key_values = outputs
            outputs = outputs[:1] + (unfreeze(past_key_values["cache"]),) + outputs[1:]

        return outputs
    

class FlaxMistralLayerCollection(nn.Module):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`MistralDecoderLayer`]

    Args:
        config: MistralConfig
    """
    
    config: MistralConfig
    dtype: jnp.dtype = jnp.float32
    
    def setup(self):
        self.blocks = [
            FlaxMistralDecoderLayer(self.config, dtype=self.dtype, name=str(i)) for i in range(self.config.num_hidden_layers)
        ]
        
    def __call__(self,
                 hidden_states: jnp.ndarray = None,
                 attention_mask: Optional[jnp.ndarray] = None,
                 position_ids: Optional[jnp.ndarray] = None,
                 past_key_values: Optional[List[jnp.ndarray]] = None,
                 init_cache: Optional[bool] = None,
                 output_attentions: Optional[bool] = None,
                 output_hidden_states: Optional[bool] = None,
                 return_dict: Optional[bool] = None,) -> Any:
        
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        next_decoder_cache = () if init_cache else None

        for idx, decoder_layer in enumerate(self.blocks):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            # past_key_value = past_key_values[idx] if past_key_values is not None else None
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                output_attentions=output_attentions,
                init_cache=init_cache,
            )

            hidden_states = layer_outputs[0]


            if output_attentions:
                all_attentions += (layer_outputs[1],)
                
        outputs = (hidden_states, all_hidden_states, all_attentions)

        return outputs 



class FlaxMistralModule(nn.Module):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`MistralDecoderLayer`]

    Args:
        config: MistralConfig
    """

    config: MistralConfig
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation

    def setup(self):
        self.padding_idx = self.config.pad_token_id
        self.vocab_size = self.config.vocab_size
        self.embed_tokens = nn.Embed(self.config.vocab_size, self.config.hidden_size, dtype=self.dtype)
        self.layers = FlaxMistralLayerCollection(self.config, dtype=self.dtype)
        self.norm = FlaxMistralRMSNorm(self.config.hidden_size, eps=self.config.rms_norm_eps, dtype=self.dtype)

    def __call__(
        self,
        input_ids: jnp.ndarray = None,
        attention_mask: Optional[jnp.ndarray] = None,
        position_ids: Optional[jnp.ndarray] = None,
        past_key_values: Optional[List[jnp.ndarray]] = None,
        inputs_embeds: Optional[jnp.ndarray] = None,
        init_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, FlaxBaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        seq_length_with_past = seq_length


        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        # # embed positions
        # padding_mask = None
        # if attention_mask is None:
        #     attention_mask = jnp.ones((batch_size, seq_length_with_past), dtype=jnp.bool_)
        # #     padding_mask = None
        # # else:
        # #     if (0 == attention_mask).any():
        # #         padding_mask = attention_mask
        # #     else:
        # #         padding_mask = None

        # attention_mask = _prepare_decoder_attention_mask(
        #     attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length, sliding_window=self.config.sliding_window,
        # )

        hidden_states = inputs_embeds

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = () if init_cache else None
        
        outputs = self.layers(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            init_cache=init_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        
        hidden_states = outputs[0]
        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states = outputs[1] + (hidden_states,)
            outputs = (hidden_states, all_hidden_states) + outputs[2:]
        else:
            outputs = (hidden_states,) + outputs[1:]

        next_cache = outputs[1] if init_cache else None
        if not return_dict:
            return tuple(v for v in outputs if v is not None)
        
        return FlaxBaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=outputs[-1] if output_attentions else None,
        )

@add_start_docstrings(
    "The bare Mistral Model transformer outputting raw hidden-states without any specific head on top.",
    MISTRAL_START_DOCSTRING,
)
class FlaxMistralModel(FlaxMistralPreTrainedModel):
    module_class = FlaxMistralModule
append_call_sample_docstring(FlaxMistralModel, _CHECKPOINT_FOR_DOC, FlaxBaseModelOutputWithPast, _CONFIG_FOR_DOC)

class FlaxMistralForCausalLMModule(nn.Module):
    config: MistralConfig
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation

    def setup(self):
        self.model = FlaxMistralModule(self.config, dtype=self.dtype)
        self.vocab_size = self.config.vocab_size
        self.lm_head = nn.Dense(self.config.vocab_size, use_bias=False, dtype=self.dtype)

    def __call__(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        init_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, FlaxCausalLMOutputWithCrossAttentions]:
        

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            init_cache=init_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]

        logits = self.lm_head(hidden_states)
        
        if not return_dict:
            output = (logits,) + outputs[1:]
            return output
        
        return FlaxCausalLMOutputWithCrossAttentions(
            logits=logits,
            past_key_values=outputs.past_key_values if return_dict else outputs[1],
            hidden_states=outputs.hidden_states if return_dict else outputs[2],
            attentions=outputs.attentions if return_dict else outputs[3],
        )

@add_start_docstrings(
    """
    The Mistral Model transformer with a language modeling head (linear layer) on top.
    """,
    MISTRAL_START_DOCSTRING,
)
class FlaxMistralForCausalLM(FlaxMistralPreTrainedModel):
    module_class = FlaxMistralForCausalLMModule

    def prepare_inputs_for_generation(
        self, input_ids, max_length, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        # initializing the cache
        batch_size, seq_length = input_ids.shape
        if past_key_values is not None:
            past_length = past_key_values[0][0].shape[2]

            # Some generation methods already pass only the last input ID
            if input_ids.shape[1] > past_length:
                remove_prefix_length = past_length
            else:
                # Default to old behavior: keep only final ID
                remove_prefix_length = input_ids.shape[1] - 1

            input_ids = input_ids[:, remove_prefix_length:]
        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.cumsum(-1) - 1
            position_ids = position_ids * attention_mask + (1 - attention_mask)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1] :]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {}

        model_inputs.update(
            {
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "init_cache": kwargs.get("init_cache"),
                "attention_mask": attention_mask,
            }
        )
        return model_inputs

    def update_inputs_for_generation(self, model_outputs, model_kwargs):
        attention_mask = model_kwargs["attention_mask"]
        model_kwargs["attention_mask"] = jnp.concatenate(
            [attention_mask, jnp.ones((attention_mask.shape[0], 1))], axis=1
        )
        model_kwargs["past_key_values"] = model_outputs.past_key_values
        model_kwargs["position_ids"] = model_kwargs["position_ids"][:, -1:] + 1
        return model_kwargs

append_call_sample_docstring(FlaxMistralForCausalLM, _CHECKPOINT_FOR_DOC, FlaxCausalLMOutputWithCrossAttentions, _CONFIG_FOR_DOC)

class FlaxMistralForSequenceClassificationModule(nn.Module):

    config: MistralConfig
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation

    def setup(self):
        self.num_labels = self.config.num_labels
        self.model = FlaxMistralModule(self.config, dtype=self.dtype)
        self.score = nn.Dense(self.num_labels, use_bias=False, dtype=self.dtype)

        # Initialize weights and apply final processing

    def __call__(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        init_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, FlaxSequenceClassifierOutput]:

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            init_cache=init_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]
        logits = self.score(hidden_states)

        if input_ids is not None:
            batch_size = input_ids.shape[0]
        else:
            batch_size = inputs_embeds.shape[0]

        if self.config.pad_token_id is None and batch_size != 1:
            raise ValueError("Cannot handle batch sizes > 1 if no padding token is defined.")
        if self.config.pad_token_id is None:
            sequence_lengths = -1
        else:
            if input_ids is not None:
                sequence_lengths = jax.lax.eq(input_ids, self.config.pad_token_id).argmax(-1) - 1
            else:
                sequence_lengths = -1

        pooled_logits = logits[jnp.arange(batch_size), sequence_lengths]

        if not return_dict:
            output = (pooled_logits,) + transformer_outputs[1:]
            return output

        return FlaxSequenceClassifierOutput(
            logits=pooled_logits,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )

@add_start_docstrings(
    """
    The Mistral Model transformer with a sequence classification  head (linear layer) on top.
    """,
    MISTRAL_START_DOCSTRING,
)
class FlaxMistralForSequenceClassification(FlaxMistralPreTrainedModel):
    module_class = FlaxMistralForSequenceClassificationModule

append_call_sample_docstring(FlaxMistralForSequenceClassification, _CHECKPOINT_FOR_DOC, FlaxSequenceClassifierOutput, _CONFIG_FOR_DOC)