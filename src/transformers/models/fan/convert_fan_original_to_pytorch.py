# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team.
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
"""Convert Fan checkpoints."""


import functools
import re

from transformers.utils import logging


logging.set_verbosity_info()
logger = logging.get_logger(__name__)


def replace_linear_keys(key):
    match = re.match(r"linear_c\d{1}", key)
    if match:
        start, end = match.span()
        output = key[start : end - 1] + "." + str(int(key[end - 1]) - 1) + key[end:]
        return output
    return key


def fix_linear_fuse(key):
    match_conv = re.match(r"linear_fuse\.conv", key)
    match_bn = re.match(r"linear_fuse\.bn", key)
    match_clf = re.match(r"linear_pred", key)
    if match_conv:
        return key.replace("linear_fuse.conv", "linear_fuse")
    if match_bn:
        return key.replace("linear_fuse.bn", "batch_norm")
    if match_clf:
        return key.replace("linear_pred", "classifier")
    return key


def compose(*functions):
    return functools.reduce(lambda f, g: lambda x: f(g(x)), functions, lambda x: x)


def remap_patch_embed(key):
    return key.replace("patch_embed", "patch_embeddings")


def remap_embeddings(key):
    if "embed" in key:
        return f"fan.embeddings.{key}"
    return key


def remap_gamma(key):
    return key.replace("gamma", "weight")


def remap_head(key):
    if key.split(".")[0] in ("norm", "head"):
        return f"head.{key}"
    return key


def remap_encoder(key):
    if any(x in key for x in ["fan", "head"]):
        return key
    return f"fan.encoder.{key}"


def remap_blocks(key):
    pattern = "([a-z\.]*blocks\.\d*\.)"
    if re.match(pattern, key):
        return re.sub(pattern, "\\1block.", key)
    return key


def remap_proj_keys(key):
    pattern = "([a-z\.]*patch_embed\.proj\.\d*\.)"
    if re.match(pattern, key):
        stem = ".".join(key.split(".")[:-3])
        first = int(key.split(".")[-3])
        second = int(key.split(".")[-2])
        name = key.split(".")[-1]
        return f"{stem}.{first + first//2 + second}.{name}"
    return key


def remap_segmentation_linear(key):
    if "decode_head.linear_fuse.conv" in key:
        return key.replace("decode_head.linear_fuse.conv", "decode_head.linear_fuse")
    if "decode_head.linear_fuse.bn" in key:
        return key.replace("decode_head.linear_fuse.bn", "decode_head.batch_norm")
    if "decode_head.linear_pred" in key:
        return key.replace("decode_head.linear_pred", "decode_head.classifier")
    return key


def remap_linear_fuse(key):
    for num in range(4):
        if f"decode_head.linear_c{num+1}" in key:
            return key.replace(f"decode_head.linear_c{num+1}", f"decode_head.linear_c.{num}")
    return key


def remap_qkv(key):
    elements = key.split(".")
    mapping_dict = {"q": "query", "v": "value", "k": "key", "kv": "key_value"}
    return ".".join([mapping_dict.get(elem, elem) for elem in elements])


remap_fn = compose(
    remap_segmentation_linear,
    remap_linear_fuse,
    remap_blocks,
    remap_encoder,
    remap_gamma,
    remap_head,
    remap_embeddings,
    remap_patch_embed,
    remap_proj_keys,
    remap_qkv,
)


def remap_state(state_dict):
    return {remap_fn(key): weights for key, weights in state_dict.items()}
