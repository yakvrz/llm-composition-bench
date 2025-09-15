#!/usr/bin/env python3
"""
Shared synthetic graph utilities for the n-hop benchmark generators.

Functions:
- layer_prefix(i): letter prefix for layer i
- make_layer(prefix, size, id_width): tokens for a layer with configurable zero-padded width
- permute_map(src, dst): random bijection between same-sized lists
- build_bijections(hops, size_M, seed, id_width): layers and hop functions f1..fn
- compose_chain(x, functions): follow chain, returning intermediates and final target
"""
from typing import Dict, List, Tuple
import random


def layer_prefix(i: int) -> str:
    return chr(ord('A') + i)


def make_layer(prefix: str, size: int, id_width: int = 4) -> List[str]:
    fmt = f"{{:0{id_width}d}}"
    return [f"{prefix}_{fmt.format(j)}" for j in range(size)]


def permute_map(src: List[str], dst: List[str]) -> Dict[str, str]:
    shuffled = dst[:]
    random.shuffle(shuffled)
    return dict(zip(src, shuffled))


def build_bijections(hops: int, size_M: int, seed: int, id_width: int = 4) -> Tuple[List[List[str]], List[Dict[str, str]]]:
    random.seed(seed)
    layers: List[List[str]] = [make_layer(layer_prefix(i), size_M, id_width=id_width) for i in range(hops + 1)]
    functions: List[Dict[str, str]] = []
    for i in range(1, hops + 1):
        src = layers[i - 1]
        dst = layers[i]
        functions.append(permute_map(src, dst))
    return layers, functions


def compose_chain(x: str, functions: List[Dict[str, str]]) -> Tuple[List[str], str]:
    intermediates: List[str] = []
    cur = x
    for f in functions:
        cur = f[cur]
        intermediates.append(cur)
    return intermediates[:-1], intermediates[-1]



