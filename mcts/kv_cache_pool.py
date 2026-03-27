"""KV cache management for MCTS tree branches.

Manages sharing and lifecycle of KV caches across tree nodes to minimize
memory usage during tree search.
"""

import torch
from typing import Optional, Tuple


def clone_kv_cache(kv_cache: Optional[Tuple]) -> Optional[Tuple]:
    """Deep clone a KV cache so modifications don't affect the original."""
    if kv_cache is None:
        return None
    return tuple(
        (k.clone(), v.clone()) for k, v in kv_cache
    )


def trim_kv_cache(kv_cache: Optional[Tuple], seq_len: int) -> Optional[Tuple]:
    """Trim KV cache to a specific sequence length."""
    if kv_cache is None:
        return None
    return tuple(
        (k[:, :, :seq_len, :], v[:, :, :seq_len, :]) for k, v in kv_cache
    )


def get_kv_cache_memory_bytes(kv_cache: Optional[Tuple]) -> int:
    """Estimate memory usage of a KV cache in bytes."""
    if kv_cache is None:
        return 0
    total = 0
    for k, v in kv_cache:
        total += k.nelement() * k.element_size()
        total += v.nelement() * v.element_size()
    return total


class KVCachePool:
    """Manages KV caches across MCTS tree to stay within memory budget."""

    def __init__(self, max_memory_bytes: int = 6 * 1024 * 1024 * 1024):  # 6GB default
        self.max_memory_bytes = max_memory_bytes
        self._current_usage = 0

    @property
    def usage_bytes(self) -> int:
        return self._current_usage

    @property
    def usage_mb(self) -> float:
        return self._current_usage / (1024 * 1024)

    def can_allocate(self, kv_cache: Optional[Tuple]) -> bool:
        """Check if we have room for this cache."""
        needed = get_kv_cache_memory_bytes(kv_cache)
        return (self._current_usage + needed) <= self.max_memory_bytes

    def register(self, kv_cache: Optional[Tuple]):
        """Track a newly stored KV cache."""
        self._current_usage += get_kv_cache_memory_bytes(kv_cache)

    def release(self, kv_cache: Optional[Tuple]):
        """Release a KV cache from tracking."""
        freed = get_kv_cache_memory_bytes(kv_cache)
        self._current_usage = max(0, self._current_usage - freed)
