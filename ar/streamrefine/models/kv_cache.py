"""Tensor-derived attention caches with no external geometry constants."""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any
import torch


@dataclass(frozen=True)
class KVCacheEntry:
    """Projected attention keys/values in ``[B,H,N,D]`` convention."""

    key: torch.Tensor
    value: torch.Tensor
    valid_mask: torch.Tensor | None = None

    def __post_init__(self) -> None:
        if self.key.ndim != 4 or self.value.ndim != 4:
            raise ValueError(
                f"key/value must be [B,H,N,D], got {tuple(self.key.shape)} and {tuple(self.value.shape)}"
            )
        if self.key.shape != self.value.shape:
            raise ValueError(
                f"key/value shapes must match, got {tuple(self.key.shape)} and {tuple(self.value.shape)}"
            )
        if self.valid_mask is not None and self.valid_mask.shape != (
            self.key.shape[0],
            self.key.shape[2],
        ):
            raise ValueError(
                f"valid_mask must be [B,N] matching key/value, got {tuple(self.valid_mask.shape)}"
            )

    @property
    def num_tokens(self) -> int:
        return int(self.key.shape[2])

    def detached(self) -> "KVCacheEntry":
        return KVCacheEntry(
            key=self.key.detach(),
            value=self.value.detach(),
            valid_mask=None if self.valid_mask is None else self.valid_mask.detach(),
        )


class LayerKVCache:
    """Layer-indexed static cache, primarily for per-window source K/V."""

    def __init__(self, num_layers: int, *, identity: Any = None) -> None:
        num_layers = int(num_layers)
        if num_layers <= 0:
            raise ValueError(f"num_layers must be positive, got {num_layers}")
        self._entries: list[KVCacheEntry | None] = [None] * num_layers
        self.identity = identity

    def __len__(self) -> int:
        return len(self._entries)

    def _validate_layer(self, layer: int) -> int:
        layer = int(layer)
        if not 0 <= layer < len(self._entries):
            raise IndexError(f"layer {layer} is outside [0, {len(self._entries)})")
        return layer

    def get(self, layer: int) -> KVCacheEntry | None:
        return self._entries[self._validate_layer(layer)]

    def set(self, layer: int, entry: KVCacheEntry, *, detach: bool = True) -> None:
        layer = self._validate_layer(layer)
        self._entries[layer] = entry.detached() if detach else entry

    def clear(self) -> None:
        for layer in range(len(self._entries)):
            self._entries[layer] = None

    @property
    def complete(self) -> bool:
        return all((entry is not None for entry in self._entries))

    def metadata(self) -> dict[str, Any]:
        tokens = [
            None if entry is None else entry.num_tokens for entry in self._entries
        ]
        return {
            "num_layers": len(self._entries),
            "populated_layers": sum((entry is not None for entry in self._entries)),
            "tokens_per_layer": tokens,
            "identity_bound": self.identity is not None,
        }


class AppendOnlyLayerKVCache(LayerKVCache):
    """Append complete target states while deriving capacity from model geometry.

    ``append`` remains available for low-level cache tests.  Model code should use
    :meth:`append_state`, which validates every layer before mutating the cache so a
    failed commit cannot leave a partially appended refinement state.
    """

    def __init__(
        self,
        num_layers: int,
        *,
        max_tokens: int | None = None,
        tokens_per_state: int | None = None,
        max_states: int | None = None,
        identity: Any = None,
    ) -> None:
        super().__init__(num_layers, identity=identity)
        if max_tokens is not None and int(max_tokens) <= 0:
            raise ValueError(f"max_tokens must be positive or None, got {max_tokens}")
        if tokens_per_state is not None and int(tokens_per_state) <= 0:
            raise ValueError(
                f"tokens_per_state must be positive or None, got {tokens_per_state}"
            )
        if max_states is not None and int(max_states) <= 0:
            raise ValueError(f"max_states must be positive or None, got {max_states}")
        self.max_tokens = None if max_tokens is None else int(max_tokens)
        self.tokens_per_state = (
            None if tokens_per_state is None else int(tokens_per_state)
        )
        self.max_states = None if max_states is None else int(max_states)
        if self.tokens_per_state is not None and self.max_states is not None:
            derived_max_tokens = self.tokens_per_state * self.max_states
            if self.max_tokens is None:
                self.max_tokens = derived_max_tokens
            elif self.max_tokens != derived_max_tokens:
                raise ValueError(
                    "max_tokens must equal tokens_per_state * max_states when all three values are provided"
                )

    def append(
        self, layer: int, entry: KVCacheEntry, *, detach: bool = True
    ) -> KVCacheEntry:
        layer = self._validate_layer(layer)
        previous = self._entries[layer]
        incoming = entry.detached() if detach else entry
        if previous is None:
            combined = incoming
        else:
            if (
                previous.key.shape[:2] + previous.key.shape[3:]
                != incoming.key.shape[:2] + incoming.key.shape[3:]
            ):
                raise ValueError(
                    f"cannot append K/V with different batch, head, or head dimensions: {tuple(previous.key.shape)} vs {tuple(incoming.key.shape)}"
                )
            key = torch.cat([previous.key, incoming.key], dim=2)
            value = torch.cat([previous.value, incoming.value], dim=2)
            if previous.valid_mask is None and incoming.valid_mask is None:
                valid_mask = None
            elif previous.valid_mask is not None and incoming.valid_mask is not None:
                valid_mask = torch.cat(
                    [previous.valid_mask, incoming.valid_mask], dim=1
                )
            else:
                raise ValueError(
                    "all appended K/V entries must agree on whether valid_mask is present"
                )
            combined = KVCacheEntry(key=key, value=value, valid_mask=valid_mask)
        if self.max_tokens is not None and combined.num_tokens > self.max_tokens:
            raise ValueError(
                f"KV cache overflow: {combined.num_tokens} tokens exceed configured {self.max_tokens}"
            )
        self._entries[layer] = combined
        return combined

    @property
    def num_states(self) -> int | None:
        """Return the number of atomically committed states when geometry is known."""
        if self.tokens_per_state is None:
            return None
        populated = [entry for entry in self._entries if entry is not None]
        if not populated:
            return 0
        if len(populated) != len(self._entries):
            raise RuntimeError("target-history cache contains a partial state commit")
        token_counts = {entry.num_tokens for entry in populated}
        if len(token_counts) != 1:
            raise RuntimeError(
                "target-history cache layers contain different token counts"
            )
        token_count = token_counts.pop()
        if token_count % self.tokens_per_state:
            raise RuntimeError(
                f"target-history cache has {token_count} tokens, which is not divisible by tokens_per_state={self.tokens_per_state}"
            )
        return token_count // self.tokens_per_state

    def append_state(
        self,
        entries: list[KVCacheEntry] | tuple[KVCacheEntry, ...],
        *,
        detach: bool = True,
    ) -> None:
        """Atomically append one complete state's projected K/V to every layer."""
        if len(entries) != len(self._entries):
            raise ValueError(
                f"one target K/V entry is required for each of {len(self._entries)} layers"
            )
        incoming = [entry.detached() if detach else entry for entry in entries]
        incoming_counts = {entry.num_tokens for entry in incoming}
        if len(incoming_counts) != 1:
            raise ValueError(
                "all layers of one target state must contain the same token count"
            )
        incoming_tokens = incoming_counts.pop()
        if (
            self.tokens_per_state is not None
            and incoming_tokens != self.tokens_per_state
        ):
            raise ValueError(
                f"target state contains {incoming_tokens} tokens, expected tokens_per_state={self.tokens_per_state}"
            )
        current_states = self.num_states
        if (
            current_states is not None
            and self.max_states is not None
            and (current_states >= self.max_states)
        ):
            raise ValueError(
                f"target-history cache already contains configured max_states={self.max_states}"
            )
        combined_entries: list[KVCacheEntry] = []
        for previous, new_entry in zip(self._entries, incoming):
            if previous is None:
                combined = new_entry
            else:
                if (
                    previous.key.shape[:2] + previous.key.shape[3:]
                    != new_entry.key.shape[:2] + new_entry.key.shape[3:]
                ):
                    raise ValueError(
                        f"cannot append target K/V with different batch, head, or head dimensions: {tuple(previous.key.shape)} vs {tuple(new_entry.key.shape)}"
                    )
                if previous.key.device != new_entry.key.device:
                    raise ValueError(
                        "all target-history K/V entries must share a device"
                    )
                if previous.key.dtype != new_entry.key.dtype:
                    raise ValueError(
                        "all target-history K/V entries must share a dtype"
                    )
                key = torch.cat([previous.key, new_entry.key], dim=2)
                value = torch.cat([previous.value, new_entry.value], dim=2)
                if previous.valid_mask is None and new_entry.valid_mask is None:
                    valid_mask = None
                elif (
                    previous.valid_mask is not None and new_entry.valid_mask is not None
                ):
                    valid_mask = torch.cat(
                        [previous.valid_mask, new_entry.valid_mask], dim=1
                    )
                else:
                    raise ValueError(
                        "all target-history K/V entries must agree on valid_mask presence"
                    )
                combined = KVCacheEntry(key=key, value=value, valid_mask=valid_mask)
            if self.max_tokens is not None and combined.num_tokens > self.max_tokens:
                raise ValueError(
                    f"KV cache overflow: {combined.num_tokens} tokens exceed configured {self.max_tokens}"
                )
            combined_entries.append(combined)
        self._entries[:] = combined_entries

    def metadata(self) -> dict[str, Any]:
        metadata = super().metadata()
        metadata.update(
            {
                "cache_type": "target_history",
                "tokens_per_state": self.tokens_per_state,
                "max_tokens": self.max_tokens,
                "max_states": self.max_states,
                "num_states": self.num_states,
            }
        )
        return metadata


TargetHistoryKVCache = AppendOnlyLayerKVCache
__all__ = [
    "AppendOnlyLayerKVCache",
    "KVCacheEntry",
    "LayerKVCache",
    "TargetHistoryKVCache",
]
