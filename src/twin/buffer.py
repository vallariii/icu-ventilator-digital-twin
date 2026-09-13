"""Circular buffer for rolling-window sensor storage (PDF Section 8)."""
from __future__ import annotations
import numpy as np


class CircularBuffer:
    """Fixed-size ring buffer of multi-channel samples.

    Append is O(1). `window()` returns an in-order numpy view of the last
    min(count, size) samples, shape (count, n_channels).
    """

    __slots__ = ('size', 'n_channels', '_buf', '_idx', '_filled')

    def __init__(self, size: int, n_channels: int = 1):
        self.size = size
        self.n_channels = n_channels
        self._buf = np.zeros((size, n_channels), dtype=np.float32)
        self._idx = 0
        self._filled = 0

    def append(self, sample):
        self._buf[self._idx] = sample
        self._idx = (self._idx + 1) % self.size
        if self._filled < self.size:
            self._filled += 1

    def window(self, n: int | None = None) -> np.ndarray:
        """Return the last `n` samples in chronological order."""
        n = self._filled if n is None else min(n, self._filled)
        if n == 0:
            return self._buf[:0]
        if self._filled < self.size:
            return self._buf[:self._filled][-n:]
        # Full buffer — unwrap the ring
        start = self._idx
        return np.concatenate([self._buf[start:], self._buf[:start]])[-n:]

    def __len__(self):
        return self._filled
