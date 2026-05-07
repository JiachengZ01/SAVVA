#!/usr/bin/env python3
"""
Base Strategy for Attention Modification

Defines the abstract interface and basic implementations for attention boost strategies.

Author: SAVVA Project
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
import torch


class BaseBoostStrategy(ABC):
    """
    Abstract base class for attention boost strategies.

    All strategies must implement the `apply` method which modifies
    attention weights in-place or returns modified weights.
    """

    def __init__(self, name: str = "base"):
        self.name = name
        self._enabled = False
        self._config = {}

    @abstractmethod
    def apply(
        self,
        attn_weights: torch.Tensor,
        visual_indices: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:
        """
        Apply the boost strategy to attention weights.

        Args:
            attn_weights: Pre-softmax attention weights [batch, heads, q_len, kv_len]
            visual_indices: Tensor of visual token indices to boost
            **kwargs: Strategy-specific parameters

        Returns:
            Modified attention weights (same shape as input)
        """
        pass

    def configure(self, **kwargs):
        """Configure the strategy parameters."""
        self._config.update(kwargs)
        return self

    def enable(self):
        """Enable the strategy."""
        self._enabled = True
        return self

    def disable(self):
        """Disable the strategy."""
        self._enabled = False
        return self

    @property
    def enabled(self) -> bool:
        return self._enabled

    def get_config(self) -> Dict[str, Any]:
        """Get current configuration."""
        return self._config.copy()

    def __repr__(self):
        return f"{self.__class__.__name__}(name={self.name}, enabled={self._enabled})"


