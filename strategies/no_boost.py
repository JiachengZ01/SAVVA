#!/usr/bin/env python3
"""
No Boost Strategy (Baseline)

This strategy does nothing - returns attention weights unchanged.
Used as a baseline for comparison.

Author: AdaVBoost Project
"""

import torch
from .base_strategy import BaseBoostStrategy


class NoBoostStrategy(BaseBoostStrategy):
    """
    Baseline strategy that applies no boost.

    Simply returns the attention weights unchanged.
    This is the true baseline for comparison.
    """

    def __init__(self):
        super().__init__(name="no_boost")
        # Always disabled - never modifies attention
        self._enabled = False

    def configure(self, **kwargs):
        """No configuration needed."""
        return self

    def enable(self):
        """Cannot be enabled - always returns original attention."""
        pass

    def disable(self):
        """Already disabled."""
        pass

    def apply(
        self,
        attn_weights: torch.Tensor,
        visual_indices: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:
        """Return attention weights unchanged."""
        return attn_weights

    def __repr__(self):
        return "NoBoostStrategy(baseline)"
