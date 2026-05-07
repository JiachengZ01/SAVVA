"""
SAVVA: Mitigating Hallucinations in LVLMs via Step-wise Adaptive Visual Attention Amplification

Attention modification strategies for reducing hallucination in VLMs.

Main components:
- SAVVAStrategy: Our method with VGE-based risk estimation
- NoBoostStrategy: Baseline without modification
"""

from .base_strategy import BaseBoostStrategy
from .no_boost import NoBoostStrategy

# SAVVA
from .ours import (
    SAVVAStrategy,
    RiskLogitsProcessor,
    compute_vge,
    compute_grounding_scores,
    RiskEstimator,
    load_savva_config,
    get_model_config,
)

__all__ = [
    # Base
    'BaseBoostStrategy',
    'NoBoostStrategy',
    # SAVVA
    'SAVVAStrategy',
    'RiskLogitsProcessor',
    'compute_vge',
    'compute_grounding_scores',
    'RiskEstimator',
    'load_savva_config',
    'get_model_config',
]
