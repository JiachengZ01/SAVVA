"""
AdaVBoost: Mitigating Hallucinations in LVLMs via Token-Level Adaptive Visual Attention Boosting

Attention modification strategies for reducing hallucination in VLMs.

Main components:
- AdaVBoostStrategy: Our method with VGE-based risk estimation
- NoBoostStrategy: Baseline without modification
"""

from .base_strategy import BaseBoostStrategy
from .no_boost import NoBoostStrategy

# AdaVBoost
from .ours import (
    AdaVBoostStrategy,
    RiskLogitsProcessor,
    compute_vge,
    compute_grounding_scores,
    RiskEstimator,
    load_adavboost_config,
    get_model_config,
)

__all__ = [
    # Base
    'BaseBoostStrategy',
    'NoBoostStrategy',
    # AdaVBoost
    'AdaVBoostStrategy',
    'RiskLogitsProcessor',
    'compute_vge',
    'compute_grounding_scores',
    'RiskEstimator',
    'load_adavboost_config',
    'get_model_config',
]
