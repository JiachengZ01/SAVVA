"""
SAVVA: Mitigating Hallucinations in LVLMs via Step-wise Adaptive Visual Attention Amplification

Key features:
1. VGE (Visual Grounding Entropy) as risk signal
2. Adaptive visual token boosting based on hallucination risk
3. Text token suppression for input tokens
4. Model-specific layer config

Usage:
    from strategies.ours import SAVVAStrategy, RiskLogitsProcessor

    # Create strategy for specific model
    strategy = SAVVAStrategy(model_name='llava')
    strategy.enable()

    # Create logits processor for generation
    processor = RiskLogitsProcessor(strategy)

    # In model.generate():
    outputs = model.generate(..., logits_processor=[processor])
"""

from .savva import SAVVAStrategy, load_savva_config, get_model_config
from .logits_processor import RiskLogitsProcessor
from .vge import compute_vge, compute_grounding_scores, RiskEstimator

__all__ = [
    'SAVVAStrategy',
    'RiskLogitsProcessor',
    'compute_vge',
    'compute_grounding_scores',
    'RiskEstimator',
    'load_savva_config',
    'get_model_config',
]
