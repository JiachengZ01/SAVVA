"""
AdaVBoost Strategy

Model-specific config loaded from configs/ours.yaml
"""

import os
import yaml
import torch
from typing import Optional, Dict, Any
from ..base_strategy import BaseBoostStrategy
from .vge import (
    RiskEstimator,
    compute_vge,
    compute_grounding_scores,
)


# =============================================================================
# Configuration Loading
# =============================================================================

def _get_config_path():
    """Get path to configs/ours.yaml."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(current_dir, '..', '..', 'configs', 'ours.yaml')
    return os.path.normpath(config_path)


def load_adavboost_config(config_path: str = None) -> Dict:
    """Load AdaVBoost configuration from YAML file."""
    if config_path is None:
        config_path = _get_config_path()

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def get_model_config(model_name: str, config: Dict = None) -> Dict:
    """Get model-specific configuration."""
    if config is None:
        config = load_adavboost_config()

    if model_name not in config['models']:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(config['models'].keys())}")

    model_cfg = config['models'][model_name]

    return {
        'start_layer': model_cfg['layers']['start'],
        'end_layer': model_cfg['layers']['end'],
        'm_max_visual': model_cfg['m']['max_visual'],
        'm_max_text': model_cfg['m']['max_text'],
        'vge_alpha': model_cfg['vge']['alpha'],
        'risk_scale': model_cfg['vge']['risk_scale'],
    }


class AdaVBoostStrategy(BaseBoostStrategy):
    """
    AdaVBoost: Mitigating Hallucinations in LVLMs via Token-Level Adaptive Visual Attention Boosting.

    Uses VGE (Visual Grounding Entropy) for risk estimation and applies
    adaptive attention reweighting to reduce hallucination.

    VGE = alpha * entropy + (1 - alpha) * (1 - G)
    where G is visual grounding score.

    Key parameters:
    - m_max_visual: Max multiplier for visual token boosting (visual tokens × m_t)
    - m_max_text: Max multiplier for text token suppression (text tokens × 1/m_max_text)

    Behavior:
    - Visual tokens are boosted based on risk: m_t = 1 + (m_max_visual - 1) * risk
    - Input text tokens (before image and after image, excluding output) are suppressed by 1/m_max_text
    """

    def __init__(
        self,
        model_name: str = 'llava',
        config_path: Optional[str] = None,
        risk_scale: Optional[float] = None,
        m_max_visual: Optional[float] = None,
        m_max_text: Optional[float] = None,
        start_layer: Optional[int] = None,
        end_layer: Optional[int] = None,
        vge_alpha: Optional[float] = None,
    ):
        super().__init__(name="adavboost")

        self.model_name = model_name

        cfg = get_model_config(model_name, load_adavboost_config(config_path) if config_path else None)

        self.start_layer = start_layer if start_layer is not None else cfg['start_layer']
        self.end_layer = end_layer if end_layer is not None else cfg['end_layer']

        risk_scale = risk_scale if risk_scale is not None else cfg['risk_scale']
        vge_alpha = vge_alpha if vge_alpha is not None else cfg['vge_alpha']
        m_max_visual = m_max_visual if m_max_visual is not None else cfg['m_max_visual']
        m_max_text = m_max_text if m_max_text is not None else cfg['m_max_text']

        self.vge_alpha = vge_alpha
        self.params = {
            'model_name': model_name,
            'risk_scale': risk_scale,
            'vge_alpha': vge_alpha,
            'm_max_visual': m_max_visual,
            'm_max_text': m_max_text,
            'start_layer': self.start_layer,
            'end_layer': self.end_layer,
        }

        self.risk_estimator = RiskEstimator(scale=risk_scale)
        self.m_max_visual = m_max_visual
        self.m_max_text = m_max_text

        self._current_risk = 0.0
        self._current_vge = 0.0
        self._step_count = 0
        self._num_visual_tokens = None
        self._prompt_length = None
        self._grounding_scores = None

    def update_risk_from_logits(self, logits: torch.Tensor):
        """Update risk from logits (call this each generation step)."""
        self._current_vge = compute_vge(
            logits,
            grounding_scores=self._grounding_scores,
            alpha=self.vge_alpha,
        )
        self._current_risk = self.risk_estimator.compute_risk(self._current_vge)

    def set_risk(self, risk: float):
        """Manually set risk (for compatibility)."""
        self._current_risk = max(0.0, min(1.0, risk))

    def set_grounding_scores(
        self,
        prefill_logits: torch.Tensor,
        image_start: int,
        image_end: int,
    ):
        """
        Compute and store grounding scores from prefill logits.

        Call this after prefill, before generation starts.

        Args:
            prefill_logits: Logits from prefill, shape (seq_len, vocab_size)
            image_start: Start index of image tokens
            image_end: End index of image tokens (exclusive)
        """
        self._grounding_scores = compute_grounding_scores(
            prefill_logits, image_start, image_end
        )

    def clear_grounding_scores(self):
        """Clear stored grounding scores."""
        self._grounding_scores = None

    def reset(self):
        """Reset for new generation session."""
        self.risk_estimator.reset()
        self._current_risk = 0.0
        self._current_vge = 0.0
        self._step_count = 0
        self._num_visual_tokens = None
        self._prompt_length = None
        self._grounding_scores = None

    def _compute_m(self) -> float:
        """Compute m_t for visual boosting based on risk.

        m_t = 1.0 + (m_max_visual - 1.0) * risk
        When risk=0: m_t=1.0 (no boost)
        When risk=1: m_t=m_max_visual (max boost)
        """
        m_t = 1.0 + (self.m_max_visual - 1.0) * self._current_risk
        return max(1.0, min(self.m_max_visual, m_t))

    def _compute_suppress_factor(self) -> float:
        """Compute suppression factor for text tokens.

        Returns 1/m_max_text (fixed suppression based on config).
        """
        return 1.0 / self.m_max_text

    def _get_text_indices(
        self,
        total_len: int,
        visual_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Get indices of input text tokens to suppress (all_input scope).

        Suppresses all input text tokens (before and after image), excluding:
        - Visual tokens
        - Output tokens (generated after prompt)
        """
        visual_set = set(visual_indices.tolist())

        # Suppress all input text (up to prompt_length), excluding visual tokens
        if self._prompt_length is not None:
            end_idx = self._prompt_length
        else:
            end_idx = total_len

        text_indices = set(range(0, end_idx)) - visual_set

        if not text_indices:
            return torch.tensor([], dtype=torch.long, device=visual_indices.device)

        return torch.tensor(
            sorted(list(text_indices)),
            dtype=torch.long,
            device=visual_indices.device
        )

    def should_apply_to_layer(self, layer_idx: int) -> bool:
        """Check if should apply to this layer."""
        return self.start_layer <= layer_idx < self.end_layer

    def apply(
        self,
        attn_weights: torch.Tensor,
        visual_indices: torch.Tensor,
        layer_idx: int = 0,
        current_kv_len: int = None,
        **kwargs
    ) -> torch.Tensor:
        """Apply AdaVBoost attention reweighting strategy."""
        if not self._enabled:
            return attn_weights

        if not self.should_apply_to_layer(layer_idx):
            return attn_weights

        if visual_indices is None or visual_indices.numel() == 0:
            return attn_weights

        # Filter valid indices
        if current_kv_len is not None:
            valid_mask = visual_indices < current_kv_len
            valid_indices = visual_indices[valid_mask]
            kv_len = current_kv_len
        else:
            valid_indices = visual_indices
            kv_len = attn_weights.shape[-1]

        if valid_indices.numel() == 0:
            return attn_weights

        # Record prompt length on first call
        if self._prompt_length is None:
            self._prompt_length = kv_len

        # Track visual token count
        if self._num_visual_tokens is None:
            self._num_visual_tokens = valid_indices.numel()

        m_t = self._compute_m()

        # Skip if m_t ≈ 1
        if abs(m_t - 1.0) < 1e-6:
            return attn_weights

        # Boost ALL visual tokens
        attn_weights[:, :, -1, valid_indices] = (
            attn_weights[:, :, -1, valid_indices] * m_t
        )

        # Suppress input text tokens
        text_indices = self._get_text_indices(kv_len, valid_indices)
        if text_indices.numel() > 0:
            suppress_factor = self._compute_suppress_factor()
            attn_weights[:, :, -1, text_indices] = (
                attn_weights[:, :, -1, text_indices] * suppress_factor
            )

        self._step_count += 1

        return attn_weights

    def get_current_state(self) -> Dict[str, Any]:
        """Get current state for logging/debugging."""
        m_t = self._compute_m()
        return {
            'risk': self._current_risk,
            'vge': self._current_vge,
            'm_t': m_t,
            'suppress_factor': self._compute_suppress_factor(),
            'layer_range': (self.start_layer, self.end_layer),
            'prompt_length': self._prompt_length,
            'step_count': self._step_count,
            'm_max_visual': self.m_max_visual,
            'm_max_text': self.m_max_text,
            'has_grounding': self._grounding_scores is not None,
            'params': self.params.copy(),
        }

    def __repr__(self):
        return (
            f"AdaVBoostStrategy("
            f"model={self.model_name}, "
            f"vge_alpha={self.vge_alpha}, "
            f"m_visual=[1.0, {self.m_max_visual}], "
            f"m_text={self.m_max_text}, "
            f"layers=[{self.start_layer}, {self.end_layer}])"
        )
