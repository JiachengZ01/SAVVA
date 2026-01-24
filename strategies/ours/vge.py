"""
VGE (Visual Grounding Entropy) Module

Computes hallucination risk signal for AdaVBoost strategy.

VGE = alpha * norm_entropy + (1 - alpha) * (1 - G(v))

Where G(v) = max_{i in image_tokens} softmax(logits_i)[v]
represents visual grounding - how well the image supports vocab word v.

Intuition:
- High entropy = model uncertain = risky
- Low G = image doesn't support this word = risky
- Combining both signals improves hallucination detection

Author: AdaVBoost Project
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Optional


def compute_grounding_scores(
    prefill_logits: torch.Tensor,
    image_start: int,
    image_end: int,
) -> Optional[torch.Tensor]:
    """
    Compute visual grounding scores G(v) from prefill logits.

    G(v) = max_{i in image_tokens} softmax(logits_i)[v]

    This represents how well the image "supports" each vocabulary word.
    Higher G means the image can predict this word with high probability.

    Args:
        prefill_logits: Logits from prefill, shape (seq_len, vocab_size)
        image_start: Start index of image tokens
        image_end: End index of image tokens (exclusive)

    Returns:
        G: Grounding scores for each vocab word, shape (vocab_size,)
        Returns None if image_start/end are invalid
    """
    if image_start is None or image_end is None:
        return None

    if image_end > prefill_logits.shape[0] or image_start >= image_end:
        return None

    # Extract image token logits
    image_logits = prefill_logits[image_start:image_end]  # (num_img_tokens, vocab_size)

    if image_logits.shape[0] == 0:
        return None

    # Compute softmax probabilities
    image_probs = F.softmax(image_logits, dim=-1)  # (num_img_tokens, vocab_size)

    # G(v) = max probability across image tokens
    G = image_probs.max(dim=0).values  # (vocab_size,)

    return G


def compute_vge(
    logits: torch.Tensor,
    grounding_scores: Optional[torch.Tensor],
    alpha: float = 0.75,
    vocab_size: Optional[int] = None,
) -> float:
    """
    Compute VGE (Visual Grounding Entropy) score.

    VGE = alpha * norm_entropy + (1 - alpha) * (1 - G(v_pred))

    Where v_pred is the predicted token (argmax of logits).

    Intuition:
    - Entropy captures model uncertainty
    - (1-G) captures lack of visual grounding
    - High VGE = risky (uncertain OR not grounded in image)

    Args:
        logits: Token logits, shape [..., vocab_size]
        grounding_scores: Precomputed G(v) from compute_grounding_scores()
        alpha: Weight for entropy (default 0.75)
        vocab_size: Vocabulary size (auto-detected if None)

    Returns:
        vge: Risk signal in [0, 1]
    """
    # Flatten logits if needed
    if logits.dim() > 1:
        logits = logits.view(-1, logits.shape[-1])[-1]  # Get last position

    if vocab_size is None:
        vocab_size = logits.shape[-1]

    # Compute probabilities
    probs = F.softmax(logits, dim=-1)
    log_probs = F.log_softmax(logits, dim=-1)

    # 1. Compute normalized entropy
    entropy = -(probs * log_probs).sum().item()
    max_entropy = np.log(vocab_size)
    norm_entropy = entropy / max_entropy  # [0, 1]

    # 2. Compute visual grounding risk (1 - G)
    if grounding_scores is not None:
        # Get predicted token
        pred_token = logits.argmax().item()
        g_score = grounding_scores[pred_token].item()
        grounding_risk = 1.0 - g_score  # [0, 1], high = not grounded
    else:
        # No visual grounding available, use entropy only
        grounding_risk = norm_entropy

    # 3. Combine: VGE = alpha * entropy + (1-alpha) * (1-G)
    vge = alpha * norm_entropy + (1 - alpha) * grounding_risk

    return vge


class RiskEstimator:
    """
    Estimates hallucination risk from VGE signal.

    Uses simple linear scaling: risk = vge / scale (clipped to [0, 1])
    """

    def __init__(self, scale: float = 0.5):
        """
        Initialize RiskEstimator.

        Args:
            scale: VGE value that maps to risk=1.0
        """
        self.scale = scale
        self.step_count = 0

    def compute_risk(self, vge: float) -> float:
        """
        Compute hallucination risk from VGE.

        Args:
            vge: VGE value in [0, 1]

        Returns:
            risk: Hallucination risk in [0, 1]
        """
        # Linear mapping with clip: [0, scale] -> [0, 1]
        risk = min(1.0, max(0.0, vge / self.scale))
        self.step_count += 1
        return risk

    def reset(self):
        """Reset state for new generation session."""
        self.step_count = 0

    def __repr__(self):
        return f"RiskEstimator(scale={self.scale})"
