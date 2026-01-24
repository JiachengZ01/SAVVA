#!/usr/bin/env python3
"""
Risk Update LogitsProcessor

Updates strategy's risk value at each generation step.
This is CRITICAL - without this, the strategy uses constant risk=0.

Author: AdaVBoost Project
"""

import torch
from transformers import LogitsProcessor


class RiskLogitsProcessor(LogitsProcessor):
    """
    LogitsProcessor that updates strategy's risk at each step.

    This must be used during generation for AdaVBoost strategy to work properly.
    Without this, the risk stays at 0 and the strategy has no effect.

    Usage:
        processor = RiskLogitsProcessor(strategy)
        outputs = model.generate(
            ...,
            logits_processor=LogitsProcessorList([processor])
        )
    """

    def __init__(self, strategy, debug=False):
        """
        Initialize RiskLogitsProcessor.

        Args:
            strategy: Strategy instance (must have update_risk_from_logits method)
            debug: Whether to print debug info
        """
        self.strategy = strategy
        self.debug = debug
        self.step_count = 0

    def __call__(
        self,
        input_ids: torch.LongTensor,
        scores: torch.FloatTensor,
    ) -> torch.FloatTensor:
        """
        Update strategy's risk from current logits.

        Args:
            input_ids: Current input IDs [batch, seq_len]
            scores: Current logits [batch, vocab_size]

        Returns:
            Unmodified scores (we only update the strategy state)
        """
        # Update strategy's risk value
        if hasattr(self.strategy, 'update_risk_from_logits'):
            self.strategy.update_risk_from_logits(scores)

            # Debug output (every 50 steps)
            if self.debug:
                self.step_count += 1
                if self.step_count % 50 == 1:  # Print at step 1, 51, 101, ...
                    state = self.strategy.get_current_state() if hasattr(self.strategy, 'get_current_state') else {}
                    risk = getattr(self.strategy, '_current_risk', 0)
                    vge = getattr(self.strategy, '_current_vge', 0)
                    print(f"  [Step {self.step_count}] vge={vge:.4f}, "
                          f"risk={risk:.4f}, "
                          f"m={state.get('m_t', 0):.3f}")

        # Return scores unchanged - we don't modify logits, just update state
        return scores

    def reset(self):
        """Reset for new generation session."""
        if hasattr(self.strategy, 'reset'):
            self.strategy.reset()
