"""
Qwen Model Module

Contains Qwen3-VL forward interface with strategy-based attention boost.
"""

from .boosted_interface import QwenBoostedInterface

__all__ = [
    'QwenBoostedInterface',
]
