"""
LLaVA Model Module

Contains LLaVA forward interface with strategy-based attention boost.
"""

from .boosted_interface import LLaVABoostedInterface

__all__ = [
    'LLaVABoostedInterface',
]
