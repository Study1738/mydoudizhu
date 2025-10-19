"""
Action-Level Fusion Strategy

This module implements action-level fusion between two DouZero models.
Both models compute action values, and a fusion MLP decides how to combine them
or which one to trust for each specific action.
"""

from .dual_model import DualModelInference
from .fusion_network import ActionFusionNetwork, GatingFusionNetwork
from .trainer import ActionFusionTrainer

__all__ = ['DualModelInference', 'ActionFusionNetwork', 'GatingFusionNetwork', 'ActionFusionTrainer']
