"""
Fusion Networks for Action-Level Fusion

Different architectures for combining outputs from two models.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class ActionFusionNetwork(nn.Module):
    pass


class GatingFusionNetwork(nn.Module):
    """
    Gating network that outputs weights to combine two model outputs

    Input: [values_a, values_b, state_features]
    Output: Gating weights β to combine outputs
    """

    def __init__(
        self,
        num_actions=309,
        state_feature_dim=512,
        hidden_dim=256,
        gating_type='global',
        initial_gate=0.5  # 改为0.5，即50-50平衡开始
    ):
        """Initialise gating network.

        The ``num_actions`` argument is kept for backwards compatibility but no
        longer drives the layer dimensions – the network now supports dynamic
        numbers of legal actions by summarising the two model outputs.
        """
        super(GatingFusionNetwork, self).__init__()

        if gating_type != 'global':
            raise ValueError("Only 'global' gating is supported for dynamic action spaces")

        if not (0.1 < initial_gate < 0.9):
            raise ValueError("initial_gate must be within (0.1, 0.9) to respect protection bounds")

        self.gating_type = gating_type
        self.state_feature_dim = state_feature_dim
        self.initial_gate = float(initial_gate)

        # We append lightweight summary statistics of the model outputs
        # (mean/max and disagreement metrics) so that the gating decision is
        # aware of per-step confidence without depending on the raw action
        # vector length.
        self._summary_dim = 6

        backbone_input_dim = state_feature_dim + self._summary_dim
        self.fc1 = nn.Linear(backbone_input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.gate = nn.Linear(hidden_dim // 2, 1)

        self._reset_parameters()

    def _reset_parameters(self):
        """Initialise layers so that the starting gate outputs ~initial_gate."""
        nn.init.kaiming_uniform_(self.fc1.weight, a=math.sqrt(5))
        nn.init.zeros_(self.fc1.bias)

        nn.init.kaiming_uniform_(self.fc2.weight, a=math.sqrt(5))
        nn.init.zeros_(self.fc2.bias)

        nn.init.zeros_(self.gate.weight)

        # Solve for bias term in gate activation:
        # gate = 0.1 + 0.8 * sigmoid(z) = initial_gate  => sigmoid(z) = (initial_gate - 0.1) / 0.8
        target = (self.initial_gate - 0.1) / 0.8
        # Numerical safety to keep value inside (0, 1)
        target = min(max(target, 1e-5), 1 - 1e-5)
        bias = math.log(target / (1 - target))
        nn.init.constant_(self.gate.bias, bias)

        # Debug: verify initialization
        print(f"[GatingFusion Init] target_sigmoid={target:.4f}, bias={bias:.4f}, expected_gate={self.initial_gate:.4f}")

    def forward(self, values_a, values_b, state_features):
        """
        Forward pass

        Args:
            values_a: Action values from model A (batch_size, num_actions)
            values_b: Action values from model B (batch_size, num_actions)
            state_features: State features (batch_size, feature_dim)

        Returns:
            dict with 'fused_values' and 'gate_weights'
        """
        if values_a.dim() == 1:
            values_a = values_a.unsqueeze(0)
        if values_b.dim() == 1:
            values_b = values_b.unsqueeze(0)
        if state_features.dim() == 1:
            state_features = state_features.unsqueeze(0)

        values_a = values_a.to(state_features)
        values_b = values_b.to(state_features)

        # Safety check for empty values (shouldn't happen, but handle gracefully)
        if values_a.size(-1) == 0 or values_b.size(-1) == 0:
            # Return zero fused values - this should be caught by caller
            batch_size = values_a.size(0) if values_a.size(0) > 0 else 1
            return {
                'fused_values': torch.zeros((batch_size, 0), device=state_features.device),
                'gate_weights': torch.tensor([[0.5]], device=state_features.device)
            }

        diff = values_a - values_b
        summary = torch.cat([
            values_a.mean(dim=-1, keepdim=True),
            values_b.mean(dim=-1, keepdim=True),
            values_a.max(dim=-1, keepdim=True).values,
            values_b.max(dim=-1, keepdim=True).values,
            diff.mean(dim=-1, keepdim=True),
            diff.abs().max(dim=-1, keepdim=True).values,
        ], dim=-1)

        x = torch.cat([state_features, summary], dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        # Apply 10% weight protection: gate_weights ∈ [0.1, 0.9]
        # Formula: gate = 0.1 + 0.8 * sigmoid(x)
        # This ensures both models contribute at least 10%
        gate_weights = 0.1 + 0.8 * torch.sigmoid(self.gate(x))

        fused_values = gate_weights * values_a + (1 - gate_weights) * values_b

        return {
            'fused_values': fused_values,
            'gate_weights': gate_weights
        }


class AttentionFusionNetwork(nn.Module):
    pass
    


class MixtureOfExpertsNetwork(nn.Module):
    pass
    


class AdaptiveFusionNetwork(nn.Module):
    pass
    
