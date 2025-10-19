"""
Dual Model Inference

Manages parallel inference from two DouZero models.
"""

import os
import torch
import numpy as np
from collections import OrderedDict

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from douzero.dmc.models import LandlordLstmModel, FarmerLstmModel


class DualModelInference:
    """
    Manages inference from two DouZero models in parallel
    """

    def __init__(self, model_path_a, model_path_b, position, device='cpu'):
        """
        Args:
            model_path_a: Path to first model checkpoint
            model_path_b: Path to second model checkpoint
            position: Position ('landlord', 'landlord_up', 'landlord_down')
            device: Device to run inference on
        """
        self.position = position
        self.device = device
        # Persist model paths for downstream components that may need to
        # reload checkpoints (e.g. evaluator creating new agents).
        self.model_path_a = os.path.abspath(model_path_a)
        self.model_path_b = os.path.abspath(model_path_b)

        # Load both models
        self.model_a = self._load_model(model_path_a, position, device)
        self.model_b = self._load_model(model_path_b, position, device)

        # Set to eval mode
        self.model_a.eval()
        self.model_b.eval()

    def _load_model(self, checkpoint_path, position, device):
        """
        Load a DouZero model from checkpoint

        Args:
            checkpoint_path: Path to checkpoint file
            position: Position of the model
            device: Device to load on

        Returns:
            Loaded model
        """
        # Create model based on position
        if position == 'landlord':
            model = LandlordLstmModel().to(device)
        else:
            model = FarmerLstmModel().to(device)

        # Load checkpoint
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=device)

        # Handle different checkpoint formats
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint

        model.load_state_dict(state_dict)

        return model

    @staticmethod
    def _format_action_values(values_tensor):
        """Ensure action-value tensor shaped as (batch, num_actions)."""
        if values_tensor.dim() == 1:
            values_tensor = values_tensor.unsqueeze(0)
        elif values_tensor.dim() == 2 and values_tensor.size(-1) == 1:
            values_tensor = values_tensor.squeeze(-1).unsqueeze(0)
        elif values_tensor.dim() == 2 and values_tensor.size(0) == 1:
            # Already in expected shape (1, num_actions)
            pass
        elif values_tensor.dim() == 3 and values_tensor.size(-1) == 1:
            # Rare case: (batch, num_actions, 1)
            values_tensor = values_tensor.squeeze(-1)
        return values_tensor.float().contiguous()

    def format_action_values(self, values_tensor):
        """Public helper for reformatting cached action-value tensors."""
        return self._format_action_values(values_tensor)

    def dual_inference(self, z, x):
        """
        Perform inference with both models

        Args:
            z: LSTM input features (batch_size, seq_len, 162)
            x: Action features (batch_size, feature_dim)

        Returns:
            dict with 'values_a' and 'values_b' (action values from both models)
        """
        with torch.no_grad():
            # Model A inference
            output_a = self.model_a(z, x, return_value=True)
            values_a = self._format_action_values(output_a['values'])

            # Model B inference
            output_b = self.model_b(z, x, return_value=True)
            values_b = self._format_action_values(output_b['values'])

        return {
            'values_a': values_a,
            'values_b': values_b
        }

    def get_dual_action_values(self, infoset):
        """
        Get action values from both models for all legal actions

        Args:
            infoset: Game information set

        Returns:
            dict with action values from both models
        """
        # Encode state (using DouZero's encoding)
        # This is a simplified version; actual implementation needs proper encoding
        z_batch, x_batch = self._encode_state(infoset)

        # Get values from both models
        results = self.dual_inference(z_batch, x_batch)

        return results

    def _encode_state(self, infoset):
        """
        Encode game state for model input

        Args:
            infoset: Game information set

        Returns:
            tuple (z_batch, x_batch) for model input
        """
        # Use DouZero's built-in encoding
        from douzero.env.env import get_obs

        obs = get_obs(infoset)
        z_batch = torch.from_numpy(obs['z_batch']).float().to(self.device)
        x_batch = torch.from_numpy(obs['x_batch']).float().to(self.device)

        return z_batch, x_batch


class DualModelAgent:
    """
    Agent that uses dual model inference with fusion strategy
    """

    def __init__(self, model_path_a, model_path_b, position, fusion_network, device='cpu'):
        """
        Args:
            model_path_a: Path to first model
            model_path_b: Path to second model
            position: Agent position
            fusion_network: Fusion network for combining outputs
            device: Device for inference
        """
        self.dual_inference = DualModelInference(
            model_path_a, model_path_b, position, device
        )
        self.fusion_network = fusion_network
        self.position = position
        self.device = device

    def act(self, infoset):
        """
        Select action using fusion strategy

        Args:
            infoset: Game information set

        Returns:
            Selected action
        """
        # Get values from both models
        dual_values = self.dual_inference.get_dual_action_values(infoset)
        values_a = self.dual_inference.format_action_values(dual_values['values_a'])
        values_b = self.dual_inference.format_action_values(dual_values['values_b'])

        # Extract state features for fusion network
        state_features = self._extract_features(infoset)

        # Fuse the outputs
        fused_output = self.fusion_network(
            values_a,
            values_b,
            state_features
        )

        # Select action based on fused output
        if 'action_probs' in fused_output:
            # Probabilistic selection
            action_idx = torch.multinomial(fused_output['action_probs'], 1).item()
        else:
            # Greedy selection
            action_idx = torch.argmax(fused_output['fused_values']).item()

        # Map action index to actual action
        action = infoset.legal_actions[action_idx]

        return action

    def _extract_features(self, infoset):
        """
        Extract features from infoset for fusion network

        Args:
            infoset: Game information set

        Returns:
            Feature tensor
        """
        from .feature_extractor import ActionLevelFeatureExtractor

        # Use the feature extractor
        feature_extractor = ActionLevelFeatureExtractor(feature_dim=512)
        features = feature_extractor.extract_features(infoset, self.position, None)

        return features.unsqueeze(0).to(self.device)


class EnsemblePredictor:
    """
    Simple ensemble predictor for baseline comparison
    """

    def __init__(self, model_path_a, model_path_b, position, ensemble_method='average', device='cpu'):
        """
        Args:
            model_path_a: Path to first model
            model_path_b: Path to second model
            position: Position
            ensemble_method: 'average', 'max', 'vote'
            device: Device
        """
        self.dual_inference = DualModelInference(
            model_path_a, model_path_b, position, device
        )
        self.ensemble_method = ensemble_method

    def predict(self, infoset):
        """
        Predict action using simple ensemble

        Args:
            infoset: Game information set

        Returns:
            Action
        """
        dual_values = self.dual_inference.get_dual_action_values(infoset)

        values_a = dual_values['values_a']
        values_b = dual_values['values_b']

        if self.ensemble_method == 'average':
            # Average ensemble
            fused_values = (values_a + values_b) / 2.0
        elif self.ensemble_method == 'max':
            # Max ensemble
            fused_values = torch.max(values_a, values_b)
        elif self.ensemble_method == 'vote':
            # Voting ensemble
            action_a = torch.argmax(values_a)
            action_b = torch.argmax(values_b)
            # If different, pick the one with higher value
            if action_a == action_b:
                action_idx = action_a
            else:
                if values_a[action_a] > values_b[action_b]:
                    action_idx = action_a
                else:
                    action_idx = action_b
            return infoset.legal_actions[action_idx]
        else:
            raise ValueError(f"Unknown ensemble method: {self.ensemble_method}")

        action_idx = torch.argmax(fused_values).item()
        return infoset.legal_actions[action_idx]
