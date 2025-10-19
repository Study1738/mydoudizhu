"""
Multi-Model Inference

Manages parallel inference from multiple DouZero models (3 models: ADP, WP, SL).
"""

import os
import torch
import numpy as np
from douzero.dmc.models import LandlordLstmModel, FarmerLstmModel
from douzero.env.env import get_obs


class MultiModelInference:
    """
    Manages inference from 3 DouZero models in parallel
    """

    def __init__(self, position, device='cpu',
                 model_dir='/home/dingjiahe/mydoudizhu/DouZero/model'):
        """
        Args:
            position: Position ('landlord', 'landlord_up', 'landlord_down')
            device: Device to run inference on
            model_dir: Base directory containing model folders
        """
        self.position = position
        self.device = device

        # Define paths to 3 models
        self.model_paths = {
            'ADP': os.path.join(model_dir, 'douzero_ADP', f'{position}.ckpt'),
            'WP': os.path.join(model_dir, 'douzero_WP', f'{position}.ckpt'),
            'SL': os.path.join(model_dir, 'sl', f'{position}.ckpt')
        }

        # Load all 3 models
        self.models = {}
        for name, path in self.model_paths.items():
            print(f"Loading {name} model from {path}")
            self.models[name] = self._load_model(path, position, device)
            self.models[name].eval()

        print(f"Successfully loaded 3 models for position: {position}")

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

    def get_all_action_values(self, infoset):
        """
        Get action values from all 3 models for all legal actions

        Args:
            infoset: Game information set

        Returns:
            dict with action values from all models: {'ADP': tensor, 'WP': tensor, 'SL': tensor}
        """
        # Encode state using DouZero's encoding
        z_batch, x_batch = self._encode_state(infoset)

        results = {}

        with torch.no_grad():
            for name, model in self.models.items():
                output = model(z_batch, x_batch, return_value=True)
                results[name] = output['values']

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
        obs = get_obs(infoset)
        z_batch = torch.from_numpy(obs['z_batch']).float().to(self.device)
        x_batch = torch.from_numpy(obs['x_batch']).float().to(self.device)

        return z_batch, x_batch


class MultiModelFusionAgent:
    """
    Agent that uses multi-model inference with learned fusion strategy
    """

    def __init__(self, position, fusion_network, device='cpu', model_dir=None):
        """
        Args:
            position: Agent position
            fusion_network: Fusion network for combining outputs
            device: Device for inference
            model_dir: Base model directory (optional)
        """
        if model_dir is None:
            model_dir = '/home/dingjiahe/mydoudizhu/DouZero/model'

        self.multi_inference = MultiModelInference(
            position=position,
            device=device,
            model_dir=model_dir
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
        # Only one legal action, return it directly
        if len(infoset.legal_actions) == 1:
            return infoset.legal_actions[0]

        # Get values from all 3 models
        all_values = self.multi_inference.get_all_action_values(infoset)

        # Stack values: shape (3, num_actions)
        values_tensor = torch.stack([
            all_values['ADP'],
            all_values['WP'],
            all_values['SL']
        ], dim=0).to(self.device)  # Shape: (3, 1, num_actions)

        # Extract state features for fusion network
        from .feature_extractor import ActionLevelFeatureExtractor
        feature_extractor = ActionLevelFeatureExtractor(feature_dim=512)

        # Create a dict with all model outputs for feature extraction
        model_outputs = {
            'values_a': all_values['ADP'],
            'values_b': all_values['WP'],
            'values_c': all_values['SL']
        }

        state_features = feature_extractor.extract_features(
            infoset, self.position, model_outputs
        ).unsqueeze(0).to(self.device)

        # Fuse the outputs using fusion network
        # Note: fusion_network needs to be updated to handle 3 models
        fused_output = self.fusion_network(
            values_tensor.squeeze(1),  # Shape: (3, num_actions)
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
