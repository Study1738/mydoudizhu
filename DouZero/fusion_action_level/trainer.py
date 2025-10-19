"""
Trainer for Action-Level Fusion

Implements PPO/A2C training for the fusion network.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque

from .fusion_network import GatingFusionNetwork, ActionFusionNetwork
from .dual_model import DualModelInference
from .feature_extractor import ActionLevelFeatureExtractor


class ActionFusionTrainer:
    """
    Trainer for action-level fusion network
    """

    def __init__(
        self,
        model_path_a,
        model_path_b,
        position,
        fusion_type='gating',  # 'gating', 'action', 'attention', 'moe'
        num_actions=309,
        feature_dim=512,
        lr=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_epsilon=0.2,
        value_loss_coef=0.5,
        entropy_coef=0.01,
        max_grad_norm=0.5,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        objective='adp',
        gate_lr_scale=0.2,
        gate_warmup_iters=0,
        gate_reg_coeff=1e-3
    ):
        """
        Args:
            model_path_a: Path to first model
            model_path_b: Path to second model
            position: Position for this trainer
            fusion_type: Type of fusion network
            num_actions: Number of actions
            feature_dim: Feature dimension
            lr: Learning rate
            gamma: Discount factor
            gae_lambda: GAE lambda
            clip_epsilon: PPO clip epsilon
            value_loss_coef: Value loss coefficient
            entropy_coef: Entropy coefficient
            max_grad_norm: Max gradient norm
            device: Training device
            gate_lr_scale: Learning rate multiplier for gating network
            gate_warmup_iters: Number of early updates to freeze gate parameters
            gate_reg_coeff: L2 regularisation strength pulling gate towards initial value
        """
        self.device = device
        self.position = position
        self.num_actions = num_actions
        self.objective = objective
        self.gate_lr_scale = gate_lr_scale
        self.gate_warmup_iters = gate_warmup_iters
        self.gate_reg_coeff = gate_reg_coeff
        self._update_calls = 0

        # Initialize dual model inference (frozen)
        self.dual_inference = DualModelInference(
            model_path_a, model_path_b, position, device
        )

        # Initialize fusion network (trainable)
        self.fusion_type = fusion_type
        if fusion_type == 'gating':
            if not (0 < gate_lr_scale <= 1.0):
                raise ValueError("gate_lr_scale must be in (0, 1]")
            if gate_warmup_iters < 0:
                raise ValueError("gate_warmup_iters must be >= 0")
            if gate_reg_coeff < 0:
                raise ValueError("gate_reg_coeff must be non-negative")
            self.fusion_net = GatingFusionNetwork(
                num_actions=num_actions,
                state_feature_dim=feature_dim,
                gating_type='global'
            ).to(device)
        elif fusion_type == 'action':
            self.fusion_net = ActionFusionNetwork(
                num_actions=num_actions,
                state_feature_dim=feature_dim
            ).to(device)
        elif fusion_type == 'attention':
            from .fusion_network import AttentionFusionNetwork
            self.fusion_net = AttentionFusionNetwork(
                num_actions=num_actions,
                state_feature_dim=feature_dim
            ).to(device)
        elif fusion_type == 'moe':
            from .fusion_network import MixtureOfExpertsNetwork
            self.fusion_net = MixtureOfExpertsNetwork(
                num_actions=num_actions,
                state_feature_dim=feature_dim
            ).to(device)
        else:
            raise ValueError(f"Unknown fusion type: {fusion_type}")

        # Value network (critic)
        self.value_net = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        ).to(device)

        # Optimizers
        fusion_lr = lr
        if self.fusion_type == 'gating':
            fusion_lr = lr * self.gate_lr_scale

        self.fusion_optimizer = optim.Adam(self.fusion_net.parameters(), lr=fusion_lr)
        self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=lr)

        # Feature extractor
        self.feature_extractor = ActionLevelFeatureExtractor(feature_dim)

        # Training hyperparameters
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm

        # Training statistics
        self.stats = {
            'policy_losses': deque(maxlen=100),
            'value_losses': deque(maxlen=100),
            'total_losses': deque(maxlen=100),
            'episode_rewards': deque(maxlen=100)
        }

    def select_action(self, infoset, deterministic=False):
        """
        Select action using fusion network

        Args:
            infoset: Game information set
            deterministic: Whether to select deterministically

        Returns:
            Selected action and metadata
        """
        # Get dual model outputs
        dual_outputs = self.dual_inference.get_dual_action_values(infoset)
        values_a = dual_outputs['values_a']
        values_b = dual_outputs['values_b']

        # Extract features
        state_features = self.feature_extractor.extract_features(
            infoset, self.position, dual_outputs
        ).unsqueeze(0).to(self.device)

        # Fusion network forward
        with torch.no_grad():
            fusion_output = self.fusion_net(values_a, values_b, state_features)

        # Select action
        if 'action_probs' in fusion_output:
            # Probabilistic output
            probs = fusion_output['action_probs']
            if deterministic:
                action_idx = torch.argmax(probs, dim=-1).item()
            else:
                action_idx = torch.multinomial(probs, 1).item()
        else:
            # Value-based output
            fused_values = fusion_output['fused_values']
            action_idx = torch.argmax(fused_values, dim=-1).item()

        # Map to legal action
        action = infoset.legal_actions[action_idx]

        return {
            'action': action,
            'action_idx': action_idx,
            'fusion_output': fusion_output,
            'state_features': state_features
        }

    def compute_gae(self, rewards, values, dones):
        """Compute Generalized Advantage Estimation"""
        advantages = []
        returns = []
        gae = 0

        rewards = np.array(rewards)
        values = np.array(values)
        dones = np.array(dones)

        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0
            else:
                next_value = values[t + 1]

            delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae

            advantages.insert(0, gae)
            returns.insert(0, gae + values[t])

        return np.array(advantages), np.array(returns)

    def update(self, trajectories, num_epochs=4):
        """
        Update fusion network using simplified loss (like DouZero)

        Following DouZero's approach: directly optimize MSE between
        fused predictions and targets, no PPO complexity.

        Args:
            trajectories: List of trajectory dicts
            num_epochs: Number of update epochs

        Returns:
            Training statistics
        """
        policy_losses = []
        value_losses = []
        total_losses = []
        reg_losses = []

        for epoch in range(num_epochs):
            for traj in trajectories:
                # Get trajectory data
                obs_z_list = traj['obs_z']
                obs_x_list = traj['obs_x']
                action_indices = traj['action_indices']
                legal_actions_list = traj['legal_actions']
                cached_features = traj.get('state_features')
                target = traj['target']  # Final reward
                num_steps = traj['num_steps']

                for step_idx in range(num_steps):
                    # Get this step's data
                    obs_z = obs_z_list[step_idx].to(self.device)
                    obs_x = obs_x_list[step_idx].to(self.device)
                    action_idx = action_indices[step_idx]
                    legal_actions = legal_actions_list[step_idx]

                    # Recompute values from both models
                    with torch.no_grad():
                        output_a = self.dual_inference.model_a(obs_z, obs_x, return_value=True)
                        output_b = self.dual_inference.model_b(obs_z, obs_x, return_value=True)
                        values_a = self.dual_inference.format_action_values(output_a['values'])
                        values_b = self.dual_inference.format_action_values(output_b['values'])

                    if cached_features:
                        state_feat = cached_features[step_idx]
                        if isinstance(state_feat, torch.Tensor):
                            state_features = state_feat.to(self.device)
                        else:
                            state_features = torch.tensor(state_feat, dtype=torch.float32, device=self.device)
                        if state_features.dim() == 1:
                            state_features = state_features.unsqueeze(0)
                    else:
                        stats = torch.tensor([[
                            values_a.mean().item(),
                            values_b.mean().item(),
                            values_a.max().item(),
                            values_b.max().item(),
                            float(len(legal_actions)),
                            float(len(legal_actions) > 0)
                        ]], dtype=torch.float32, device=self.device)

                        state_features = torch.nn.functional.pad(
                            stats, (0, self.feature_extractor.feature_dim - stats.size(-1))
                        )

                    # Fusion network forward
                    fusion_output = self.fusion_net(values_a, values_b, state_features)
                    fused_values = fusion_output['fused_values']  # Should be (1, num_legal_actions)

                    # Store gate weights for statistics
                    if 'gate_weights' in fusion_output:
                        self._last_gate_weights = fusion_output['gate_weights'].mean().item()

                    # Debug: Check dimensions
                    if fused_values.size(1) == 1:
                        # Something went wrong, fused_values should have same size as values_a
                        # This happens when values_a/values_b are not passed correctly
                        print(f"Warning: fused_values shape {fused_values.shape}, values_a shape {values_a.shape}")
                        # Use simple average as fallback
                        fused_values = (values_a + values_b) / 2.0

                    # Get the value for the selected action
                    predicted_values = fused_values[:, action_idx]

                    target_tensor = torch.tensor([target], dtype=torch.float32, device=self.device)

                    # Policy loss: fused action value vs final return
                    policy_loss = nn.functional.mse_loss(predicted_values, target_tensor)

                    # Value loss: critic prediction vs final return
                    value_pred = self.value_net(state_features).squeeze(-1)
                    value_loss = nn.functional.mse_loss(value_pred, target_tensor)

                    total_loss = policy_loss + self.value_loss_coef * value_loss

                    if self.fusion_type == 'gating' and 'gate_weights' in fusion_output and self.gate_reg_coeff > 0:
                        gate_weights = fusion_output['gate_weights']
                        target_gate = getattr(self.fusion_net, 'initial_gate', 0.55)
                        gate_reg = self.gate_reg_coeff * (gate_weights - target_gate) ** 2
                        gate_reg = gate_reg.mean()
                        total_loss = total_loss + gate_reg
                        reg_losses.append(gate_reg.item())

                    # Optimize
                    self.fusion_optimizer.zero_grad()
                    self.value_optimizer.zero_grad()
                    total_loss.backward()
                    nn.utils.clip_grad_norm_(self.fusion_net.parameters(), self.max_grad_norm)
                    nn.utils.clip_grad_norm_(self.value_net.parameters(), self.max_grad_norm)

                    freeze_gate = self.fusion_type == 'gating' and self._update_calls < self.gate_warmup_iters
                    if not freeze_gate:
                        self.fusion_optimizer.step()

                    self.value_optimizer.step()

                    policy_losses.append(policy_loss.item())
                    value_losses.append(value_loss.item())
                    total_losses.append(total_loss.item())

        # Update statistics
        mean_policy_loss = np.mean(policy_losses) if policy_losses else 0.0
        mean_value_loss = np.mean(value_losses) if value_losses else 0.0
        mean_total_loss = np.mean(total_losses) if total_losses else mean_policy_loss

        # Collect fusion weights statistics (from last batch)
        fusion_stats = {
            'policy_loss': mean_policy_loss,
            'value_loss': mean_value_loss,
            'total_loss': mean_total_loss
        }

        # Add gating weights info if available
        if hasattr(self, '_last_gate_weights'):
            fusion_stats['avg_gate_weight'] = self._last_gate_weights
        if reg_losses:
            fusion_stats['gate_reg_loss'] = float(np.mean(reg_losses))

        self._update_calls += 1

        return fusion_stats

    def _compute_log_probs(self, fusion_output, actions):
        """Compute log probabilities of actions"""
        if 'action_probs' in fusion_output:
            probs = fusion_output['action_probs']
            log_probs = torch.log(probs.gather(1, actions.unsqueeze(1)).squeeze() + 1e-8)
        else:
            # For value-based, use softmax
            values = fusion_output['fused_values']
            probs = torch.softmax(values, dim=-1)
            log_probs = torch.log(probs.gather(1, actions.unsqueeze(1)).squeeze() + 1e-8)

        return log_probs

    def save_checkpoint(self, save_path, epoch, stats=None):
        """Save training checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'fusion_net_state_dict': self.fusion_net.state_dict(),
            'value_net_state_dict': self.value_net.state_dict(),
            'fusion_optimizer_state_dict': self.fusion_optimizer.state_dict(),
            'value_optimizer_state_dict': self.value_optimizer.state_dict(),
            'stats': stats,
            'training_stats': dict(self.stats)
        }

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(checkpoint, save_path)

    def load_checkpoint(self, checkpoint_path):
        """Load training checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.fusion_net.load_state_dict(checkpoint['fusion_net_state_dict'])
        self.value_net.load_state_dict(checkpoint['value_net_state_dict'])
        self.fusion_optimizer.load_state_dict(checkpoint['fusion_optimizer_state_dict'])
        self.value_optimizer.load_state_dict(checkpoint['value_optimizer_state_dict'])

        return checkpoint.get('epoch', 0), checkpoint.get('stats', {})
