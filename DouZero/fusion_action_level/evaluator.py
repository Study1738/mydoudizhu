"""
Evaluator for Action-Level Fusion

Handles evaluation and trajectory collection.
"""

import os
import pickle
import copy
import numpy as np
from collections import defaultdict

POSITIONS = ('landlord', 'landlord_up', 'landlord_down')

class ActionFusionEvaluator:
    """
    Evaluator for action-level fusion strategy
    """

    def __init__(self, trainer, eval_data_path, device='cpu', objective='wp', opponent_map=None):
        """
        Args:
            trainer: ActionFusionTrainer instance
            eval_data_path: Path to evaluation data
            device: Device
        """
        self.trainer = trainer
        self.eval_data_path = eval_data_path
        self.device = device
        self.objective = objective

        # Load evaluation data
        with open(eval_data_path, 'rb') as f:
            self.eval_data = pickle.load(f)

        self._opponent_map = self._normalise_opponent_map(opponent_map)

    @staticmethod
    def _normalise_opponent_map(opponent_map):
        if not opponent_map:
            return {}
        normalised = {}
        for pos in POSITIONS:
            value = opponent_map.get(pos) if isinstance(opponent_map, dict) else None
            if value is not None:
                normalised[pos] = value
        return normalised

    def _resolve_opponent_types(self, default_type, override_map=None):
        resolved = {pos: default_type for pos in POSITIONS}
        for mapping in (self._opponent_map, override_map or {}):
            if not mapping:
                continue
            for pos, value in mapping.items():
                if pos in resolved and value is not None:
                    resolved[pos] = value
        return resolved

    def _get_reward_components(self, env):
        if self.trainer.position == 'landlord':
            base_reward = float(env.player_utility_dict['landlord'])
        else:
            base_reward = float(env.player_utility_dict['farmer'])
        sign = 1.0 if base_reward > 0 else -1.0
        bomb_num = getattr(env, 'bomb_num', 0)
        return sign, bomb_num, base_reward

    def _compute_reward(self, env, mode=None):
        sign, bomb_num, base_reward = self._get_reward_components(env)
        mode = mode or self.objective
        if mode == 'wp':
            return sign
        elif mode == 'logadp':
            return sign * (bomb_num + 1.0)
        else:  # default adp
            return sign * (2 ** bomb_num)

    def evaluate(self, num_games=1000, opponent_type='perfectdou', opponent_map=None):
        """
        Evaluate fusion strategy

        Args:
            num_games: Number of games
            opponent_type: Opponent type
            opponent_map: Optional dict overriding opponent type per position

        Returns:
            Evaluation statistics
        """
        stats = {
            'num_games': num_games,
            'wins': 0,
            'losses': 0,
            'total_reward': 0.0,
            'total_reward_wp': 0.0,
            'total_reward_adp': 0.0,
            'fusion_stats': defaultdict(list)
        }

        # Evaluation loop
        game_data_subset = self.eval_data[:num_games]
        resolved_opponents = self._resolve_opponent_types(opponent_type, opponent_map)

        for game_idx, card_play_data in enumerate(game_data_subset):
            try:
                # Deep copy to avoid data corruption
                import copy
                card_play_data_copy = copy.deepcopy(card_play_data)

                # Play game and collect stats
                game_result = self._play_game(card_play_data_copy, resolved_opponents, stats)

                if game_idx % 100 == 0:
                    print(f"Evaluated {game_idx}/{num_games} games")
            except (RuntimeError, ValueError, IndexError):
                # Skip invalid games
                continue
            except Exception as e:
                # Unexpected errors
                if game_idx < 5:
                    print(f"  Eval error in game {game_idx}: {type(e).__name__}")
                continue

        # Calculate final statistics
        total_games = stats['wins'] + stats['losses']
        stats['win_rate'] = stats['wins'] / total_games if total_games > 0 else 0.0
        stats['avg_reward'] = stats['total_reward'] / total_games if total_games > 0 else 0.0
        stats['avg_reward_wp'] = stats['total_reward_wp'] / total_games if total_games > 0 else 0.0
        stats['avg_reward_adp'] = stats['total_reward_adp'] / total_games if total_games > 0 else 0.0

        return stats

    def _play_game(self, card_play_data, opponent_types, stats):
        """
        Play a single game

        Args:
            card_play_data: Initial card distribution
            opponent_types: Dict mapping position to opponent descriptor
            stats: Statistics dict to update

        Returns:
            Game result
        """
        from douzero.env.game import GameEnv
        from .opponent_loader import create_opponent_dict
        from .dual_model import DualModelAgent

        opponent_types = opponent_types or {}

        # Create fusion agent for the trained position
        fusion_agent = DualModelAgent(
            model_path_a=self.trainer.dual_inference.model_path_a,
            model_path_b=self.trainer.dual_inference.model_path_b,
            position=self.trainer.position,
            fusion_network=self.trainer.fusion_net,
            device=self.device
        )

        # Create opponents for other positions
        opponents = create_opponent_dict(
            landlord_type=opponent_types.get('landlord'),
            landlord_up_type=opponent_types.get('landlord_up'),
            landlord_down_type=opponent_types.get('landlord_down')
        )

        # Replace our position with fusion agent
        opponents[self.trainer.position] = fusion_agent

        # Create game environment
        env = GameEnv(opponents)

        # Initialize game with card distribution
        env.card_play_init(card_play_data)

        # Play game until done
        while not env.game_over:
            env.step()

        # Record result
        if env.winner == 'landlord':
            stats['wins'] += 1 if self.trainer.position == 'landlord' else 0
            stats['losses'] += 1 if self.trainer.position != 'landlord' else 0
        else:  # farmer wins
            stats['wins'] += 1 if self.trainer.position != 'landlord' else 0
            stats['losses'] += 1 if self.trainer.position == 'landlord' else 0

        reward = self._compute_reward(env)
        reward_wp = self._compute_reward(env, mode='wp')
        reward_adp = self._compute_reward(env, mode='adp')

        stats['total_reward'] += reward
        stats['total_reward_wp'] += reward_wp
        stats['total_reward_adp'] += reward_adp

        return {
            'winner': env.winner,
            'reward': reward,
            'reward_wp': reward_wp,
            'reward_adp': reward_adp,
            'bomb_num': getattr(env, 'bomb_num', 0)
        }

    def collect_trajectories(self, num_games=100, opponent_type='perfectdou', opponent_map=None):
        """
        Collect training trajectories

        Args:
            num_games: Number of games to play
            opponent_type: Opponent type
            opponent_map: Optional dict overriding opponent type per position

        Returns:
            List of trajectory dicts
        """
        import torch

        # Set fusion network to eval mode during collection
        self.trainer.fusion_net.eval()

        trajectories = []

        game_data_subset = self.eval_data[:num_games]
        resolved_opponents = self._resolve_opponent_types(opponent_type, opponent_map)

        for idx, card_play_data in enumerate(game_data_subset):
            try:
                card_play_data_copy = copy.deepcopy(card_play_data)
                trajectory = self._collect_single_trajectory(card_play_data_copy, resolved_opponents)
                if trajectory is not None:
                    trajectories.append(trajectory)

                if (idx + 1) % 10 == 0 or (idx + 1) == num_games:
                    print(f"  Progress: {idx + 1}/{num_games} games, collected {len(trajectories)} valid trajectories", flush=True)
            except (RuntimeError, ValueError, IndexError):
                continue
            except Exception as e:
                if idx < 5:
                    print(f"  Unexpected error in game {idx}: {type(e).__name__}: {str(e)}")
                continue

        # Set back to train mode
        self.trainer.fusion_net.train()

        return trajectories

    def _collect_single_trajectory(self, card_play_data, opponent_types):
        """
        Collect trajectory from a single game

        Following DouZero's design: only store observations and actions,
        recompute values during training.

        Returns:
            Trajectory dict with:
                - obs_z: List of LSTM input tensors (state history)
                - obs_x: List of state feature tensors
                - action_indices: List of action indices
                - legal_actions: List of legal action lists
                - target: Final reward (repeated for all steps)
        """
        import torch
        from douzero.env.game import GameEnv
        from .opponent_loader import create_opponent_dict
        from douzero.env.env import get_obs

        opponent_types = opponent_types or {}

        trajectory = {
            'obs_z': [],      # LSTM input (state history)
            'obs_x': [],      # State features (no action)
            'action_indices': [],  # Selected action indices
            'legal_actions': [],   # Legal actions at each step
            'state_features': [],  # Cached fusion features
        }

        # Create a custom agent that records trajectory
        class TrajectoryRecordingAgent:
            def __init__(self, trainer, trajectory_buffer):
                self.trainer = trainer
                self.trajectory_buffer = trajectory_buffer

            def act(self, infoset):
                # Safety check: if no legal actions, skip this game gracefully
                if not infoset.legal_actions or len(infoset.legal_actions) == 0:
                    # Game is in invalid state, return empty action to trigger game end
                    # This will be caught by the exception handler
                    raise RuntimeError(f"Invalid game state: no legal actions for {infoset.player_position}")

                # Only record if it's our position's turn (forced move)
                if len(infoset.legal_actions) == 1:
                    return infoset.legal_actions[0]

                try:
                    # Get observation using DouZero's encoding
                    obs = get_obs(infoset)

                    with torch.no_grad():
                        # Get dual model outputs for action selection
                        dual_outputs = self.trainer.dual_inference.get_dual_action_values(infoset)
                        values_a = self.trainer.dual_inference.format_action_values(dual_outputs['values_a'])
                        values_b = self.trainer.dual_inference.format_action_values(dual_outputs['values_b'])

                        # Safety check: if no actions available, use fallback (first action)
                        if values_a.numel() == 0 or values_b.numel() == 0:
                            return infoset.legal_actions[0]

                        # Extract state features for fusion network
                        state_features = self.trainer.feature_extractor.extract_features(
                            infoset, self.trainer.position, dual_outputs
                        )

                        if state_features.dim() == 1:
                            state_features = state_features.unsqueeze(0)
                        state_features = state_features.to(self.trainer.device)

                        # Get fusion network output
                        fusion_output = self.trainer.fusion_net(values_a, values_b, state_features)

                    # Select action
                    if 'action_probs' in fusion_output:
                        probs = fusion_output['action_probs']
                        if probs.dim() > 1:
                            probs = probs.squeeze(0)
                        if probs.numel() == 0:  # Safety check
                            return infoset.legal_actions[0]
                        action_idx = torch.multinomial(probs, 1).item()
                    else:
                        fused_values = fusion_output['fused_values']
                        if fused_values.dim() > 1:
                            fused_values = fused_values.squeeze(0)
                        if fused_values.numel() == 0:  # Safety check
                            return infoset.legal_actions[0]
                        action_idx = torch.argmax(fused_values).item()

                    # Clamp action_idx to valid range
                    action_idx = min(action_idx, len(infoset.legal_actions) - 1)
                    action_idx = max(action_idx, 0)

                except Exception as e:
                    # If anything goes wrong, fallback to first legal action
                    print(f"Warning in trajectory collection: {e}")
                    return infoset.legal_actions[0]

                # Record to trajectory (DouZero style)
                self.trajectory_buffer['obs_z'].append(torch.from_numpy(obs['z_batch']).float())
                self.trajectory_buffer['obs_x'].append(torch.from_numpy(obs['x_batch']).float())
                self.trajectory_buffer['action_indices'].append(action_idx)
                self.trajectory_buffer['legal_actions'].append(infoset.legal_actions.copy())
                self.trajectory_buffer['state_features'].append(state_features.squeeze(0).cpu())

                # Return action
                return infoset.legal_actions[action_idx]

        # Create recording agent for our position
        recording_agent = TrajectoryRecordingAgent(self.trainer, trajectory)

        # Create opponents
        opponents = create_opponent_dict(
            landlord_type=opponent_types.get('landlord'),
            landlord_up_type=opponent_types.get('landlord_up'),
            landlord_down_type=opponent_types.get('landlord_down')
        )

        # Replace our position with recording agent
        opponents[self.trainer.position] = recording_agent

        # Create game environment
        env = GameEnv(opponents)

        # Initialize game
        env.card_play_init(card_play_data)

        # Play game
        while not env.game_over:
            env.step()

        final_reward = self._compute_reward(env)
        reward_wp = self._compute_reward(env, mode='wp')
        reward_adp = self._compute_reward(env, mode='adp')

        num_steps = len(trajectory['action_indices'])
        if num_steps == 0:
            return None

        trajectory['target'] = final_reward
        trajectory['reward_wp'] = reward_wp
        trajectory['reward_adp'] = reward_adp
        trajectory['num_steps'] = num_steps

        return trajectory


class SimpleActionEvaluator:
    """
    Simple evaluator for quick testing
    """

    def __init__(self):
        pass

    def compare_fusion_methods(self, model_path_a, model_path_b, position, eval_data_path):
        """
        Compare different fusion methods

        Args:
            model_path_a: Path to model A
            model_path_b: Path to model B
            position: Position
            eval_data_path: Evaluation data path

        Returns:
            Comparison results
        """
        from .dual_model import EnsemblePredictor

        results = {}

        # Test different ensemble methods
        for method in ['average', 'max', 'vote']:
            print(f"\nTesting ensemble method: {method}")
            predictor = EnsemblePredictor(
                model_path_a, model_path_b, position, method
            )

            # Evaluate (placeholder)
            result = {
                'method': method,
                'win_rate': 0.0,  # To be implemented
                'avg_reward': 0.0  # To be implemented
            }

            results[method] = result

        return results

    def analyze_model_disagreement(self, model_path_a, model_path_b, position, eval_data_path):
        """
        Analyze how often two models disagree

        Args:
            model_path_a: Path to model A
            model_path_b: Path to model B
            position: Position
            eval_data_path: Evaluation data path

        Returns:
            Disagreement analysis
        """
        from .dual_model import DualModelInference

        dual_inference = DualModelInference(model_path_a, model_path_b, position)

        disagreement_stats = {
            'total_decisions': 0,
            'same_top_action': 0,
            'top_5_overlap': [],
            'value_differences': []
        }

        # Load eval data
        with open(eval_data_path, 'rb') as f:
            eval_data = pickle.load(f)

        # Analyze disagreements
        for card_play_data in eval_data[:100]:  # Sample 100 games
            # TODO: Implement disagreement analysis
            # This requires environment integration
            pass

        # Calculate statistics
        if disagreement_stats['total_decisions'] > 0:
            disagreement_stats['agreement_rate'] = (
                disagreement_stats['same_top_action'] / disagreement_stats['total_decisions']
            )
            disagreement_stats['avg_top5_overlap'] = np.mean(disagreement_stats['top_5_overlap'])
            disagreement_stats['avg_value_diff'] = np.mean(disagreement_stats['value_differences'])

        return disagreement_stats
