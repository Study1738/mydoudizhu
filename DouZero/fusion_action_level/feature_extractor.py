"""
Feature Extractor for Action-Level Fusion

Extracts features from game state for the fusion network.
Similar to weight-level but may include additional action-specific features.
"""

import numpy as np
import torch
from collections import Counter


class ActionLevelFeatureExtractor:
    """
    Extracts features for action-level fusion network
    """

    def __init__(self, feature_dim=512):
        """
        Args:
            feature_dim: Output feature dimension
        """
        self.feature_dim = feature_dim

    def extract_features(self, infoset, position, model_outputs=None):
        """
        Extract comprehensive features

        Args:
            infoset: Game information set
            position: Current position
            model_outputs: Optional dict with model A and B outputs for feature extraction

        Returns:
            Feature tensor of shape (feature_dim,)
        """
        features = []

        # 1. Basic state features
        hand_features = self._encode_hand_cards(infoset.player_hand_cards)
        features.append(hand_features)

        # 2. Game phase
        phase_features = self._encode_game_phase(infoset)
        features.append(phase_features)

        # 3. Position encoding
        position_features = self._encode_position(position)
        features.append(position_features)

        # 4. Action history
        history_features = self._encode_action_history(infoset)
        features.append(history_features)

        # 5. Opponent modeling
        opponent_features = self._encode_opponents(infoset)
        features.append(opponent_features)

        # 6. Strategic features
        strategic_features = self._encode_strategic_features(infoset)
        features.append(strategic_features)

        # 7. Model disagreement features (if model outputs provided)
        if model_outputs is not None:
            disagreement_features = self._encode_model_disagreement(model_outputs)
            features.append(disagreement_features)

        # Concatenate and pad/truncate
        all_features = np.concatenate(features)

        if len(all_features) < self.feature_dim:
            all_features = np.pad(all_features, (0, self.feature_dim - len(all_features)))
        else:
            all_features = all_features[:self.feature_dim]

        return torch.tensor(all_features, dtype=torch.float32)

    def _encode_hand_cards(self, hand_cards):
        """Encode hand cards distribution"""
        # Card type counts
        card_counts = np.zeros(15)  # 13 ranks + 2 jokers

        for card in hand_cards:
            if 3 <= card <= 14:
                card_counts[card - 3] += 1
            elif card == 17:  # 2
                card_counts[11] += 1
            elif card == 20:  # Black Joker
                card_counts[13] += 1
            elif card == 30:  # Red Joker
                card_counts[14] += 1

        # Normalize
        card_counts = card_counts / 4.0

        # Card combinations
        counter = Counter(hand_cards)
        num_singles = sum(1 for c in counter.values() if c == 1)
        num_pairs = sum(1 for c in counter.values() if c == 2)
        num_triples = sum(1 for c in counter.values() if c == 3)
        num_bombs = sum(1 for c in counter.values() if c == 4)

        combo_features = np.array([
            num_singles / 20.0,
            num_pairs / 10.0,
            num_triples / 5.0,
            num_bombs / 4.0,
            len(hand_cards) / 20.0
        ])

        return np.concatenate([card_counts, combo_features])

    def _encode_game_phase(self, infoset):
        """Encode game phase indicators"""
        total_played = len(infoset.played_cards) if hasattr(infoset, 'played_cards') else 0
        phase_ratio = total_played / 54.0

        # Multi-hot encoding for fine-grained phases
        phases = np.array([
            1.0 if phase_ratio < 0.25 else 0.0,  # Very early
            1.0 if 0.25 <= phase_ratio < 0.5 else 0.0,  # Early
            1.0 if 0.5 <= phase_ratio < 0.75 else 0.0,  # Mid
            1.0 if phase_ratio >= 0.75 else 0.0,  # Late
            phase_ratio  # Continuous value
        ])

        return phases

    def _encode_position(self, position):
        """One-hot position encoding"""
        if position == 'landlord':
            return np.array([1, 0, 0], dtype=np.float32)
        elif position == 'landlord_up':
            return np.array([0, 1, 0], dtype=np.float32)
        else:
            return np.array([0, 0, 1], dtype=np.float32)

    def _encode_action_history(self, infoset):
        """Encode recent action history"""
        # Count card types played
        if hasattr(infoset, 'played_cards') and infoset.played_cards:
            # Flatten if nested list
            flat_cards = []
            for item in infoset.played_cards:
                if isinstance(item, list):
                    flat_cards.extend(item)
                else:
                    flat_cards.append(item)

            played_counter = Counter(flat_cards)

            played_counts = np.zeros(15)
            for card, count in played_counter.items():
                # Convert to int if needed
                try:
                    card = int(card)
                except (ValueError, TypeError):
                    continue

                if 3 <= card <= 14:
                    played_counts[card - 3] += count
                elif card == 17:
                    played_counts[11] += count
                elif card == 20:
                    played_counts[13] += count
                elif card == 30:
                    played_counts[14] += count

            played_counts = played_counts / max(played_counts.max(), 1.0)
        else:
            played_counts = np.zeros(15)

        # Recent action patterns
        total_cards = 0
        if hasattr(infoset, 'played_cards') and infoset.played_cards:
            for item in infoset.played_cards:
                if isinstance(item, list):
                    total_cards += len(item)
                else:
                    total_cards += 1

        recent_patterns = np.array([total_cards / 54.0])

        return np.concatenate([played_counts, recent_patterns])

    def _encode_opponents(self, infoset):
        """Encode opponent state information"""
        features = []

        if hasattr(infoset, 'num_cards_left'):
            for pos in ['landlord', 'landlord_up', 'landlord_down']:
                if pos in infoset.num_cards_left:
                    cards_left = infoset.num_cards_left[pos]
                    features.extend([
                        cards_left / 20.0,
                        1.0 if cards_left <= 5 else 0.0,  # Critical state
                        1.0 if cards_left == 1 else 0.0   # About to win
                    ])
                else:
                    features.extend([0.0, 0.0, 0.0])
        else:
            features = [0.0] * 9

        return np.array(features, dtype=np.float32)

    def _encode_strategic_features(self, infoset):
        """Encode strategic situation features"""
        features = []

        # Control status
        has_control = 1.0 if hasattr(infoset, 'last_action') and infoset.last_action is None else 0.0
        features.append(has_control)

        # Bomb availability
        if hasattr(infoset, 'player_hand_cards'):
            hand_counter = Counter(infoset.player_hand_cards)
            num_bombs = sum(1 for c in hand_counter.values() if c == 4)
            has_rocket = 1.0 if (20 in infoset.player_hand_cards and 30 in infoset.player_hand_cards) else 0.0

            features.extend([
                num_bombs / 4.0,
                has_rocket
            ])
        else:
            features.extend([0.0, 0.0])

        # Winning probability estimate (simple heuristic)
        if hasattr(infoset, 'num_cards_left') and hasattr(infoset, 'player_position'):
            my_cards = infoset.num_cards_left.get(infoset.player_position, 20)
            opponent_min_cards = min([
                infoset.num_cards_left.get(pos, 20)
                for pos in ['landlord', 'landlord_up', 'landlord_down']
                if pos != infoset.player_position
            ])
            win_urgency = 1.0 if opponent_min_cards <= 3 else 0.0
            features.append(win_urgency)
        else:
            features.append(0.0)

        return np.array(features, dtype=np.float32)

    def _encode_model_disagreement(self, model_outputs):
        """
        Encode disagreement between two models

        Args:
            model_outputs: dict with 'values_a' and 'values_b'

        Returns:
            Disagreement features
        """
        values_a = model_outputs.get('values_a')
        values_b = model_outputs.get('values_b')

        if isinstance(values_a, torch.Tensor):
            values_a = values_a.detach().cpu().numpy()
        if isinstance(values_b, torch.Tensor):
            values_b = values_b.detach().cpu().numpy()

        values_a = np.asarray(values_a, dtype=np.float32).flatten()
        values_b = np.asarray(values_b, dtype=np.float32).flatten()

        # Handle edge case of empty values
        if values_a.size == 0 or values_b.size == 0:
            return np.zeros(7, dtype=np.float32)

        # Compute disagreement metrics
        # 1. Action ranking disagreement
        top_k = min(5, len(values_a))
        if top_k == 0:
            return np.zeros(7, dtype=np.float32)

        top_actions_a = np.argsort(values_a)[-top_k:]
        top_actions_b = np.argsort(values_b)[-top_k:]
        # Convert to int for hashability
        ranking_agreement = len(set(top_actions_a.tolist()) & set(top_actions_b.tolist())) / top_k

        # 2. Value difference statistics
        value_diff = np.abs(values_a - values_b)
        mean_diff = np.mean(value_diff)
        max_diff = np.max(value_diff)
        std_diff = np.std(value_diff)

        # 3. Confidence indicators
        confidence_a = np.max(values_a) - np.mean(values_a)
        confidence_b = np.max(values_b) - np.mean(values_b)

        features = np.array([
            ranking_agreement,
            mean_diff,
            max_diff,
            std_diff,
            confidence_a,
            confidence_b,
            np.abs(confidence_a - confidence_b)
        ], dtype=np.float32)

        return features


class BatchActionFeatureExtractor:
    """
    Batch feature extractor for efficient processing
    """

    def __init__(self, feature_dim=512):
        self.extractor = ActionLevelFeatureExtractor(feature_dim)

    def extract_batch(self, infosets, positions, model_outputs_list=None):
        """
        Extract features for a batch

        Args:
            infosets: List of infosets
            positions: List of positions
            model_outputs_list: Optional list of model output dicts

        Returns:
            Batched feature tensor
        """
        features_list = []

        for i, (infoset, position) in enumerate(zip(infosets, positions)):
            model_outputs = model_outputs_list[i] if model_outputs_list else None
            features = self.extractor.extract_features(infoset, position, model_outputs)
            features_list.append(features)

        return torch.stack(features_list)
