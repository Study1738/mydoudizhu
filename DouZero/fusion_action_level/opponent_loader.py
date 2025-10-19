"""
Opponent Loader

Loads different types of opponents for training/evaluation.
"""

import os
import sys
import numpy as np

try:
    import onnxruntime as ort
except ImportError:
    ort = None

from douzero.env.env import get_obs

encode_obs_landlord = None
encode_obs_peasant = None
_decode_action = None

# Project root (DouZero/)
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Add project root for imports
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)


def load_opponent_agent(opponent_type, position):
    """
    Load an opponent agent

    Args:
        opponent_type: Type of opponent ('perfectdou', 'douzero', 'random', 'rlcard')
        position: Position of the agent

    Returns:
        Agent object with act(infoset) method
    """
    if opponent_type == 'random':
        from douzero.evaluation.random_agent import RandomAgent
        return RandomAgent()

    elif opponent_type == 'rlcard':
        from douzero.evaluation.rlcard_agent import RLCardAgent
        return RLCardAgent(position)

    elif opponent_type == 'douzero':
        # Load default DouZero model (ADP version)
        from douzero.evaluation.deep_agent import DeepAgent
        model_path = os.path.join(ROOT_DIR, 'model', 'douzero_ADP', f'{position}.ckpt')
        return DeepAgent(position, model_path)

    elif opponent_type == 'perfectdou':
        # Load PerfectDou agent
        return load_perfectdou_agent(position)

    else:
        # Treat explicit file paths separately
        if isinstance(opponent_type, str):
            path = os.path.abspath(opponent_type)
            if os.path.exists(path):
                lower = path.lower()
                if lower.endswith('.onnx'):
                    if ort is None:
                        raise ImportError(
                            "onnxruntime is required to load PerfectDou ONNX models. "
                            "Install onnxruntime or provide a DouZero checkpoint instead."
                        )
                    return _PerfectDouOnnxAgent(path, position)
                if lower.endswith(('.pt', '.ckpt')):
                    if 'perfectdou' in lower.replace('\\', '/'):
                        try:
                            from perfectdou.evaluation.perfectdou_agent import PerfectDouAgent
                        except ImportError as exc:
                            raise ImportError(
                                "PerfectDouAgent is required to load PerfectDou checkpoint files. "
                                "Ensure the PerfectDou package is installable on PYTHONPATH."
                            ) from exc
                        model_dir = os.path.dirname(path)
                        if not os.path.isdir(model_dir):
                            raise FileNotFoundError(f"Directory for PerfectDou checkpoint not found: {model_dir}")
                        return PerfectDouAgent(position, model_dir)
                    from douzero.evaluation.deep_agent import DeepAgent
                    return DeepAgent(position, path)
                from douzero.evaluation.deep_agent import DeepAgent
                return DeepAgent(position, path)

        # Fallback: assume keyword or DeepAgent path
        from douzero.evaluation.deep_agent import DeepAgent
        return DeepAgent(position, opponent_type)


# Global flag to only warn once about PerfectDou
_perfectdou_warning_shown = False

def load_perfectdou_agent(position):
    """
    Load PerfectDou agent

    Args:
        position: Position of the agent

    Returns:
        PerfectDou agent
    """
    global _perfectdou_warning_shown

    model_path = _find_perfectdou_model(position)

    # Prefer ONNX when available
    if model_path and model_path.endswith('.onnx'):
        if ort is None:
            if not _perfectdou_warning_shown:
                print("Warning: Found PerfectDou ONNX weights but onnxruntime is not installed. Falling back to DouZero agent.")
                _perfectdou_warning_shown = True
        else:
            return _PerfectDouOnnxAgent(model_path, position)

    try:
        from perfectdou.evaluation.perfectdou_agent import PerfectDouAgent
        if model_path and model_path.endswith(('.pt', '.ckpt')):
            model_dir = os.path.dirname(model_path)
            return PerfectDouAgent(position, model_dir)
        # Let PerfectDouAgent resolve its own default paths
        return PerfectDouAgent(position)

    except (ImportError, FileNotFoundError, OSError) as e:
        if not _perfectdou_warning_shown:
            print(f"Warning: Could not load PerfectDou agent ({e}). Falling back to DouZero deep agent.")
            _perfectdou_warning_shown = True

        from douzero.evaluation.deep_agent import DeepAgent
        fallback_path = os.path.join(ROOT_DIR, 'model', 'douzero_ADP', f'{position}.ckpt')
        return DeepAgent(position, fallback_path)


def create_opponent_dict(landlord_type='perfectdou',
                         landlord_up_type='perfectdou',
                         landlord_down_type='perfectdou'):
    """
    Create a dictionary of opponent agents for all positions

    Args:
        landlord_type: Opponent type for landlord
        landlord_up_type: Opponent type for landlord_up
        landlord_down_type: Opponent type for landlord_down

    Returns:
        Dict mapping position to agent
    """
    return {
        'landlord': load_opponent_agent(landlord_type, 'landlord'),
        'landlord_up': load_opponent_agent(landlord_up_type, 'landlord_up'),
        'landlord_down': load_opponent_agent(landlord_down_type, 'landlord_down')
    }


def _find_perfectdou_model(position):
    """
    Locate PerfectDou model file for a given position.
    Supports .pt/.ckpt (PyTorch) and .onnx (ONNX) weights.
    """
    search_dirs = []

    env_dir = os.environ.get('PERFECTDOU_MODEL_DIR')
    if env_dir:
        search_dirs.append(env_dir)

    search_dirs.extend([
        os.path.join(ROOT_DIR, 'perfectdou', 'model'),
        os.path.join(ROOT_DIR, 'model', 'perfectdou')
    ])

    preferred_exts = ['.pt', '.ckpt', '.onnx']

    for directory in search_dirs:
        if not directory:
            continue
        for ext in preferred_exts:
            candidate = os.path.join(directory, f'{position}{ext}')
            if os.path.exists(candidate):
                return candidate
    return None


class _PerfectDouOnnxAgent:
    """ONNX-backed PerfectDou agent using onnxruntime."""

    def __init__(self, model_path, position):
        self.position = position
        self.model_path = model_path
        self.session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
        self.input_names = [inp.name for inp in self.session.get_inputs()]
        self.output_names = [out.name for out in self.session.get_outputs()]
        self._ensure_encoders()

        self.rlcard2env = {
            "3": 3,
            "4": 4,
            "5": 5,
            "6": 6,
            "7": 7,
            "8": 8,
            "9": 9,
            "T": 10,
            "J": 11,
            "Q": 12,
            "K": 13,
            "A": 14,
            "2": 17,
            "B": 20,
            "R": 30,
        }

    def _ensure_encoders(self):
        global encode_obs_landlord, encode_obs_peasant, _decode_action
        if encode_obs_landlord and encode_obs_peasant and _decode_action:
            return

        candidate_roots = []
        env_root = os.environ.get('PERFECTDOU_ROOT')
        if env_root:
            candidate_roots.append(env_root)

        candidate_roots.extend([
            os.path.join(ROOT_DIR, 'perfectdou'),
            os.path.join(os.path.dirname(ROOT_DIR), 'PerfectDou'),
            os.path.join(os.path.dirname(ROOT_DIR), 'perfectdou'),
        ])

        for root in candidate_roots:
            if not root:
                continue
            pkg_path = root
            if os.path.isdir(os.path.join(root, 'perfectdou')):
                pkg_path = root
            elif os.path.isdir(root) and os.path.basename(root) == 'perfectdou':
                pkg_path = os.path.dirname(root)
            if os.path.isdir(pkg_path) and pkg_path not in sys.path:
                sys.path.insert(0, pkg_path)

        try:
            from perfectdou.env.encode import encode_obs_landlord as _landlord_encoder
            from perfectdou.env.encode import encode_obs_peasant as _peasant_encoder
            from perfectdou.env.encode import _decode_action as _decoder
            encode_obs_landlord = _landlord_encoder
            encode_obs_peasant = _peasant_encoder
            _decode_action = _decoder
        except ImportError as exc:
            raise ImportError(
                "perfectdou.env.encode is required to run PerfectDou ONNX models. "
                "Ensure the PerfectDou package is installed or set PERFECTDOU_ROOT."
            ) from exc

    def act(self, infoset):
        legal_actions = getattr(infoset, 'legal_actions', [])
        if not legal_actions:
            return []
        if len(legal_actions) == 1:
            return legal_actions[0]

        if self.position == 'landlord':
            obs = encode_obs_landlord(infoset)
        else:
            obs = encode_obs_peasant(infoset)

        x_no_action = obs.get("x_no_action")
        legal_arr = obs.get("legal_actions_arr")
        actions = obs.get("actions")
        current_hand = obs.get("current_hand")

        if x_no_action is None or legal_arr is None or actions is None or current_hand is None:
            # Fallback: try to map back to legal action list
            return legal_actions[0]

        input_vec = np.concatenate(
            [np.asarray(x_no_action, dtype=np.float32).ravel(),
             np.asarray(legal_arr, dtype=np.float32).ravel()]
        ).reshape(1, -1)

        feeds = {self.input_names[0]: input_vec}
        output = self.session.run(self.output_names, feeds)[0]
        logit = np.asarray(output, dtype=np.float32).reshape(-1)
        action_id = int(np.argmax(logit))

        decoded = _decode_action(action_id, current_hand, actions)
        if decoded == "pass":
            return []
        converted = []
        for card in decoded:
            converted.append(self.rlcard2env.get(card, card))
        # Ensure the chosen action is legal; fallback otherwise
        if converted not in legal_actions:
            for action in legal_actions:
                if set(action) == set(converted):
                    return action
            return legal_actions[0]
        return converted
