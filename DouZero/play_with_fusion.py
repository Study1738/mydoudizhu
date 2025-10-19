"""
Play Doudizhu with trained fusion model

This script demonstrates how to use a trained fusion model in actual gameplay.
"""
import os
import argparse
import torch
import pickle
from douzero.env.game import GameEnv
from fusion_action_level.trainer import ActionFusionTrainer
from fusion_action_level.dual_model import DualModelAgent
from fusion_action_level.opponent_loader import create_opponent_dict


POSITIONS = ('landlord', 'landlord_up', 'landlord_down')


def load_fusion_agent(checkpoint_path, model_a_path, model_b_path, position, device='cuda'):
    """
    Load trained fusion agent for gameplay

    Args:
        checkpoint_path: Path to trained fusion checkpoint
        model_a_path: Path to base model A
        model_b_path: Path to base model B
        position: Player position
        device: Device

    Returns:
        DualModelAgent ready for gameplay
    """
    print(f"Loading fusion agent...")
    print(f"  Checkpoint: {checkpoint_path}")
    print(f"  Model A: {model_a_path}")
    print(f"  Model B: {model_b_path}")
    print(f"  Position: {position}")

    # Initialize trainer to get network architecture
    trainer = ActionFusionTrainer(
        model_path_a=model_a_path,
        model_path_b=model_b_path,
        position=position,
        fusion_type='gating',
        device=device
    )

    # Load trained weights
    checkpoint = torch.load(checkpoint_path, map_location=device)
    trainer.fusion_net.load_state_dict(checkpoint['fusion_net_state_dict'])
    trainer.fusion_net.eval()

    print(f"\nCheckpoint Info:")
    print(f"  Epoch: {checkpoint.get('epoch', 'N/A')}")
    if 'stats' in checkpoint and checkpoint['stats']:
        stats = checkpoint['stats']
        if 'win_rate' in stats:
            print(f"  Training Win Rate: {stats['win_rate']:.2%}")
        if 'avg_reward' in stats:
            print(f"  Training Avg Reward: {stats['avg_reward']:.4f}")

    # Create agent
    agent = DualModelAgent(
        model_path_a=model_a_path,
        model_path_b=model_b_path,
        position=position,
        fusion_network=trainer.fusion_net,
        device=device
    )

    print("[OK] Fusion agent loaded successfully!\n")
    return agent


def play_single_game(
        fusion_agents,
        primary_position,
        card_play_data,
        opponent_type='douzero',
        opponent_types=None):
    """
    Play a single game with fusion agent.
    MODIFIED: This version explicitly calculates and returns separate scores
    for the Landlord and the sum of both Farmers.
    """
    opponent_types = opponent_types or {}
    landlord_type = opponent_types.get('landlord', opponent_type)
    landlord_up_type = opponent_types.get('landlord_up', opponent_type)
    landlord_down_type = opponent_types.get('landlord_down', opponent_type)

    # Create opponents
    opponents = create_opponent_dict(
        landlord_type=landlord_type,
        landlord_up_type=landlord_up_type,
        landlord_down_type=landlord_down_type
    )

    # Replace with fusion agent
    for pos, agent in fusion_agents.items():
        opponents[pos] = agent

    # Create game
    env = GameEnv(opponents)

    # Initialize with specific card distribution
    env.card_play_init(card_play_data)

    # Play
    while not env.game_over:
        env.step()

    # ==================== MODIFICATION START ====================
    # Calculate separate scores for landlord and farmers
    multiplier = 2 ** env.bomb_num
    is_landlord_win = (env.winner == 'landlord')

    landlord_reward = (2 if is_landlord_win else -2) * multiplier
    farmers_total_reward = (2 if not is_landlord_win else -2) * multiplier

    # Determine win status from the perspective of the primary position
    if primary_position == 'landlord':
        win = is_landlord_win
    else:
        win = not is_landlord_win
    # ===================== MODIFICATION END =====================

    return {
        'winner': env.winner,
        'win': win,
        'landlord_reward': landlord_reward,
        'farmers_total_reward': farmers_total_reward,
        'bomb_num': env.bomb_num
    }


def play_games(
        fusion_agents,
        primary_position,
        eval_data_path,
        num_games=100,
        opponent_type='douzero',
        opponent_types=None):
    """
    Play multiple games and collect statistics.
    MODIFIED: Aggregates landlord and farmer scores separately.
    """
    # Load evaluation data
    print(f"Loading evaluation data from: {eval_data_path}")
    with open(eval_data_path, 'rb') as f:
        eval_data = pickle.load(f)

    print(f"Playing {num_games} games...")
    print(f"  Primary position: {primary_position}")
    print(f"  Opponent: {opponent_type}")
    if opponent_types:
        print(f"  Opponent overrides: {opponent_types}")
    print("=" * 60)

    wins = 0
    total_landlord_reward = 0.0
    total_farmers_reward = 0.0
    results = []

    for i in range(num_games):
        card_play_data = eval_data[i % len(eval_data)]

        try:
            import copy
            result = play_single_game(
                fusion_agents,
                primary_position,
                copy.deepcopy(card_play_data),
                opponent_type,
                opponent_types
            )

            if result['win']:
                wins += 1

            total_landlord_reward += result['landlord_reward']
            total_farmers_reward += result['farmers_total_reward']
            results.append(result)

            if (i + 1) % 500 == 0:
                current_wr = wins / (i + 1)
                current_landlord_ar = total_landlord_reward / (i + 1)
                current_farmer_ar = total_farmers_reward / (i + 1)
                print(f"  [{i+1}/{num_games}] Win Rate: {current_wr:.2%}, "
                      f"Landlord ADP: {current_landlord_ar:.4f}, Farmer ADP: {current_farmer_ar:.4f}")

        except Exception as e:
            print(f"  Error in game {i}: {type(e).__name__}")
            continue

    # Calculate final stats
    valid_games = len(results)
    win_rate = wins / valid_games if valid_games > 0 else 0.0
    avg_landlord_adp = total_landlord_reward / valid_games if valid_games > 0 else 0.0
    avg_farmers_adp = total_farmers_reward / valid_games if valid_games > 0 else 0.0

    return {
        'num_games': valid_games,
        'wins': wins,
        'losses': valid_games - wins,
        'win_rate': win_rate,
        'avg_landlord_adp': avg_landlord_adp,
        'avg_farmers_adp': avg_farmers_adp,
        'results': results
    }


def main():
    parser = argparse.ArgumentParser(description='Play Doudizhu with trained fusion model')

    # Fusion model
    parser.add_argument('--checkpoint', type=str,
                        default='checkpoints/action_fusion_landlordup_wp/best_fusion_wp.pt',
                        help='Path to fusion checkpoint')
    parser.add_argument('--model_a', type=str,
                        default='model/douzero_ADP/landlord_up.ckpt',
                        help='Path to base model A')
    parser.add_argument('--model_b', type=str,
                        default='model/douzero_WP/landlord_up.ckpt',
                        help='Path to base model B')

    # Game settings
    parser.add_argument('--position', type=str, default='landlord',
                        choices=['landlord', 'landlord_up', 'landlord_down'],
                        help='Position for fusion agent')
    parser.add_argument('--opponent', type=str, default='douzero',
                        choices=['douzero', 'perfectdou', 'random'],
                        help='Opponent type')
    parser.add_argument('--opponent_landlord', type=str, default=None,
                        help='Opponent override for landlord (type keyword or checkpoint path)')
    parser.add_argument('--opponent_landlord_up', type=str, default=None,
                        help='Opponent override for landlord_up')
    parser.add_argument('--opponent_landlord_down', type=str, default=None,
                        help='Opponent override for landlord_down')
    parser.add_argument('--eval_data', type=str, default='eval_data_copy.pkl',
                        help='Evaluation data path')
    parser.add_argument('--num_games', type=int, default=100,
                        help='Number of games to play')

    # Device
    parser.add_argument('--device', type=str,
                        default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device')

    # Optional per-position overrides
    for pos in POSITIONS:
        parser.add_argument(f'--{pos}_checkpoint', type=str, default=None,
                            help=f'Checkpoint for fusion agent at {pos} (overrides --checkpoint for that position)')
        parser.add_argument(f'--{pos}_model_a', type=str, default=None,
                            help=f'Model A path for {pos}')
        parser.add_argument(f'--{pos}_model_b', type=str, default=None,
                            help=f'Model B path for {pos}')

    args = parser.parse_args()

    print("=" * 60)
    print("Play Doudizhu with Fusion Model")
    print("=" * 60)

    def resolve_agent_args(position):
        return (
            getattr(args, f'{position}_checkpoint'),
            getattr(args, f'{position}_model_a'),
            getattr(args, f'{position}_model_b')
        )

    fusion_agents = {}

    # Primary position (always required)
    primary_ckpt, primary_a, primary_b = resolve_agent_args(args.position)
    fusion_agents[args.position] = load_fusion_agent(
        checkpoint_path=primary_ckpt or args.checkpoint,
        model_a_path=primary_a or args.model_a,
        model_b_path=primary_b or args.model_b,
        position=args.position,
        device=args.device
    )

    # Additional positions (optional overrides)
    for pos in POSITIONS:
        if pos == args.position:
            continue
        ckpt, model_a, model_b = resolve_agent_args(pos)
        if ckpt and model_a and model_b:
            fusion_agents[pos] = load_fusion_agent(
                checkpoint_path=ckpt,
                model_a_path=model_a,
                model_b_path=model_b,
                position=pos,
                device=args.device
            )

    opponent_overrides = {
        'landlord': args.opponent_landlord,
        'landlord_up': args.opponent_landlord_up,
        'landlord_down': args.opponent_landlord_down
    }
    opponent_overrides = {k: v for k, v in opponent_overrides.items() if v is not None}

    # Play games
    stats = play_games(
        fusion_agents=fusion_agents,
        primary_position=args.position,
        eval_data_path=args.eval_data,
        num_games=args.num_games,
        opponent_type=args.opponent,
        opponent_types=opponent_overrides
    )

    # ==================== MODIFICATION START ====================
    # Print results with separate scores
    print("\n" + "=" * 60)
    print("GAMEPLAY RESULTS")
    print("=" * 60)
    print(f"Total Games Played: {stats['num_games']}")
    print(f"Primary Position ({args.position}) Wins: {stats['wins']} ({stats['win_rate']:.2%})")
    print("-" * 60)
    print(f"Avg Landlord ADP: {stats['avg_landlord_adp']:.4f}")
    print(f"Avg Farmer (Total) ADP: {stats['avg_farmers_adp']:.4f}")
    print("=" * 60)

    # Show sample game results
    if stats['results']:
        print("\nSample Game Results (last 5):")
        for i, result in enumerate(stats['results'][-5:]):
            game_num = stats['num_games'] - 5 + i + 1
            win_symbol = "[WIN]" if result['win'] else "[LOSS]"
            print(f"  Game {game_num}: {win_symbol} Winner={result['winner']}, "
                  f"LL Reward={result['landlord_reward']:.2f}, "
                  f"Farmer Reward={result['farmers_total_reward']:.2f}, "
                  f"Bombs={result['bomb_num']}")
    # ===================== MODIFICATION END =====================


if __name__ == '__main__':
    main()