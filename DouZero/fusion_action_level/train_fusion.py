"""
Training Script for Action-Level Fusion

Main entry point for training action-level fusion networks.
"""

import os
import argparse
import torch
from datetime import datetime

from .trainer import ActionFusionTrainer

POSITIONS = ('landlord', 'landlord_up', 'landlord_down')


def parse_args():
    parser = argparse.ArgumentParser(description='Train Action-Level Fusion Network')

    # Model paths
    parser.add_argument('--model_a', type=str, required=True,
                        help='Path to first model checkpoint')
    parser.add_argument('--model_b', type=str, required=True,
                        help='Path to second model checkpoint')
    parser.add_argument('--position', type=str, required=True,
                        choices=['landlord', 'landlord_up', 'landlord_down'],
                        help='Position to train for')

    # Fusion network type
    parser.add_argument('--fusion_type', type=str, default='gating',
                        choices=['gating'],
                        help='Type of fusion network')

    # Training parameters
    parser.add_argument('--num_iterations', type=int, default=20,
                        help='Number of training iterations')
    parser.add_argument('--games_per_iteration', type=int, default=1000,
                        help='Games per iteration')
    parser.add_argument('--eval_interval', type=int, default=1,
                        help='Evaluation interval')
    parser.add_argument('--eval_games', type=int, default=1000,
                        help='Number of evaluation games')

    # Reward objective
    parser.add_argument('--objective', type=str, default='wp',
                        choices=['adp', 'wp', 'logadp'],
                        help='Reward objective to optimize')

    # Gating dynamics
    parser.add_argument('--gate_lr_scale', type=float, default=0.05,
                        help='Learning rate multiplier for gating fusion network')
    parser.add_argument('--gate_warmup_iters', type=int, default=3,
                        help='Number of initial update steps to freeze gate parameters')
    parser.add_argument('--gate_reg_coeff', type=float, default=1e-3,
                        help='Quadratic regularisation strength pulling gate towards its initial value')

    # Network parameters
    parser.add_argument('--num_actions', type=int, default=309,
                        help='Number of actions')
    parser.add_argument('--feature_dim', type=int, default=512,
                        help='Feature dimension')
    parser.add_argument('--lr', type=float, default=3e-4,
                        help='Learning rate')
    parser.add_argument('--gamma', type=float, default=0.99,
                        help='Discount factor')
    parser.add_argument('--gae_lambda', type=float, default=0.95,
                        help='GAE lambda')
    parser.add_argument('--clip_epsilon', type=float, default=0.2,
                        help='PPO clip epsilon')
    parser.add_argument('--value_loss_coef', type=float, default=0.5,
                        help='Value loss coefficient')
    parser.add_argument('--entropy_coef', type=float, default=0.01,
                        help='Entropy coefficient')

    # Opponent
    parser.add_argument('--opponent', type=str, default='perfectdou',
                        choices=['perfectdou', 'douzero', 'random', 'rlcard'],
                        help='Opponent type')
    parser.add_argument('--opponent_landlord', type=str, default=None,
                        help='Override opponent for landlord position (path or keyword)')
    parser.add_argument('--opponent_landlord_up', type=str, default=None,
                        help='Override opponent for landlord_up position')
    parser.add_argument('--opponent_landlord_down', type=str, default=None,
                        help='Override opponent for landlord_down position')

    # Evaluation data
    parser.add_argument('--eval_data', type=str, default='eval_data_copy.pkl',
                        help='Evaluation data path')

    # Device
    parser.add_argument('--device', type=str,
                        default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Training device')

    # Save settings
    parser.add_argument('--save_dir', type=str, default='checkpoints/action_fusion_test',
                        help='Save directory')
    parser.add_argument('--save_interval', type=int, default=100,
                        help='Save interval')
    parser.add_argument('--resume', type=str, default=None,
                        help='Resume from checkpoint')

    return parser.parse_args()


def train(args):
    """Main training loop"""

    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)

    # Initialize trainer
    print(f"Initializing Action-Level Fusion Trainer...")
    print(f"  Model A: {args.model_a}")
    print(f"  Model B: {args.model_b}")
    print(f"  Position: {args.position}")
    print(f"  Fusion Type: {args.fusion_type}")
    print(f"  Device: {args.device}")
    print(f"  Objective: {args.objective}")
    if args.fusion_type == 'gating':
        print(f"  Gate LR scale: {args.gate_lr_scale}")
        print(f"  Gate warmup iters: {args.gate_warmup_iters}")
        print(f"  Gate reg coeff: {args.gate_reg_coeff}")

    opponent_map = {
        'landlord': args.opponent_landlord,
        'landlord_up': args.opponent_landlord_up,
        'landlord_down': args.opponent_landlord_down
    }
    opponent_map = {k: v for k, v in opponent_map.items() if v is not None}

    trainer = ActionFusionTrainer(
        model_path_a=args.model_a,
        model_path_b=args.model_b,
        position=args.position,
        fusion_type=args.fusion_type,
        num_actions=args.num_actions,
        objective=args.objective,
        feature_dim=args.feature_dim,
        lr=args.lr,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_epsilon=args.clip_epsilon,
        value_loss_coef=args.value_loss_coef,
        entropy_coef=args.entropy_coef,
        device=args.device,
        gate_lr_scale=args.gate_lr_scale,
        gate_warmup_iters=args.gate_warmup_iters,
        gate_reg_coeff=args.gate_reg_coeff
    )

    # Resume if specified
    start_iteration = 0
    if args.resume:
        print(f"Resuming from: {args.resume}")
        start_iteration, _ = trainer.load_checkpoint(args.resume)
        start_iteration += 1

    # Training loop
    print(f"\nStarting training for {args.num_iterations} iterations...")
    print(f"  Games per iteration: {args.games_per_iteration}")
    print(f"  Opponent: {args.opponent}")
    if opponent_map:
        print("  Opponent overrides:")
        for pos in POSITIONS:
            if pos in opponent_map:
                print(f"    {pos}: {opponent_map[pos]}")
    print("-" * 60)

    # Best metric tracking based on objective
    if args.objective == 'adp' or args.objective == 'logadp':
        best_metric = float('-inf')  # ADP/logADP can be negative, maximize it
        metric_name = args.objective.upper()
    else:  # wp
        best_metric = 0.0  # Win rate, maximize it
        metric_name = 'Win Rate'

    print(f"Best model selection criterion: {metric_name}")
    print("-" * 60)

    # Create evaluator for trajectory collection
    from .evaluator import ActionFusionEvaluator

    evaluator = ActionFusionEvaluator(
        trainer=trainer,
        eval_data_path=args.eval_data,
        device=args.device,
        objective=args.objective,
        opponent_map=opponent_map
    )

    # Tracking stats
    recent_rewards = []
    recent_rewards_wp = []
    recent_rewards_adp = []
    recent_policy_losses = []
    recent_value_losses = []

    for iteration in range(start_iteration, args.num_iterations):
        print(f"\n=== Iteration {iteration + 1}/{args.num_iterations} ===")

        # Collect trajectories
        print(f"Collecting {args.games_per_iteration} game trajectories...")
        trajectories = evaluator.collect_trajectories(
            num_games=args.games_per_iteration,
            opponent_type=args.opponent,
            opponent_map=opponent_map
        )

        if not trajectories:
            print("  Warning: No trajectories collected, skipping update")
            continue

        print(f"  Collected {len(trajectories)} valid trajectories")

        # Compute average reward (target is the final reward for each trajectory)
        avg_reward = sum([t['target'] for t in trajectories]) / len(trajectories)
        avg_reward_wp = sum([t.get('reward_wp', 0.0) for t in trajectories]) / len(trajectories)
        avg_reward_adp = sum([t.get('reward_adp', 0.0) for t in trajectories]) / len(trajectories)
        recent_rewards.append(avg_reward)
        recent_rewards_wp.append(avg_reward_wp)
        recent_rewards_adp.append(avg_reward_adp)
        print(f"  Avg trajectory reward: {avg_reward:.4f}")

        # Update network
        print("Updating fusion network...")
        update_stats = trainer.update(trajectories)

        # Record stats
        recent_policy_losses.append(update_stats['policy_loss'])
        recent_value_losses.append(update_stats['value_loss'])

        # Print training stats
        print(f"  Policy Loss: {update_stats['policy_loss']:.4f}")
        print(f"  Value Loss: {update_stats['value_loss']:.4f}")
        print(f"  Total Loss: {update_stats['total_loss']:.4f}")
        if 'gate_reg_loss' in update_stats:
            print(f"  Gate Reg Loss: {update_stats['gate_reg_loss']:.6f}")

        # Print fusion gate weights (ADP vs WP) - ALWAYS print
        if 'avg_gate_weight' in update_stats:
            gate_weight = update_stats['avg_gate_weight']
            adp_weight = gate_weight * 100
            wp_weight = (1 - gate_weight) * 100
            print(f"  Fusion Weights: ADP={adp_weight:.1f}% | WP={wp_weight:.1f}%")
        else:
            # If no gate weight info available, try to extract from fusion network
            print(f"  Fusion Weights: (not using gating fusion)")

        # Evaluation - now run every iteration
        if True:  # Changed from: (iteration + 1) % args.eval_interval == 0
            print(f"\n--- Evaluation at iteration {iteration + 1} ---")

            # Evaluate against opponent
            print(f"Evaluating on {args.eval_games} games...")
            eval_stats = evaluator.evaluate(
                num_games=args.eval_games,
                opponent_type=args.opponent,
                opponent_map=opponent_map
            )

            obj_label = args.objective.upper()
            print(f"  Evaluation Results:")
            print(f"    Win Rate: {eval_stats['win_rate']:.2%}")
            print(f"    Wins: {eval_stats['wins']}, Losses: {eval_stats['losses']}")
            print(f"    Avg Reward (objective {obj_label}): {eval_stats['avg_reward']:.4f}")
            wp_metric = eval_stats.get('avg_reward_wp', 0.0)
            wp_win_rate = max(0.0, min((wp_metric + 1.0) / 2.0, 1.0))
            print(f"    WP Win Rate (metric derived): {wp_win_rate:.2%} (raw {wp_metric:.4f})")
            print(f"    Avg ADP: {eval_stats.get('avg_reward_adp', 0.0):.4f}")

            # Print averaged training stats
            if recent_policy_losses:
                window = min(args.eval_interval, len(recent_policy_losses))
                avg_policy_loss = sum(recent_policy_losses[-window:]) / window
                avg_value_loss = sum(recent_value_losses[-window:]) / window
                print(f"  Training Stats (last {window} iters):")
                print(f"    Avg Policy Loss: {avg_policy_loss:.4f}")
                print(f"    Avg Value Loss: {avg_value_loss:.4f}")

            if recent_rewards:
                window = min(args.eval_interval, len(recent_rewards))
                avg_reward = sum(recent_rewards[-window:]) / window
                print(f"    Avg Training Reward: {avg_reward:.4f}")

            # Determine current metric based on objective
            if args.objective == 'wp':
                current_metric = eval_stats['win_rate']
            elif args.objective == 'adp':
                current_metric = eval_stats.get('avg_reward_adp', eval_stats['avg_reward'])
            else:  # logadp
                current_metric = eval_stats['avg_reward']

            print(f"  >> Current {metric_name}: {current_metric:.4f}")

            # Save best model based on objective
            if current_metric > best_metric:
                best_metric = current_metric
                best_path = os.path.join(args.save_dir, f'best_fusion_{args.objective}.pt')
                trainer.save_checkpoint(best_path, iteration, eval_stats)
                print(f"  ✓ New best model saved! {metric_name}: {best_metric:.4f}")
                print(f"    Saved to: {best_path}")

        # Save checkpoint
        if (iteration + 1) % args.save_interval == 0:
            checkpoint_path = os.path.join(
                args.save_dir,
                f'checkpoint_iter_{iteration + 1}.pt'
            )
            trainer.save_checkpoint(checkpoint_path, iteration)
            print(f"Checkpoint saved to {checkpoint_path}")

    # Save final
    final_path = os.path.join(args.save_dir, 'final_fusion.pt')
    trainer.save_checkpoint(final_path, args.num_iterations - 1)

    # Print final summary
    print("\n" + "=" * 60)
    print("TRAINING SUMMARY")
    print("=" * 60)
    print(f"Total Iterations: {args.num_iterations}")
    print(f"Training Objective: {args.objective.upper()}")
    print(f"Best {metric_name}: {best_metric:.4f}")
    print(f"Best model saved to: {os.path.join(args.save_dir, f'best_fusion_{args.objective}.pt')}")

    if recent_policy_losses:
        print(f"\nLoss Statistics:")
        print(f"  Final Policy Loss: {recent_policy_losses[-1]:.4f}")
        print(f"  Final Value Loss: {recent_value_losses[-1]:.4f}")
        window = min(100, len(recent_policy_losses))
        print(f"  Avg Policy Loss (last {window}): {sum(recent_policy_losses[-window:]) / window:.4f}")
        print(f"  Avg Value Loss (last {window}): {sum(recent_value_losses[-window:]) / window:.4f}")

    if recent_rewards:
        window = min(100, len(recent_rewards))
        obj_label = args.objective.upper()
        print(f"\nReward Statistics:")
        print(f"  Final Avg Reward (objective {obj_label}): {recent_rewards[-1]:.4f}")
        print(f"  Avg Reward (objective {obj_label} last {window}): {sum(recent_rewards[-window:]) / window:.4f}")
        if len(recent_rewards) > 1:
            print(f"  Improvement: {recent_rewards[-1] - recent_rewards[0]:+.4f}")

    if recent_rewards_wp:
        window_wp = min(100, len(recent_rewards_wp))
        print(f"  Final Avg Reward (WP metric): {recent_rewards_wp[-1]:.4f}")
        print(f"  Avg Reward (WP metric last {window_wp}): {sum(recent_rewards_wp[-window_wp:]) / window_wp:.4f}")
    if recent_rewards_adp:
        window_adp = min(100, len(recent_rewards_adp))
        print(f"  Final Avg Reward (ADP metric): {recent_rewards_adp[-1]:.4f}")
        print(f"  Avg Reward (ADP metric last {window_adp}): {sum(recent_rewards_adp[-window_adp:]) / window_adp:.4f}")

    print("=" * 60)
    print(f"\nTraining completed! Final model saved to {final_path}")
    print(f"Best model saved to {os.path.join(args.save_dir, f'best_fusion_{args.objective}.pt')}")


if __name__ == '__main__':
    args = parse_args()

    print("=" * 60)
    print("Action-Level Fusion Training")
    print("=" * 60)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\nConfiguration:")
    for arg, value in vars(args).items():
        print(f"  {arg}: {value}")
    print("=" * 60)

    train(args)
