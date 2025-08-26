#!/usr/bin/env python3
"""
Training script for DQN agent on Dinosaur Game
Run this script to train your agent on the Chrome Dino game.
"""

import argparse
import os
import sys
import numpy as np
from datetime import datetime

# Add the current directory to Python path for imports
sys.path.append(os.path.dirname(__file__))

# Import your custom modules
try:
    from dino_env import SimplifiedDinosaurGameEnv, DinosaurGameEnv
    from dqn_agent import DQNAgent, train_agent, test_agent
    print("✓ All modules imported successfully")
except ImportError as e:
    print(f"✗ Import error: {e}")
    print("Make sure dino_env.py and dqn_agent.py are in the same directory")
    sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description='Train DQN agent on Dinosaur Game')
    
    # Environment options
    parser.add_argument('--env-type', choices=['simple', 'pixel'], default='simple',
                        help='Environment type: simple (features) or pixel (images)')
    parser.add_argument('--render', action='store_true', default=False,
                        help='Render environment during training')
    
    # Training options
    parser.add_argument('--episodes', type=int, default=1000,
                        help='Number of training episodes')
    parser.add_argument('--max-steps', type=int, default=10000,
                        help='Maximum steps per episode')
    
    # Agent hyperparameters
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--gamma', type=float, default=0.99,
                        help='Discount factor')
    parser.add_argument('--epsilon-start', type=float, default=1.0,
                        help='Initial epsilon for exploration')
    parser.add_argument('--epsilon-end', type=float, default=0.01,
                        help='Final epsilon for exploration')
    parser.add_argument('--epsilon-decay', type=float, default=0.995,
                        help='Epsilon decay rate')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size for training')
    parser.add_argument('--buffer-size', type=int, default=100000,
                        help='Replay buffer size')
    parser.add_argument('--target-update', type=int, default=1000,
                        help='Target network update frequency')
    
    # Model options
    parser.add_argument('--model-dir', type=str, default='models',
                        help='Directory to save models')
    parser.add_argument('--save-every', type=int, default=100,
                        help='Save model every N episodes')
    parser.add_argument('--load-model', type=str, default=None,
                        help='Path to load existing model')
    
    # Testing options
    parser.add_argument('--test-only', action='store_true', default=False,
                        help='Only test the agent (requires --load-model)')
    parser.add_argument('--test-episodes', type=int, default=10,
                        help='Number of test episodes')
    
    args = parser.parse_args()
    
    # Create environment
    print(f"Creating {args.env_type} environment...")
    render_mode = "human" if args.render else None
    
    if args.env_type == 'simple':
        env = SimplifiedDinosaurGameEnv(render_mode=render_mode, max_steps=args.max_steps)
        state_size = env.observation_space.shape[0]
        use_conv = False
        frame_stack = 1
        print(f"✓ Simple environment created - State size: {state_size}")
    else:
        env = DinosaurGameEnv(render_mode=render_mode, max_steps=args.max_steps)
        state_size = env.observation_space.shape
        use_conv = True
        frame_stack = state_size[2]  # Number of stacked frames
        print(f"✓ Pixel environment created - State shape: {state_size}")
    
    action_size = env.action_space.n
    print(f"Action space size: {action_size}")
    
    # Create agent
    print("Creating DQN agent...")
    agent = DQNAgent(
        state_size=state_size[0] * state_size[1] if use_conv else state_size,
        action_size=action_size,
        use_conv=use_conv,
        frame_stack=frame_stack,
        lr=args.lr,
        gamma=args.gamma,
        epsilon_start=args.epsilon_start,
        epsilon_end=args.epsilon_end,
        epsilon_decay=args.epsilon_decay,
        buffer_size=args.buffer_size,
        batch_size=args.batch_size,
        target_update=args.target_update
    )
    
    # Load existing model if specified
    if args.load_model:
        if os.path.exists(args.load_model):
            print(f"Loading model from {args.load_model}...")
            agent.load_model(args.load_model)
            print("✓ Model loaded successfully")
        else:
            print(f"✗ Model file {args.load_model} not found")
            sys.exit(1)
    
    # Test only mode
    if args.test_only:
        if not args.load_model:
            print("✗ --test-only requires --load-model")
            sys.exit(1)
        
        print("Testing agent...")
        test_agent(env, agent, args.test_episodes, render=True)
        return
    
    # Training mode
    print(f"Starting training...")
    print(f"Episodes: {args.episodes}")
    print(f"Max steps per episode: {args.max_steps}")
    print(f"Environment type: {args.env_type}")
    print(f"Using CNN: {use_conv}")
    print(f"Render: {args.render}")
    print("-" * 50)
    
    try:
        # Train the agent
        trained_agent = train_agent(
            env=env,
            agent=agent,
            n_episodes=args.episodes,
            max_steps=args.max_steps,
            save_every=args.save_every,
            model_dir=args.model_dir
        )
        
        # Save final model
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        final_model_path = os.path.join(args.model_dir, f"dino_dqn_final_{timestamp}.pth")
        trained_agent.save_model(final_model_path)
        print(f"Final model saved: {final_model_path}")
        
        # Plot training progress
        print("Plotting training progress...")
        plot_path = os.path.join(args.model_dir, f"training_progress_{timestamp}.png")
        trained_agent.plot_training_progress(save_path=plot_path)
        
        # Test the trained agent
        print("Testing trained agent...")
        test_agent(env, trained_agent, args.test_episodes, render=True)
        
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        
        # Save interrupted training
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        interrupted_model_path = os.path.join(args.model_dir, f"dino_dqn_interrupted_{timestamp}.pth")
        agent.save_model(interrupted_model_path)
        print(f"Model saved: {interrupted_model_path}")
        
    except Exception as e:
        print(f"✗ Training failed with error: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        env.close()
        print("Environment closed")

def quick_test():
    """Quick test function to verify everything works"""
    print("Running quick test...")
    
    # Test simple environment
    try:
        env = SimplifiedDinosaurGameEnv()
        obs, info = env.reset()
        
        # Create simple agent
        agent = DQNAgent(
            state_size=env.observation_space.shape[0],
            action_size=env.action_space.n,
            use_conv=False,
            epsilon_start=1.0
        )
        
        # Run a few steps
        total_reward = 0
        for i in range(10):
            action = agent.act(obs, training=True)
            next_obs, reward, terminated, truncated, info = env.step(action)
            agent.remember(obs, action, reward, next_obs, terminated or truncated)
            obs = next_obs
            total_reward += reward
            
            if terminated or truncated:
                break
        
        print(f"✓ Quick test passed - Total reward: {total_reward:.2f}")
        env.close()
        
    except Exception as e:
        print(f"✗ Quick test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("DQN Training Script for Dinosaur Game")
    print("=" * 50)
    
    # Check if running with arguments
    if len(sys.argv) == 1:
        print("No arguments provided. Running quick test...")
        quick_test()
        print("\nTo start training, use:")
        print("python train_dino.py --episodes 1000 --env-type simple")
        print("python train_dino.py --episodes 1000 --env-type pixel")
        print("\nFor help: python train_dino.py --help")
    else:
        main()