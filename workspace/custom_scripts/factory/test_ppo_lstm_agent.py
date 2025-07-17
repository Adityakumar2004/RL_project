import argparse
import os
import torch
import sys

# Ensure the script can import from the current directory
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ppo_lstm_1 import make_env, Agent, TestingAgent


def main():
    parser = argparse.ArgumentParser(description="Test PPO LSTM Agent")
    parser.add_argument('--checkpoint', type=str, default=os.path.join("custom_scripts", "logs", "ppo_factory", "checkpoints", "cp_lstm_1.pt"), help='Path to the checkpoint file')
    parser.add_argument('--num_episodes', type=int, default=5, help='Number of test episodes')
    parser.add_argument('--video', action='store_true', help='Enable video recording')
    parser.add_argument('--cuda', action='store_true', help='Use CUDA if available')
    args = parser.parse_args()

    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")

    # Set up video folder if recording
    video_folder = None
    if args.video:
        video_folder = os.path.join("custom_scripts", "logs", "ppo_factory", "videos_lstm_1_test")
        os.makedirs(video_folder, exist_ok=True)

    # Create environment
    env = make_env(video_folder=video_folder, output_type="torch")
    env.eval()  # Set to evaluation mode

    # Create agent and load checkpoint
    agent = Agent(env)
    agent.to(device)
    if not os.path.exists(args.checkpoint):
        print(f"Checkpoint not found at {args.checkpoint}")
        return
    checkpoint = torch.load(args.checkpoint, map_location=device)
    agent.load_state_dict(checkpoint["agent"])
    print(f"Loaded checkpoint from {args.checkpoint}")

    # Run evaluation
    avg_reward = TestingAgent(env, agent, num_episodes=args.num_episodes, recording_enabled=args.video)
    print(f"Average reward over {args.num_episodes} episodes: {avg_reward:.2f}")

    env.close()

if __name__ == "__main__":
    main() 