#!/usr/bin/env python3
"""Standalone PPO training script."""
import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml
from bemo.algorithms.ppo_planner import PPOPlanner


def main():
    parser = argparse.ArgumentParser(description="Train PPO UAV agent")
    parser.add_argument("--config", default="config/default_config.yaml")
    parser.add_argument("--timesteps", type=int, default=1_000_000)
    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument("--save-path", default="models/ppo_uav")
    parser.add_argument("--log-dir", default="logs/ppo")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    planner = PPOPlanner.from_config(config)
    print(f"Training PPO for {args.timesteps:,} timesteps with {args.n_envs} envs...")
    planner.train(
        total_timesteps=args.timesteps,
        n_envs=args.n_envs,
        log_dir=args.log_dir,
        save_path=args.save_path,
    )


if __name__ == "__main__":
    main()
