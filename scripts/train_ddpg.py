#!/usr/bin/env python3
"""Standalone DDPG training script."""
import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml
from bemo.algorithms.ddpg_planner import DDPGPlanner


def main():
    parser = argparse.ArgumentParser(description="Train DDPG UAV agent")
    parser.add_argument("--config", default="config/default_config.yaml")
    parser.add_argument("--timesteps", type=int, default=500_000)
    parser.add_argument("--save-path", default="models/ddpg_uav")
    parser.add_argument("--log-dir", default="logs/ddpg")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    planner = DDPGPlanner.from_config(config)
    print(f"Training DDPG for {args.timesteps:,} timesteps...")
    planner.train(
        total_timesteps=args.timesteps,
        log_dir=args.log_dir,
        save_path=args.save_path,
    )


if __name__ == "__main__":
    main()
