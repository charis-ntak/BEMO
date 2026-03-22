# BEMO — Multi-UAV Path Planning & Obstacle Avoidance

A Python testbed for autonomous UAV navigation in 3D lab-scale environments, comparing four planning algorithms using multi-objective evaluation.

## Features

- **Random scenario generation**: Box and cylinder obstacles, multiple UAVs, configurable 3D lab space
- **4 planning algorithms**: RRT*, APF-PSO, PPO (deep RL), DDPG (deep RL)
- **4 objectives**: path length, energy efficiency, trajectory smoothness, obstacle clearance
- **Multi-objective aggregation**: Pareto optimality (NSGA-II sorting) + Mamdani Fuzzy Inference System
- **Evaluation**: Benchmarking suite with comparative metrics, statistical analysis
- **Visualization**: 3D path plots, Pareto front scatter, radar/bar dashboards

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Plan (single run, compare algorithms)
```bash
python main.py plan --algorithm all --n-obstacles 5 --n-uavs 1 --aggregation both
python main.py plan --algorithm rrt_star --n-obstacles 8 --visualize
```

### Train RL agents
```bash
python main.py train --algorithm ppo --timesteps 1000000 --n-envs 8
python main.py train --algorithm ddpg --timesteps 500000
```

### Benchmark (compare all algorithms)
```bash
python main.py benchmark --n-scenarios 20 --n-runs 5 --output-dir results/
```

## Architecture

```
bemo/
├── environment/      # World, obstacles (SDF-based), UAV dynamics, Gym wrapper
├── objectives/       # Path length, energy, smoothness, safety
├── aggregation/      # Pareto front (NSGA-II), Mamdani FIS (scikit-fuzzy)
├── algorithms/       # RRT*, APF-PSO, PPO, DDPG planners
├── evaluation/       # Metrics, benchmark suite, CSV/JSON reporter
└── visualization/    # 3D scene, Pareto plot, metrics dashboard
```

### Key Design: SDF-based Geometry
All collision and clearance queries use Signed Distance Functions (SDF) through a unified `ObstacleMap.clearance(point)` interface, enabling consistent geometry across objectives, planners, and RL reward shaping.

### Aggregation
- **Pareto**: Non-dominated sorting across [path_length, energy, smoothness, −safety]. Hypervolume as scalar quality indicator.
- **Mamdani FIS**: 4 antecedents → 1 fitness score. Safety antecedent dominates via rule `safety=poor → fitness=very_poor`.

## Configuration

See `config/default_config.yaml` for all tunable parameters.

## Tests

```bash
python -m pytest tests/ -v
```
