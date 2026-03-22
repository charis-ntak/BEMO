"""Results export to JSON, CSV, and summary statistics."""
from __future__ import annotations

import json
import os
from typing import Dict, List, Optional

import numpy as np


class Reporter:
    """Saves benchmark results to disk."""

    def __init__(self, output_dir: str = "results/"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def save_metrics(self, metrics, filename: str = "metrics.json") -> str:
        path = os.path.join(self.output_dir, filename)
        data = [m.to_dict() for m in metrics]
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=self._json_default)
        print(f"Metrics saved to {path}")
        return path

    def save_summary(self, summary: Dict, filename: str = "summary.json") -> str:
        path = os.path.join(self.output_dir, filename)
        with open(path, "w") as f:
            json.dump(summary, f, indent=2, default=self._json_default)
        print(f"Summary saved to {path}")
        return path

    def save_csv(self, metrics, filename: str = "metrics.csv") -> str:
        path = os.path.join(self.output_dir, filename)
        try:
            import pandas as pd
            df = pd.DataFrame([m.to_dict() for m in metrics])
            df.to_csv(path, index=False)
            print(f"CSV saved to {path}")
        except ImportError:
            # Fallback: manual CSV
            if metrics:
                keys = list(metrics[0].to_dict().keys())
                with open(path, "w") as f:
                    f.write(",".join(keys) + "\n")
                    for m in metrics:
                        row = m.to_dict()
                        f.write(",".join(str(row[k]) for k in keys) + "\n")
        return path

    def print_summary(self, summary: Dict) -> None:
        print("\n" + "=" * 60)
        print("BENCHMARK SUMMARY")
        print("=" * 60)
        for alg, stats in summary.items():
            print(f"\n{alg}:")
            for key, val in stats.items():
                print(f"  {key:<30} {val:.4f}")

    @staticmethod
    def _json_default(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, bool):
            return bool(obj)
        return str(obj)
