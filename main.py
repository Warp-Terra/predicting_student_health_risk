#!/usr/bin/env python3
from argparse import Namespace
from datetime import datetime
import os

from run_experiment import run


def main():
    run_name = f"balacc_seed2026_{datetime.now():%Y%m%d_%H%M%S}"
    run(
        Namespace(
            name=run_name,
            iterations=1500,
            weight_power=1.0,
            learning_rate=0.05,
            num_leaves=63,
            min_data_in_leaf=100,
            model_seed=2026,
            early_stop_metric="balanced_accuracy",
            reuse_model_prefix=None,
            threads=max(1, min(6, os.cpu_count() or 1)),
        )
    )


if __name__ == "__main__":
    main()
