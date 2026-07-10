import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import ID_COL, ROOT, SUBMISSION_DIR, TEST_PATH
from run_experiment import save_submission

def main():
    parser = argparse.ArgumentParser(description="Rebuild a submission from saved probabilities")
    parser.add_argument("artifact", help="Artifact directory containing test.npy")
    parser.add_argument("--name", default="resubmission")
    args = parser.parse_args()

    artifact = Path(args.artifact)
    if not artifact.is_absolute():
        artifact = ROOT / artifact
    probabilities = np.load(artifact / "test.npy")
    test_ids = pd.read_csv(TEST_PATH, usecols=[ID_COL])[ID_COL]
    path = SUBMISSION_DIR / f"submission_{args.name}.csv"
    save_submission(probabilities, test_ids, path)
    print(f"Submission saved to {path}")

if __name__ == "__main__":
    main()
