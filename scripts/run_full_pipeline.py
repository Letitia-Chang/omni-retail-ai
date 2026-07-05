import subprocess
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]

PIPELINE_STEPS = [
    "scripts/run_customer_segmentation.py",
    "scripts/run_product_enrichment.py",
    "scripts/run_purchase_intent.py",
    "scripts/run_campaign_ranking.py",
    "scripts/run_campaign_generation.py",
]


def run_step(script_path):
    print(f"\nRunning {script_path}...")

    result = subprocess.run(
        [sys.executable, script_path],
        cwd=ROOT_DIR,
    )

    if result.returncode != 0:
        raise RuntimeError(f"Pipeline failed at: {script_path}")

    print(f"Finished {script_path}")


def main():
    print("Starting OmniRetail AI pipeline...")

    for step in PIPELINE_STEPS:
        run_step(step)

    print("\nFull pipeline completed successfully.")


if __name__ == "__main__":
    main()