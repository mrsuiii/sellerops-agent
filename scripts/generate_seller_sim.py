"""Write the seller simulator dataset and eval tasks to data/seller_sim/.

    python scripts/generate_seller_sim.py [--seed N]

Deterministic: the same catalog and seed give byte-identical files (the
manifests carry SHA-256 checksums; tests compare against them).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sellerops.evaluation.seller_tasks import write_tasks
from sellerops.settings import get_settings
from sellerops.simulator.generator import DEFAULT_SEED, write_dataset
from sellerops.simulator.store import load_sim_data


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()

    settings = get_settings()
    out_dir = settings.data_dir / "seller_sim"
    catalog = json.loads(settings.catalog_path.read_text(encoding="utf-8"))
    manifest = write_dataset(catalog, out_dir, seed=args.seed)
    print(f"dataset {manifest['data_version']} seed={manifest['seed']} {manifest['counts']} -> {out_dir}")
    tasks_manifest = write_tasks(load_sim_data(out_dir), out_dir)
    print(f"tasks {tasks_manifest['counts']}")


if __name__ == "__main__":
    main()
