#!/usr/bin/env python
"""Generate the deterministic SellerOps dataset under data/."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sellerops.dataset import DEFAULT_SEED, write_dataset
from sellerops.settings import get_settings


def main() -> None:
    """Parse CLI options, generate dataset files, and print the manifest.

    Args:
        None. Arguments are read from the process command line.

    Returns:
        ``None`` after writing the selected output directory. Command-line
        parsing handles the output path and deterministic seed.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=get_settings().data_dir)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()

    write_dataset(args.output_dir, args.seed)
    manifest = json.loads((args.output_dir / "manifest.json").read_text())
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
