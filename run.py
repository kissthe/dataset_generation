from __future__ import annotations

import argparse
from pathlib import Path

from src.config import AppConfig
from src.pipeline import GenerationPipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate long-term dialogue benchmark data")
    parser.add_argument("--config", type=Path, default=Path("config.json"))
    parser.add_argument("--case", type=Path, default=Path("cases/case_a.json"))
    parser.add_argument("--output", type=Path, default=Path("outputs/case_a"))
    args = parser.parse_args()
    config = AppConfig.load(args.config.resolve())
    artifact, qa = GenerationPipeline(config).run(args.case.resolve(), args.output.resolve())
    if artifact.name == "benchmark.json":
        print(f"benchmark: {artifact}")
    if qa:
        print(f"qa: {qa}")


if __name__ == "__main__":
    main()
