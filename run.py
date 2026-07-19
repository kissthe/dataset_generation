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
    parser.add_argument(
        "--prepare-blueprints", action="store_true",
        help="Generate reviewable Blueprint candidates, then stop.",
    )
    parser.add_argument(
        "--prepare-plans", action="store_true",
        help="Generate Plan candidates for --select-blueprint, then stop.",
    )
    parser.add_argument("--blueprint-count", type=int, default=3)
    parser.add_argument("--plan-count", type=int, default=3)
    parser.add_argument(
        "--prepare-candidates", action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--candidate-count", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--select-blueprint", type=str)
    parser.add_argument("--select-plan", type=str)
    args = parser.parse_args()
    config = AppConfig.load(args.config.resolve())
    pipeline = GenerationPipeline(config)
    if args.prepare_blueprints or args.prepare_candidates:
        if args.prepare_plans:
            parser.error("--prepare-blueprints and --prepare-plans cannot be used together")
        blueprint_count = (
            args.candidate_count
            if args.prepare_candidates and args.candidate_count is not None
            else args.blueprint_count
        )
        artifact = pipeline.prepare_blueprint_candidates(
            args.case.resolve(), args.output.resolve(), blueprint_count
        )
        qa = None
    elif args.prepare_plans:
        if not args.select_blueprint:
            parser.error("--prepare-plans requires --select-blueprint")
        if args.select_plan:
            parser.error("--select-plan is only used when continuing the pipeline")
        artifact = pipeline.prepare_plan_candidates(
            args.case.resolve(), args.output.resolve(),
            args.select_blueprint, args.plan_count,
        )
        qa = None
    else:
        if bool(args.select_blueprint) != bool(args.select_plan):
            parser.error("--select-blueprint and --select-plan must be used together")
        if args.select_blueprint and args.select_plan:
            pipeline.select_candidates(
                args.output.resolve(), args.select_blueprint, args.select_plan
            )
        artifact, qa = pipeline.run(args.case.resolve(), args.output.resolve())
    if artifact.name == "benchmark.json":
        print(f"benchmark: {artifact}")
    if qa:
        print(f"qa: {qa}")


if __name__ == "__main__":
    main()
