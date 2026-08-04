#!/usr/bin/env python3
"""Summarize harmonic C_V files from the MOF-5 + 100 CH4 temperature sweep."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


PROJECT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_TEMPERATURES = (100, 200, 300, 400, 500)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--temperatures",
        default=",".join(str(value) for value in DEFAULT_TEMPERATURES),
        help="Comma-separated MD temperatures (default: 100,200,300,400,500)",
    )
    parser.add_argument(
        "--frame",
        type=int,
        default=4000,
        help="Analyzed trajectory frame used in result filenames (default: 4000)",
    )
    parser.add_argument(
        "--analysis-temperature",
        type=float,
        help="Report every result at this temperature instead of its MD temperature",
    )
    return parser.parse_args()


def requested_temperatures(specification: str) -> list[int]:
    values = [int(item.strip()) for item in specification.split(",") if item.strip()]
    if not values:
        raise ValueError("--temperatures must not be empty")
    unsupported = sorted(set(values) - set(DEFAULT_TEMPERATURES))
    if unsupported:
        raise ValueError(f"unsupported MD temperatures: {unsupported}")
    return values


def scalar_text(value: np.ndarray) -> str:
    return str(np.asarray(value).reshape(-1)[0])


def main() -> None:
    args = parse_args()
    if args.frame < 0:
        raise ValueError("--frame must be non-negative")

    print("MD_temperature_K,frame,analysis_temperature_K,Cv_J_per_gK,result")
    for md_temperature in requested_temperatures(args.temperatures):
        result = (
            PROJECT_DIR
            / f"output/mof5-100ch4-{md_temperature}K-test"
            / f"heat-capacity-frame-{args.frame}.npz"
        )
        if not result.is_file():
            raise FileNotFoundError(f"heat-capacity result not found: {result}")

        with np.load(result, allow_pickle=False) as data:
            temperatures = np.asarray(data["temperatures_K"], dtype=float)
            cv = np.asarray(data["cv_J_per_gK"], dtype=float)
            frames = np.asarray(data["frame_indices"], dtype=int)
            run_name = scalar_text(data["run_name"])

        if cv.shape != (len(frames), len(temperatures)):
            raise ValueError(
                f"unexpected C_V shape {cv.shape} in {result}; "
                f"expected ({len(frames)}, {len(temperatures)})"
            )
        if len(frames) != 1 or int(frames[0]) != args.frame:
            raise ValueError(f"unexpected frame indices {frames.tolist()} in {result}")

        target = float(args.analysis_temperature or md_temperature)
        matches = np.flatnonzero(np.isclose(temperatures, target, rtol=0.0, atol=1e-8))
        if len(matches) != 1:
            raise ValueError(f"temperature {target:g} K is absent from {result}")

        value = cv[0, int(matches[0])]
        relative_result = result.relative_to(PROJECT_DIR)
        print(
            f"{md_temperature},{args.frame},{target:g},{value:.8g},"
            f"{relative_result} ({run_name})"
        )


if __name__ == "__main__":
    main()

