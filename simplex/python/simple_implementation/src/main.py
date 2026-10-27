import argparse
import sys

import numpy as np

from core.models import OptimalResult, UnboundedResult, InfeasibleResult
from core.parser import load_lp, parse_lp, prompt_lp
from core.solver import solve


def main():
    parser = argparse.ArgumentParser(
        description="Solve a linear program in standard equality form using the simplex method."
    )

    parser.add_argument(
        "file",
        nargs="?",
        help="LP input file. Use '-' to read from stdin. If omitted, interactive input is used.",
    )

    args = parser.parse_args()

    try:
        if args.file == "-":
            lp = parse_lp(sys.stdin.read())
        elif args.file is not None:
            lp = load_lp(args.file)
        else:
            lp = prompt_lp()

        result = solve(lp)
        print_result(result)

    except (ValueError, OSError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)


def print_result(result):
    print()

    if isinstance(result, OptimalResult):
        print("OPTIMAL")
        print(f"Objective value: {format_number(result.objective_value)}")
        print(f"Solution:        {format_vector(result.solution)}")
        print(f"Basis:           {format_basis(result.basis.indices)}")
        print(f"Certificate:     {format_vector(result.certificate)}")

    elif isinstance(result, UnboundedResult):
        print("UNBOUNDED")
        print(f"Feasible point:  {format_vector(result.solution)}")
        print(f"Direction:       {format_vector(result.direction)}")

    elif isinstance(result, InfeasibleResult):
        print("INFEASIBLE")
        print(f"Certificate:     {format_vector(result.certificate)}")


def format_number(value: float) -> str:
    value = float(value)

    if abs(value) < 1e-12:
        value = 0.0

    return f"{value:.10g}"


def format_vector(values: np.ndarray) -> str:
    return "[" + ", ".join(format_number(value) for value in values) + "]"


def format_basis(indices) -> str:
    return "[" + ", ".join(f"x{i + 1}" for i in indices) + "]"


if __name__ == "__main__":
    main()