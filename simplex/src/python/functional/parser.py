import argparse
import json
import numpy as np

from models import SEFLinearProgram


def parse_vector(value: str) -> np.ndarray:
    return np.array(
        [float(x.strip()) for x in value.split(",")],
        dtype=float,
    )


def parse_matrix(value: str) -> np.ndarray:
    return np.array(
        [
            [float(x.strip()) for x in row.split(",")]
            for row in value.split(";")
        ],
        dtype=float,
    )


def parse_file(path: str) -> SEFLinearProgram:
    with open(path, "r") as f:
        data = json.load(f)

    c = np.array(data["c"], dtype=float)
    A = np.array(data["A"], dtype=float)
    b = np.array(data["b"], dtype=float)

    objective_offset = float(data.get("objective_offset", 0.0))

    _validate_sef(c, A, b)

    return SEFLinearProgram(
        c=c,
        A=A,
        b=b,
        objective_offset=objective_offset,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Solve a linear program in standard equality form."
    )

    parser.add_argument(
        "--file",
        type=str,
        help="Path to a JSON file containing the LP.",
    )

    parser.add_argument(
        "--c",
        type=str,
        help='Objective vector, e.g. "3,2,1"',
    )

    parser.add_argument(
        "--A",
        type=str,
        help='Constraint matrix, e.g. "1,2;3,4"',
    )

    parser.add_argument(
        "--b",
        type=str,
        help='Right-hand side vector, e.g. "5,6"',
    )

    parser.add_argument(
        "--offset",
        type=float,
        default=0.0,
        help="Objective constant offset.",
    )

    return parser.parse_args()


def build_lp(args: argparse.Namespace) -> SEFLinearProgram:
    if args.file is not None:
        if any(value is not None for value in (args.c, args.A, args.b)):
            raise ValueError(
                "Use either --file or --c/--A/--b, not both."
            )

        return parse_file(args.file)

    if args.c is None or args.A is None or args.b is None:
        raise ValueError(
            "Provide either --file or all of --c, --A and --b."
        )

    c = parse_vector(args.c)
    A = parse_matrix(args.A)
    b = parse_vector(args.b)

    _validate_sef(c, A, b)

    return SEFLinearProgram(
        c=c,
        A=A,
        b=b,
        objective_offset=args.offset,
    )


def _validate_sef(
    c: np.ndarray,
    A: np.ndarray,
    b: np.ndarray,
) -> None:
    if c.ndim != 1:
        raise ValueError("c must be a vector.")

    if b.ndim != 1:
        raise ValueError("b must be a vector.")

    if A.ndim != 2:
        raise ValueError("A must be a matrix.")

    m, n = A.shape

    if len(c) != n:
        raise ValueError(
            f"A has {n} columns but c has length {len(c)}."
        )

    if len(b) != m:
        raise ValueError(
            f"A has {m} rows but b has length {len(b)}."
        )