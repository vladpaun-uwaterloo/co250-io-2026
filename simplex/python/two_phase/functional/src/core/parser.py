import numpy as np

from .models import SEFLinearProgram


def parse_lp(text: str) -> SEFLinearProgram:
    lines = []

    for line in text.splitlines():
        line = line.split("#", 1)[0].strip()

        if line:
            lines.append(line)

    m = None
    n = None
    c = None
    A = None
    b = None
    objective_offset = 0.0

    i = 0

    while i < len(lines):
        parts = lines[i].split()
        keyword = parts[0].lower()

        if keyword == "m":
            if len(parts) != 2:
                raise ValueError("Expected: m <number of constraints>")

            m = int(parts[1])
            i += 1

        elif keyword == "n":
            if len(parts) != 2:
                raise ValueError("Expected: n <number of variables>")

            n = int(parts[1])
            i += 1

        elif keyword == "c":
            c = np.array([float(value) for value in parts[1:]], dtype=float)
            i += 1

        elif keyword == "a":
            if m is None or n is None:
                raise ValueError("m and n must be specified before A.")

            rows = []

            for row in range(m):
                i += 1

                if i >= len(lines):
                    raise ValueError(f"Expected {m} rows after A.")

                values = [float(value) for value in lines[i].split()]

                if len(values) != n:
                    raise ValueError(f"Row {row + 1} of A has {len(values)} entries, expected {n}.")

                rows.append(values)

            A = np.array(rows, dtype=float)
            i += 1

        elif keyword == "b":
            b = np.array([float(value) for value in parts[1:]], dtype=float)
            i += 1

        elif keyword == "offset":
            if len(parts) != 2:
                raise ValueError("Expected: offset <number>")

            objective_offset = float(parts[1])
            i += 1

        else:
            raise ValueError(f"Unknown input keyword: {parts[0]}")

    if m is None:
        raise ValueError("Missing m.")

    if n is None:
        raise ValueError("Missing n.")

    if c is None:
        raise ValueError("Missing c.")

    if A is None:
        raise ValueError("Missing A.")

    if b is None:
        raise ValueError("Missing b.")

    if len(c) != n:
        raise ValueError(f"c has {len(c)} entries, expected {n}.")

    if A.shape != (m, n):
        raise ValueError(f"A has shape {A.shape}, expected ({m}, {n}).")

    if len(b) != m:
        raise ValueError(f"b has {len(b)} entries, expected {m}.")

    return SEFLinearProgram(
        c=c,
        A=A,
        b=b,
        objective_offset=objective_offset,
    )


def load_lp(filename: str) -> SEFLinearProgram:
    with open(filename, "r") as file:
        return parse_lp(file.read())


def prompt_lp() -> SEFLinearProgram:
    print("Enter an LP in standard equality form:")
    print()
    print("    maximize c^T x + offset")
    print("    subject to Ax = b")
    print("               x >= 0")
    print()

    m = int(input("Number of constraints (m): "))
    n = int(input("Number of variables (n): "))

    print()
    c = _read_vector("c", n)

    print()
    print(f"Enter the {m} rows of A:")

    A = np.empty((m, n), dtype=float)

    for i in range(m):
        A[i, :] = _read_vector(f"A[{i + 1}]", n)

    print()
    b = _read_vector("b", m)

    print()
    offset_text = input("Objective offset [0]: ").strip()
    objective_offset = float(offset_text) if offset_text else 0.0

    return SEFLinearProgram(
        c=c,
        A=A,
        b=b,
        objective_offset=objective_offset,
    )


def _read_vector(name: str, length: int) -> np.ndarray:
    while True:
        text = input(f"{name} ({length} numbers): ")

        try:
            values = np.array([float(value) for value in text.split()], dtype=float)

        except ValueError:
            print("Invalid number. Try again.")
            continue

        if len(values) != length:
            print(f"Expected {length} numbers, got {len(values)}.")
            continue

        return values