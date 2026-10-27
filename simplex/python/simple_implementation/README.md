# Functional Simplex Solver

A small functional implementation of the simplex method written as a learning exercise for CO250.

The implementation intentionally focuses only on linear programs already written in **standard equality form (SEF)**:

$$
\max c^Tx+c_0
$$

subject to

$$
Ax=b,\qquad x\ge0.
$$

The more complete object-oriented implementation in `../oop/` supports general LP input and performs the transformations to SEF automatically.

## Structure

```text
functional/
├── src/
│   ├── main.py
│   └── core/
│       ├── models.py
│       ├── parser.py
│       ├── simplex.py
│       ├── solver.py
│       └── transform.py
├── examples/
├── test/
├── pytest.ini
└── requirements.txt
```

The main components are:

- `models.py` — LP, basis, and result data structures.
- `parser.py` — file and interactive terminal input.
- `transform.py` — transformations required by the simplex algorithm.
- `simplex.py` — simplex iterations.
- `solver.py` — Phase I, Phase II, and top-level solving logic.
- `main.py` — command-line entry point.

## Algorithm

The solver implements simplex using bases and canonical forms rather than the tableau method.

Phase I constructs an auxiliary LP with artificial variables. It either finds a feasible basis for the original problem or produces an infeasibility certificate.

Phase II starts from the feasible basis and returns either:

- an optimal solution and optimality certificate, or
- a feasible point and recession direction proving unboundedness.

## Input file format

A problem file contains the dimensions, objective vector, constraint matrix, right-hand side, and optional objective offset.

For example:

```text
m 2
n 4

c 3 2 0 0

A
1 1 1 0
2 1 0 1

b 4 5

offset 0
```

This represents

$$
\max 3x_1+2x_2
$$

subject to

$$
x_1+x_2+x_3=4,
$$

$$
2x_1+x_2+x_4=5,
$$

and

$$
x\ge0.
$$

Blank lines are ignored and `#` introduces a comment.

## Running interactively

Run:

```bash
python src/main.py
```

The program will prompt for the dimensions, objective, constraint rows, right-hand side, and objective offset.

## Running from a file

Run:

```bash
python src/main.py examples/example.lp
```

Input can also be piped through stdin:

```bash
cat examples/example.lp | python src/main.py -
```

## Example output

```text
OPTIMAL
Objective value: 9
Solution:        [1, 3, 0, 0]
Basis:           [x2, x1]
Certificate:     [1, 1]
```

## Tests

Install dependencies:

```bash
pip install -r requirements.txt
```

Then run:

```bash
pytest
```

## Scope

This version deliberately does not convert general LPs into SEF.

Supporting minimization, inequalities, arbitrary bounds, and free variables requires substantial additional transformation and certificate bookkeeping. That functionality is implemented in the object-oriented version instead.

The purpose of this version is to keep the implementation of the simplex algorithm itself small and easy to follow.