# Simplex Solver — Python Implementations

This directory contains two Python implementations of the simplex method written while studying linear optimization in CO250.

```text
python/
├── simple_implementation/
└── oop/
```

They implement the same underlying simplex algorithm but intentionally have different scopes and program structures.

## Functional implementation

The `simple_implementation/` version is the smaller implementation.

It operates only on linear programs already written in standard equality form:

$$
\max c^Tx + c_0
$$

subject to

$$
Ax=b,\qquad x\ge0.
$$

It is organized primarily around functions implementing:

- canonicalization with respect to a basis,
- Phase I,
- auxiliary LP construction,
- removal of artificial variables,
- Phase II,
- simplex iterations,
- optimality, infeasibility, and unboundedness certificates.

The program has no graphical frontend. Problems can either be entered interactively in the terminal or loaded from a small text file.

Its purpose is to keep the implementation close to the mathematical simplex algorithm and avoid unrelated transformation bookkeeping.

## Object-oriented implementation

The `oop/` version is the more complete solver.

It supports both standard equality form and general LP input, including:

- maximization and minimization,
- `<=`, `=`, and `>=` constraints,
- finite lower and upper bounds,
- free variables,
- objective offsets.

General LPs are converted internally to standard equality form.

The conversion is reversible: the solver records how variables, constraints, bounds, and the objective are transformed so that solutions, directions, objective values, and certificates can be mapped back to the original LP.

The OOP version also includes a PySide6 graphical frontend.

## Why two versions?

The goal is not to maintain two feature-identical solvers.

Instead:

- `functional/` is a compact implementation of the simplex algorithm itself.
- `oop/` explores a larger solver architecture with objects and reversible transformations.

This makes the two versions useful for comparing both the mathematics and the software-design approaches.