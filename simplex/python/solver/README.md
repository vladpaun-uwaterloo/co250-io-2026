# Object-Oriented Simplex Solver

An object-oriented simplex solver written as an extension of the smaller functional implementation in `../functional/`.

The project is a learning exercise in both linear optimization and object-oriented software design.

## Supported problems

The solver accepts both standard equality form problems and general linear programs.

General LPs may contain:

- maximization or minimization objectives,
- `<=`, `=`, and `>=` constraints,
- finite lower bounds,
- finite upper bounds,
- variables bounded on both sides,
- free variables,
- objective offsets.

The simplex algorithm itself still operates on standard equality form:

$$
\max c^Tx+c_0
$$

subject to

$$
Ax=b,\qquad x\ge0.
$$

General problems are converted internally before simplex is applied.

## Main objects

The implementation is organized around objects representing the mathematical concepts involved in the algorithm.

Important classes include:

- `LinearProgram` — a general LP.
- `SEFLinearProgram` — an LP in standard equality form.
- `Basis` — the current simplex basis.
- `SimplexSolver` — Phase I, Phase II, and simplex.
- `OptimalResult`, `UnboundedResult`, and `InfeasibleResult` — solver outcomes.
- transformation objects — bookkeeping required to map SEF results back to the original LP.

Operations naturally associated with an LP are implemented as methods, for example:

```python
lp.to_sef()
lp.normalize()
lp.canonicalize(basis)
lp.build_auxiliary()
```

The simplex algorithm itself remains the responsibility of `SimplexSolver`.

## Reversible general-LP transformations

General variables are converted into nonnegative SEF variables using an affine transformation

$$
x=q+Tz.
$$

Examples include:

### Lower-bounded variable

For

$$
x\ge L,
$$

use

$$
x=L+z,\qquad z\ge0.
$$

### Upper-bounded variable

For

$$
x\le U,
$$

use

$$
x=U-z,\qquad z\ge0.
$$

### Free variable

For an unrestricted variable,

$$
x=z^+-z^-,
$$

with

$$
z^+,z^-\ge0.
$$

The transformation is recorded so that SEF solutions and unbounded directions can later be mapped back to the original variables.

Objective transformations and bound constraints are tracked as well.

## Row transformation tracking

The SEF solver also records row operations performed during normalization and canonicalization.

If the current constraint system is related to the original one by

$$
A_{\text{current}}=RA_{\text{original}},
$$

and

$$
b_{\text{current}}=Rb_{\text{original}},
$$

then a certificate \(y\) for the current system can be mapped to a certificate for the original system using

$$
R^Ty.
$$

This remains valid even when Phase I performs canonical row operations or removes redundant constraints.

## Solver phases

The solver performs:

1. conversion to SEF when required,
2. right-hand-side normalization,
3. construction of the Phase I auxiliary LP,
4. Phase I simplex,
5. removal of artificial variables and redundant rows,
6. Phase II simplex,
7. mapping of the result back to the original general LP.

The final result may contain:

- an optimal solution and certificate,
- an unbounded feasible point and direction,
- an infeasibility certificate.

## Frontend

Unlike the functional version, this implementation includes a PySide6 graphical frontend.

The GUI supports both:

- standard equality form input,
- general LP input.

Both are passed through the same `SimplexSolver`.

## Running

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the GUI:

```bash
python src/main.py
```

Run the tests:

```bash
pytest
```

## Purpose

The functional implementation emphasizes the simplex algorithm itself.

This version deliberately goes further and explores how LP representations, mutable bases, transformations, certificates, and solver state can be organized using objects while preserving the mathematical relationship between the transformed problem and the original one.