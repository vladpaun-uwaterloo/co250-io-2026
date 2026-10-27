import numpy as np

from core.models import (
    SEFLinearProgram,
    Basis,
    OptimalResult,
    UnboundedResult,
    InfeasibleResult,
)

from core.simplex import (
    simplex,
    phase_one,
    phase_two,
)


TOL = 1e-8


def test_simplex_finds_optimum():
    # max 3x1 + 2x2
    #
    # x1 + x2 + s1      = 4
    # 2x1 + x2      + s2 = 5
    #
    # Initial slack basis: {s1, s2}

    lp = SEFLinearProgram(
        c=np.array([3.0, 2.0, 0.0, 0.0]),
        A=np.array([
            [1.0, 1.0, 1.0, 0.0],
            [2.0, 1.0, 0.0, 1.0],
        ]),
        b=np.array([4.0, 5.0]),
        objective_offset=0.0,
    )

    basis = Basis((2, 3))

    result = simplex(lp, basis)

    assert isinstance(result, OptimalResult)

    np.testing.assert_allclose(
        result.solution,
        np.array([1.0, 3.0, 0.0, 0.0]),
        atol=TOL,
    )

    assert abs(result.objective_value - 9.0) <= TOL

    # Primal feasibility
    np.testing.assert_allclose(
        lp.A @ result.solution,
        lp.b,
        atol=TOL,
    )

    assert np.all(result.solution >= -TOL)

    # Optimality certificate:
    # c - A^T y <= 0
    assert np.all(
        lp.c - lp.A.T @ result.certificate
        <= TOL
    )

    # c^T x = b^T y
    assert abs(
        lp.c @ result.solution
        - lp.b @ result.certificate
    ) <= TOL


def test_simplex_detects_unbounded_problem():
    # max x1
    #
    # x1 - x2 = 1
    # x >= 0
    #
    # Starting basis {x1}, BFS = (1, 0).
    #
    # Direction (1,1) preserves the equality and improves
    # the objective forever.

    lp = SEFLinearProgram(
        c=np.array([1.0, 0.0]),
        A=np.array([
            [1.0, -1.0],
        ]),
        b=np.array([1.0]),
        objective_offset=0.0,
    )

    basis = Basis((0,))

    result = simplex(lp, basis)

    assert isinstance(result, UnboundedResult)

    # Starting solution is feasible.
    np.testing.assert_allclose(
        lp.A @ result.solution,
        lp.b,
        atol=TOL,
    )

    assert np.all(result.solution >= -TOL)

    # Direction must be a recession direction.
    np.testing.assert_allclose(
        lp.A @ result.direction,
        np.zeros(1),
        atol=TOL,
    )

    assert np.all(result.direction >= -TOL)

    # Objective strictly improves in this direction.
    assert lp.c @ result.direction > TOL


def test_phase_one_detects_infeasibility_and_reverses_normalization():
    # x1 = -1, x1 >= 0
    #
    # Clearly impossible.
    #
    # This intentionally has negative RHS so we also test that
    # the Phase I certificate is mapped back through normalization.

    lp = SEFLinearProgram(
        c=np.array([0.0]),
        A=np.array([
            [1.0],
        ]),
        b=np.array([-1.0]),
        objective_offset=0.0,
    )

    result = phase_one(lp)

    assert isinstance(result, InfeasibleResult)

    y = result.certificate

    # Infeasibility certificate for ORIGINAL system:
    #
    # A^T y >= 0
    # b^T y < 0

    assert np.all(
        lp.A.T @ y >= -TOL
    )

    assert lp.b @ y < -TOL


def test_phase_one_removes_redundant_constraint():
    # Second equation is exactly twice the first:
    #
    # x1 + x2 = 1
    # 2x1 + 2x2 = 2

    lp = SEFLinearProgram(
        c=np.array([1.0, 0.0]),
        A=np.array([
            [1.0, 1.0],
            [2.0, 2.0],
        ]),
        b=np.array([1.0, 2.0]),
        objective_offset=0.0,
    )

    result = phase_one(lp)

    assert not isinstance(result, InfeasibleResult)

    clean_lp, basis = result

    # Rank is only 1, so Phase I should remove one redundant row.
    assert clean_lp.A.shape == (1, 2)
    assert clean_lp.b.shape == (1,)
    assert len(basis.indices) == 1

    # The resulting basis should contain only original variables.
    assert all(
        j < 2
        for j in basis.indices
    )

    # And Phase II should still solve the same problem.
    result = phase_two(
        clean_lp,
        basis,
    )

    assert isinstance(result, OptimalResult)

    assert abs(
        result.objective_value - 1.0
    ) <= TOL

    np.testing.assert_allclose(
        result.solution,
        np.array([1.0, 0.0]),
        atol=TOL,
    )