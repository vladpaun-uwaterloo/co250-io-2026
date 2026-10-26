import numpy as np

from core.models import (
    SEFLinearProgram,
    OptimalResult,
    UnboundedResult,
    InfeasibleResult,
)

from core.solver import solve


TOL = 1e-8


def test_solver_assignment_5_problem_1():
    # max x4
    #
    # x1 - x3           = 1
    # x2 - x3           = -2
    # x1 - x2 - x3 - x4 = 1
    #
    # x >= 0
    #
    # Unique feasible solution:
    # (3, 0, 2, 0)

    lp = SEFLinearProgram(
        c=np.array([
            0.0,
            0.0,
            0.0,
            1.0,
        ]),
        A=np.array([
            [1.0,  0.0, -1.0,  0.0],
            [0.0,  1.0, -1.0,  0.0],
            [1.0, -1.0, -1.0, -1.0],
        ]),
        b=np.array([
            1.0,
            -2.0,
            1.0,
        ]),
        objective_offset=0.0,
    )

    result = solve(lp)

    assert isinstance(
        result,
        OptimalResult,
    )

    np.testing.assert_allclose(
        result.solution,
        np.array([
            3.0,
            0.0,
            2.0,
            0.0,
        ]),
        atol=TOL,
    )

    assert abs(
        result.objective_value
    ) <= TOL


def test_solver_infeasible():
    # x = 1
    # x = 2
    # x >= 0

    lp = SEFLinearProgram(
        c=np.array([1.0]),
        A=np.array([
            [1.0],
            [1.0],
        ]),
        b=np.array([
            1.0,
            2.0,
        ]),
        objective_offset=0.0,
    )

    result = solve(lp)

    assert isinstance(
        result,
        InfeasibleResult,
    )

    y = result.certificate

    assert np.all(
        lp.A.T @ y >= -TOL
    )

    assert lp.b @ y < -TOL


def test_solver_unbounded():
    # max x1
    #
    # x1 - x2 = 1
    # x >= 0

    lp = SEFLinearProgram(
        c=np.array([
            1.0,
            0.0,
        ]),
        A=np.array([
            [1.0, -1.0],
        ]),
        b=np.array([1.0]),
        objective_offset=0.0,
    )

    result = solve(lp)

    assert isinstance(
        result,
        UnboundedResult,
    )

    np.testing.assert_allclose(
        lp.A @ result.solution,
        lp.b,
        atol=TOL,
    )

    np.testing.assert_allclose(
        lp.A @ result.direction,
        np.zeros(1),
        atol=TOL,
    )

    assert np.all(
        result.direction >= -TOL
    )

    assert (
        lp.c @ result.direction
        > TOL
    )


def test_solver_respects_objective_offset():
    # max 2x + 7
    #
    # x = 3
    #
    # optimum = 13

    lp = SEFLinearProgram(
        c=np.array([2.0]),
        A=np.array([
            [1.0],
        ]),
        b=np.array([3.0]),
        objective_offset=7.0,
    )

    result = solve(lp)

    assert isinstance(
        result,
        OptimalResult,
    )

    np.testing.assert_allclose(
        result.solution,
        np.array([3.0]),
        atol=TOL,
    )

    assert abs(
        result.objective_value - 13.0
    ) <= TOL