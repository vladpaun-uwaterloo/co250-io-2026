import numpy as np

from core.linear_program import (
    SEFLinearProgram,
)

from core.result import (
    OptimalResult,
    UnboundedResult,
    InfeasibleResult,
)

from core.solver import SimplexSolver


TOL = 1e-8


def test_solver_optimal():
    # max 3x1 + 2x2
    #
    # x1 + x2 + s1       = 4
    # 2x1 + x2      + s2 = 5

    lp = SEFLinearProgram(
        c=np.array([
            3.0,
            2.0,
            0.0,
            0.0,
        ]),
        A=np.array([
            [1.0, 1.0, 1.0, 0.0],
            [2.0, 1.0, 0.0, 1.0],
        ]),
        b=np.array([
            4.0,
            5.0,
        ]),
    )

    original_A = lp.A.copy()
    original_b = lp.b.copy()
    original_c = lp.c.copy()

    result = SimplexSolver().solve(lp)

    assert isinstance(
        result,
        OptimalResult,
    )

    np.testing.assert_allclose(
        result.solution,
        np.array([
            1.0,
            3.0,
            0.0,
            0.0,
        ]),
        atol=TOL,
    )

    assert abs(
        result.objective_value - 9.0
    ) <= TOL

    # Feasible in original LP.
    np.testing.assert_allclose(
        original_A @ result.solution,
        original_b,
        atol=TOL,
    )

    assert np.all(
        result.solution >= -TOL
    )

    # Optimality certificate.
    assert np.all(
        original_c
        - original_A.T @ result.certificate
        <= TOL
    )

    assert abs(
        original_c @ result.solution
        - original_b @ result.certificate
    ) <= TOL


def test_solver_unbounded():
    # max x1
    #
    # x1 - x2 = 1

    lp = SEFLinearProgram(
        c=np.array([
            1.0,
            0.0,
        ]),
        A=np.array([
            [1.0, -1.0],
        ]),
        b=np.array([1.0]),
    )

    A = lp.A.copy()
    b = lp.b.copy()
    c = lp.c.copy()

    result = SimplexSolver().solve(lp)

    assert isinstance(
        result,
        UnboundedResult,
    )

    np.testing.assert_allclose(
        A @ result.solution,
        b,
        atol=TOL,
    )

    np.testing.assert_allclose(
        A @ result.direction,
        np.zeros(
            len(b)
        ),
        atol=TOL,
    )

    assert np.all(
        result.direction >= -TOL
    )

    assert (
        c @ result.direction
        > TOL
    )


def test_solver_infeasible():
    # x = 1
    # x = 2

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
    )

    A = lp.A.copy()
    b = lp.b.copy()

    result = SimplexSolver().solve(lp)

    assert isinstance(
        result,
        InfeasibleResult,
    )

    y = result.certificate

    assert np.all(
        A.T @ y >= -TOL
    )

    assert (
        b @ y
        < -TOL
    )


def test_infeasibility_certificate_reverses_normalization():
    # Original system:
    #
    # x = -1
    # x >= 0
    #
    # Phase I internally turns this into -x = 1.

    lp = SEFLinearProgram(
        c=np.array([0.0]),
        A=np.array([
            [1.0],
        ]),
        b=np.array([
            -1.0,
        ]),
    )

    original_A = lp.A.copy()
    original_b = lp.b.copy()

    result = SimplexSolver().solve(lp)

    assert isinstance(
        result,
        InfeasibleResult,
    )

    y = result.certificate

    assert np.all(
        original_A.T @ y
        >= -TOL
    )

    assert (
        original_b @ y
        < -TOL
    )


def test_assignment_5_problem_1():
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
    )

    result = SimplexSolver().solve(lp)

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


def test_redundant_constraint_cleanup():
    # Second equation is twice the first.
    #
    # x1 + x2 = 1
    # 2x1 + 2x2 = 2

    lp = SEFLinearProgram(
        c=np.array([
            1.0,
            0.0,
        ]),
        A=np.array([
            [1.0, 1.0],
            [2.0, 2.0],
        ]),
        b=np.array([
            1.0,
            2.0,
        ]),
    )

    solver = SimplexSolver()

    phase_one_result = (
        solver.phase_one(lp)
    )

    assert not isinstance(
        phase_one_result,
        InfeasibleResult,
    )

    clean_lp, basis = (
        phase_one_result
    )

    assert clean_lp.A.shape == (
        1,
        2,
    )

    assert len(
        basis.indices
    ) == 1

    assert all(
        index < 2
        for index in basis.indices
    )

    result = solver.phase_two(
        clean_lp,
        basis,
    )

    assert isinstance(
        result,
        OptimalResult,
    )

    assert abs(
        result.objective_value - 1.0
    ) <= TOL


def test_objective_offset():
    # max 2x + 7
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

    result = SimplexSolver().solve(lp)

    assert isinstance(
        result,
        OptimalResult,
    )

    assert abs(
        result.objective_value
        - 13.0
    ) <= TOL