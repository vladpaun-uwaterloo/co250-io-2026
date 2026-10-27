import numpy as np

from core.linear_program import SEFLinearProgram
from core.result import OptimalResult, UnboundedResult, InfeasibleResult
from core.solver import SimplexSolver


TOL = 1e-8


def test_solver_optimal():
    lp = SEFLinearProgram(
        c=np.array([3.0, 2.0, 0.0, 0.0]),
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

    assert isinstance(result, OptimalResult)

    np.testing.assert_allclose(
        result.solution,
        np.array([1.0, 3.0, 0.0, 0.0]),
        atol=TOL,
    )

    assert abs(result.objective_value - 9.0) <= TOL

    np.testing.assert_allclose(
        original_A @ result.solution,
        original_b,
        atol=TOL,
    )

    assert np.all(result.solution >= -TOL)

    assert np.all(
        original_c - original_A.T @ result.certificate <= TOL
    )

    assert abs(
        original_c @ result.solution
        - original_b @ result.certificate
    ) <= TOL

    np.testing.assert_allclose(
        result.certificate,
        np.array([1.0, 1.0]),
        atol=TOL,
    )


def test_solver_unbounded():
    lp = SEFLinearProgram(
        c=np.array([1.0, 0.0]),
        A=np.array([
            [1.0, -1.0],
        ]),
        b=np.array([1.0]),
    )

    original_A = lp.A.copy()
    original_b = lp.b.copy()
    original_c = lp.c.copy()

    result = SimplexSolver().solve(lp)

    assert isinstance(result, UnboundedResult)

    np.testing.assert_allclose(
        original_A @ result.solution,
        original_b,
        atol=TOL,
    )

    np.testing.assert_allclose(
        original_A @ result.direction,
        np.zeros(len(original_b)),
        atol=TOL,
    )

    assert np.all(result.direction >= -TOL)
    assert original_c @ result.direction > TOL


def test_solver_infeasible():
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

    original_A = lp.A.copy()
    original_b = lp.b.copy()

    result = SimplexSolver().solve(lp)

    assert isinstance(result, InfeasibleResult)

    y = result.certificate

    assert y.shape == (2,)
    assert np.all(original_A.T @ y >= -TOL)
    assert original_b @ y < -TOL


def test_infeasibility_certificate_with_normalization():
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

    assert isinstance(result, InfeasibleResult)

    y = result.certificate

    assert np.all(original_A.T @ y >= -TOL)
    assert original_b @ y < -TOL


def test_assignment_5_problem_1():
    lp = SEFLinearProgram(
        c=np.array([0.0, 0.0, 0.0, 1.0]),
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

    original_A = lp.A.copy()
    original_b = lp.b.copy()
    original_c = lp.c.copy()

    result = SimplexSolver().solve(lp)

    assert isinstance(result, OptimalResult)

    np.testing.assert_allclose(
        result.solution,
        np.array([3.0, 0.0, 2.0, 0.0]),
        atol=TOL,
    )

    assert abs(result.objective_value) <= TOL

    assert np.all(
        original_c - original_A.T @ result.certificate <= TOL
    )

    assert abs(
        original_c @ result.solution
        - original_b @ result.certificate
    ) <= TOL


def test_redundant_constraint_certificate_mapping():
    # Second constraint is redundant:
    #
    # x1 + x2 = 1
    # 2x1 + 2x2 = 2
    #
    # max x1

    lp = SEFLinearProgram(
        c=np.array([1.0, 0.0]),
        A=np.array([
            [1.0, 1.0],
            [2.0, 2.0],
        ]),
        b=np.array([
            1.0,
            2.0,
        ]),
    )

    original_A = lp.A.copy()
    original_b = lp.b.copy()
    original_c = lp.c.copy()

    result = SimplexSolver().solve(lp)

    assert isinstance(result, OptimalResult)

    np.testing.assert_allclose(
        result.solution,
        np.array([1.0, 0.0]),
        atol=TOL,
    )

    assert abs(result.objective_value - 1.0) <= TOL

    # Even though one row was removed internally, the certificate
    # is expressed in terms of the original two constraints.
    assert result.certificate.shape == (2,)

    assert np.all(
        original_c - original_A.T @ result.certificate <= TOL
    )

    assert abs(
        original_c @ result.solution
        - original_b @ result.certificate
    ) <= TOL


def test_objective_offset():
    lp = SEFLinearProgram(
        c=np.array([2.0]),
        A=np.array([
            [1.0],
        ]),
        b=np.array([3.0]),
        objective_offset=7.0,
    )

    result = SimplexSolver().solve(lp)

    assert isinstance(result, OptimalResult)

    np.testing.assert_allclose(
        result.solution,
        np.array([3.0]),
        atol=TOL,
    )

    assert abs(result.objective_value - 13.0) <= TOL