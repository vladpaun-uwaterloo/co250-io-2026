import numpy as np
import pytest

from core import (
    LinearProgram,
    SEFLinearProgram,
    ObjectiveSense,
    ConstraintSense,
    GeneralCertificate,
    OptimalResult,
    UnboundedResult,
    InfeasibleResult,
)

from core.solvers import CanonicalSimplexSolver


TOL = 1e-8


SOLVER_CLASSES = [
    CanonicalSimplexSolver,
]


@pytest.fixture(params=SOLVER_CLASSES, ids=lambda solver_class: solver_class.__name__)
def solver(request):
    return request.param()


def test_optimal_sef(solver):
    # max 3x1 + 2x2
    #
    # x1 + x2 + x3      = 4
    # 2x1 + x2      + x4 = 5
    #
    # x >= 0
    #
    # optimum: (1, 3, 0, 0)
    # value: 9

    lp = SEFLinearProgram(
        c=np.array([3.0, 2.0, 0.0, 0.0]),
        A=np.array([
            [1.0, 1.0, 1.0, 0.0],
            [2.0, 1.0, 0.0, 1.0],
        ]),
        b=np.array([4.0, 5.0]),
    )

    original_A = lp.A.copy()
    original_b = lp.b.copy()
    original_c = lp.c.copy()

    result = solver.solve(lp)

    assert isinstance(result, OptimalResult)
    assert isinstance(result.certificate, np.ndarray)

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

    # Optimality certificate:
    #
    # c - A^T y <= 0
    # c^T x = b^T y
    assert np.all(
        original_c - original_A.T @ result.certificate <= TOL
    )

    assert abs(
        original_c @ result.solution - original_b @ result.certificate
    ) <= TOL


def test_sef_solver_does_not_mutate_input(solver):
    lp = SEFLinearProgram(
        c=np.array([3.0, 2.0, 0.0, 0.0]),
        A=np.array([
            [1.0, 1.0, 1.0, 0.0],
            [2.0, 1.0, 0.0, 1.0],
        ]),
        b=np.array([4.0, 5.0]),
        objective_offset=7.0,
    )

    original_c = lp.c.copy()
    original_A = lp.A.copy()
    original_b = lp.b.copy()
    original_offset = lp.objective_offset

    solver.solve(lp)

    np.testing.assert_allclose(lp.c, original_c)
    np.testing.assert_allclose(lp.A, original_A)
    np.testing.assert_allclose(lp.b, original_b)
    assert lp.objective_offset == original_offset


def test_unbounded_sef(solver):
    # max x1
    #
    # x1 - x2 = 1
    # x >= 0

    lp = SEFLinearProgram(
        c=np.array([1.0, 0.0]),
        A=np.array([
            [1.0, -1.0],
        ]),
        b=np.array([1.0]),
    )

    A = lp.A.copy()
    b = lp.b.copy()
    c = lp.c.copy()

    result = solver.solve(lp)

    assert isinstance(result, UnboundedResult)

    # x is feasible.
    np.testing.assert_allclose(
        A @ result.solution,
        b,
        atol=TOL,
    )

    assert np.all(result.solution >= -TOL)

    # d is a recession direction.
    np.testing.assert_allclose(
        A @ result.direction,
        np.zeros(len(b)),
        atol=TOL,
    )

    assert np.all(result.direction >= -TOL)
    assert c @ result.direction > TOL


def test_infeasible_sef(solver):
    # x = 1
    # x = 2
    # x >= 0

    lp = SEFLinearProgram(
        c=np.array([1.0]),
        A=np.array([
            [1.0],
            [1.0],
        ]),
        b=np.array([1.0, 2.0]),
    )

    A = lp.A.copy()
    b = lp.b.copy()

    result = solver.solve(lp)

    assert isinstance(result, InfeasibleResult)
    assert isinstance(result.certificate, np.ndarray)

    y = result.certificate

    # Infeasibility certificate:
    #
    # A^T y >= 0
    # b^T y < 0
    assert np.all(A.T @ y >= -TOL)
    assert b @ y < -TOL


def test_infeasible_sef_with_negative_rhs(solver):
    # x = -1
    # x >= 0

    lp = SEFLinearProgram(
        c=np.array([0.0]),
        A=np.array([
            [1.0],
        ]),
        b=np.array([-1.0]),
    )

    A = lp.A.copy()
    b = lp.b.copy()

    result = solver.solve(lp)

    assert isinstance(result, InfeasibleResult)
    assert isinstance(result.certificate, np.ndarray)

    y = result.certificate

    assert np.all(A.T @ y >= -TOL)
    assert b @ y < -TOL


def test_assignment_5_problem_1(solver):
    # max x4
    #
    # x1 - x3           =  1
    #      x2 - x3      = -2
    # x1 - x2 - x3 - x4 = 1
    #
    # x >= 0

    lp = SEFLinearProgram(
        c=np.array([0.0, 0.0, 0.0, 1.0]),
        A=np.array([
            [1.0,  0.0, -1.0,  0.0],
            [0.0,  1.0, -1.0,  0.0],
            [1.0, -1.0, -1.0, -1.0],
        ]),
        b=np.array([1.0, -2.0, 1.0]),
    )

    original_A = lp.A.copy()
    original_b = lp.b.copy()
    original_c = lp.c.copy()

    result = solver.solve(lp)

    assert isinstance(result, OptimalResult)
    assert isinstance(result.certificate, np.ndarray)

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
        original_c @ result.solution - original_b @ result.certificate
    ) <= TOL


def test_redundant_constraint(solver):
    # max x1
    #
    # x1 + x2     = 1
    # 2x1 + 2x2   = 2
    #
    # Second row is redundant.

    lp = SEFLinearProgram(
        c=np.array([1.0, 0.0]),
        A=np.array([
            [1.0, 1.0],
            [2.0, 2.0],
        ]),
        b=np.array([1.0, 2.0]),
    )

    original_A = lp.A.copy()
    original_b = lp.b.copy()
    original_c = lp.c.copy()

    result = solver.solve(lp)

    assert isinstance(result, OptimalResult)
    assert isinstance(result.certificate, np.ndarray)

    np.testing.assert_allclose(
        result.solution,
        np.array([1.0, 0.0]),
        atol=TOL,
    )

    assert abs(result.objective_value - 1.0) <= TOL

    # Certificate must still refer to both original rows.
    assert result.certificate.shape == (2,)

    assert np.all(
        original_c - original_A.T @ result.certificate <= TOL
    )

    assert abs(
        original_c @ result.solution - original_b @ result.certificate
    ) <= TOL


def test_objective_offset(solver):
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

    result = solver.solve(lp)

    assert isinstance(result, OptimalResult)

    np.testing.assert_allclose(
        result.solution,
        np.array([3.0]),
        atol=TOL,
    )

    assert abs(result.objective_value - 13.0) <= TOL


def test_zero_rhs_constraint(solver):
    # max x2
    #
    # x1      = 0
    # x1 + x2 = 1
    #
    # optimum: x = (0, 1)

    lp = SEFLinearProgram(
        c=np.array([0.0, 1.0]),
        A=np.array([
            [1.0, 0.0],
            [1.0, 1.0],
        ]),
        b=np.array([0.0, 1.0]),
    )

    result = solver.solve(lp)

    assert isinstance(result, OptimalResult)

    np.testing.assert_allclose(
        result.solution,
        np.array([0.0, 1.0]),
        atol=TOL,
    )

    assert abs(result.objective_value - 1.0) <= TOL


def test_general_max_lp(solver):
    # max 3x + 2y
    #
    # x + y  <= 4
    # 2x + y <= 5
    # x, y >= 0
    #
    # optimum: (1, 3)
    # value: 9

    lp = LinearProgram(
        c=np.array([3.0, 2.0]),
        A=np.array([
            [1.0, 1.0],
            [2.0, 1.0],
        ]),
        b=np.array([4.0, 5.0]),
        objective_sense=ObjectiveSense.MAX,
        constraint_senses=[
            ConstraintSense.LE,
            ConstraintSense.LE,
        ],
        lower_bounds=np.array([0.0, 0.0]),
        upper_bounds=np.array([np.inf, np.inf]),
    )

    result = solver.solve(lp)

    assert isinstance(result, OptimalResult)
    assert isinstance(result.certificate, GeneralCertificate)

    np.testing.assert_allclose(
        result.solution,
        np.array([1.0, 3.0]),
        atol=TOL,
    )

    assert abs(result.objective_value - 9.0) <= TOL

    np.testing.assert_allclose(
        result.certificate.constraint_multipliers,
        np.array([1.0, 1.0]),
        atol=TOL,
    )


def test_general_min_lp(solver):
    # min x
    #
    # x >= 2
    # x >= 0
    #
    # optimum x = 2

    lp = LinearProgram(
        c=np.array([1.0]),
        A=np.array([
            [1.0],
        ]),
        b=np.array([2.0]),
        objective_sense=ObjectiveSense.MIN,
        constraint_senses=[
            ConstraintSense.GE,
        ],
        lower_bounds=np.array([0.0]),
        upper_bounds=np.array([np.inf]),
    )

    result = solver.solve(lp)

    assert isinstance(result, OptimalResult)
    assert isinstance(result.certificate, GeneralCertificate)

    np.testing.assert_allclose(
        result.solution,
        np.array([2.0]),
        atol=TOL,
    )

    assert abs(result.objective_value - 2.0) <= TOL

    # For this minimization problem the >= constraint multiplier
    # should be nonnegative after mapping back.
    assert result.certificate.constraint_multipliers[0] >= -TOL


def test_general_finite_lower_and_upper_bound(solver):
    # max x
    #
    # x <= 100
    # 2 <= x <= 7
    #
    # optimum x = 7

    lp = LinearProgram(
        c=np.array([1.0]),
        A=np.array([
            [1.0],
        ]),
        b=np.array([100.0]),
        objective_sense=ObjectiveSense.MAX,
        constraint_senses=[
            ConstraintSense.LE,
        ],
        lower_bounds=np.array([2.0]),
        upper_bounds=np.array([7.0]),
    )

    result = solver.solve(lp)

    assert isinstance(result, OptimalResult)

    np.testing.assert_allclose(
        result.solution,
        np.array([7.0]),
        atol=TOL,
    )

    assert abs(result.objective_value - 7.0) <= TOL


def test_general_upper_bound_only(solver):
    # max x
    #
    # x >= 0
    # x <= 5 from its explicit variable bound.
    #
    # to_sef uses x = 5 - z.

    lp = LinearProgram(
        c=np.array([1.0]),
        A=np.array([
            [1.0],
        ]),
        b=np.array([0.0]),
        objective_sense=ObjectiveSense.MAX,
        constraint_senses=[
            ConstraintSense.GE,
        ],
        lower_bounds=np.array([-np.inf]),
        upper_bounds=np.array([5.0]),
    )

    result = solver.solve(lp)

    assert isinstance(result, OptimalResult)

    np.testing.assert_allclose(
        result.solution,
        np.array([5.0]),
        atol=TOL,
    )

    assert abs(result.objective_value - 5.0) <= TOL


def test_general_free_variable(solver):
    # max x
    #
    # x = 3
    # x free

    lp = LinearProgram(
        c=np.array([1.0]),
        A=np.array([
            [1.0],
        ]),
        b=np.array([3.0]),
        objective_sense=ObjectiveSense.MAX,
        constraint_senses=[
            ConstraintSense.EQ,
        ],
        lower_bounds=np.array([-np.inf]),
        upper_bounds=np.array([np.inf]),
    )

    result = solver.solve(lp)

    assert isinstance(result, OptimalResult)

    np.testing.assert_allclose(
        result.solution,
        np.array([3.0]),
        atol=TOL,
    )

    assert abs(result.objective_value - 3.0) <= TOL


def test_general_unbounded_direction_is_mapped_back(solver):
    # max x
    #
    # x - y = 0
    #
    # x free
    # y >= 0
    #
    # Increasing both x and y gives an unbounded direction.

    lp = LinearProgram(
        c=np.array([1.0, 0.0]),
        A=np.array([
            [1.0, -1.0],
        ]),
        b=np.array([0.0]),
        objective_sense=ObjectiveSense.MAX,
        constraint_senses=[
            ConstraintSense.EQ,
        ],
        lower_bounds=np.array([-np.inf, 0.0]),
        upper_bounds=np.array([np.inf, np.inf]),
    )

    result = solver.solve(lp)

    assert isinstance(result, UnboundedResult)

    np.testing.assert_allclose(
        lp.A @ result.solution,
        lp.b,
        atol=TOL,
    )

    np.testing.assert_allclose(
        lp.A @ result.direction,
        np.zeros(len(lp.b)),
        atol=TOL,
    )

    assert lp.c @ result.direction > TOL


def test_general_infeasible_certificate(solver):
    # x = -1
    # x >= 0

    lp = LinearProgram(
        c=np.array([0.0]),
        A=np.array([
            [1.0],
        ]),
        b=np.array([-1.0]),
        objective_sense=ObjectiveSense.MAX,
        constraint_senses=[
            ConstraintSense.EQ,
        ],
        lower_bounds=np.array([0.0]),
        upper_bounds=np.array([np.inf]),
    )

    result = solver.solve(lp)

    assert isinstance(result, InfeasibleResult)
    assert isinstance(result.certificate, GeneralCertificate)

    certificate = result.certificate

    # The original constraint and bound multipliers must cancel
    # the coefficient of x.
    np.testing.assert_allclose(
        lp.A.T @ certificate.constraint_multipliers
        + certificate.lower_bound_multipliers
        + certificate.upper_bound_multipliers,
        np.zeros(len(lp.c)),
        atol=TOL,
    )

    # Their RHS combination must produce a contradiction.
    rhs = lp.b @ certificate.constraint_multipliers

    for j in range(len(lp.c)):
        if np.isfinite(lp.lower_bounds[j]):
            rhs += lp.lower_bounds[j] * certificate.lower_bound_multipliers[j]

        if np.isfinite(lp.upper_bounds[j]):
            rhs += lp.upper_bounds[j] * certificate.upper_bound_multipliers[j]

    assert rhs < -TOL