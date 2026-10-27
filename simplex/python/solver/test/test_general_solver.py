import numpy as np

from core.linear_program import LinearProgram, ObjectiveSense, ConstraintSense
from core.result import OptimalResult, UnboundedResult, InfeasibleResult, GeneralCertificate
from core.solver import SimplexSolver


TOL = 1e-8


def test_general_max_lp():
    # max 3x + 2y
    #
    # x + y <= 4
    # 2x + y <= 5
    # x, y >= 0
    #
    # optimum: (1, 3), value 9

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

    result = SimplexSolver().solve(lp)

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


def test_general_min_lp():
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

    result = SimplexSolver().solve(lp)

    assert isinstance(result, OptimalResult)
    assert isinstance(result.certificate, GeneralCertificate)

    np.testing.assert_allclose(
        result.solution,
        np.array([2.0]),
        atol=TOL,
    )

    assert abs(result.objective_value - 2.0) <= TOL

    # For a minimization problem, a >= constraint has a
    # nonnegative multiplier.
    assert result.certificate.constraint_multipliers[0] >= -TOL


def test_finite_lower_and_upper_bound():
    # max x
    #
    # 2 <= x <= 7
    #
    # plus x <= 100 just so there is an ordinary constraint too.
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

    result = SimplexSolver().solve(lp)

    assert isinstance(result, OptimalResult)

    np.testing.assert_allclose(
        result.solution,
        np.array([7.0]),
        atol=TOL,
    )

    assert abs(result.objective_value - 7.0) <= TOL


def test_upper_bound_only_variable():
    # max x
    #
    # x >= 0
    # x <= 5 from the variable bound.
    #
    # The variable itself has no finite lower bound, so to_sef uses
    # x = 5 - z.

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

    result = SimplexSolver().solve(lp)

    assert isinstance(result, OptimalResult)

    np.testing.assert_allclose(
        result.solution,
        np.array([5.0]),
        atol=TOL,
    )

    assert abs(result.objective_value - 5.0) <= TOL


def test_free_variable():
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

    result = SimplexSolver().solve(lp)

    assert isinstance(result, OptimalResult)

    np.testing.assert_allclose(
        result.solution,
        np.array([3.0]),
        atol=TOL,
    )

    assert abs(result.objective_value - 3.0) <= TOL


def test_general_unbounded_direction_is_mapped_back():
    # max x
    #
    # x - y = 0
    #
    # x free
    # y >= 0
    #
    # Direction in original coordinates should increase both x and y.

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

    result = SimplexSolver().solve(lp)

    assert isinstance(result, UnboundedResult)

    np.testing.assert_allclose(
        lp.A @ result.direction,
        np.zeros(1),
        atol=TOL,
    )

    assert lp.c @ result.direction > TOL


def test_general_infeasible_certificate():
    # x = -1
    # x >= 0 from the lower bound

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

    result = SimplexSolver().solve(lp)

    assert isinstance(result, InfeasibleResult)
    assert isinstance(result.certificate, GeneralCertificate)

    certificate = result.certificate

    # Combining the original constraint and bound multipliers
    # should eliminate x.
    np.testing.assert_allclose(
        lp.A.T @ certificate.constraint_multipliers
        + certificate.lower_bound_multipliers
        + certificate.upper_bound_multipliers,
        np.zeros(1),
        atol=TOL,
    )

    # The RHS combination gives a contradiction.
    rhs = lp.b @ certificate.constraint_multipliers

    for j in range(len(lp.c)):
        if np.isfinite(lp.lower_bounds[j]):
            rhs += lp.lower_bounds[j] * certificate.lower_bound_multipliers[j]

        if np.isfinite(lp.upper_bounds[j]):
            rhs += lp.upper_bounds[j] * certificate.upper_bound_multipliers[j]

    assert rhs < -TOL