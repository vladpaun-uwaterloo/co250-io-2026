import numpy as np

from core import (
    Basis,
    LinearProgram,
    SEFLinearProgram,
    ObjectiveSense,
    ConstraintSense,
    VariableTransformKind,
)


TOL = 1e-9


def test_basis_accepts_iterable():
    basis = Basis((0, 2, 4))

    assert basis.indices == [0, 2, 4]


def test_basis_replace_mutates():
    basis = Basis([0, 2, 4])

    returned = basis.replace(1, 3)

    assert returned is basis
    assert basis.indices == [0, 3, 4]


def test_basis_remove_removes_row():
    basis = Basis([1, 4, 7])

    returned = basis.remove(1)

    assert returned is basis
    assert basis.indices == [1, 7]


def test_basis_copy_is_independent():
    basis = Basis([0, 1, 2])
    copied = basis.copy()

    copied.replace(1, 5)

    assert basis.indices == [0, 1, 2]
    assert copied.indices == [0, 5, 2]


def test_sef_copy_is_independent():
    lp = SEFLinearProgram(
        c=np.array([1.0, 2.0]),
        A=np.array([
            [1.0, 0.0],
            [0.0, 1.0],
        ]),
        b=np.array([3.0, 4.0]),
        objective_offset=5.0,
    )

    copied = lp.copy()

    copied.c[0] = 100.0
    copied.A[0, 0] = 100.0
    copied.b[0] = 100.0
    copied.objective_offset = 100.0

    np.testing.assert_allclose(lp.c, np.array([1.0, 2.0]))
    np.testing.assert_allclose(lp.A, np.eye(2))
    np.testing.assert_allclose(lp.b, np.array([3.0, 4.0]))
    assert lp.objective_offset == 5.0


def test_to_sef_with_inequalities():
    lp = LinearProgram(
        c=np.array([3.0, 2.0]),
        A=np.array([
            [1.0, 1.0],
            [1.0, 0.0],
        ]),
        b=np.array([4.0, 1.0]),
        objective_sense=ObjectiveSense.MAX,
        constraint_senses=[
            ConstraintSense.LE,
            ConstraintSense.GE,
        ],
        lower_bounds=np.array([0.0, 0.0]),
        upper_bounds=np.array([np.inf, np.inf]),
    )

    sef, transformation = lp.to_sef()

    np.testing.assert_allclose(
        sef.c,
        np.array([3.0, 2.0, 0.0, 0.0]),
    )

    np.testing.assert_allclose(
        sef.A,
        np.array([
            [1.0, 1.0, 1.0, 0.0],
            [1.0, 0.0, 0.0, -1.0],
        ]),
    )

    np.testing.assert_allclose(sef.b, np.array([4.0, 1.0]))

    assert transformation.transformed_variable_count == 2
    assert transformation.original_constraint_count == 2


def test_to_sef_does_not_mutate_original_lp():
    lp = LinearProgram(
        c=np.array([1.0, 2.0]),
        A=np.array([
            [1.0, 2.0],
        ]),
        b=np.array([3.0]),
        objective_sense=ObjectiveSense.MAX,
        constraint_senses=[ConstraintSense.LE],
        lower_bounds=np.array([1.0, 0.0]),
        upper_bounds=np.array([5.0, np.inf]),
        objective_offset=7.0,
    )

    original_c = lp.c.copy()
    original_A = lp.A.copy()
    original_b = lp.b.copy()
    original_lower = lp.lower_bounds.copy()
    original_upper = lp.upper_bounds.copy()

    lp.to_sef()

    np.testing.assert_allclose(lp.c, original_c)
    np.testing.assert_allclose(lp.A, original_A)
    np.testing.assert_allclose(lp.b, original_b)
    np.testing.assert_allclose(lp.lower_bounds, original_lower)
    np.testing.assert_allclose(lp.upper_bounds, original_upper)
    assert lp.objective_offset == 7.0


def test_variable_transformation_maps_solution_back():
    lp = LinearProgram(
        c=np.array([1.0, 1.0, 1.0]),
        A=np.array([
            [1.0, 1.0, 1.0],
        ]),
        b=np.array([10.0]),
        objective_sense=ObjectiveSense.MAX,
        constraint_senses=[ConstraintSense.EQ],
        lower_bounds=np.array([2.0, -np.inf, -np.inf]),
        upper_bounds=np.array([7.0, 5.0, np.inf]),
    )

    sef, transformation = lp.to_sef()

    # x1 = 2 + z1
    # x2 = 5 - z2
    # x3 = z3+ - z3-
    #
    # z = (3, 1, 4, 2)
    #
    # x = (5, 4, 2)
    sef_solution = np.array([3.0, 1.0, 4.0, 2.0])

    original_solution = transformation.map_solution_to_original(sef_solution)

    np.testing.assert_allclose(
        original_solution,
        np.array([5.0, 4.0, 2.0]),
    )

    assert transformation.variable_transformations[0].kind == VariableTransformKind.LOWER_SHIFT
    assert transformation.variable_transformations[1].kind == VariableTransformKind.UPPER_SHIFT
    assert transformation.variable_transformations[2].kind == VariableTransformKind.FREE

    # x1 has both a finite lower and upper bound, so to_sef() should
    # have added one extra constraint.
    assert sef.A.shape[0] == 2


def test_variable_transformation_maps_direction_back():
    lp = LinearProgram(
        c=np.array([1.0, 1.0, 1.0]),
        A=np.array([
            [1.0, 1.0, 1.0],
        ]),
        b=np.array([0.0]),
        objective_sense=ObjectiveSense.MAX,
        constraint_senses=[ConstraintSense.EQ],
        lower_bounds=np.array([2.0, -np.inf, -np.inf]),
        upper_bounds=np.array([7.0, 5.0, np.inf]),
    )

    _, transformation = lp.to_sef()

    # dx1 = dz1
    # dx2 = -dz2
    # dx3 = dz3+ - dz3-
    sef_direction = np.array([1.0, 2.0, 3.0, 1.0])

    original_direction = transformation.map_direction_to_original(sef_direction)

    np.testing.assert_allclose(
        original_direction,
        np.array([1.0, -2.0, 2.0]),
    )


def test_min_objective_is_reversed_in_sef():
    lp = LinearProgram(
        c=np.array([2.0]),
        A=np.array([
            [1.0],
        ]),
        b=np.array([3.0]),
        objective_sense=ObjectiveSense.MIN,
        constraint_senses=[ConstraintSense.EQ],
        lower_bounds=np.array([0.0]),
        upper_bounds=np.array([np.inf]),
        objective_offset=5.0,
    )

    sef, transformation = lp.to_sef()

    np.testing.assert_allclose(sef.c, np.array([-2.0]))
    assert sef.objective_offset == -5.0
    assert transformation.objective_multiplier == -1.0

    assert transformation.map_objective_value_to_original(-11.0) == 11.0


def test_lower_and_upper_bound_creates_constraint():
    lp = LinearProgram(
        c=np.array([1.0]),
        A=np.array([
            [1.0],
        ]),
        b=np.array([100.0]),
        objective_sense=ObjectiveSense.MAX,
        constraint_senses=[ConstraintSense.LE],
        lower_bounds=np.array([2.0]),
        upper_bounds=np.array([7.0]),
    )

    sef, transformation = lp.to_sef()

    variable = transformation.variable_transformations[0]

    assert variable.kind == VariableTransformKind.LOWER_SHIFT
    assert variable.upper_bound_row == 1

    # Original constraint plus x <= 7 transformed to z <= 5.
    assert sef.A.shape[0] == 2
    np.testing.assert_allclose(sef.b, np.array([98.0, 5.0]))