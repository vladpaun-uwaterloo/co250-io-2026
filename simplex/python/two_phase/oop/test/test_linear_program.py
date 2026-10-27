import numpy as np

from core.basis import Basis
from core.linear_program import LinearProgram, SEFLinearProgram, ObjectiveSense, ConstraintSense


TOL = 1e-9


def test_normalize_tracks_row_transform():
    original_A = np.array([
        [1.0, 2.0],
        [3.0, 4.0],
    ])

    original_b = np.array([
        -5.0,
        6.0,
    ])

    lp = SEFLinearProgram(
        c=np.array([1.0, 2.0]),
        A=original_A.copy(),
        b=original_b.copy(),
    )

    returned = lp.normalize()

    assert returned is lp

    np.testing.assert_allclose(
        lp.A,
        np.array([
            [-1.0, -2.0],
            [3.0, 4.0],
        ]),
    )

    np.testing.assert_allclose(
        lp.b,
        np.array([
            5.0,
            6.0,
        ]),
    )

    np.testing.assert_allclose(
        lp._row_transform @ original_A,
        lp.A,
        atol=TOL,
    )

    np.testing.assert_allclose(
        lp._row_transform @ original_b,
        lp.b,
        atol=TOL,
    )


def test_normalization_certificate_mapping():
    lp = SEFLinearProgram(
        c=np.array([0.0]),
        A=np.array([
            [1.0],
        ]),
        b=np.array([
            -1.0,
        ]),
    )

    lp.normalize()

    # The normalized system is -x = 1.
    # y = -1 is therefore a certificate in the normalized row representation.
    y_normalized = np.array([-1.0])

    y_original = lp.map_certificate_to_original(y_normalized)

    np.testing.assert_allclose(
        y_original,
        np.array([1.0]),
        atol=TOL,
    )


def test_copy_preserves_row_transform_but_is_independent():
    lp = SEFLinearProgram(
        c=np.array([1.0]),
        A=np.array([[2.0]]),
        b=np.array([-3.0]),
    )

    lp.normalize()

    copied = lp.copy()

    np.testing.assert_allclose(
        copied._row_transform,
        lp._row_transform,
    )

    copied.A[0, 0] = 99
    copied.b[0] = 20
    copied.c[0] = 7
    copied._row_transform[0, 0] = 50

    assert lp.A[0, 0] == -2
    assert lp.b[0] == 3
    assert lp.c[0] == 1
    assert lp._row_transform[0, 0] == -1


def test_canonicalize_tracks_row_transform():
    original_A = np.array([
        [1.0, 1.0, 1.0, 0.0],
        [2.0, 1.0, 0.0, 1.0],
    ])

    original_b = np.array([
        4.0,
        5.0,
    ])

    lp = SEFLinearProgram(
        c=np.array([3.0, 2.0, 0.0, 0.0]),
        A=original_A.copy(),
        b=original_b.copy(),
    )

    basis = Basis([0, 1])

    returned = lp.canonicalize(basis)

    assert returned is lp

    np.testing.assert_allclose(
        lp.A[:, basis.indices],
        np.eye(2),
        atol=TOL,
    )

    np.testing.assert_allclose(
        lp.b,
        np.array([1.0, 3.0]),
        atol=TOL,
    )

    np.testing.assert_allclose(
        lp.c,
        np.array([0.0, 0.0, -1.0, -1.0]),
        atol=TOL,
    )

    assert abs(lp.objective_offset - 9.0) <= TOL

    # Verify the row-transform invariant directly.
    np.testing.assert_allclose(
        lp._row_transform @ original_A,
        lp.A,
        atol=TOL,
    )

    np.testing.assert_allclose(
        lp._row_transform @ original_b,
        lp.b,
        atol=TOL,
    )


def test_remove_constraint_updates_row_transform():
    original_A = np.array([
        [1.0, 1.0],
        [2.0, 2.0],
    ])

    original_b = np.array([
        1.0,
        2.0,
    ])

    lp = SEFLinearProgram(
        c=np.array([1.0, 0.0]),
        A=original_A.copy(),
        b=original_b.copy(),
    )

    lp.remove_constraint(1)

    assert lp.A.shape == (1, 2)
    assert lp.b.shape == (1,)
    assert lp._row_transform.shape == (1, 2)

    np.testing.assert_allclose(
        lp._row_transform @ original_A,
        lp.A,
        atol=TOL,
    )

    np.testing.assert_allclose(
        lp._row_transform @ original_b,
        lp.b,
        atol=TOL,
    )

    certificate = lp.map_certificate_to_original(np.array([1.0]))

    assert certificate.shape == (2,)


def test_build_auxiliary():
    lp = SEFLinearProgram(
        c=np.array([3.0, 2.0]),
        A=np.array([
            [1.0, 1.0],
            [2.0, 1.0],
        ]),
        b=np.array([
            4.0,
            5.0,
        ]),
    )

    aux, basis = lp.build_auxiliary()

    assert aux.A.shape == (2, 4)

    np.testing.assert_allclose(
        aux.A[:, :2],
        lp.A,
    )

    np.testing.assert_allclose(
        aux.A[:, 2:],
        np.eye(2),
    )

    np.testing.assert_allclose(
        aux.c,
        np.array([0.0, 0.0, -1.0, -1.0]),
    )

    np.testing.assert_allclose(
        aux.b,
        lp.b,
    )

    np.testing.assert_allclose(
        aux._row_transform,
        lp._row_transform,
    )

    assert basis.indices == [2, 3]


def test_general_lp_to_sef():
    lp = LinearProgram(
        c=np.array([3.0, 2.0]),
        A=np.array([
            [1.0, 1.0],
            [1.0, 0.0],
        ]),
        b=np.array([
            4.0,
            1.0,
        ]),
        objective_sense=ObjectiveSense.MAX,
        constraint_senses=[
            ConstraintSense.LE,
            ConstraintSense.GE,
        ],
        lower_bounds=np.array([
            0.0,
            0.0,
        ]),
        upper_bounds=np.array([
            np.inf,
            np.inf,
        ]),
    )

    sef = lp.to_sef()

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

    np.testing.assert_allclose(
        sef.b,
        np.array([4.0, 1.0]),
    )

def test_row_transform_starts_as_identity():
    lp = SEFLinearProgram(
        c=np.array([1.0, 2.0]),
        A=np.array([
            [1.0, 0.0],
            [0.0, 1.0],
        ]),
        b=np.array([3.0, 4.0]),
    )

    np.testing.assert_allclose(
        lp._row_transform,
        np.eye(2),
    )