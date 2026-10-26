import numpy as np

from core.basis import Basis
from core.linear_program import (
    LinearProgram,
    SEFLinearProgram,
    ObjectiveSense,
    ConstraintSense,
)


TOL = 1e-9


def test_normalize_mutates_and_returns_self():
    lp = SEFLinearProgram(
        c=np.array([1.0, 2.0]),
        A=np.array([
            [1.0, 2.0],
            [3.0, 4.0],
        ]),
        b=np.array([
            -5.0,
            6.0,
        ]),
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

    assert lp._flipped_rows == [0]


def test_reverse_normalization_certificate():
    lp = SEFLinearProgram(
        c=np.array([0.0]),
        A=np.array([[1.0]]),
        b=np.array([-1.0]),
    )

    lp.normalize()

    certificate = np.array([-1.0])

    original_certificate = (
        lp.reverse_normalization_certificate(
            certificate
        )
    )

    np.testing.assert_allclose(
        original_certificate,
        np.array([1.0]),
    )

    # Input certificate should not be mutated.
    np.testing.assert_allclose(
        certificate,
        np.array([-1.0]),
    )


def test_copy_is_independent():
    lp = SEFLinearProgram(
        c=np.array([1.0]),
        A=np.array([[2.0]]),
        b=np.array([-3.0]),
    )

    copied = lp.copy()

    copied.A[0, 0] = 99
    copied.b[0] = 20
    copied.c[0] = 7

    assert lp.A[0, 0] == 2
    assert lp.b[0] == -3
    assert lp.c[0] == 1


def test_canonicalize_mutates_lp():
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

    basis = Basis([0, 1])

    returned = lp.canonicalize(
        basis
    )

    assert returned is lp

    np.testing.assert_allclose(
        lp.A[:, [0, 1]],
        np.eye(2),
        atol=TOL,
    )

    np.testing.assert_allclose(
        lp.b,
        np.array([
            1.0,
            3.0,
        ]),
        atol=TOL,
    )

    np.testing.assert_allclose(
        lp.c,
        np.array([
            0.0,
            0.0,
            -1.0,
            -1.0,
        ]),
        atol=TOL,
    )

    assert abs(
        lp.objective_offset - 9.0
    ) <= TOL


def test_build_auxiliary():
    lp = SEFLinearProgram(
        c=np.array([
            3.0,
            2.0,
        ]),
        A=np.array([
            [1.0, 1.0],
            [2.0, 1.0],
        ]),
        b=np.array([
            4.0,
            5.0,
        ]),
    )

    aux, basis = (
        lp.build_auxiliary()
    )

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
        np.array([
            0.0,
            0.0,
            -1.0,
            -1.0,
        ]),
    )

    assert basis.indices == [2, 3]


def test_general_lp_to_sef():
    lp = LinearProgram(
        c=np.array([
            3.0,
            2.0,
        ]),
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
        np.array([
            3.0,
            2.0,
            0.0,
            0.0,
        ]),
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
        np.array([
            4.0,
            1.0,
        ]),
    )