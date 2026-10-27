import numpy as np

from core.models import SEFLinearProgram, Basis
from core.transform import normalize, canonicalize, build_aux


TOL = 1e-9


def test_normalize_flips_negative_rhs_rows():
    lp = SEFLinearProgram(
        c=np.array([1.0, 2.0]),
        A=np.array([
            [1.0, 2.0],
            [3.0, 4.0],
        ]),
        b=np.array([-5.0, 6.0]),
        objective_offset=0.0,
    )

    normalized, flipped_rows = normalize(lp)

    np.testing.assert_allclose(
        normalized.A,
        np.array([
            [-1.0, -2.0],
            [3.0, 4.0],
        ]),
    )

    np.testing.assert_allclose(
        normalized.b,
        np.array([5.0, 6.0]),
    )

    assert flipped_rows == [0]


def test_normalize_does_not_change_nonnegative_rhs():
    lp = SEFLinearProgram(
        c=np.array([1.0]),
        A=np.array([
            [1.0],
            [2.0],
        ]),
        b=np.array([0.0, 3.0]),
        objective_offset=0.0,
    )

    normalized, flipped_rows = normalize(lp)

    np.testing.assert_allclose(
        normalized.A,
        lp.A,
    )

    np.testing.assert_allclose(
        normalized.b,
        lp.b,
    )

    assert flipped_rows == []


def test_canonicalize():
    lp = SEFLinearProgram(
        c=np.array([3.0, 2.0, 0.0, 0.0]),
        A=np.array([
            [1.0, 1.0, 1.0, 0.0],
            [2.0, 1.0, 0.0, 1.0],
        ]),
        b=np.array([4.0, 5.0]),
        objective_offset=0.0,
    )

    basis = Basis((0, 1))

    canonical, _ = canonicalize(lp, basis)

    # Basis columns must become the identity.
    np.testing.assert_allclose(
        canonical.A[:, [0, 1]],
        np.eye(2),
        atol=TOL,
    )

    # Basic solution is x1 = 1, x2 = 3.
    np.testing.assert_allclose(
        canonical.b,
        np.array([1.0, 3.0]),
        atol=TOL,
    )

    # Basic reduced costs must be zero.
    np.testing.assert_allclose(
        canonical.c[[0, 1]],
        np.zeros(2),
        atol=TOL,
    )

    # Here the full reduced-cost vector is known exactly.
    np.testing.assert_allclose(
        canonical.c,
        np.array([0.0, 0.0, -1.0, -1.0]),
        atol=TOL,
    )

    assert abs(canonical.objective_offset - 9.0) <= TOL


def test_build_auxiliary_problem():
    lp = SEFLinearProgram(
        c=np.array([3.0, 2.0]),
        A=np.array([
            [1.0, 1.0],
            [2.0, 1.0],
        ]),
        b=np.array([4.0, 5.0]),
        objective_offset=0.0,
    )

    aux, basis = build_aux(lp)

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

    assert basis.indices == (2, 3)