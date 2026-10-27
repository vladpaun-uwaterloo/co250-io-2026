import numpy as np

from core import Basis, SEFLinearProgram
from core.solvers.canonical import CanonicalState


TOL = 1e-9


def test_canonical_state_starts_with_identity_row_transform():
    lp = SEFLinearProgram(
        c=np.array([1.0, 2.0]),
        A=np.array([
            [1.0, 0.0],
            [0.0, 1.0],
        ]),
        b=np.array([3.0, 4.0]),
    )

    state = CanonicalState.from_lp(lp)

    np.testing.assert_allclose(
        state.row_transform,
        np.eye(2),
    )


def test_canonical_state_copies_lp():
    lp = SEFLinearProgram(
        c=np.array([1.0]),
        A=np.array([
            [2.0],
        ]),
        b=np.array([3.0]),
    )

    state = CanonicalState.from_lp(lp)

    state.lp.A[0, 0] = 10.0
    state.lp.b[0] = 20.0

    assert lp.A[0, 0] == 2.0
    assert lp.b[0] == 3.0


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

    state = CanonicalState.from_lp(lp)
    returned = state.normalize()

    assert returned is state

    np.testing.assert_allclose(
        state.lp.A,
        np.array([
            [-1.0, -2.0],
            [3.0, 4.0],
        ]),
    )

    np.testing.assert_allclose(
        state.lp.b,
        np.array([5.0, 6.0]),
    )

    np.testing.assert_allclose(
        state.row_transform,
        np.array([
            [-1.0, 0.0],
            [0.0, 1.0],
        ]),
    )

    np.testing.assert_allclose(
        state.row_transform @ original_A,
        state.lp.A,
        atol=TOL,
    )

    np.testing.assert_allclose(
        state.row_transform @ original_b,
        state.lp.b,
        atol=TOL,
    )


def test_normalization_maps_certificate_back():
    lp = SEFLinearProgram(
        c=np.array([0.0]),
        A=np.array([
            [1.0],
        ]),
        b=np.array([-1.0]),
    )

    state = CanonicalState.from_lp(lp)
    state.normalize()

    # Current normalized row is -x = 1.
    current_certificate = np.array([-1.0])

    original_certificate = state.map_certificate_to_original(current_certificate)

    np.testing.assert_allclose(
        original_certificate,
        np.array([1.0]),
    )


def test_canonicalize():
    original_A = np.array([
        [1.0, 1.0, 1.0, 0.0],
        [2.0, 1.0, 0.0, 1.0],
    ])

    original_b = np.array([4.0, 5.0])

    lp = SEFLinearProgram(
        c=np.array([3.0, 2.0, 0.0, 0.0]),
        A=original_A.copy(),
        b=original_b.copy(),
    )

    state = CanonicalState.from_lp(lp)
    basis = Basis([0, 1])

    returned = state.canonicalize(basis)

    assert returned is state

    np.testing.assert_allclose(
        state.lp.A,
        np.array([
            [1.0, 0.0, -1.0, 1.0],
            [0.0, 1.0, 2.0, -1.0],
        ]),
        atol=TOL,
    )

    np.testing.assert_allclose(
        state.lp.b,
        np.array([1.0, 3.0]),
        atol=TOL,
    )

    np.testing.assert_allclose(
        state.lp.c,
        np.array([0.0, 0.0, -1.0, -1.0]),
        atol=TOL,
    )

    assert abs(state.lp.objective_offset - 9.0) <= TOL

    np.testing.assert_allclose(
        state.lp.A[:, basis.indices],
        np.eye(2),
        atol=TOL,
    )

    np.testing.assert_allclose(
        state.row_transform @ original_A,
        state.lp.A,
        atol=TOL,
    )

    np.testing.assert_allclose(
        state.row_transform @ original_b,
        state.lp.b,
        atol=TOL,
    )


def test_build_auxiliary():
    lp = SEFLinearProgram(
        c=np.array([3.0, 2.0]),
        A=np.array([
            [1.0, 1.0],
            [2.0, 1.0],
        ]),
        b=np.array([4.0, 5.0]),
    )

    state = CanonicalState.from_lp(lp)

    aux_state, basis = state.build_auxiliary()

    np.testing.assert_allclose(
        aux_state.lp.A,
        np.array([
            [1.0, 1.0, 1.0, 0.0],
            [2.0, 1.0, 0.0, 1.0],
        ]),
    )

    np.testing.assert_allclose(
        aux_state.lp.c,
        np.array([0.0, 0.0, -1.0, -1.0]),
    )

    np.testing.assert_allclose(aux_state.lp.b, np.array([4.0, 5.0]))
    np.testing.assert_allclose(aux_state.row_transform, np.eye(2))

    assert basis.indices == [2, 3]


def test_build_auxiliary_preserves_existing_row_transform():
    lp = SEFLinearProgram(
        c=np.array([1.0]),
        A=np.array([
            [1.0],
        ]),
        b=np.array([-2.0]),
    )

    state = CanonicalState.from_lp(lp)
    state.normalize()

    aux_state, _ = state.build_auxiliary()

    np.testing.assert_allclose(
        aux_state.row_transform,
        state.row_transform,
    )


def test_remove_constraint_tracks_original_rows():
    original_A = np.array([
        [1.0, 1.0],
        [2.0, 2.0],
    ])

    original_b = np.array([1.0, 2.0])

    lp = SEFLinearProgram(
        c=np.array([1.0, 0.0]),
        A=original_A.copy(),
        b=original_b.copy(),
    )

    state = CanonicalState.from_lp(lp)

    state.remove_constraint(1)

    assert state.lp.A.shape == (1, 2)
    assert state.lp.b.shape == (1,)
    assert state.row_transform.shape == (1, 2)

    np.testing.assert_allclose(
        state.row_transform @ original_A,
        state.lp.A,
        atol=TOL,
    )

    np.testing.assert_allclose(
        state.row_transform @ original_b,
        state.lp.b,
        atol=TOL,
    )


def test_state_copy_is_independent():
    lp = SEFLinearProgram(
        c=np.array([1.0, 2.0]),
        A=np.eye(2),
        b=np.array([3.0, 4.0]),
    )

    state = CanonicalState.from_lp(lp)
    copied = state.copy()

    copied.lp.A[0, 0] = 100.0
    copied.row_transform[0, 0] = 100.0

    assert state.lp.A[0, 0] == 1.0
    assert state.row_transform[0, 0] == 1.0