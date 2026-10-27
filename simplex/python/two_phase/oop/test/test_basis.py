from core.basis import Basis


def test_basis_accepts_any_iterable():
    basis = Basis((0, 2, 4))

    assert basis.indices == [0, 2, 4]


def test_replace_mutates_basis():
    basis = Basis([0, 2, 4])

    returned = basis.replace(
        1,
        3,
    )

    assert returned is basis
    assert basis.indices == [0, 3, 4]


def test_remove_removes_basis_row():
    basis = Basis([1, 4, 7])

    returned = basis.remove(1)

    assert returned is basis
    assert basis.indices == [1, 7]