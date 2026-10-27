from __future__ import annotations

from dataclasses import dataclass
from typing import Self

import numpy as np

from .base import SimplexSolver
from ..models import (
    SEFLinearProgram,
    Basis,
    SolveResult,
    OptimalResult,
    InfeasibleResult,
    UnboundedResult,
)


@dataclass
class CanonicalState:
    lp: SEFLinearProgram
    row_transform: np.ndarray

    @classmethod
    def from_lp(cls, lp: SEFLinearProgram) -> Self:
        return cls(
            lp=lp.copy(),
            row_transform=np.eye(lp.A.shape[0]),
        )

    def copy(self) -> Self:
        return type(self)(
            lp=self.lp.copy(),
            row_transform=self.row_transform.copy(),
        )

    def normalize(self) -> Self:
        for i in range(len(self.lp.b)):
            if self.lp.b[i] < 0:
                self.lp.A[i, :] *= -1
                self.lp.b[i] *= -1
                self.row_transform[i, :] *= -1

        return self

    def canonicalize(self, basis: Basis) -> Self:
        A_B = self.lp.A[:, basis.indices]
        c_B = self.lp.c[basis.indices]

        A_bar = np.linalg.solve(A_B, self.lp.A)
        b_bar = np.linalg.solve(A_B, self.lp.b)

        row_transform_bar = np.linalg.solve(A_B, self.row_transform)

        # A_B^T y = c_B
        y = np.linalg.solve(A_B.T, c_B)

        c_bar = self.lp.c - self.lp.A.T @ y
        objective_offset = self.lp.objective_offset + float(self.lp.b @ y)

        self.lp.A = A_bar
        self.lp.b = b_bar
        self.lp.c = c_bar
        self.lp.objective_offset = objective_offset
        self.row_transform = row_transform_bar

        return self

    def build_auxiliary(self) -> tuple[CanonicalState, Basis]:
        m, n = self.lp.A.shape

        aux_c = np.concatenate((np.zeros(n), -np.ones(m)))
        aux_A = np.hstack((self.lp.A.copy(), np.eye(m)))

        aux_lp = SEFLinearProgram(
            c=aux_c,
            A=aux_A,
            b=self.lp.b.copy(),
            objective_offset=0.0,
        )

        aux_state = CanonicalState(
            lp=aux_lp,
            row_transform=self.row_transform.copy(),
        )

        basis = Basis(range(n, n + m))

        return aux_state, basis

    def remove_constraint(self, row: int) -> Self:
        self.lp.A = np.delete(self.lp.A, row, axis=0)
        self.lp.b = np.delete(self.lp.b, row)
        self.row_transform = np.delete(self.row_transform, row, axis=0)

        return self

    def map_certificate_to_original(self, certificate: np.ndarray) -> np.ndarray:
        return self.row_transform.T @ certificate


class CanonicalSimplexSolver(SimplexSolver):
    def solve_sef(self, lp: SEFLinearProgram) -> SolveResult:
        state = CanonicalState.from_lp(lp)

        result = self.phase_one(state)

        if isinstance(result, InfeasibleResult):
            return result

        state, basis = result

        return self.phase_two(state, basis)

    def phase_one(self, state: CanonicalState) -> InfeasibleResult | tuple[CanonicalState, Basis]:
        state.normalize()

        aux_state, init_aux_basis = state.build_auxiliary()

        result = self.simplex(aux_state, init_aux_basis)

        assert isinstance(result, OptimalResult)
        assert isinstance(result.certificate, np.ndarray)

        if result.objective_value < -self.tol:
            return InfeasibleResult(result.certificate)

        return self._remove_artificials(state, aux_state, result.basis)

    def phase_two(self, state: CanonicalState, basis: Basis) -> OptimalResult | UnboundedResult:
        return self.simplex(state, basis)

    def simplex(self, state: CanonicalState, basis: Basis) -> OptimalResult | UnboundedResult:
        original_state = state.copy()

        while True:
            state.canonicalize(basis)

            m, n = state.lp.A.shape

            bfs = np.zeros(n)
            bfs[basis.indices] = state.lp.b

            if np.all(state.lp.c <= self.tol):
                return self._generate_optimal_result(original_state, state, basis, bfs)

            entering = next(
                i
                for i, ci in enumerate(state.lp.c)
                if ci > self.tol and i not in basis.indices
            )

            if np.all(state.lp.A[:, entering] <= self.tol):
                return self._generate_unbounded_result(state, basis, bfs, entering)

            ts = []

            for i in range(m):
                if state.lp.A[i, entering] > self.tol:
                    ts.append((i, state.lp.b[i] / state.lp.A[i, entering]))

            min_t = min(t for _, t in ts)
            leaving_candidates = [i for i, t in ts if abs(t - min_t) <= self.tol]

            # Bland's rule for ties
            leaving = min(leaving_candidates, key=lambda i: basis.indices[i])

            basis.replace(leaving, entering)

    def _remove_artificials(
        self,
        original_state: CanonicalState,
        aux_state: CanonicalState,
        basis: Basis,
    ) -> tuple[CanonicalState, Basis]:
        _, n = original_state.lp.A.shape

        while True:
            aux_state.canonicalize(basis)

            # Find a basic artificial variable.
            r = next(
                (r for r, j in enumerate(basis.indices) if j >= n),
                None,
            )

            # No artificial variables remain in the basis.
            if r is None:
                clean_lp = SEFLinearProgram(
                    c=original_state.lp.c.copy(),
                    A=aux_state.lp.A[:, :n].copy(),
                    b=aux_state.lp.b.copy(),
                    objective_offset=original_state.lp.objective_offset,
                )

                clean_state = CanonicalState(
                    lp=clean_lp,
                    row_transform=aux_state.row_transform.copy(),
                )

                return clean_state, basis

            row = aux_state.lp.A[r, :]
            basic = set(basis.indices)

            # Find an original nonbasic variable that can replace the artificial.
            entering = next(
                (j for j in range(n) if abs(row[j]) > self.tol and j not in basic),
                None,
            )

            if entering is None:
                # The row is redundant.
                aux_state.remove_constraint(r)
                basis.remove(r)

            else:
                # Degenerate pivot replacing the zero-valued artificial variable.
                basis.replace(r, entering)

    def _generate_optimal_result(
        self,
        original_state: CanonicalState,
        canonical_state: CanonicalState,
        basis: Basis,
        bfs: np.ndarray,
    ) -> OptimalResult:
        A_B = original_state.lp.A[:, basis.indices]
        c_B = original_state.lp.c[basis.indices]

        y = np.linalg.solve(A_B.T, c_B)

        # Map the certificate through normalization, previous canonicalizations,
        # and any deleted redundant rows.
        y = original_state.map_certificate_to_original(y)

        return OptimalResult(
            solution=bfs,
            objective_value=canonical_state.lp.objective_offset,
            basis=basis,
            certificate=y,
        )

    def _generate_unbounded_result(
        self,
        state: CanonicalState,
        basis: Basis,
        bfs: np.ndarray,
        entering: int,
    ) -> UnboundedResult:
        d = np.zeros(state.lp.A.shape[1])

        d[entering] = 1.0
        d[basis.indices] = -state.lp.A[:, entering]

        return UnboundedResult(
            solution=bfs,
            direction=d,
        )