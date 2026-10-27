from .result import SolveResult, OptimalResult, InfeasibleResult, UnboundedResult, GeneralCertificate
from .linear_program import (
    LinearProgram,
    SEFLinearProgram,
    SEFTransformation,
    VariableTransformKind,
)
from .basis import Basis

import numpy as np


class SimplexSolver:
    def __init__(self, tol: float = 1e-9):
        self.tol = tol

    def solve(self, lp: LinearProgram | SEFLinearProgram) -> SolveResult:
        if isinstance(lp, LinearProgram):
            return self._solve_general(lp)

        return self._solve_sef(lp)

    def _solve_general(self, lp: LinearProgram) -> SolveResult:
        sef_lp, transformation = lp.to_sef()

        # Simplex mutates the SEF LP, so preserve the original SEF representation.
        # We need it later when translating certificates.
        original_sef_lp = sef_lp.copy()

        result = self._solve_sef(sef_lp)

        if isinstance(result, OptimalResult):
            assert isinstance(result.certificate, np.ndarray)

            return OptimalResult(
                solution=transformation.map_solution_to_original(result.solution),
                objective_value=transformation.map_objective_value_to_original(result.objective_value),
                basis=result.basis,
                certificate=self._map_optimal_certificate(original_sef_lp, transformation, result.certificate),
            )

        if isinstance(result, UnboundedResult):
            return UnboundedResult(
                solution=transformation.map_solution_to_original(result.solution),
                direction=transformation.map_direction_to_original(result.direction),
            )

        assert isinstance(result, InfeasibleResult)
        assert isinstance(result.certificate, np.ndarray)

        return InfeasibleResult(
            certificate=self._map_infeasible_certificate(original_sef_lp, transformation, result.certificate)
        )

    def _solve_sef(self, lp: SEFLinearProgram) -> SolveResult:
        result = self.phase_one(lp)

        if isinstance(result, InfeasibleResult):
            return result

        lp, basis = result
        return self.phase_two(lp, basis)

    def phase_one(self, lp: SEFLinearProgram) -> InfeasibleResult | tuple[SEFLinearProgram, Basis]:
        aux_lp, init_aux_basis = lp.normalize().build_auxiliary()

        result = self.simplex(aux_lp, init_aux_basis)
        assert isinstance(result, OptimalResult)
        assert isinstance(result.certificate, np.ndarray)

        if result.objective_value < -self.tol:
            return InfeasibleResult(result.certificate)

        return self._remove_artificials(lp, aux_lp, result.basis)

    def phase_two(self, lp: SEFLinearProgram, basis: Basis) -> OptimalResult | UnboundedResult:
        return self.simplex(lp, basis)

    def simplex(self, lp: SEFLinearProgram, basis: Basis) -> OptimalResult | UnboundedResult:
        original_lp = lp.copy()

        while True:
            lp.canonicalize(basis)

            m, n = lp.A.shape

            bfs = np.zeros(n)
            bfs[basis.indices] = lp.b

            if np.all(lp.c <= self.tol):
                return self._generate_optimal_result(original_lp, lp, basis, bfs)

            entering = next(i for i, ci in enumerate(lp.c) if ci > self.tol and i not in basis.indices)

            if np.all(lp.A[:, entering] <= self.tol):
                return self._generate_unbounded_result(lp, basis, bfs, entering)

            ts = []

            for i in range(m):
                if lp.A[i, entering] > self.tol:
                    ts.append((i, lp.b[i] / lp.A[i, entering]))

            min_t = min(t for _, t in ts)
            leaving_candidates = [i for i, t in ts if abs(t - min_t) <= self.tol]

            leaving = min(leaving_candidates, key=lambda i: basis.indices[i])

            basis.replace(leaving, entering)

    def _remove_artificials(
        self,
        lp: SEFLinearProgram,
        aux_lp: SEFLinearProgram,
        basis: Basis,
    ) -> tuple[SEFLinearProgram, Basis]:
        _, n = lp.A.shape

        while True:
            aux_lp.canonicalize(basis)

            r = next(
                (r for r, j in enumerate(basis.indices) if j >= n),
                None,
            )

            if r is None:
                lp.adopt_constraint_system(aux_lp, n)
                return lp, basis

            row = aux_lp.A[r, :]

            basic = set(basis.indices)

            entering = next(
                (j for j in range(n) if abs(row[j]) > self.tol and j not in basic),
                None,
            )

            if entering is None:
                aux_lp.remove_constraint(r)
                basis.remove(r)
            else:
                basis.replace(r, entering)

    def _generate_optimal_result(
        self,
        original_lp: SEFLinearProgram,
        canonical_lp: SEFLinearProgram,
        basis: Basis,
        bfs: np.ndarray,
    ) -> OptimalResult:
        A_B = original_lp.A[:, basis.indices]
        c_B = original_lp.c[basis.indices]

        y = np.linalg.solve(A_B.T, c_B)
        y = original_lp.map_certificate_to_original(y)

        return OptimalResult(
            solution=bfs,
            objective_value=canonical_lp.objective_offset,
            basis=basis,
            certificate=y,
        )

    def _generate_unbounded_result(
        self,
        lp: SEFLinearProgram,
        basis: Basis,
        bfs: np.ndarray,
        entering: int,
    ) -> UnboundedResult:
        d = np.zeros(lp.A.shape[1])

        d[entering] = 1.0
        d[basis.indices] = -lp.A[:, entering]

        return UnboundedResult(
            solution=bfs,
            direction=d,
        )

    def _map_optimal_certificate(
        self,
        sef_lp: SEFLinearProgram,
        transformation: SEFTransformation,
        certificate: np.ndarray,
    ) -> GeneralCertificate:
        m = transformation.original_constraint_count
        n = len(transformation.variable_transformations)

        constraint_multipliers = certificate[:m].copy()
        lower_bound_multipliers = np.zeros(n)
        upper_bound_multipliers = np.zeros(n)

        # For the standard equality form:
        #
        # reduced_cost = c - A^T y <= 0
        reduced_costs = sef_lp.c - sef_lp.A.T @ certificate

        for j, variable_transformation in enumerate(transformation.variable_transformations):
            if variable_transformation.kind == VariableTransformKind.LOWER_SHIFT:
                transformed_index = variable_transformation.indices[0]

                # The nonnegativity z >= 0 corresponds to the original
                # lower bound x >= L.
                lower_bound_multipliers[j] = reduced_costs[transformed_index]

                # If x also had a finite upper bound, it produced an explicit
                # SEF row z <= U - L.
                if variable_transformation.upper_bound_row >= 0:
                    upper_bound_multipliers[j] = certificate[variable_transformation.upper_bound_row]

            elif variable_transformation.kind == VariableTransformKind.UPPER_SHIFT:
                transformed_index = variable_transformation.indices[0]

                # x = U - z, so z >= 0 corresponds to x <= U.
                upper_bound_multipliers[j] = -reduced_costs[transformed_index]

            elif variable_transformation.kind == VariableTransformKind.FREE:
                # A free variable has no bound multipliers.
                pass

        # Everything above certifies the maximization-equivalent objective.
        #
        # For an original MIN problem the SEF objective was negated.
        # Multiplying all multipliers by -1 maps the certificate back to
        # the original minimization problem.
        multiplier = transformation.objective_multiplier

        constraint_multipliers *= multiplier
        lower_bound_multipliers *= multiplier
        upper_bound_multipliers *= multiplier

        return GeneralCertificate(
            constraint_multipliers=constraint_multipliers,
            lower_bound_multipliers=lower_bound_multipliers,
            upper_bound_multipliers=upper_bound_multipliers,
        )

    def _map_infeasible_certificate(
        self,
        sef_lp: SEFLinearProgram,
        transformation: SEFTransformation,
        certificate: np.ndarray,
    ) -> GeneralCertificate:
        m = transformation.original_constraint_count
        n = len(transformation.variable_transformations)

        constraint_multipliers = certificate[:m].copy()
        lower_bound_multipliers = np.zeros(n)
        upper_bound_multipliers = np.zeros(n)

        # For an SEF infeasibility certificate:
        #
        # A^T y >= 0
        column_products = sef_lp.A.T @ certificate

        for j, variable_transformation in enumerate(transformation.variable_transformations):
            if variable_transformation.kind == VariableTransformKind.LOWER_SHIFT:
                transformed_index = variable_transformation.indices[0]

                # A^T y + lambda_L + lambda_U = 0
                lower_bound_multipliers[j] = -column_products[transformed_index]

                if variable_transformation.upper_bound_row >= 0:
                    upper_bound_multipliers[j] = certificate[variable_transformation.upper_bound_row]

            elif variable_transformation.kind == VariableTransformKind.UPPER_SHIFT:
                transformed_index = variable_transformation.indices[0]

                upper_bound_multipliers[j] = column_products[transformed_index]

            elif variable_transformation.kind == VariableTransformKind.FREE:
                pass

        return GeneralCertificate(
            constraint_multipliers=constraint_multipliers,
            lower_bound_multipliers=lower_bound_multipliers,
            upper_bound_multipliers=upper_bound_multipliers,
        )