"""
constraint_solver

name: constraint_solver.py
by:   Dmytry Lavrov
date: August 2025
desc:
    Solves geometric constraints for code CAD.
license:

    Copyright 2025 Dmytry Lavrov

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.

"""

"""constraint_solver.py

A pure NumPy & Dual-Number constraint solver for code CAD.
"""

import collections
import functools
import math
import operator
import numpy as np


class SolverError(Exception):
    pass


class SolverSettings:
    max_tolerance = 1e-7
    verbose = False
    settings_metadata = {"max_tolerance": {"combine": min}}

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            self.__dict__[k] = v

    def append_settings(self, other: "SolverSettings"):
        for k, v in other.__dict__.items():
            if k in self.__dict__:
                if k in self.settings_metadata:
                    try:
                        self.__dict__[k] = self.settings_metadata[k]["combine"](
                            self.__dict__[k], v
                        )
                    except KeyError:
                        pass
            else:
                self.__dict__[k] = v


solver_settings = SolverSettings()


def set_verbose(v=True):
    global solver_settings
    solver_settings.verbose = v


opportunistic = False


def set_opportunistic(v=True):
    global opportunistic
    opportunistic = v


# ==============================================================================
# Dual Number Forward-Mode Automatic Differentiation Engine
# ==============================================================================


class Dual:
    """Forward-mode automatic differentiation using vector Dual Numbers."""

    __slots__ = ("val", "grad")

    def __init__(self, val, grad=None):
        self.val = float(val)
        self.grad = grad  # 1D numpy array representing partial derivatives

    @classmethod
    def lift(cls, x):
        if isinstance(x, Dual):
            return x
        return cls(float(x), None)

    def __float__(self):
        return float(self.val)

    def __int__(self):
        return int(self.val)

    def __round__(self, ndigits=None):
        return round(self.val, ndigits) if ndigits is not None else round(self.val)

    def __eq__(self, other):
        if isinstance(other, Dual):
            return self.val == other.val
        return self.val == float(other)

    def __lt__(self, other):
        if isinstance(other, Dual):
            return self.val < other.val
        return self.val < float(other)

    def __le__(self, other):
        if isinstance(other, Dual):
            return self.val <= other.val
        return self.val <= float(other)

    def __gt__(self, other):
        if isinstance(other, Dual):
            return self.val > other.val
        return self.val > float(other)

    def __ge__(self, other):
        if isinstance(other, Dual):
            return self.val >= other.val
        return self.val >= float(other)

    def make_zero(self, name=""):
        if abs(self.val) > 1e-7:
            raise SolverError(f"Inconsistent constant constraint: {self.val} == 0")
        return self

    def __add__(self, other):
        if isinstance(other, Dual):
            if self.grad is not None and other.grad is not None:
                g = self.grad + other.grad
            else:
                g = self.grad if self.grad is not None else other.grad
            return Dual(self.val + other.val, g)
        return Dual(self.val + float(other), self.grad)

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        if isinstance(other, Dual):
            if self.grad is not None and other.grad is not None:
                g = self.grad - other.grad
            elif self.grad is not None:
                g = self.grad
            elif other.grad is not None:
                g = -other.grad
            else:
                g = None
            return Dual(self.val - other.val, g)
        return Dual(self.val - float(other), self.grad)

    def __rsub__(self, other):
        g = -self.grad if self.grad is not None else None
        return Dual(float(other) - self.val, g)

    def __mul__(self, other):
        if isinstance(other, Dual):
            g_s = self.grad if self.grad is not None else 0.0
            g_o = other.grad if other.grad is not None else 0.0
            return Dual(self.val * other.val, self.val * g_o + other.val * g_s)
        g = self.grad * float(other) if self.grad is not None else None
        return Dual(self.val * float(other), g)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        if isinstance(other, Dual):
            denom = other.val
            inv = 1.0 / denom
            g_s = self.grad if self.grad is not None else 0.0
            g_o = other.grad if other.grad is not None else 0.0
            g = (g_s * denom - self.val * g_o) * (inv * inv)
            return Dual(self.val * inv, g)
        inv = 1.0 / float(other)
        g = self.grad * inv if self.grad is not None else None
        return Dual(self.val * inv, g)

    def __rtruediv__(self, other):
        denom_sq = self.val * self.val
        g = (-float(other) / denom_sq) * self.grad if self.grad is not None else None
        return Dual(float(other) / self.val, g)

    def __pow__(self, power):
        p = float(power)
        val = self.val**p
        if self.grad is not None:
            g = (p * (self.val ** (p - 1.0))) * self.grad
        else:
            g = None
        return Dual(val, g)

    def __mod__(self, other):
        val = self.val % float(other)
        return Dual(val, self.grad)

    def __neg__(self):
        return Dual(-self.val, -self.grad if self.grad is not None else None)

    def __pos__(self):
        return self

    def __abs__(self):
        sign = 1.0 if self.val >= 0 else -1.0
        return Dual(abs(self.val), sign * self.grad if self.grad is not None else None)

    def __repr__(self):
        return f"Dual(val={self.val:.6f})"


# Elementary Dual Math Functions
def d_sqrt(x):
    if isinstance(x, Dual):
        val = math.sqrt(max(x.val, 0.0))
        g = (0.5 / (val + 1e-30)) * x.grad if x.grad is not None else None
        return Dual(val, g)
    return math.sqrt(max(float(x), 0.0))


def d_hypot(x, y):
    if isinstance(x, Dual) or isinstance(y, Dual):
        dx = Dual.lift(x)
        dy = Dual.lift(y)
        val = math.hypot(dx.val, dy.val)
        inv = 1.0 / (val + 1e-30)
        gx = dx.grad if dx.grad is not None else 0.0
        gy = dy.grad if dy.grad is not None else 0.0
        return Dual(val, (dx.val * gx + dy.val * gy) * inv)
    return math.hypot(float(x), float(y))


def d_sin(x):
    if isinstance(x, Dual):
        return Dual(
            math.sin(x.val),
            math.cos(x.val) * x.grad if x.grad is not None else None,
        )
    return math.sin(float(x))


def d_cos(x):
    if isinstance(x, Dual):
        return Dual(
            math.cos(x.val),
            -math.sin(x.val) * x.grad if x.grad is not None else None,
        )
    return math.cos(float(x))


def d_atan2(y, x):
    if isinstance(y, Dual) or isinstance(x, Dual):
        dy = Dual.lift(y)
        dx = Dual.lift(x)
        val = math.atan2(dy.val, dx.val)
        # Floor prevents division by ~0 from dominating the Jacobian SVD
        denom = max(dx.val**2 + dy.val**2, 1e-10)
        gy = dy.grad if dy.grad is not None else 0.0
        gx = dx.grad if dx.grad is not None else 0.0
        return Dual(val, (dx.val * gy - dy.val * gx) / denom)
    return math.atan2(float(y), float(x))


def d_abs(x):
    if isinstance(x, Dual):
        return abs(x)
    return abs(float(x))


# ==============================================================================
# Graph Traversal, AST, and Variables
# ==============================================================================


def is_iterable(a):
    return isinstance(a, collections.abc.Sequence) and not isinstance(
        a, (str, bytes, Dual)
    )


def recursive_unpack(list_or_item):
    if is_iterable(list_or_item):
        for i in list_or_item:
            yield from recursive_unpack(i)
    else:
        yield list_or_item


class SolverEntity:
    settings = None
    name = ""

    def append_settings_to(self, s):
        if self.settings:
            s.append_settings(self.settings)


class Variable(SolverEntity):
    """A value that the solver will solve for."""

    def __init__(self, initial_value=0.0):
        self.initial_value = float(initial_value)
        self.constraints = set()
        self.solution = None

    def solve(self):
        if self.solution is None:
            solve_everything(self)
        return float(self.solution)

    def solution_as_float_or_none(self):
        return float(self.solution) if self.solution is not None else None

    @property
    def s(self):
        return self.solve()

    @property
    def magic(self):
        return self

    @magic.setter
    def magic(self, v):
        (self - v).make_zero()


class WrappedFunction(SolverEntity):
    """Represents a function used as a geometric constraint."""

    good_func = None

    def __init__(self, function_or_variable, arguments=None):
        if isinstance(function_or_variable, Variable):
            self.arguments = [function_or_variable]
            self.function = lambda v: v
        else:
            self.function = function_or_variable
            self.arguments = arguments if arguments is not None else []

    def make_zero(self, name=""):
        self.name = name
        variables_to_solve = set()
        for a in recursive_unpack(self.arguments):
            if isinstance(a, Variable):
                a.constraints.add(self)
                variables_to_solve.add(a)

        if opportunistic:
            for a in variables_to_solve:
                solve_everything(a, solve_even_if_underconstrained=False)
        return self

    @property
    def magic(self):
        return self

    @magic.setter
    def magic(self, v):
        (self - v).make_zero()

    @property
    def initial_value(self):
        return self.function(*get_initial_value(self.arguments))


def deep_filter(f, *args):
    if len(args) == 1:
        a = args[0]
    else:
        a = args
    if is_iterable(a):
        return a.__class__(deep_filter(f, b) for b in a)
    return f(a)


def var(*a):
    return deep_filter(Variable, *a)


def get_initial_value(*a):
    def f(b):
        if isinstance(b, (Variable, WrappedFunction)):
            return b.initial_value
        return b

    return deep_filter(f, *a)


def absvar(*a):
    return deep_filter(lambda p: make_wrapper(d_abs)(Variable(p)), *a)


def to_float(val):
    """Unwraps Dual or array numbers into native Python floats."""
    if isinstance(val, Dual):
        return float(val.val)
    if isinstance(val, (np.ndarray, np.generic)) and val.size == 1:
        return float(val)
    return float(val) if isinstance(val, (int, float)) else val


def unjax(*a):
    """Backward-compatible alias for unwrapping solver values."""
    return deep_filter(to_float, *a)


def solve(*a):
    def f(b):
        if isinstance(b, Variable):
            return to_float(b.solve())
        elif isinstance(b, WrappedFunction):
            return to_float(b.function(*solve(b.arguments)))
        return to_float(b)

    return deep_filter(f, *a)


_results_cache = {}


def make_wrapper(f):
    @functools.wraps(f)
    def result(*args_of_first_invocation):
        var_to_index = {}

        def inner_f(*args_of_solver_invocation):
            nonlocal args_of_first_invocation

            def arg_unflatten_filter(a):
                if isinstance(a, WrappedFunction):
                    if a in _results_cache:
                        return _results_cache[a]
                    res = a.function(*(arg_unflatten_filter(b) for b in a.arguments))
                    _results_cache[a] = res
                    return res
                elif isinstance(a, Variable):
                    if (
                        a.solution is None
                        and a in var_to_index
                        and var_to_index[a] < len(args_of_solver_invocation)
                    ):
                        return args_of_solver_invocation[var_to_index[a]]
                    return a.solution_as_float_or_none()
                return a

            return f(*deep_filter(arg_unflatten_filter, args_of_first_invocation))

        relevant_args = []

        def arg_flatten_filter(a):
            if isinstance(a, WrappedFunction):
                for b in a.arguments:
                    arg_flatten_filter(b)
                return a
            elif isinstance(a, Variable):
                if a not in var_to_index:
                    var_to_index[a] = len(relevant_args)
                    if a.solution is None:
                        relevant_args.append(a)
                    else:
                        return a.solution_as_float_or_none()
            return a

        deep_filter(arg_flatten_filter, args_of_first_invocation)
        if len(relevant_args) == 0:
            return inner_f()
        return WrappedFunction(inner_f, relevant_args)

    return result


def swap_args(f):
    return lambda a, b: f(b, a)


def _add_operator_wrapper(name, wrapper):
    setattr(WrappedFunction, name, wrapper)
    setattr(Variable, name, wrapper)


for o in ["add", "sub", "mul", "truediv", "pow", "mod"]:
    _add_operator_wrapper(f"__{o}__", make_wrapper(operator.__dict__.get(o, None)))
    _add_operator_wrapper(
        f"__r{o}__", make_wrapper(swap_args(operator.__dict__.get(o, None)))
    )

for o in ["neg", "pos", "abs"]:
    _add_operator_wrapper(f"__{o}__", make_wrapper(operator.__dict__[o]))


class MagicalZero:
    @property
    def zero(self):
        return 0.0

    @zero.setter
    def zero(self, val):
        if hasattr(val, "make_zero"):
            val.make_zero()
        elif isinstance(val, (int, float, Dual)):
            if abs(float(val)) > 1e-7:
                raise SolverError(f"Inconsistent constant constraint: {val} == 0")


magic = MagicalZero()


# ==============================================================================
# Numerical LM Solver (Direct QR / SVD)
# ==============================================================================


class SimpleSolver:
    """Direct SVD Levenberg-Marquardt Solver with Marquardt Column Scaling and Backtracking Line Search."""
    lm_dampings = []
    verbose = False

    def __init__(self, tol=1e-11, max_iter=400):
        self.tol = tol
        self.max_iter = max_iter

    def step(self, J, r, lam, angle_mask=None):
        col_norms = np.linalg.norm(J, axis=0)
        col_norms = np.where(col_norms < 1e-12, 1.0, col_norms)
        J_scaled = J / col_norms
        U, s, Vt = np.linalg.svd(J_scaled, full_matrices=False)
        factors = np.where(s > 1e-12, s / (s**2 + lam), 0.0)
        delta_scaled = -Vt.T @ (factors * (U.T @ r))
        delta = delta_scaled / col_norms

        # Component-aware step clamping
        if angle_mask is not None and len(angle_mask) == len(delta):
            for i in range(len(delta)):
                limit = 90.0 if angle_mask[i] else 50.0
                if abs(delta[i]) > limit:
                    delta[i] = math.copysign(limit, delta[i])
        else:
            max_delta = np.max(np.abs(delta))
            if max_delta > 50.0:
                delta = delta * (50.0 / max_delta)
        return delta

    def solve(self, eval_fn, x0, angle_mask=None):
        x = np.array(x0, dtype=np.float64)
        lam = 1e-4
        r, J = eval_fn(x)
        cost = 0.5 * np.dot(r, r)
        if len(r) == 0 or np.max(np.abs(r)) < self.tol:
            return x, r, True

        rejection_count = 0
        for i in range(self.max_iter):
            current_lam = (
                self.lm_dampings[min(i, len(self.lm_dampings) - 1)]
                if len(self.lm_dampings) > 0
                else lam
            )
            delta = self.step(J, r, current_lam, angle_mask=angle_mask)

            step_accepted = False
            for alpha in [1.0, 0.75, 0.5, 0.25, 0.1, 0.02, 0.005]:
                x_try = x + alpha * delta
                r_try, J_try = eval_fn(x_try)
                cost_try = 0.5 * np.dot(r_try, r_try)
                if cost_try < cost or len(self.lm_dampings) > 0:
                    x, r, J, cost = x_try, r_try, J_try, cost_try
                    lam = max(lam / 5.0, 1e-15)
                    rejection_count = 0
                    step_accepted = True
                    if np.max(np.abs(r)) < self.tol:
                        return x, r, True
                    break

            if not step_accepted:
                lam = min(lam * 5.0, 1e9)
                rejection_count += 1
                if rejection_count > 35:
                    break

        return x, r, np.max(np.abs(r)) < self.tol


def solve_everything(first_variable_or_function, solve_even_if_underconstrained=True):
    """Solves all constraints and variables associated with the provided root entity."""
    global _results_cache
    _results_cache.clear()

    all_variables = []
    var_set = set()
    all_constraints = []
    const_set = set()
    settings = SolverSettings()
    settings.append_settings(solver_settings)

    def collect(item):
        if isinstance(item, Variable):
            if item.solution is None and item not in var_set:
                item.append_settings_to(settings)
                var_set.add(item)
                all_variables.append(item)
                for c in item.constraints:
                    collect(c)
        elif isinstance(item, WrappedFunction):
            if item.good_func is None or item.good_func(item):
                if item not in const_set:
                    item.append_settings_to(settings)
                    const_set.add(item)
                    all_constraints.append(item)
                    for arg in recursive_unpack(item.arguments):
                        collect(arg)

    if isinstance(first_variable_or_function, WrappedFunction):
        for arg in recursive_unpack(first_variable_or_function.arguments):
            collect(arg)
    else:
        collect(first_variable_or_function)

    if not all_variables or not all_constraints:
        return

    n_vars = len(all_variables)
    var_to_idx = {v: i for i, v in enumerate(all_variables)}
    x0 = np.array([v.initial_value for v in all_variables], dtype=np.float64)

    def eval_sketch(x_vec):
        global _results_cache
        _results_cache.clear()
        dual_vars = [Dual(x_vec[i], np.eye(n_vars)[i]) for i in range(n_vars)]

        residuals = []
        jacobian_rows = []

        for c in all_constraints:
            args_dual = [
                dual_vars[var_to_idx[a]]
                if a in var_to_idx
                else Dual(
                    float(a.solution if a.solution is not None else a.initial_value),
                    np.zeros(n_vars, dtype=np.float64),
                )
                for a in c.arguments
            ]
            res = c.function(*args_dual)
            res_items = res if is_iterable(res) else [res]

            for r in res_items:
                if isinstance(r, Dual):
                    residuals.append(r.val)
                    jacobian_rows.append(
                        r.grad
                        if r.grad is not None
                        else np.zeros(n_vars, dtype=np.float64)
                    )
                else:
                    residuals.append(float(r))
                    jacobian_rows.append(np.zeros(n_vars, dtype=np.float64))

        _results_cache.clear()
        return np.array(residuals, dtype=np.float64), np.array(
            jacobian_rows, dtype=np.float64
        )

    initial_r, _ = eval_sketch(x0)
    if len(initial_r) < n_vars and not solve_even_if_underconstrained:
        return

    solver = SimpleSolver(tol=1e-8, max_iter=250)
    solver.verbose = settings.verbose

    # Primary pass from initial guesses
    x_opt, residuals, success = solver.solve(eval_sketch, x0)
    total_error = np.linalg.norm(residuals) if len(residuals) > 0 else 0.0

    # Tolerance thresholding with sub-micron floor for extreme scale ratios
    base_tol = getattr(settings, "max_tolerance", 1e-4) or 1e-4
    effective_tol = max(base_tol, 1e-3) if total_error < 1e-3 else base_tol

    # Basin-hopping recovery: triggers only if primary pass fails or exceeds tolerance
    if not success or total_error > effective_tol:
        best_x, best_r, best_err = x_opt, residuals, total_error

    # Identify variable roles and angle mask
    angle_indices = []
    forward_indices = []
    angle_mask = np.zeros(n_vars, dtype=bool)

    for idx, v in enumerate(all_variables):
        name = (getattr(v, "name", "") or "").lower()
        if any(k in name for k in ("angle", "a1", "a2", "a3", "heading", "left", "right")):
            angle_indices.append(idx)
            angle_mask[idx] = True
        elif "forward" in name or "dist" in name:
            forward_indices.append(idx)

    solver = SimpleSolver(tol=1e-11, max_iter=300)
    solver.verbose = settings.verbose

    # Primary pass from initial guesses
    x_opt, residuals, success = solver.solve(eval_sketch, x0, angle_mask=angle_mask)
    total_error = np.linalg.norm(residuals) if len(residuals) > 0 else 0.0

    base_tol = getattr(settings, "max_tolerance", 1e-7) or 1e-7

    # Basin-hopping recovery if primary pass did not reach tight tolerance
    if not success or total_error > base_tol:
        best_x, best_r, best_err = x_opt, residuals, total_error

        candidates = []

        def get_degree_shifts(idx):
            name = (getattr(all_variables[idx], "name", "") or "").lower()
            is_deg = any(k in name for k in ("bad_a", "a1", "a2", "a3", "heading", "left", "right"))
            unit = 180.0 if (is_deg or abs(x0[idx]) > 6.5) else math.pi
            return [unit, -unit, 0.5 * unit, -0.5 * unit, 2 * unit]

        # 1. Single and dual angle flips
        for a_idx in angle_indices:
            for shift_val in get_degree_shifts(a_idx):
                x_cand = x0.copy()
                x_cand[a_idx] += shift_val
                candidates.append(x_cand)

        for i in range(len(angle_indices)):
            for j in range(i + 1, len(angle_indices)):
                a_i, a_j = angle_indices[i], angle_indices[j]
                shifts_i = get_degree_shifts(a_i)[:2]
                shifts_j = get_degree_shifts(a_j)[:2]
                for si in shifts_i:
                    for sj in shifts_j:
                        x_cand = x0.copy()
                        x_cand[a_i] += si
                        x_cand[a_j] += sj
                        candidates.append(x_cand)

        # 2. Forward variations paired with angle flips
        for f_idx in forward_indices:
            for f_val in [1.0, 5.0, 20.0, 50.0]:
                x_cand = x0.copy()
                x_cand[f_idx] = f_val
                candidates.append(x_cand)

        # 3. Deterministic random restarts
        rng = np.random.default_rng(7056)
        for scale in (10.0, 30.0, 90.0):
            for _ in range(5):
                jitter = rng.normal(0.0, scale, size=x0.shape)
                candidates.append(x0 + jitter)

        for x_cand in candidates:
            x_try, r_try, ok_try = solver.solve(eval_sketch, x_cand, angle_mask=angle_mask)
            err_try = np.linalg.norm(r_try) if len(r_try) > 0 else 0.0
            if err_try < best_err:
                best_x, best_r, best_err, success = x_try, r_try, err_try, ok_try
            if best_err <= base_tol:
                break

        x_opt, residuals, total_error = best_x, best_r, best_err

    # Final polish pass if close to solution
    if total_error < 1e-3 and total_error > 1e-10:
        solver_tight = SimpleSolver(tol=1e-11, max_iter=100)
        x_opt, residuals, _ = solver_tight.solve(eval_sketch, x_opt, angle_mask=angle_mask)
        total_error = np.linalg.norm(residuals) if len(residuals) > 0 else 0.0

    for v, val in zip(all_variables, x_opt):
        v.solution = float(val)

    _results_cache.clear()

    if total_error > base_tol:
        msg = f"Solver failed to converge with total error {total_error}."
        if len(residuals) > n_vars:
            msg += " The solver is over constrained."
        raise SolverError(msg)

# ==============================================================================
# Constraint Decorators & Built-in Primitives
# ==============================================================================


def make_constraint(f):
    @functools.wraps(f)
    def result(*args):
        wrapper = make_wrapper(f)(*args)
        if hasattr(wrapper, "make_zero"):
            return wrapper.make_zero()
        return wrapper

    return result


@make_constraint
def sum_constraint(a, b, c):
    if is_iterable(a):
        return [(a[i] + b[i] - c[i]) for i in range(len(a))]
    return a + b - c


@make_constraint
def distance_constraint(a, b, c):
    if is_iterable(a):
        diffs = [a[i] - b[i] for i in range(len(a))]
        if len(diffs) == 2:
            return d_hypot(diffs[0], diffs[1]) - c
        sum_sq = sum(d**2 for d in diffs)
        return d_sqrt(sum_sq) - c
    return d_abs(a - b) - c


@make_constraint
def coincident(a, b):
    if is_iterable(a):
        return [(a[i] - b[i]) for i in range(len(a))]
    return a - b


@make_constraint
def line_point_distance_constraint(line, point, desired_dist=0.0):
    p0, p1 = line[0], line[1]
    dx = p1[0] - p0[0]
    dy = p1[1] - p0[1]
    p_dx = point[0] - p0[0]
    p_dy = point[1] - p0[1]
    cross = dx * p_dy - dy * p_dx
    length = d_hypot(dx, dy)
    return d_abs(cross / (length + 1e-30)) - desired_dist


@make_constraint
def line_contains_point_2d_constraint(line, point):
    p0, p1 = line[0], line[1]
    dx = p1[0] - p0[0]
    dy = p1[1] - p0[1]
    return dx * (point[1] - p0[1]) - dy * (point[0] - p0[0])


@make_constraint
def line_left_distance_to_point_2d_constraint(line, point, desired_dist=0.0):
    p0, p1 = line[0], line[1]
    dx = p1[0] - p0[0]
    dy = p1[1] - p0[1]
    cross = dx * (point[1] - p0[1]) - dy * (point[0] - p0[0])
    return (cross / (d_hypot(dx, dy) + 1e-30)) - desired_dist


def line_pt_dist(line, point):
    dx = line[1][0] - line[0][0]
    dy = line[1][1] - line[0][1]
    cross = dx * (point[1] - line[0][1]) - dy * (point[0] - line[0][0])
    return cross / (make_wrapper(d_hypot)(dx, dy) + 1e-30)


@make_constraint
def parallel_2d_constraint(linea, lineb):
    da_x = linea[1][0] - linea[0][0]
    da_y = linea[1][1] - linea[0][1]
    db_x = lineb[1][0] - lineb[0][0]
    db_y = lineb[1][1] - lineb[0][1]
    return da_x * db_y - da_y * db_x


@make_constraint
def angle_2d_constraint(linea, lineb, angle):
    da_x = linea[1][0] - linea[0][0]
    da_y = linea[1][1] - linea[0][1]
    db_x = lineb[1][0] - lineb[0][0]
    db_y = lineb[1][1] - lineb[0][1]
    s = math.sin(angle)
    c = math.cos(angle)
    rot_x = da_x * c - da_y * s
    rot_y = da_x * s + da_y * c
    return rot_x * db_y - rot_y * db_x
