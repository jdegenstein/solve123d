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

import collections
import copy
import functools
import operator
import typing

import jax
import jax.numpy as jnp
import jaxopt

jax.config.update("jax_enable_x64", True)


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


# Note: opportunistic is not a solver setting, but a constraints setting.
opportunistic = False


def set_opportunistic(v=True):
    global opportunistic
    opportunistic = v


# TODO: see if the parameter unpacking-repacking could be replaced easily with pytrees
# Usage of no coverage pragmas: only for "pass" statements due to defensive programming when traversing argument trees (which currently contain only Variable instances, but that may change)


def is_iterable(a):  # TODO: support things that aren't sequences?
    """Used to decide what values to iterate"""
    return isinstance(a, collections.abc.Sequence)


def recursive_unpack(list_or_item):
    """Depth-first iterates over all leaves in list_or_item"""
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
    """A value that the solver will solve for"""

    def __init__(self, initial_value=0.0):
        self.initial_value = initial_value
        self.constraints = set()
        self.solution = None

    def solve(self):
        """Returns the solution to the system of equations connected to the variable
        Runs solver on the first call (all other variables end up solved afterwards)
        """
        if self.solution is None:
            solve_everything(self)
        if isinstance(self.solution, jax.Array) and self.solution.size == 1:
            return float(self.solution)
        return self.solution

    def solution_as_float_or_none(self):
        """If already solved, gives solution, else returns None"""
        if self.solution is None:
            return None
        if isinstance(self.solution, jax.Array) and self.solution.size == 1:
            return float(self.solution)
        return self.solution

    @property
    def s(self):
        """Abbreviation for solve()
        Returns the solution to the system of equations connected to the variable
        Runs solver on the first call (all other variables end up solved afterwards)"""
        return self.solve()


class WrappedFunction(SolverEntity):
    """Represents a function used as a geometric constraint"""

    cached_initial_value = None
    good_func = None

    def make_zero(self):
        """Creates a constraint that the wrapped function equates to zero"""
        variables_to_solve = set()
        for a in recursive_unpack(self.arguments):
            if isinstance(a, Variable):
                a.constraints.add(self)
                variables_to_solve.add(a)
            else:  # pragma: no cover
                pass
        if opportunistic:  # Attempt to solve as soon as possible
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
        """Computes the function on initial values of variables passed to it"""
        if self.cached_initial_value is None:
            self.cached_initial_value = self.function(
                *get_initial_value(self.arguments)
            )
        return self.cached_initial_value

    # arguments is a list of variables that are parameters to the function
    def __init__(self, function_or_variable, arguments=None):
        # Trivial case of wrapping a Variable in a Function
        if isinstance(function_or_variable, Variable):
            self.arguments = [function_or_variable]

            def passthrough(v):
                return v

            self.function = passthrough
        else:
            self.function = function_or_variable
            self.arguments = arguments


def deep_filter(f, *args):
    """Filters *args through f, like a deep copy (sequences are also filtered)"""
    if len(args) == 1:
        a = args[0]
    else:
        a = args
    if isinstance(a, collections.abc.Sequence):
        return a.__class__(deep_filter(f, b) for b in a)
    else:
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
    return deep_filter(lambda p: make_wrapper(jnp.abs)(Variable(p)), *a)


def unjax(*a):
    def f(a):
        if isinstance(a, jax.Array) and a.size == 1:
            return float(a)
        else:
            return a

    return deep_filter(f, *a)


def solve(*a):
    def f(a):
        if isinstance(a, Variable):
            return a.solve()
        elif isinstance(a, WrappedFunction):
            return a.function(*solve(a.arguments))
        else:
            return a

    return deep_filter(f, *a)


# TODO: figure type hints to make this work correctly with linters
# def make_deep_filter(f: typing.Callable[[typing.Any], typing.Any]) -> typing.Callable[..., typing.Any] :
#     return lambda *a: deep_filter(f, *a)

# @make_deep_filter
# def var(a):
#     return Variable(a)

# @make_deep_filter
# def unjax(a):
#     if isinstance(a, jax.Array) and a.size == 1:
#         return float(a)
#     else:
#         return a

# @make_deep_filter
# def solve(a):
#     if isinstance(a, Variable):
#         return a.solve()
#     elif isinstance(a, WrappedFunction):
#         return a.function(*solve(a.arguments))
#     else:
#         return a

# TODO: improve results_cache implementation
_results_cache = {}


def make_wrapper(f):
    """
    High order function that creates a function that will output a wrapper object around
    f' where f' is f with the non-variable values bound to it and variable values as arguments
    :f: Function to be wrapped
    """

    @functools.wraps(f)
    def result(*args_of_first_invocation):
        var_to_index = {}

        def inner_f(*args_of_solver_invocation):
            nonlocal args_of_first_invocation

            def arg_unflatten_filter(a):
                nonlocal args_of_solver_invocation, args_of_first_invocation
                if isinstance(a, WrappedFunction):
                    if a in _results_cache:
                        result = _results_cache[a]
                    else:
                        result = a.function(
                            *(arg_unflatten_filter(b) for b in a.arguments)
                        )
                        _results_cache[a] = result
                elif isinstance(a, Variable):
                    if a.solution is None:
                        # result = args_of_solver_invocation[indices[index]][0]
                        result = args_of_solver_invocation[var_to_index[a]]
                    else:
                        result = a.solution_as_float_or_none()
                else:
                    result = a
                return result

            return f(*deep_filter(arg_unflatten_filter, args_of_first_invocation))

        relevant_args = []

        # Flattens and deduplicates arguments
        def arg_flatten_filter(a):
            if isinstance(a, WrappedFunction):
                # relevant_args.append(a.arguments)
                for b in a.arguments:  # shallowly append child arguments
                    arg_flatten_filter(b)
                return a
            elif isinstance(a, Variable):
                if a not in var_to_index:
                    var_to_index[a] = len(relevant_args)
                    if a.solution is None:
                        relevant_args.append(a)
                    else:
                        return a.solution_as_float_or_none()
            else:
                return a

        deep_filter(arg_flatten_filter, args_of_first_invocation)
        # If none of arguments will be substituted, evaluate wrapped function here and now instead of delaying evaluation
        # TODO: make it bypass the inner_f as well?
        if len(relevant_args) == 0:
            return inner_f()
            # return f(*bypass)
        return WrappedFunction(inner_f, relevant_args)

    return result


def swap_args(f):
    """Higher order function that converts f(a,b) into f(b,a)"""

    def result(a, b):
        return f(b, a)

    return result


def _add_operator_wrapper(name, wrapper):
    setattr(WrappedFunction, name, wrapper)
    setattr(Variable, name, wrapper)


_binary_operators_to_wrap = ["add", "sub", "mul", "truediv", "pow"]
for o in _binary_operators_to_wrap:
    _add_operator_wrapper(f"__{o}__", make_wrapper(operator.__dict__[o]))
    _add_operator_wrapper(f"__r{o}__", make_wrapper(swap_args(operator.__dict__[o])))

_unary_operators_to_wrap = ["neg", "pos"]
for o in _unary_operators_to_wrap:
    _add_operator_wrapper(f"__{o}__", make_wrapper(operator.__dict__[o]))


class MagicalZero:
    """Used to allow creation of constraints via `magic.zero=a+2*b` syntax"""

    @property
    def zero(self):
        """Used to allow creation of constraints via `magic.zero=a+2*b` syntax
        Getter simply returns 0.0
        """
        return 0.0

    @zero.setter
    def zero(self, val):
        """Used to allow creation of constraints via `magic.zero=a+2*b` syntax"""
        val.make_zero()


magic = MagicalZero()


# Makes a deep copy but replacing variable instances.
# TODO: replace with deep_filter
def _recursive_substitute(args, state, indices):
    """Makes a deep copy but substitutes Variables with values from state"""

    def f(a):
        if isinstance(a, Variable):
            return state[indices[a]]
        else:  # pragma: no cover
            return a

    return deep_filter(f, args)


class _SimpleFunctionCache:
    def __init__(self, f):
        self.f = f
        self.cached_for = None
        self.cached_result = None

    def __call__(self, a):
        if a is not self.cached_for:
            self.cached_result = self.f(a)
            self.cached_for = a
        return self.cached_result

debug_nan=True

# Solver that works even if a is singular (by using QR decomposition)
def solve_via_qr(a, b):
    eps=1E-50
    q,r=jnp.linalg.qr(a)
    p=jnp.dot(q.T, b)
    
    def replace_small_with_one(a):
        return jax.lax.cond(jnp.abs(a)>eps, lambda a: 0.0, lambda a: 1.0, a)
    r_sanitizer=jax.vmap(replace_small_with_one)(jnp.diag(r))
    return jax.scipy.linalg.solve_triangular(r+jnp.diag(r_sanitizer), p)

class SimpleSolver:
    def __init__(self, residual_f, jacobian_f, tolerance, max_iter=100):
        self.residual_f = _SimpleFunctionCache(residual_f)
        self.jacobian_f = _SimpleFunctionCache(jacobian_f)
        self.norm_f = _SimpleFunctionCache(jnp.linalg.norm)
        self.tolerance = tolerance
        self.max_iter = max_iter
        self.total_err = 1e20
        #self.lm_dampings = [0.5, 0.5, 0.5, 0.25, 0.125, 0]
        self.lm_dampings=[]

    def update(self, state, damping=0):
        print(state)
        r = self.residual_f(state)
        self.total_err = self.norm_f(r)
        print(f"Solver error: {self.total_err}")
        if jnp.isnan(self.total_err):
            raise SolverError("NAN when solving!")
        if self.total_err < self.tolerance:
            return state
        j = self.jacobian_f(state)
        if debug_nan and jnp.any(jnp.isnan(j)):
            raise SolverError("NAN when solving!")
        jtj = j.transpose() @ j
        jte = j.transpose() @ r
        if debug_nan and jnp.any(jnp.isnan(jte)):
            raise SolverError("NAN when solving!")
        if damping > 0:
            damped_jtj = jtj + damping * jnp.diag(jnp.diag(jtj))
        else:
            damped_jtj = jtj
        if debug_nan and jnp.any(jnp.isnan(damped_jtj)):
            raise SolverError("NAN when solving!")
        #delta = jax.numpy.linalg.solve(damped_jtj, jte)
        delta=solve_via_qr(damped_jtj, jte)

        if debug_nan and jnp.any(jnp.isnan(delta)):
            raise SolverError("NAN when solving!")

        result = state - delta
        delta_scale = 0.5
        while self.norm_f(self.residual_f(result)) > self.total_err:
            result = state - delta_scale * delta
            delta_scale = delta_scale * 0.5
        return result

    def run(self, state):
        for i in range(0, self.max_iter):
            if len(self.lm_dampings)>0:
                damping = self.lm_dampings[min(i, len(self.lm_dampings) - 1)]
            else:
                damping=0
            new_state = self.update(state, damping)
            if new_state is state:
                break
            state = new_state
        print(f"Solved in {i} iterations")
        return state


def solve_everything(
    first_variable_or_function: Variable | WrappedFunction,
    solve_even_if_underconstrained=True,
):
    """Solves all constraints and variables associated with the provided argument.
    :param first_variable: the variable to use as starting point for traversal.
    """
    use_jit = True
    use_custom_solver = True

    all_variables = set()
    all_constraints = set()

    settings = SolverSettings()
    settings.append_settings(solver_settings)

    def filter_item(a):
        if isinstance(a, Variable):
            if a.solution is None:
                if a not in all_variables:
                    a.append_settings_to(settings)
                    all_variables.add(a)
                    deep_filter(filter_item, *a.constraints)
        elif isinstance(a, WrappedFunction):
            if a.good_func is None or a.good_func(a):
                if a not in all_constraints:
                    a.append_settings_to(settings)
                    all_constraints.add(a)
                    deep_filter(filter_item, a.arguments)
            else:
                print("Constraint elision")
        else:  # pragma: no cover
            pass

    # Do not treat WrappedFunction itself as an ==0 constraint
    if isinstance(first_variable_or_function, WrappedFunction):
        deep_filter(filter_item, first_variable_or_function.arguments)
    else:
        deep_filter(filter_item, first_variable_or_function)

    verbose = settings.verbose

    if verbose:
        print("Constraint solver invoked")

    # slightly more efficient but duplicative code
    # def recurse_constraint(c):
    #     if c not in all_constraints:
    #         all_constraints.add(c)
    #         for a in c.arguments:
    #             recurse_variable(a)
    # def recurse_variable(a):
    #     if is_iterable(a):
    #         for v in a:
    #             recurse_variable(v)
    #     elif isinstance(a, Variable):
    #         if a not in all_variables:
    #             all_variables.add(a)
    #             for c in a.constraints:
    #                 recurse_constraint(c)
    #     else:  # pragma: no cover
    #         pass
    # recurse_variable(first_variable)

    if len(all_constraints) == 0:  # pragma: no cover
        print("No constraints")
        return
    if len(all_variables) == 0:  # pragma: no cover
        print("No variables")
        return

    variable_indices = {}
    cur_index = 0
    for v in all_variables:
        variable_indices[v] = cur_index
        cur_index += 1

    params = jnp.array([v.initial_value for v in all_variables], dtype=jnp.float64)

    residuals_count = 0

    # Make one function to solve, out of all known constraints
    def all_constraints_function(input_state):
        nonlocal residuals_count
        global _results_cache
        if verbose:
            print("running original all_constraints_function")
        _results_cache = {}
        # Todo: create jnp.array directly
        result = []
        for c in all_constraints:
            args = c.arguments
            # if verbose:
            #    print(f"Constraint {c} with arguments {args}")
            args_copy = _recursive_substitute(args, input_state, variable_indices)
            # r=c.function(*(input_state[variable_indices[v]] for v in c.arguments))
            r = c.function(*args_copy)
            # print(r)
            if is_iterable(r):
                result += r
            else:
                result.append(r)
        residuals_count = len(result)
        jax_result = jnp.array(result)
        _results_cache = {}
        return jax_result

    if use_custom_solver:
        fast_residual = jax.jit(all_constraints_function)
        jac = jax.jacfwd(all_constraints_function)
        fast_jac = jax.jit(jac)
        solver = SimpleSolver(fast_residual, fast_jac, 1e-12, 100)
        residual=solver.residual_f(params)

        if residuals_count < len(params):
            if not solve_even_if_underconstrained:
                print("Not solving underconstrained")
                return

        result_params = solver.run(params)
        residuals = solver.residual_f.cached_result
        print(f"Residuals: {residuals}")
    else:
        if use_jit:
            if verbose:
                print("Starting jit compile")
            fast_residual = jax.jit(all_constraints_function)
            jac = jax.jacfwd(all_constraints_function)
            fast_jac = jax.jit(jac)
            # TODO: diagnostic messages (e.g. if under or over constrained, if fails to converge).
            # solver = jaxopt.LevenbergMarquardt(residual_fun=all_constraints_function)
            solver = jaxopt.LevenbergMarquardt(
                residual_fun=fast_residual, maxiter=30, tol=1e-15, gtol=1e-15, jit=True
            )
            # solver=jaxopt.GaussNewton(residual_fun=all_constraints_function, tol=1E-15, verbose=True)

            solver_run = jax.jit(solver.run)
            # solver_run=solver.run
            if verbose:
                print("experiment: forcing compilation")
                residuals = fast_residual(params)
                jac_result = fast_jac(params)
                print(f"experiment: done forcing compilation, jac={jac_result}")

            if verbose:
                print("Running solver")
            result_params, state = solver_run(params)
            if verbose:
                print("Computing final residuals")
            residuals = fast_residual(result_params)
            print(f"Residuals: {residuals}")
        else:  # pragma: no cover
            solver = jaxopt.LevenbergMarquardt(
                residual_fun=all_constraints_function, maxiter=30, tol=1e-15, gtol=1e-15
            )
            result_params, _ = solver.run(params)
            residuals = all_constraints_function(result_params)

    if residuals_count < len(params):
        if not solve_even_if_underconstrained:
            print("Not solving underconstrained")
            return
        print(
            f"Under constrained: {len(params)} degrees of freedom but only {residuals_count} constraints"
        )

    for v in all_variables:
        # If we want to deal with jax.Array values
        # v.solution = result_params[variable_indices[v]]
        v.solution = float(result_params[variable_indices[v]])
        if verbose:
            print(f"Var {v.name}: solution:{v.solution}")

    if residuals_count > len(params):
        print(
            f"Over or redundantly constrained: {len(params)} degrees of freedom and {residuals_count} constraints"
        )

    if verbose:
        print(f"initial params = {params} , result_params={result_params}")

    # Raise at the end so that all verbose prints complete
    total_error = jnp.linalg.norm(residuals)
    if settings.max_tolerance is not None:
        if total_error > settings.max_tolerance:
            error_message = (
                f"Solver failed to converge with total error {total_error}. "
            )
            if residuals_count > len(params):
                error_message += "The solver is over constrained."
            error_message += " You may need to provide initial guesses and/or remove conflicting constraints."
            raise SolverError(error_message)
    # print(fast_residual(params))
    # print(fast_jac(params))


def make_constraint(f):
    """Decorator that makes a constraint out of function f
    :f: function returning a tuple or array of errors (residuals).
    """

    @functools.wraps(f)
    def result(*args):
        wrapper_f = make_wrapper(f)
        wrapper = wrapper_f(*args)
        return wrapper.make_zero()

    return result


@make_constraint
def sum_constraint(a, b, c):
    """Simple test constraint, enforces that a+b=c"""
    if is_iterable(a):
        return jnp.array(
            [(a[i] + b[i] - c[i]) for i, v in enumerate(a)], dtype=jnp.float64
        )
    else:
        return a + b - c


@make_constraint
def distance_constraint(a, b, c):
    """Constrains N-dimensional distance between a and b to be equal to c.
    :a: n-dimensional vector
    :b: n-dimensional vector
    :c: distance
    """
    return (
        jnp.linalg.norm(
            jnp.subtract(
                jnp.array(a, dtype=jnp.float64), jnp.array(b, dtype=jnp.float64)
            )
        )
        - c
    )


@make_constraint
def coincident(a, b):
    """Coincident constraint, requires that a=b .
    It is faster to just use the same variable for a and b instead."""
    if is_iterable(a):
        return [(a[i] - b[i]) for i, v in enumerate(a)]
    else:
        return a - b


@make_constraint
def line_point_distance_constraint(line, point, desired_dist=0.0):
    """N-d line to point distance constraint (point is constrained to the surface of a cylinder around the line)
    In 2D construction, use line_left_distance_to_point_2d_constraint as it provides signed distance instead.
    :line: Tuple of two points defining a line
    :point: The point being constrained
    :desired_dist: Desired distance to the line
    """
    p0 = jnp.array(line[0], dtype=jnp.float64)
    p1 = jnp.array(line[1], dtype=jnp.float64)
    pt = jnp.array(point)
    direction = jnp.subtract(p1, p0)
    ptdelta = pt - p0
    dirnorm2 = jnp.dot(direction, direction)
    return (
        jnp.linalg.norm(
            jnp.subtract(ptdelta, direction * (jnp.dot(direction, ptdelta) / dirnorm2))
        )
        - desired_dist
    )


@make_constraint
def line_contains_point_2d_constraint(line, point):
    """Constrains 2d point to the 2d line (better for 2D construction than using generic N-dimensional verison)
    :line: Tuple of two points defining a line
    :point: the point being constrained

    """
    p0 = jnp.array(line[0], dtype=jnp.float64)
    p1 = jnp.array(line[1], dtype=jnp.float64)
    pt = jnp.array(point, dtype=jnp.float64)
    direction = jnp.subtract(p1, p0)
    dir_to_pt = jnp.subtract(pt, p0)
    return direction[0] * dir_to_pt[1] - direction[1] * dir_to_pt[0]


@make_constraint
def line_left_distance_to_point_2d_constraint(line, point, desired_dist=0.0):
    """Constrains signed distance between 2D line and 2D point
    Looking from the first point of the line to the second point, the distance is given in the left direction.
    For example if the line is ((0,0), (1,0)) then the distance is equal to point[1]
    :line: Tuple of two points defining a line
    :point: the point being constrained
    """
    p0 = jnp.array(line[0], dtype=jnp.float64)
    p1 = jnp.array(line[1], dtype=jnp.float64)
    pt = jnp.array(point, dtype=jnp.float64)
    direction = jnp.subtract(p1, p0)
    dir_to_pt = jnp.subtract(pt, p0)
    return (
        direction[0] * dir_to_pt[1] - direction[1] * dir_to_pt[0]
    ) / jnp.linalg.norm(direction) - desired_dist


def line_pt_dist(line, point):
    direction = (line[1][0] - line[0][0], line[1][1] - line[0][1])
    dir_to_pt = (point[0] - line[0][0], point[1] - line[0][1])
    return (direction[0] * dir_to_pt[1] - direction[1] * dir_to_pt[0]) / make_wrapper(
        jnp.sqrt
    )(direction[0] ** 2 + direction[1] ** 2)


@make_constraint
def parallel_2d_constraint(linea, lineb):
    """Makes two lines parallel, in 2D
    :linea: First line, a tuple of two points
    :lineb: Second line, a tuple of two points
    """
    a0 = jnp.array(linea[0], dtype=jnp.float64)
    a1 = jnp.array(linea[1], dtype=jnp.float64)

    b0 = jnp.array(lineb[0], dtype=jnp.float64)
    b1 = jnp.array(lineb[1], dtype=jnp.float64)

    da = jnp.subtract(a1, a0)
    db = jnp.subtract(b1, b0)
    return da[0] * db[1] - da[1] * db[0]


@make_constraint
def angle_2d_constraint(linea, lineb, angle):
    """Makes two lines be at an angle, in 2D
    :linea: First line, a tuple of two points
    :lineb: Second line, a tuple of two points
    :angle: angle, in radians, positive counter-clockwise
    """
    a0 = jnp.array(linea[0], dtype=jnp.float64)
    a1 = jnp.array(linea[1], dtype=jnp.float64)

    b0 = jnp.array(lineb[0], dtype=jnp.float64)
    b1 = jnp.array(lineb[1], dtype=jnp.float64)

    da = jnp.subtract(a1, a0)
    db = jnp.subtract(b1, b0)
    s = jnp.sin(angle)
    c = jnp.cos(angle)
    da_rot = jnp.array([da[0] * c - da[1] * s, da[0] * s + da[1] * c])

    return da_rot[0] * db[1] - da_rot[1] * db[0]
