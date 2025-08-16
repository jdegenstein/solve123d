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

import jax
import jax.numpy as jnp
import jaxopt

jax.config.update("jax_enable_x64", True)


# TODO: see if the parameter unpacking-repacking could be replaced easily with pytrees


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


class Variable:
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


# TODO: refactor common functionality among different substitution methods
def var(*args):
    """Helper method. Converts each value in the arguments to Variable(value)
    Intended usage: var(1,2) is equivalent to (Variable(1), Variable(2))
     var([1,2]) is equivalent to ([Variable(1), Variable(2)])
     and so on.
    """
    if len(args) == 1:
        a = args[0]
    else:
        a = args
    if isinstance(a, collections.abc.Sequence):
        return a.__class__(var(b) for b in a)
    else:
        return Variable(a)

def unjax(*args):
    if len(args) == 1:
        a = args[0]
    else:
        a = args
    if isinstance(a, collections.abc.Sequence):
        return a.__class__(unjax(b) for b in a)
    elif isinstance(a, jax.Array) and a.size==1:
        return float(a)
    else:
        return a

def solve(*args):
    """Helper method. Converts each Variable in the argument to the solution, similarly to var()
    Intended usage: solve(a, b) is equivalent to (a.s, b.s)
    Note that basic arithmetics is also supported, solve(a+1, b) would also work as expected
    """
    if len(args) == 1:
        a = args[0]
    else:
        a = args
    if isinstance(a, collections.abc.Sequence):
        return a.__class__(solve(b) for b in a)
    elif isinstance(a, Variable):
        return a.solve()
    elif isinstance(a, WrappedFunction):
        return a.function(*solve(a.arguments))
    else:
        return a


def make_wrapper(f):
    """
    High order function that creates a function that will output a wrapper object around
    f' where f' is f with the non-variable values bound to it and variable values as arguments
    :f: Function to be wrapped
    """

    @functools.wraps(f)
    def result(*args_of_first_invocation):
        indices=[]
        def inner_f(*args_of_solver_invocation):
            nonlocal args_of_first_invocation
            def process_argument(i, a):
                if isinstance(a, WrappedFunction):
                    return a.function(*args_of_solver_invocation[indices[i]])
                elif isinstance(a, Variable):
                    if a.solution is None:
                        return args_of_solver_invocation[indices[i]][0]
                    else:
                        return a.solution_as_float_or_none()
                else:
                    return a

            return f(
                *[
                    process_argument(i, a)
                    for i, a in enumerate(args_of_first_invocation)
                ]
            )
        relevant_args = []
        args_for_bypass = []
        # TODO: handle arguments that are tuples of tuples etc
        for i, a in enumerate(args_of_first_invocation):
            indices.append(len(relevant_args))
            if isinstance(a, WrappedFunction):                
                relevant_args.append(a.arguments)
            elif isinstance(a, Variable):
                if a.solution is None:
                    relevant_args.append([a])                    
                else:
                    args_for_bypass.append(a.solution_as_float_or_none())
            else:                
                args_for_bypass.append(a)
        # If none of arguments will be substituted, evaluate wrapped function here and now instead of delaying evaluation
        if len(relevant_args) == 0:
            return f(*args_for_bypass)
        return WrappedFunction(inner_f, relevant_args)

    return result


class WrappedFunction:
    """Represents a function used as a geometric constraint"""

    def make_zero(self):
        """Creates a constraint that the wrapped function equates to zero"""
        for a in recursive_unpack(self.arguments):
            if isinstance(a, Variable):
                a.constraints.add(self)
        return self

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
def _recursive_substitute(args, state, indices):
    """Makes a deep coopy but substitutes Variables with values from state"""

    def recursive_substitute_inner(arg):
        def process(a):
            if isinstance(a, Variable):
                return state[indices[a]]
            elif is_iterable(a):
                return recursive_substitute_inner(a)
            else:
                return a

        # TODO: refactor to use same path for tuples and arrays
        if isinstance(arg, tuple):
            return tuple(process(a) for a in arg)
        else:
            result = copy.copy(arg)
            for i, a in enumerate(arg):
                if isinstance(a, Variable):
                    result[i] = state[indices[a]]
                elif is_iterable(a):
                    result[i] = recursive_substitute_inner(a)
            return result

    return recursive_substitute_inner(args)


# def recursive_set_solution(args, state, indices):
#     def recursive_set_solution_inner(arg):
#         for i, a in enumerate(arg):
#             if isinstance(a, Variable):
#                 a.solution = state[indices[a]]
#             elif is_iterable(a):
#                 recursive_set_solution_inner(a)

#     recursive_set_solution_inner(args)


def solve_everything(first_variable: Variable):
    """Solves all constraints and variables associated with the provided argument.
    :param first_variable: Variable the variable to use as starting point for traversal.
    """
    print("Constraint solver invoked")
    all_variables = set()
    all_constraints = set()

    def recurse_constraint(c):
        if c not in all_constraints:
            all_constraints.add(c)
            for a in c.arguments:
                recurse_variable(a)

    def recurse_variable(a):
        if is_iterable(a):
            for v in a:
                recurse_variable(v)
        elif isinstance(a, Variable):
            if a not in all_variables:
                all_variables.add(a)
                for c in a.constraints:
                    recurse_constraint(c)

    recurse_variable(first_variable)

    variable_indices = dict()
    cur_index = 0
    for v in all_variables:
        print(v.initial_value)
        variable_indices[v] = cur_index
        cur_index += 1

    params = jnp.array([v.initial_value for v in all_variables], dtype=jnp.float64)

    residuals_count = 0

    # Make one function to solve, out of all known constraints
    def all_constraints_function(input_state):
        nonlocal residuals_count
        # Todo: create jnp.array directly
        result = []
        for c in all_constraints:
            args = c.arguments
            args_copy = _recursive_substitute(args, input_state, variable_indices)
            # r=c.function(*(input_state[variable_indices[v]] for v in c.arguments))
            r = c.function(*args_copy)
            if is_iterable(r):
                result += r
            else:
                result.append(r)
        residuals_count = len(result)
        jax_result = jnp.array(result)
        return jax_result

    fast_residual = jax.jit(all_constraints_function)
    # jac = jax.jacfwd(all_constraints_function)
    # fast_jac=jax.jit(jac)
    # TODO: diagnostic messages (e.g. if under or over constrained, if fails to converge).
    # solver = jaxopt.LevenbergMarquardt(residual_fun=all_constraints_function)
    solver = jaxopt.LevenbergMarquardt(
        residual_fun=fast_residual, maxiter=30, tol=1e-15, gtol=1e-15
    )

    jit_solver = jax.jit(solver.run)

    # result_params, state = solver.run(params)
    result_params, _ = jit_solver(params)

    for v in all_variables:
        # If we want to deal with jax.Array values
        # v.solution = result_params[variable_indices[v]]
        v.solution = float(result_params[variable_indices[v]])

    if residuals_count < len(params):
        print(
            f"Under constrained: {len(params)} degrees of freedom but only {residuals_count} constraints"
        )

    if residuals_count > len(params):
        print(
            f"Over or redundantly constrained: {len(params)} degrees of freedom and {residuals_count} constraints"
        )

    print(f"initial params = {params} , result_params={result_params}")
    # print(fast_residual(params))
    # print(fast_jac(params))


def make_constraint(f):
    """Decorator that makes a constraint out of function f
    :f: function returning a tuple or array of errors (residuals).
    """

    @functools.wraps(f)
    def result(*args):
        # TODO: use make_wrapper once it supports functions whose arguments are tuples containing other wrapped functions etc
        #wrapper_f=make_wrapper(f)
        #wrapper=wrapper_f(*args)
        #return wrapper.make_zero()
        return WrappedFunction(f, [*args]).make_zero()

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
    direction = (line[1][0]-line[0][0], line[1][1]-line[0][1])
    dir_to_pt = (point[0]-line[0][0], point[1]-line[0][1])
    return (
        direction[0] * dir_to_pt[1] - direction[1] * dir_to_pt[0]
    ) / make_wrapper(jnp.sqrt)(direction[0]**2 + direction[1]**2)


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
