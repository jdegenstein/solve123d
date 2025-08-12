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
import jax
import jaxopt
import jax.numpy as jnp
import weakref  # todo: use weakrefs in constraints
import copy
import operator
import functools

jax.config.update("jax_enable_x64", True)


# TODO: see if the parameter unpacking-repacking could be replaced easily with pytrees


def is_iterable(a):  # TODO: support things that aren't sequences?
    return isinstance(a, collections.abc.Sequence)


def recursive_unpack(list_or_item):
    if is_iterable(list_or_item):
        for i in list_or_item:
            yield from recursive_unpack(i)
    else:
        yield list_or_item


class Variable:
    def __init__(self, initial_value=0.0):
        self.initial_value = initial_value
        self.constraints = set()
        self.solution = None

    def solve(self):
        if self.solution is None:
            solve_everything(self)
        if isinstance(self.solution, jax.Array) and self.solution.size == 1:
            return float(self.solution)
        return self.solution

    def solution_as_float_or_none(self):
        if self.solution is None:
            return None
        if isinstance(self.solution, jax.Array) and self.solution.size == 1:
            return float(self.solution)
        return self.solution

    @property
    def s(self):
        return self.solve()


# TODO: refactor common functionality among different substitution methods
def var(*args):
    if len(args) == 1:
        a = args[0]
    else:
        a = args
    if isinstance(a, collections.abc.Sequence):
        return a.__class__(var(b) for b in a)
    else:
        return Variable(a)


def solve(*args):
    if len(args) == 1:
        a = args[0]
    else:
        a = args
    if isinstance(a, collections.abc.Sequence):
        return a.__class__(solve(b) for b in a)
    elif isinstance(a, Variable):
        return a.solve()
    elif isinstance(a, WrappedFunction):
        return a.function(solve(*a.arguments))
    else:
        return a


# Creates a function that will output a wrapper object around f' where f' is f with the non-variable values bound to it and variable values as arguments
def make_wrapper(f):
    @functools.wraps(f)
    def result(*args_of_first_invocation):
        def inner_f(*args_of_solver_invocation):
            def eval(i, a):
                if isinstance(a, WrappedFunction):
                    return a.function(*args_of_solver_invocation[i])
                elif isinstance(a, Variable) and a.solution is None:
                    return args_of_solver_invocation[i][0]
                else:
                    return a

            return f(*[eval(i, a) for i, a in enumerate(args_of_first_invocation)])

        relevant_args = []
        args_for_bypass = []
        for i, a in enumerate(args_of_first_invocation):
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
    def make_zero(self):
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
    def result(a, b):
        return f(b, a)

    return result


def add_operator_wrapper(name, wrapper):
    setattr(WrappedFunction, name, wrapper)
    setattr(Variable, name, wrapper)


binary_operators_to_wrap = ["add", "sub", "mul", "truediv", "pow"]
for o in binary_operators_to_wrap:
    add_operator_wrapper(f"__{o}__", make_wrapper(operator.__dict__[o]))
    add_operator_wrapper(f"__r{o}__", make_wrapper(swap_args(operator.__dict__[o])))

unary_operators_to_wrap = ["neg", "pos"]
for o in unary_operators_to_wrap:
    add_operator_wrapper(f"__{o}__", make_wrapper(operator.__dict__[o]))


class MagicalZero:
    @property
    def zero(self):
        return 0.0

    @zero.setter
    def zero(self, val):
        val.make_zero()


magic = MagicalZero()


# Makes a deep copy but replacing variable instances.
# TODO: refactor to use same path for tuples and arrays
def recursive_substitute(args, state, indices):
    def recursive_substitute_inner(arg):
        def process(a):
            if isinstance(a, Variable):
                return state[indices[a]]
            elif is_iterable(a):
                return recursive_substitute_inner(a)
            else:
                return a

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


def solve_everything(first_variable):
    print("Constraint solver invoked")
    all_variables = set()
    all_constraints = set()

    def recurse_constraint(c):
        if c not in all_constraints:
            all_constraints.add(c)
            for a in c.arguments:
                recurse_variable(a)

    def recurse_variable(var):
        if is_iterable(var):
            for v in var:
                recurse_variable(v)
        elif isinstance(var, Variable):
            if var not in all_variables:
                all_variables.add(var)
                for c in var.constraints:
                    recurse_constraint(c)

    recurse_variable(first_variable)

    variable_indices = dict()
    cur_index = 0
    for v in all_variables:
        print(v.initial_value)
        variable_indices[v] = cur_index
        cur_index += 1

    params = jnp.array([v.initial_value for v in all_variables], dtype=jnp.float64)

    # Make one function to solve, out of all known constraints
    def all_constraints_function(input_state):
        # Todo: create jnp.array directly
        result = []
        for c in all_constraints:
            args = c.arguments
            args_copy = recursive_substitute(args, input_state, variable_indices)
            # r=c.function(*(input_state[variable_indices[v]] for v in c.arguments))
            r = c.function(*args_copy)
            if is_iterable(r):
                result += r
            else:
                result.append(r)
        return jnp.array(result)

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
    result_params, state = jit_solver(params)

    for v in all_variables:
        v.solution = result_params[variable_indices[v]]

    print(f"initial params = {params} , result_params={result_params}")
    # print(fast_residual(params))
    # print(fast_jac(params))


def make_constraint(f):
    @functools.wraps(f)
    def result(*args):
        return WrappedFunction(f, [*args]).make_zero()

    return result


@make_constraint
def sum_constraint(a, b, c):
    if is_iterable(a):
        return jnp.array(
            [(a[i] + b[i] - c[i]) for i, v in enumerate(a)], dtype=jnp.float64
        )
    else:
        return a + b - c


@make_constraint
def distance_constraint(a, b, c):
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
    if is_iterable(a):
        return [(a[i] - b[i]) for i, v in enumerate(a)]
    else:
        return a - b


@make_constraint
def line_point_distance_constraint(line, point, desired_dist=0.0):
    p0 = jnp.array(line[0], dtype=jnp.float64)
    p1 = jnp.array(line[1], dtype=jnp.float64)
    pt = jnp.array(point)
    dir = jnp.subtract(p1, p0)
    ptdelta = pt - p0
    dirnorm2 = jnp.dot(dir, dir)
    return (
        jnp.linalg.norm(jnp.subtract(ptdelta, dir * (jnp.dot(dir, ptdelta) / dirnorm2)))
        - desired_dist
    )


@make_constraint
def line_contains_point_2d_constraint(line, point):
    p0 = jnp.array(line[0], dtype=jnp.float64)
    p1 = jnp.array(line[1], dtype=jnp.float64)
    pt = jnp.array(point, dtype=jnp.float64)
    direction = jnp.subtract(p1, p0)
    dir_to_pt = jnp.subtract(pt, p0)
    return direction[0] * dir_to_pt[1] - direction[1] * dir_to_pt[0]


@make_constraint
def line_left_distance_to_point_2d_constraint(line, point, desired_dist=0.0):
    p0 = jnp.array(line[0], dtype=jnp.float64)
    p1 = jnp.array(line[1], dtype=jnp.float64)
    pt = jnp.array(point, dtype=jnp.float64)
    direction = jnp.subtract(p1, p0)
    dir_to_pt = jnp.subtract(pt, p0)
    return (
        direction[0] * dir_to_pt[1] - direction[1] * dir_to_pt[0]
    ) / jnp.linalg.norm(direction) - desired_dist


@make_constraint
def parallel_constraint(linea, lineb):
    a0 = jnp.array(linea[0], dtype=jnp.float64)
    a1 = jnp.array(linea[1], dtype=jnp.float64)

    b0 = jnp.array(lineb[0], dtype=jnp.float64)
    b1 = jnp.array(lineb[1], dtype=jnp.float64)

    da = jnp.subtract(a1, a0)
    db = jnp.subtract(b1, b0)
    return da[0] * db[1] - da[1] * db[0]


@make_constraint
def angle_constraint(linea, lineb, angle):
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
