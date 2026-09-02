"""Targeted unit tests to achieve comprehensive coverage for solve123d.

Directly exercises edge cases, fallback branches, dual arithmetic operators,
and geometric constraints in constraint_solver.py and turtle.py.
"""

import math
import numpy as np
import pytest
import solve123d as cs
from solve123d.turtle import (
    Center,
    DirectionAngle,
    Turtle,
    TurtleArc,
    TurtleLine,
    make_non_zero,
)


def test_solver_settings_metadata_keyerror():
    """Covers lines in SolverSettings.append_settings when combine key is missing."""
    s1 = cs.SolverSettings(custom_param=1)
    s1.settings_metadata = {"custom_param": {}}  # Omit "combine" key intentionally
    s2 = cs.SolverSettings(custom_param=2)
    s1.append_settings(s2)
    assert s1.custom_param == 1


def test_dual_number_dunder_methods():
    """Covers Dual comparisons, conversions, and grad-None branches."""
    d1 = cs.Dual(3.75, np.array([1.0, 2.0]))
    d2 = cs.Dual(3.75, np.array([1.0, 2.0]))
    d3 = cs.Dual(5.0, None)
    d_zero = cs.Dual(0.0)

    # Conversions (__float__)
    assert float(d1) == 3.75
    assert int(d1) == 3
    assert round(d1) == 4
    assert round(d1, 1) == 3.8

    # Equality and relational comparisons
    assert d1 == d2
    assert d1 == 3.75
    assert not (d1 == 4.0)

    assert d1 < d3
    assert d1 < 5.0
    assert not (d3 < d1)

    assert d1 <= d2
    assert d1 <= 3.75
    assert d1 <= 5.0

    assert d3 > d1
    assert d3 > 3.75
    assert not (d1 > d3)

    assert d1 >= d2
    assert d1 >= 3.75
    assert d3 >= 5.0

    # make_zero on Dual
    assert d_zero.make_zero() is d_zero
    with pytest.raises(cs.SolverError, match="Inconsistent constant constraint"):
        d3.make_zero()

    # Addition branches with mixed/none gradients
    d_no_grad = cs.Dual(2.0, None)
    d_grad = cs.Dual(3.0, np.array([1.0]))

    add_rhs_no_grad = d_grad + d_no_grad
    assert np.allclose(add_rhs_no_grad.grad, [1.0])

    add_lhs_no_grad = d_no_grad + d_grad
    assert np.allclose(add_lhs_no_grad.grad, [1.0])

    add_both_no_grad = d_no_grad + cs.Dual(1.0, None)
    assert add_both_no_grad.grad is None

    # Subtraction branches (self.grad is not None and other.grad is None)
    sub_grad_lhs = d_grad - d_no_grad
    assert np.allclose(sub_grad_lhs.grad, [1.0])

    sub_grad_rhs = d_no_grad - d_grad
    assert np.allclose(sub_grad_rhs.grad, [-1.0])

    sub_no_grads = d_no_grad - cs.Dual(1.0, None)
    assert sub_no_grads.grad is None

    rsub_grad = 10.0 - d_grad
    assert np.allclose(rsub_grad.grad, [-1.0])

    rsub_no_grad = 10.0 - d_no_grad
    assert rsub_no_grad.grad is None

    # Multiplication branches
    mul_scalar = d_no_grad * 3.0
    assert mul_scalar.grad is None

    rmul_scalar = 3.0 * d_no_grad
    assert rmul_scalar.grad is None

    # Division and power branches
    div_scalar = d_no_grad / 2.0
    assert div_scalar.grad is None

    rdiv_no_grad = 6.0 / d_no_grad
    assert rdiv_no_grad.grad is None

    rdiv_grad = 6.0 / d_grad
    assert np.allclose(rdiv_grad.grad, [-6.0 / 9.0])

    pow_no_grad = d_no_grad**3
    assert pow_no_grad.grad is None

    # Modulo, unary pos, abs, and representation
    mod_val = d_grad % 2.0
    assert mod_val.val == 1.0

    pos_val = +d_grad
    assert pos_val is d_grad

    abs_no_grad = abs(cs.Dual(-4.0, None))
    assert abs_no_grad.val == 4.0 and abs_no_grad.grad is None

    abs_grad = abs(cs.Dual(4.0, np.array([2.0])))
    assert np.allclose(abs_grad.grad, [2.0])

    assert repr(cs.Dual(1.2345678)) == "Dual(val=1.234568)"


def test_elementary_dual_math_functions():
    """Covers elementary Dual functions when given raw floats or None gradients."""
    # d_sqrt
    assert cs.d_sqrt(9.0) == 3.0
    assert cs.d_sqrt(cs.Dual(9.0, None)).grad is None
    assert np.allclose(cs.d_sqrt(cs.Dual(9.0, np.array([1.0]))).grad, [0.5 / 3.0])

    # d_hypot
    assert cs.d_hypot(3.0, 4.0) == 5.0
    h_dual = cs.d_hypot(cs.Dual(3.0, None), 4.0)
    assert h_dual.val == 5.0

    # d_sin and d_cos
    assert cs.d_sin(0.0) == 0.0
    assert cs.d_sin(cs.Dual(0.0, None)).grad is None
    assert cs.d_cos(0.0) == 1.0
    assert cs.d_cos(cs.Dual(0.0, None)).grad is None

    # d_atan2 and d_abs
    assert cs.d_atan2(0.0, 1.0) == 0.0
    atan_dual = cs.d_atan2(cs.Dual(1.0, None), 1.0)
    assert math.isclose(atan_dual.val, math.pi / 4)

    assert cs.d_abs(-42.0) == 42.0
    assert cs.d_abs(cs.Dual(-42.0)).val == 42.0


def test_magical_zero_edge_cases():
    """Covers MagicalZero direct scalar and Dual assignments."""
    cs.magic.zero = 0.0
    cs.magic.zero = cs.Dual(0.0)
    cs.magic.zero = 0

    with pytest.raises(cs.SolverError, match="Inconsistent constant constraint"):
        cs.magic.zero = 5.0

    with pytest.raises(cs.SolverError, match="Inconsistent constant constraint"):
        cs.magic.zero = cs.Dual(5.0)


def test_solve_everything_empty_and_error_paths():
    """Covers early-returns and diagnostic branches in solve_everything."""
    # Variable with no constraints returns early
    v = cs.Variable(42.0)
    cs.solve_everything(v)
    assert v.solution is None

    # WrappedFunction with no arguments returns early
    wf = cs.WrappedFunction(lambda: 0.0)
    cs.solve_everything(wf)

    # Initial guess already at exact residual 0
    zero_var = cs.var(0.0)
    zero_var.magic = 0.0
    assert cs.solve(zero_var) == 0.0

    # Overconstrained failure triggering the "over constrained" message
    a = cs.var(1.0)
    cs.magic.zero = a - 1.0
    cs.magic.zero = a - 2.0
    cs.magic.zero = a - 3.0
    with pytest.raises(cs.SolverError, match="over constrained"):
        cs.solve(a)

    # Inconsistent square system
    b = cs.var(0.0)
    cs.magic.zero = b**2 + 10.0
    with pytest.raises(cs.SolverError) as exc_info:
        cs.solve(b)
    assert "over constrained" not in str(exc_info.value)


def test_eval_sketch_dual_none_gradient():
    """Covers lines 670-671: Dual constraint result where grad is None."""
    x = cs.var(2.0)
    cs.magic.zero = x - 5.0

    # Constraint returning a Dual instance with grad=None
    def c_with_none_grad(v):
        return cs.Dual(v.val - 5.0, None)

    wf = cs.WrappedFunction(c_with_none_grad, [x])
    cs.magic.zero = wf
    assert math.isclose(cs.solve(x), 5.0, abs_tol=1e-5)


def test_basin_hopping_and_polishing_paths():
    """Covers lines 771-774 and 782-785: candidate polish and safety polish."""
    a = cs.var(1.0)
    a.name = "test_angle"
    # Setting max_tolerance to 1e-15 keeps the stopping residual (~3.4e-11)
    # between 1e-3 and base_tol, triggering candidate polish and safety polish.
    a.settings = cs.SolverSettings(max_tolerance=1e-15)
    cs.magic.zero = cs.make_wrapper(cs.d_sin)(a * (math.pi / 180.0)) - 0.5

    assert math.isclose(cs.solve(a), 30.0, abs_tol=1e-4)


def test_make_constraint_without_variables():
    """Covers line in make_constraint when called with constant scalar arguments."""
    res_sum = cs.sum_constraint(1.0, 2.0, 3.0)
    assert res_sum == 0.0

    res_dist = cs.distance_constraint((0.0, 0.0), (3.0, 4.0), 5.0)
    assert math.isclose(res_dist, 0.0)


def test_geometric_constraints_direct_execution():
    """Covers built-in constraints executed directly with symbolic variables."""
    # 3D distance constraint
    p0 = (0.0, 0.0, 0.0)
    p1 = cs.var((0.0, 0.0, 2.0))
    cs.distance_constraint(p0, p1, 5.0)
    cs.magic.zero = p1[0]
    cs.magic.zero = p1[1]
    assert math.isclose(cs.solve(p1[2]), 5.0, abs_tol=1e-5)

    # Scalar (1D) distance constraint
    s1 = cs.var(0.0)
    cs.distance_constraint(s1, 10.0, 3.0)
    assert math.isclose(abs(cs.solve(s1) - 10.0), 3.0, abs_tol=1e-5)

    # line_point_distance_constraint executed as constraint
    line = ((0.0, 0.0), (10.0, 0.0))
    pt = cs.var((5.0, 4.0))
    cs.line_point_distance_constraint(line, pt, 2.0)
    cs.magic.zero = pt[0] - 5.0
    assert math.isclose(cs.solve(pt[1]), 2.0, abs_tol=1e-5)

    # line_contains_point_2d_constraint executed as constraint
    pt2 = cs.var((3.0, 1.0))
    cs.line_contains_point_2d_constraint(((0.0, 0.0), (5.0, 5.0)), pt2)
    cs.magic.zero = pt2[0] - 4.0
    assert math.isclose(cs.solve(pt2[1]), 4.0, abs_tol=1e-5)

    # parallel_2d_constraint executed as constraint
    l_base = ((0.0, 0.0), (1.0, 0.0))
    l_parallel = ((0.0, 0.0), cs.var((2.0, 0.5)))
    cs.parallel_2d_constraint(l_base, l_parallel)
    cs.magic.zero = l_parallel[1][0] - 2.0
    assert math.isclose(cs.solve(l_parallel[1][1]), 0.0, abs_tol=1e-5)

    # angle_2d_constraint executed as constraint
    l_perp = ((0.0, 0.0), cs.var((0.1, 2.0)))
    cs.angle_2d_constraint(l_base, l_perp, math.pi / 2)
    cs.magic.zero = l_perp[1][1] - 2.0
    assert math.isclose(cs.solve(l_perp[1][0]), 0.0, abs_tol=1e-5)


def test_turtle_coverage_gaps():
    """Covers Center properties, partial center setters, and debug printers in turtle.py."""
    # Center x and y property getters and setters
    c = Center((cs.var(1.0), cs.var(2.0)))
    assert c.x.initial_value == 1.0
    assert c.y.initial_value == 2.0
    c.x = 10.0
    c.y = 20.0
    assert math.isclose(cs.solve(c.x), 10.0, abs_tol=1e-5)
    assert math.isclose(cs.solve(c.y), 20.0, abs_tol=1e-5)

    # TurtleArc.center validation error
    arc = TurtleArc((0, 0), (1, 0), (1, 1), (0, 1), 1)
    with pytest.raises(ValueError, match="arc.center must be a 2D coordinate sequence"):
        arc.center = 42
    with pytest.raises(ValueError, match="arc.center must be a 2D coordinate sequence"):
        arc.center = (1, 2, 3)

    # Partial center tuple assignment on an arc with symbolic center variables
    arc_var = TurtleArc((0, 0), (1, 0), (1, 1), Center((cs.var(0.0), cs.var(0.0))), 1)
    arc_var.center = (15.0, None)
    arc_var.center = (None, 25.0)
    assert math.isclose(cs.solve(arc_var.center.x), 15.0, abs_tol=1e-5)
    assert math.isclose(cs.solve(arc_var.center.y), 25.0, abs_tol=1e-5)

    # make_non_zero with Variable (triggers float(Variable) -> TypeError)
    v = cs.var(0.0)
    non_zero_v = make_non_zero(v)
    assert isinstance(non_zero_v, cs.WrappedFunction)

    # Debug print methods
    line_prim = TurtleLine((0, 0), (1, 1))
    line_prim.debug_print()
    arc.debug_print()


def test_helper_utilities_and_operator_wrappers():
    """Covers to_float, absvar, operator wrappers, and unmasked step clamping."""
    assert cs.to_float(cs.Dual(3.14)) == 3.14
    assert cs.to_float(np.array(2.5)) == 2.5
    assert cs.to_float(np.float64(2.5)) == 2.5
    assert cs.to_float("plain_string") == "plain_string"

    abs_vars = cs.absvar((1.0, -2.0))
    assert len(abs_vars) == 2

    x = cs.var(3.0)
    _ = 2 + x
    _ = 2 - x
    _ = 2 * x
    _ = 6 / x
    _ = 2**x
    _ = 7 % x
    _ = -x
    _ = +x
    _ = abs(x)

    # SimpleSolver step clamping with and without angle_mask
    solver = cs.SimpleSolver(tol=1e-6, max_iter=10)
    J = np.eye(2)
    r = np.array([200.0, 200.0])

    delta_masked = solver.step(J, r, 1e-4, angle_mask=np.array([True, False]))
    assert abs(delta_masked[0]) <= 90.0
    assert abs(delta_masked[1]) <= 50.0

    delta_unmasked = solver.step(J, r, 1e-4, angle_mask=None)
    assert np.max(np.abs(delta_unmasked)) <= 50.0 + 1e-9