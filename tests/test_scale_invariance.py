"""Scale-invariance test for the solve123d Turtle builder.

Tests whether the solver's trust region, step clamping (e.g. 50.0 unit limits),
and residual tolerances converge consistently across micro, standard, and macro scales.
"""

import math
import pytest
import solve123d as cs
from solve123d.turtle import (
    Turtle,
    close,
    closing_constraint,
    forward,
    heading,
    left,
    pen_down,
    pen_up,
)


@pytest.mark.parametrize(
    "scale",
    [
        pytest.param(1e-3, id="micro_1e-3"),
        pytest.param(1e-1, id="sub_millimeter_0.1"),
        pytest.param(1.0, id="standard_1.0"),
        pytest.param(10.0, id="tens_10.0"),
        pytest.param(1e2, id="hundreds_100.0"),
        pytest.param(1e3, id="large_1000.0"),
        pytest.param(1e4, id="macro_10000.0"),
        pytest.param(1e6, id="huge_1000000.0"),
    ],
)
def test_turtle_l_bracket_varying_scale(scale: float):
    """Solves an L-profile with an arc corner across 7 orders of magnitude.

    Probes:
      - Step clamping: Initial guesses start at 50% of the true distance.
        At scale=1e4, the distance to travel is (18 - 9) * 1e4 = 90,000 units.
        If delta is hard-clamped to 50.0 units, it requires ~1,800 iterations
        and will exhaust max_iter.
      - Tolerance: At scale=1e-3, absolute tolerance (1e-7) requires
        high relative precision (1e-4), testing numerical noise limits.
    """
    turn_r = 2.0 * scale
    target_x = 20.0 * scale
    target_y = 30.0 * scale

    expected_l1 = target_x - turn_r  # 18.0 * scale
    expected_l2 = target_y - turn_r  # 28.0 * scale

    with Turtle() as t:
        heading(0)

        # Proportional initial guesses (half of expected length)
        l1 = cs.var(0.5 * expected_l1)
        l1.name = "horizontal_length"
        forward(l1)

        # Corner turn
        left(90, turn_radius=turn_r)

        # Second unknown segment
        l2 = cs.var(0.5 * expected_l2)
        l2.name = "vertical_length"
        forward(l2)

        # Target constraints
        t.x = target_x
        t.y = target_y

        close()

    # Assert convergence and relative accuracy
    sol_l1 = cs.solve(l1)
    sol_l2 = cs.solve(l2)

    assert math.isclose(
        sol_l1, expected_l1, rel_tol=1e-4
    ), f"l1 failed at scale {scale}: expected {expected_l1}, got {sol_l1}"

    assert math.isclose(
        sol_l2, expected_l2, rel_tol=1e-4
    ), f"l2 failed at scale {scale}: expected {expected_l2}, got {sol_l2}"


@pytest.mark.parametrize(
    "scale",
    [
        pytest.param(1e-2, id="small_0.01"),
        pytest.param(1.0, id="standard_1.0"),
        pytest.param(1e1, id="tens_10.0"),
        pytest.param(1e3, id="large_1000.0"),
    ],
)
def test_turtle_hull_varying_scale(scale: float):
    """Tests coupled non-linear angles and distance variables at varying scales."""
    r1 = 10.0 * scale
    r2 = 5.0 * scale
    h = 25.0 * scale

    # Theoretical straight tangent length between two external circles:
    # L = sqrt(h^2 - (r1 - r2)^2)
    expected_tangent = math.sqrt(h**2 - (r1 - r2)**2)

    with Turtle() as t:
        pen_up()
        heading(cs.var(-180.0))
        forward(r1)
        left(90)
        pen_down()

        left(cs.var(180.0), turn_radius=r1)

        f1 = cs.var(0.5 * expected_tangent)
        f1.name = "tangent_1"
        forward(f1)

        arc2_c = left(cs.var(180.0), turn_radius=r2).center
        arc2_c[0].magic = 0.0
        arc2_c[1].magic = h

        f2 = cs.var(0.5 * expected_tangent)
        f2.name = "tangent_2"
        forward(f2)

        closing_constraint(tangency=True)

    sol_f1 = cs.solve(f1)
    sol_f2 = cs.solve(f2)

    assert math.isclose(
        sol_f1, expected_tangent, rel_tol=1e-4
    ), f"Tangent 1 failed at scale {scale}: expected {expected_tangent}, got {sol_f1}"

    assert math.isclose(
        sol_f2, expected_tangent, rel_tol=1e-4
    ), f"Tangent 2 failed at scale {scale}: expected {expected_tangent}, got {sol_f2}"