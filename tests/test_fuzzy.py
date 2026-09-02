"""Fuzzy stress tests for hull of two circles.

Tests the geometric constraint solver across difficult non-linear search
configurations with random angle and distance guesses. Based on the test
from the fuzzy_tests folder, but focused on the 7000 to 7100 range as
there are particularly difficult cases in this range and it should run
relatively faster than that.
"""

import random
import pytest
import solve123d as cs
from solve123d.turtle import (
    Turtle,
    closing_constraint,
    forward,
    heading,
    left,
    pen_down,
    pen_up,
)


@pytest.mark.parametrize("seed", range(7000, 7101))
def test_hull_of_two_circles_fuzzy(seed: int):
    rng = random.Random(seed)

    # 1. Deterministic sequence matching original fuzzy generation
    r1 = 10.0 * rng.random()
    r2 = 5.0 * rng.random()
    h = 20.0 * rng.random() + r1 + r2

    a1_val = 720.0 * rng.random() - 360.0
    a2_val = 720.0 * rng.random() - 360.0
    f1_val = rng.random() * 100.0
    a3_val = 720.0 * rng.random() - 360.0
    f2_val = rng.random() * 100.0

    # 2. Construct constrained sketch
    with Turtle() as t:
        t.simplify_equations = False
        pen_up()

        a1 = cs.var(a1_val)
        a1.name = "bad_a1"
        heading(a1)
        forward(r1)
        left(90)
        pen_down()

        a2 = cs.var(a2_val)
        a2.name = "bad_a2"
        left(a2, turn_radius=r1)

        bad_forward_1 = cs.var(f1_val)
        bad_forward_1.name = "bad_forward_1"
        forward(bad_forward_1)

        a3 = cs.var(a3_val)
        a3.name = "bad_a3"
        arc2_c = left(a3, turn_radius=r2).center
        arc2_c[0].magic = 0
        arc2_c[1].magic = h

        bad_forward_2 = cs.var(f2_val)
        bad_forward_2.name = "bad_forward_2"
        forward(bad_forward_2)

        closing_constraint(tangency=True)

    # 3. Solve and assert convergence with failure diagnostics
    try:
        line = t.to_build123d()
        assert line is not None
    except cs.SolverError as err:
        # Collect primitive states for debugging
        primitives_report = []
        for i, p in enumerate(t.primitive_list):
            try:
                if hasattr(p, "points"):
                    p0 = cs.solve(p.points[0])
                    p1 = cs.solve(p.points[1])
                    primitives_report.append(f"    line {i}: {p0} -> {p1}")
                elif hasattr(p, "center"):
                    p0 = cs.solve(p.start_point)
                    p1 = cs.solve(p.end_point)
                    c = cs.solve(p.center)
                    primitives_report.append(f"    arc {i}: center={c}, {p0} -> {p1}")
            except Exception:
                primitives_report.append(f"    primitive {i}: could not unwrap coordinates")

        report = "\n".join(primitives_report)

        pytest.fail(
            f"\nConstraint Solver failed to converge on seed {seed}:\n"
            f"Geometry:\n"
            f"  r1 = {r1:.6f}\n"
            f"  r2 = {r2:.6f}\n"
            f"  h  = {h:.6f}\n"
            f"Initial Guesses:\n"
            f"  bad_a1        = {a1_val:.6f} deg\n"
            f"  bad_a2        = {a2_val:.6f} deg\n"
            f"  bad_forward_1 = {f1_val:.6f}\n"
            f"  bad_a3        = {a3_val:.6f} deg\n"
            f"  bad_forward_2 = {f2_val:.6f}\n"
            f"Primitives:\n{report}\n"
            f"Solver Message:\n  {err}"
        )