import math
import solve123d as cs
from solve123d import *
from solve123d.turtle import *
import random

from build123d import *
import ocp_vscode

cs.set_verbose(True)


def random_angle():
    return 720 * random.random() - 360


failures = 0

# currently fails on seed 56
for i in range(7000, 7100):
    random.seed(i)
    print(f"Seed: {i}")
    r1 = 10 * random.random()
    r2 = 5 * random.random()
    h = 20 * random.random() + r1 + r2
    with Turtle() as t:
        t.simplify_equations = False
        pen_up()

        a1 = var(720 * random.random() - 360)
        a1.name = "bad_a1"
        heading(a1)
        forward(r1)
        left(90)
        pen_down()

        a2 = var(720 * random.random() - 360)
        a2.name = "bad_a2"

        left(a2, turn_radius=r1)
        bad_forward_1 = cs.var(random.random() * 100)
        bad_forward_1.name = "bad_forward_1"
        forward(bad_forward_1)
        a3 = var(random_angle())
        a3.name = "bad_a3"
        arc2_c = left(a3, turn_radius=r2).center
        arc2_c[0].magic = 0
        arc2_c[1].magic = h
        bad_forward_2 = cs.var(random.random() * 100)
        bad_forward_2.name = "bad_forward_2"
        forward(bad_forward_2)
        closing_constraint(tangency=True)
    try:
        t.debug_print_solution()
    except cs.SolverError:
        failures += 1
        print(f"Failure {failures} at seed {i}")
        t.debug_print_solution()

        # line=t.to_build123d()
        circle1 = Circle(r1)
        circle2 = Pos(0, h) * Circle(r2)
        prims = [*t.to_build123d_list(ignore_errors=True, debug_objects=True)]
        ocp_vscode.show(circle1, circle2, prims)
        breakpoint()

        # raise
