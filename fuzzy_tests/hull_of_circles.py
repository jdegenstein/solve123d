import math
import solve123d as cs
from solve123d import *
from solve123d.turtle import *
import random

cs.set_verbose(True)


def random_angle():
    return 720 * random.random() - 360


# currently fails on seed 56
for i in range(0, 10000):
    random.seed(i)
    print(f"Seed: {i}")
    r1 = 10 * random.random()
    r2 = 5 * random.random()
    h = 20 * random.random() + r1 + r2
    with Turtle() as t:
        t.simplify_equations = False
        pen_up()

        heading(var(720 * random.random() - 360))
        forward(r1)
        left(90)
        start_tangent = t.heading_vector
        start_point = t.position
        pen_down()

        left(var(random_angle()), turn_radius=r1)
        forward(cs.var(random.random() * 100))
        arc2_c = left(var(random_angle()), turn_radius=r2).center
        arc2_c[0].magic = 0
        arc2_c[1].magic = h
        forward(cs.var(random.random() * 100))
        closing_constraint(tangency=True)
    t.debug_print_solution()
