"""
Test turtle with bad initial guesses

name: test_turtle.py
by:   Dmytry Lavrov
date: August 2025
desc:
    This unit test tests turtle graphics - inspired construction, with deliberately bad initial guesses
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

import math
import unittest
import solve123d as cs
from solve123d.turtle import *
import build123d

import jax
import jax.numpy as jnp

wrapped_abs = cs.make_wrapper(jnp.abs)
cs.set_verbose(True)


class TurtleBadGuessTest(unittest.TestCase):

    def test_hull_of_two_circles(self):
        r1 = 10
        r2 = 5
        h = 20
        with Turtle() as t:
            pen_up()
            # we don't know where the bottom arc starts, but we know it's somewhere around -r1,0
            heading(cs.var(-180))
            forward(r1)
            left(90)
            # the tangent of the arc is 90 degrees to the left of where radius is pointing
            pen_down()
            left(turn_radius=r1)
            forward()
            arc2_c = left(turn_radius=r2).center
            arc2_c[0].magic = 0
            arc2_c[1].magic = h
            forward()
            closing_constraint(tangency=True)
        line = t.to_build123d()
        face = build123d.make_face(line)
        a = face.area
        self.assertAlmostEqual(a, 505.77431095057443)

    def test_rounded_triangle(self):
        with Turtle() as t:
            t.turn_radius = 1
            l = cs.var(1)
            forward(l)
            left(120)
            t.primitive_list[-1].center[0].magic = 10
            l2 = wrapped_abs(cs.var(1))
            forward(l2)
            left(cs.var(340))
            forward(l)
            left(120)
            closing_constraint(tangency=True)
        t.debug_print_solution()
        line = t.to_build123d()
        face = build123d.make_face(line)
        a = face.area
        self.assertAlmostEqual(a, 76.44286284281172)
        if __name__ == "__main__":  # pragma: no cover
            import ocp_vscode

            ocp_vscode.show(line, face)

    def test_almost_straight(self):
        r = 1
        with Turtle() as t:
            forward(100)
            left(turn_radius=r)
            forward()
            a1 = left(turn_radius=r)
            a1.center[0].magic = 200
            a1.center[1].magic = 2
            forward()
            closing_constraint()
        t.debug_print_solution()
        line = t.to_build123d()
        face = build123d.make_face(line)
        a = face.area
        print(a)
        self.assertAlmostEqual(a, 251.58704585025112)
        if __name__ == "__main__":  # pragma: no cover
            import ocp_vscode

            ocp_vscode.show(line)

    def test_almost_straight_bad_guess(self):
        r = 1
        with Turtle() as t:
            forward(100)
            bad_guess = cs.var(181)
            bad_guess.name = "bad guess for angle"
            left(bad_guess, turn_radius=r)
            forward()
            a1 = left(turn_radius=r)
            a1.center[0].magic = 200
            a1.center[1].magic = 2
            forward()
            closing_constraint()
        t.debug_print_solution()
        line = t.to_build123d()
        face = build123d.make_face(line)
        a = face.area
        print(a)
        self.assertAlmostEqual(a, 251.58704585025112)
        if __name__ == "__main__":  # pragma: no cover
            import ocp_vscode

            ocp_vscode.show(line)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
