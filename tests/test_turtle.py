"""
Test turtle

name: test_turtle.py
by:   Dmytry Lavrov
date: August 2025
desc:
    This unit test tests turtle graphics - inspired construction
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


class TurtleTest(unittest.TestCase):
    # TODO: more coverage for Turtle

    def test_hull_of_two_circles(self):
        r1 = 10
        r2 = 5
        h = 20
        with Turtle() as t:
            pen_up()
            # we don't know where the bottom arc starts, but we know it's somewhere around -r1,0
            # so we provide initial guess of -180 degrees here
            heading(cs.var(-180))
            forward(r1)
            left(90)
            # the tangent of the arc is 90 degrees to the left of where radius is pointing
            pen_down()
            # Likewise, initial guess of 180 for the arc angle
            left(cs.var(180), turn_radius=r1)
            forward(cs.var(h))
            arc2_c = left(cs.var(180), turn_radius=r2).center
            arc2_c[0].magic = 0
            arc2_c[1].magic = h
            forward()
            closing_constraint(tangency=True)
        line = t.to_build123d()
        face = build123d.make_face(line)
        a = face.area
        self.assertAlmostEqual(a, 505.77431095057443)

    def test_arc_error(self):
        excepted = False
        try:
            with Turtle() as t:
                t.simplify_equations = False
                forward()
                pen_up()
                t.turn_radius = 1
                left(90)
                forward()
                t.x = 10
                t.y = 10
                t.turn_radius = 0
                right()  ## ends up turning it 180 degrees wrong way and walking backwards TODO: fix or output a warning when that happens
                l = cs.var(1)
                forward(l)
                t.x = 20
                t.y = 20
                t.turn_radius = 2
                heading(
                    90
                )  # this statement should throw because direction is variables

        except RuntimeError:
            excepted = True
        self.assertTrue(excepted)
        # TODO: fix backwards solution problem if possible, and remove abs here
        self.assertAlmostEqual(abs(cs.solve(l)), math.sqrt(2.0) * 10)

    def test_rounded_triangle(self):
        with Turtle() as t:
            t.turn_radius = 1
            l = cs.var(1)
            forward(l)
            left(120)
            t.primitive_list[-1].center[0].magic = 10
            forward()
            left()
            forward(l)
            left(120)
            t.heading_vector[0].magic = 1
            t.heading_vector[1].magic = 0
            closing_constraint()
        line = t.to_build123d()
        face = build123d.make_face(line)
        a = face.area
        self.assertAlmostEqual(a, 76.44286284281172)
        if __name__ == "__main__":  # pragma: no cover
            import ocp_vscode

            ocp_vscode.show(face)

    def too_tall_toby(self):
        """See examples/turtle_sketching.py for better code. This does weird stuff to increase coverage"""
        with Turtle() as t:
            pen_up()  # Disables appending of primitives (moves work the same)
            heading(270 + 25)
            forward(33 - 10)
            pen_down()  # Start adding primitives, we start at the point that is 33-10 units away from the circle center.
            forward(10)
            # For coverage, does not fully verify left turn circles work
            t.turn_radius = 1e-10
            left(90)
            t.turn_radius = 0
            self.assertTrue(cs.turtle.all_values(t.position))
            forward()  # Move forward an unknown distance (the constraint solver will solve for the distance that meets constraints)
            self.assertFalse(cs.turtle.all_values(t.position))
            # heading(90)
            t.turn_radius = 1e-10
            heading(0, 1)
            # can also write t.x=
            t.x.magic = 33  # Constrain turtle's x coordinate to 33 (which resolves the unknown amount above)
            t.turn_radius = 0
            forward()  # Another unknown distance move
            heading(180 - 35)
            forward()  # and another
            # heading(90)
            # For better coverage:
            right(90 - 35)

            # Test nested turtle, todo: do something useful with it
            with Turtle() as t2:
                left(90)
                with Turtle(use_stack=False) as t3:
                    left(90)

            forward(5)
            # heading(180)
            heading((-1, 0))
            forward(6)
            # Now at upper left corner, whose position is known
            t.x = -8  # Two constraints for two unknown-distance moves above
            # Longer way to write a constraint
            t.y.magic = 120
            # The upper rectangle thingy
            heading(-90)
            forward(22)
            heading(0)
            forward(6)
            heading(90)
            forward(5)
            heading(-40)
            forward()
            # Uppermost arc
            t.turn_radius = (
                13  # With turn_radius set to nonzero, all turns will add arcs
            )
            heading(-90)  # Create the topmost arc
            t.x = 33 - 10  # Constrain x at the end of the arc
            forward()
            heading(180 + (90 - 65))
            l = forward()  # Remember the line created by this forward move
            t.x = 0
            t.y = 33
            t.turn_radius = 0  # No more arcs, do sharp corners
            heading(-90)
            forward()  # Unknown distance move at the "10" constraint in the sketch right above the hole

            # Distance from the saved line, to turtle position, must be 10 (will solve for zero of the equation on the right side).
            # Distances are signed, positive leftwards of the line, negative rightwards.
            cs.magic.zero = cs.line_pt_dist(l, t.position) - 10

            heading(90 - 65)
            forward()
            # t.turn_radius = 13  # Do the last two arcs
            # Use parameter turn_radius instead
            heading(-90, turn_radius=13)  # This adds 2nd arc from the bottom
            t.x = 33 - 10  # Constraint needs to be done at the end of the arc
            forward()
            heading(180 + 25, turn_radius=13)
            forward()
            # t.turn_radius = 0  # Turn arcs off again
            closing_constraint()  # Solve for the end point to match exactly the starting point
        line = t.to_build123d()
        face = build123d.make_face(line)
        a = face.area
        self.assertAlmostEqual(
            a,
            2150.2234406261177,
            places=5,
            msg="Note: the value was not independently calculated",
        )

    def test_too_tall_toby_normal(self):
        self.too_tall_toby()

    def test_too_tall_toby_opportunistic(self):
        cs.set_opportunistic(True)
        self.too_tall_toby()
        cs.set_opportunistic(False)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
