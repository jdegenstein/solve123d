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
from build123d import *

wrapped_abs = cs.make_wrapper(cs.d_abs)
cs.set_verbose(True)

# cs.set_opportunistic(True)


class TurtleTest(unittest.TestCase):
    # TODO: more coverage for Turtle

    def test_connector(self):
        w = 11
        l = 66
        r = 22
        with Turtle() as t:
            left(90)
            forward(w / 2)
            left(90)
            right(cs.var(20), turn_radius=cs.var(l))
            c = left(cs.var(90), turn_radius=r).center
            c[0].magic = -l
            c[1].magic = 0
            t.y = 0
            heading(0)
            forward()
            closing_constraint(tangency=False)
        line = t.to_build123d()
        if __name__ == "__main__":  # pragma: no cover
            import ocp_vscode

            ocp_vscode.show(line)

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
                be_at(y=10)
                t.turn_radius = 0
                right()  ## ends up turning it 180 degrees wrong way and walking backwards TODO: fix or output a warning when that happens
                l = cs.absvar(1)
                forward(l)
                # For better coverage of be_at
                t.simplify_equations = True
                be_at((20, 20))
                t.simplify_equations = False
                t.turn_radius = 2
                heading(90)  # this statement should throw because direction is variable

        except RuntimeError:
            excepted = True
        self.assertTrue(excepted)
        # TODO: Add correct check
        # self.assertAlmostEqual(abs(cs.solve(l)), math.sqrt(2.0) * 10)

    def test_teleport(self):
        cs.solver_settings.max_tolerance = 1e10
        with Turtle() as t:
            teleport(1, 1)
            forward(10)
            left(90)
            forward(10)
            left(90)
            forward(10)
            r = cs.var(1)
            r.name = "r"
            left(turn_radius=r)
            closing_constraint()
            teleport((1, 2))
            self.assertEqual(t.x, 1)
            self.assertEqual(t.y, 2)

        line = t.to_build123d()
        face = build123d.make_face(line)
        a = face.area
        self.assertAlmostEqual(a, math.pi * 5 * 5 / 2 + 10 * 10)

    def test_closing_constraint_warning(self):
        with Turtle() as t:
            teleport(1, 1)
            forward(10)
            left(90)
            forward(10)
            left(90)
            forward(10)
            left(180, turn_radius=5)
            closing_constraint()
        line = t.to_build123d()
        face = build123d.make_face(line)
        a = face.area
        self.assertAlmostEqual(a, math.pi * 5 * 5 / 2 + 10 * 10)

    def test_ignore_errors(self):
        with Turtle() as t:
            teleport(1, 1)
            forward(10)
            left(90)
            forward(10)
            left(90)
            forward(10)
            forward(0)
            forward(1e-12)
            left(180, turn_radius=5)
            closing_constraint()
        line = t.to_build123d(ignore_errors=True)
        prims = [*t.to_build123d_list(debug_objects=True, ignore_errors=True)]

        with self.assertRaises(Exception):
            line = t.to_build123d(ignore_errors=False)
        with self.assertRaises(Exception):
            prims = [*t.to_build123d_list(debug_objects=False, ignore_errors=False)]

        with Turtle() as t3:
            left(180, turn_radius=1e-40)
            left(180, turn_radius=float("nan"))
            right(180, turn_radius=-1e40)
            right(180, turn_radius=-20)
            forward(1e-40)
            left(90)
            forward(1e40)
        t3.primitive_list[0].end_point = (1, 0)

        with self.assertRaises(Exception):
            line = t3.to_build123d(ignore_errors=False)
        with self.assertRaises(Exception):
            prims = [*t3.to_build123d_list(debug_objects=True)]
        # face = build123d.make_face(line)
        # a = face.area
        # self.assertAlmostEqual(a, math.pi * 5 * 5 / 2 + 10 * 10)

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
            be_heading(0)
            # test the warning for empty be_at
            be_at()
            closing_constraint()
        line = t.to_build123d()
        # for coverage
        line_parts = [*t.to_build123d_list(ignore_errors=True, debug_objects=True)]
        face = build123d.make_face(line)
        a = face.area
        h = math.sqrt(0.75) * 10
        triangle_area = 10 * h / 2
        side_area = 3 * 10 * 1
        circle_area = 1 * 1 * math.pi
        self.assertAlmostEqual(a, triangle_area + side_area + circle_area)
        if __name__ == "__main__":  # pragma: no cover
            import ocp_vscode

            ocp_vscode.show(face, line_parts)

    def too_tall_toby(self, simplify=True):
        """See examples/turtle_sketching.py for better code. This does weird stuff to increase coverage"""
        with Turtle() as t:
            t.simplify_equations = simplify
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
            be_at(
                x=33
            )  # Constrain turtle's x coordinate to 33 (which resolves the unknown amount above)

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
            # Two constraints for two unknown-distance moves above

            be_at(x=-8, y=120)
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
            be_at(y=33)
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
            be_at(x=33 - 10)  # Constraint needs to be done at the end of the arc
            forward()
            heading(180 + 25, turn_radius=13)
            forward()
            # Check deprecated
            print(t.heading_vector)
            # t.turn_radius = 0  # Turn arcs off again
            closing_constraint()  # Solve for the end point to match exactly the starting point
        line = t.to_build123d()
        prims = [*t.to_build123d_list(debug_objects=False, ignore_errors=False)]
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
        self.too_tall_toby(False)

    def test_ttt_23_t_42(self):
        # ----------------------------------------------------------------------
        # Sketch 1: Main Arm Profile
        # ----------------------------------------------------------------------
        with Turtle() as t:
            heading(0)
            forward()

            # Corner radius set inline with 90 deg turn
            left(90, turn_radius=55)
            forward(11)

            # Directly assign the arc center coordinates
            arc = left(turn_radius=20)
            arc.center = (150, 66)

            # Reverse corner arc into return path
            right(turn_radius=35)
            t.y = 40
            forward()
            t.x = 0
            t.y = 40

            # Unconstrains heading, steps home to (0, 0), and closes the wire
            close()

        line = t.to_build123d()
        print(f"Sketch 1 size: {line.bounding_box().size}")

        # ----------------------------------------------------------------------
        # Sketch 2: Cross Section Profile
        # ----------------------------------------------------------------------
        with Turtle() as t2:
            heading(-90)
            forward()

            # Turn to arc extrema
            left(turn_radius=28)
            t2.y = -160 / 2 - 28
            t2.x = 28

            # Turn again with matching radius
            left(turn_radius=28)
            forward()
            t2.x = 100
            t2.y = -40

            # Sharp corner turn up to y=0
            left(turn_radius=0)
            forward()
            t2.x = 100
            t2.y = 0

            # Steps home from (100, 0) to (0, 0) and closes the wire
            close()

        line2 = Plane.XZ * t2.to_build123d()
        print(f"Sketch 2 size: {line2.bounding_box().size}")

        # ----------------------------------------------------------------------
        # Solid Modeling Pipeline
        # ----------------------------------------------------------------------
        with BuildPart() as p:
            with BuildSketch() as s:
                add(line)  # Already closed by close()
                make_face()
                split(bisect_by=Plane.YZ.offset(40))
            extrude(amount=-40)

            with BuildSketch(Plane.XZ) as s2:
                add(line2)  # Already closed by close()
                make_face()
            extrude(amount=-20)

            with BuildSketch(Plane.ZY):
                with Locations((0, 40)):
                    Rectangle(80 - 28, 60, align=(Align.CENTER, Align.MIN))
            extrude(amount=-200, mode=Mode.SUBTRACT)

            with BuildSketch(Plane.XZ):
                with Locations((28, -80)):
                    Circle(28)
            extrude(amount=-30)

            with Locations(Plane.XZ):
                with Locations((28, -80)):
                    Hole(27 / 2)

            with BuildSketch(Plane.XZ):
                with Locations((40, 0)):
                    Circle(40)
            extrude(amount=-60)

            with Locations(Plane.XZ):
                with Locations((40, 0)):
                    Hole(20)

            with Locations((150, 66)):
                Hole(22 / 2)

            mirror(about=Plane.XY)

        densa = 7800 / 1e6  # Carbon steel density (g/mm^3)
        mass = p.part.volume * densa
        self.assertAlmostEqual(
            mass,
            5815.00,
            places=2,
        )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
