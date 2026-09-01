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
import solve123d.turtle as turtle
import build123d

cs.set_verbose(True)

# cs.set_opportunistic(True)


class LowLevelTurtleTest(unittest.TestCase):
    # TODO: more coverage for Turtle
    def test_wrapped_angle(self):
        self.assertAlmostEqual(turtle.normalize_angle(math.pi + 0.1), -math.pi + 0.1)
        self.assertAlmostEqual(
            turtle.normalize_angle(3.0 * math.pi + 0.1), -math.pi + 0.1
        )
        self.assertAlmostEqual(
            turtle.normalize_angle(2.0 * math.pi + math.pi + 0.1), -math.pi + 0.1
        )

        self.assertAlmostEqual(
            turtle.normalize_angle(math.radians(-370)), math.radians(-10)
        )
        self.assertAlmostEqual(
            turtle.normalize_angle(math.radians(190)), math.radians(-170)
        )
        self.assertAlmostEqual(
            turtle.normalize_angle(math.radians(-190)), math.radians(170)
        )

    def test_angle_error(self):
        e = turtle.angle_error((1, 0), (0, 1), math.radians(90))
        self.assertAlmostEqual(e, 0.0)

        e = turtle.angle_error((1, 0), (0, 1), math.radians(0))
        self.assertAlmostEqual(e, math.radians(90))
        e = turtle.angle_error((1, 0), (-1, 1), math.radians(0))
        self.assertAlmostEqual(e, math.radians(135))

        e = turtle.angle_error((0, 1), (-1, -1), math.radians(0))
        self.assertAlmostEqual(e, math.radians(135))

    def test_directions(self):
        a = math.radians(71)
        b = math.radians(13)
        c = math.radians(17)
        d = math.radians(27)

        sc, cc = turtle.solver_sincos(c)
        to_test = (
            turtle.DirectionAngle(a),
            turtle.DirectionNormalized((math.cos(b), math.sin(b))),
            turtle.DirectionUnnormalized((cc * 2, sc * 2)),
            turtle.DirectionDiff(((2, 3), (2 + math.cos(d) * 2, 3 + math.sin(d) * 2))),
        )
        angles = (a, b, c, d)
        for i, x in enumerate(to_test):
            self.assertTrue(x.known())
            cmp = turtle.DirectionAngle(angles[i])
            self.assertIs(turtle.make_direction_from_user_params(x, None, None), x)
            self.assertAlmostEqual(turtle.angle_error(x.dir_n(), cmp.dir_n()), 0.0)
            for j, y in enumerate(to_test):
                combined = x.combine(y)
                compare = turtle.DirectionAngle(angles[i] + angles[j])
                self.assertAlmostEqual(
                    turtle.angle_error(combined.dir_n(), compare.dir_n()), 0.0
                )
                self.assertAlmostEqual(
                    turtle.angle_error(combined.dir_u(), compare.dir_u()), 0.0
                )
                self.assertAlmostEqual(
                    turtle.angle_error(combined.dir_u(), compare.dir_n()), 0.0
                )

                combined = x.combine(y.negate())
                compare = turtle.DirectionAngle(angles[i] - angles[j])
                self.assertAlmostEqual(
                    turtle.angle_error(combined.dir_n(), compare.dir_n()), 0.0
                )
                self.assertAlmostEqual(
                    turtle.angle_error(combined.dir_u(), compare.dir_u()), 0.0
                )
                self.assertAlmostEqual(
                    turtle.angle_error(combined.dir_u(), compare.dir_n()), 0.0
                )
                with self.assertRaises(ValueError):
                    turtle.make_parallel(x, y)

        for x in to_test:
            with self.assertRaises(ValueError):
                blah = x.combine(turtle.Direction())
            with self.assertRaises(ValueError):
                blah = turtle.Direction().combine(x)
            turtle.make_parallel(x, turtle.Direction())

        u1 = turtle.Direction()
        u2 = turtle.Direction()
        self.assertTrue(u1.combine(u2).__class__ is turtle.Direction)
        self.assertTrue(u1.negate().__class__ is turtle.Direction)

    def test_make_direction(self):
        a = turtle.make_direction_from_user_params(0.5, math.sqrt(0.75), 0)
        b = turtle.make_direction_from_user_params(60, None, math.pi / 180.0)
        c = turtle.make_direction_from_user_params(
            (0.5, math.sqrt(0.75)), None, math.pi / 180.0
        )
        self.assertAlmostEqual(turtle.angle_error(a.dir_n(), b.dir_n()), 0.0)
        self.assertAlmostEqual(turtle.angle_error(a.dir_n(), c.dir_n()), 0.0)
        self.assertIs(
            turtle.make_direction_from_user_params(
                turtle.Direction(), None, None
            ).__class__,
            turtle.Direction,
        )

    def test_unspecified_dir(self):
        # To provide coverage
        with turtle.Turtle() as t:
            t.direction = turtle.Direction()
            t.forward(1)
            self.assertFalse(t.direction.known())
            self.assertIsInstance(t.direction, turtle.DirectionDiff)

    def test_make_parallel(self):

        vec1 = cs.var(turtle.angle_to_dir(1.2))
        (vec1[0] ** 2 + vec1[1] ** 2 - 1.0).make_zero()
        vec2 = cs.var(turtle.angle_to_dir(1.7))
        (vec2[0] ** 2 + vec2[1] ** 2 - 3.0).make_zero()
        vec3 = cs.var(turtle.angle_to_dir(2))
        (vec3[0] ** 2 + vec3[1] ** 2 - 2.0).make_zero()
        to_test = (
            turtle.DirectionAngle(cs.var(1)),
            turtle.DirectionNormalized(vec1),
            turtle.DirectionUnnormalized(vec2),
            turtle.DirectionDiff(((2, 3), (2 + vec3[0], 3 + vec3[1]))),
        )

        # also test decoupling
        a2 = turtle.decouple_value(to_test[0].angle)

        for i, x in enumerate(to_test):
            self.assertFalse(x.known())
            for j, y in enumerate(to_test):
                turtle.make_parallel(x, y)

        for i, x in enumerate(to_test):
            for j, y in enumerate(to_test):
                self.assertAlmostEqual(
                    turtle.angle_error(*cs.solve(x.dir_n(), y.dir_n())), 0.0
                )
        self.assertAlmostEqual(cs.solve(a2), cs.solve(to_test[0].angle))
        a3 = turtle.decouple_value(to_test[0].angle)
        self.assertAlmostEqual(cs.solve(a3), cs.solve(to_test[0].angle))

    def test_decouple_value(self):
        self.assertEqual(turtle.decouple_value(1.23), 1.23)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
