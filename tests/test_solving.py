"""
test solving

name: test_solving.py
by:   Dmytry Lavrov
date: August 2025
desc:
    This unit test tests solve123d constraint solver and higher-order functions
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

import unittest
import math
import solve123d as cs
import jax


def unwrap(f):
    return f.__wrapped__


class SolvingTest(unittest.TestCase):

    def test_conversion(self):
        a=cs.var(0)
        a.solution=jax.numpy.array(1.0)
        b=a.solution_as_float_or_none()
        c=a.solve()
        self.assertEqual(b, 1.0)
        self.assertEqual(c, 1.0)

    def test_scalar_sumconstraint(self):
        a = cs.var(0)
        b = 2
        c = 3
        cs.sum_constraint(a, b, c)
        self.assertAlmostEqual(a.s + b - c, 0)

    def test_coincident(self):
        a = cs.var(5, 5)
        b = (1, 2)
        cs.coincident(a, b)
        self.assertAlmostEqual(a[0].s, 1)
        self.assertAlmostEqual(a[1].s, 2)

        c = 1
        d = cs.var(2)
        self.assertTrue(d.solution_as_float_or_none() is None)
        cs.coincident(c, d)
        self.assertAlmostEqual(d.s, c)
        # test difficult to trigger branches
        d.solution = float(d.solution)
        self.assertAlmostEqual(d.s, c)
        self.assertAlmostEqual(d.solution_as_float_or_none(), c)

    def test_wrapped_var(self):
        a = cs.var(1)
        wf = cs.WrappedFunction(a)
        cs.magic.zero = wf + 2
        self.assertAlmostEqual(a.s, -2)
        self.assertEqual(cs.magic.zero, 0.0)

    def test_constraints(self):
        aa = [cs.Variable(1), cs.Variable(1.1)]
        bb = cs.var((2, 2.2))
        cc = [cs.var(2.5), 2.66]
        cs.sum_constraint(aa, bb, cc)
        x = aa[0].solve()

        self.assertAlmostEqual(aa[0] + bb[0] - cc[0], 0.0)
        self.assertAlmostEqual(aa[1] + bb[1] - cc[1], 0.0)

        triangle_a = [0.0, 0.0]
        triangle_b = [1.0, 0.0]
        triangle_c = cs.var([0.5, 0.5])

        point_p = cs.var(0.71, 0.32)
        side_length = cs.var(0.5)

        cs.distance_constraint(triangle_a, triangle_b, side_length)
        cs.distance_constraint(triangle_b, triangle_c, side_length)
        cs.distance_constraint(triangle_c, triangle_a, side_length)

        # Place a circle in the corner, tangent to both sides
        cs.line_left_distance_to_point_2d_constraint(
            [triangle_a, triangle_b], point_p, 0.1
        )
        cs.line_left_distance_to_point_2d_constraint(
            [triangle_a, triangle_c], point_p, -0.1
        )

        self.assertAlmostEqual(triangle_c[0].s, 0.5)
        self.assertAlmostEqual(triangle_c[1].s, math.sqrt(0.75))

        self.assertAlmostEqual(point_p[1].s, 0.1)
        self.assertAlmostEqual(point_p[0].s, 0.1 / math.tan(math.radians(30)))

    def test_magic(self):
        a = cs.Variable(1.2345)
        b = cs.var(1.2345)
        cs.magic.zero = 2.0 - a * 3.0 + b
        cs.magic.zero = a - b * 2.0 + 1.0

        test = cs.solve(2.0 - a * 3.0 + b, a - b * 2.0 + 1.0)

        #self.assertTrue(isinstance(test[0], float))


        self.assertAlmostEqual(test[0], 0)
        self.assertAlmostEqual(test[1], 0)
        self.assertAlmostEqual(2.0 - a.s * 3.0 + b.s, 0)
        self.assertAlmostEqual(a.s - b.s * 2.0 + 1.0, 0)

    def test_overconstrained(self):
        a = cs.Variable(1.2345)
        b = cs.var(1.2345)
        cs.magic.zero = 2.0 - a * 3.0 + b
        cs.magic.zero = a - b * 2.0 + 1.0
        cs.magic.zero = a - b * 1.0 + 1.0

        test = cs.solve(2.0 - a * 3.0 + b, a - b * 2.0 + 1.0)
        print(test)

    def test_presolve_magic(self):
        a = cs.Variable(1.2345)
        b = cs.var(1.2345)
        cs.magic.zero = 2.0 - a * 3.0 + b
        cs.magic.zero = a - b * 2.0 + 1.0

        a_val=a.solve()

        test = cs.solve(2.0 - a * 3.0 + b, a - b * 2.0 + 1.0)

        #self.assertTrue(isinstance(test[0], float))


        self.assertAlmostEqual(test[0], 0)
        self.assertAlmostEqual(test[1], 0)
        self.assertAlmostEqual(2.0 - a.s * 3.0 + b.s, 0)
        self.assertAlmostEqual(a.s - b.s * 2.0 + 1.0, 0)

    def test_dovetail(self):
        r = 20
        d = 5
        w = 2
        pt1 = cs.var(0, r)
        pt2 = cs.var(0, r)
        cs.distance_constraint(pt1, (0, 0), r)
        cs.distance_constraint(pt2, (0, 0), r + d)
        cs.line_left_distance_to_point_2d_constraint(((0, 0), (0, 1)), pt1, -w / 2)
        cs.angle_2d_constraint(((0, 0), (0, 1)), (pt1, pt2), -30 * math.pi / 180.0)

        para = unwrap(cs.parallel_2d_constraint)

        dist=unwrap(cs.distance_constraint)(cs.solve(pt1), (0, 0), 0)

        dist=cs.unjax(dist)
        self.assertIsInstance(dist, float)
        dist=cs.unjax(dist)
        self.assertIsInstance(dist, float)

        self.assertAlmostEqual(
            dist, r
        )
        self.assertAlmostEqual(
            unwrap(cs.distance_constraint)(cs.solve(pt2), (0, 0), 0), r + d
        )
        self.assertAlmostEqual(
            unwrap(cs.line_point_distance_constraint)(((0, 0), (0, 1)), cs.solve(pt1)),
            w / 2,
        )
        l1 = ((0, 0), (0, 1))
        l2 = (cs.solve(pt1), cs.solve(pt2))
        self.assertAlmostEqual(
            unwrap(cs.angle_2d_constraint)(l1, l2, -30 * math.pi / 180.0), 0.0
        )

        self.assertAlmostEqual(
            para(
                (cs.solve(pt1), cs.solve(pt2)),
                (cs.solve(0.0, 0.0), (0.5, math.sqrt(0.75))),
            ),
            0,
        )

        lcp = unwrap(cs.line_contains_point_2d_constraint)
        self.assertAlmostEqual(lcp(l2, (pt1[0].s + 0.5, pt1[1].s + math.sqrt(0.75))), 0)

    def test_dovetail_redundant_constraint(self):
        r = 20
        d = 5
        w = 2
        pt1 = cs.var(0, r)
        pt2 = cs.var(0, r)
        cs.distance_constraint(pt1, (0, 0), r)
        cs.distance_constraint(pt2, (0, 0), r + d)
        cs.line_left_distance_to_point_2d_constraint(((0, 0), (0, 1)), pt1, -w / 2)
        cs.magic.zero=cs.line_pt_dist(((0, 0), (0, 1)), pt1) + w/2
        
        cs.angle_2d_constraint(((0, 0), (0, 1)), (pt1, pt2), -30 * math.pi / 180.0)

        para = unwrap(cs.parallel_2d_constraint)

        dist=unwrap(cs.distance_constraint)(cs.solve(pt1), (0, 0), 0)

        dist=cs.unjax(dist)
        self.assertIsInstance(dist, float)
        dist=cs.unjax(dist)
        self.assertIsInstance(dist, float)

        self.assertAlmostEqual(
            dist, r
        )
        self.assertAlmostEqual(
            unwrap(cs.distance_constraint)(cs.solve(pt2), (0, 0), 0), r + d
        )
        self.assertAlmostEqual(
            unwrap(cs.line_point_distance_constraint)(((0, 0), (0, 1)), cs.solve(pt1)),
            w / 2,
        )
        l1 = ((0, 0), (0, 1))
        l2 = (cs.solve(pt1), cs.solve(pt2))
        self.assertAlmostEqual(
            unwrap(cs.angle_2d_constraint)(l1, l2, -30 * math.pi / 180.0), 0.0
        )

        self.assertAlmostEqual(
            para(
                (cs.solve(pt1), cs.solve(pt2)),
                (cs.solve(0.0, 0.0), (0.5, math.sqrt(0.75))),
            ),
            0,
        )

        lcp = unwrap(cs.line_contains_point_2d_constraint)
        self.assertAlmostEqual(lcp(l2, (pt1[0].s + 0.5, pt1[1].s + math.sqrt(0.75))), 0)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
