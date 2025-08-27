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

import jax
import jax.numpy as jnp

#
# jax.config.update("jax_debug_nans", True)

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


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
