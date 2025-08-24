"""
turtle

name: turtle.py
by:   Dmytry Lavrov
date: August 2025
desc:
    A turtle-graphics-meets-constraints builder for code cad.
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

import copy
from enum import Enum
import collections
import solve123d as cs
import jax
import jax.numpy as jnp
import build123d
import math

type FloatLike = cs.Variable | cs.WrappedFunction | float


class Primitive:
    pass


class Line(Primitive):
    def __init__(self, p1, p2):
        self.points = (p1, p2)


class TArc(Primitive):
    # Redundant information is useful for the constraint solver
    # Only start, tangent, and end_point are used to draw the arc
    def __init__(self, start_point, tangent_at_start, end_point, center, radius):
        self.start_point = start_point
        self.tangent_at_start = tangent_at_start
        self.end_point = end_point
        self.center = center
        self.radius = radius


class TurnDir(Enum):
    AUTO = 0
    LEFT = 1
    RIGHT = 2


# Doing turns with arcs:
# If direction is not given, see if directions are floats and use that
# If directions are variables (unknown), raise an error


# equivalent to complex product
def rotate(a, b):
    return (a[0] * b[0] - a[1] * b[1], a[0] * b[1] + a[1] * b[0])


def conjugate(a):
    return (a[0], -a[1])


def add(a, b):
    return (a[0] + b[0], a[1] + b[1])


def sub(a, b):
    return (a[0] - b[0], a[1] - b[1])


def all_values(*args):
    for a in cs.recursive_unpack(args):
        if isinstance(a, (cs.WrappedFunction, cs.Variable)):
            return False
    return True


class Turtle:
    """Turtle graphics + constraints for CAD sketching
    Intended usage:
    ```
    with Turtle() as t:
        A bunch of turtle operations using either global functions or t. methods
    ```
    """

    _turtle_stack = []
    """Global stack of nested `with Turtle as t:` contexts """

    simplify_equations = True
    """
    When using .x= and .y= constraint, set the value after setting constraint. 
    That avoids pile up of very deep expressions by consecutive Turtle operations
    In over constrained cases, that can cause gaps
    TODO: Sum sequences of moves
    TODO: opportunistically solve whenever a subset of equations is fully constrained.
    """

    @classmethod
    def top(cls):
        return cls._turtle_stack[-1]

    def __init__(self, use_stack=True):
        self.point_list = []
        self.primitive_list = []
        self.is_down = True
        self.heading_vector = (1, 0)
        self.position = (0, 0)
        self.turn_radius = 0
        self._use_stack = use_stack
        self.angle_scale = math.pi / 180.0
        self.first_position = None
        self.first_heading_vector = None

    def __enter__(self):
        print("Turtle start")
        if len(Turtle._turtle_stack) > 0 and self._use_stack:
            self.primitive_list = copy.copy(Turtle.top().primitive_list)
            self.is_down = Turtle.top().is_down
            self.heading_vector = Turtle.top().heading_vector
            self.position = Turtle.top().position
            self.turn_radius = Turtle.top().turn_radius
        Turtle._turtle_stack.append(self)
        return self

    def __exit__(self, type, value, traceback):
        print("Turtle end")
        Turtle._global_turtle = Turtle._turtle_stack[-1]
        del Turtle._turtle_stack[-1]

    # Does not make a line
    def teleport(self, x_or_pos, y=None):
        """Moves turtle to a point without creating a constraint or drawing a line"""
        if cs.is_iterable(x_or_pos):
            assert y is None
            self.position = x_or_pos
        else:
            self.position = (x_or_pos, y)

    def forward(
        self, dist
    ) -> tuple[tuple[FloatLike, FloatLike], tuple[FloatLike, FloatLike]]:
        """Move the turtle forward by dist
        Args:
            dist: Distance to move by
        Returns:
            A line as a tuple of two points
        """
        new_pos = (
            self.position[0] + self.heading_vector[0] * dist,
            self.position[1] + self.heading_vector[1] * dist,
        )
        if self.is_down:
            self.point_list.append(self.position)
            self.primitive_list.append(Line(self.position, new_pos))
            if self.first_heading_vector is None:
                self.first_heading_vector = self.heading_vector
                self.first_position = self.position
        result = (self.position, new_pos)
        self.position = new_pos
        return result

    def left(self, angle, *, turn_radius=None) -> TArc:
        """Turn the turtle left by angle
        Args:
            angle: Angle for the turn, scaled with self.angle_scale
            turn_radius Overrides self.corner radius
        Returns:
            An arc (possibly zero-radius) which you can use in constraints
        """
        c = cs.make_wrapper(jnp.cos)(self.angle_scale * angle)
        s = cs.make_wrapper(jnp.sin)(self.angle_scale * angle)
        return self.change_heading_to(
            rotate(self.heading_vector, (c, s)),
            turn_radius=turn_radius,
            turn_dir=TurnDir.LEFT,
        )

    def right(self, angle, *, turn_radius=None) -> TArc:
        """Turn the turtle right by angle
        Args:
            angle: Angle for the turn, scaled with self.angle_scale
            turn_radius Overrides self.corner radius
        Returns:
            An arc (possibly zero-radius) which you can use in constraints
        """
        c = cs.make_wrapper(jnp.cos)(-self.angle_scale * angle)
        s = cs.make_wrapper(jnp.sin)(-self.angle_scale * angle)
        return self.change_heading_to(
            rotate(self.heading_vector, (c, s)),
            turn_radius=turn_radius,
            turn_dir=TurnDir.RIGHT,
        )

    def heading(
        self,
        angle_or_x,
        y=None,
        *,
        turn_radius=None,
        turn_dir: TurnDir = TurnDir.AUTO,
    ) -> TArc:
        """Turn the turtle to point in a desired direction
        Args:
            angle_or_x: Angle for the turn, scaled with self.angle_scale, or a direction vector as a sequence, or x-component of direction
            y: y component of direction
            turn_radius Overrides self.turn_radius
            turn_dir Tells the turtle which way it should turn (which matters when corner radius is non zero)
        Returns:
            An arc (possibly zero-radius) which you can use in constraints
        Raises:
            Runtime error when turn_dir is not provided but is required due to potential dependence of turn direction on solver.
        """
        if y is None:
            if isinstance(angle_or_x, collections.abc.Sequence):
                scale = 1.0 / cs.make_wrapper(jnp.sqrt)(
                    angle_or_x[0] ** 2 + angle_or_x[1] ** 2
                )
                return self.change_heading_to(
                    (scale * angle_or_x[0], scale * angle_or_x[1]), turn_dir=turn_dir
                )
            else:
                c = cs.make_wrapper(jnp.cos)(self.angle_scale * angle_or_x)
                s = cs.make_wrapper(jnp.sin)(self.angle_scale * angle_or_x)
                return self.change_heading_to(
                    (c, s), turn_radius=turn_radius, turn_dir=turn_dir
                )
        else:
            scale = 1.0 / cs.make_wrapper(jnp.sqrt)(angle_or_x**2 + y**2)
            return self.change_heading_to(
                (scale * angle_or_x, scale * y),
                turn_radius=turn_radius,
                turn_dir=turn_dir,
            )

    def change_heading_to(
        self,
        new_heading_vector,
        *,
        turn_radius=None,
        turn_dir: TurnDir = TurnDir.AUTO,
    ) -> TArc:
        """Turn the turtle to point in a desired direction
        Args:
            new_heading_vector: New direction that the turtle must point in. Must be unit-length.
            turn_radius Overrides self.turn_radius
            turn_dir Tells the turtle which way it should turn (which matters when corner radius is non zero).
        Returns:
            An arc (possibly zero-radius) which you can use in constraints
        Raises:
            Runtime error when turn_dir is not provided but is required due to potential dependence of turn direction on solver.
        """
        r = turn_radius if turn_radius is not None else self.turn_radius
        if isinstance(r, (cs.WrappedFunction, cs.Variable, jax.Array)) or r != 0:
            if turn_dir == TurnDir.AUTO:
                if all_values(self.heading_vector, new_heading_vector):
                    delta = rotate(new_heading_vector, conjugate(self.heading_vector))
                    if delta[1] > 0:
                        turn_dir = TurnDir.LEFT
                    else:
                        turn_dir = TurnDir.RIGHT
                else:
                    raise RuntimeError(
                        "Arc turn direction can not depend on solver variables. Please use turn_dir parameter to specify if ark needs to be left or right."
                    )
            d = 1 if turn_dir == TurnDir.LEFT else -1
            center_offset = rotate(self.heading_vector, (0, d * r))
            center = add(self.position, center_offset)
            center_offset_new = rotate(new_heading_vector, (0, d * r))
            new_point = sub(center, center_offset_new)

            result = TArc(self.position, self.heading_vector, new_point, center, r)
            if self.is_down:
                self.primitive_list.append(result)
                if self.first_heading_vector is None:
                    self.first_heading_vector = self.heading_vector
                    self.first_position = self.position
            self.position = new_point
        else:  # turn_radius==0 , return zero radius arc for consistently
            result = TArc(
                self.position, self.heading_vector, self.position, self.position, r
            )
        self.heading_vector = new_heading_vector
        return result

    def closing_constraint(self, tangency=False):
        """Close the sketch by constraining the current point to the start point when the pen was first down
        Args:
            tangency: Whether to constrain tangency
        """
        applied_constraint = False
        if not all_values(self.position[0], self.first_position[0]):
            cs.magic.zero = self.position[0] - self.first_position[0]
            applied_constraint = True
        if not all_values(self.position[1], self.first_position[1]):
            cs.magic.zero = self.position[1] - self.first_position[1]
            applied_constraint = True
        # Tangency (if free enough)
        if tangency:
            if not all_values(self.heading_vector[0], self.first_heading_vector[0]):
                cs.magic.zero = self.heading_vector[0] - self.first_heading_vector[0]
                applied_constraint = True
            if not all_values(self.heading_vector[1], self.first_heading_vector[1]):
                cs.magic.zero = self.heading_vector[1] - self.first_heading_vector[1]
                applied_constraint = True
        if not applied_constraint:
            print(
                "Turtle warning: closing_constraint() does nothing (starting and ending points are constrained)"
            )

    def to_build123d(self):
        """Convert to a build123d line"""
        with build123d.BuildLine() as l:
            for p in self.primitive_list:
                if isinstance(p, Line):
                    p0 = cs.unjax(cs.solve(p.points[0]))
                    p1 = cs.unjax(cs.solve(p.points[1]))
                    build123d.Line(p0, p1)
                elif isinstance(p, TArc):
                    build123d.TangentArc(
                        cs.unjax(cs.solve(p.start_point)),
                        cs.unjax(cs.solve(p.end_point)),
                        tangent=cs.unjax(cs.solve(p.tangent_at_start)),
                    )
                else:  # pragma: no cover
                    raise RuntimeError("Unsupported primitive for build123d")

        return l.line

    def pen_up(self):
        """Disable appending primitives to primitive list"""
        self.is_down = False

    def pen_down(self):
        """Enable appending primitives to primitive list"""
        self.is_down = True

    @property
    def x(self):
        """Magical property, use .x=[some expression] to create a constraint on the x coordinate"""
        return self.position[0]

    @x.setter
    def x(self, value):
        cs.magic.zero = self.position[0] - value
        if self.simplify_equations:
            self.position = (value, self.position[1])
            # Does not work, it may be underconstrained
            # self.position=(cs.solve(self.position[0]), self.position[1])

    @property
    def y(self):
        """Magical property, use .y=[some expression] to create a constraint on the y coordinate"""
        return self.position[1]

    @y.setter
    def y(self, value):
        cs.magic.zero = self.position[1] - value
        if self.simplify_equations:
            self.position = (self.position[0], value)
            # Does not work, it may be underconstrained
            # self.position=(self.position[0], cs.solve(self.position[1]))


def teleport(x_or_pos, y=None):
    """Moves turtle to a point without creating a constraint or drawing a line"""
    Turtle.top().teleport(x_or_pos, y)


def forward(
    dist=None,
) -> tuple[tuple[FloatLike, FloatLike], tuple[FloatLike, FloatLike]]:
    """Move the turtle forward by dist
    Args:
        dist: Distance to move by. If None, will create a variable you can constrain later.
    Returns:
        A line as a tuple of two points
    """
    if dist is None:
        dist = cs.var(1)
    return Turtle.top().forward(dist)


def left(angle=None, *, turn_radius=None) -> TArc:
    """Turn the turtle left by angle
    Args:
        angle: Angle for the turn, scaled with self.angle_scale. If None, will create a variable you can constrain later.
        turn_radius Overrides self.corner radius
    Returns:
        An arc (possibly zero-radius) which you can use in constraints
    """
    if angle is None:
        angle = cs.var(1.0 / Turtle.top().angle_scale)
    return Turtle.top().left(angle, turn_radius=turn_radius)


def right(angle=None, *, turn_radius=None) -> TArc:
    """Turn the turtle right by angle
    Args:
        angle: Angle for the turn, scaled with self.angle_scale. If None, will create a variable you can constrain later.
        turn_radius Overrides self.corner radius
    Returns:
        An arc (possibly zero-radius) which you can use in constraints
    """
    if angle is None:
        angle = cs.var(1.0 / Turtle.top().angle_scale)
    return Turtle.top().right(angle, turn_radius=turn_radius)


def heading(
    angle_or_x_or_dir, y=None, *, turn_radius=None, turn_dir: TurnDir = TurnDir.AUTO
) -> TArc:
    """Turn the turtle to point in a desired direction
    Args:
        angle_or_x: Angle for the turn, scaled with self.angle_scale, or a direction vector as a sequence, or x-component of direction
        y: y component of direction
        turn_radius Overrides self.turn_radius
        turn_dir Tells the turtle which way it should turn (which matters when corner radius is non zero)
    Returns:
        An arc (possibly zero-radius) which you can use in constraints
    Raises:
        Runtime error when turn_dir is not provided but is required due to potential dependence of turn direction on solver.
    """
    return Turtle.top().heading(
        angle_or_x_or_dir, y, turn_radius=turn_radius, turn_dir=turn_dir
    )


def closing_constraint(tangency=False):
    """Close the sketch by constraining the current point to the start point when the pen was first down"""
    Turtle.top().closing_constraint(tangency)


def pen_down():
    """Enable appending primitives to primitive list"""
    Turtle.top().pen_down()


def pen_up():
    """Disable appending primitives to primitive list"""
    Turtle.top().pen_up()


__all__ = [
    "Turtle",
    "teleport",
    "forward",
    "left",
    "right",
    "heading",
    "closing_constraint",
    "pen_down",
    "pen_up",
]
