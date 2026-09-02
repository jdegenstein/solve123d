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

import collections
import copy
from enum import Enum
import math
from warnings import warn
import numpy as np
import build123d
import solve123d as cs
from typing import TypeAlias

FloatLike: TypeAlias = (
    cs.Variable | cs.WrappedFunction | float | int | cs.Dual | np.ndarray
)


SCALE_FOR_ANGLE_PARALELISM_CONSTRAINTS = 0.1


class Primitive:
    pass


class Line(Primitive):
    def __init__(self, p1, p2):
        self.points = (p1, p2)

    def debug_print(self):
        print(
            f"line {cs.solve(cs.solve(self.points[0]))}->{cs.solve(cs.solve(self.points[1]))}"
        )


class TArc(Primitive):
    def __init__(self, start_point, tangent_at_start, end_point, center, radius):
        self.start_point = start_point
        self.tangent_at_start = tangent_at_start
        self.end_point = end_point
        self.center = center
        self.radius = radius

    def debug_print(self):
        start_point = (cs.solve(cs.solve(self.start_point)),)
        end_point = (cs.solve(cs.solve(self.end_point)),)
        tangent = cs.solve(cs.solve(self.tangent_at_start))
        print(f"arc {start_point}_{tangent}->{end_point}")


class TurnDir(Enum):
    AUTO = 0
    LEFT = 1
    RIGHT = 2


def rotate(a, b):
    return (a[0] * b[0] - a[1] * b[1], a[0] * b[1] + a[1] * b[0])


def conjugate(a):
    return (a[0], -a[1])


def add(a, b):
    return (a[0] + b[0], a[1] + b[1])


def sub(a, b):
    return (a[0] - b[0], a[1] - b[1])


def norm(a):
    return cs.make_wrapper(cs.d_hypot)(a[0], a[1]) + 1e-25


def vec_scale(a, s):
    return (a[0] * s, a[1] * s)


def normalized(a):
    return vec_scale(a, 1.0 / norm(a))


def all_values(*args):
    for a in cs.recursive_unpack(args):
        if isinstance(a, (cs.WrappedFunction, cs.Variable)):
            return False
    return True


def decouple_value(v):
    if all_values(v):
        return v
    if isinstance(v, cs.Variable) and v.solution is not None:
        return v.solution
    result = cs.var(v.initial_value)
    result.name = "decoupling variable"
    cs.magic.zero = v - result
    return result


# Use constraint_solver Dual implementations
wrapped_abs = cs.make_wrapper(cs.d_abs)
wrapped_atan2 = cs.make_wrapper(cs.d_atan2)
wrapped_sin = cs.make_wrapper(cs.d_sin)
wrapped_cos = cs.make_wrapper(cs.d_cos)


def angle_to_dir(a):
    return wrapped_cos(a), wrapped_sin(a)


def solver_sincos(a):
    return wrapped_sin(a), wrapped_cos(a)


def normalize_angle(a):
    return ((a + math.pi) % (2.0 * math.pi)) - math.pi


wrapped_normalize_angle = cs.make_wrapper(normalize_angle)


def make_non_zero(a, eps=1e-20):
    val = a.val if isinstance(a, cs.Dual) else float(a)
    sign = 1.0 if val >= 0 else -1.0
    return a + eps * sign


wrapped_make_non_zero = cs.make_wrapper(make_non_zero)


def wrapped_safe_atan2(y, x):
    return wrapped_atan2(y, wrapped_make_non_zero(x))


def angle_error(dir1, dir2, angle=float(0.0)):
    ddir = rotate(dir2, conjugate(dir1))
    alpha = wrapped_atan2(cs.make_wrapper(make_non_zero)(ddir[1]), ddir[0])
    if isinstance(angle, (float, int)) and angle == 0.0:
        return alpha
    return wrapped_normalize_angle(alpha - angle)


class Direction:
    _ORDER = 0

    def known(self):
        assert self.__class__ == Direction
        return False

    def negate(self):
        assert self.__class__ == Direction
        return self

    def combine(self, other):
        if self._ORDER <= other._ORDER:
            return self._combine(other)
        return other._combine(self)

    def _combine(self, other):
        if other._ORDER > 0:
            raise ValueError(
                "Combining unspecified direction with a specified direction is disallowed."
            )
        assert self.__class__ == Direction
        return self


class DirectionAngle(Direction):
    _ORDER = 1

    def __init__(self, a):
        self.angle = a

    def known(self):
        return all_values(self.angle)

    def negate(self):
        return DirectionAngle(-self.angle)

    def dir_u(self):
        return self.dir_n()

    def dir_n(self):
        return angle_to_dir(self.angle)

    def _combine(self, other):
        if isinstance(other, DirectionAngle):
            return DirectionAngle(self.angle + other.angle)
        if other._ORDER > 1:
            return DirectionNormalized(self.dir_n())._combine(other)


class DirectionNormalized(Direction):
    _ORDER = 2

    def __init__(self, direction):
        self.direction = direction

    def known(self):
        return all_values(self.direction)

    def dir_u(self):
        return self.direction

    def dir_n(self):
        return self.direction

    def negate(self):
        return DirectionNormalized(conjugate(self.direction))

    def _combine(self, other):
        if isinstance(other, DirectionNormalized):
            return DirectionNormalized(rotate(self.direction, other.direction))
        if isinstance(other, (DirectionUnnormalized, DirectionDiff)):
            return DirectionUnnormalized(rotate(self.direction, other.dir_u()))


class DirectionUnnormalized(Direction):
    _ORDER = 3

    def __init__(self, direction):
        assert len(direction) == 2
        self.direction = direction

    def known(self):
        return all_values(self.direction)

    def dir_u(self):
        return self.direction

    def dir_n(self):
        return normalized(self.direction)

    def negate(self):
        return DirectionUnnormalized(conjugate(self.direction))

    def _combine(self, other):
        if isinstance(other, (DirectionUnnormalized, DirectionDiff)):
            return DirectionUnnormalized(rotate(self.direction, other.dir_u()))


class DirectionDiff(Direction):
    _ORDER = 4

    def __init__(self, points):
        self.points = points

    def known(self):
        return all_values(self.points)

    def dir_u(self):
        return sub(self.points[1], self.points[0])

    def dir_n(self):
        return normalized(self.dir_u())

    def negate(self):
        return DirectionDiff(
            (
                self.points[0],
                (self.points[1][0], self.points[0][1] * 2 - self.points[1][1]),
            )
        )

    def _combine(self, other):
        if isinstance(other, DirectionDiff):
            return DirectionUnnormalized(rotate(self.dir_u(), other.dir_u()))


AnyDirection: TypeAlias = (
    Direction
    | DirectionAngle
    | DirectionNormalized
    | DirectionUnnormalized
    | DirectionDiff
)


def make_direction_from_user_params(a, b, angle_scale):
    if b is None:
        if isinstance(a, Direction):
            return a
        elif isinstance(a, collections.abc.Sequence):
            return DirectionUnnormalized(a)
        return DirectionAngle(a * angle_scale)
    return DirectionUnnormalized((a, b))


def make_parallel(a: AnyDirection, b: AnyDirection, name=None):
    if a.__class__ == Direction or b.__class__ == Direction:
        return
    if a.known() and b.known():
        raise ValueError("Trying to constrain two known directions")

    if a._ORDER > b._ORDER:
        a, b = b, a
    if isinstance(a, DirectionAngle):
        if isinstance(b, DirectionAngle):
            wrapped_normalize_angle(a.angle - b.angle).make_zero("parallel angle angle")
        else:
            x, y = b.dir_u()
            wrapped_normalize_angle(a.angle - wrapped_safe_atan2(y, x)).make_zero(
                "parallel angle vector"
            )
    else:
        d = a.combine(b.negate())
        x, y = d.dir_u()
        wrapped_safe_atan2(y, x).make_zero(name or "parallel vector vector")


class Turtle:
    _turtle_stack = []
    simplify_equations = True
    no_reparametrize_hack = False

    @classmethod
    def top(cls) -> "Turtle":
        return cls._turtle_stack[-1]

    def __init__(self, use_stack=True):
        self.point_list = []
        self.primitive_list = []
        self.is_down = True
        self.direction = DirectionAngle(0)
        self.position = (0, 0)
        self.turn_radius = 0
        self._use_stack = use_stack
        self.angle_scale = math.pi / 180.0
        self.first_position = None
        self.first_direction = None

    def __enter__(self):
        if len(Turtle._turtle_stack) > 0 and self._use_stack:
            self.primitive_list = copy.copy(Turtle.top().primitive_list)
            self.is_down = Turtle.top().is_down
            self.direction = Turtle.top().direction
            self.position = Turtle.top().position
            self.turn_radius = Turtle.top().turn_radius
        Turtle._turtle_stack.append(self)
        return self

    def __exit__(self, type, value, traceback):
        Turtle._global_turtle = Turtle._turtle_stack[-1]
        del Turtle._turtle_stack[-1]

    def teleport(self, x_or_pos, y=None):
        if cs.is_iterable(x_or_pos):
            assert y is None
            self.position = x_or_pos
        else:
            self.position = (x_or_pos, y)

    def forward(self, dist=None, draw_line=None):
        dist_was_whatever = False
        if dist is None:
            dist = cs.absvar(1.239459234564)
            dist_was_whatever = True
        if self.direction.known():
            new_pos = add(self.position, vec_scale(self.direction.dir_n(), dist))
        else:
            if self.direction.__class__ == Direction:
                new_pos = cs.var(
                    add(
                        cs.get_initial_value(self.position),
                        (cs.get_initial_value(dist), 0.1234234651234),
                    )
                )
            else:
                new_pos = cs.var(
                    cs.get_initial_value(
                        add(self.position, vec_scale(self.direction.dir_n(), dist))
                    )
                )
            new_dir = DirectionDiff((self.position, new_pos))
            new_pos[0].name = "substituted position x"
            new_pos[1].name = "substituted position y"
            make_parallel(self.direction, new_dir)
            self.direction = new_dir
            if not dist_was_whatever:
                (norm(sub(new_pos, self.position)) - dist).make_zero(
                    "forward distance constraint"
                )

        if self.is_down and ((draw_line is None) or draw_line):
            self.point_list.append(self.position)
            self.primitive_list.append(Line(self.position, new_pos))
            if self.first_direction is None:
                self.first_direction = self.direction
                self.first_position = self.position
        result = (self.position, new_pos)
        self.position = new_pos
        return result

    def left(self, angle=None, *, turn_radius=None) -> TArc:
        if angle is None:
            angle = cs.var(178.1234123452345)
            angle.name = "left() unknown angle"
        new_direction = self.direction.combine(DirectionAngle(angle * self.angle_scale))
        return self.change_heading_to(
            new_direction, turn_radius=turn_radius, turn_dir=TurnDir.LEFT
        )

    def right(self, angle=None, *, turn_radius=None) -> TArc:
        if angle is None:
            angle = cs.var(178.1234123452345)
            angle.name = "right() unknown angle"
        new_direction = self.direction.combine(
            DirectionAngle(angle * (-self.angle_scale))
        )
        return self.change_heading_to(
            new_direction, turn_radius=turn_radius, turn_dir=TurnDir.RIGHT
        )

    def heading(
        self, angle_or_x, y=None, *, turn_radius=None, turn_dir: TurnDir = TurnDir.AUTO
    ) -> TArc:
        return self.change_heading_to(
            make_direction_from_user_params(angle_or_x, y, self.angle_scale),
            turn_radius=turn_radius,
            turn_dir=turn_dir,
        )

    def change_heading_to(
        self,
        new_direction: AnyDirection,
        *,
        turn_radius=None,
        turn_dir: TurnDir = TurnDir.AUTO,
    ) -> TArc:
        r = turn_radius if turn_radius is not None else self.turn_radius
        if (
            isinstance(r, (cs.WrappedFunction, cs.Variable, cs.Dual, np.ndarray))
            or r != 0
        ):
            if turn_dir == TurnDir.AUTO:
                if self.direction.known() and new_direction.known():
                    if new_direction.combine(self.direction.negate()).dir_u()[1] > 0:
                        turn_dir = TurnDir.LEFT
                    else:
                        turn_dir = TurnDir.RIGHT
                else:
                    raise RuntimeError(
                        "Arc turn direction cannot depend on solver variables. Use turn_dir."
                    )
            initial_dir = self.direction.dir_n()
            initial_pos = self.position
            if self.is_down:
                if self.first_direction is None:
                    self.first_direction = self.direction
                    self.first_position = self.position
            api_90_deg = 0.5 * math.pi / self.angle_scale
            if turn_dir == TurnDir.LEFT:
                self.left(api_90_deg, turn_radius=0)
                self.forward(r, draw_line=False)
                center = self.position
                self.change_heading_to(
                    new_direction.combine(DirectionAngle(-0.5 * math.pi)), turn_radius=0
                )
                self.forward(r, draw_line=False)
                left(api_90_deg, turn_radius=0)
            else:
                self.right(api_90_deg, turn_radius=0)
                self.forward(r, draw_line=False)
                center = self.position
                self.change_heading_to(
                    new_direction.combine(DirectionAngle(0.5 * math.pi)), turn_radius=0
                )
                self.forward(r, draw_line=False)
                self.right(api_90_deg, turn_radius=0)

            result = TArc(initial_pos, initial_dir, self.position, center, r)
            if self.is_down:
                self.primitive_list.append(result)
        else:
            result = TArc(
                self.position, self.direction.dir_n(), self.position, self.position, r
            )
            self.direction = new_direction
        return result

    def closing_constraint(self, tangency=False):
        applied_constraint = False
        x_diff = self.position[0] - self.first_position[0]
        if not all_values(x_diff):
            x_diff.make_zero()
            applied_constraint = True
        y_diff = self.position[1] - self.first_position[1]
        if not all_values(y_diff):
            y_diff.make_zero()
            applied_constraint = True
        if tangency:
            make_parallel(self.direction, self.first_direction)

    def debug_print_solution(self):
        for p in self.primitive_list:
            p.debug_print()

    def to_build123d_list(self, ignore_errors=False, debug_objects=False):
        for i, p in enumerate(self.primitive_list):
            if isinstance(p, Line):
                p0 = cs.solve(cs.solve(p.points[0]))
                p1 = cs.solve(cs.solve(p.points[1]))
                try:
                    line = build123d.Line(p0, p1)
                    line.name = f"l{i}"
                    yield line
                except Exception:
                    if not ignore_errors:
                        raise
            elif isinstance(p, TArc):
                try:
                    arc = build123d.TangentArc(
                        cs.solve(cs.solve(p.start_point)),
                        cs.solve(cs.solve(p.end_point)),
                        tangent=cs.solve(cs.solve(p.tangent_at_start)),
                    )
                    arc.name = f"a{i}"
                    yield arc
                    if debug_objects:
                        circle = build123d.CenterArc(
                            cs.solve(p.center),
                            cs.solve(p.radius),
                            start_angle=0,
                            arc_size=360,
                        )
                        circle.name = f"a{i} radius"
                        circle.color = "green"
                        yield circle
                        arc_start = build123d.CenterArc(
                            cs.solve(p.start_point), 0.1, start_angle=0, arc_size=360
                        )
                        arc_start.color = "red"
                        arc_start.name = f"a{i} start"
                        yield arc_start
                except Exception:
                    if not ignore_errors:
                        raise

    def to_build123d(self, ignore_errors=False):
        with build123d.BuildLine() as l:
            for i, p in enumerate(self.primitive_list):
                if isinstance(p, Line):
                    p0 = cs.solve(cs.solve(p.points[0]))
                    p1 = cs.solve(cs.solve(p.points[1]))
                    try:
                        build123d.Line(p0, p1).name = f"l{i}"
                    except Exception:
                        if not ignore_errors:
                            raise
                elif isinstance(p, TArc):
                    try:
                        build123d.TangentArc(
                            cs.solve(cs.solve(p.start_point)),
                            cs.solve(cs.solve(p.end_point)),
                            tangent=cs.solve(cs.solve(p.tangent_at_start)),
                        ).name = f"a{i}"
                    except Exception:
                        if not ignore_errors:
                            raise
        return l.line

    def pen_up(self):
        self.is_down = False

    def pen_down(self):
        self.is_down = True

    @property
    def x(self):
        return self.position[0]

    @x.setter
    def x(self, value):
        cs.magic.zero = self.position[0] - value
        if self.simplify_equations:
            self.position = (value, self.position[1])

    @property
    def y(self):
        return self.position[1]

    @y.setter
    def y(self, value):
        cs.magic.zero = self.position[1] - value
        if self.simplify_equations:
            self.position = (self.position[0], value)

    def be_at(self, x=None, y=None):
        if isinstance(x, collections.abc.Sequence):
            (self.position[0] - x[0]).make_zero("be_at x constraint")
            (self.position[1] - x[1]).make_zero("be_at y constraint")
            if self.simplify_equations:
                self.position = tuple(x)
            assert y is None
        else:
            if x is not None:
                (self.position[0] - x).make_zero("be_at x constraint")
            if y is not None:
                (self.position[1] - y).make_zero("be_at y constraint")
            if self.simplify_equations:
                if x is not None and y is not None:
                    self.position = (x, y)
                elif x is not None:
                    self.position = (x, self.position[1])
                elif y is not None:
                    self.position = (self.position[0], y)

    def be_heading(self, angle_or_x_or_dir, y=None, *, name="be_heading constraint"):
        new_direction = make_direction_from_user_params(
            angle_or_x_or_dir, y, self.angle_scale
        )
        make_parallel(new_direction, self.direction, name)
        if self.simplify_equations:
            self.direction = new_direction

    @property
    def heading_vector(self):
        return self.direction.dir_n()


def teleport(x_or_pos, y=None):
    Turtle.top().teleport(x_or_pos, y)


def forward(dist=None):
    return Turtle.top().forward(dist)


def left(angle=None, *, turn_radius=None) -> TArc:
    return Turtle.top().left(angle, turn_radius=turn_radius)


def right(angle=None, *, turn_radius=None) -> TArc:
    return Turtle.top().right(angle, turn_radius=turn_radius)


def heading(
    angle_or_x_or_dir, y=None, *, turn_radius=None, turn_dir: TurnDir = TurnDir.AUTO
) -> TArc:
    return Turtle.top().heading(
        angle_or_x_or_dir, y, turn_radius=turn_radius, turn_dir=turn_dir
    )


def be_at(x=None, y=None):
    return Turtle.top().be_at(x, y)


def be_heading(angle_or_x_or_dir=None, y=None):
    return Turtle.top().be_heading(angle_or_x_or_dir, y)


def closing_constraint(tangency=False):
    Turtle.top().closing_constraint(tangency)


def pen_down():
    Turtle.top().pen_down()


def pen_up():
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
    "be_at",
    "be_heading",
]
