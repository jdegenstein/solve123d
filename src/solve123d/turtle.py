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

type FloatLike = cs.Variable | cs.WrappedFunction | float | int | jax.Array

SCALE_FOR_ANGLE_PARALELISM_CONSTRAINTS = 0.1


class Primitive:
    pass


class Line(Primitive):
    def __init__(self, p1, p2):
        self.points = (p1, p2)

    def debug_print(self):
        """Prints values for debugging"""
        print(
            f"line {cs.unjax(cs.solve(self.points[0]))}->{cs.unjax(cs.solve(self.points[1]))}"
        )


class TArc(Primitive):
    # Redundant information is useful for the constraint solver
    # Only start, tangent, and end_point are used to draw the arc
    def __init__(self, start_point, tangent_at_start, end_point, center, radius):
        self.start_point = start_point
        self.tangent_at_start = tangent_at_start
        self.end_point = end_point
        self.center = center
        self.radius = radius

    def debug_print(self):
        """Prints values for debugging"""
        start_point = (cs.unjax(cs.solve(self.start_point)),)
        end_point = (cs.unjax(cs.solve(self.end_point)),)
        tangent = cs.unjax(cs.solve(self.tangent_at_start))
        print(f"arc {start_point}_{tangent}->{end_point}")


class TurnDir(Enum):
    """When creating an arc, the solver needs to know which way the arc should go
    (because that determines if the circle center is to the left or to the right)

    For constants, arc direction can be inferred, but for variables TurnDir is used as a parameter specifying arc direction.
    """

    AUTO = 0
    LEFT = 1
    RIGHT = 2


# equivalent to complex product
def rotate(a, b):
    return (a[0] * b[0] - a[1] * b[1], a[0] * b[1] + a[1] * b[0])


def conjugate(a):
    return (a[0], -a[1])


def add(a, b):
    return (a[0] + b[0], a[1] + b[1])


def sub(a, b):
    return (a[0] - b[0], a[1] - b[1])


def norm(a):
    return cs.make_wrapper(jnp.hypot)(a[0], a[1]) + 1e-25


def normalized(a):
    return (1.0 / norm(a)) * a


def vec_scale(a, s):
    return (a[0] * s, a[1] * s)


def all_values(*args):
    for a in cs.recursive_unpack(args):
        if isinstance(a, (cs.WrappedFunction, cs.Variable)):
            return False
    return True


@cs.make_wrapper
def parallel_metric(a, b):
    sa = 1.0 / jnp.hypot(a[0], a[1])
    sb = 1.0 / jnp.hypot(b[0], b[1])
    return jnp.hypot(a[0] * sa - b[0] * sb, a[1] * sa - b[1] * sb)


# Experimental attempt to add extra variables to allow it to hop across a circle. Does not work well.
def decouple_value(v):
    if all_values(v):
        return v
    if isinstance(v, cs.Variable) and v.solution is not None:
        return v.solution
    result = cs.var(v.initial_value)
    result.name = "decoupling variable"
    # result=cs.var(0)
    # TODO: figure out good value for the magic number here
    cs.magic.zero = v - result
    return result


wrapped_abs = cs.make_wrapper(jnp.abs)


wrapped_atan2 = cs.make_wrapper(jnp.atan2)

# Avoid using wrapped_sin and wrapped_cos at all costs - use sin and cos on regular values, and
# for SolverEntity find a way to express equivalent constraint with atan2
# Rationale: sin and cos appearing in constraint expressions are problematic for the solver
wrapped_sin = cs.make_wrapper(jnp.sin)
wrapped_cos = cs.make_wrapper(jnp.cos)


def decoupled_sin(v):
    return decouple_value(wrapped_sin(v))


def decoupled_cos(v):
    return decouple_value(wrapped_cos(v))


def angle_to_dir(a):
    return jnp.cos(a), jnp.sin(a)


def solver_sincos(a):
    return wrapped_sin(a), wrapped_cos(a)


def normalize_angle(a):
    return ((a + jnp.pi) % (2.0 * jnp.pi)) - jnp.pi


wrapped_normalize_angle = cs.make_wrapper(normalize_angle)


def make_non_zero(a, eps=1e-20):
    return a + eps * jnp.sign(a + eps / 2)


# angle is rotation from dir1 to dir2
def angle_error(dir1, dir2, angle=float(0.0)):
    ddir = rotate(dir2, conjugate(dir1))
    # is this where it fails?

    alpha = wrapped_atan2(cs.make_wrapper(make_non_zero)(ddir[1]), ddir[0])
    # Don't shortcircuit wrapped_normalized_angle if its one of jax types or anything else weird
    if isinstance(angle, (float, int)) and angle == 0.0:
        return alpha
    return wrapped_normalize_angle(alpha - angle)


class SolverDirectionVector:
    """
    Solver-friendly direction vector (as opposed to solver-unfriendly direction vector obtained with sin and cos of an angle)

    Purpose: avoid sin(a)*dist , cos(a)*dist like expressions (with unknown a and dist) from appearing in the equations being solved by the
    constraint solver, by representing unknown directions with non normalized vectors"""

    # TODO: maybe upgrade to a two-point representation, to avoid potentially long sums of unknown vectors?
    delta = (1, 0)
    constrained_to_a_circle = False
    original_angle_constraint = None
    original_angle_entity = None

    def __init__(self, angle, relative_to=(1, 0), make_constraint=False):
        if isinstance(relative_to, SolverDirectionVector):
            relative_to = relative_to.get_normalized_dir()
        if isinstance(angle, cs.SolverEntity):
            self.original_angle_entity = angle
            self.delta = cs.var(
                rotate(
                    angle_to_dir(angle.initial_value), cs.get_initial_value(relative_to)
                )
            )
            if make_constraint:
                self.original_angle_constraint = angle_error(
                    relative_to, self.delta, angle
                )
                self.original_angle_constraint.make_zero()
                # TODO: set up for constraint elision for unnecessary angles
        else:
            # TODO: fix or rethink, this is actually very problematic (relative_to may be unknown)
            self.delta = rotate(angle_to_dir(angle), relative_to)
            self.constrained_to_a_circle = True

    def be_on_circle(self, r):
        assert not self.constrained_to_a_circle
        circle_constraint = norm(self.delta) - r
        circle_constraint.name = "direction vector to circle constraint"
        circle_constraint.make_zero()
        self.constrained_to_a_circle = True

    def get_normalized_dir(self):
        s = 1.0 / norm(self.delta)
        return (s * self.delta[0], s * self.delta[1])

    def get_scaled_dir(self, r):
        if self.constrained_to_a_circle:
            return vec_scale(self.get_normalized_dir(), r)
        self.be_on_circle(r)
        return self.delta

    def be_parallel(self, other):
        if isinstance(other, SolverDirectionVector):
            other = other.delta
        constraint = (
            angle_error(self.delta, other) * SCALE_FOR_ANGLE_PARALELISM_CONSTRAINTS
        )
        constraint.name = "Parallel constraint"
        constraint.make_zero()


class Direction:
    """Base direction class. Also represents a direction we don't care about"""

    _ORDER = 0  # Used for re-ordering

    def known(self):
        """True if its all values, false otherwise"""
        assert self.__class__ == Direction
        return False

    def negate(self):
        """Negative angle (left vs right)"""
        assert self.__class__ == Direction
        return self

    def combine(self, other):
        """Works like addition of angles."""
        if self._ORDER <= other._ORDER:
            return self._combine(other)
        else:
            return other._combine(self)

    def _combine(self, other):
        """Every implementation can assume that self._ORDER<=other._ORDER"""
        if other._ORDER > 0:
            raise ValueError(
                "Combining unspecified direction with a specified direction is disallowed. "
                "Unspecified direction should only be used to signify that you don't know and don't care about the direction"
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
        return self.dir_n

    def dir_n(self):
        return angle_to_dir(self.angle)

    def _combine(self, other):
        if isinstance(other, DirectionAngle):
            return DirectionAngle(self.angle + other.angle)
        if other._ORDER > 1:
            return DirectionNormalized(self.dir_n)._combine(other)


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
        if isinstance(other, DirectionUnnormalized):
            return DirectionUnnormalized(rotate(self.direction, other.direction))
        if isinstance(other, DirectionDiff):
            return DirectionUnnormalized(rotate(self.direction, other.dir_u()))
        assert False


class DirectionUnnormalized(Direction):
    _ORDER = 3

    def __init__(self, dir):
        self.direction = dir

    def known(self):
        return all_values(self.direction)

    def dir_u(self):
        return self.direction

    def dir_n(self):
        return normalized(self.direction)

    def negate(self):
        return DirectionUnnormalized(conjugate(self.direction))

    def _combine(self, other):
        if isinstance(other, DirectionUnnormalized):
            print(
                "Product of two unnormalized directions. Solver might benefit from normalization"
            )
            return DirectionUnnormalized(rotate(self.direction, other.direction))
        if isinstance(other, DirectionDiff):
            print(
                "Product of unnormalized direction and difference direction. Solver might benefit from normalization"
            )
            return DirectionUnnormalized(rotate(self.direction, other.dir_u()))
        assert False

class DirectionDiff(Direction):
    _ORDER = 4

    def __init__(self, points):
        self.points = points

    def known(self):
        return all_values(self.points)

    def dir_u(self):
        return sub(self.points[1], self.points[0])

    def dir_n(self):
        return normalized(self.dir_u)

    def negate(self):
        return DirectionDiff(
            (
                self.points[0],
                (self.points[1][0], self.points[0][1] * 2 - self.points[1][1]),
            )
        )

    def _combine(self, other):
        if isinstance(other, DirectionDiff):
            print(
                "Product of difference direction and difference direction. Solver might benefit from normalization"
            )
            return DirectionUnnormalized(rotate(self.dir_u(), other.dir_u()))


# class DirectionKind(Enum):
#     NOT_SET=0 # Nobody specified the direction.
#     ANGLE=1
#     NORMALIZED=2
#     UNNORMALIZED=3
#     DIFFERENCE=4

# class Direction:
#     kind=DirectionKind.NOT_SET
#     def __init__(*, angle=None, dir=None, udir=None, pts=None):
#         if angle is not None:
#             self.kind=DirectionKind.ANGLE
#             self.angle=angle

#     @property
#     def is_known(self):


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
    no_reparametrize_hack = False
    """
    When using .x= and .y= constraint, set the value after setting constraint. 
    That avoids pile up of very deep expressions by consecutive Turtle operations
    In over constrained cases, that can cause gaps
    TODO: Sum sequences of moves
    TODO: opportunistically solve whenever a subset of equations is fully constrained.
    """

    @classmethod
    def top(cls) -> "Turtle":
        return cls._turtle_stack[-1]

    def __init__(self, use_stack=True):
        self.point_list = []
        self.primitive_list = []
        self.is_down = True

        # self.heading_vector = (1, 0)
        self._heading_vector = SolverDirectionVector(angle=0)

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
            self._heading_vector = Turtle.top()._heading_vector
            self.position = Turtle.top().position
            self.turn_radius = Turtle.top().turn_radius
        Turtle._turtle_stack.append(self)
        return self

    def __exit__(self, type, value, traceback):
        print("Turtle end")
        if not self._heading_vector.constrained_to_a_circle:
            self._heading_vector.be_on_circle(1)
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
        self, dist=None
    ) -> tuple[tuple[FloatLike, FloatLike], tuple[FloatLike, FloatLike]]:
        """Move the turtle forward by dist
        Args:
            dist: Distance to move by
        Returns:
            A line as a tuple of two points
        """
        if all_values(self._heading_vector.delta):
            if dist is None:
                # some value that is not 1 (to tiebreak solver, todo get values from a deterministic sequence)
                dist = cs.absvar(1.239459234564)
            new_pos = (
                self.position[0] + self._heading_vector.delta[0] * dist,
                self.position[1] + self._heading_vector.delta[1] * dist,
            )
        else:
            if self._heading_vector.constrained_to_a_circle:
                new_pos = cs.var(
                    cs.get_initial_value(
                        self.position[0] + self._heading_vector.delta[0] * dist,
                        self.position[1] + self._heading_vector.delta[1] * dist,
                    )
                )
                new_pos[0].name = "substituted position x"
                new_pos[1].name = "substituted position y"
                delta = sub(new_pos, self.position)

                self._heading_vector = SolverDirectionVector(0, delta, True)
                self._heading_vector.constrained_to_a_circle = False
                if dist is not None:
                    self._heading_vector.be_on_circle(dist)

                # refactored into SolverDirectionVector
                # a = angle_error(delta, self._heading_vector.delta)
                # a.name = "paralelism constraint"
                # a.make_zero()
                # self._heading_vector.delta=delta
                # if dist is not None:
                #     (norm(delta) - dist).make_zero()
                #     self._heading_vector.constrained_to_a_circle=True

            else:
                new_pos = (
                    self.position[0] + self._heading_vector.delta[0],
                    self.position[1] + self._heading_vector.delta[1],
                )

                delta_len = norm(self._heading_vector.delta)
                if dist is not None:
                    (delta_len - dist).make_zero()

                self._heading_vector.constrained_to_a_circle = True

            # new_pos = cs.var(
            #     cs.get_initial_value(
            #         self.position[0] + self.heading_vector[0] * dist,
            #         self.position[1] + self.heading_vector[1] * dist,
            #     )
            # )
            # new_pos[0].name="substituted position x"
            # new_pos[1].name="substituted position y"
            # delta = sub(new_pos, self.position)
            # delta_len = cs.make_wrapper(jnp.hypot)(delta[0], delta[1])

            # angle_constraint=parallel_metric(self.heading_vector, delta) * SCALE_FOR_ANGLE_PARALELISM_CONSTRAINTS
            # angle_constraint.make_zero()

            # # Ensure correct new heading
            # self.heading_vector=(delta[0]/delta_len, delta[1]/delta_len)

            # # Prepare elision of angle_constraint if angle is not used for anything
            # heading_angle_var=self._heading_angle
            # if isinstance(heading_angle_var, cs.WrappedFunction):
            #     if len(heading_angle_var.arguments) == 1 :
            #         heading_angle_var=heading_angle_var.arguments[0]
            #     else:
            #         print("Weird heading angle, no elision")
            # if isinstance(heading_angle_var, cs.Variable) :
            #     for v in angle_constraint.arguments :
            #         if v is self._heading_angle :
            #             def good_func(a):
            #                 return len(heading_angle_var.constraints)>1
            #             angle_constraint.good_func = good_func
            #             break

        if self.is_down:
            self.point_list.append(self.position)
            self.primitive_list.append(Line(self.position, new_pos))
            if self.first_heading_vector is None:
                self.first_heading_vector = self._heading_vector
                self.first_position = self.position
        result = (self.position, new_pos)
        self.position = new_pos
        return result

    def left(self, angle, *, turn_radius=None, angle_is_unimportant=False) -> TArc:
        """Turn the turtle left by angle
        Args:
            angle: Angle for the turn, scaled with self.angle_scale
            turn_radius Overrides self.corner radius
        Returns:
            An arc (possibly zero-radius) which you can use in constraints
        """

        new_heading = SolverDirectionVector(
            angle=angle * self.angle_scale,
            relative_to=self._heading_vector,
            make_constraint=not angle_is_unimportant,
        )

        return self.change_heading_to(
            new_heading,
            turn_radius=turn_radius,
            turn_dir=TurnDir.LEFT,
            angle_is_unimportant=angle_is_unimportant,
        )

    def right(self, angle, *, turn_radius=None, angle_is_unimportant=False) -> TArc:
        """Turn the turtle right by angle
        Args:
            angle: Angle for the turn, scaled with self.angle_scale
            turn_radius Overrides self.corner radius
        Returns:
            An arc (possibly zero-radius) which you can use in constraints
        """
        new_heading = SolverDirectionVector(
            angle=-angle * self.angle_scale,
            relative_to=self._heading_vector,
            make_constraint=not angle_is_unimportant,
        )
        return self.change_heading_to(
            new_heading,
            turn_radius=turn_radius,
            turn_dir=TurnDir.RIGHT,
            angle_is_unimportant=angle_is_unimportant,
        )

    def heading(
        self,
        angle_or_x,
        y=None,
        *,
        turn_radius=None,
        turn_dir: TurnDir = TurnDir.AUTO,
        angle_is_unimportant=False,
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
                new_heading = SolverDirectionVector(
                    angle=0,
                    relative_to=angle_or_x,
                    make_constraint=not angle_is_unimportant,
                )
            else:
                new_heading = SolverDirectionVector(
                    angle=angle_or_x * self.angle_scale,
                    make_constraint=not angle_is_unimportant,
                )
        else:
            new_heading = SolverDirectionVector(
                angle=0,
                relative_to=(angle_or_x, y),
                make_constraint=not angle_is_unimportant,
            )

        return self.change_heading_to(
            new_heading,
            turn_radius=turn_radius,
            turn_dir=turn_dir,
            angle_is_unimportant=angle_is_unimportant,
        )

    def change_heading_to(
        self,
        new_heading_vector: SolverDirectionVector,
        *,
        turn_radius=None,
        turn_dir: TurnDir = TurnDir.AUTO,
        angle_is_unimportant=False,
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
                if all_values(self._heading_vector.delta, new_heading_vector.delta):
                    delta = rotate(
                        new_heading_vector.delta, conjugate(self._heading_vector.delta)
                    )
                    if delta[1] > 0:
                        turn_dir = TurnDir.LEFT
                    else:
                        turn_dir = TurnDir.RIGHT
                else:
                    raise RuntimeError(
                        "Arc turn direction can not depend on solver variables. Please use turn_dir parameter to specify if ark needs to be left or right."
                    )
            d = 1 if turn_dir == TurnDir.LEFT else -1
            # scale = 1.0 / cs.make_wrapper(jnp.sqrt)(self.heading_vector[0]**2 + self.heading_vector[1]**2)

            center_offset = rotate(self._heading_vector.get_scaled_dir(r), (0, d))

            center = add(self.position, center_offset)
            # center2 = cs.var(cs.get_initial_value(center))
            # (center2[0]-center[0]).make_zero()
            # (center2[1]-center[1]).make_zero()
            # center=center2

            center_offset_new = rotate(new_heading_vector.get_scaled_dir(r), (0, d))
            new_point = sub(center, center_offset_new)

            result = TArc(
                self.position,
                self._heading_vector.get_normalized_dir(),
                new_point,
                center,
                r,
            )
            if self.is_down:
                self.primitive_list.append(result)
                if self.first_heading_vector is None:
                    self.first_heading_vector = self._heading_vector
                    self.first_position = self.position
            self.position = new_point
        else:  # turn_radius==0 , return zero radius arc for consistently
            result = TArc(
                self.position,
                self._heading_vector.get_normalized_dir(),
                self.position,
                self.position,
                r,
            )
        self._heading_vector = new_heading_vector
        return result

    def closing_constraint(self, tangency=False):
        """Close the sketch by constraining the current point to the start point when the pen was first down
        Args:
            tangency: Whether to constrain tangency
        """
        applied_constraint = False
        x_diff = self.position[0] - self.first_position[0]
        if not all_values(x_diff):
            x_diff.make_zero()
            applied_constraint = True
        y_diff = self.position[1] - self.first_position[1]
        if not all_values(y_diff):
            y_diff.make_zero()
            applied_constraint = True
        # Tangency (if free enough)
        if tangency:
            # parallel_metric(self.heading_vector, self.first_heading_vector).make_zero()
            self._heading_vector.be_parallel(self.first_heading_vector)
        if not applied_constraint:
            print(
                "Turtle warning: closing_constraint() does nothing (starting and ending points are constrained)"
            )

    def debug_print_solution(self):
        for p in self.primitive_list:
            p.debug_print()

    def to_build123d_list(self, ignore_errors=False, debug_objects=True):
        """Iterable list of build123d objects"""
        for i, p in enumerate(self.primitive_list):
            if isinstance(p, Line):
                p0 = cs.unjax(cs.solve(p.points[0]))
                p1 = cs.unjax(cs.solve(p.points[1]))
                try:
                    line = build123d.Line(p0, p1)
                    line.name = f"l{i}"
                    yield line
                except:
                    if not ignore_errors:
                        raise
            elif isinstance(p, TArc):
                try:
                    arc = build123d.TangentArc(
                        cs.unjax(cs.solve(p.start_point)),
                        cs.unjax(cs.solve(p.end_point)),
                        tangent=cs.unjax(cs.solve(p.tangent_at_start)),
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
                        # arc_arrow=build123d.Arrow(0.2, arc, shaft_width=0.05, head_at_start=True)
                        # arc_arrow.color='red'
                        arc_start = build123d.CenterArc(
                            cs.solve(p.start_point), 0.1, start_angle=0, arc_size=360
                        )
                        arc_start.color = "red"
                        arc_start.name = f"a{i} start"
                        yield arc_start
                except:
                    if not ignore_errors:
                        raise
            else:  # pragma: no cover
                raise RuntimeError("Unsupported primitive for build123d")

    def to_build123d(self, ignore_errors=False):
        """Convert to a build123d line"""
        with build123d.BuildLine() as l:
            for i, p in enumerate(self.primitive_list):
                if isinstance(p, Line):
                    p0 = cs.unjax(cs.solve(p.points[0]))
                    p1 = cs.unjax(cs.solve(p.points[1]))
                    try:
                        build123d.Line(p0, p1).name = f"l{i}"
                    except:
                        if not ignore_errors:
                            raise
                elif isinstance(p, TArc):
                    try:
                        build123d.TangentArc(
                            cs.unjax(cs.solve(p.start_point)),
                            cs.unjax(cs.solve(p.end_point)),
                            tangent=cs.unjax(cs.solve(p.tangent_at_start)),
                        ).name = f"a{i}"
                    except:
                        if not ignore_errors:
                            raise
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
        dist_non_abs = cs.var(1.1239452983467823465)
        dist_non_abs.name = "forward dist"
        dist = wrapped_abs(dist_non_abs)
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
        angle = cs.var(0.99 * math.pi / Turtle.top().angle_scale)
        angle.name = "left turn angle"
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
        angle = cs.var(0.99 * math.pi / Turtle.top().angle_scale)
        angle.name = "right turn angle"
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
