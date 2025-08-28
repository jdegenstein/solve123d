"""
Turtle sketching example

name: arc_tangent_to_arc_and_line.py
by:   Dmytry Lavrov
date: August 2025
desc:
    Just a simple shape from SourceCAD, with an arc that is tangent to another arc and a line
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
import solve123d as cs
from solve123d.turtle import *
from build123d import *
import ocp_vscode

# Outer line
cs.set_verbose(True)
cs.solver_settings.max_tolerance = 1e10
center_1 = (4, -1)
with Turtle() as t:
    # t.simplify_equations=False
    pen_up()
    heading(-90)
    forward(2 + 1)
    left(90)
    pen_down()
    forward(5 - 1)
    first_left=cs.var(45)
    first_left.name="first_left"
    left(first_left, turn_radius=2) # unknown 1
    # provide initial guess
    forward(cs.var(6)) # unknown 2
    arc = left(cs.var(90), turn_radius=1.4) # unknown 3
    (arc.center[0]-7.8).make_zero("custom - arc center x")
    (arc.center[1]-4.8).make_zero("custom - arc center y")
    first_right=cs.var(45)
    first_right.name="first_right"
    right(first_right, turn_radius=3) #unknown 4
    (t.y - (2 + 1)).make_zero("custom - penultimate y")
    forward() # unknown 5
    (t.x - 0).make_zero("custom - end x")
    (t.y- (2 + 1)).make_zero("custom - end y")
line = t.to_build123d()
# line += Rot(0, 0, 180) * line
ocp_vscode.show(line, [*t.to_build123d_list(ignore_errors=True, debug_objects=True)])
