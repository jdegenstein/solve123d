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
center_1 = (4, -1)
with Turtle() as t:
    pen_up()
    heading(-90)
    forward(2 + 1)
    left(90)
    pen_down()
    forward(5 - 1)
    left(turn_radius=2)
    # provide initial guess
    forward(cs.var(6))
    arc = left(cs.var(90), turn_radius=1.4)
    arc.center[0].magic = 7.8
    arc.center[1].magic = 4.8
    right(turn_radius=3)
    t.y = 2 + 1
    forward()
    t.x = 0
    t.y = 2 + 1
line = t.to_build123d()
line += Rot(0, 0, 180) * line
ocp_vscode.show(line)
