"""
Turtle sketching example - hull of two circles

name: turtle_sketching.py
by:   Dmytry Lavrov
date: August 2025
desc:
    Demo of turtle - inspired construction
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
from solve123d import *
from solve123d.turtle import *
import build123d
import ocp_vscode


cs.set_verbose(True)
# cs.set_opportunistic(True)

r1 = 10
r2 = 5
h = 20
with Turtle() as t:
    pen_up()
    # we don't know where the bottom arc starts, but we know it's somewhere around -r1,0
    # so we provide initial guess of -180 degrees here
    heading(var(-180))
    forward(r1)
    left(
        90
    )  # the tangent of the arc is 90 degrees to the left of where radius is pointing
    start_tangent = t.heading_vector
    start_point = t.position
    pen_down()
    # Likewise, initial guess of 180 for the arc angle
    left(var(180), turn_radius=r1)
    forward(var(h))
    arc2_c = left(var(180), turn_radius=r2).center
    arc2_c[0].magic = 0
    arc2_c[1].magic = h
    forward()
    closing_constraint(tangency=True)
line = t.to_build123d()
face = build123d.make_face(line)
a = face.area
print(a)
# face = build123d.make_face(line)
ocp_vscode.show(line)
