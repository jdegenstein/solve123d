import math
import solve123d as cs
from solve123d.turtle import *
from ocp_vscode import *


# with Turtle() as t:
#     heading(1, 0)
#     t.corner_radius=1
#     forward(10)
#     left(math.radians(120))
#     forward(10)
#     left(math.radians(120))
#     forward()
#     left(math.radians(120))
#     close()

# #print(pts)
# line=t.to_build123d()

with Turtle() as t:
    pen_up()
    heading(270 + 25)
    forward(33 - 10)
    pen_down()
    forward(10)
    left(90)
    forward()
    t.x = 33
    heading(90)
    forward()
    heading(180 - 35)
    forward()
    heading(90)
    forward(5)
    heading(180)
    forward(6)
    t.x = -8
    t.y = 120

    heading(-90)
    forward(22)
    heading(0)
    forward(6)
    heading(90)
    forward(5)
    heading(-40)
    forward()
    t.corner_radius = 13
    heading(-90)
    t.x = 33 - 10
    forward()
    heading(180 + (90 - 65))
    l = forward()
    t.x = 0
    t.y = 33
    t.corner_radius = 0
    heading(-90)
    forward()
    cs.magic.zero = cs.line_pt_dist(l, t.position) - 10
    heading(90 - 65)
    forward()
    t.corner_radius = 13
    heading(-90)
    t.x = 33 - 10
    forward()
    heading(180 + 25)
    forward()
    t.corner_radius = 0
    close()

line = t.to_build123d()
face = build123d.make_face(line)
show(line, build123d.Pos(0, 0, 1) * face)


"""
pen_up()
heading(-Y)
left(25)
forward(33-10)
pen_down()
forward(10)
left(90)
l0=forward()
magic.x=33
heading(+Y)
forward()
heading(180-35)
forward()
heading(+Y)
forward(5)
heading(-X)
forward(6)
magic.x=-8
magic.y=120
heading(-Y)
forward()
heading(+X)
forward(6)
heading(+Y)
forward(5)
heading(-40)
forward()
turn_radius(13)
heading(-Y)
magic.x=33-10
forward()
heading(90-65)
l1=forward()
turn_radius(0)
magic.y=33
heading(-Y)
forward()
heading(-l1)
distance(l1, 10)
turn_radius(13)
forward()
heading(-Y)
forward()
heading(-l0)
turn_radius(0)
close()
"""
