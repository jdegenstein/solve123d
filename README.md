# Extremely basic geometric constraint solver

Meant for use with build123d. The project is in very early stage and constraints api is likely to change a lot. 

# Installation

pip install git+https://gitlab.com/dmytrylk/solve123d.git

# Basic idea

You create a set of variables and impose constraints on them, then call solve() to obtain a solution (if multiple solutions exist you may end up with not the solution you're looking for). The first solve() call solves all connected variables.

For example, you could construct an equilateral triangle like this:
```py
import solve123d as cs

triangle_a = (0.0, 0.0)
triangle_b = (1.0, 0.0)
triangle_c = cs.var(0.5, 0.5) # equivalent to (cs.var(0.5), cs.var(0.5))
side_length = cs.var(0.5)
cs.distance_constraint(triangle_a, triangle_b, side_length)
cs.distance_constraint(triangle_b, triangle_c, side_length)
cs.distance_constraint(triangle_c, triangle_a, side_length)
c = cs.solve(triangle_c)
```
c will be equal to (0.5, math.sqrt(0.75))

# Solving equations

It is also able to solve systems of linear and non linear equations like
```py
import solve123d as cs
a = cs.var(1.2345)
b = cs.var(1.2345)
cs.magic.zero = 2.0 - a * 3.0 + b
cs.magic.zero = a - b * 2.0 + 1.0
```

Afterwards, you can use a.s or solve(a) to obtain the value.

# TODO
Some documentation

# License

Apache 2.0, see LICENSE file