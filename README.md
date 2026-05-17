# Mechanics DSL

A small SymPy-backed DSL for modelling A-Level mechanics force diagrams and
Newton's second law equations.

```python
from mechanics import g, particle, system, unknown_acceleration, unknown_force

A = particle(mass=5)
B = particle(mass=2)
T = unknown_force()
a = unknown_acceleration()

A.force(vertical=5)
A.force(vertical=-3 * g)
B.force(horizontal=-T)

s = system(A, B)
s.acceleration(horizontal=5)
s.solve(T)
s.equations()
s.diagrams()
s.acceleration(vertical=a)
```

In Jupyter notebooks, `equations()`, `solve()`, and `solve_exact()` render
as LaTeX when they are the final expression in a cell. Use `print(...)` around
them if you want the plain-text representation instead.

`solve()` returns a dict-like solution object with cleaner display. It uses
`g = 9.8` where needed and rounds numeric answers to 3 significant figures by
default:

```python
s.solve()
```

```text
T = -10 N
a = -4.88 m s^-2
```

For a numeric value such as `1/3`, the display is rounded:

```text
T = 0.333 N
```

If you pass variables, only those are solved. If you pass no variables, only
unknowns created with `unknown_force()` and `unknown_acceleration()` are solved.
Ordinary SymPy symbols in equations are treated as parameters.

Use `solve_exact()` when you want exact SymPy values in the display:

```python
s.solve_exact()
```

```text
T = -10 N
a = (1 - 3g/5) m s^-2
```

`diagrams()` prints ASCII free-body diagrams by default:

```text
              ↑ R N
┌─────────┐
 T N ←│    P    │→ 40 N
10 N ←│   5 kg  │
└─────────┘
   ↓ 5g N
    ↓ F N
```

Use SVG output in notebooks with:

```python
s.diagrams(format="svg")
```

SVG diagrams require the `drawsvg` dependency.

The sign convention is explicit: positive `horizontal` and `vertical` values are
in the positive directions chosen for the problem. Use negative values for
opposing forces.

`unknown_force()` and `unknown_acceleration()` return SymPy symbols. They infer
the left-hand variable name in normal scripts, so `T = unknown_force()` creates
a symbol named `T`. In notebooks, if direct source-line inference is not
available, the system resolves the name from the notebook namespace when you use
the symbol in equations, solving, or diagrams.

```python
T = unknown_force()
a = unknown_acceleration()
```
