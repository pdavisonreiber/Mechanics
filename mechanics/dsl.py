from __future__ import annotations

from dataclasses import dataclass, field
import ast
import inspect
from types import FrameType
from typing import Iterable

import sympy as sp


g = sp.Symbol("g", positive=True)
AXES = ("horizontal", "vertical")
_UNKNOWN_SYMBOLS: set[sp.Symbol] = set()
_UNKNOWN_UNITS: dict[sp.Symbol, str] = {}
_UNKNOWN_SYMBOL_IDS: set[int] = set()
_UNKNOWN_UNITS_BY_ID: dict[int, str] = {}
_PARTICLE_COUNT = 0
_UNKNOWN_COUNT = 0


class Solution(dict[sp.Symbol, sp.Expr]):
    def __init__(
        self,
        *args: object,
        significant_figures: int | None = None,
        **kwargs: object,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.significant_figures = significant_figures

    def __repr__(self) -> str:
        if not self:
            return "No unknowns to solve."
        return "\n".join(
            f"{_display_expr(symbol)} = {_solution_value(symbol, value, self.significant_figures)}"
            for symbol, value in self.items()
        )

    __str__ = __repr__

    def _repr_pretty_(self, printer: object, cycle: bool) -> None:
        printer.text("..." if cycle else repr(self))

    def _repr_latex_(self) -> str:
        if not self:
            return r"$\text{No unknowns to solve.}$"
        lines = [
            rf"{sp.latex(symbol)} &= {_latex_solution_value(symbol, value, self.significant_figures)}"
            for symbol, value in self.items()
        ]
        return r"$\begin{aligned}" + r"\\".join(lines) + r"\end{aligned}$"


class EquationDisplay:
    def __init__(self, equations: Iterable[sp.Eq], solution: Solution | list[Solution] | None = None) -> None:
        self.equations = list(equations)
        self.solution = solution

    def __repr__(self) -> str:
        lines = [sp.pretty(equation) for equation in self.equations]
        if self.solution is not None:
            lines.append("Solution:")
            lines.append(str(self.solution))
        return "\n".join(lines)

    __str__ = __repr__

    def _repr_pretty_(self, printer: object, cycle: bool) -> None:
        printer.text("..." if cycle else repr(self))

    def _repr_latex_(self) -> str:
        if not self.equations and self.solution is None:
            return r"$\text{No equations.}$"

        lines = [
            rf"{_equation_latex(equation.lhs)} &= {_equation_latex(equation.rhs)}"
            for equation in self.equations
        ]
        if self.solution is not None:
            lines.append(r"\text{Solution:}")
            if isinstance(self.solution, list):
                for index, solution in enumerate(self.solution, start=1):
                    lines.append(rf"\text{{Solution {index}:}}")
                    lines.extend(_latex_solution_lines(solution))
            else:
                lines.extend(_latex_solution_lines(self.solution))
        return r"$\begin{aligned}" + r"\\".join(lines) + r"\end{aligned}$"


def _infer_lhs_name() -> str | None:
    frame = inspect.currentframe()
    if frame is None:
        return None

    caller = frame
    internal_names = {"_infer_lhs_name", "_new_unknown", "unknown_force", "unknown_acceleration", "particle"}
    while caller.f_back is not None and caller.f_code.co_name in internal_names:
        caller = caller.f_back

    info = inspect.getframeinfo(caller, context=1)
    if not info.code_context:
        return None

    try:
        tree = ast.parse(info.code_context[0].strip())
    except SyntaxError:
        return None

    if not tree.body or not isinstance(tree.body[0], ast.Assign):
        return None

    target = tree.body[0].targets[0]
    if isinstance(target, ast.Name):
        return target.id
    return None


def _pick_axis(long_name: str, long_value: object, short_name: str, short_value: object, *, allow_none: bool = False) -> object:
    default = None if allow_none else 0
    long_given = long_value is not default
    short_given = short_value is not default
    if long_given and short_given:
        raise TypeError(f"Specify either {long_name}= or {short_name}=, not both")
    return short_value if short_given else long_value


def _sympify(value: object) -> sp.Expr:
    if isinstance(value, float):
        return sp.Float(str(value))
    return sp.sympify(value)


def _new_unknown(prefix: str, unit: str) -> sp.Symbol:
    global _UNKNOWN_COUNT
    symbol_name = _infer_lhs_name()
    if symbol_name is None:
        _UNKNOWN_COUNT += 1
        symbol_name = f"{prefix}_{_UNKNOWN_COUNT}"

    symbol = sp.Symbol(symbol_name, mechanics_unknown=True)
    _UNKNOWN_SYMBOLS.add(symbol)
    _UNKNOWN_UNITS[symbol] = unit
    _UNKNOWN_SYMBOL_IDS.add(id(symbol))
    _UNKNOWN_UNITS_BY_ID[id(symbol)] = unit
    return symbol


def unknown_force() -> sp.Symbol:
    return _new_unknown("F", "N")


def unknown_acceleration() -> sp.Symbol:
    return _new_unknown("a", "m s^-2")


@dataclass(frozen=True)
class Force:
    horizontal: sp.Expr = sp.Integer(0)
    vertical: sp.Expr = sp.Integer(0)
    label: str | None = None

    def components(self) -> dict[str, sp.Expr]:
        return {"horizontal": self.horizontal, "vertical": self.vertical}


@dataclass
class Particle:
    mass: sp.Expr
    name: str
    forces: list[Force] = field(default_factory=list)
    accelerations: dict[str, sp.Expr] = field(default_factory=dict)

    def force(
        self,
        *,
        horizontal: object = 0,
        vertical: object = 0,
        x: object = 0,
        y: object = 0,
        label: str | None = None,
    ) -> Force:
        horizontal = _pick_axis("horizontal", horizontal, "x", x)
        vertical = _pick_axis("vertical", vertical, "y", y)
        force = Force(_sympify(horizontal), _sympify(vertical), label)
        self.forces.append(force)
        return force

    def acceleration(
        self,
        *,
        horizontal: object | None = None,
        vertical: object | None = None,
        x: object | None = None,
        y: object | None = None,
    ) -> None:
        horizontal = _pick_axis("horizontal", horizontal, "x", x, allow_none=True)
        vertical = _pick_axis("vertical", vertical, "y", y, allow_none=True)
        if horizontal is not None:
            self.accelerations["horizontal"] = _sympify(horizontal)
        if vertical is not None:
            self.accelerations["vertical"] = _sympify(vertical)

    def resultant_force(self, axis: str) -> sp.Expr:
        return sp.simplify(sum(force.components()[axis] for force in self.forces))


def particle(*, mass: object, name: str | None = None) -> Particle:
    global _PARTICLE_COUNT
    particle_name = name or _infer_lhs_name()
    if particle_name is None:
        _PARTICLE_COUNT += 1
        particle_name = f"P{_PARTICLE_COUNT}"
    return Particle(mass=_sympify(mass), name=particle_name)


@dataclass
class System:
    particles: tuple[Particle, ...]
    accelerations: dict[str, sp.Expr] = field(default_factory=dict)
    built_equations: list[sp.Eq] = field(default_factory=list)
    solution: Solution | list[Solution] | None = None

    def acceleration(
        self,
        *,
        horizontal: object | None = None,
        vertical: object | None = None,
        x: object | None = None,
        y: object | None = None,
    ) -> None:
        horizontal = _pick_axis("horizontal", horizontal, "x", x, allow_none=True)
        vertical = _pick_axis("vertical", vertical, "y", y, allow_none=True)
        if horizontal is not None:
            self.accelerations["horizontal"] = _sympify(horizontal)
        if vertical is not None:
            self.accelerations["vertical"] = _sympify(vertical)
        self._bind_names_from_call_stack()
        self.solution = None
        self.build_equations()

    def build_equations(self) -> list[sp.Eq]:
        equations: list[sp.Eq] = []
        for particle in self.particles:
            for axis in AXES:
                if not self._has_axis_information(particle, axis):
                    continue
                acceleration = particle.accelerations.get(axis, self.accelerations.get(axis, sp.Integer(0)))
                equations.append(sp.Eq(particle.resultant_force(axis), _second_law_rhs(particle.mass, acceleration), evaluate=False))

        self.built_equations = equations
        return equations

    def solve(self, *unknowns: sp.Symbol) -> Solution | list[Solution]:
        return self._solve(*unknowns, significant_figures=3)

    def solve_exact(self, *unknowns: sp.Symbol) -> Solution | list[Solution]:
        return self._solve(*unknowns, significant_figures=None)

    def _solve(
        self,
        *unknowns: sp.Symbol,
        significant_figures: int | None,
    ) -> Solution | list[Solution]:
        substitutions = self._bind_names_from_call_stack()
        equations = self.build_equations()
        if not unknowns:
            unknowns = tuple(sorted(self._unknowns_in(equations), key=str))
        unknowns = tuple(substitutions.get(unknown, unknown) for unknown in unknowns)

        unknown_set = set(unknowns)
        equations = [equation for equation in equations if equation.free_symbols & unknown_set]
        if not unknowns:
            self.solution = Solution(significant_figures=significant_figures)
            return self.solution

        solved = sp.solve(equations, unknowns, dict=True)
        solutions = [
            Solution(solution, significant_figures=significant_figures)
            for solution in solved
        ]
        self.solution = solutions[0] if len(solutions) == 1 else solutions
        return self.solution

    def equations(self) -> EquationDisplay:
        self._bind_names_from_call_stack()
        equations = self.build_equations()
        return EquationDisplay(equations, self.solution)

    def build_diagrams(self) -> str:
        self._bind_names_from_call_stack()
        blocks = [
            _particle_diagram(
                particle,
                particle.accelerations,
            )
            for particle in self.particles
        ]
        return _arrange_blocks(blocks, self.accelerations)

    def diagrams(self) -> None:
        print(self.build_diagrams())

    def _has_axis_information(self, particle: Particle, axis: str) -> bool:
        has_force = any(force.components()[axis] != 0 for force in particle.forces)
        has_acceleration = axis in particle.accelerations
        return has_force or has_acceleration

    @staticmethod
    def _unknowns_in(equations: Iterable[sp.Eq]) -> set[sp.Symbol]:
        symbols: set[sp.Symbol] = set()
        for equation in equations:
            symbols.update(equation.free_symbols)
        return {symbol for symbol in symbols if _is_unknown(symbol)}

    def _bind_names_from_call_stack(self) -> dict[sp.Symbol, sp.Symbol]:
        substitutions: dict[sp.Symbol, sp.Symbol] = {}
        frame = inspect.currentframe()
        if frame is None:
            return substitutions

        caller = frame.f_back
        while caller is not None:
            if caller.f_code.co_filename != __file__:
                substitutions.update(self._bind_names_from_frame(caller))
                if substitutions:
                    self._replace_symbols(substitutions)
                return substitutions
            caller = caller.f_back
        return substitutions

    def _bind_names_from_frame(self, frame: FrameType) -> dict[sp.Symbol, sp.Symbol]:
        substitutions: dict[sp.Symbol, sp.Symbol] = {}
        namespace = {**frame.f_globals, **frame.f_locals}

        for name, value in namespace.items():
            if not name.isidentifier() or name.startswith("_"):
                continue

            for particle in self.particles:
                if value is particle and particle.name.startswith("P"):
                    particle.name = name

            if isinstance(value, sp.Symbol) and _is_unknown(value) and str(value) != name:
                replacement = sp.Symbol(name, mechanics_unknown=True)
                substitutions[value] = replacement

        return substitutions

    def _replace_symbols(self, substitutions: dict[sp.Symbol, sp.Symbol]) -> None:
        if not substitutions:
            return

        for new in substitutions.values():
            _UNKNOWN_SYMBOLS.add(new)
            _UNKNOWN_SYMBOL_IDS.add(id(new))
        for old, new in substitutions.items():
            if old in _UNKNOWN_UNITS:
                _UNKNOWN_UNITS[new] = _UNKNOWN_UNITS[old]
            unit = _unknown_unit(old)
            if unit is not None:
                _UNKNOWN_UNITS_BY_ID[id(new)] = unit

        for particle in self.particles:
            particle.mass = particle.mass.xreplace(substitutions)
            particle.accelerations = {
                axis: value.xreplace(substitutions)
                for axis, value in particle.accelerations.items()
            }
            particle.forces = [
                Force(
                    force.horizontal.xreplace(substitutions),
                    force.vertical.xreplace(substitutions),
                    force.label,
                )
                for force in particle.forces
            ]

        self.accelerations = {
            axis: value.xreplace(substitutions)
            for axis, value in self.accelerations.items()
        }
        self.built_equations = [equation.xreplace(substitutions) for equation in self.built_equations]


def system(*particles: Particle) -> System:
    return System(tuple(particles))


def _second_law_rhs(mass: sp.Expr, acceleration: sp.Expr) -> sp.Expr:
    if mass.is_number and acceleration.is_number:
        return sp.Mul(mass, acceleration, evaluate=False)
    return mass * acceleration


def _particle_diagram(particle: Particle, accelerations: dict[str, sp.Expr]) -> list[str]:
    forces = _directional_force_labels(particle)
    acceleration_labels = _directional_acceleration_labels(accelerations)
    left = forces["left"]
    right = forces["right"]
    up = forces["up"]
    down = forces["down"]
    acceleration_left = acceleration_labels["left"]
    acceleration_right = acceleration_labels["right"]
    acceleration_up = acceleration_labels["up"]
    acceleration_down = acceleration_labels["down"]

    mass_label = f"{_diagram_expr(particle.mass)} kg"
    box_inner_width = max(len(particle.name) + 2, len(mass_label) + 2, 9)
    box_width = box_inner_width + 2
    left_width = max([len(label) + 2 for label in left + acceleration_left] + [0])
    right_width = max([len(label) + 2 for label in right + acceleration_right] + [8])
    gap = 0
    total_width = left_width + gap + box_width + gap + right_width
    center_start = left_width + gap
    center_width = box_width

    rows: list[str] = []
    rows.extend(_vertical_rows(acceleration_up, total_width, center_start, center_width, "↟"))
    rows.extend(_vertical_rows(up, total_width, center_start, center_width, "↑"))

    rows.append(_with_sides("", _box_line(box_inner_width, "top"), "", left_width, right_width, gap))
    left_side = [(label, "←") for label in left] + [(label, "↞") for label in acceleration_left]
    right_side = [("→", label) for label in right] + [("↠", label) for label in acceleration_right]
    side_rows = max(len(left_side), len(right_side), 2)
    for index in range(side_rows):
        left_label = f"{left_side[index][0]} {left_side[index][1]}" if index < len(left_side) else ""
        right_label = f"{right_side[index][0]} {right_side[index][1]}" if index < len(right_side) else ""
        if index == 0:
            content = particle.name
        elif index == 1:
            content = mass_label
        else:
            content = ""
        rows.append(
            _with_sides(
                left_label,
                _box_line(box_inner_width, content),
                right_label,
                left_width,
                right_width,
                gap,
            )
        )

    rows.append(_with_sides("", _box_line(box_inner_width, "bottom"), "", left_width, right_width, gap))
    rows.extend(_vertical_rows(down, total_width, center_start, center_width, "↓"))
    rows.extend(_vertical_rows(acceleration_down, total_width, center_start, center_width, "↡"))
    return rows


def _directional_force_labels(particle: Particle) -> dict[str, list[str]]:
    labels = {"left": [], "right": [], "up": [], "down": []}
    for force in particle.forces:
        if force.horizontal != 0:
            direction = "left" if force.horizontal.could_extract_minus_sign() else "right"
            labels[direction].append(_force_component_label(force, force.horizontal))
        if force.vertical != 0:
            direction = "down" if force.vertical.could_extract_minus_sign() else "up"
            labels[direction].append(_force_component_label(force, force.vertical))
    return labels


def _directional_acceleration_labels(accelerations: dict[str, sp.Expr]) -> dict[str, list[str]]:
    labels = {"left": [], "right": [], "up": [], "down": []}
    horizontal = accelerations.get("horizontal")
    if horizontal is not None and horizontal != 0:
        direction = "left" if horizontal.could_extract_minus_sign() else "right"
        labels[direction].append(_acceleration_label(horizontal))

    vertical = accelerations.get("vertical")
    if vertical is not None and vertical != 0:
        direction = "down" if vertical.could_extract_minus_sign() else "up"
        labels[direction].append(_acceleration_label(vertical))
    return labels


def _acceleration_label(acceleration: sp.Expr) -> str:
    magnitude = -acceleration if acceleration.could_extract_minus_sign() else acceleration
    simplified = sp.simplify(magnitude)
    if isinstance(simplified, sp.Symbol) and _unknown_unit(simplified) == "m s^-2":
        return _diagram_expr(simplified)
    return f"{_diagram_expr(simplified)} m s^-2"


def _force_component_label(force: Force, component: sp.Expr) -> str:
    magnitude = -component if component.could_extract_minus_sign() else component
    simplified = sp.simplify(magnitude)
    if isinstance(simplified, sp.Symbol) and _is_unknown(simplified):
        return _diagram_expr(simplified)
    return f"{_diagram_expr(simplified)} N"


def _diagram_expr(value: sp.Expr) -> str:
    return _display_expr(value)


def _display_expr(value: sp.Expr) -> str:
    if value.is_Float:
        return f"{float(value):.12g}"
    return sp.sstr(value).replace("*", "")


def _equation_latex(value: sp.Expr) -> str:
    if isinstance(value, sp.Mul) and _is_explicit_numeric_product(value):
        return sp.latex(value, mul_symbol=r"\times ")
    return sp.latex(value)


def _is_explicit_numeric_product(value: sp.Expr) -> bool:
    return (
        isinstance(value, sp.Mul)
        and len(value.args) == 2
        and all(arg.is_number for arg in value.args)
    )


def _is_unknown(symbol: sp.Symbol) -> bool:
    return bool(symbol.assumptions0.get("mechanics_unknown")) and id(symbol) in _UNKNOWN_SYMBOL_IDS


def _unknown_unit(symbol: sp.Symbol) -> str | None:
    return _UNKNOWN_UNITS_BY_ID.get(id(symbol), _UNKNOWN_UNITS.get(symbol))


def _solution_value(
    symbol: sp.Symbol,
    value: sp.Expr,
    significant_figures: int | None,
) -> str:
    unit = _unknown_unit(symbol)
    expression = _display_solution_expr(value, significant_figures)
    if unit is None:
        return expression
    if significant_figures is None and isinstance(value, sp.Add):
        expression = f"({expression})"
    return f"{expression} {unit}"


def _latex_solution_value(
    symbol: sp.Symbol,
    value: sp.Expr,
    significant_figures: int | None,
) -> str:
    unit = _unknown_unit(symbol)
    rounded = _round_numeric_expr(value, significant_figures)
    expression = sp.latex(rounded) if isinstance(rounded, sp.Expr) else rounded
    if unit is None:
        return expression
    if significant_figures is None and isinstance(value, sp.Add):
        expression = rf"\left({expression}\right)"
    return rf"{expression}\,{_latex_unit(unit)}"


def _latex_solution_lines(solution: Solution) -> list[str]:
    return [
        rf"{sp.latex(symbol)} &= {_latex_solution_value(symbol, value, solution.significant_figures)}"
        for symbol, value in solution.items()
    ]


def _latex_unit(unit: str) -> str:
    if unit == "N":
        return r"\mathrm{N}"
    if unit == "m s^-2":
        return r"\mathrm{m\,s^{-2}}"
    return rf"\mathrm{{{unit}}}"


def _display_solution_expr(value: sp.Expr, significant_figures: int | None) -> str:
    rounded = _round_numeric_expr(value, significant_figures)
    if isinstance(rounded, str):
        return rounded
    return _display_expr(rounded)


def _round_numeric_expr(value: sp.Expr, significant_figures: int | None) -> sp.Expr | str:
    if significant_figures is None:
        return value
    value = value.subs(g, sp.Float(9.8))
    if value.free_symbols:
        return value
    if not value.is_number:
        return value
    return f"{float(value):.{significant_figures}g}"


def _vertical_rows(
    labels: list[str],
    total_width: int,
    center_start: int,
    center_width: int,
    arrow: str,
) -> list[str]:
    rows = []
    for label in labels:
        text = f"{arrow} {label}"
        rows.append((" " * center_start + text.center(center_width)).ljust(total_width).rstrip())
    return rows


def _box_line(inner_width: int, content: str) -> str:
    if content == "top":
        return "┌" + "─" * inner_width + "┐"
    if content == "bottom":
        return "└" + "─" * inner_width + "┘"
    return "│" + content.center(inner_width) + "│"


def _with_sides(left: str, center: str, right: str, left_width: int, right_width: int, gap: int) -> str:
    return (
        left.rjust(left_width)
        + " " * gap
        + center
        + " " * gap
        + right.ljust(right_width)
    ).rstrip()


def _arrange_blocks(blocks: list[list[str]], accelerations: dict[str, sp.Expr] | None = None, spacing: int = 4) -> str:
    if not blocks:
        return ""

    box_top_rows = [_first_box_top_row(block) for block in blocks]
    target_box_top = max(box_top_rows)
    blocks = [
        [""] * (target_box_top - box_top_row) + block
        for block, box_top_row in zip(blocks, box_top_rows)
    ]

    if accelerations:
        acceleration_block = _system_acceleration_block(accelerations, target_box_top)
        blocks.append(acceleration_block)

    height = max(len(block) for block in blocks)
    widths = [max(len(line) for line in block) for block in blocks]

    padded_blocks = [
        block + [" " * width] * (height - len(block))
        for block, width in zip(blocks, widths)
    ]

    lines = []
    gap = " " * spacing
    for row in range(height):
        lines.append(gap.join(block[row].ljust(widths[index]) for index, block in enumerate(padded_blocks)))
    return "\n".join(lines)


def _first_box_top_row(block: list[str]) -> int:
    for index, line in enumerate(block):
        if "┌" in line:
            return index
    return 0


def _system_acceleration_block(accelerations: dict[str, sp.Expr], box_top_row: int) -> list[str]:
    labels = _directional_acceleration_labels(accelerations)
    lines = (
        [f"{label} ↞" for label in labels["left"]]
        + [f"↠ {label}" for label in labels["right"]]
        + [f"↟ {label}" for label in labels["up"]]
        + [f"↡ {label}" for label in labels["down"]]
    )
    if not lines:
        return [""]
    return [""] * (box_top_row + 1) + lines
