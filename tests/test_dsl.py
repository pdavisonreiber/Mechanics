import unittest

import sympy as sp

from mechanics import g, particle, system, unknown_acceleration, unknown_force


class MechanicsDslTests(unittest.TestCase):
    def test_unknown_constructors_do_not_accept_duplicate_names(self):
        with self.assertRaises(TypeError):
            unknown_force("T")

        with self.assertRaises(TypeError):
            unknown_acceleration("a")

    def test_solves_unknown_tension_from_horizontal_motion(self):
        A = particle(mass=5)
        T = unknown_force()

        A.force(horizontal=T)

        s = system(A)
        s.acceleration(horizontal=3)

        self.assertEqual(s.solve(T), {T: 15})
        self.assertEqual(str(s.solve(T)), "T = 15 N")
        self.assertEqual(str(s.solve_exact(T)), "T = 15 N")

    def test_builds_vertical_equation_with_unknown_acceleration(self):
        A = particle(mass=5)
        a = unknown_acceleration()

        A.force(vertical=5)
        A.force(vertical=-3 * g)

        s = system(A)
        s.acceleration(vertical=a)

        self.assertEqual(s.build_equations(), [sp.Eq(5 - 3 * g, 5 * a)])
        self.assertEqual(s.solve(a), {a: 1 - 3 * g / 5})
        self.assertEqual(str(s.solve(a)), "a = -4.88 m s^-2")
        self.assertEqual(str(s.solve_exact(a)), "a = (1 - 3g/5) m s^-2")

    def test_solve_rounds_numeric_answers_to_three_significant_figures(self):
        A = particle(mass=1)
        T = unknown_force()

        A.force(horizontal=T)

        s = system(A)
        s.acceleration(horizontal=sp.Rational(1, 3))

        self.assertEqual(s.solve(T), {T: sp.Rational(1, 3)})
        self.assertEqual(str(s.solve(T)), "T = 0.333 N")
        self.assertEqual(str(s.solve_exact(T)), "T = 1/3 N")

    def test_solve_without_parameters_solves_all_unknowns(self):
        A = particle(mass=5)
        B = particle(mass=2)
        T = unknown_force()
        a = unknown_acceleration()

        A.force(horizontal=T)
        B.force(horizontal=10)
        B.force(vertical=6)

        s = system(A, B)
        s.acceleration(horizontal=2, vertical=a)

        solution = s.solve()

        self.assertEqual(solution, {T: 10, a: 3})
        self.assertEqual(str(solution), "T = 10 N\na = 3 m s^-2")

    def test_solve_without_parameters_ignores_plain_sympy_symbols(self):
        T = unknown_force()
        plain_T = sp.Symbol(str(T))
        A = particle(mass=5)

        A.force(horizontal=plain_T)

        s = system(A)
        s.acceleration(horizontal=3)

        self.assertEqual(s.solve(), {})
        self.assertEqual(str(s.solve()), "No unknowns to solve.")

    def test_equations_returns_latex_display(self):
        A = particle(mass=5)
        a = unknown_acceleration()

        A.force(vertical=5)

        s = system(A)
        s.acceleration(vertical=a)

        equations = s.equations()

        self.assertEqual(str(equations), "5 = 5⋅a")
        self.assertEqual(equations._repr_latex_(), r"$\begin{aligned}5 &= 5 a\end{aligned}$")

    def test_equations_keeps_known_mass_times_acceleration_visible(self):
        A = particle(mass=0.4)
        A.force(vertical=5)

        s = system(A)
        s.acceleration(vertical=0.5)
        equations = s.equations()

        self.assertEqual(str(equations), "5 = 0.4⋅0.5")
        self.assertEqual(equations._repr_latex_(), r"$\begin{aligned}5 &= 0.4\times 0.5\end{aligned}$")

    def test_equation_latex_does_not_use_times_for_decimal_g_terms(self):
        A = particle(mass=0.4)
        R = unknown_force()

        A.force(vertical=R)
        A.force(vertical=-0.4 * g)

        s = system(A)
        s.acceleration(vertical=0.5)

        self.assertEqual(
            s.equations()._repr_latex_(),
            r"$\begin{aligned}R - 0.4 g &= 0.4\times 0.5\end{aligned}$",
        )

    def test_solution_latex_includes_units(self):
        A = particle(mass=1)
        T = unknown_force()

        A.force(horizontal=T)

        s = system(A)
        s.acceleration(horizontal=sp.Rational(1, 3))
        solution = s.solve(T)

        self.assertEqual(solution._repr_latex_(), r"$\begin{aligned}T &= 0.333\,\mathrm{N}\end{aligned}$")

    def test_solve_exact_latex_aligns_solutions(self):
        P = particle(mass=5)
        Q = particle(mass=3)
        T = unknown_force()
        a = unknown_acceleration()

        P.force(horizontal=40)
        P.force(horizontal=-10)
        P.force(horizontal=-T)
        Q.force(horizontal=T)
        Q.force(horizontal=-6)

        s = system(P, Q)
        s.acceleration(horizontal=a)

        self.assertEqual(
            s.solve_exact()._repr_latex_(),
            r"$\begin{aligned}T &= 15\,\mathrm{N}\\a &= 3\,\mathrm{m\,s^{-2}}\end{aligned}$",
        )

    def test_binds_fallback_unknown_name_from_calling_namespace(self):
        def make_acceleration():
            return unknown_acceleration()

        A = particle(mass=5)
        a = make_acceleration()

        A.force(vertical=5)

        s = system(A)
        s.acceleration(vertical=a)

        self.assertEqual(str(s.build_equations()[0]), "Eq(5, 5*a)")
        self.assertEqual(list(s.solve(a).values()), [1])
        self.assertEqual(str(s.solve(a)), "a = 1 m s^-2")

    def test_solves_requested_unknown_without_unrelated_known_equation(self):
        A = particle(mass=5)
        B = particle(mass=2)
        T = unknown_force()

        A.force(vertical=5)
        A.force(vertical=-3 * g)
        B.force(horizontal=-T)

        s = system(A, B)
        s.acceleration(horizontal=5)

        self.assertEqual(s.solve(T), {T: -10})

    def test_diagrams_returns_text(self):
        A = particle(mass=5)
        A.force(vertical=-5 * g, label="weight")

        diagram = system(A).build_diagrams()

        self.assertIn("│    A    │", diagram)
        self.assertIn("│   5 kg  │", diagram)
        self.assertIn("↓ 5g N", diagram)
        self.assertNotIn("weight", diagram)

    def test_diagrams_formats_decimal_masses_cleanly(self):
        A = particle(mass=0.4)
        A.force(vertical=-0.4 * g)

        diagram = system(A).build_diagrams()

        self.assertIn("│  0.4 kg │", diagram)
        self.assertIn("↓ 0.4g N", diagram)
        self.assertNotIn("0.400000", diagram)

    def test_diagrams_arranges_particles_horizontally_with_forces(self):
        A = particle(mass=5)
        B = particle(mass=2)
        T = unknown_force()
        a = unknown_acceleration()

        A.force(horizontal=T)
        A.force(vertical=-5 * g)
        B.force(horizontal=-T)
        B.force(vertical=3)

        s = system(A, B)
        s.acceleration(horizontal=a)
        diagram = s.build_diagrams()

        self.assertIn("│    A    │", diagram)
        self.assertIn("│   5 kg  │", diagram)
        self.assertIn("│    B    │", diagram)
        self.assertIn("│   2 kg  │", diagram)
        self.assertIn("→ T", diagram)
        self.assertIn("T ←", diagram)
        self.assertIn("↓ 5g N", diagram)
        self.assertIn("↑ 3 N", diagram)
        self.assertIn("↠ a", diagram)
        self.assertEqual(diagram.count("↠ a"), 1)
        box_top_lines = [line for line in diagram.splitlines() if "┌" in line]
        self.assertEqual(len(box_top_lines), 1)
        self.assertEqual(box_top_lines[0].count("┌"), 2)
        self.assertIsNone(s.diagrams())

    def test_diagrams_includes_vertical_numeric_acceleration(self):
        A = particle(mass=5)
        A.force(vertical=-5 * g)

        s = system(A)
        s.acceleration(vertical=0.5)
        diagram = s.build_diagrams()

        self.assertIn("↟ 0.5 m s^-2", diagram)
        self.assertEqual(diagram.count("↟ 0.5 m s^-2"), 1)


if __name__ == "__main__":
    unittest.main()
