
import unittest
import numpy as np
import sys
import os
from unittest.mock import patch

# Ensure src is in path
sys.path.append(os.path.abspath('src'))

from mech390.physics import dynamics, kinematics, mass_properties
from mech390.datagen import generate, stage1_kinematic, stage2_embodiment
from mech390.config import get_baseline_config

class TestKinematicsAndDatagen(unittest.TestCase):

    @staticmethod
    def _sample_valid_2d_design():
        return {
            "r": 0.08,
            "l": 0.32,
            "e": 0.03,
            "ROM": 0.25,
            "QRR": 1.9,
            "theta_min": 0.1,
            "theta_max": 2.2,
        }
    
    def test_kinematic_feasibility(self):
        """Test valid and invalid geometry detection."""
        # Valid geometry: r=0.1, l=0.3, e=0.0
        # l > r + e -> 0.3 > 0.1 + 0 -> True
        metrics = kinematics.calculate_metrics(0.1, 0.3, 0.0)
        self.assertTrue(metrics['valid'], "Geometry 0.1, 0.3, 0.0 should be valid")
        
        # Invalid geometry: r=0.2, l=0.1, e=0.0 (crank bigger than rod)
        # Should fail or return invalid
        # calculate_metrics handles this by checking dead centers or sqrt
        # 0.1 < 0.2 + 0 -> Invalid
        try:
             metrics_inv = kinematics.calculate_metrics(0.2, 0.1, 0.0)
             self.assertFalse(metrics_inv['valid'], "Geometry 0.2, 0.1, 0.0 should be invalid")
        except ValueError:
             # If it raises, that's also acceptable "detection"
             pass

    def test_dead_center_detection(self):
        """Test that exactly 2 dead centers are found for valid geometry."""
        r, l, e = 0.1, 0.4, 0.05
        roots = kinematics.get_dead_center_angles(r, l, e)
        self.assertEqual(len(roots), 2, "Should find exactly 2 dead centers")
        
        # Check they are in [0, 2pi)
        for root in roots:
            self.assertTrue(0 <= root < 2*np.pi)

    def test_rom_target_solve(self):
        """Test that stage1 solver hits the target ROM, using r bounds from baseline.yaml."""
        config = get_baseline_config()
        r_range = config['geometry']['r']
        r_min = r_range['min']
        r_max = r_range['max']

        target_rom = config['operating']['ROM']
        l = 0.4
        e = 0.02

        # Solve for r using bounds from config
        r_sol = stage1_kinematic.solve_for_r_given_rom(l, e, target_rom, r_min=r_min, r_max=r_max)
        
        self.assertIsNotNone(r_sol, "Solver should find a solution for reasonable inputs")
        
        # Verify result
        metrics = kinematics.calculate_metrics(r_sol, l, e)
        self.assertTrue(metrics['valid'])
        self.assertAlmostEqual(metrics['ROM'], target_rom, places=4, 
                               msg=f"Solved ROM {metrics['ROM']} should match target {target_rom}")

    def test_qrr_computation(self):
        """Test QRR computation correctness."""
        # Centered slider-crank (e=0) should have QRR = 1.0
        r, l, e = 0.1, 0.4, 0.0
        metrics = kinematics.calculate_metrics(r, l, e)
        self.assertAlmostEqual(metrics['QRR'], 1.0, places=4, msg="QRR for e=0 should be 1.0")
        
        # Offset slider-crank (e > 0) should have QRR > 1.0 (usually)
        # Check geometry direction.
        # If offset is small positive, depending on rotation direction QRR != 1.
        r, l, e = 0.1, 0.4, 0.05
        metrics = kinematics.calculate_metrics(r, l, e)
        self.assertNotEqual(metrics['QRR'], 1.0, "QRR for offset mechanism should not be exactly 1.0")
        self.assertTrue(metrics['QRR'] > 0, "QRR should be positive")

    def test_stage1_constrained_candidate_sampling(self):
        """Constrained Stage-1 candidate sampler should enforce pre-feasibility inequalities."""
        config = get_baseline_config()
        target_rom = config['operating']['ROM']
        l_range = config['geometry']['l']
        e_range = config['geometry']['e']

        for method in ["random", "latin_hypercube"]:
            with self.subTest(method=method):
                candidates = stage1_kinematic._generate_constrained_stage1_candidates(
                    method=method,
                    n_samples=100,
                    seed=123,
                    l_range=l_range,
                    e_range=e_range,
                    target_rom=target_rom,
                    max_draws=2000,
                    strict_eps=1e-12,
                )

                self.assertGreater(len(candidates), 0, "Expected at least one feasible constrained candidate")

                for cand in candidates:
                    l = cand["l"]
                    e = cand["e"]

                    self.assertLess(abs(e), l, "Constrained sampler must enforce |e| < l")
                    self.assertLess(target_rom, 2.0 * l, "Constrained sampler must enforce S < 2l")
                    self.assertLess(
                        target_rom,
                        2.0 * np.sqrt(l**2 - e**2),
                        "Constrained sampler must enforce S < 2*sqrt(l^2-e^2)",
                    )

    def test_stage1_respects_sampling_constraints_from_config(self):
        """Stage-1 must enforce sampling.constraints expressions from config."""
        config = get_baseline_config()
        config['sampling']['method'] = 'random'
        config['sampling']['n_samples'] = 2000
        # Impossible with current baseline bounds.
        config['sampling']['constraints'] = ["l >= 100*r"]

        rows = stage1_kinematic.generate_valid_2d_mechanisms(config)
        self.assertEqual(len(rows), 0, "Expected zero valid designs for impossible constraints")

    def test_config_normalizes_scientific_notation(self):
        """Config loader should coerce scientific-notation strings to numeric values."""
        config = get_baseline_config()

        self.assertIsInstance(config["limits"]["sigma_allow"], float)
        self.assertAlmostEqual(config["limits"]["sigma_allow"], 180e6)

        self.assertIsInstance(config["material"]["yield_stress"]["min"], float)
        self.assertAlmostEqual(config["material"]["yield_stress"]["min"], 100e6)

    def test_stage2_generates_multiple_variants_with_constraints(self):
        """Stage-2 should produce N valid variants per 2D design and enforce width/pin constraints."""
        config = get_baseline_config()
        config["sampling"]["method"] = "random"
        config["sampling"]["n_variants_per_2d"] = 12
        config["sampling"]["stage2_max_attempts_per_2d"] = 3000

        rows = stage2_embodiment.expand_to_3d([self._sample_valid_2d_design()], config)
        self.assertEqual(len(rows), 12)

        width_r_range = config["geometry"]["widths"]["width_r"]
        width_l_range = config["geometry"]["widths"]["width_l"]
        thickness_r_range = config["geometry"]["thicknesses"]["thickness_r"]
        thickness_l_range = config["geometry"]["thicknesses"]["thickness_l"]
        pin_a_range = config["geometry"]["pin_diameters"]["pin_diameter_A"]
        pin_b_range = config["geometry"]["pin_diameters"]["pin_diameter_B"]
        pin_c_range = config["geometry"]["pin_diameters"]["pin_diameter_C"]

        for row in rows:
            self.assertEqual(row["r"], 0.08)
            self.assertEqual(row["l"], 0.32)
            self.assertEqual(row["e"], 0.03)

            self.assertTrue(width_r_range["min"] <= row["width_r"] <= width_r_range["max"])
            self.assertTrue(width_l_range["min"] <= row["width_l"] <= width_l_range["max"])
            self.assertTrue(thickness_r_range["min"] <= row["thickness_r"] <= thickness_r_range["max"])
            self.assertTrue(thickness_l_range["min"] <= row["thickness_l"] <= thickness_l_range["max"])
            self.assertTrue(pin_a_range["min"] <= row["pin_diameter_A"] <= pin_a_range["max"])
            self.assertTrue(pin_b_range["min"] <= row["pin_diameter_B"] <= pin_b_range["max"])
            self.assertTrue(pin_c_range["min"] <= row["pin_diameter_C"] <= pin_c_range["max"])

            self.assertGreater(row["width_r"], row["pin_diameter_A"])
            self.assertGreater(row["width_r"], row["pin_diameter_B"])
            self.assertGreater(row["width_l"], row["pin_diameter_B"])
            self.assertGreater(row["width_l"], row["pin_diameter_C"])

    def test_stage2_supports_legacy_flat_geometry_keys(self):
        """Stage-2 should accept legacy flat geometry keys when grouped keys are absent."""
        baseline = get_baseline_config()
        config = {
            "random_seed": 101,
            "sampling": {
                "method": "random",
                "n_variants_per_2d": 4,
                "stage2_max_attempts_per_2d": 500,
            },
            "geometry": {
                "width_r": baseline["geometry"]["widths"]["width_r"],
                "width_l": baseline["geometry"]["widths"]["width_l"],
                "thickness_r": baseline["geometry"]["thicknesses"]["thickness_r"],
                "thickness_l": baseline["geometry"]["thicknesses"]["thickness_l"],
                "pin_diameter_A": baseline["geometry"]["pin_diameters"]["pin_diameter_A"],
                "pin_diameter_B": baseline["geometry"]["pin_diameters"]["pin_diameter_B"],
                "pin_diameter_C": baseline["geometry"]["pin_diameters"]["pin_diameter_C"],
            },
        }

        rows = stage2_embodiment.expand_to_3d([self._sample_valid_2d_design()], config)
        self.assertEqual(len(rows), 4)

    def test_stage2_raises_on_impossible_constraints(self):
        """Stage-2 should fail clearly when configured ranges make constraints impossible."""
        config = get_baseline_config()
        config["sampling"]["method"] = "random"
        config["sampling"]["n_variants_per_2d"] = 3
        config["sampling"]["stage2_max_attempts_per_2d"] = 200
        config["geometry"]["widths"]["width_r"] = {"min": 0.01, "max": 0.01}
        config["geometry"]["widths"]["width_l"] = {"min": 0.01, "max": 0.01}
        config["geometry"]["pin_diameters"]["pin_diameter_A"] = {"min": 0.02, "max": 0.02}
        config["geometry"]["pin_diameters"]["pin_diameter_B"] = {"min": 0.02, "max": 0.02}
        config["geometry"]["pin_diameters"]["pin_diameter_C"] = {"min": 0.02, "max": 0.02}

        with self.assertRaises(ValueError):
            stage2_embodiment.expand_to_3d([self._sample_valid_2d_design()], config)

    def test_generate_dataset_uses_streaming_stage2_iterator(self):
        """Generate should consume Stage-2 iterator and report n_stage2 correctly."""
        mocked_stage2 = iter(
            [
                {
                    "r": 0.1,
                    "l": 0.4,
                    "e": 0.02,
                    "width_r": 0.02,
                    "width_l": 0.02,
                    "thickness_r": 0.015,
                    "thickness_l": 0.015,
                    "pin_diameter_A": 0.01,
                    "pin_diameter_B": 0.01,
                    "pin_diameter_C": 0.01,
                },
                {
                    "r": 0.1,
                    "l": 0.4,
                    "e": 0.02,
                    "width_r": 0.019,
                    "width_l": 0.019,
                    "thickness_r": 0.014,
                    "thickness_l": 0.014,
                    "pin_diameter_A": 0.01,
                    "pin_diameter_B": 0.01,
                    "pin_diameter_C": 0.01,
                },
            ]
        )

        config = {
            "random_seed": 11,
            "limits": {"sigma_allow": 180e6, "tau_allow": 100e6, "safety_factor": 1.0},
        }

        with patch(
            "mech390.datagen.generate.stage1_kinematic.generate_valid_2d_mechanisms",
            return_value=[{"r": 0.1, "l": 0.4, "e": 0.02}],
        ), patch(
            "mech390.datagen.generate.stage2_embodiment.iter_expand_to_3d",
            return_value=mocked_stage2,
        ):
            result = generate.generate_dataset(config)

        self.assertEqual(result.summary["n_stage1"], 1)
        self.assertEqual(result.summary["n_stage2"], 2)
        self.assertEqual(len(result.all_cases), 2)
        self.assertIn("pass_fail", result.all_cases.columns)

    def test_generate_dataset_injects_mu_from_config(self):
        """Dataset generation should inject operating.mu into each evaluated design."""
        seen_mu = []

        def fake_eval(design, _engine):
            seen_mu.append(design.get("mu"))
            return {
                "valid_physics": True,
                "sigma_max": 0.0,
                "tau_max": 0.0,
                "theta_sigma_max": 0.0,
                "theta_tau_max": 0.0,
            }

        config = {
            "random_seed": 7,
            "operating": {"mu": 0.37},
            "limits": {"sigma_allow": 180e6, "tau_allow": 100e6, "safety_factor": 1.0},
        }

        with patch(
            "mech390.datagen.generate.stage1_kinematic.generate_valid_2d_mechanisms",
            return_value=[{"r": 0.1, "l": 0.4, "e": 0.02}],
        ), patch(
            "mech390.datagen.generate.stage2_embodiment.iter_expand_to_3d",
            return_value=iter([{"r": 0.1, "l": 0.4, "e": 0.02}]),
        ), patch(
            "mech390.datagen.generate._evaluate_physics",
            side_effect=fake_eval,
        ):
            result = generate.generate_dataset(config)

        self.assertEqual(seen_mu, [0.37])
        self.assertAlmostEqual(result.all_cases.iloc[0]["mu"], 0.37)

    def test_mass_properties_fixed_rho_policy(self):
        """material.rho must be fixed (scalar or min==max)."""
        cfg_scalar = {"material": {"rho": 7800}}
        self.assertEqual(mass_properties._fixed_rho_from_config(cfg_scalar), 7800.0)

        cfg_fixed_range = {"material": {"rho": {"min": 7850, "max": 7850}}}
        self.assertEqual(mass_properties._fixed_rho_from_config(cfg_fixed_range), 7850.0)

        cfg_variable_range = {"material": {"rho": {"min": 7700, "max": 7900}}}
        with self.assertRaises(ValueError):
            mass_properties._fixed_rho_from_config(cfg_variable_range)

    def test_link_mass_matches_manual_formula(self):
        """link_mass should match net-area times thickness times density."""
        c = 0.10
        width = 0.020
        thickness = 0.012
        d_left = 0.010
        d_right = 0.011
        rho = 7800.0

        length = c + 0.5 * d_left + 0.5 * d_right
        expected_area = length * width - (np.pi / 4.0) * (d_left**2 + d_right**2)
        expected_mass = expected_area * thickness * rho

        mass = mass_properties.link_mass(c, width, thickness, d_left, d_right, rho)
        self.assertAlmostEqual(mass, expected_mass, places=12)

    def test_link_mass_moi_positive_and_increasing(self):
        """Link mass MOI should be positive and generally increase with section size."""
        c = 0.11
        d_left = 0.010
        d_right = 0.010
        rho = 7800.0

        i_small = mass_properties.link_mass_moi_cg_z(
            c, width=0.015, thickness=0.010, d_left=d_left, d_right=d_right, rho=rho
        )
        i_large = mass_properties.link_mass_moi_cg_z(
            c, width=0.022, thickness=0.014, d_left=d_left, d_right=d_right, rho=rho
        )
        self.assertGreater(i_small, 0.0)
        self.assertGreater(i_large, i_small)

    def test_area_moments_gross_formulas(self):
        """Gross area moments should match rectangle formulas."""
        width = 0.02
        thickness = 0.01
        link_i = mass_properties.link_area_moments_gross(width, thickness)
        self.assertAlmostEqual(link_i["Iyy"], width * thickness**3 / 12.0, places=15)
        self.assertAlmostEqual(link_i["Izz"], thickness * width**3 / 12.0, places=15)

        slider_w = 0.2
        slider_h = 0.3
        slider_i = mass_properties.slider_area_moments_gross(slider_w, slider_h)
        self.assertAlmostEqual(slider_i["I_area_x"], slider_w * slider_h**3 / 12.0, places=15)
        self.assertAlmostEqual(slider_i["I_area_y"], slider_h * slider_w**3 / 12.0, places=15)

    def test_slider_mass_matches_manual_formula(self):
        """Slider mass should subtract one pin-C hole from plan area."""
        length = 0.2
        width = 0.18
        height = 0.12
        d_c = 0.015
        rho = 7800.0

        expected_mass = (length * width - (np.pi / 4.0) * d_c**2) * height * rho
        got_mass = mass_properties.slider_mass(length, width, height, d_c, rho)
        self.assertAlmostEqual(got_mass, expected_mass, places=12)

    def test_invalid_geometry_raises_for_negative_net_area(self):
        """Mass helpers should raise when hole area exceeds rectangle area."""
        with self.assertRaises(ValueError):
            mass_properties.link_plan_area_net(
                center_distance=0.005,
                width=0.006,
                d_left=0.02,
                d_right=0.02,
            )

    def test_compute_design_mass_properties_schema(self):
        """Design-level aggregator should return all required mass/inertia keys."""
        config = get_baseline_config()
        design = {
            "r": 0.08,
            "l": 0.30,
            "width_r": 0.018,
            "width_l": 0.017,
            "thickness_r": 0.012,
            "thickness_l": 0.011,
            "pin_diameter_A": 0.010,
            "pin_diameter_B": 0.010,
            "pin_diameter_C": 0.009,
        }

        props = mass_properties.compute_design_mass_properties(design, config)
        required = {
            "rho",
            "mass_crank",
            "mass_rod",
            "mass_slider",
            "I_mass_crank_cg_z",
            "I_mass_rod_cg_z",
            "I_mass_slider_cg_z",
            "I_area_crank_yy",
            "I_area_crank_zz",
            "I_area_rod_yy",
            "I_area_rod_zz",
            "I_area_slider_x",
            "I_area_slider_y",
        }
        self.assertTrue(required.issubset(set(props.keys())))

        for key in required:
            self.assertTrue(np.isfinite(props[key]), f"{key} should be finite")
            self.assertGreater(props[key], 0.0, f"{key} should be positive")

    def test_mass_properties_cog_helpers_backward_compatibility(self):
        """Existing COG helpers should still return 2D vectors."""
        theta, r, l, e = np.pi / 6, 0.1, 0.4, 0.02
        c_crank = mass_properties.crank_cog(theta, r)
        c_rod = mass_properties.rod_cog(theta, r, l, e)
        c_slider = mass_properties.slider_cog(theta, r, l, e)

        for name, vec in [("crank_cog", c_crank), ("rod_cog", c_rod), ("slider_cog", c_slider)]:
            self.assertIsInstance(vec, np.ndarray, f"{name} should return np.ndarray")
            self.assertEqual(vec.shape, (2,), f"{name} should have shape (2,)")

    def test_slider_returns_vector(self):
        """Slider pos/vel/acc must return np.ndarray of shape (2,) with y == 0."""
        r, l, e, theta, omega = 0.1, 0.4, 0.02, np.pi / 4, 10.0

        pos = kinematics.slider_position(theta, r, l, e)
        vel = kinematics.slider_velocity(theta, omega, r, l, e)
        acc = kinematics.slider_acceleration(theta, omega, r, l, e)

        for name, vec in [("slider_position", pos), ("slider_velocity", vel), ("slider_acceleration", acc)]:
            self.assertIsInstance(vec, np.ndarray, f"{name} should return np.ndarray")
            self.assertEqual(vec.shape, (2,), f"{name} should have shape (2,)")
            self.assertEqual(vec[1], 0.0, f"{name} y-component should be 0.0")

    def test_crank_pin_returns_vector(self):
        """Crank-pin pos/vel/acc must return np.ndarray of shape (2,) with both components non-trivially set."""
        r, theta, omega = 0.1, np.pi / 3, 10.0

        pos = kinematics.crank_pin_position(theta, r)
        vel = kinematics.crank_pin_velocity(theta, omega, r)
        acc = kinematics.crank_pin_acceleration(theta, omega, r)

        for name, vec in [("crank_pin_position", pos), ("crank_pin_velocity", vel), ("crank_pin_acceleration", acc)]:
            self.assertIsInstance(vec, np.ndarray, f"{name} should return np.ndarray")
            self.assertEqual(vec.shape, (2,), f"{name} should have shape (2,)")

        # Verify values analytically
        np.testing.assert_allclose(pos, [r * np.cos(theta), r * np.sin(theta)], rtol=1e-10)
        np.testing.assert_allclose(vel, [-r * omega * np.sin(theta), r * omega * np.cos(theta)], rtol=1e-10)
        np.testing.assert_allclose(acc, [-r * omega**2 * np.cos(theta), -r * omega**2 * np.sin(theta)], rtol=1e-10)

    def test_rod_kinematics_helpers(self):
        """Verify the new rod-angle/omega/alpha helper functions behave correctly."""
        r, l, e = 0.1, 0.4, 0.02
        theta = np.pi / 6
        omega = 5.0
        alpha2 = 0.0

        # phi consistency
        phi = kinematics.rod_angle(theta, r, l, e)
        self.assertAlmostEqual(np.sin(phi), -(e + r * np.sin(theta)) / l, places=10)
        self.assertGreater(np.cos(phi), 0.0, "rod angle should pick positive cosine branch")

        # slider position via phi should match original function once the
        # offset translation is accounted for.  The rod-angle derivation
        # assumes the slider line at y = -e (pivot at origin), whereas
        # ``slider_position`` returns a frame where the slider is at y = 0.
        posB = kinematics.crank_pin_position(theta, r)
        posC_phi = posB + np.array([l * np.cos(phi), l * np.sin(phi)])
        # translate upward by +e to move the slider line to y=0
        posC_phi_shifted = posC_phi + np.array([0.0, e])
        posC = kinematics.slider_position(theta, r, l, e)
        np.testing.assert_allclose(posC, posC_phi_shifted, rtol=1e-8, atol=1e-12)

        # angular velocity via helper equals implicitly derived value
        omega_cb = kinematics.rod_angular_velocity(theta, omega, r, l, e)
        # compute using slider velocity formula: should satisfy V_C = V_B + V_C/B
        V_B = kinematics.crank_pin_velocity(theta, omega, r)
        V_C = kinematics.slider_velocity(theta, omega, r, l, e)
        V_rel = np.array([-omega_cb * l * np.sin(phi), omega_cb * l * np.cos(phi)])
        np.testing.assert_allclose(V_C, V_B + V_rel, rtol=1e-8, atol=1e-12)

        # angular acceleration helper consistency
        alpha_cb = kinematics.rod_angular_acceleration(theta, omega, r, l, e, alpha2=alpha2)
        # use acceleration relationship: a_C = a_B + a_rel
        a_B = kinematics.crank_pin_acceleration(theta, omega, r)
        a_C = kinematics.slider_acceleration(theta, omega, r, l, e)
        a_rel = np.array([
            -alpha_cb * l * np.sin(phi) - omega_cb ** 2 * l * np.cos(phi),
            alpha_cb * l * np.cos(phi) - omega_cb ** 2 * l * np.sin(phi),
        ])
        np.testing.assert_allclose(a_C, a_B + a_rel, rtol=1e-8, atol=1e-12)

        # verify invalid geometry for rod_angle raises
        with self.assertRaises(ValueError):
            # choose e large so sin_phi magnitude >1
            kinematics.rod_angle(theta, r, l, e=l + 1.0)

    def test_newton_euler_solver_residuals(self):
        """Newton-Euler linear system residuals should be near zero."""
        theta = 0.9
        omega = 8.0
        r, l, e = 0.08, 0.32, 0.03
        m_r, m_l, m_s = 1.2, 1.6, 2.0
        i_r, i_l = 0.02, 0.03
        mu = 0.2
        g = 9.81
        alpha_r = 0.0

        out = dynamics.solve_joint_reactions_newton_euler(
            theta=theta,
            omega=omega,
            r=r,
            l=l,
            e=e,
            mass_crank=m_r,
            mass_rod=m_l,
            mass_slider=m_s,
            I_crank=i_r,
            I_rod=i_l,
            mu=mu,
            g=g,
            alpha_r=alpha_r,
            v_eps=1e-9,
        )

        F_A = out["F_A"]
        F_B = out["F_B"]
        F_C = out["F_C"]
        N = out["N"]
        tau_A = out["tau_A"]

        r_A = np.array([0.0, 0.0])
        r_B = kinematics.crank_pin_position(theta, r)
        r_C = kinematics.slider_position(theta, r, l, e)
        r_Gr = mass_properties.crank_cog(theta, r)
        r_Gl = mass_properties.rod_cog(theta, r, l, e)

        a_B = kinematics.crank_pin_acceleration(theta, omega, r)
        a_C = kinematics.slider_acceleration(theta, omega, r, l, e)
        a_Gr = 0.5 * a_B
        a_Gl = 0.5 * (a_B + a_C)
        a_Gs = a_C

        alpha_l = kinematics.rod_angular_acceleration(theta, omega, r, l, e, alpha2=alpha_r)
        v_sx = float(kinematics.slider_velocity(theta, omega, r, l, e)[0])
        s = 0.0 if abs(v_sx) <= 1e-9 else np.sign(v_sx)

        r_A_Gr = r_A - r_Gr
        r_B_Gr = r_B - r_Gr
        r_B_Gl = r_B - r_Gl
        r_C_Gl = r_C - r_Gl

        residuals = np.array(
            [
                F_A[0] + F_B[0] - m_r * a_Gr[0],
                F_A[1] + F_B[1] - (m_r * a_Gr[1] + m_r * g),
                (r_A_Gr[0] * F_A[1] - r_A_Gr[1] * F_A[0])
                + (r_B_Gr[0] * F_B[1] - r_B_Gr[1] * F_B[0])
                + tau_A
                - i_r * alpha_r,
                -F_B[0] + F_C[0] - m_l * a_Gl[0],
                -F_B[1] + F_C[1] - (m_l * a_Gl[1] + m_l * g),
                (-r_B_Gl[0] * F_B[1] + r_B_Gl[1] * F_B[0])
                + (r_C_Gl[0] * F_C[1] - r_C_Gl[1] * F_C[0])
                - i_l * alpha_l,
                -F_C[0] - mu * s * N - m_s * a_Gs[0],
                -F_C[1] + N - (m_s * a_Gs[1] + m_s * g),
            ]
        )
        np.testing.assert_allclose(residuals, np.zeros_like(residuals), atol=1e-8, rtol=1e-8)

    def test_joint_reaction_wrapper_back_compat_keys(self):
        """Compatibility wrapper must still return legacy keys and F_O alias."""
        out = dynamics.joint_reaction_forces(
            theta=0.8,
            omega=6.0,
            r=0.08,
            l=0.32,
            e=0.03,
            mass_crank=1.0,
            mass_rod=1.1,
            mass_slider=1.3,
            I_crank=0.02,
            I_rod=0.03,
            mu=0.2,
        )

        for key in ["F_A", "F_B", "F_C", "F_O"]:
            self.assertIn(key, out)
            self.assertIsInstance(out[key], np.ndarray)
            self.assertEqual(out[key].shape, (2,))

        np.testing.assert_allclose(out["F_O"], out["F_A"])
        self.assertTrue(np.isfinite(out["tau_A"]))

    def test_friction_sign_flips_with_slider_velocity_direction(self):
        """Friction sign should flip when slider velocity sign flips."""
        common = {
            "omega": 8.0,
            "r": 0.08,
            "l": 0.32,
            "e": 0.03,
            "mass_crank": 1.1,
            "mass_rod": 1.2,
            "mass_slider": 1.4,
            "I_crank": 0.02,
            "I_rod": 0.03,
            "mu": 0.25,
            "v_eps": 1e-9,
        }
        theta_neg = 1.0
        theta_pos = 4.5

        out_neg = dynamics.solve_joint_reactions_newton_euler(theta=theta_neg, **common)
        out_pos = dynamics.solve_joint_reactions_newton_euler(theta=theta_pos, **common)

        v_neg = float(kinematics.slider_velocity(theta_neg, common["omega"], common["r"], common["l"], common["e"])[0])
        v_pos = float(kinematics.slider_velocity(theta_pos, common["omega"], common["r"], common["l"], common["e"])[0])

        self.assertLess(v_neg, 0.0)
        self.assertGreater(v_pos, 0.0)
        self.assertGreater(out_neg["F_f"], 0.0)
        self.assertLess(out_pos["F_f"], 0.0)

    def test_newton_euler_invalid_geometry_raises(self):
        """Invalid geometry should fail clearly in the dynamics solver."""
        with self.assertRaises(ValueError):
            dynamics.solve_joint_reactions_newton_euler(
                theta=0.5,
                omega=6.0,
                r=0.1,
                l=0.09,
                e=0.2,
                mass_crank=1.0,
                mass_rod=1.0,
                mass_slider=1.0,
            )


if __name__ == '__main__':
    unittest.main()
