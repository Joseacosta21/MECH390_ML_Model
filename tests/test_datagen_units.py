
import unittest
import numpy as np
import sys
import os

# Ensure src is in path
sys.path.append(os.path.abspath('src'))

from mech390.physics import kinematics
from mech390.datagen import stage1_kinematic
from mech390.config import get_baseline_config

class TestKinematicsAndDatagen(unittest.TestCase):
    
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


if __name__ == '__main__':
    unittest.main()
