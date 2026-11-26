"""Regression tests for generation, geometry, trajectory and optional urdfpy FK."""

import importlib.util
import json
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np
from build_mailer import Parameters, generate, subtract_rectangles
from kinematics import Model
from motion import Motion
from validate_mailer import metrics, validate


class MailerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.path, _ = generate(Parameters(), self.root / "default")
        self.model = Model(self.path, "reference")
        self.motion = Motion(self.path.with_suffix(".json"))

    def tearDown(self):
        self.temp.cleanup()

    def test_default_collision_limits_inertia(self):
        report = validate(self.model, self.motion, 401)
        self.assertTrue(report["passed"], report)
        self.assertEqual(report["num_links"], 27)
        self.assertEqual(report["num_joints"], 26)

    def test_slot_empty_partition_area(self):
        vals = subtract_rectangles((0, 4, 0, 3), [(1, 2, 1, 2)])
        self.assertAlmostEqual(sum((b - a) * (d - c) for a, b, c, d in vals), 11)

    def test_zero_is_planar(self):
        for b, T in self.model.box_poses(self.motion.zero):
            self.assertAlmostEqual(T[2, 3], 0)
            self.assertAlmostEqual(abs(T[2, 2]), 1)

    def test_insertion_centering_all_intermediate_times(self):
        for k in (7, 14):
            for time in np.linspace(
                self.motion.bounds[k], self.motion.bounds[k + 1], 31
            ):
                mm = metrics(
                    self.model, self.motion, self.motion.evaluate(float(time))[0]
                )
                names = (
                    [f"{s}_lock_{i}" for s in ("left", "right") for i in range(2)]
                    if k == 7
                    else ["front"]
                )
                for key in names:
                    self.assertLess(mm[key]["centering_error_m"], 1e-10)
                    self.assertAlmostEqual(
                        mm[key]["min_face_clearance_m"], 0.00045, places=10
                    )

    def test_continuity_at_boundaries(self):
        for t in self.motion.bounds[1:-1]:
            qa = self.motion.evaluate(float(t) - 1e-6)[0]
            qb = self.motion.evaluate(float(t) + 1e-6)[0]
            self.assertLess(max(abs(qa[k] - qb[k]) for k in qa), 1e-9)

    def test_uniform_scale_invariance(self):
        p = Parameters(width=0.4, depth=0.3, height=0.09, thickness=0.003)
        path, _ = generate(p, self.root / "double")
        m = Model(path, "reference")
        motion = Motion(path.with_suffix(".json"))
        for t in (0, 9, self.motion.open_time, self.motion.total):
            a = self.model.fk(self.motion.evaluate(t)[0])
            b = m.fk(motion.evaluate(t)[0])
            for key in a:
                self.assertTrue(np.allclose(a[key][:3, :3], b[key][:3, :3], atol=1e-10))
                self.assertTrue(
                    np.allclose(a[key][:3, 3] * 2, b[key][:3, 3], atol=1e-10)
                )

    def test_different_dimensions(self):
        p = Parameters(width=0.28, depth=0.2, height=0.06, thickness=0.002)
        path, _ = generate(p, self.root / "custom")
        m = Model(path, "reference")
        motion = Motion(path.with_suffix(".json"))
        self.assertEqual(m.joint_names, self.model.joint_names)
        self.assertTrue(validate(m, motion, 251)["passed"])

    def test_invalid_inputs_rejected(self):
        for values in [
            dict(width=-1),
            dict(height=0.004),
            dict(width=0.08, height=0.08),
            dict(thickness=math.nan),
            dict(front_tab_width=0.4),
            dict(front_tab_length=0.2),
            dict(lid_wing_depth=0.15),
            dict(face_clearance=0.00001),
        ]:
            with self.subTest(values=values), self.assertRaises(ValueError):
                generate(Parameters(**values), self.root / "bad")

    def test_sample_times_and_nan(self):
        ts, qs, stages = self.motion.sample(12)
        self.assertTrue(np.all(np.diff(ts) > 0))
        self.assertAlmostEqual(ts[-1], self.motion.total)
        self.assertEqual(len(ts), len(qs))
        self.assertEqual(len(ts), len(stages))
        with self.assertRaises(ValueError):
            self.motion.evaluate(math.nan)

    @unittest.skipUnless(
        importlib.util.find_spec("urdfpy") is not None, "upstream urdfpy not installed"
    )
    def test_real_urdfpy_transforms(self):
        m = Model(self.path, "urdfpy")
        for t in np.linspace(0, self.motion.total, 101):
            self.assertLess(
                m.crosscheck_urdfpy(self.motion.evaluate(float(t))[0]), 1e-9
            )


if __name__ == "__main__":
    unittest.main()
