"""Run offline regression tests: python -m unittest -v test_carton.

The real urdfpy backend has a separate optional test; it is not substituted.
"""

from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np
from build_carton import Parameters, generate
from kinematics import Model
from motion import Motion
from validate_carton import insertion_metrics, sat_overlap, validate


class CartonTests(unittest.TestCase):
    def make(self, directory, p=Parameters(), backend="reference"):
        urdf, meta = generate(p, Path(directory))
        return Model(urdf, backend), Motion(meta)

    def test_default_motion(self):
        with tempfile.TemporaryDirectory() as temp:
            model, motion = self.make(temp)
            report = validate(model, motion, 1001)
            self.assertTrue(report["passed"], report)
            self.assertEqual((len(model.links), len(model.joints)), (22, 21))
            self.assertTrue(report["all_interlink_collision_pairs_enabled"])

    def test_dimensions_change_nonuniformly(self):
        cases = [
            (0.045, 0.035, 0.080, 0.0005),
            (0.060, 0.045, 0.240, 0.0008),
            (0.150, 0.050, 0.100, 0.001),
            (0.050, 0.110, 0.140, 0.0007),
            (0.160, 0.120, 0.220, 0.0015),
            (0.180, 0.140, 0.070, 0.0012),
            (0.240, 0.180, 0.200, 0.0004),
            (0.050, 0.040, 0.070, 0.0015),
        ]
        for W, D, H, t in cases:
            with (
                self.subTest(W=W, D=D, H=H, t=t),
                tempfile.TemporaryDirectory() as temp,
            ):
                model, motion = self.make(
                    temp, Parameters(width=W, depth=D, height=H, thickness=t)
                )
                report = validate(model, motion, 401)
                self.assertTrue(report["passed"], report)

    def test_uniform_scaling(self):
        p = Parameters().resolved()
        for factor in (0.3, 2.7):
            scaled = Parameters(
                width=p.width * factor,
                depth=p.depth * factor,
                height=p.height * factor,
                thickness=p.thickness * factor,
            ).resolved()
            for name, value in p.derived().items():
                self.assertAlmostEqual(
                    scaled.derived()[name], factor * value, places=12
                )
            self.assertAlmostEqual(scaled.tab_length, factor * p.tab_length, places=12)

    def test_flat_zero_pose(self):
        with tempfile.TemporaryDirectory() as temp:
            model, motion = self.make(temp)
            for box, pose in model.box_poses(motion.zero):
                self.assertAlmostEqual(pose[2, 3], 0.0, places=12)
            self.assertTrue(all(q == 0 for q in motion.evaluate(0)[0].values()))

    def test_inserted_tabs_stay_centered(self):
        with tempfile.TemporaryDirectory() as temp:
            model, motion = self.make(temp)
            for start, stop, end in ((9, 10, "bottom"), (13, 14, "top")):
                depths = []
                for t in np.linspace(motion.bounds[start], motion.bounds[stop], 151):
                    q = motion.evaluate(float(t))[0]
                    metrics = insertion_metrics(model, motion, q, end)
                    depths.append(metrics["tip_insertion_depth_m"])
                    self.assertAlmostEqual(
                        metrics["wall_face_clearance_m"],
                        motion.p["face_clearance"],
                        places=10,
                    )
                    self.assertAlmostEqual(
                        metrics["keeper_face_clearance_m"],
                        motion.p["face_clearance"],
                        places=10,
                    )
                self.assertTrue(np.all(np.diff(depths) >= -1e-10))
                self.assertAlmostEqual(depths[-1], motion.p["tab_length"], places=10)

    def test_invalid_parameters_are_rejected(self):
        for p in (
            Parameters(thickness=-0.001),
            Parameters(width=0.001),
            Parameters(tab_length=0.1),
            Parameters(tab_width=0.20),
            Parameters(face_clearance=0.00001),
            Parameters(depth=float("nan")),
            Parameters(width=None),
        ):
            with self.subTest(p=p):
                with self.assertRaises((ValueError, TypeError)):
                    p.validate()

    def test_separating_axis_checker_detects_overlap(self):
        A, B, h = np.eye(4), np.eye(4), np.array([0.5, 0.5, 0.5])
        B[0, 3] = 0.9
        self.assertGreater(sat_overlap(A, h, B, h), 0.09)
        B[0, 3] = 1.1
        self.assertLess(sat_overlap(A, h, B, h), -0.09)

    @unittest.skipUnless(importlib.util.find_spec("urdfpy"), "urdfpy is not installed")
    def test_real_urdfpy_agrees_with_reference(self):
        with tempfile.TemporaryDirectory() as temp:
            model, motion = self.make(temp, backend="urdfpy")
            for t in np.linspace(0, motion.total, 21):
                self.assertLess(
                    model.crosscheck_urdfpy(motion.evaluate(float(t))[0]), 1e-10
                )


if __name__ == "__main__":
    unittest.main()
