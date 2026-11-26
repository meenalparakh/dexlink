"""Continuous folding trajectory, with analytic tuck-tab insertion.

No poses of links are directly overwritten. Every motion is a URDF joint
angle. During insertion the tab remains vertical and centered in its slot.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class Stage:
    name: str
    duration: float
    end: str = ""
    kind: str = "ramp"


STAGES = [
    Stage("Flat net", 0.7, kind="hold"),
    Stage("Deploy open-mouth receivers", 1.8),
    Stage("Raise left wall", 1.5),
    Stage("Raise right wall", 1.5),
    Stage("Close front wall", 1.7),
    Stage("Fold offset glue seam", 1.5),
    Stage("Fold bottom dust flaps", 1.5, "bottom"),
    Stage("Approach bottom receiver", 1.5, "bottom"),
    Stage("Align bottom tuck tab", 1.5, "bottom"),
    Stage("Insert bottom tab", 2.4, "bottom", "insert"),
    Stage("Fold top dust flaps", 1.5, "top"),
    Stage("Approach top receiver", 1.5, "top"),
    Stage("Align top tuck tab", 1.5, "top"),
    Stage("Insert top tab", 2.4, "top", "insert"),
    Stage("Assembled carton", 2.0, kind="hold"),
]


def smooth(u: float) -> float:
    u = float(np.clip(u, 0, 1))
    return u * u * u * (10 + u * (-15 + 6 * u))


class Motion:
    def __init__(self, meta_path: Path):
        self.meta_path = Path(meta_path)
        self.meta = json.loads(self.meta_path.read_text())
        self.p, self.d = self.meta["parameters"], self.meta["derived"]
        self.zero = {name: 0.0 for name in self.meta["joints"]}
        self.total = sum(s.duration for s in STAGES)
        self.bounds = np.r_[0, np.cumsum([s.duration for s in STAGES])]
        a, L, T = self.d["tuck_margin"], self.d["lid_length"], self.p["tab_length"]
        target = T + self.d["entry_tip_clearance"]
        lo, hi = math.asin(1 - a / L) + 1e-9, math.pi / 2
        if self.root_height(lo) <= target:
            raise ValueError(
                "No collision-clear vertical-tab entry pose for these dimensions"
            )
        for _ in range(80):
            mid = (lo + hi) / 2
            if self.root_height(mid) > target:
                lo = mid
            else:
                hi = mid
        self.entry_angle = (lo + hi) / 2
        self.keyframes = [self.zero.copy()]
        for stage in STAGES:
            self.keyframes.append(self.finish(stage, self.keyframes[-1]))

    def insertion_angles(self, theta: float) -> tuple[float, float, float]:
        """Returns lid angle, relative margin angle, relative tab angle.

        z_tab = L*sin(theta) + a*sin(alpha) = L
        tab absolute angle beta = theta + q_margin + q_tab = pi.
        Select alpha = pi - asin(L*(1-sin(theta))/a).
        """
        a, L = self.d["tuck_margin"], self.d["lid_length"]
        sine = L * (1 - math.sin(theta)) / a
        if sine < -1e-9 or sine > 1 + 1e-9:
            raise ValueError("Requested insertion angle is outside reachable branch")
        alpha = math.pi - math.asin(float(np.clip(sine, 0, 1)))
        return theta, alpha - theta, math.pi - alpha

    def root_height(self, theta: float) -> float:
        a, L = self.d["tuck_margin"], self.d["lid_length"]
        _, qm, _ = self.insertion_angles(theta)
        return a + L * math.cos(theta) + a * math.cos(theta + qm)

    def set_lid(self, cfg: dict[str, float], end: str, theta: float) -> None:
        q = self.insertion_angles(theta)
        for suffix, value in zip(("lid_fold", "tuck_margin_fold", "tuck_tab_fold"), q):
            cfg[f"{end}_{suffix}"] = value

    def finish(self, stage: Stage, old: dict[str, float]) -> dict[str, float]:
        q, pi2 = old.copy(), math.pi / 2
        name = stage.name
        if name == "Deploy open-mouth receivers":
            for end in ("top", "bottom"):
                q[f"{end}_socket_base"] = pi2
                q[f"{end}_socket_return"] = pi2
        elif name == "Raise left wall":
            q["fold_left_wall"] = pi2
        elif name == "Raise right wall":
            q["fold_right_wall"] = pi2
        elif name == "Close front wall":
            q["fold_front_wall"] = pi2
        elif name == "Fold offset glue seam":
            q["glue_flap_fold"] = pi2
        elif "dust flaps" in name:
            for side in ("left", "right"):
                q[f"{stage.end}_{side}_dust_fold"] = pi2
        elif name.startswith("Approach"):
            q[f"{stage.end}_lid_fold"] = self.entry_angle
        elif name.startswith("Align"):
            self.set_lid(q, stage.end, self.entry_angle)
        elif stage.kind == "insert":
            self.set_lid(q, stage.end, pi2)
        return q

    def evaluate(self, t: float) -> tuple[dict[str, float], str, int]:
        t = float(np.clip(t, 0, self.total))
        idx = min(
            int(np.searchsorted(self.bounds, t, side="right") - 1), len(STAGES) - 1
        )
        stage = STAGES[idx]
        u = smooth((t - self.bounds[idx]) / stage.duration)
        q0, q1 = self.keyframes[idx : idx + 2]
        q = {name: q0[name] + u * (q1[name] - q0[name]) for name in q0}
        if stage.kind == "insert":
            theta = self.entry_angle + u * (math.pi / 2 - self.entry_angle)
            self.set_lid(q, stage.end, theta)
        return q, stage.name, idx

    def sample(
        self, fps: float
    ) -> tuple[np.ndarray, list[dict[str, float]], list[str]]:
        if fps <= 0:
            raise ValueError("fps must be positive")
        times = np.linspace(0, self.total, int(round(self.total * fps)) + 1)
        results = [self.evaluate(float(t)) for t in times]
        return times, [r[0] for r in results], [r[1] for r in results]

    def save(self, path: Path, fps: float = 120.0) -> None:
        times, cfgs, labels = self.sample(fps)
        joints = list(self.zero)
        data = {
            "units": {"time": "s", "joint_position": "rad"},
            "joint_names": joints,
            "times": times.tolist(),
            "positions": [[q[j] for j in joints] for q in cfgs],
            "stage": labels,
            "note": "Uniform samples of motion.py; use Motion.evaluate(t) for exact continuous IK.",
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, separators=(",", ":")) + "\n")
