"""Continuous revolute-only assembly motion, with analytic insertion segments."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

HALF = math.pi / 2


def smooth(u):
    u = float(np.clip(u, 0, 1))
    return u * u * u * (10 + u * (-15 + 6 * u))


def bisect_root(fn, lo, hi, target):
    if not fn(lo) <= target <= fn(hi):
        raise ValueError(
            f"Insertion reach not feasible: {fn(lo):.6g} <= {target:.6g} <= {fn(hi):.6g} failed"
        )
    for _ in range(70):
        m = (lo + hi) / 2
        if fn(m) < target:
            lo = m
        else:
            hi = m
    return (lo + hi) / 2


class Motion:
    def __init__(self, path: Path | dict):
        self.meta = (
            path if isinstance(path, dict) else json.loads(Path(path).read_text())
        )
        self.p = self.meta["parameters"]
        self.d = self.meta["derived"]
        if self.meta.get("model_type") != "hinged_lid_rollover_mailer":
            raise ValueError("Metadata is not a hinged-lid mailer model")
        p, d = self.p, self.d
        A, L = d["foot_margin"], d["liner_length"]

        # delta is the small inward inclination of the liner; alpha is foot angle above horizontal.
        def lock_root_height(alpha):
            delta = math.asin(A * (1 - math.cos(alpha)) / L)
            return p["height"] - L * math.cos(delta) + A * math.sin(alpha)

        self.lock_entry = bisect_root(
            lock_root_height,
            0,
            math.radians(35),
            d["lock_tab_length"] + d["lock_tip_entry_height"],
        )
        a, lidL = d["lid_front_margin"], d["lid_length"]
        max_theta = math.acos(1 - a / lidL)

        def front_tip_height(theta):
            c = lidL * (1 - math.cos(theta)) / a
            return (
                d["lid_lift"]
                + lidL * math.sin(theta)
                - a * math.sqrt(max(0, 1 - c * c))
                - p["front_tab_length"]
            )

        self.front_entry = bisect_root(
            front_tip_height, 0, max_theta, d["front_entry_tip_height"]
        )
        self.zero = {j: 0.0 for j in self.meta["joint_names"]}
        self.stages = []
        self.keyframes = [self.zero.copy()]

        def add(name, duration, changes=None, kind="lerp"):
            q = self.keyframes[-1].copy()
            q.update(changes or {})
            self.stages.append(dict(name=name, duration=duration, kind=kind))
            self.keyframes.append(q)

        add("Flat one-piece simulation net", 1.4)
        add(
            "Raise front wall and rear lid spine",
            2.4,
            {
                "rear_rise": HALF,
                "front_rise": HALF,
                "lid_fold": HALF - d["lid_open_angle"],
            },
        )
        add(
            "Fold four corner ears inward",
            1.8,
            {
                f"{e}_corner_{k}_fold": HALF
                for e in ("rear", "front")
                for k in ("a", "b")
            },
        )
        add(
            "Raise the tray side walls",
            2.2,
            {f"{s}_rise": HALF for s in ("left", "right")},
        )
        add(
            "Fold the two rollover clearance strips",
            1.6,
            {f"{s}_roll_base": HALF for s in ("left", "right")},
        )
        prep = self.lock_angles(self.lock_entry)
        add(
            "Pre-bend the lower locking strips",
            1.8,
            {j: v for j, v in prep.items() if "roll_return" not in j},
        )
        add("Roll liners down and align the floor tabs", 2.2, prep)
        add(
            "Insert four locking tabs into base slots",
            2.0,
            self.lock_angles(0),
            kind="base_insert",
        )
        add("Deploy front receiver spacing ears", 1.5, {"front_socket_base": HALF})
        add("Return the front keeper into the tray", 1.6, {"front_socket_return": HALF})
        add("Tray assembled / lid open", 2.0)
        self.tray_time = sum(s["duration"] for s in self.stages)
        add(
            "Fold the lid side wings",
            1.8,
            {"lid_left_wing_fold": HALF, "lid_right_wing_fold": HALF},
        )
        ent = self.front_angles(self.front_entry)
        add(
            "Pre-fold the front tuck margin and flap",
            1.5,
            {j: v for j, v in ent.items() if j != "lid_fold"},
        )
        add("Lower the lid and enter the side channels", 2.8, ent)
        add(
            "Insert the front flap into its open receiver",
            2.8,
            self.front_angles(0),
            kind="front_insert",
        )
        add("Closed hinged-lid mailer", 2.0)
        self.bounds = np.r_[0, np.cumsum([s["duration"] for s in self.stages])]
        self.total = float(self.bounds[-1])
        self.open_time = float(self.bounds[13])
        self.lock_insert_index = 7
        self.front_insert_index = 14

    def lock_angles(self, alpha):
        d = self.d
        delta = math.asin(d["foot_margin"] * (1 - math.cos(alpha)) / d["liner_length"])
        q = {}
        for s in ("left", "right"):
            q[f"{s}_roll_return"] = HALF - delta
            q[f"{s}_foot_fold"] = -HALF - alpha + delta
            for k in range(2):
                q[f"{s}_lock_{k}_fold"] = HALF + alpha
        return q

    def front_angles(self, theta):
        d = self.d
        c = d["lid_length"] * (1 - math.cos(theta)) / d["lid_front_margin"]
        beta = math.acos(float(np.clip(c, -1, 1)))
        return {
            "lid_fold": HALF - theta,
            "lid_front_margin_fold": theta + beta,
            "lid_front_tab_fold": HALF - beta,
        }

    def evaluate(self, t):
        if not math.isfinite(t):
            raise ValueError("time must be finite")
        t = float(np.clip(t, 0, self.total))
        k = min(
            int(np.searchsorted(self.bounds, t, side="right") - 1), len(self.stages) - 1
        )
        u = smooth((t - self.bounds[k]) / (self.bounds[k + 1] - self.bounds[k]))
        a, b = self.keyframes[k], self.keyframes[k + 1]
        q = {j: a[j] + u * (b[j] - a[j]) for j in self.zero}
        if self.stages[k]["kind"] == "base_insert":
            q.update(self.lock_angles((1 - u) * self.lock_entry))
        if self.stages[k]["kind"] == "front_insert":
            q.update(self.front_angles((1 - u) * self.front_entry))
        return q, self.stages[k]["name"], k

    def sample(self, fps=30):
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError("fps must be finite and positive")
        times = np.linspace(0, self.total, math.ceil(self.total * fps) + 1)
        values = [self.evaluate(float(t)) for t in times]
        return times, [v[0] for v in values], [v[1] for v in values]

    def save(self, path, fps=120):
        ts, qs, labels = self.sample(fps)
        data = {
            "units": {"time": "s", "angles": "rad"},
            "joint_names": list(self.zero),
            "times": ts.tolist(),
            "positions": [[q[j] for j in self.zero] for q in qs],
            "stages": labels,
            "notice": "Position targets / reference kinematics only, not recorded dynamics.",
        }
        Path(path).write_text(json.dumps(data, separators=(",", ":")) + "\n")
