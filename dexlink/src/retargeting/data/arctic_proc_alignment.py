import math
from pathlib import Path

import numpy as np
from urdfpy import URDF

from dexlink.utils.retargeting.path_utils import get_assets_dir

ARCTIC_ABS_URDF_PATHS = {
    "box": get_assets_dir() / "arctic_box" / "decomp" / "box_decomp_abs.urdf",
    "waffleiron": get_assets_dir()
    / "arctic_waffleiron"
    / "decomp"
    / "waffleiron_decomp_abs.urdf",
    "notebook": get_assets_dir()
    / "arctic_notebook"
    / "decomp"
    / "notebook_decomp_abs.urdf",
    "ketchup": get_assets_dir()
    / "arctic_ketchup"
    / "decomp"
    / "ketchup_decomp_abs.urdf",
    "laptop": get_assets_dir() / "arctic_laptop" / "decomp" / "laptop_decomp_abs.urdf",
}


ASSET_AXIS_MODE = {
    # use OBB smallest extent axis as link normal for plate/bin-like bodies
    "box": {"bottom": "min", "top": "min"},
    "laptop": {"bottom": "min", "top": "min"},
    "notebook": {"bottom": "min", "top": "min"},
    "waffleiron": {"bottom": "min", "top": "min"},
    # bottle body is tall; its opening/stack axis is the largest-extent axis
    "ketchup": {"bottom": "max", "top": "min"},
}


def _urdf_single_joint_cfg(urdf_inst: URDF, theta: float):
    cfg = {}
    for joint in urdf_inst.joints:
        if joint.joint_type != "fixed":
            cfg[joint] = float(theta)
    return cfg


def _visual_mesh_entries_with_link(urdf_inst: URDF, joint_cfg: dict):
    link_fk = urdf_inst.link_fk(cfg=joint_cfg, links=None)
    for link, link_pose in link_fk.items():
        for visual in getattr(link, "visuals", None) or []:
            geom = getattr(visual, "geometry", None)
            if geom is None:
                continue
            origin = visual.origin if visual.origin is not None else np.eye(4)
            for mesh in getattr(geom, "meshes", None) or []:
                pose = link_pose @ origin
                mesh_field = getattr(geom, "mesh", None)
                scale = (
                    getattr(mesh_field, "scale", None)
                    if mesh_field is not None
                    else None
                )
                if scale is not None:
                    S = np.eye(4, dtype=np.float64)
                    S[:3, :3] = np.diag(scale)
                    pose = pose @ S
                yield link.name, mesh, pose


def _link_points_world(urdf_inst: URDF, theta: float, link_name: str) -> np.ndarray:
    pts = []
    joint_cfg = _urdf_single_joint_cfg(urdf_inst, theta)
    for cur_link_name, mesh, pose in _visual_mesh_entries_with_link(
        urdf_inst, joint_cfg
    ):
        if cur_link_name != link_name:
            continue
        verts = np.asarray(mesh.vertices)
        if verts.size == 0:
            continue
        verts_h = np.concatenate([verts, np.ones((verts.shape[0], 1))], axis=1)
        pts.append((pose @ verts_h.T).T[:, :3])
    if not pts:
        raise ValueError(f"No visual mesh vertices found for link '{link_name}'.")
    return np.concatenate(pts, axis=0)


def _obb_center_axis(points: np.ndarray, axis_mode: str = "min"):
    mean = points.mean(axis=0)
    centered = points - mean
    cov = centered.T @ centered / max(points.shape[0], 1)
    evals, evecs = np.linalg.eigh(cov)  # ascending
    local = centered @ evecs
    mins = local.min(axis=0)
    maxs = local.max(axis=0)
    extents = maxs - mins
    center_local = 0.5 * (mins + maxs)
    center_world = mean + evecs @ center_local

    if axis_mode == "min":
        axis_idx = int(np.argmin(extents))
    elif axis_mode == "max":
        axis_idx = int(np.argmax(extents))
    elif axis_mode == "mid":
        axis_idx = int(np.argsort(extents)[1])
    else:
        raise ValueError(f"Unknown axis_mode: {axis_mode}")

    axis = evecs[:, axis_idx]
    axis = axis / max(np.linalg.norm(axis), 1e-12)
    return center_world, axis, extents, evecs


def _find_articulated_joint(urdf_inst: URDF):
    joints = [j for j in urdf_inst.joints if j.joint_type != "fixed"]
    if not joints:
        return None
    if len(joints) > 1:
        # Current ARCTIC/procedural assets are expected to have one articulation.
        return joints[0]
    return joints[0]


def _hinge_summary(urdf_inst: URDF, theta: float):
    joint = _find_articulated_joint(urdf_inst)
    if joint is None:
        return None

    joint_cfg = _urdf_single_joint_cfg(urdf_inst, theta)
    link_fk = urdf_inst.link_fk(cfg=joint_cfg, links=None)
    parent_link = next(
        (link for link in urdf_inst.links if link.name == str(joint.parent)), None
    )
    if parent_link is None:
        # urdfpy sometimes stores names directly in joint.parent/child
        parent_name = (
            joint.parent.name if hasattr(joint.parent, "name") else str(joint.parent)
        )
        parent_link = next(
            link for link in urdf_inst.links if link.name == parent_name
        )
    parent_pose = link_fk[parent_link]

    joint_origin = joint.origin if joint.origin is not None else np.eye(4)
    joint_pose = parent_pose @ joint_origin
    axis_local = np.asarray(
        joint.axis if joint.axis is not None else [1.0, 0.0, 0.0], dtype=float
    )
    axis_local /= max(np.linalg.norm(axis_local), 1e-12)
    axis_world = joint_pose[:3, :3] @ axis_local
    axis_world /= max(np.linalg.norm(axis_world), 1e-12)

    return {
        "joint_name": joint.name,
        "origin": joint_pose[:3, 3],
        "axis": axis_world,
    }


def _fit_rigid_transform(src_pts: np.ndarray, dst_pts: np.ndarray):
    src_mean = src_pts.mean(axis=0)
    dst_mean = dst_pts.mean(axis=0)
    H = (src_pts - src_mean).T @ (dst_pts - dst_mean)
    U, _, Vt = np.linalg.svd(H)
    R_fit = Vt.T @ U.T
    if np.linalg.det(R_fit) < 0:
        Vt[-1, :] *= -1
        R_fit = Vt.T @ U.T
    t_fit = dst_mean - R_fit @ src_mean
    T = np.eye(4)
    T[:3, :3] = R_fit
    T[:3, 3] = t_fit
    return T


def _apply_T(T: np.ndarray, p: np.ndarray):
    p = np.asarray(p, dtype=float)
    if p.ndim == 1:
        return T[:3, :3] @ p + T[:3, 3]
    return (T[:3, :3] @ p.T).T + T[:3, 3]


def _asset_link_summary(urdf_path: str | Path, theta: float, asset_name: str):
    urdf_inst = URDF.load(str(urdf_path))
    modes = ASSET_AXIS_MODE[asset_name]
    out = {}
    for link_name in ("bottom", "top"):
        pts = _link_points_world(urdf_inst, theta=theta, link_name=link_name)
        center, normal, extents, basis = _obb_center_axis(
            pts, axis_mode=modes[link_name]
        )
        out[link_name] = {
            "center": center,
            "normal": normal,
            "extents_pca": extents,
            "basis_pca": basis,
            "n_points": int(pts.shape[0]),
        }
    out["hinge"] = _hinge_summary(urdf_inst, theta=theta)
    return out


def _sign_flip_search_and_fit(arctic_summary: dict, proc_summary: dict):
    best = None
    for s_bottom in (-1.0, 1.0):
        for s_top in (-1.0, 1.0):
            for s_hinge in (-1.0, 1.0):
                a_bottom_n = s_bottom * arctic_summary["bottom"]["normal"]
                a_top_n = s_top * arctic_summary["top"]["normal"]
                p_bottom_n = proc_summary["bottom"]["normal"]
                p_top_n = proc_summary["top"]["normal"]

                src_pts = [
                    arctic_summary["bottom"]["center"],
                    arctic_summary["bottom"]["center"] + a_bottom_n,
                    arctic_summary["top"]["center"],
                    arctic_summary["top"]["center"] + a_top_n,
                ]
                dst_pts = [
                    proc_summary["bottom"]["center"],
                    proc_summary["bottom"]["center"] + p_bottom_n,
                    proc_summary["top"]["center"],
                    proc_summary["top"]["center"] + p_top_n,
                ]

                use_hinge = (
                    arctic_summary.get("hinge") is not None
                    and proc_summary.get("hinge") is not None
                )
                if use_hinge:
                    a_hinge_axis = s_hinge * arctic_summary["hinge"]["axis"]
                    p_hinge_axis = proc_summary["hinge"]["axis"]
                    src_pts.extend(
                        [
                            arctic_summary["hinge"]["origin"],
                            arctic_summary["hinge"]["origin"] + a_hinge_axis,
                        ]
                    )
                    dst_pts.extend(
                        [
                            proc_summary["hinge"]["origin"],
                            proc_summary["hinge"]["origin"] + p_hinge_axis,
                        ]
                    )

                src_pts = np.stack(src_pts, axis=0)
                dst_pts = np.stack(dst_pts, axis=0)
                T = _fit_rigid_transform(src_pts, dst_pts)

                pos_err = np.linalg.norm(_apply_T(T, src_pts) - dst_pts, axis=1).mean()
                a_bottom_n_aligned = _apply_T(T, a_bottom_n) - _apply_T(T, np.zeros(3))
                a_top_n_aligned = _apply_T(T, a_top_n) - _apply_T(T, np.zeros(3))
                a_bottom_n_aligned /= max(np.linalg.norm(a_bottom_n_aligned), 1e-12)
                a_top_n_aligned /= max(np.linalg.norm(a_top_n_aligned), 1e-12)
                ang_err = math.acos(
                    np.clip(np.dot(a_bottom_n_aligned, p_bottom_n), -1.0, 1.0)
                ) + math.acos(np.clip(np.dot(a_top_n_aligned, p_top_n), -1.0, 1.0))
                if use_hinge:
                    a_hinge_aligned = _apply_T(T, a_hinge_axis) - _apply_T(
                        T, np.zeros(3)
                    )
                    a_hinge_aligned /= max(np.linalg.norm(a_hinge_aligned), 1e-12)
                    ang_err += math.acos(
                        np.clip(np.dot(a_hinge_aligned, p_hinge_axis), -1.0, 1.0)
                    )

                score = pos_err + 0.05 * ang_err
                cand = {
                    "score": score,
                    "sign_flips": {"bottom": s_bottom, "top": s_top, "hinge": s_hinge},
                    "T_arctic_to_proc": T,
                    "pos_err_mean": pos_err,
                    "ang_err_sum_rad": ang_err,
                }
                if best is None or cand["score"] < best["score"]:
                    best = cand
    return best


def estimate_hinged_asset_arctic_to_proc_alignment(
    asset_name: str,
    proc_urdf_path: str | Path,
    arctic_urdf_path: str | Path | None = None,
    theta_proc: float = math.radians(45.0),
    theta_arctic: float | None = None,
    articulation_relation_hint: str | None = None,
):
    if asset_name not in ASSET_AXIS_MODE:
        raise ValueError(f"Unsupported asset_name: {asset_name}")

    if arctic_urdf_path is None:
        arctic_urdf_path = ARCTIC_ABS_URDF_PATHS[asset_name]
    if theta_arctic is None:
        if asset_name == "box":
            theta_arctic = math.pi - theta_proc
            articulation_relation_hint = "theta_arctic = pi - theta_proc"
        else:
            theta_arctic = theta_proc
            articulation_relation_hint = (
                articulation_relation_hint
                or "theta_arctic ~= theta_proc (default guess)"
            )
    else:
        articulation_relation_hint = (
            articulation_relation_hint or "user-provided theta_arctic"
        )

    arctic = _asset_link_summary(
        arctic_urdf_path, theta=float(theta_arctic), asset_name=asset_name
    )
    proc = _asset_link_summary(
        proc_urdf_path, theta=float(theta_proc), asset_name=asset_name
    )
    best = _sign_flip_search_and_fit(arctic, proc)

    return {
        "asset_name": asset_name,
        "theta_proc": float(theta_proc),
        "theta_arctic": float(theta_arctic),
        "articulation_relation": articulation_relation_hint,
        "arctic": arctic,
        "proc": proc,
        **best,
    }


def estimate_box_arctic_to_proc_alignment(
    proc_urdf_path: str | Path,
    arctic_urdf_path: str | Path | None = None,
    theta_proc: float = math.radians(60.0),
    theta_arctic: float | None = None,
):
    return estimate_hinged_asset_arctic_to_proc_alignment(
        asset_name="box",
        proc_urdf_path=proc_urdf_path,
        arctic_urdf_path=arctic_urdf_path,
        theta_proc=theta_proc,
        theta_arctic=theta_arctic,
        articulation_relation_hint="theta_arctic = pi - theta_proc",
    )


def estimate_laptop_arctic_to_proc_alignment(
    proc_urdf_path: str | Path,
    arctic_urdf_path: str | Path | None = None,
    theta_proc: float = math.radians(45.0),
    theta_arctic: float | None = None,
):
    return estimate_hinged_asset_arctic_to_proc_alignment(
        asset_name="laptop",
        proc_urdf_path=proc_urdf_path,
        arctic_urdf_path=arctic_urdf_path,
        theta_proc=theta_proc,
        theta_arctic=theta_arctic,
    )


def estimate_notebook_arctic_to_proc_alignment(
    proc_urdf_path: str | Path,
    arctic_urdf_path: str | Path | None = None,
    theta_proc: float = math.radians(45.0),
    theta_arctic: float | None = None,
):
    return estimate_hinged_asset_arctic_to_proc_alignment(
        asset_name="notebook",
        proc_urdf_path=proc_urdf_path,
        arctic_urdf_path=arctic_urdf_path,
        theta_proc=theta_proc,
        theta_arctic=theta_arctic,
    )


def estimate_waffleiron_arctic_to_proc_alignment(
    proc_urdf_path: str | Path,
    arctic_urdf_path: str | Path | None = None,
    theta_proc: float = math.radians(45.0),
    theta_arctic: float | None = None,
):
    return estimate_hinged_asset_arctic_to_proc_alignment(
        asset_name="waffleiron",
        proc_urdf_path=proc_urdf_path,
        arctic_urdf_path=arctic_urdf_path,
        theta_proc=theta_proc,
        theta_arctic=theta_arctic,
    )


def estimate_ketchup_arctic_to_proc_alignment(
    proc_urdf_path: str | Path,
    arctic_urdf_path: str | Path | None = None,
    theta_proc: float = math.radians(45.0),
    theta_arctic: float | None = None,
):
    return estimate_hinged_asset_arctic_to_proc_alignment(
        asset_name="ketchup",
        proc_urdf_path=proc_urdf_path,
        arctic_urdf_path=arctic_urdf_path,
        theta_proc=theta_proc,
        theta_arctic=theta_arctic,
    )


def print_alignment_report(
    asset_name: str,
    proc_urdf_path: str | Path,
    arctic_urdf_path: str | Path | None = None,
    theta_proc: float = math.radians(45.0),
    theta_arctic: float | None = None,
):
    res = estimate_hinged_asset_arctic_to_proc_alignment(
        asset_name=asset_name,
        proc_urdf_path=proc_urdf_path,
        arctic_urdf_path=arctic_urdf_path,
        theta_proc=theta_proc,
        theta_arctic=theta_arctic,
    )
    print(f"=== {asset_name} ARCTIC -> Procedural Alignment ===")
    print(f"proc urdf:   {proc_urdf_path}")
    print(f"arctic urdf: {arctic_urdf_path or ARCTIC_ABS_URDF_PATHS[asset_name]}")
    print(
        f"theta_proc={res['theta_proc']:.6f} rad ({math.degrees(res['theta_proc']):.2f} deg), "
        f"theta_arctic={res['theta_arctic']:.6f} rad ({math.degrees(res['theta_arctic']):.2f} deg)"
    )
    print(f"articulation relation hint: {res['articulation_relation']}")
    for family in ("arctic", "proc"):
        for link_name in ("bottom", "top"):
            info = res[family][link_name]
            print(
                f"{family}.{link_name}.center = "
                f"{np.array2string(info['center'], precision=6, suppress_small=True)}"
            )
            print(
                f"{family}.{link_name}.normal = "
                f"{np.array2string(info['normal'], precision=6, suppress_small=True)}"
            )
            print(
                f"{family}.{link_name}.extents_pca = "
                f"{np.array2string(info['extents_pca'], precision=6, suppress_small=True)}"
            )
        if res[family].get("hinge") is not None:
            print(
                f"{family}.hinge.origin = "
                f"{np.array2string(res[family]['hinge']['origin'], precision=6, suppress_small=True)}"
            )
            print(
                f"{family}.hinge.axis = "
                f"{np.array2string(res[family]['hinge']['axis'], precision=6, suppress_small=True)}"
            )
    print(f"chosen sign flips (ARCTIC): {res['sign_flips']}")
    print(f"mean anchor position error: {res['pos_err_mean']:.6e} m")
    print(f"angular error sum:         {res['ang_err_sum_rad']:.6e} rad")
    print("T_arctic_to_proc =")
    print(np.array2string(res["T_arctic_to_proc"], precision=6, suppress_small=True))
    return res
