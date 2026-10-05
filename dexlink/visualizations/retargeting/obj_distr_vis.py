import argparse
import colorsys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R
from urdfpy import URDF

from dexlink.utils.retargeting.path_utils import get_assets_dir
from dexlink.utils.retargeting.vis_server import VizServer

DEFAULT_PROC_ASSETS = ["box", "laptop", "notebook", "ketchup", "waffleiron"]


def _parse_proc_assets(raw: str) -> list[str]:
    if not raw.strip():
        return []
    return [x.strip() for x in raw.split(",") if x.strip()]


def _grid_pose(xyz, rpy):
    pose = np.eye(4)
    pose[:3, 3] = xyz
    pose[:3, :3] = R.from_euler("xyz", rpy).as_matrix()
    return pose


def _link_name(link_or_name) -> str:
    return link_or_name.name if hasattr(link_or_name, "name") else str(link_or_name)


def _joint_cfg(
    urdf_inst: URDF,
    rng: np.random.Generator,
    articulation_value_deg: float | None = None,
) -> dict:
    cfg = {}
    for joint in urdf_inst.joints:
        jt = joint.joint_type
        if jt == "continuous":
            if articulation_value_deg is None:
                cfg[joint] = float(rng.uniform(-np.pi, np.pi))
            else:
                cfg[joint] = float(np.deg2rad(articulation_value_deg))
            continue
        if jt not in {"revolute", "prismatic"}:
            continue

        limit = getattr(joint, "limit", None)
        lower = getattr(limit, "lower", None)
        upper = getattr(limit, "upper", None)
        if lower is None or upper is None:
            if articulation_value_deg is None:
                cfg[joint] = 0.0
            else:
                cfg[joint] = float(
                    np.deg2rad(articulation_value_deg)
                    if jt == "revolute"
                    else articulation_value_deg
                )
            continue
        lower = float(lower)
        upper = float(upper)
        if not (np.isfinite(lower) and np.isfinite(upper)) or upper <= lower:
            if articulation_value_deg is None:
                cfg[joint] = 0.0
            else:
                cfg[joint] = float(
                    np.deg2rad(articulation_value_deg)
                    if jt == "revolute"
                    else articulation_value_deg
                )
            continue
        if articulation_value_deg is None:
            cfg[joint] = float(rng.uniform(lower, upper))
        else:
            target = (
                np.deg2rad(articulation_value_deg)
                if jt == "revolute"
                else articulation_value_deg
            )
            cfg[joint] = float(np.clip(target, lower, upper))
    return cfg


def _fixed_link_groups(urdf_inst: URDF) -> dict[str, int]:
    adj: dict[str, set[str]] = {_link_name(link): set() for link in urdf_inst.links}
    for joint in urdf_inst.joints:
        if joint.joint_type != "fixed":
            continue
        parent = _link_name(joint.parent)
        child = _link_name(joint.child)
        adj.setdefault(parent, set()).add(child)
        adj.setdefault(child, set()).add(parent)

    comp: dict[str, int] = {}
    comp_id = 0
    for link_name in adj:
        if link_name in comp:
            continue
        stack = [link_name]
        comp[link_name] = comp_id
        while stack:
            cur = stack.pop()
            for nei in adj[cur]:
                if nei in comp:
                    continue
                comp[nei] = comp_id
                stack.append(nei)
        comp_id += 1
    return comp


def _color_from_group(group_id: int, palette: str = "proc") -> int:
    h = (0.61803398875 * (group_id + 1)) % 1.0
    if palette == "arctic":
        # Cool, lower-saturation palette to distinguish ARCTIC references.
        h = (0.52 + 0.18 * h) % 1.0
        s, v = 0.35, 0.9
    else:
        s, v = 0.65, 0.95
    r, g, b = colorsys.hsv_to_rgb(h, s, v)
    return (int(r * 255) << 16) + (int(g * 255) << 8) + int(b * 255)


def _visual_mesh_entries(urdf_inst: URDF, joint_cfg: dict):
    link_fk = urdf_inst.link_fk(cfg=joint_cfg, links=None)
    for link, link_pose in link_fk.items():
        visuals = getattr(link, "visuals", None) or []
        for visual_idx, visual in enumerate(visuals):
            geom = getattr(visual, "geometry", None)
            if geom is None:
                continue
            origin = visual.origin if visual.origin is not None else np.eye(4)
            meshes = getattr(geom, "meshes", None) or []
            for mesh_idx, mesh in enumerate(meshes):
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
                yield _link_name(link), f"{visual_idx}_{mesh_idx}", mesh, pose


def _view_urdf(
    vis: VizServer,
    urdf_path: Path,
    name: str,
    base_pose: np.ndarray,
    alpha: float,
    rng: np.random.Generator,
    palette: str = "proc",
    articulation_value: float | None = None,
):
    urdf_inst = URDF.load(str(urdf_path))
    joint_cfg = _joint_cfg(urdf_inst, rng, articulation_value_deg=articulation_value)
    fixed_groups = _fixed_link_groups(urdf_inst)

    if joint_cfg:
        joint_desc = ", ".join(
            f"{getattr(j, 'name', 'joint')}={v:.4f}" for j, v in joint_cfg.items()
        )
        print(f"  [joints] {name}: {joint_desc}")
    else:
        print(f"  [joints] {name}: none")

    for link_name, mesh_tag, mesh, pose in _visual_mesh_entries(urdf_inst, joint_cfg):
        group_id = fixed_groups.get(link_name, 0)
        vis.view_trimesh(
            name=f"{name}/{link_name}/{mesh_tag}",
            mesh=mesh,
            pose=base_pose @ pose,
            color=_color_from_group(group_id, palette=palette),
            opacity=alpha,
        )


def _collect_urdfs(proc_asset: str, proc_root_dir: Path | None = None) -> list[Path]:
    candidate_dirs: list[Path] = []
    if proc_root_dir is None:
        candidate_dirs.append(get_assets_dir() / f"proc_{proc_asset}")
    else:
        candidate_dirs.extend(
            [
                proc_root_dir / f"proc_{proc_asset}",
                proc_root_dir / proc_asset,
            ]
        )
        if proc_root_dir.name in {proc_asset, f"proc_{proc_asset}"}:
            candidate_dirs.append(proc_root_dir)

    proc_dir = next((d for d in candidate_dirs if d.exists()), None)
    if proc_dir is None:
        return []

    # New layout: assets/proc_<asset>/<id>/<id>.urdf (ignore *_6dof.urdf).
    nested = []
    for d in sorted(
        [p for p in proc_dir.iterdir() if p.is_dir()],
        key=lambda p: (not p.name.isdigit(), p.name),
    ):
        cand = d / f"{d.name}.urdf"
        if cand.exists():
            nested.append(cand)
            continue
        # Fallback: any non-6dof urdf directly inside the sample dir.
        alts = sorted(p for p in d.glob("*.urdf") if not p.name.endswith("_6dof.urdf"))
        if alts:
            nested.append(alts[0])
    if nested:
        return nested

    # Backward-compatible flat layout.
    prefixed = sorted(proc_dir.glob(f"{proc_asset}_*.urdf"))
    if prefixed:
        return [p for p in prefixed if not p.name.endswith("_6dof.urdf")]
    return sorted(
        p for p in proc_dir.glob("*.urdf") if not p.name.endswith("_6dof.urdf")
    )


def _get_arctic_urdf(proc_asset: str) -> Path | None:
    decomp_dir = get_assets_dir() / f"arctic_{proc_asset}" / "decomp"
    if not decomp_dir.exists():
        return None
    exact = decomp_dir / f"{proc_asset}_decomp_abs.urdf"
    if exact.exists():
        return exact
    matches = sorted(decomp_dir.glob("*_decomp_abs.urdf"))
    if matches:
        return matches[0]
    return None


def _compute_alignment_T(
    asset_name: str,
    proc_urdf_path: Path,
    proc_articulation_deg: float | None,
    arctic_articulation_deg: float | None,
) -> np.ndarray | None:
    try:
        from dexlink.src.retargeting.data.arctic_proc_alignment import (
            estimate_box_arctic_to_proc_alignment,
            estimate_ketchup_arctic_to_proc_alignment,
            estimate_laptop_arctic_to_proc_alignment,
            estimate_notebook_arctic_to_proc_alignment,
            estimate_waffleiron_arctic_to_proc_alignment,
        )
    except Exception as exc:
        print(f"[warn] could not import box alignment helper: {exc}")
        return None

    fn_map = {
        "box": estimate_box_arctic_to_proc_alignment,
        "laptop": estimate_laptop_arctic_to_proc_alignment,
        "notebook": estimate_notebook_arctic_to_proc_alignment,
        "waffleiron": estimate_waffleiron_arctic_to_proc_alignment,
        "ketchup": estimate_ketchup_arctic_to_proc_alignment,
    }
    fn = fn_map.get(asset_name)
    if fn is None:
        print(f"[warn] no alignment wrapper available for asset '{asset_name}'")
        return None

    kwargs = {"proc_urdf_path": proc_urdf_path}
    if proc_articulation_deg is not None:
        theta_proc = float(np.deg2rad(proc_articulation_deg))
        kwargs["theta_proc"] = theta_proc
        if arctic_articulation_deg is None:
            if asset_name == "box":
                kwargs["theta_arctic"] = float(np.pi - theta_proc)
            else:
                kwargs["theta_arctic"] = theta_proc
    if arctic_articulation_deg is not None:
        kwargs["theta_arctic"] = float(np.deg2rad(arctic_articulation_deg))
    try:
        res = fn(**kwargs)
    except Exception as exc:
        print(f"[warn] failed to estimate {asset_name} ARCTIC->proc alignment: {exc}")
        return None

    print(
        f"[{asset_name}-align] using T_arctic_to_proc estimate. "
        f"proc={proc_urdf_path.name}, theta_proc={np.rad2deg(res['theta_proc']):.2f} deg, "
        f"theta_arctic={np.rad2deg(res['theta_arctic']):.2f} deg"
    )
    print(f"[{asset_name}-align] T_arctic_to_proc =")
    print(np.array2string(res["T_arctic_to_proc"], precision=6, suppress_small=True))
    return res["T_arctic_to_proc"]


def main():
    parser = argparse.ArgumentParser(
        description="Visualize generated procedural URDFs in a grid."
    )
    parser.add_argument(
        "--proc_assets",
        type=str,
        default=",".join(DEFAULT_PROC_ASSETS),
        help="Comma-separated procedural asset names (e.g. box,laptop,notebook).",
    )
    parser.add_argument(
        "--proc_root_dir",
        type=str,
        default=None,
        help="Optional root directory containing generated procedural assets "
        "(expects proc_<asset>/... or <asset>/... subdirs).",
    )
    parser.add_argument(
        "--max_per_asset", type=int, default=5, help="Max URDFs per asset row."
    )
    parser.add_argument(
        "--row_spacing", type=float, default=0.65, help="Grid row spacing (meters)."
    )
    parser.add_argument(
        "--col_spacing", type=float, default=0.55, help="Grid col spacing (meters)."
    )
    parser.add_argument(
        "--z", type=float, default=0.0, help="Base z position for all objects."
    )
    parser.add_argument(
        "--proc_rpy_deg",
        type=float,
        nargs=3,
        default=[0.0, 0.0, 0.0],
        help="Euler rotation (deg) for procedural objects.",
    )
    parser.add_argument(
        "--arctic_rpy_deg",
        type=float,
        nargs=3,
        default=[0.0, 0.0, 0.0],
        help="Euler rotation (deg) for ARCTIC objects.",
    )
    parser.add_argument("--alpha", type=float, default=0.9, help="Mesh opacity.")
    parser.add_argument("--port", type=int, default=6000, help="Meshcat ZMQ port.")
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for articulated joint sampling.",
    )
    parser.add_argument(
        "--align_arctic_to_proc",
        action="store_true",
        help="Estimate/apply ARCTIC->procedural alignment transforms for each supported asset row using dexlink.src.rl.sim.arctic_proc_alignment.",
    )
    parser.add_argument(
        "--align_box_arctic_to_proc",
        action="store_true",
        help="Compatibility alias: only align the box row.",
    )
    parser.add_argument(
        "--align_proc_idx",
        type=int,
        default=0,
        help="Which procedural variant index per row to use when estimating alignment transforms.",
    )
    parser.add_argument(
        "--proc_art",
        "--proc_articulation",
        dest="proc_articulation",
        type=float,
        default=None,
        help="If set, use this exact articulation value in degrees for revolute/continuous joints (prismatic joints use the raw value in meters); clamped to limits when available.",
    )
    parser.add_argument(
        "--arctic_art",
        "--arctic_articulation",
        dest="arctic_articulation",
        type=float,
        default=None,
        help="If set, use this exact articulation value in degrees for revolute/continuous joints (prismatic joints use the raw value in meters); clamped to limits when available.",
    )
    args = parser.parse_args()

    proc_assets = _parse_proc_assets(args.proc_assets)
    proc_root_dir = (
        Path(args.proc_root_dir).expanduser().resolve()
        if args.proc_root_dir is not None
        else None
    )
    rows: list[tuple[str, Path | None, list[Path]]] = []
    for asset_name in proc_assets:
        urdfs = _collect_urdfs(asset_name, proc_root_dir=proc_root_dir)[
            : max(args.max_per_asset, 0)
        ]
        if not urdfs:
            print(f"[skip] no URDFs found for proc_{asset_name}")
            continue
        arctic_urdf = _get_arctic_urdf(asset_name)
        if arctic_urdf is None:
            print(
                f"[warn] no ARCTIC URDF found for arctic_{asset_name}; row will show procedural only"
            )
        rows.append((asset_name, arctic_urdf, urdfs))

    if not rows:
        if proc_root_dir is None:
            raise SystemExit(
                "No procedural URDFs found. Generate some with --gen first."
            )
        raise SystemExit(
            f"No procedural URDFs found under {proc_root_dir}. "
            "Generate some with --gen --out_dir=<that_dir> first."
        )

    vis = VizServer(port_vis=args.port)
    proc_rpy = np.deg2rad(np.asarray(args.proc_rpy_deg, dtype=float))
    arctic_rpy = np.deg2rad(np.asarray(args.arctic_rpy_deg, dtype=float))
    rng = np.random.default_rng(args.seed)
    alignment_flags_enabled = args.align_arctic_to_proc or args.align_box_arctic_to_proc
    alignment_T_by_asset: dict[str, np.ndarray] = {}
    if alignment_flags_enabled:
        align_only_box = args.align_box_arctic_to_proc and not args.align_arctic_to_proc
        for asset_name, _, urdfs in rows:
            if align_only_box and asset_name != "box":
                continue
            if not urdfs:
                continue
            idx = int(np.clip(args.align_proc_idx, 0, len(urdfs) - 1))
            T = _compute_alignment_T(
                asset_name=asset_name,
                proc_urdf_path=urdfs[idx],
                proc_articulation_deg=args.proc_articulation,
                arctic_articulation_deg=args.arctic_articulation,
            )
            if T is not None:
                alignment_T_by_asset[asset_name] = T

    n_rows = len(rows)
    n_cols = max(
        len(urdfs) + (1 if arctic_urdf is not None else 0)
        for _, arctic_urdf, urdfs in rows
    )
    origin_x = -0.5 * (max(n_cols - 1, 0) * args.col_spacing)
    origin_y = -0.5 * (max(n_rows - 1, 0) * args.row_spacing)

    for row_idx, (asset_name, arctic_urdf, urdfs) in enumerate(rows):
        y = origin_y + row_idx * args.row_spacing
        col_offset = 0
        if arctic_urdf is not None:
            x = origin_x
            base_pose = _grid_pose([x, y, args.z], arctic_rpy)
            if asset_name in alignment_T_by_asset:
                base_pose = base_pose @ alignment_T_by_asset[asset_name]
            obj_name = f"arctic_{asset_name}"
            print(f"[view] {obj_name}: {arctic_urdf}")
            _view_urdf(
                vis,
                arctic_urdf,
                obj_name,
                base_pose,
                alpha=args.alpha,
                rng=rng,
                palette="arctic",
                articulation_value=args.arctic_articulation,
            )
            col_offset = 1

        for col_idx, urdf_path in enumerate(urdfs):
            x = origin_x + (col_idx + col_offset) * args.col_spacing
            base_pose = _grid_pose([x, y, args.z], proc_rpy)
            obj_name = f"{asset_name}_{col_idx}"
            print(f"[view] {obj_name}: {urdf_path}")
            _view_urdf(
                vis,
                urdf_path,
                obj_name,
                base_pose,
                alpha=args.alpha,
                rng=rng,
                palette="proc",
                articulation_value=args.proc_articulation,
            )

    print("Grid loaded in MeshCat. Keep this process running while viewing.")


if __name__ == "__main__":
    main()
