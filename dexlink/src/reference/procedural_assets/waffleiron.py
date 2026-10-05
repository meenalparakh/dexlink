import math
import random
import sys

from dexlink.src.reference.procedural_assets.proc_asset_io import (
    save_proc_generation_params,
    save_proc_variation_pair,
)
from dexlink.utils.retargeting.path_utils import get_assets_dir

# procedural waffle iron:
# - bottom: thin cylinder ("plate")
# - top: thin cylinder ("lid")
# - top connected to bottom via revolute hinge on the rim


def _validate_params(params_dict):
    required = ["radius", "bottom_thickness", "top_thickness"]
    missing = [k for k in required if k not in params_dict]
    if missing:
        raise ValueError(f"Missing required params: {missing}")

    radius = float(params_dict["radius"])
    bottom_thickness = float(params_dict["bottom_thickness"])
    top_thickness = float(params_dict["top_thickness"])

    if min(radius, bottom_thickness, top_thickness) <= 0:
        raise ValueError("All dimensions must be > 0.")

    return radius, bottom_thickness, top_thickness


def _cylinder_link_xml(
    link_name,
    radius,
    length,
    mass,
    x_center=0.0,
    y_center=0.0,
    z_center=0.0,
):
    return f"""
  <link name="{link_name}">
    <inertial>
      <origin xyz="{x_center} {y_center} {z_center}" rpy="0 0 0"/>
      <mass value="{mass}"/>
      <inertia ixx="1e-4" ixy="0" ixz="0" iyy="1e-4" iyz="0" izz="1e-4"/>
    </inertial>

    <visual>
      <origin xyz="{x_center} {y_center} {z_center}" rpy="0 0 0"/>
      <geometry><cylinder radius="{radius}" length="{length}"/></geometry>
    </visual>
    <collision>
      <origin xyz="{x_center} {y_center} {z_center}" rpy="0 0 0"/>
      <geometry><cylinder radius="{radius}" length="{length}"/></geometry>
    </collision>
  </link>"""


def _cuboid_link_xml(
    link_name,
    size_x,
    size_y,
    size_z,
    mass,
    x_center=0.0,
    y_center=0.0,
    z_center=0.0,
):
    size = f"{size_x} {size_y} {size_z}"
    return f"""
  <link name="{link_name}">
    <inertial>
      <origin xyz="{x_center} {y_center} {z_center}" rpy="0 0 0"/>
      <mass value="{mass}"/>
      <inertia ixx="1e-4" ixy="0" ixz="0" iyy="1e-4" iyz="0" izz="1e-4"/>
    </inertial>

    <visual>
      <origin xyz="{x_center} {y_center} {z_center}" rpy="0 0 0"/>
      <geometry><box size="{size}"/></geometry>
    </visual>
    <collision>
      <origin xyz="{x_center} {y_center} {z_center}" rpy="0 0 0"/>
      <geometry><box size="{size}"/></geometry>
    </collision>
  </link>"""


def construct_waffleiron(params_dict):
    radius, bottom_thickness, top_thickness = _validate_params(params_dict)

    robot_name = params_dict.get("name", "procedural_waffleiron")
    hinge_lower = float(params_dict.get("hinge_lower", 0.0))
    hinge_upper = float(params_dict.get("hinge_upper", 2.4))
    hinge_effort = float(params_dict.get("hinge_effort", 50.0))
    hinge_velocity = float(params_dict.get("hinge_velocity", 50.0))
    bottom_mass = float(params_dict.get("bottom_mass", 1.0))
    top_mass = float(params_dict.get("top_mass", 0.6))
    support_mass = float(params_dict.get("support_mass", 0.2))
    grip_mass = float(params_dict.get("grip_mass", 0.08))
    top_radius_scale = float(params_dict.get("top_radius_scale", 1.0))
    if top_radius_scale <= 0:
        raise ValueError("top_radius_scale must be > 0.")

    top_radius = radius * top_radius_scale
    support_size_x = float(params_dict.get("support_size_x", 0.02))
    support_size_y = float(params_dict.get("support_size_y", 0.06))
    support_size_z = float(params_dict.get("support_size_z", 0.01))
    support_x_offset = float(params_dict.get("support_x_offset", 0.5 * support_size_x))
    # Default places the support centered on the hinge height (not entirely above it).
    support_z_offset = float(params_dict.get("support_z_offset", 0.0))
    if min(support_size_x, support_size_y, support_size_z) <= 0:
        raise ValueError("Support dimensions must be > 0.")

    grip_size_x = float(params_dict.get("grip_size_x", 0.04))
    grip_size_y = float(params_dict.get("grip_size_y", 0.05))
    grip_size_z = float(params_dict.get("grip_size_z", 0.01))
    grip_overlap_x = float(params_dict.get("grip_overlap_x", 0.006))
    grip_tilt_deg = float(params_dict.get("grip_tilt_deg", 12.0))
    bottom_grip_pitch_deg = float(
        params_dict.get("bottom_grip_pitch_deg", -grip_tilt_deg)
    )
    top_grip_pitch_deg = float(params_dict.get("top_grip_pitch_deg", grip_tilt_deg))
    bottom_grip_z_offset = float(params_dict.get("bottom_grip_z_offset", 0.0))
    top_grip_z_offset = float(params_dict.get("top_grip_z_offset", 0.0))
    if min(grip_size_x, grip_size_y, grip_size_z) <= 0:
        raise ValueError("Grip dimensions must be > 0.")

    # Bottom link frame at the cylinder center.
    bottom_link = _cylinder_link_xml(
        link_name="bottom",
        radius=radius,
        length=bottom_thickness,
        mass=bottom_mass,
        z_center=0.5 * bottom_thickness,
    )

    # Top link frame is placed at the hinge line (rim point, lower face plane).
    # URDF cylinders are centered on their link frame by default, so offset the cylinder center.
    top_link = _cylinder_link_xml(
        link_name="top",
        radius=top_radius,
        length=top_thickness,
        mass=top_mass,
        x_center=-top_radius,
        z_center=0.5 * top_thickness,
    )

    # Hinge axis is tangent to the disk rim (parallel to y-axis), at the top of the bottom plate.
    hinge_x = radius
    hinge_z = bottom_thickness

    # Rear support block sits behind the hinge (positive x).
    # By default it is centered on the hinge height, so it spans above and below the hinge plane.
    support_link = _cuboid_link_xml(
        link_name="hinge_support",
        size_x=support_size_x,
        size_y=support_size_y,
        size_z=support_size_z,
        mass=support_mass,
    )
    support_center_x = hinge_x + support_x_offset * 0.8
    support_center_z = hinge_z + support_z_offset

    # Front grip tabs are opposite the hinge (negative x direction).
    bottom_grip_link = _cuboid_link_xml(
        link_name="bottom_grip",
        size_x=grip_size_x,
        size_y=grip_size_y,
        size_z=grip_size_z,
        mass=grip_mass,
    )
    top_grip_link = _cuboid_link_xml(
        link_name="top_grip",
        size_x=grip_size_x,
        size_y=grip_size_y,
        size_z=grip_size_z,
        mass=grip_mass,
    )

    bottom_front_rim_x = -radius
    bottom_grip_center_x = bottom_front_rim_x - 0.5 * grip_size_x + grip_overlap_x
    bottom_grip_center_z = 0.5 * bottom_thickness + bottom_grip_z_offset

    # In the top-link frame, hinge is at x=0 and the front rim is at x=-2*top_radius.
    top_front_rim_x = -2.0 * top_radius
    top_grip_center_x = top_front_rim_x - 0.5 * grip_size_x + grip_overlap_x
    top_grip_center_z = 0.5 * top_thickness + top_grip_z_offset

    bottom_grip_pitch = math.radians(bottom_grip_pitch_deg)
    top_grip_pitch = math.radians(top_grip_pitch_deg)

    urdf = f"""<?xml version="1.0" ?>
<robot name="{robot_name}">
  <link name="root">
    <inertial>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <mass value="0.01"/>
      <inertia ixx="1e-6" ixy="0" ixz="0" iyy="1e-6" iyz="0" izz="1e-6"/>
    </inertial>
  </link>

  <joint name="base_fixed" type="fixed">
    <parent link="root"/>
    <child link="bottom"/>
    <origin xyz="0 0 0" rpy="0 0 0"/>
  </joint>
{bottom_link}

  <joint name="hinge_support_fixed" type="fixed">
    <parent link="bottom"/>
    <child link="hinge_support"/>
    <origin xyz="{support_center_x} 0 {support_center_z}" rpy="0 0 0"/>
  </joint>
{support_link}

  <joint name="bottom_grip_fixed" type="fixed">
    <parent link="bottom"/>
    <child link="bottom_grip"/>
    <origin xyz="{bottom_grip_center_x} 0 {bottom_grip_center_z}" rpy="0 {bottom_grip_pitch} 0"/>
  </joint>
{bottom_grip_link}

  <joint name="rotation" type="revolute">
    <parent link="bottom"/>
    <child link="top"/>
    <origin xyz="{hinge_x} 0 {hinge_z}" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <dynamics damping="0.01" friction="0.0"/>
    <limit effort="{hinge_effort}" velocity="{hinge_velocity}" lower="{hinge_lower}" upper="{hinge_upper}"/>
  </joint>
{top_link}

  <joint name="top_grip_fixed" type="fixed">
    <parent link="top"/>
    <child link="top_grip"/>
    <origin xyz="{top_grip_center_x} 0 {top_grip_center_z}" rpy="0 {top_grip_pitch} 0"/>
  </joint>
{top_grip_link}
</robot>
"""
    return urdf


def _parse_seed_from_args(arg_list):
    for arg in arg_list:
        if arg.startswith("--seed="):
            _, _, raw_seed = arg.partition("=")
            if not raw_seed:
                raise SystemExit("Usage: --seed=<int>")
            try:
                return int(raw_seed)
            except ValueError as exc:
                raise SystemExit("--seed must be an integer.") from exc
    return None


def _parse_out_dir_from_args(arg_list):
    for arg in arg_list:
        if arg.startswith("--out_dir="):
            _, _, raw_path = arg.partition("=")
            if not raw_path:
                raise SystemExit("Usage: --out_dir=<path>")
            return raw_path
    return None


def create_variations(n=5, seed=42, output_root=None):
    rng = random.Random(seed) if seed is not None else random
    radius = (0.08, 0.15)
    bottom_thickness = (0.008, 0.02)
    top_thickness = (0.006, 0.016)
    top_radius_scale = (0.95, 1.02)
    support_size_x = (0.012, 0.03)
    support_size_y = (0.04, 0.10)
    support_size_z = (0.006, 0.02)
    support_z_offset = (-0.003, 0.003)
    grip_size_x = (0.015, 0.035)
    grip_size_y = (0.03, 0.07)
    grip_size_z = (0.003, 0.012)
    grip_overlap_x = (0.002, 0.01)
    grip_tilt_deg = (6.0, 18.0)
    bottom_grip_pitch_deg = (-18.0, -6.0)
    top_grip_pitch_deg = (6.0, 18.0)
    bottom_grip_z_offset = (-0.002, 0.002)
    top_grip_z_offset = (-0.002, 0.002)
    variation_params = []

    for i in range(n):
        bt = rng.uniform(*bottom_thickness)
        tt = rng.uniform(*top_thickness)

        params = dict(
            name=f"procedural_waffleiron_{i}",
            radius=str(rng.uniform(*radius)),
            bottom_thickness=str(bt),
            top_thickness=str(tt),
            top_radius_scale=str(rng.uniform(*top_radius_scale)),
            support_size_x=str(rng.uniform(*support_size_x)),
            support_size_y=str(rng.uniform(*support_size_y)),
            support_size_z=str(rng.uniform(*support_size_z)),
            support_z_offset=str(rng.uniform(*support_z_offset)),
            grip_size_x=str(rng.uniform(*grip_size_x)),
            grip_size_y=str(rng.uniform(*grip_size_y)),
            grip_size_z=str(rng.uniform(*grip_size_z)),
            grip_overlap_x=str(rng.uniform(*grip_overlap_x)),
            grip_tilt_deg=str(rng.uniform(*grip_tilt_deg)),
            bottom_grip_pitch_deg=str(rng.uniform(*bottom_grip_pitch_deg)),
            top_grip_pitch_deg=str(rng.uniform(*top_grip_pitch_deg)),
            bottom_grip_z_offset=str(rng.uniform(*bottom_grip_z_offset)),
            top_grip_z_offset=str(rng.uniform(*top_grip_z_offset)),
        )
        variation_params.append({"sample_id": i + 1, "params": params})
        urdf = construct_waffleiron(params)
        urdf_path, urdf_6dof_path = save_proc_variation_pair(
            "waffleiron", i, urdf, output_root=output_root
        )
        print(f"URDF saved to {urdf_path}")
        print(f"URDF saved to {urdf_6dof_path}")
    params_path = save_proc_generation_params("waffleiron", variation_params, seed=seed)
    print(f"Params saved to {params_path}")


if __name__ == "__main__":
    args_list = sys.argv[1:]
    args = set(args_list)
    unknown_args = [
        arg
        for arg in args_list
        if arg not in {"--demo", "--gen"}
        and not arg.startswith("--seed=")
        and not arg.startswith("--out_dir=")
    ]
    if unknown_args:
        raise SystemExit(
            "Usage: python waffleiron.py [--demo | --gen] [--seed=<int>] [--out_dir=<path>]"
        )
    seed = _parse_seed_from_args(args_list)
    out_dir = _parse_out_dir_from_args(args_list)
    if seed is None:
        seed = 42
    if "--gen" in args:
        create_variations(seed=seed, output_root=out_dir)
        sys.exit(0)
    if args and "--demo" not in args and "--gen" not in args:
        raise SystemExit(
            "Usage: python waffleiron.py [--demo | --gen] [--seed=<int>] [--out_dir=<path>]"
        )

    params = dict(
        radius=0.11,
        bottom_thickness=0.012,
        top_thickness=0.01,
        top_radius_scale=0.99,
        support_size_x=0.02,
        support_size_y=0.06,
        support_size_z=0.01,
        grip_size_x=0.02,
        grip_size_y=0.04,
        grip_size_z=0.005,
        grip_tilt_deg=10.0,
        bottom_grip_pitch_deg=-10.0,
        top_grip_pitch_deg=10.0,
        bottom_grip_z_offset=0.0,
        top_grip_z_offset=0.0,
    )

    urdf = construct_waffleiron(params)
    urdf_path = get_assets_dir() / "proc_waffleiron" / "waffleiron.urdf"
    urdf_path.parent.mkdir(parents=True, exist_ok=True)
    with open(urdf_path, "w", encoding="utf-8") as f:
        f.write(urdf)

    print(f"URDF saved to {urdf_path}")
