import math
import random
import sys

from dexlink.src.reference.procedural_assets.proc_asset_io import (
    save_proc_generation_params,
    save_proc_variation_pair,
)
from dexlink.utils.retargeting.path_utils import get_assets_dir

# create a ketchup bottle
# with top and bottom parts
# the top part is cylindrical, connected to the bottom via a revolute link
# the bottom part is also cylindrical, but scaled differently to look like elliptical cross-section


def _validate_params(params_dict):
    required = ["width", "length", "bottom_height", "top_height"]
    missing = [k for k in required if k not in params_dict]
    if missing:
        raise ValueError(f"Missing required params: {missing}")

    width = float(params_dict["width"])
    length = float(params_dict["length"])
    bottom_height = float(params_dict["bottom_height"])
    top_height = float(params_dict["top_height"])

    if min(width, length, bottom_height, top_height) <= 0:
        raise ValueError("All dimensions must be > 0.")

    top_radius = float(params_dict.get("top_radius", 0.25 * min(width, length)))
    if top_radius <= 0:
        raise ValueError("top_radius must be > 0.")

    approx_slices = int(params_dict.get("ellipse_slices", 9))
    if approx_slices < 3:
        raise ValueError("ellipse_slices must be >= 3.")

    neck_height = float(params_dict.get("neck_height", 0.25 * top_height))
    neck_radius = float(params_dict.get("neck_radius", top_radius))
    if neck_height <= 0:
        raise ValueError("neck_height must be > 0.")
    if neck_radius <= 0:
        raise ValueError("neck_radius must be > 0.")
    if neck_radius >= 0.5 * min(width, length):
        raise ValueError("neck_radius must be smaller than half of width and length.")

    return (
        width,
        length,
        bottom_height,
        top_height,
        top_radius,
        approx_slices,
        neck_height,
        neck_radius,
    )


def _single_cylinder_xml(link_name, radius, height, mass, x_offset=0.0, y_offset=0.0):
    z_center = 0.5 * height
    return f"""
  <link name="{link_name}">
    <inertial>
      <origin xyz="{x_offset} {y_offset} {z_center}" rpy="0 0 0"/>
      <mass value="{mass}"/>
      <inertia ixx="1e-4" ixy="0" ixz="0" iyy="1e-4" iyz="0" izz="1e-4"/>
    </inertial>

    <visual>
      <origin xyz="{x_offset} {y_offset} {z_center}" rpy="0 0 0"/>
      <geometry><cylinder radius="{radius}" length="{height}"/></geometry>
    </visual>
    <collision>
      <origin xyz="{x_offset} {y_offset} {z_center}" rpy="0 0 0"/>
      <geometry><cylinder radius="{radius}" length="{height}"/></geometry>
    </collision>
  </link>"""


def _elliptical_cylinder_xml(
    link_name,
    radius_x,
    radius_y,
    height,
    mass,
    slices=9,
):
    # URDF primitive cylinders are circular. Approximate an elliptical cross-section
    # by union of shifted cylinders with varying radii.
    z_center = 0.5 * height

    if abs(radius_x - radius_y) < 1e-9:
        return _single_cylinder_xml(link_name, radius_x, height, mass)

    major_is_x = radius_x >= radius_y
    r_major = radius_x if major_is_x else radius_y
    r_minor = radius_y if major_is_x else radius_x

    visuals = []
    collisions = []
    for i in range(slices):
        t = -1.0 + 2.0 * i / (slices - 1)
        major_coord = t * r_major
        local_minor = r_minor * math.sqrt(max(0.0, 1.0 - (major_coord / r_major) ** 2))
        if local_minor <= 1e-6:
            continue

        if major_is_x:
            x, y = major_coord, 0.0
        else:
            x, y = 0.0, major_coord

        geom = (
            f'<geometry><cylinder radius="{local_minor}" length="{height}"/></geometry>'
        )
        visuals.append(
            f"""
    <visual>
      <origin xyz="{x} {y} {z_center}" rpy="0 0 0"/>
      {geom}
    </visual>"""
        )
        collisions.append(
            f"""
    <collision>
      <origin xyz="{x} {y} {z_center}" rpy="0 0 0"/>
      {geom}
    </collision>"""
        )

    return f"""
  <link name="{link_name}">
    <inertial>
      <origin xyz="0 0 {z_center}" rpy="0 0 0"/>
      <mass value="{mass}"/>
      <inertia ixx="1e-4" ixy="0" ixz="0" iyy="1e-4" iyz="0" izz="1e-4"/>
    </inertial>
{''.join(visuals)}
{''.join(collisions)}
  </link>"""


def _bottom_with_neck_xml(
    width,
    length,
    bottom_height,
    neck_height,
    neck_radius,
    mass,
    slices=9,
):
    # Build one link composed of:
    # 1) an elliptical-like main body
    # 2) a cylindrical neck on top
    radius_x = 0.5 * width
    radius_y = 0.5 * length

    major_is_x = radius_x >= radius_y
    r_major = radius_x if major_is_x else radius_y
    r_minor = radius_y if major_is_x else radius_x

    body_visuals = []
    body_collisions = []
    for i in range(slices):
        t = -1.0 + 2.0 * i / (slices - 1)
        major_coord = t * r_major
        local_minor = r_minor * math.sqrt(max(0.0, 1.0 - (major_coord / r_major) ** 2))
        if local_minor <= 1e-6:
            continue

        if major_is_x:
            x, y = major_coord, 0.0
        else:
            x, y = 0.0, major_coord

        geom = f'<geometry><cylinder radius="{local_minor}" length="{bottom_height}"/></geometry>'
        body_visuals.append(
            f"""
    <visual>
      <origin xyz="{x} {y} {0.5 * bottom_height}" rpy="0 0 0"/>
      {geom}
    </visual>"""
        )
        body_collisions.append(
            f"""
    <collision>
      <origin xyz="{x} {y} {0.5 * bottom_height}" rpy="0 0 0"/>
      {geom}
    </collision>"""
        )

    neck_z = bottom_height + 0.5 * neck_height
    neck_geom = f'<geometry><cylinder radius="{neck_radius}" length="{neck_height}"/></geometry>'
    neck_visual = f"""
    <visual>
      <origin xyz="0 0 {neck_z}" rpy="0 0 0"/>
      {neck_geom}
    </visual>"""
    neck_collision = f"""
    <collision>
      <origin xyz="0 0 {neck_z}" rpy="0 0 0"/>
      {neck_geom}
    </collision>"""

    # Approximate COM using body+neck volumes.
    body_vol = math.pi * radius_x * radius_y * bottom_height
    neck_vol = math.pi * neck_radius * neck_radius * neck_height
    z_com = (
        body_vol * (0.5 * bottom_height)
        + neck_vol * (bottom_height + 0.5 * neck_height)
    ) / max(body_vol + neck_vol, 1e-9)

    return f"""
  <link name="bottom">
    <inertial>
      <origin xyz="0 0 {z_com}" rpy="0 0 0"/>
      <mass value="{mass}"/>
      <inertia ixx="1e-4" ixy="0" ixz="0" iyy="1e-4" iyz="0" izz="1e-4"/>
    </inertial>
{''.join(body_visuals)}
{''.join(body_collisions)}
{neck_visual}
{neck_collision}
  </link>"""


def construct_ketchup(params_dict):
    (
        width,
        length,
        bottom_height,
        top_height,
        top_radius,
        ellipse_slices,
        neck_height,
        neck_radius,
    ) = _validate_params(params_dict)

    robot_name = params_dict.get("name", "procedural_ketchup")
    hinge_lower = float(params_dict.get("hinge_lower", 0.0))
    hinge_upper = float(params_dict.get("hinge_upper", 2.85))
    hinge_effort = float(params_dict.get("hinge_effort", 1000.0))
    hinge_velocity = float(params_dict.get("hinge_velocity", 200.0))
    bottom_mass = float(params_dict.get("bottom_mass", 0.6))
    top_mass = float(params_dict.get("top_mass", 0.08))

    bottom_link = _bottom_with_neck_xml(
        width=width,
        length=length,
        bottom_height=bottom_height,
        neck_height=neck_height,
        neck_radius=neck_radius,
        mass=bottom_mass,
        slices=ellipse_slices,
    )
    top_link = _single_cylinder_xml(
        link_name="top",
        radius=top_radius,
        height=top_height,
        mass=top_mass,
        # Put top frame on the outer hinge edge of the cap.
        x_offset=top_radius,
    )

    # Hinge along bottle length direction (y-axis), on the neck top edge.
    hinge_x = neck_radius
    hinge_z = bottom_height + neck_height

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

  <joint name="rotation" type="revolute">
    <parent link="bottom"/>
    <child link="top"/>
    <origin xyz="{-hinge_x} 0 {hinge_z}" rpy="0 0 0"/>
    <axis xyz="0 -1 0"/>
    <dynamics damping="0.05" friction="0.0001"/>
    <limit effort="{hinge_effort}" velocity="{hinge_velocity}" lower="{hinge_lower}" upper="{hinge_upper}"/>
  </joint>
{top_link}
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
    width = (0.03, 0.06)
    length = (0.05, 0.1)
    bottom_height = (0.1, 0.2)
    top_height = (0.005, 0.015)
    neck_height = (0.01, 0.05)
    neck_radius = (0.01, 0.035)
    top_radius_scale = (1.01, 1.1)
    variation_params = []

    for i in range(n):
        w = rng.uniform(*width)
        length_value = rng.uniform(*length)
        nr_max = min(neck_radius[1], 0.5 * min(w, length_value) * 0.99)
        nr_min = neck_radius[0]
        if nr_max <= nr_min:
            nr = (
                0.5 * (nr_min + nr_max)
                if nr_max > 0
                else 0.25 * min(w, length_value)
            )
            nr = min(nr, 0.5 * min(w, length_value) * 0.99)
        else:
            nr = rng.uniform(nr_min, nr_max)
        params = dict(
            name=f"procedural_ketchup_{i}",
            width=str(w),
            length=str(length_value),
            bottom_height=str(rng.uniform(*bottom_height)),
            top_height=str(rng.uniform(*top_height)),
            top_radius=str(nr * rng.uniform(*top_radius_scale)),
            neck_height=str(rng.uniform(*neck_height)),
            neck_radius=str(nr),
        )
        variation_params.append({"sample_id": i + 1, "params": params})
        urdf = construct_ketchup(params)
        urdf_path, urdf_6dof_path = save_proc_variation_pair(
            "ketchup", i, urdf, output_root=output_root
        )
        print(f"URDF saved to {urdf_path}")
        print(f"URDF saved to {urdf_6dof_path}")
    params_path = save_proc_generation_params("ketchup", variation_params, seed=seed)
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
            "Usage: python ketchup.py [--demo | --gen] [--seed=<int>] [--out_dir=<path>]"
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
            "Usage: python ketchup.py [--demo | --gen] [--seed=<int>] [--out_dir=<path>]"
        )

    params = dict(
        width=0.07,
        length=0.1,
        bottom_height=0.20,
        top_height=0.01,
        top_radius=0.027,
        neck_height=0.03,
        neck_radius=0.025,
        ellipse_slices=11,
    )

    urdf = construct_ketchup(params)
    urdf_path = get_assets_dir() / "proc_ketchup" / "ketchup.urdf"
    urdf_path.parent.mkdir(parents=True, exist_ok=True)
    with open(urdf_path, "w", encoding="utf-8") as f:
        f.write(urdf)

    print(f"URDF saved to {urdf_path}")
