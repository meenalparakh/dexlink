import random
import sys

from dexlink.src.reference.procedural_assets.proc_asset_io import (
    save_proc_generation_params,
    save_proc_variation_pair,
)
from dexlink.utils.retargeting.path_utils import get_assets_dir

# procedural laptop:
# - bottom: solid cuboid base
# - top: solid cuboid screen
# - top connected to bottom via revolute hinge


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

    return width, length, bottom_height, top_height


def _cuboid_link_xml(
    link_name,
    width,
    length,
    height,
    mass,
    x_center=0.0,
    y_center=0.0,
    z_center=0.0,
):
    size = f"{width} {length} {height}"
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


def construct_laptop(params_dict):
    width, length, bottom_height, top_height = _validate_params(params_dict)

    robot_name = params_dict.get("name", "procedural_laptop")
    hinge_lower = float(params_dict.get("hinge_lower", 0.0))
    hinge_upper = float(params_dict.get("hinge_upper", 2.6))
    hinge_effort = float(params_dict.get("hinge_effort", 50.0))
    hinge_velocity = float(params_dict.get("hinge_velocity", 50.0))
    bottom_mass = float(params_dict.get("bottom_mass", 1.0))
    top_mass = float(params_dict.get("top_mass", 0.2))
    top_width_scale = float(params_dict.get("top_width_scale", 1.06))
    if top_width_scale <= 0:
        raise ValueError("top_width_scale must be > 0.")

    top_width = width * top_width_scale

    bottom_link = _cuboid_link_xml(
        link_name="bottom",
        width=width,
        length=length,
        height=bottom_height,
        mass=bottom_mass,
        z_center=0.5 * bottom_height,
    )

    # Top link frame is on the hinge edge (outer side of the top cuboid).
    top_link = _cuboid_link_xml(
        link_name="top",
        width=top_width,
        length=length,
        height=top_height,
        mass=top_mass,
        x_center=-0.5 * top_width,
        z_center=0.5 * top_height,
    )

    # Hinge along length (y-axis), on top rear edge of the bottom cuboid.
    hinge_x = 0.5 * width
    hinge_z = bottom_height

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
    <origin xyz="{hinge_x} 0 {hinge_z}" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <dynamics damping="0.01" friction="0.0"/>
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
    width = (0.18, 0.22)
    length = (0.2, 0.32)
    bottom_height = (0.008, 0.015)
    top_height = (0.004, 0.015)
    top_width_scale = (0.98, 1.05)
    variation_params = []

    for i in range(n):
        params = dict(
            name=f"procedural_laptop_{i}",
            width=str(rng.uniform(*width)),
            length=str(rng.uniform(*length)),
            bottom_height=str(rng.uniform(*bottom_height)),
            top_height=str(rng.uniform(*top_height)),
            top_width_scale=str(rng.uniform(*top_width_scale)),
        )
        variation_params.append({"sample_id": i + 1, "params": params})
        urdf = construct_laptop(params)
        urdf_path, urdf_6dof_path = save_proc_variation_pair(
            "laptop", i, urdf, output_root=output_root
        )
        print(f"URDF saved to {urdf_path}")
        print(f"URDF saved to {urdf_6dof_path}")
    params_path = save_proc_generation_params("laptop", variation_params, seed=seed)
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
            "Usage: python laptop.py [--demo | --gen] [--seed=<int>] [--out_dir=<path>]"
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
            "Usage: python laptop.py [--demo | --gen] [--seed=<int>] [--out_dir=<path>]"
        )

    params = dict(
        width=0.16,
        length=0.24,
        bottom_height=0.015,
        top_height=0.01,
        top_width_scale=1.03,
    )

    urdf = construct_laptop(params)
    urdf_path = get_assets_dir() / "proc_laptop" / "laptop.urdf"
    urdf_path.parent.mkdir(parents=True, exist_ok=True)
    with open(urdf_path, "w", encoding="utf-8") as f:
        f.write(urdf)

    print(f"URDF saved to {urdf_path}")
