import math
import random
import sys

from dexlink.src.reference.procedural_assets.proc_asset_io import (
    save_proc_generation_params,
    save_proc_variation_pair,
)
from dexlink.utils.retargeting.path_utils import get_assets_dir

# write urdf with primitive geometry nodes to create a box
# the box has top and bottom parts
# each part (top and bottom) is bin-shaped
# each should ideally use the same function, but with different depth, and matching length and width
# the top part is connected via a revolute joint with the bottom part.
# the urdf should be procedural (allow for a configurable depth, width, and height and thickness)


def _validate_params(params_dict):
    required = ["width", "length", "bottom_height", "top_height", "thickness"]
    missing = [k for k in required if k not in params_dict]
    if missing:
        raise ValueError(f"Missing required params: {missing}")

    width = float(params_dict["width"])
    length = float(params_dict["length"])
    bottom_height = float(params_dict["bottom_height"])
    top_height = float(params_dict["top_height"])
    thickness = float(params_dict["thickness"])

    if min(width, length, bottom_height, top_height, thickness) <= 0:
        raise ValueError("All dimensions must be > 0.")
    if thickness >= 0.5 * min(width, length):
        raise ValueError("thickness must be smaller than half of width/length.")

    return width, length, bottom_height, top_height, thickness


def _bin_link_xml(
    link_name,
    width,
    length,
    height,
    thickness,
    mass,
    x_offset=0.0,
    y_offset=0.0,
    inverted=False,
):
    # A bin is built from 5 primitive boxes:
    # one base plate + four side walls.
    half_w = width / 2.0
    half_l = length / 2.0
    if inverted:
        # Open rim is at z=0; geometry extends in -z direction.
        base_z = -(height + 0.5 * thickness)
        wall_z = -0.5 * height
        inertial_z = -0.5 * height
    else:
        base_z = 0.5 * thickness
        wall_z = 0.5 * (height + thickness)
        inertial_z = 0.5 * height

    base_size = f"{width} {length} {thickness}"
    long_wall_size = f"{width} {thickness} {height}"
    short_wall_size = f"{thickness} {length - 2.0 * thickness} {height}"

    return f"""
  <link name="{link_name}">
    <inertial>
      <origin xyz="{x_offset} {y_offset} {inertial_z}" rpy="0 0 0"/>
      <mass value="{mass}"/>
      <inertia ixx="1e-4" ixy="0" ixz="0" iyy="1e-4" iyz="0" izz="1e-4"/>
    </inertial>

    <visual>
      <origin xyz="{x_offset} {y_offset} {base_z}" rpy="0 0 0"/>
      <geometry><box size="{base_size}"/></geometry>
    </visual>
    <collision>
      <origin xyz="{x_offset} {y_offset} {base_z}" rpy="0 0 0"/>
      <geometry><box size="{base_size}"/></geometry>
    </collision>

    <visual>
      <origin xyz="{x_offset} {y_offset + half_l - 0.5 * thickness} {wall_z}" rpy="0 0 0"/>
      <geometry><box size="{long_wall_size}"/></geometry>
    </visual>
    <collision>
      <origin xyz="{x_offset} {y_offset + half_l - 0.5 * thickness} {wall_z}" rpy="0 0 0"/>
      <geometry><box size="{long_wall_size}"/></geometry>
    </collision>

    <visual>
      <origin xyz="{x_offset} {y_offset - half_l + 0.5 * thickness} {wall_z}" rpy="0 0 0"/>
      <geometry><box size="{long_wall_size}"/></geometry>
    </visual>
    <collision>
      <origin xyz="{x_offset} {y_offset - half_l + 0.5 * thickness} {wall_z}" rpy="0 0 0"/>
      <geometry><box size="{long_wall_size}"/></geometry>
    </collision>

    <visual>
      <origin xyz="{x_offset + half_w - 0.5 * thickness} {y_offset} {wall_z}" rpy="0 0 0"/>
      <geometry><box size="{short_wall_size}"/></geometry>
    </visual>
    <collision>
      <origin xyz="{x_offset + half_w - 0.5 * thickness} {y_offset} {wall_z}" rpy="0 0 0"/>
      <geometry><box size="{short_wall_size}"/></geometry>
    </collision>

    <visual>
      <origin xyz="{x_offset - half_w + 0.5 * thickness} {y_offset} {wall_z}" rpy="0 0 0"/>
      <geometry><box size="{short_wall_size}"/></geometry>
    </visual>
    <collision>
      <origin xyz="{x_offset - half_w + 0.5 * thickness} {y_offset} {wall_z}" rpy="0 0 0"/>
      <geometry><box size="{short_wall_size}"/></geometry>
    </collision>
  </link>"""


def construct_box(params_dict):
    width, length, bottom_height, top_height, thickness = _validate_params(params_dict)

    robot_name = params_dict.get("name", "procedural_box")
    hinge_lower = float(params_dict.get("hinge_lower", math.pi / 6))
    hinge_upper = float(params_dict.get("hinge_upper", math.pi))
    hinge_effort = float(params_dict.get("hinge_effort", 50.0))
    hinge_velocity = float(params_dict.get("hinge_velocity", 50.0))
    bottom_mass = float(params_dict.get("bottom_mass", 1.0))
    top_mass = float(params_dict.get("top_mass", 0.2))

    bottom_link = _bin_link_xml(
        link_name="bottom",
        width=width,
        length=length,
        height=bottom_height,
        thickness=thickness,
        mass=bottom_mass,
    )
    top_link = _bin_link_xml(
        link_name="top",
        width=width + thickness,
        length=length,
        height=top_height,
        thickness=thickness,
        mass=top_mass,
        # Put top frame on the OUTER hinge edge (open rim) of the lid.
        x_offset=0.5 * width + 0.5 * thickness,
        inverted=True,
    )

    # Hinge runs along the length dimension (y-axis in this convention).
    hinge_x = 0.5 * width + 0.5 * thickness
    hinge_z = bottom_height + thickness

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
    <axis xyz="0 -1 0"/>
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
    width = (0.12, 0.28)
    length = (0.18, 0.40)
    bottom_height = (0.05, 0.15)
    top_height = (0.02, 0.05)
    thickness = (0.003, 0.01)
    max_total_height = 0.22
    variation_params = []

    for i in range(n):
        w = rng.uniform(*width)
        length_value = rng.uniform(*length)
        t = min(rng.uniform(*thickness), 0.45 * min(w, length_value))
        for _ in range(64):
            bh = rng.uniform(*bottom_height)
            th = rng.uniform(*top_height)
            if bh + th < max_total_height:
                break
        else:
            bh = bottom_height[0]
            th = min(top_height[0], max_total_height - bh - 1e-4)
        params = dict(
            name=f"procedural_box_{i}",
            width=str(w),
            length=str(length_value),
            bottom_height=str(bh),
            top_height=str(th),
            thickness=str(t),
        )
        variation_params.append({"sample_id": i + 1, "params": params})
        urdf = construct_box(params)
        urdf_path, urdf_6dof_path = save_proc_variation_pair(
            "box", i, urdf, output_root=output_root
        )
        print(f"URDF saved to {urdf_path}")
        print(f"URDF saved to {urdf_6dof_path}")
    params_path = save_proc_generation_params("box", variation_params, seed=seed)
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
            "Usage: python box.py [--demo | --gen] [--seed=<int>] [--out_dir=<path>]"
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
            "Usage: python box.py [--demo | --gen] [--seed=<int>] [--out_dir=<path>]"
        )

    params = dict(
        width=0.2,
        length=0.35,
        bottom_height=0.12,
        top_height=0.04,
        thickness=0.005,
    )

    urdf = construct_box(params)

    # save urdf
    urdf_path = get_assets_dir() / "proc_box" / "box.urdf"
    urdf_path.parent.mkdir(parents=True, exist_ok=True)
    with open(urdf_path, "w", encoding="utf-8") as f:
        f.write(urdf)

    print(f"URDF saved to {urdf_path}")
