import random
import sys

from dexlink.src.reference.procedural_assets.proc_asset_io import (
    save_proc_generation_params,
    save_proc_variation_pair,
)
from dexlink.utils.retargeting.path_utils import get_assets_dir

# procedural notebook:
# - bottom: inner page block / notebook body (cuboid)
# - top: inner cover/page block (cuboid)
# - top connected to bottom via revolute hinge
# - larger rectangular hard-cover supports attached to both top and bottom


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


def construct_notebook(params_dict):
    width, length, bottom_height, top_height = _validate_params(params_dict)

    robot_name = params_dict.get("name", "procedural_notebook")
    hinge_lower = float(params_dict.get("hinge_lower", 0.0))
    hinge_upper = float(params_dict.get("hinge_upper", 3.0))
    hinge_effort = float(params_dict.get("hinge_effort", 50.0))
    hinge_velocity = float(params_dict.get("hinge_velocity", 50.0))

    bottom_mass = float(params_dict.get("bottom_mass", 0.5))
    top_mass = float(params_dict.get("top_mass", 0.35))
    bottom_cover_mass = float(params_dict.get("bottom_cover_mass", 0.2))
    top_cover_mass = float(params_dict.get("top_cover_mass", 0.2))

    top_width_scale = float(params_dict.get("top_width_scale", 1.0))
    cover_width_scale = float(params_dict.get("cover_width_scale", 1.08))
    cover_length_scale = float(params_dict.get("cover_length_scale", 1.06))
    bottom_cover_thickness = float(params_dict.get("bottom_cover_thickness", 0.003))
    top_cover_thickness = float(params_dict.get("top_cover_thickness", 0.003))

    if (
        min(
            top_width_scale,
            cover_width_scale,
            cover_length_scale,
            bottom_cover_thickness,
            top_cover_thickness,
        )
        <= 0
    ):
        raise ValueError("Scales and cover thicknesses must be > 0.")

    top_width = width * top_width_scale
    cover_width = width * cover_width_scale
    cover_length = length * cover_length_scale
    top_cover_width = top_width * cover_width_scale

    bottom_link = _cuboid_link_xml(
        link_name="bottom",
        width=width,
        length=length,
        height=bottom_height,
        mass=bottom_mass,
        z_center=0.5 * bottom_height,
    )

    # Top link frame is on the hinge edge, matching laptop.py convention.
    top_link = _cuboid_link_xml(
        link_name="top",
        width=top_width,
        length=length,
        height=top_height,
        mass=top_mass,
        x_center=-0.5 * top_width,
        z_center=0.5 * top_height,
    )

    # Hard covers are larger than the inner bodies and attached as fixed links.
    bottom_cover_link = _cuboid_link_xml(
        link_name="bottom_cover",
        width=cover_width,
        length=cover_length,
        height=bottom_cover_thickness,
        mass=bottom_cover_mass,
    )

    top_cover_link = _cuboid_link_xml(
        link_name="top_cover",
        width=top_cover_width,
        length=cover_length,
        height=top_cover_thickness,
        mass=top_cover_mass,
    )

    hinge_x = 0.5 * width
    hinge_z = bottom_height

    # Bottom cover ("back cover") is shifted so its back edge aligns with the page block
    # at the hinge/spine side (no protrusion past the spine).
    bottom_cover_center_x = 0.5 * (width - cover_width)
    bottom_cover_center_z = -0.5 * bottom_cover_thickness

    # Top cover is attached on top of the top link, using the top link's hinge-edge frame.
    top_cover_center_x = -0.5 * top_cover_width
    top_cover_center_z = top_height + 0.5 * top_cover_thickness

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

  <joint name="bottom_cover_fixed" type="fixed">
    <parent link="bottom"/>
    <child link="bottom_cover"/>
    <origin xyz="{bottom_cover_center_x} 0 {bottom_cover_center_z}" rpy="0 0 0"/>
  </joint>
{bottom_cover_link}

  <joint name="rotation" type="revolute">
    <parent link="bottom"/>
    <child link="top"/>
    <origin xyz="{hinge_x} 0 {hinge_z}" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <dynamics damping="0.01" friction="0.0"/>
    <limit effort="{hinge_effort}" velocity="{hinge_velocity}" lower="{hinge_lower}" upper="{hinge_upper}"/>
  </joint>
{top_link}

  <joint name="top_cover_fixed" type="fixed">
    <parent link="top"/>
    <child link="top_cover"/>
    <origin xyz="{top_cover_center_x} 0 {top_cover_center_z}" rpy="0 0 0"/>
  </joint>
{top_cover_link}
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
    width = (0.12, 0.2)
    length = (0.17, 0.25)
    bottom_height = (0.008, 0.02)
    top_height = (0.005, 0.015)
    cover_width_scale = (1.02, 1.12)
    cover_length_scale = (1.01, 1.10)
    cover_thickness = (0.0015, 0.005)
    variation_params = []

    for i in range(n):
        params = dict(
            name=f"procedural_notebook_{i}",
            width=str(rng.uniform(*width)),
            length=str(rng.uniform(*length)),
            bottom_height=str(rng.uniform(*bottom_height)),
            top_height=str(rng.uniform(*top_height)),
            top_width_scale=str(rng.uniform(0.98, 1.03)),
            cover_width_scale=str(rng.uniform(*cover_width_scale)),
            cover_length_scale=str(rng.uniform(*cover_length_scale)),
            bottom_cover_thickness=str(rng.uniform(*cover_thickness)),
            top_cover_thickness=str(rng.uniform(*cover_thickness)),
        )
        variation_params.append({"sample_id": i + 1, "params": params})
        urdf = construct_notebook(params)
        urdf_path, urdf_6dof_path = save_proc_variation_pair(
            "notebook", i, urdf, output_root=output_root
        )
        print(f"URDF saved to {urdf_path}")
        print(f"URDF saved to {urdf_6dof_path}")
    params_path = save_proc_generation_params("notebook", variation_params, seed=seed)
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
            "Usage: python notebook.py [--demo | --gen] [--seed=<int>] [--out_dir=<path>]"
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
            "Usage: python notebook.py [--demo | --gen] [--seed=<int>] [--out_dir=<path>]"
        )

    params = dict(
        width=0.16,
        length=0.24,
        bottom_height=0.012,
        top_height=0.008,
        top_width_scale=1.0,
        cover_width_scale=1.08,
        cover_length_scale=1.05,
        bottom_cover_thickness=0.003,
        top_cover_thickness=0.003,
    )

    urdf = construct_notebook(params)
    urdf_path = get_assets_dir() / "proc_notebook" / "notebook.urdf"
    urdf_path.parent.mkdir(parents=True, exist_ok=True)
    with open(urdf_path, "w", encoding="utf-8") as f:
        f.write(urdf)

    print(f"URDF saved to {urdf_path}")
