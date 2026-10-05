import json
from datetime import datetime
from pathlib import Path

from dexlink.utils.retargeting.path_utils import get_assets_dir

_SMALL_LINK_XML = """  <link name="{name}">
    <inertial>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <mass value="0.01"/>
      <inertia ixx="1e-6" ixy="0" ixz="0" iyy="1e-6" iyz="0" izz="1e-6"/>
    </inertial>
  </link>
"""

DEFAULT_PARAMS_DIR = Path("~/Desktop/PhD/params").expanduser()


def make_6dof_variant_urdf(urdf_str: str) -> str:
    """Wrap procedural URDF with a 6-DoF base chain by rerouting `base_fixed` to `yaw_link`."""
    if "tx_joint" in urdf_str and "yaw_joint" in urdf_str:
        return urdf_str
    if '<joint name="base_fixed" type="fixed">' not in urdf_str:
        raise ValueError("Expected a base_fixed joint in procedural URDF.")
    if '<parent link="root"/>' not in urdf_str:
        raise ValueError("Expected base_fixed parent to be root in procedural URDF.")
    if "</robot>" not in urdf_str:
        raise ValueError("Malformed URDF: missing </robot>.")

    # Redirect the existing fixed attachment from root->bottom to yaw_link->bottom.
    patched = urdf_str.replace('<parent link="root"/>', '<parent link="yaw_link"/>', 1)

    chain = (
        "\n"
        + _SMALL_LINK_XML.format(name="tx_link")
        + """
  <joint name="tx_joint" type="prismatic">
    <parent link="root"/>
    <child link="tx_link"/>
    <axis xyz="1 0 0"/>
    <limit lower="-5.0" upper="5.0" effort="50" velocity="1"/>
  </joint>
"""
        + _SMALL_LINK_XML.format(name="ty_link")
        + """
  <joint name="ty_joint" type="prismatic">
    <parent link="tx_link"/>
    <child link="ty_link"/>
    <axis xyz="0 1 0"/>
    <limit lower="-5.0" upper="5.0" effort="50" velocity="1"/>
  </joint>
"""
        + _SMALL_LINK_XML.format(name="tz_link")
        + """
  <joint name="tz_joint" type="prismatic">
    <parent link="ty_link"/>
    <child link="tz_link"/>
    <axis xyz="0 0 1"/>
    <limit lower="-5.0" upper="5.0" effort="50" velocity="1"/>
  </joint>
"""
        + _SMALL_LINK_XML.format(name="roll_link")
        + """
  <joint name="roll_joint" type="revolute">
    <parent link="tz_link"/>
    <child link="roll_link"/>
    <axis xyz="1 0 0"/>
    <limit lower="-31.41592653589793" upper="31.41592653589793" effort="50" velocity="1"/>
  </joint>
"""
        + _SMALL_LINK_XML.format(name="pitch_link")
        + """
  <joint name="pitch_joint" type="revolute">
    <parent link="roll_link"/>
    <child link="pitch_link"/>
    <axis xyz="0 1 0"/>
    <limit lower="-31.41592653589793" upper="31.41592653589793" effort="50" velocity="1"/>
  </joint>
"""
        + _SMALL_LINK_XML.format(name="yaw_link")
        + """
  <joint name="yaw_joint" type="revolute">
    <parent link="pitch_link"/>
    <child link="yaw_link"/>
    <axis xyz="0 0 1"/>
    <limit lower="-31.41592653589793" upper="31.41592653589793" effort="50" velocity="1"/>
  </joint>
"""
    )
    return patched.replace("</robot>", chain + "\n</robot>", 1)


def save_proc_variation_pair(
    asset_name: str, idx: int, urdf_str: str, output_root: str | Path | None = None
):
    """
    Save:
      <output_root>/proc_<asset>/<idx+1>/<idx+1>.urdf
      <output_root>/proc_<asset>/<idx+1>/<idx+1>_6dof.urdf
    where <output_root> defaults to assets/.
    """
    sample_id = str(idx + 1)
    root_dir = (
        get_assets_dir()
        if output_root is None
        else Path(output_root).expanduser().resolve()
    )
    out_dir = root_dir / f"proc_{asset_name}" / sample_id
    out_dir.mkdir(parents=True, exist_ok=True)

    base_path = out_dir / f"{sample_id}.urdf"
    urdf_6dof_path = out_dir / f"{sample_id}_6dof.urdf"

    with open(base_path, "w", encoding="utf-8") as f:
        f.write(urdf_str)
    with open(urdf_6dof_path, "w", encoding="utf-8") as f:
        f.write(make_6dof_variant_urdf(urdf_str))

    return base_path, urdf_6dof_path


def save_proc_generation_params(
    asset_name: str,
    variation_params: list[dict],
    seed: int | None = None,
    output_root: str | Path | None = None,
) -> Path:
    root_dir = (
        DEFAULT_PARAMS_DIR
        if output_root is None
        else Path(output_root).expanduser().resolve()
    )
    out_dir = root_dir / f"proc_{asset_name}"
    out_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    seed_tag = f"_seed{seed}" if seed is not None else ""
    out_path = out_dir / f"{asset_name}_params_{ts}{seed_tag}.json"
    suffix = 1
    while out_path.exists():
        out_path = out_dir / f"{asset_name}_params_{ts}{seed_tag}_{suffix}.json"
        suffix += 1

    payload = {
        "asset_name": asset_name,
        "seed": seed,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "num_variations": len(variation_params),
        "variations": variation_params,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    return out_path
