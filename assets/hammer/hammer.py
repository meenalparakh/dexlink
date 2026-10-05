import zipfile
from pathlib import Path

from dexlink.utils.retargeting.path_utils import get_assets_dir

out_dir = get_assets_dir() / "hammer"
out_dir.mkdir(parents=True, exist_ok=True)

urdf_path = out_dir / "hammer.urdf"

handle_radius = 0.015
handle_length = 0.25
handle_mass = 0.10

head_radius = 0.02
head_length = 0.07
head_mass = 0.50


def cyl_inertia_z(m, r, h):
    ixx = (1.0 / 12.0) * m * (3 * r * r + h * h)
    iyy = ixx
    izz = 0.5 * m * r * r
    return ixx, iyy, izz


handle_ixx, handle_iyy, handle_izz = cyl_inertia_z(
    handle_mass, handle_radius, handle_length
)
head_ixx_z, head_iyy_z, head_izz_z = cyl_inertia_z(head_mass, head_radius, head_length)

# Head cylinder visual/collision is rotated so its cylinder axis is along X.
head_ixx = head_izz_z
head_iyy = head_ixx_z
head_izz = head_iyy_z

urdf = f"""<?xml version="1.0"?>
<robot name="two_link_hammer">

  <material name="handle_brown">
    <color rgba="0.45 0.22 0.08 1.0"/>
  </material>

  <material name="head_dark_grey">
    <color rgba="0.08 0.08 0.08 1.0"/>
  </material>

  <link name="handle_link">
    <inertial>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <mass value="{handle_mass:.6f}"/>
      <inertia
        ixx="{handle_ixx:.9f}"
        ixy="0"
        ixz="0"
        iyy="{handle_iyy:.9f}"
        iyz="0"
        izz="{handle_izz:.9f}"/>
    </inertial>

    <visual>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry>
        <cylinder radius="{handle_radius:.6f}" length="{handle_length:.6f}"/>
      </geometry>
      <material name="handle_brown"/>
    </visual>

    <collision>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry>
        <cylinder radius="{handle_radius:.6f}" length="{handle_length:.6f}"/>
      </geometry>
    </collision>
  </link>

  <link name="head_link">
    <inertial>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <mass value="{head_mass:.6f}"/>
      <inertia
        ixx="{head_ixx:.9f}"
        ixy="0"
        ixz="0"
        iyy="{head_iyy:.9f}"
        iyz="0"
        izz="{head_izz:.9f}"/>
    </inertial>

    <visual>
      <origin xyz="0 0 0" rpy="0 1.57079632679 0"/>
      <geometry>
        <cylinder radius="{head_radius:.6f}" length="{head_length:.6f}"/>
      </geometry>
      <material name="head_dark_grey"/>
    </visual>

    <collision>
      <origin xyz="0 0 0" rpy="0 1.57079632679 0"/>
      <geometry>
        <cylinder radius="{head_radius:.6f}" length="{head_length:.6f}"/>
      </geometry>
    </collision>
  </link>

  <joint name="handle_to_head_fixed" type="fixed">
    <parent link="handle_link"/>
    <child link="head_link"/>
    <origin xyz="0 0 {handle_length / 2:.6f}" rpy="0 0 0"/>
  </joint>

</robot>
"""

urdf_path.write_text(urdf)
