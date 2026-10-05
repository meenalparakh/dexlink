"""
# OakInk-Shape
OAKINK_DIR/shape/ # OakInk-Shape root
OAKINK_DIR/shape/metaV2/ # OakInk-Shape meta data
    object_id.json # source obj name_to_id mapping
    virtual_obj_id.json # target obj name_to_id mapping
    yodaobject_cat.json # category id_to_name mapping
OAKINK_DIR/shape/oakink_shape_v2/ # OakInk-Shape's grasping poses root
    bottle # category bottle's root
        A16012/ # source obj ID, obj that appear in OakInk-Image
            0cc013118e/ # source grasp ID, grasp that appears in OakInk-Image, record on A16012
                hand_param.pkl # hand param of the source grasp 0cc013118e on source obj A16012
                    pose: # MANO pose param (in quaternion, 16x4) in obj-canonical space.
                    shape: # MANO shape param (10 dim)
                    tsl: # 3D hand translation in obj-canonical space.
                source.txt # where the source grasp 0cc013118e appears in OakInk-Image
                s16101/ # target obj ID
                    hand_param.pkl # target grasp, 0cc013118e on target obj s16101
                ...
OAKINK_DIR/shape/OakInkObjectsV2/ # source object models root, obj that appear in OakInk-Image
    006_mustard_bottle/ # obj instance name, name_to_id mappings in metaV2/object_id.json
        align/ # rot and scale aligned obj model
        align_ds/ # downsampled aligned obj model
    ...
OAKINK_DIR/shape/OakInkVirtualObjectsV2/ # target object models root, obj that not appear in OakInk-Image
    binoculars_0/ # obj instance name, name_to_id mappings in metaV2/virtual_obj_id.json
        align/ # rot and scale aligned obj model
        align_ds/ # downsampled aligned obj model
    ...
"""

import os

from dexlink.utils.retargeting.path_utils import get_external_data_dir


OAKINK_DIR = get_external_data_dir() / "oakink"
# for the (21) dimension sequence of joints. the following is the mapping
hand_joint_map = {
    "wrist": 0,
    "thumb_tip": 4,
    "index_tip": 8,
    "middle_tip": 12,
    "ring_tip": 16,
    "little_tip": 20,
    "thumb_base": 2,
    "index_base": 5,
    "middle_base": 9,
    "ring_base": 13,
    "little_base": 17,
}


def create_urdf_from_obj(obj_path, save_dir, scale=1.0):
    """
    Create a simple URDF file that wraps the given OBJ file.
    """
    urdf_content = f"""<?xml version="1.0" ?>
<robot name="simple_object">
  <link name="base_link">
    <visual>
      <geometry>
        <mesh filename="{obj_path}" scale="{scale} {scale} {scale}"/>
      </geometry>
    </visual>
    <collision>
      <geometry>
        <mesh filename="{obj_path}" scale="{scale} {scale} {scale}"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="0.2"/>
      <inertia ixx="0.1" ixy="0.0" ixz="0.0" iyy="0.1" iyz="0.0" izz="0.1"/>
    </inertial>
  </link>
</robot>
"""
    urdf_path = os.path.join(save_dir, "simple_object.urdf")
    with open(urdf_path, "w") as f:
        f.write(urdf_content)
    return urdf_path
