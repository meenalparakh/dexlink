import os
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import trimesh
from numpy.random import uniform as U
from urdfpy import URDF

from dexlink.utils.retargeting.vis_server import VizServer


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
      <mass value="1.0"/>
      <inertia ixx="0.1" ixy="0.0" ixz="0.0" iyy="0.1" iyz="0.0" izz="0.1"/>
    </inertial>
  </link>
</robot>
"""
    urdf_path = os.path.join(save_dir, "simple_object.urdf")
    with open(urdf_path, "w") as f:
        f.write(urdf_content)
    return urdf_path


def scale_urdf(input_path, output_path, scale=0.5):
    if scale <= 0.0:
        raise ValueError(f"`scale` must be positive, got {scale}.")
    tree = ET.parse(input_path)
    root = tree.getroot()

    # Scale all mesh geometries
    for mesh in root.findall(".//mesh"):
        s = f"{scale} {scale} {scale}"
        mesh.set("scale", s)

    # Scale all origins (xyz)
    for origin in root.findall(".//origin"):
        if "xyz" in origin.attrib:
            xyz = [float(x) * scale for x in origin.attrib["xyz"].split()]
            origin.set("xyz", " ".join(map(str, xyz)))

    # Scale all inertial masses and inertia tensors
    # for inertial in root.findall(".//inertial"):
    #     mass_elem = inertial.find("mass")
    #     inertia_elem = inertial.find("inertia")

    #     if mass_elem is not None and 'value' in mass_elem.attrib:
    #         m = float(mass_elem.attrib['value'])
    #         mass_elem.set('value', str(m * scale**3))

    #     if inertia_elem is not None:
    #         for attr in inertia_elem.attrib:
    #             val = float(inertia_elem.attrib[attr])
    #             inertia_elem.set(attr, str(val * scale**5))

    tree.write(output_path)
    print(f"✅ Scaled URDF saved to {output_path}")


def materialize_scaled_urdf(
    input_path: str | Path,
    output_dir: str | Path,
    scale: float,
    output_name: str | None = None,
) -> Path:
    input_path = Path(input_path)
    if input_path.suffix.lower() != ".urdf":
        raise ValueError(f"Expected a .urdf input path, got {input_path}.")
    if not input_path.exists():
        raise FileNotFoundError(f"URDF file not found: {input_path}")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if output_name is None:
        output_name = f"{input_path.stem}_scaled.urdf"
    output_path = output_dir / output_name
    scale_urdf(input_path, output_path, scale)
    return output_path


def load_urdf_combined_mesh(
    urdf_path: str | Path,
    mesh_type: str = "visual",
) -> trimesh.Trimesh:
    urdf_path = Path(urdf_path)
    if not urdf_path.exists():
        raise FileNotFoundError(f"URDF file not found: {urdf_path}")
    if mesh_type not in ("visual", "collision"):
        raise ValueError(
            f"`mesh_type` must be 'visual' or 'collision', got {mesh_type!r}."
        )

    urdf_inst = URDF.load(str(urdf_path))
    if mesh_type == "visual":
        mesh_fk = urdf_inst.visual_trimesh_fk()
    else:
        mesh_fk = urdf_inst.collision_trimesh_fk()

    pieces = []
    for mesh, pose in mesh_fk.items():
        transformed_mesh = mesh.copy()
        transformed_mesh.apply_transform(pose)
        pieces.append(transformed_mesh)

    if not pieces:
        raise ValueError(f"No {mesh_type} meshes found in URDF: {urdf_path}")
    return trimesh.util.concatenate(pieces)


def get_parent_child_link_names(robo_urdfpy):
    parent_child_links = set()
    for j in robo_urdfpy.joints:
        # if j.joint_type == "fixed":
        parent_child_links.add((j.parent, j.child))

    return parent_child_links


def check_is_collision(
    robo_urdfpy: URDF,
    jdict,
    obj_dict={},
    art_jdict={},
    view=False,
    base_pose=np.eye(4),
    collision_manager=None,
    collision_threshold=-0.02,
):
    if view:
        vis = VizServer()
    else:
        vis = None

    if collision_manager is None:
        collision_manager = trimesh.collision.CollisionManager()

    parent_child_links = get_parent_child_link_names(robo_urdfpy)
    link_fk = robo_urdfpy.link_fk(cfg=jdict)
    # link_fk = robo_urdfpy.collision_trimesh_fk(cfg=jdict)

    # get mesh for linkname 'link_0'

    min_distances = []
    names = []
    meshes = []
    poses = []

    # for fixed links, we don't check their collision with their parent link

    if "ground" in obj_dict:
        g_ht = obj_dict["ground"].pose[2, 3]
    else:
        g_ht = -10.0

    for idx, (link, pose) in enumerate(link_fk.items()):
        mesh = link.collision_mesh

        if mesh is None:
            continue

        meshes.append(mesh)
        poses.append(pose)

        dist, name = collision_manager.min_distance_single(
            mesh, transform=base_pose @ pose, return_name=True
        )

        m_ht = (base_pose @ pose)[2, 3]
        if m_ht < g_ht:
            is_collision = True

        else:
            is_collision = dist < collision_threshold
            min_distances.append(dist)
            names.append(name)

        if is_collision:
            if (link.name, name) in parent_child_links:
                is_collision = False
            if (name, link.name) in parent_child_links:
                is_collision = False

        if name == "ground":
            if dist < 0.0:
                is_collision = True

        # is_collision = collision_manager.in_collision_single(
        #     link_mesh, transform=pose,
        # )

        if is_collision:
            if view:
                assert vis is not None, "vis is None"
                for obs in obj_dict.values():
                    vis.view_trimesh(
                        f"obs_{obs.obj_id}", obs.mesh, obs.pose, 0x00FF00, opacity=0.5
                    )

                for idx, (mesh, pose) in enumerate(zip(meshes, poses)):
                    vis.view_trimesh(
                        f"{link.name}", mesh, base_pose @ pose, 0xFFFFFF, opacity=0.5
                    )

                input("Press enter to continue")
            return True
        else:
            collision_manager.add_object(link.name, mesh, transform=base_pose @ pose)

    return False


def get_non_colliding_jpos_general(
    robo_urdfpy,
    obj_dict={},
    art_jdict={},
    partial_dict=None,
    view=False,
    max_tries=200,
    return_dict=True,
    robo_type="generated",
):
    # joint_pos = {}
    # view=True

    # get info from urdfinst

    joints = [j for j in robo_urdfpy.joints if j.joint_type != "fixed"]

    joint_names = [j.name for j in joints]
    joint_lb = [j.limit.lower for j in joints]
    joint_ub = [j.limit.upper for j in joints]

    new_joint_lb = np.array(joint_lb)
    new_joint_ub = np.array(joint_ub)

    for _ in range(max_tries):
        # get robo mesh
        jpos = U(new_joint_lb, new_joint_ub)
        jdict = dict(zip(joint_names, jpos))

        if partial_dict is not None:
            for k in partial_dict:
                jdict[k] = U(partial_dict[k][0], partial_dict[k][1])

        is_collision = check_is_collision(
            robo_urdfpy,
            jdict,
            obj_dict,
            art_jdict=art_jdict,
            view=view,
            base_pose=np.eye(4),
            collision_threshold=(-0.01) if robo_type == "real" else (-0.01),
        )
        if not is_collision:
            if return_dict:
                return jdict
            else:
                return jpos

    return None


def _normalize_obstacle_entries(obstacle_dict):
    if isinstance(obstacle_dict, dict):
        return [
            (f"obstacle_{obs_idx}", mesh, pose)
            for obs_idx, (mesh, pose) in enumerate(obstacle_dict.items())
        ]

    if isinstance(obstacle_dict, (list, tuple)):
        obstacle_entries = []
        for obs_idx, entry in enumerate(obstacle_dict):
            if not isinstance(entry, (list, tuple)):
                raise ValueError(
                    "Expected each obstacle entry to be a tuple/list, got "
                    f"{type(entry)!r} at index {obs_idx}."
                )
            if len(entry) == 2:
                mesh, pose = entry
                name = f"obstacle_{obs_idx}"
            elif len(entry) == 3:
                name, mesh, pose = entry
            else:
                raise ValueError(
                    "Expected each obstacle entry to have length 2 or 3, got "
                    f"{len(entry)} at index {obs_idx}."
                )
            obstacle_entries.append((str(name), mesh, pose))
        return obstacle_entries

    raise ValueError(
        "Expected obstacle_dict to be either a dict or a list/tuple of obstacle entries, "
        f"got {type(obstacle_dict)!r}."
    )


def _visualize_non_colliding_jpos_debug(
    urdf_inst: URDF,
    joint_names: list[str],
    q_input: np.ndarray,
    obstacle_entries,
    q_found: np.ndarray | None,
):
    vis = VizServer()
    vis.clear()

    left_shift = np.eye(4, dtype=float)
    left_shift[0, 3] = -0.35
    right_shift = np.eye(4, dtype=float)
    right_shift[0, 3] = 0.35

    vis.view_robot(
        urdf_inst=urdf_inst,
        joint_names=joint_names,
        joint_vals=q_input,
        name="debug/input_robot",
        color=0xCC4444,
        alpha=0.7,
        mesh_type="collision",
        base_pose=left_shift,
    )
    for obs_name, mesh, pose in obstacle_entries:
        pose_array = np.asarray(pose, dtype=float)
        vis.view_trimesh(
            f"debug/input_obstacles/{obs_name}",
            mesh,
            left_shift @ pose_array,
            0x44AA44,
            opacity=0.45,
        )

    if q_found is not None:
        vis.view_robot(
            urdf_inst=urdf_inst,
            joint_names=joint_names,
            joint_vals=q_found,
            name="debug/found_robot",
            color=0x4444CC,
            alpha=0.7,
            mesh_type="collision",
            base_pose=right_shift,
        )
        for obs_name, mesh, pose in obstacle_entries:
            pose_array = np.asarray(pose, dtype=float)
            vis.view_trimesh(
                f"debug/found_obstacles/{obs_name}",
                mesh,
                right_shift @ pose_array,
                0x44AA44,
                opacity=0.45,
            )
    else:
        print("No non-colliding joint configuration found after max tries.")

    input("Debug visualization loaded. Press Enter to continue...")


def _joint_sampling_order(joints) -> list[int]:
    child_links = {joint.child for joint in joints}
    root_links = [joint.parent for joint in joints if joint.parent not in child_links]
    if len(root_links) == 0:
        return list(range(len(joints)))

    root_depths = {root_link: 0 for root_link in root_links}
    joint_depths = [0] * len(joints)
    updated = True
    while updated:
        updated = False
        for joint_idx, joint in enumerate(joints):
            if joint.parent not in root_depths:
                continue
            depth = root_depths[joint.parent] + 1
            if joint.child not in root_depths or depth < root_depths[joint.child]:
                root_depths[joint.child] = depth
                joint_depths[joint_idx] = depth
                updated = True

    return sorted(range(len(joints)), key=lambda idx: (joint_depths[idx], idx))


def _link_is_collision_free(
    link_name: str,
    mesh: trimesh.Trimesh,
    pose: np.ndarray,
    collision_manager: trimesh.collision.CollisionManager,
    parent_child_links,
    collision_threshold: float,
) -> bool:
    dist, other_name = collision_manager.min_distance_single(
        mesh,
        transform=pose,
        return_name=True,
    )
    if other_name is None:
        return True

    if dist >= collision_threshold:
        return True
    if (link_name, other_name) in parent_child_links:
        return True
    if (other_name, link_name) in parent_child_links:
        return True
    if other_name == "ground" and dist < 0.0:
        return False
    return False


def _build_obstacle_collision_manager(obstacle_entries):
    collision_manager = trimesh.collision.CollisionManager()
    for obs_idx, (name, mesh, pose) in enumerate(obstacle_entries):
        if not isinstance(mesh, trimesh.Trimesh):
            raise ValueError(
                "Expected obstacle mesh to be a trimesh.Trimesh instance, got "
                f"{type(mesh)!r} at index {obs_idx}."
            )
        pose_array = np.asarray(pose, dtype=float)
        if pose_array.shape != (4, 4):
            raise ValueError(
                "Expected each obstacle pose to have shape (4, 4), got "
                f"{pose_array.shape} for obstacle index {obs_idx}."
            )
        collision_manager.add_object(str(name), mesh, transform=pose_array)
    return collision_manager


def get_non_colliding_jpos_near(
    urdf_inst,
    q,
    obstacle_dict,
    fixed_joints=None,
    std=0.5,
    max_tries=200,
    collision_threshold=0.0,
    visualize=False,
):
    # obstacle dict is a dictionary
    # with trimesh.Trimesh: pose entries
    # urdf_inst is the robot's urdf instance
    # fixed joints is a list of joint names that should not be changed
    # the other joints should be sampled within std, and joint limits
    if fixed_joints is None:
        fixed_joints = []
    if std <= 0.0:
        raise ValueError(f"Expected std > 0, got {std}.")
    if max_tries < 1:
        raise ValueError(f"Expected max_tries >= 1, got {max_tries}.")

    all_joints = list(urdf_inst.joints)
    joints = [joint for joint in all_joints if joint.joint_type != "fixed"]
    joint_names = [joint.name for joint in joints]
    joint_lbs = np.asarray([joint.limit.lower for joint in joints], dtype=float)
    joint_ubs = np.asarray([joint.limit.upper for joint in joints], dtype=float)
    joint_name_to_index = {name: idx for idx, name in enumerate(joint_names)}
    link_name_to_link = {link.name: link for link in urdf_inst.links}
    parent_child_links = get_parent_child_link_names(urdf_inst)

    unknown_fixed_joints = [
        name for name in fixed_joints if name not in joint_name_to_index
    ]
    if unknown_fixed_joints:
        raise ValueError(
            f"Fixed joints {unknown_fixed_joints} are not present in the URDF joints."
        )

    if isinstance(q, dict):
        missing_joint_names = [name for name in joint_names if name not in q]
        if missing_joint_names:
            raise ValueError(
                "Expected q dict to contain all non-fixed joint names; missing "
                f"{missing_joint_names}."
            )
        q_center = np.asarray([q[name] for name in joint_names], dtype=float)
        return_dict = True
    else:
        q_center = np.asarray(q, dtype=float)
        if q_center.ndim != 1:
            raise ValueError(
                f"Expected q to be a 1D array or dict, got array shape {q_center.shape}."
            )
        if q_center.shape[0] != len(joint_names):
            raise ValueError(
                "Expected q to match the number of non-fixed joints "
                f"({len(joint_names)}), got {q_center.shape[0]}."
            )
        return_dict = False

    fixed_mask = np.zeros(len(joint_names), dtype=bool)
    for joint_name in fixed_joints:
        fixed_mask[joint_name_to_index[joint_name]] = True

    obstacle_entries = _normalize_obstacle_entries(obstacle_dict)

    collision_manager = _build_obstacle_collision_manager(obstacle_entries)

    q_center = np.clip(q_center, joint_lbs, joint_ubs)
    candidate = q_center.copy()
    if not check_is_collision(
        urdf_inst,
        dict(zip(joint_names, candidate)),
        obj_dict={},
        collision_manager=collision_manager,
        base_pose=np.eye(4),
        collision_threshold=collision_threshold,
    ):
        if visualize:
            _visualize_non_colliding_jpos_debug(
                urdf_inst=urdf_inst,
                joint_names=joint_names,
                q_input=q_center,
                obstacle_entries=obstacle_entries,
                q_found=candidate,
            )
        if return_dict:
            return dict(zip(joint_names, candidate))
        return candidate

    ordered_all_joint_indices = _joint_sampling_order(all_joints)
    n_outer_trials = max(1, min(10, max_tries))
    for outer_trial_idx in range(n_outer_trials):
        print(
            f"Sequential non-colliding construction trial "
            f"{outer_trial_idx + 1}/{n_outer_trials}..."
        )
        candidate = q_center.copy()

        trial_collision_manager = _build_obstacle_collision_manager(obstacle_entries)

        trial_success = True
        for all_joint_idx in ordered_all_joint_indices:
            joint = all_joints[all_joint_idx]
            child_link = link_name_to_link.get(joint.child, None)
            if child_link is None or child_link.collision_mesh is None:
                continue

            if joint.joint_type == "fixed":
                jdict = dict(zip(joint_names, candidate))
                link_fk = urdf_inst.link_fk(cfg=jdict)
                child_pose = np.asarray(link_fk[child_link], dtype=float)
                if not _link_is_collision_free(
                    link_name=child_link.name,
                    mesh=child_link.collision_mesh,
                    pose=child_pose,
                    collision_manager=trial_collision_manager,
                    parent_child_links=parent_child_links,
                    collision_threshold=collision_threshold,
                ):
                    trial_success = False
                    break
                trial_collision_manager.add_object(
                    child_link.name,
                    child_link.collision_mesh,
                    transform=child_pose,
                )
                continue

            joint_idx = joint_name_to_index[joint.name]
            if fixed_mask[joint_idx]:
                jdict = dict(zip(joint_names, candidate))
                link_fk = urdf_inst.link_fk(cfg=jdict)
                child_pose = np.asarray(link_fk[child_link], dtype=float)
                if not _link_is_collision_free(
                    link_name=child_link.name,
                    mesh=child_link.collision_mesh,
                    pose=child_pose,
                    collision_manager=trial_collision_manager,
                    parent_child_links=parent_child_links,
                    collision_threshold=collision_threshold,
                ):
                    trial_success = False
                    break
                trial_collision_manager.add_object(
                    child_link.name,
                    child_link.collision_mesh,
                    transform=child_pose,
                )
                continue

            print(f"Sampling joint '{joint_names[joint_idx]}' (index {joint_idx})...")
            found_joint_value = False
            joint_std = (
                0.05
                if any(token in joint.name for token in ("tx", "ty", "tz"))
                else float(std)
            )
            for _ in range(max_tries):
                proposal = candidate.copy()
                proposal[joint_idx] = np.random.normal(
                    loc=float(q_center[joint_idx]),
                    scale=joint_std,
                )
                proposal = np.clip(proposal, joint_lbs, joint_ubs)
                proposal[fixed_mask] = q_center[fixed_mask]
                jdict = dict(zip(joint_names, proposal))
                link_fk = urdf_inst.link_fk(cfg=jdict)
                child_pose = np.asarray(link_fk[child_link], dtype=float)
                if _link_is_collision_free(
                    link_name=child_link.name,
                    mesh=child_link.collision_mesh,
                    pose=child_pose,
                    collision_manager=trial_collision_manager,
                    parent_child_links=parent_child_links,
                    collision_threshold=collision_threshold,
                ):
                    candidate = proposal
                    trial_collision_manager.add_object(
                        child_link.name,
                        child_link.collision_mesh,
                        transform=child_pose,
                    )
                    found_joint_value = True
                    break

            if not found_joint_value:
                print(
                    f"Could not find non-colliding value for joint "
                    f"'{joint_names[joint_idx]}' after {max_tries} tries."
                )
                trial_success = False
                break

        if trial_success:
            final_collision_manager = _build_obstacle_collision_manager(
                obstacle_entries
            )
            final_jdict = dict(zip(joint_names, candidate))
            if check_is_collision(
                urdf_inst,
                final_jdict,
                obj_dict={},
                collision_manager=final_collision_manager,
                base_pose=np.eye(4),
                collision_threshold=collision_threshold,
            ):
                print(
                    "Final validation detected collision after sequential construction; "
                    "retrying with a new outer trial."
                )
                continue
            if visualize:
                _visualize_non_colliding_jpos_debug(
                    urdf_inst=urdf_inst,
                    joint_names=joint_names,
                    q_input=q_center,
                    obstacle_entries=obstacle_entries,
                    q_found=candidate,
                )
            if return_dict:
                return final_jdict
            return candidate

    if visualize:
        _visualize_non_colliding_jpos_debug(
            urdf_inst=urdf_inst,
            joint_names=joint_names,
            q_input=q_center,
            obstacle_entries=obstacle_entries,
            q_found=None,
        )
    return None
