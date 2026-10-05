# from pyngrok import ngrok
import os
import random
from collections import OrderedDict

import matplotlib.cm as cm
import matplotlib.colors as mc
import meshcat
import meshcat.geometry as g
import meshcat.transformations as tf
import numpy as np
import trimesh
from matplotlib.colors import Normalize
from meshcat.animation import Animation
from scipy.spatial.transform import Rotation as R

from dexlink.utils.retargeting.math_utils import batched_rotation_matrices_from_normals
from dexlink.utils.retargeting.path_utils import get_tmp_dir


def float_to_int(color):
    color = [int(c * 255) for c in color]
    return (color[0] << 16) + (color[1] << 8) + color[2]


def _resolve_meshcat_zmq_url(port_vis: int) -> str:
    env_zmq_url = os.getenv("MESHCAT_ZMQ_URL")
    if env_zmq_url is not None:
        return env_zmq_url
    return f"tcp://127.0.0.1:{port_vis}"


class VizServer:
    def __init__(self, mc_vis=None, port_vis=6000) -> None:
        zmq_url = _resolve_meshcat_zmq_url(port_vis)

        if mc_vis is None:
            self.zmq_url = zmq_url
            self.mc_vis = meshcat.Visualizer(zmq_url=zmq_url)
            self.clear()
            self.mc_vis["/Background"].set_property("top_color", [1, 1, 1])
            self.mc_vis["/Background"].set_property("bottom_color", [1, 1, 1])
        else:
            self.mc_vis = mc_vis
            self.zmq_url = getattr(mc_vis, "zmq_url", None)

    def clear(self):
        self.mc_vis["scene"].delete()
        self.mc_vis["meshcat"].delete()
        self.mc_vis["/"].delete()

    def view_cylinder(self, ht, r, center):
        self.mc_vis["scene/cylinder"].set_object(
            g.Cylinder(ht, r), g.MeshLambertMaterial(color=0x22DD22, opacity=0.3)
        )

        X_pose = np.eye(4)
        X_pose[:3, :3] = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float32)
        X_pose[:3, 3] = center
        self.mc_vis["scene/cylinder"].set_transform(X_pose)

    def view_cylinder_pose(self, name, ht, r, pose, color):
        self.mc_vis[f"{name}"].set_object(
            g.Cylinder(ht, r), g.MeshLambertMaterial(color=color, opacity=0.7)
        )
        self.mc_vis[f"{name}"].set_transform(pose)

    def view_axis(self, name, axis_ht, axis_rad, pose, color):
        cone_rad = axis_rad * 3.0
        self.mc_vis[f"{name}/cylinder"].set_object(
            g.Cylinder(axis_ht, axis_rad),
            g.MeshLambertMaterial(color=color, opacity=0.7),
        )
        self.mc_vis[f"{name}/cone"].set_object(
            g.Sphere(cone_rad), g.MeshLambertMaterial(color=color, opacity=0.7)
        )
        X_CyCo = np.eye(4)
        X_CyCo[:3, 3] = [0, axis_ht / 2.0, 0.0]
        X_CyCo[:3, :3] = R.from_euler("xyz", [-np.pi / 2, 0, 0]).as_matrix()

        self.mc_vis[f"{name}/cylinder"].set_transform(pose)
        self.mc_vis[f"{name}/cone"].set_transform(pose @ X_CyCo)

    def view_trimesh(self, name, mesh, pose, color, opacity=0.7):
        tmp_obj_path = str(get_tmp_dir() / "tmp_obj.obj")
        import trimesh

        trimesh.exchange.export.export_mesh(mesh, tmp_obj_path)
        self.view_mesh(name, tmp_obj_path, pose, color, opacity)

    def view_mesh(self, name, obj_path, pose, color, opacity=0.7):
        if color is None:
            color = random.randrange(0, 2**24)

            # Converting that number from base-10
            # (decimal) to base-16 (hexadecimal)
            color = hex(color)
        self.mc_vis[f"scene/{name}"].set_object(
            g.ObjMeshGeometry.from_file(obj_path),
            g.MeshLambertMaterial(color=color, opacity=opacity),
        )
        self.mc_vis[f"scene/{name}"].set_transform(pose)

    def view_pcd(self, pts, colors=None, name="scene", size=0.001):
        # colors are in [0., 1.]
        if colors is None:
            colors = pts
        # self.mc_vis["scene"].delete()
        self.mc_vis["scene/" + name].set_object(
            g.PointCloud(pts.T, color=colors.T, size=size)
        )

    def view_line_segment(
        self,
        start,
        end,
        *,
        name="line_segment",
        color=(0.0, 0.0, 1.0),
        linewidth=2.0,
        opacity=1.0,
    ):
        start = np.asarray(start, dtype=np.float32)
        end = np.asarray(end, dtype=np.float32)
        if start.shape != (3,) or end.shape != (3,):
            raise ValueError(
                "Expected line segment endpoints to have shape (3,), got "
                f"{start.shape} and {end.shape}."
            )
        color = np.asarray(color, dtype=np.float32)
        if color.shape != (3,):
            raise ValueError(
                f"Expected line segment color to have shape (3,), got {color.shape}."
            )
        points = np.stack([start, end], axis=1)
        material = g.LineBasicMaterial(
            color=float_to_int(color.tolist()),
            linewidth=float(linewidth),
            opacity=float(opacity),
        )
        self.mc_vis["scene/" + name].set_object(
            g.Line(g.PointsGeometry(points), material)
        )

    def view_vector(self, pos, vector, name="positioned_vector", color=0x0000FF):
        # axis_length = 0.25
        axis_length = np.linalg.norm(vector)
        radius = 0.005

        z_axis = vector / np.linalg.norm(vector)
        tmp_y_axis = [0.0, 0.0, 1]
        if np.linalg.norm(z_axis - tmp_y_axis) < 1e-2:
            tmp_y_axis = [1.0, 0.0, 0.0]
        x_axis = np.cross(z_axis, tmp_y_axis)
        x_axis = x_axis / np.linalg.norm(x_axis)
        y_axis = np.cross(z_axis, x_axis)

        X = np.eye(4)
        X[:3, :3] = np.stack([x_axis, y_axis, z_axis]).T
        X[:3, 3] = pos

        X_FZ = np.eye(4)
        X_FZ[:3, :3] = R.from_euler("xyz", [np.pi / 2, 0, 0]).as_matrix()
        X_FZ[:3, 3] = [0, 0, axis_length / 2]
        self.view_axis(f"{name}/z", axis_length, radius, X @ X_FZ, color)

        self.mc_vis[f"{name}/base"].set_object(
            g.Sphere(radius * 2.0), g.MeshLambertMaterial(color=0x000000, opacity=0.7)
        )
        X_CyCo = np.eye(4)
        X_CyCo[:3, 3] = pos
        self.mc_vis[f"{name}/base"].set_transform(X_CyCo)

    def view_frames_pointed(self, pose, name="frame_pose"):
        X_WF = pose
        axis_length = 0.25
        radius = 0.005

        X_FZ = np.eye(4)
        X_FZ[:3, :3] = R.from_euler("xyz", [np.pi / 2, 0, 0]).as_matrix()
        X_FZ[:3, 3] = [0, 0, axis_length / 2]
        self.view_axis(f"{name}/z", axis_length, radius, X_WF @ X_FZ, 0x0000FF)

        X_FY = np.eye(4)
        X_FY[:3, :3] = R.from_euler("xyz", [0, 0, 0]).as_matrix()
        X_FY[:3, 3] = [0, axis_length / 2, 0]
        self.view_axis(f"{name}/y", axis_length, radius, X_WF @ X_FY, 0x00FF00)

        X_FX = np.eye(4)
        X_FX[:3, :3] = R.from_euler("xyz", [0, 0, -np.pi / 2]).as_matrix()
        X_FX[:3, 3] = [axis_length / 2, 0, 0]
        self.view_axis(f"{name}/x", axis_length, radius, X_WF @ X_FX, 0xFF0000)

    def view_frames(self, pose, name="frame_pose", scale=1.0):
        X_WF = pose
        axis_length = 0.25 * scale
        radius = 0.005 * scale

        X_FZ = np.eye(4)
        X_FZ[:3, :3] = R.from_euler("xyz", [np.pi / 2, 0, 0]).as_matrix()
        X_FZ[:3, 3] = [0, 0, axis_length / 2]
        self.view_cylinder_pose(f"{name}/z", axis_length, radius, X_WF @ X_FZ, 0x0000FF)

        X_FY = np.eye(4)
        X_FY[:3, :3] = R.from_euler("xyz", [0, 0, 0]).as_matrix()
        X_FY[:3, 3] = [0, axis_length / 2, 0]
        self.view_cylinder_pose(f"{name}/y", axis_length, radius, X_WF @ X_FY, 0x00FF00)

        X_FX = np.eye(4)
        X_FX[:3, :3] = R.from_euler("xyz", [0, 0, -np.pi / 2]).as_matrix()
        X_FX[:3, 3] = [axis_length / 2, 0, 0]
        self.view_cylinder_pose(f"{name}/x", axis_length, radius, X_WF @ X_FX, 0xFF0000)

    def view_position(self, position, name="point"):
        if len(position) == 2:
            position = [*position, 0.0]
        X = np.eye(4)
        X[:3, 3] = position
        self.view_frames(X, name=name)

    def custom_visual_trimesh_fk(self, urdf_inst, joint_dict=None):
        """Computes the poses of the URDF's visual trimeshes using fk.

        Parameters
        ----------
        cfg : dict or (n), float
            A map from joints or joint names to configuration values for
            each joint, or a list containing a value for each actuated joint
            in sorted order from the base link.
            If not specified, all joints are assumed to be in their default
            configurations.
        links : list of str or list of :class:`.Link`
            The links or names of links to perform forward kinematics on.
            Only trimeshes from these links will be in the returned map.
            If not specified, all links are returned.

        Returns
        -------
        fk : dict
            A map from :class:`~trimesh.base.Trimesh` objects that are
            part of the visual geometry of the specified links to the
            4x4 homogenous transform matrices that position them relative
            to the base link's frame.
        """
        lfk = urdf_inst.link_fk(cfg=joint_dict, links=None)

        fk = OrderedDict()
        for link in lfk:
            mesh_counter = 0
            for visual in link.visuals:
                for mesh in visual.geometry.meshes:
                    pose = lfk[link].dot(visual.origin)
                    if visual.geometry.mesh is not None:
                        if visual.geometry.mesh.scale is not None:
                            S = np.eye(4, dtype=np.float64)
                            S[:3, :3] = np.diag(visual.geometry.mesh.scale)
                            pose = pose.dot(S)
                    fk[mesh] = (pose, f"{link.name}_{mesh_counter}")
                    mesh_counter += 1
        return fk

    def custom_collision_trimesh_fk(self, urdf_inst, joint_dict):
        lfk = urdf_inst.link_fk(cfg=joint_dict, links=None)
        fk = OrderedDict()
        for link in lfk:
            pose = lfk[link]
            cm = link.collision_mesh
            if cm is not None:
                fk[cm] = (pose, link.name)
        return fk

    def view_robot(
        self,
        urdf_inst,
        joint_names,
        joint_vals,
        name,
        color,
        alpha=0.3,
        mesh_type="visual",
        joint_dict=None,
        base_pose=np.eye(4),
        show_link_frames=False,
    ):
        from dexlink.utils.retargeting.path_utils import get_tmp_dir

        tmp_storage = get_tmp_dir() / "meshcat" / mesh_type
        tmp_storage.mkdir(exist_ok=True, parents=True)

        if joint_dict is None:
            joint_dict = dict(zip(joint_names, joint_vals))

        if mesh_type == "collision":
            # fk = urdf_inst.collision_trimesh_fk(cfg=joint_dict)
            fk = self.custom_collision_trimesh_fk(
                urdf_inst=urdf_inst, joint_dict=joint_dict
            )
        else:
            # fk = urdf_inst.visual_trimesh_fk(cfg=joint_dict)
            fk = self.custom_visual_trimesh_fk(
                urdf_inst=urdf_inst, joint_dict=joint_dict
            )
        # override colors
        import seaborn as sns

        link_colors = sns.color_palette("coolwarm", len(fk))
        colors = [mc.rgb2hex(c)[1:] for c in link_colors]
        colors = [int(c, 16) for c in colors]

        to_delete = []
        for idx, (mesh, pose) in enumerate(fk.items()):
            fname = str(tmp_storage / f"mesh_{idx}.obj")
            # mesh to tmp obj

            if isinstance(pose, tuple):
                pose, link_name = pose
                link_name = f"_{link_name}"
            else:
                link_name = ""

            trimesh.exchange.export.export_mesh(mesh, fname, include_normals=True)
            self.view_mesh(
                f"{name}/{idx}{link_name}",
                fname,
                base_pose @ pose,
                colors[idx],
                opacity=alpha,
            )
            to_delete.append(fname)

            if show_link_frames:
                self.view_frames(
                    base_pose @ pose, name=f"{name}/frames/{idx}{link_name}", scale=0.1
                )

        for fname in to_delete:
            os.remove(fname)

    def view_ground(self, name, zval=0.0):
        self.mc_vis["scene/" + name].set_object(
            g.Box([1.0, 1.0, 0.001]), g.MeshLambertMaterial(color=0xAAAAAA, opacity=1.0)
        )
        X = np.eye(4)
        X[2, 3] = zval
        self.mc_vis["scene/" + name].set_transform(X)

    # def view_meshes(
    #     self,
    #     meshes, poses, name, color, alpha=0.3,
    # ):
    #     for idx, (mesh, pose) in enumerate(zip(meshes, poses)):
    #         self.view_trimesh(f"{name}/{idx}", mesh, pose, color, opacity=alpha)

    def view_link_geometry(self, link_info_dict, name="links"):
        for link_name, link_info in link_info_dict.items():
            geom_type = link_info[0]
            if geom_type == "box":
                self.mc_vis[f"{name}/{link_name}"].set_object(
                    g.Box(link_info[1]["dims"]),
                    g.MeshLambertMaterial(color=0xAAAAAA, opacity=1.0),
                )
            elif geom_type == "sphere":
                self.mc_vis[f"{name}/{link_name}"].set_object(
                    g.Sphere(link_info[1]["radius"]),
                    g.MeshLambertMaterial(color=0xAAAAAA, opacity=1.0),
                )
            elif geom_type == "cylinder":
                self.mc_vis[f"{name}/{link_name}"].set_object(
                    g.Cylinder(link_info[1]["height"], link_info[1]["radius"]),
                    g.MeshLambertMaterial(color=0xAAAAAA, opacity=1.0),
                )

    def view_spheres(self, centers, radii, name, color, alpha=0.3):
        for idx in range(len(centers)):
            center = centers[idx]
            radius = radii[idx]

            self.mc_vis[f"{name}/{idx}"].set_object(
                g.Sphere(radius), g.MeshLambertMaterial(color=color, opacity=alpha)
            )
            X = np.eye(4)
            X[:3, 3] = center
            self.mc_vis[f"{name}/{idx}"].set_transform(X)

    def view_fitted_box(self, mesh, mesh_origin, box_dim, box_transform, name="mesh"):
        # view mesh
        # self.view_trimesh(name, mesh, mesh_origin, 0x0000FF)

        # view box
        self.mc_vis[f"scene/fitted_box/{name}"].set_object(
            g.Box(box_dim), g.MeshLambertMaterial(color=0xAAAAAA, opacity=0.2)
        )
        self.mc_vis[f"scene/fitted_box/{name}"].set_transform(
            mesh_origin @ box_transform
        )

    def view_box(self, box_dim, box_transform, name="box", color=0xAAAAAA, alpha=0.3):
        self.mc_vis[f"scene/{name}"].set_object(
            g.Box(box_dim), g.MeshLambertMaterial(color=color, opacity=alpha)
        )
        self.mc_vis[f"scene/{name}"].set_transform(box_transform)

    def view_scene(
        self, generated_objects, object_placements, prefix="default", color=0xAA0000
    ):
        # both args are dictionaries with keys as object names
        # assuming that values of generated objects are outputs of
        # def box_generator(params):
        #     return {
        #         "type": "box",
        #         "params": params,
        #     }
        # and values in object placements are np arrays of (6,) shape
        def getX(placement):
            X = np.eye(4)
            X[:3, 3] = placement[:3]
            X[:3, :3] = R.from_euler("xyz", placement[3:]).as_matrix()
            return X

        for obj_name, obj_params in generated_objects.items():
            if obj_params["type"] == "box":
                box_dim = obj_params["params"]
                box_dim = np.array(
                    [box_dim["length"], box_dim["width"], box_dim["height"]]
                )
                box_transform = getX(object_placements[obj_name])
                self.view_box(
                    box_dim, box_transform, name=f"{prefix}/{obj_name}", color=color
                )
            elif obj_params["type"] == "mesh":
                mesh_path = obj_params["params"]
                mesh = trimesh.load(mesh_path)
                mesh_transform = getX(object_placements[obj_name])
                self.view_trimesh(obj_name, mesh, mesh_transform, 0xAAAAAA)
            else:
                raise ValueError(f"Unknown object type: {obj_params['type']}")

    def view_voxels(self, voxels, voxel_size, workspace_bounds, name="voxels"):
        # the workspace bounds is divided into voxels
        # the voxels are of size voxel_size
        # workspace_bounds = [[x_min, x_max], [y_min, y_max], [z_min, z_max]]

        # get the grid size
        grid_dims = voxels.shape

        # get the min bounds
        min_bounds = workspace_bounds[:, 0]

        # for each voxel, create a box of size voxel_size
        # compute the center of each voxel

        voxel_x_centers = (
            np.arange(grid_dims[0]) * voxel_size + min_bounds[0] + voxel_size / 2
        )
        voxel_y_centers = (
            np.arange(grid_dims[1]) * voxel_size + min_bounds[1] + voxel_size / 2
        )
        voxel_z_centers = (
            np.arange(grid_dims[2]) * voxel_size + min_bounds[2] + voxel_size / 2
        )

        # get non-zero indices
        voxel_indices = np.where(voxels)

        # get the centers of the non-zero voxels
        voxel_centers = np.stack(
            [
                voxel_x_centers[voxel_indices[0]],
                voxel_y_centers[voxel_indices[1]],
                voxel_z_centers[voxel_indices[2]],
            ],
            axis=-1,
        )
        # view as pcd
        self.view_pcd(voxel_centers, name=name, size=voxel_size)

    def close(self):
        self.mc_vis.close()


class VisAnimator:
    def __init__(self, port_vis=6000) -> None:
        zmq_url = _resolve_meshcat_zmq_url(port_vis)
        self.zmq_url = zmq_url
        self.mc_vis = meshcat.Visualizer(zmq_url=zmq_url)
        self.mc_vis["/Background"].set_property("top_color", [1, 1, 1])
        self.mc_vis["/Background"].set_property("bottom_color", [1, 1, 1])
        self.pts_dict = {}
        self.mesh_dict = {}
        self.urdfs_dict = {}
        self.vec_field_dict = {}
        self.dt = 1
        self.clear()

    def clear(self):
        self.mc_vis["scene"].delete()
        self.mc_vis["meshcat"].delete()
        self.mc_vis["/"].delete()

    def initialize_points(
        self,
        mesh_vertices,
        mesh_colors,
        posed=False,
        num_points=1000,
        name="sampled_points",
        radius=0.003,
        opacity=0.7,
    ):
        # mesh_vertices: (N, 3)
        # randomly sample num_points from mesh_vertices

        if mesh_colors.ndim == 1:
            mesh_colors = np.tile(mesh_colors[None, :], (mesh_vertices.shape[0], 1))

        assert (
            mesh_colors.shape[0] == mesh_vertices.shape[0]
        ), f"Got {mesh_colors.shape}, expected {(mesh_vertices.shape[0], 3)}"
        if mesh_vertices.shape[0] <= num_points:
            self.pts_dict[name] = (
                mesh_vertices,
                mesh_colors,
                np.arange(mesh_vertices.shape[0]),
                posed,
                float(radius),
                float(opacity),
            )
        else:
            indices = np.random.choice(
                mesh_vertices.shape[0], num_points, replace=False
            )
            self.pts_dict[name] = (
                mesh_vertices[indices],
                mesh_colors[indices],
                indices,
                posed,
                float(radius),
                float(opacity),
            )

    def initialize_urdf(
        self, urdf_inst, name, mesh_type="collision", color=0xAAAAAA, alpha=0.3
    ):
        self.urdfs_dict[name] = (urdf_inst, mesh_type, color, alpha)

    def _urdf_link_colors(self, n_links, color_spec):
        import seaborn as sns

        if isinstance(color_spec, str):
            if color_spec == "cool":
                palette = sns.color_palette("Blues", n_links + 2)[2:]
            elif color_spec == "warm":
                palette = sns.color_palette("Oranges", n_links + 2)[2:]
            elif color_spec == "green":
                palette = sns.color_palette("Greens", n_links + 2)[2:]
            else:
                palette = sns.color_palette("coolwarm", n_links)
        else:
            # Backward-compatible behavior: keep per-link palette coloring by default.
            palette = sns.color_palette("coolwarm", n_links)

        colors = [mc.rgb2hex(c)[1:] for c in palette]
        return [int(c, 16) for c in colors]

    def initialize_vector_field(self, n_vecs, name="vector_field"):
        # the magnitude of each vector is represented by the length of the arrow
        # use cylinders to represent arrows
        radius = 0.0005
        max_dist = 0.2
        ht = 0.01
        self.vec_field_dict[name] = (n_vecs, radius, ht, max_dist)

    def initialize(self):
        for name, (pts, colors, _, posed, radius, opacity) in self.pts_dict.items():
            # initialize each point as a sphere

            n = pts.shape[0]
            for idx in range(n):
                center = pts[idx]
                color = colors[idx]
                int_color = float_to_int(color)
                # color = float_to_hex(color)

                if posed:
                    # initialize flat squares if representing posed points
                    self.mc_vis[f"scene/{name}/{idx}"].set_object(
                        g.Cylinder(height=max(radius * 0.4, 0.001), radius=radius),
                        g.MeshLambertMaterial(color=int_color, opacity=opacity),
                    )
                else:
                    self.mc_vis[f"scene/{name}/{idx}"].set_object(
                        g.Sphere(radius),
                        g.MeshLambertMaterial(color=int_color, opacity=opacity),
                    )
                X = np.eye(4)
                X[:3, 3] = center
                self.mc_vis[f"scene/{name}/{idx}"].set_transform(X)

        for name, (urdf_inst, mesh_type, color, alpha) in self.urdfs_dict.items():
            tmp_storage = get_tmp_dir() / "meshcat" / name
            tmp_storage.mkdir(exist_ok=True, parents=True)

            if mesh_type == "collision":
                fk = urdf_inst.collision_trimesh_fk()
            else:
                fk = urdf_inst.visual_trimesh_fk()

            colors = self._urdf_link_colors(len(fk), color)

            for idx, (mesh, pose) in enumerate(fk.items()):
                fname = str(tmp_storage / f"mesh_{idx}.obj")
                # mesh to tmp obj
                trimesh.exchange.export.export_mesh(mesh, fname, include_normals=True)

                self.mc_vis[f"scene/{name}/{idx}"].set_object(
                    g.ObjMeshGeometry.from_file(fname),
                    g.MeshLambertMaterial(color=colors[idx], opacity=alpha),
                )
                self.mc_vis[f"scene/{name}/{idx}"].set_transform(pose)

        # initialize vec fields
        for name, (n_vecs, radius, height, _) in self.vec_field_dict.items():
            for idx in range(n_vecs):
                self.mc_vis[f"scene/{name}/{idx}"].set_object(
                    g.Cylinder(height=height, radius=radius),
                    g.MeshLambertMaterial(color=0xFFFF00, opacity=0.3),
                )
                # move the cylinder so that the end is at origin
                X = np.eye(4)
                X[1, 3] = height / 2
                self.mc_vis[f"scene/{name}/{idx}"].set_transform(X)

        self.anim = Animation()
        self.frame_idx = 0

    def set_urdf_poses(self, urdf_dict: dict[str, dict]) -> None:
        """Set current URDF poses outside the animation timeline."""
        unknown_names = sorted(set(urdf_dict) - set(self.urdfs_dict))
        if unknown_names:
            raise KeyError(
                f"URDF poses contain uninitialized scene names: {unknown_names}."
            )
        for name, urdf_info in urdf_dict.items():
            urdf_inst, mesh_type, _, _ = self.urdfs_dict[name]
            base_pose = np.eye(4)
            base_pose[:3, 3] = urdf_info["pos"]
            base_pose[:3, :3] = urdf_info["rot"]
            joint_dict = urdf_info["joints"]
            if mesh_type == "collision":
                fk = urdf_inst.collision_trimesh_fk(cfg=joint_dict)
            else:
                fk = urdf_inst.visual_trimesh_fk(cfg=joint_dict)
            for idx, pose in enumerate(fk.values()):
                self.mc_vis[f"scene/{name}/{idx}"].set_transform(base_pose @ pose)

    def add_frame(
        self,
        pts_dict={},
        pts_rot_dict={},
        urdf_dict={},
        vec_field_dict={},
        time_sec=None,
    ):
        frame_time = self.frame_idx * self.dt if time_sec is None else float(time_sec)

        with self.anim.at_frame(self.mc_vis, frame_time) as frame:
            print(f"Adding frame {self.frame_idx} at time {frame_time:.2f} seconds")
            for name, pts in pts_dict.items():
                pts_indices = self.pts_dict[name][2]
                pts_sampled = pts[pts_indices]
                is_posed = self.pts_dict[name][3]

                if is_posed:
                    if pts_rot_dict[name].ndim == 2:
                        # normal vectors are given
                        normals = pts_rot_dict[name][pts_indices]
                        assert normals.shape == (
                            pts_sampled.shape[0],
                            3,
                        ), f"Got {normals.shape}, expected {(pts_sampled.shape[0], 3)}"
                        rots_sampled = batched_rotation_matrices_from_normals(normals)

                        # cylinder is aligned with x-axis, need to rotate by 90 degrees around y-axis
                        # for the last axis, x y z -> y z x
                        R_cyl_to_normals = R.from_euler(
                            "xyz", [np.pi / 2, 0, 0]
                        ).as_matrix()[None, ...]
                        rots_sampled = rots_sampled @ R_cyl_to_normals

                    else:
                        rots_sampled = pts_rot_dict[name][pts_indices]
                else:
                    rots_sampled = np.eye(3)[None, :, :].repeat(
                        pts_sampled.shape[0], axis=0
                    )

                for idx in range(pts_sampled.shape[0]):
                    center = pts_sampled[idx]
                    X = np.eye(4)
                    X[:3, 3] = center
                    X[:3, :3] = rots_sampled[idx]
                    frame[f"scene/{name}/{idx}"].set_transform(X)

            for name, urdf_info in urdf_dict.items():
                urdf_inst = self.urdfs_dict[name][0]
                mesh_type = self.urdfs_dict[name][1]
                base_pose = np.eye(4)
                base_pose[:3, 3] = urdf_info["pos"]
                base_pose[:3, :3] = urdf_info["rot"]

                joint_dict = urdf_info["joints"]
                if mesh_type == "collision":
                    fk = urdf_inst.collision_trimesh_fk(cfg=joint_dict)
                else:
                    fk = urdf_inst.visual_trimesh_fk(cfg=joint_dict)

                for idx, (mesh, pose) in enumerate(fk.items()):
                    frame[f"scene/{name}/{idx}"].set_transform(base_pose @ pose)

            vec_field_colormap = cm.get_cmap("viridis")
            for name, field_info in vec_field_dict.items():
                positions = field_info["positions"]
                directions = field_info["directions"]
                # the magnitude of each vector is in directions

                # colors should be based on magnitude
                magnitudes = np.linalg.norm(directions, axis=-1)
                # only those in 0 to 0.2 meters are shown
                max_magnitude = self.vec_field_dict[name][3]
                valid_indices = magnitudes <= max_magnitude

                # zero the invalid directions
                directions = directions.copy()
                directions[~valid_indices] = 0.0
                magnitudes[~valid_indices] = 1e-3

                norm = Normalize(vmin=0, vmax=max_magnitude)
                colors = vec_field_colormap(norm(magnitudes))[:, :3]  # (N, 3)
                # convert to hex
                colors = [mc.rgb2hex(c)[1:] for c in colors]
                colors = [int(c, 16) for c in colors]

                n_vecs = self.vec_field_dict[name][0]
                rot_mats = batched_rotation_matrices_from_normals(directions)
                R_cyl_to_normals = R.from_euler("xyz", [np.pi / 2, 0, 0]).as_matrix()[
                    None, ...
                ]
                rot_mats = rot_mats @ R_cyl_to_normals

                for idx in range(n_vecs):

                    pos = positions[idx]
                    vec = directions[idx]
                    ht = magnitudes[idx]
                    arrow_ht = self.vec_field_dict[name][2]

                    # the center of the cylinder should be at pos + vec / 2
                    pose = np.eye(4)
                    pose[:3, 3] = pos + (arrow_ht / 2.0) * (vec / ht)
                    pose[:3, :3] = rot_mats[idx]

                    frame[f"scene/{name}/{idx}"].set_transform(
                        pose @ tf.scale_matrix(factor=ht * 10.0)
                    )
                    # frame[f'scene/{name}/{idx}'].set_property("material.color", 'Color', col)

        self.frame_idx += 1

    def play(self, repetitions: int = 1) -> None:
        self.mc_vis.set_animation(self.anim, repetitions=repetitions)
