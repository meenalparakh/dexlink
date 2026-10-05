import numpy as np
import trimesh
from trimesh.creation import box, cylinder
from trimesh.util import concatenate

from dexlink.utils.retargeting.path_utils import get_assets_dir

# ============================================================
# PARAMETERS
# ============================================================

# Container
container_outer_radius = 0.04
container_inner_radius = 0.035
container_height = 0.12
base_thickness = 0.004
num_wall_segments = 64

# Lid
lid_inner_radius = container_outer_radius + 0.004
lid_outer_radius = lid_inner_radius + 0.004
lid_height = 0.03
lid_top_thickness = 0.003
lid_segments = 64

# ============================================================
# HELPER
# ============================================================


def create_ring_wall(inner_radius, outer_radius, height, segments):
    """
    Create a cylindrical wall using thin boxes arranged in a circle.
    """

    wall_thickness = outer_radius - inner_radius
    mean_radius = (outer_radius + inner_radius) / 2.0

    arc_length = 2.0 * np.pi * mean_radius / segments

    pieces = []

    for i in range(segments):
        theta = 2.0 * np.pi * i / segments

        # Box dimensions
        extents = [arc_length, wall_thickness, height]

        piece = box(extents=extents)

        # Tangential orientation
        rot = trimesh.transformations.rotation_matrix(theta, [0, 0, 1])

        # Position around circle
        x = mean_radius * np.cos(theta)
        y = mean_radius * np.sin(theta)
        z = height / 2.0

        trans = trimesh.transformations.translation_matrix([x, y, z])

        transform = trimesh.transformations.concatenate_matrices(trans, rot)

        piece.apply_transform(transform)

        pieces.append(piece)

    return concatenate(pieces)


# ============================================================
# CONTAINER
# ============================================================

# Thin cylindrical base
base = cylinder(radius=container_outer_radius, height=base_thickness, sections=64)

# Move base so bottom sits at z=0
base.apply_translation([0, 0, base_thickness / 2.0])

# Cylindrical wall made from boxes
wall = create_ring_wall(
    inner_radius=container_inner_radius,
    outer_radius=container_outer_radius,
    height=container_height,
    segments=num_wall_segments,
)

# Combine
container_mesh = concatenate([base, wall])

# Export
container_mesh.export(str(get_assets_dir() / "box_with_lid" / "container.obj"))

print("Saved container.obj")

# ============================================================
# LID
# ============================================================

# Lid top plate
lid_top = cylinder(radius=lid_outer_radius, height=lid_top_thickness, sections=64)

lid_top.apply_translation([0, 0, lid_height - lid_top_thickness / 2.0])

# Lid skirt / side wall
lid_wall = create_ring_wall(
    inner_radius=lid_inner_radius,
    outer_radius=lid_outer_radius,
    height=lid_height,
    segments=lid_segments,
)

# Combine
lid_mesh = concatenate([lid_top, lid_wall])

# Export
lid_mesh.export(str(get_assets_dir() / "box_with_lid" / "lid.obj"))

print("Saved lid.obj")


# ============================================================
# OPTIONAL VISUALIZATION
# ============================================================

scene = trimesh.Scene()

container_vis = container_mesh.copy()
lid_vis = lid_mesh.copy()

lid_vis.apply_translation([0.0, 0.0, container_height + 0.01])

scene.add_geometry(container_vis)
scene.add_geometry(lid_vis)

scene.show()
