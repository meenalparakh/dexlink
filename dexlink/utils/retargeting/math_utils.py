import numpy as np
import torch
from scipy.spatial.transform import Rotation as R


def get_X(pos, quat):
    # quat is in (, 4) wxyz
    return_batched = len(pos.shape) > 1

    pos = pos.reshape((-1, 3))
    quat = quat.reshape((-1, 4))

    quat_xyzw = np.roll(quat, shift=-1, axis=1)

    n_batches = pos.shape[0]

    X = np.zeros((n_batches, 4, 4))
    X[:, 3, 3] = 1.0

    X[:, :3, :3] = R.from_quat(quat_xyzw).as_matrix()
    X[:, :3, 3] = pos

    if return_batched:
        return X
    else:
        return X[0]


def check_shape(tensor, expected_shape):
    if tensor.shape != expected_shape:
        raise ValueError(f"Expected shape {expected_shape}, but got {tensor.shape}.")


@torch.jit.script
def quat_conjugate(q: torch.Tensor) -> torch.Tensor:
    """Computes the conjugate of a quaternion.

    Args:
        q: The quaternion orientation in (w, x, y, z). Shape is (..., 4).

    Returns:
        The conjugate quaternion in (w, x, y, z). Shape is (..., 4).
    """
    shape = q.shape
    q = q.reshape(-1, 4)
    return torch.cat((q[:, 0:1], -q[:, 1:]), dim=-1).view(shape)


@torch.jit.script
def quat_mul(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """Multiply two quaternions together.

    Args:
        q1: The first quaternion in (w, x, y, z). Shape is (..., 4).
        q2: The second quaternion in (w, x, y, z). Shape is (..., 4).

    Returns:
        The product of the two quaternions in (w, x, y, z). Shape is (..., 4).

    Raises:
        ValueError: Input shapes of ``q1`` and ``q2`` are not matching.
    """
    # check input is correct
    if q1.shape != q2.shape:
        msg = f"Expected input quaternion shape mismatch: {q1.shape} != {q2.shape}."
        raise ValueError(msg)
    # reshape to (N, 4) for multiplication
    shape = q1.shape
    q1 = q1.reshape(-1, 4)
    q2 = q2.reshape(-1, 4)
    # extract components from quaternions
    w1, x1, y1, z1 = q1[:, 0], q1[:, 1], q1[:, 2], q1[:, 3]
    w2, x2, y2, z2 = q2[:, 0], q2[:, 1], q2[:, 2], q2[:, 3]
    # perform multiplication
    ww = (z1 + x1) * (x2 + y2)
    yy = (w1 - y1) * (w2 + z2)
    zz = (w1 + y1) * (w2 - z2)
    xx = ww + yy + zz
    qq = 0.5 * (xx + (z1 - x1) * (x2 - y2))
    w = qq - ww + (z1 - y1) * (y2 - z2)
    x = qq - xx + (x1 + w1) * (x2 + w2)
    y = qq - yy + (w1 - x1) * (y2 + z2)
    z = qq - zz + (z1 + y1) * (w2 - x2)

    return torch.stack([w, x, y, z], dim=-1).view(shape)


@torch.jit.script
def matrix_from_quat(quaternions: torch.Tensor) -> torch.Tensor:
    """Convert rotations given as quaternions to rotation matrices.

    Args:
        quaternions: The quaternion orientation in (w, x, y, z). Shape is (..., 4).

    Returns:
        Rotation matrices. The shape is (..., 3, 3).

    Reference:
        https://github.com/facebookresearch/pytorch3d/blob/main/pytorch3d/transforms/rotation_conversions.py#L41-L70
    """
    r, i, j, k = torch.unbind(quaternions, -1)
    # pyre-fixme[58]: `/` is not supported for operand types `float` and `Tensor`.
    two_s = 2.0 / (quaternions * quaternions).sum(-1)

    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))


def depth_to_pointcloud(depth, K, min_depth=0.0, max_depth=np.inf):
    """
    Convert a depth image to a 3D point cloud.

    Args:
        depth: (H, W) depth map in meters.
        K: (3, 3) camera intrinsic matrix:
           [[fx, 0, cx],
            [0, fy, cy],
            [0,  0,  1]]

    Returns:
        points: (N, 3) array of 3D points in camera coordinates.
    """
    H, W = depth.shape
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    # Create a grid of pixel coordinates
    u, v = np.meshgrid(np.arange(W), np.arange(H))

    # Flatten and convert to 3D
    Z = depth.flatten()
    X = (u.flatten() - cx) * Z / fx
    Y = (v.flatten() - cy) * Z / fy

    # Stack into Nx3
    points = np.vstack((X, Y, Z)).T

    # Remove invalid depths (0 or NaN)
    mask = (Z > min_depth) & (Z < max_depth) & np.isfinite(Z)
    points = points[mask]

    return points


def make_rotation_matrix_from_normal(normal):
    # Ensure unit normal
    n = normal / np.linalg.norm(normal)

    # Choose an arbitrary vector that’s not parallel to n
    if abs(n[2]) < 0.9:
        tangent = np.array([0, 0, 1.0])
    else:
        tangent = np.array([1.0, 0, 0])

    # Build orthonormal basis
    x_axis = np.cross(tangent, n)
    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(n, x_axis)
    y_axis /= np.linalg.norm(y_axis)

    # Rotation matrix [x y z]
    R = np.column_stack((x_axis, y_axis, n))
    return R


@torch.jit.script
def axis_angle_from_quat(quat: torch.Tensor, eps: float = 1.0e-6) -> torch.Tensor:
    """Convert rotations given as quaternions to axis/angle.

    Args:
        quat: The quaternion orientation in (w, x, y, z). Shape is (..., 4).
        eps: The tolerance for Taylor approximation. Defaults to 1.0e-6.

    Returns:
        Rotations given as a vector in axis angle form. Shape is (..., 3).
        The vector's magnitude is the angle turned anti-clockwise in radians around the vector's direction.

    Reference:
        https://github.com/facebookresearch/pytorch3d/blob/main/pytorch3d/transforms/rotation_conversions.py#L526-L554
    """
    # Modified to take in quat as [q_w, q_x, q_y, q_z]
    # Quaternion is [q_w, q_x, q_y, q_z] = [cos(theta/2), n_x * sin(theta/2), n_y * sin(theta/2), n_z * sin(theta/2)]
    # Axis-angle is [a_x, a_y, a_z] = [theta * n_x, theta * n_y, theta * n_z]
    # Thus, axis-angle is [q_x, q_y, q_z] / (sin(theta/2) / theta)
    # When theta = 0, (sin(theta/2) / theta) is undefined
    # However, as theta --> 0, we can use the Taylor approximation 1/2 - theta^2 / 48
    quat = quat * (1.0 - 2.0 * (quat[..., 0:1] < 0.0))
    mag = torch.linalg.norm(quat[..., 1:], dim=-1)
    half_angle = torch.atan2(mag, quat[..., 0])
    angle = 2.0 * half_angle
    # check whether to apply Taylor approximation
    sin_half_angles_over_angles = torch.where(
        angle.abs() > eps, torch.sin(half_angle) / angle, 0.5 - angle * angle / 48
    )
    return quat[..., 1:4] / sin_half_angles_over_angles.unsqueeze(-1)


@torch.jit.script
def quat_box_minus(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """The box-minus operator (quaternion difference) between two quaternions.

    Args:
        q1: The first quaternion in (w, x, y, z). Shape is (N, 4).
        q2: The second quaternion in (w, x, y, z). Shape is (N, 4).

    Returns:
        The difference between the two quaternions. Shape is (N, 3).

    Reference:
        https://github.com/ANYbotics/kindr/blob/master/doc/cheatsheet/cheatsheet_latest.pdf
    """
    quat_diff = quat_mul(q1, quat_conjugate(q2))  # q1 * q2^-1
    return axis_angle_from_quat(quat_diff)  # log(qd)


@torch.jit.script
def quat_error_magnitude(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """Computes the rotation difference between two quaternions.

    Args:
        q1: The first quaternion in (w, x, y, z). Shape is (..., 4).
        q2: The second quaternion in (w, x, y, z). Shape is (..., 4).

    Returns:
        Angular error between input quaternions in radians.
    """
    axis_angle_error = quat_box_minus(q1, q2)
    return torch.norm(axis_angle_error, dim=-1)


@torch.jit.script
def _sqrt_positive_part(x: torch.Tensor) -> torch.Tensor:
    """Returns torch.sqrt(torch.max(0, x)) but with a zero sub-gradient where x is 0.

    Reference:
        https://github.com/facebookresearch/pytorch3d/blob/main/pytorch3d/transforms/rotation_conversions.py#L91-L99
    """
    ret = torch.zeros_like(x)
    positive_mask = x > 0
    ret[positive_mask] = torch.sqrt(x[positive_mask])
    return ret


@torch.jit.script
def normalize(x: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    """Normalizes a given input tensor to unit length.

    Args:
        x: Input tensor of shape (N, dims).
        eps: A small value to avoid division by zero. Defaults to 1e-9.

    Returns:
        Normalized tensor of shape (N, dims).
    """
    return x / x.norm(p=2, dim=-1).clamp(min=eps, max=None).unsqueeze(-1)


@torch.jit.script
def quat_from_angle_axis(angle: torch.Tensor, axis: torch.Tensor) -> torch.Tensor:
    """Convert rotations given as angle-axis to quaternions.

    Args:
        angle: The angle turned anti-clockwise in radians around the vector's direction. Shape is (N,).
        axis: The axis of rotation. Shape is (N, 3).

    Returns:
        The quaternion in (w, x, y, z). Shape is (N, 4).
    """
    # if axis is None:
    #     _angle = torch.norm(angle, dim=-1)
    #     axis = angle / _angle.unsqueeze(-1).clamp(min=1e-9, max=None)
    #     angle = _angle

    theta = (angle / 2).unsqueeze(-1)
    xyz = normalize(axis) * theta.sin()
    w = theta.cos()
    return normalize(torch.cat([w, xyz], dim=-1))


@torch.jit.script
def quat_from_rotvec(rotvec: torch.Tensor) -> torch.Tensor:
    """Convert rotations given as axis-angle vectors to quaternions.

    Args:
        rotvec: Axis-angle vectors where direction is the rotation axis and
            magnitude is the rotation angle in radians. Shape is (..., 3).

    Returns:
        The quaternion in (w, x, y, z). Shape is (..., 4).
    """
    angle = torch.linalg.norm(rotvec, dim=-1)
    axis = normalize(rotvec)
    return quat_from_angle_axis(angle, axis)


@torch.jit.script
def quat_slerp(
    start_quat: torch.Tensor,
    end_quat: torch.Tensor,
    alpha: float,
    shortest_path: bool = True,
) -> torch.Tensor:
    """Spherically interpolate between two quaternions.

    Args:
        start_quat: Start quaternion in (w, x, y, z). Shape is (..., 4).
        end_quat: End quaternion in (w, x, y, z). Shape is (..., 4).
        alpha: Interpolation factor in [0, 1].
        shortest_path: If true, flips the target quaternion sign when needed so
            interpolation follows the shorter arc on S^3.

    Returns:
        Interpolated quaternion in (w, x, y, z). Shape is (..., 4).

    Reference:
        Shoemake, "Animating Rotation with Quaternion Curves", 1985.
    """
    if start_quat.shape != end_quat.shape:
        raise ValueError(
            "Expected matching quaternion shapes, got "
            f"{start_quat.shape} and {end_quat.shape}."
        )
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"Expected alpha in [0, 1], got {alpha}.")

    q0 = normalize(start_quat)
    q1 = normalize(end_quat)
    dot = torch.sum(q0 * q1, dim=-1, keepdim=True)
    if shortest_path:
        q1 = torch.where(dot < 0.0, -q1, q1)
        dot = torch.sum(q0 * q1, dim=-1, keepdim=True)

    dot = dot.clamp(-1.0, 1.0)
    theta = torch.acos(dot)
    sin_theta = torch.sin(theta)
    alpha_t = torch.full_like(dot, alpha)
    lerped = (1.0 - alpha_t) * q0 + alpha_t * q1
    close = sin_theta.abs() < 1.0e-6
    s0 = torch.sin((1.0 - alpha_t) * theta) / sin_theta
    s1 = torch.sin(alpha_t * theta) / sin_theta
    slerped = s0 * q0 + s1 * q1
    return normalize(torch.where(close, lerped, slerped))


@torch.jit.script
def quat_from_matrix(matrix: torch.Tensor) -> torch.Tensor:
    """Convert rotations given as rotation matrices to quaternions.

    Args:
        matrix: The rotation matrices. Shape is (..., 3, 3).

    Returns:
        The quaternion in (w, x, y, z). Shape is (..., 4).

    Reference:
        https://github.com/facebookresearch/pytorch3d/blob/main/pytorch3d/transforms/rotation_conversions.py#L102-L161
    """
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix shape {matrix.shape}.")

    batch_dim = matrix.shape[:-2]
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.unbind(
        matrix.reshape(batch_dim + (9,)), dim=-1
    )

    q_abs = _sqrt_positive_part(
        torch.stack(
            [
                1.0 + m00 + m11 + m22,
                1.0 + m00 - m11 - m22,
                1.0 - m00 + m11 - m22,
                1.0 - m00 - m11 + m22,
            ],
            dim=-1,
        )
    )

    # we produce the desired quaternion multiplied by each of r, i, j, k
    quat_by_rijk = torch.stack(
        [
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and `int`.
            torch.stack([q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and `int`.
            torch.stack([m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and `int`.
            torch.stack([m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and `int`.
            torch.stack([m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2], dim=-1),
        ],
        dim=-2,
    )

    # We floor here at 0.1 but the exact level is not important; if q_abs is small,
    # the candidate won't be picked.
    flr = torch.tensor(0.1).to(dtype=q_abs.dtype, device=q_abs.device)
    quat_candidates = quat_by_rijk / (2.0 * q_abs[..., None].max(flr))

    # if not for numerical problems, quat_candidates[i] should be same (up to a sign),
    # forall i; we pick the best-conditioned one (with the largest denominator)
    return quat_candidates[
        torch.nn.functional.one_hot(q_abs.argmax(dim=-1), num_classes=4) > 0.5, :
    ].reshape(batch_dim + (4,))


@torch.jit.script
def rot6d_from_matrix(matrix: torch.Tensor) -> torch.Tensor:
    """Convert rotation matrices to the 6D representation used by quat_from_6d.

    The 6D representation stores the first two rotation-matrix columns as two
    concatenated 3-vectors: [col0, col1].
    """
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix shape {matrix.shape}.")
    return torch.cat((matrix[..., :, 0], matrix[..., :, 1]), dim=-1)


@torch.jit.script
def quat_from_6d(x: torch.Tensor) -> torch.Tensor:
    a1 = x[..., 0:3]
    a2 = x[..., 3:6]
    b1 = normalize(a1)
    b2 = normalize(a2 - (b1 * a2).sum(-1, keepdim=True) * b1)
    b3 = torch.cross(b1, b2, dim=-1)
    return quat_from_matrix(torch.stack((b1, b2, b3), dim=-1))


def batched_rotation_matrices_from_normals(_normals):
    """
    Compute rotation matrices whose z-axis aligns with the given normals.
    normals: (N, 3) array
    Returns: (N, 3, 3) rotation matrices
    """

    normals = np.copy(_normals)

    # for normals with magnitude 0, set to default z-axis
    zero_mask = np.linalg.norm(normals, axis=1) < 1e-6
    normals[zero_mask] = np.array([0, 0, 1.0])

    n = normals / np.linalg.norm(normals, axis=1, keepdims=True)

    # Choose arbitrary tangents that are not parallel to n
    # Start with global z = [0,0,1], but fix where normals ≈ z
    ref = np.tile(np.array([0, 0, 1.0]), (len(n), 1))
    parallel = np.abs(n[:, 2]) > 0.9
    ref[parallel] = np.array([1.0, 0, 0])  # replace problematic cases

    # x-axis = tangent = ref × n
    x = np.cross(ref, n)
    x /= np.linalg.norm(x, axis=1, keepdims=True)

    # y-axis = n × x
    y = np.cross(n, x)
    y /= np.linalg.norm(y, axis=1, keepdims=True)

    # Stack axes into rotation matrices [x y z]
    R = np.stack((x, y, n), axis=-1)  # shape (N, 3, 3)
    return R


# Adapted from PyTorch3D's point-to-mesh distance tests. PyTorch3D is
# BSD-licensed; see third_party_licenses/PYTORCH3D-BSD.txt.
def eps():
    return 1e-8


def _point_to_bary(point: torch.Tensor, tri: torch.Tensor) -> torch.Tensor:
    """
    Computes the barycentric coordinates of point wrt triangle (tri)
    Note that point needs to live in the space spanned by tri = (a, b, c),
    i.e. by taking the projection of an arbitrary point on the space spanned by tri

    Args:
        point: FloatTensor of shape (3)
        tri: FloatTensor of shape (3, 3)
    Returns:
        bary: FloatTensor of shape (3)
    """
    assert point.dim() == 1 and point.shape[0] == 3
    assert tri.dim() == 2 and tri.shape[0] == 3 and tri.shape[1] == 3

    a, b, c = tri.unbind(0)

    v0 = b - a
    v1 = c - a
    v2 = point - a

    d00 = v0.dot(v0)
    d01 = v0.dot(v1)
    d11 = v1.dot(v1)
    d20 = v2.dot(v0)
    d21 = v2.dot(v1)

    denom = d00 * d11 - d01 * d01 + eps()
    s2 = (d11 * d20 - d01 * d21) / denom
    s3 = (d00 * d21 - d01 * d20) / denom
    s1 = 1.0 - s2 - s3

    bary = torch.tensor([s1, s2, s3])
    return bary


def _is_inside_triangle(point: torch.Tensor, tri: torch.Tensor) -> torch.Tensor:
    """
    Computes whether point is inside triangle tri
    Note that point needs to live in the space spanned by tri = (a, b, c)
    i.e. by taking the projection of an arbitrary point on the space spanned by tri

    Args:
        point: FloatTensor of shape (3)
        tri: FloatTensor of shape (3, 3)
    Returns:
        inside: BoolTensor of shape (1)
    """
    v0 = tri[1] - tri[0]
    v1 = tri[2] - tri[0]
    area = torch.cross(v0, v1).norm() / 2.0

    # check if triangle is a line or a point. In that case, return False
    if area < 5e-3:
        return False
    bary = _point_to_bary(point, tri)
    inside = ((bary >= 0.0) * (bary <= 1.0)).all()
    return inside


def _point_to_edge_distance(
    point: torch.Tensor, edge: torch.Tensor, return_pt_coord: bool = False
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """
    Computes the squared euclidean distance of points to edges
    Args:
        point: FloatTensor of shape (3)
        edge: FloatTensor of shape (2, 3)
    Returns:
        dist: FloatTensor of shape (1)

    If a, b are the start and end points of the segments, we
    parametrize a point p as
        x(t) = a + t * (b - a)
    To find t which describes p we minimize (x(t) - p) ^ 2
    Note that p does not need to live in the space spanned by (a, b)
    """
    s0, s1 = edge.unbind(0)

    s01 = s1 - s0
    norm_s01 = s01.dot(s01)

    same_edge = norm_s01 < eps()
    if same_edge:
        dist = 0.5 * (point - s0).dot(point - s0) + 0.5 * (point - s1).dot(point - s1)
        return dist

    t = s01.dot(point - s0) / norm_s01
    t = torch.clamp(t, min=0.0, max=1.0)
    x = s0 + t * s01
    dist = (x - point).dot(x - point)

    if return_pt_coord:
        return dist, x

    return dist


def point_to_tri_distance(point: torch.Tensor, tri: torch.Tensor) -> torch.Tensor:
    """
    Computes the squared euclidean distance of points to edges
    Args:
        point: FloatTensor of shape (3)
        tri: FloatTensor of shape (3, 3)
    Returns:
        dist: FloatTensor of shape (1)
    """
    a, b, c = tri.unbind(0)
    cross = torch.cross(b - a, c - a)
    norm = cross.norm()
    normal = torch.nn.functional.normalize(cross, dim=0)

    # p0 is the projection of p onto the plane spanned by (a, b, c)
    # p0 = p + tt * normal, s.t. (p0 - a) is orthogonal to normal
    # => tt = dot(a - p, n)
    tt = normal.dot(a) - normal.dot(point)
    p0 = point + tt * normal
    dist_p = tt * tt

    # Compute the distance of p to all edge segments
    e01_dist = _point_to_edge_distance(point, tri[[0, 1]])
    e02_dist = _point_to_edge_distance(point, tri[[0, 2]])
    e12_dist = _point_to_edge_distance(point, tri[[1, 2]])

    with torch.no_grad():
        inside_tri = _is_inside_triangle(p0, tri)

    if inside_tri and (norm > eps()):
        return dist_p
    else:
        if e01_dist.le(e02_dist) and e01_dist.le(e12_dist):
            return e01_dist
        elif e02_dist.le(e01_dist) and e02_dist.le(e12_dist):
            return e02_dist
        else:
            return e12_dist


def point_to_tri_distance_coord(
    point: torch.Tensor, tri: torch.Tensor
) -> (torch.Tensor, torch.Tensor):
    """
    same as above, but also returns the closest point coordinates on the triangle
    """

    a, b, c = tri.unbind(0)
    cross = torch.cross(b - a, c - a)
    norm = cross.norm()
    normal = torch.nn.functional.normalize(cross, dim=0)

    # p0 is the projection of p onto the plane spanned by (a, b, c)
    # p0 = p + tt * normal, s.t. (p0 - a) is orthogonal to normal
    # => tt = dot(a - p, n)
    tt = normal.dot(a) - normal.dot(point)
    p0 = point + tt * normal
    dist_p = tt * tt

    # Compute the distance of p to all edge segments
    e01_dist, e01_coord = _point_to_edge_distance(
        point, tri[[0, 1]], return_pt_coord=True
    )
    e02_dist, e02_coord = _point_to_edge_distance(
        point, tri[[0, 2]], return_pt_coord=True
    )
    e12_dist, e12_coord = _point_to_edge_distance(
        point, tri[[1, 2]], return_pt_coord=True
    )

    with torch.no_grad():
        inside_tri = _is_inside_triangle(p0, tri)

    if inside_tri and (norm > eps()):
        return dist_p, p0
    else:
        if e01_dist.le(e02_dist) and e01_dist.le(e12_dist):
            return e01_dist, e01_coord
        elif e02_dist.le(e01_dist) and e02_dist.le(e12_dist):
            return e02_dist, e02_coord
        else:
            return e12_dist, e12_coord


# ///////////////////////////////////////////////////////////////////////////////////////////////////////


def get_transform(translation, rotation):
    X = np.eye(4)
    X[:3, 3] = translation
    r = R.from_euler(rotation[0], rotation[1]).as_matrix()
    X[:3, :3] = r
    return X
