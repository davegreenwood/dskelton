import torch
import torch.nn.functional as F


def eye_batch(batch_n, eye_n, device, dtype=torch.float32):
    """Return an identity matrix of size eye_n, repeated batch_n times."""
    return torch.eye(
        eye_n, device=device, dtype=dtype)[None, ...].repeat(batch_n, 1, 1)


def rotation_6d_to_matrix(d6: torch.Tensor) -> torch.Tensor:
    """
    Converts 6D rotation representation by Zhou et al. [1] to rotation matrix
    using Gram--Schmidt orthogonalisation per Section B of [1].
    Args:
        d6: 6D rotation representation, of size (*, 6)

    Returns:
        batch of rotation matrices of size (*, 3, 3)

    [1] Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H.
    On the Continuity of Rotation Representations in Neural Networks.
    IEEE Conference on Computer Vision and Pattern Recognition, 2019.
    Retrieved from http://arxiv.org/abs/1812.07035
    """

    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = F.normalize(a1, dim=-1)
    b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
    b2 = F.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-2)


def matrix_to_rotation_6d(matrix: torch.Tensor) -> torch.Tensor:
    """
    Converts rotation matrices to 6D rotation representation by Zhou et al. [1]
    by dropping the last row. Note that 6D representation is not unique.
    Args:
        matrix: batch of rotation matrices of size (*, 3, 3)

    Returns:
        6D rotation representation, of size (*, 6)

    [1] Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H.
    On the Continuity of Rotation Representations in Neural Networks.
    IEEE Conference on Computer Vision and Pattern Recognition, 2019.
    Retrieved from http://arxiv.org/abs/1812.07035
    """
    return matrix[..., :2, :].clone().reshape(*matrix.size()[:-2], 6)


def batch_vector_rotation(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    The batch rotation matrix that rotates vector a to vector b.

    Returns R such that R @ a.T = b.T where
    'a' and 'b' must be  [N, (x, y, z)] unit vectors, and N is the batch size.

    .. math::R = I + [v]_{\times} + [v]_{\times}^2\frac{1-c}{s^2}

    See: 
    https://math.stackexchange.com/a/476311/812637

    """
    n, m, device, dtype = a.shape[0], b.shape[0], a.device, a.dtype
    assert n == m, f"a ({n}) and b ({m}) must be same size first dimension"

    # normalise
    a = F.normalize(a, dim=-1)
    b = F.normalize(b, dim=-1)

    # store results
    R = torch.zeros(n, 3, 3, dtype=dtype, device=device)

    # if vectors a & b coincide - mask out
    pos_mask = torch.isclose(a, b).sum(-1) > 2
    neg_mask = torch.isclose(a, -b).sum(-1) > 2
    R[pos_mask, ...] = torch.eye(3, dtype=dtype, device=device)
    R[neg_mask, ...] = -torch.eye(3, dtype=dtype, device=device)

    # where vectors a & b do not coincide
    inv_mask = ~ (pos_mask + neg_mask)
    R[inv_mask, ...] = _angle_between_R(a, b)
    return R


def _angle_between_R(a, b):
    """Use rodrigues to find the angle between two unit vectors."""
    n, device, dtype = a.shape[0], a.device, a.dtype
    axis = F.normalize(torch.cross(a, b), dim=1)
    x, y, z = axis[:, 0, ...], axis[:, 1, ...], axis[:, 2, ...]
    cos = (a[:, None, ...] @ b[:, ..., None])[..., 0, 0]
    sin = torch.sqrt(1 - cos ** 2)
    csm = 1.0 - cos

    # Rodrigues formula
    R = torch.zeros(n, 3, 3, device=device, dtype=dtype)
    R[:, 0, 0] = cos + x * x * csm
    R[:, 1, 0] = z * sin + y * x * csm
    R[:, 2, 0] = -y * sin + z * x * csm
    R[:, 0, 1] = -z * sin + x * y * csm
    R[:, 1, 1] = cos + y * y * csm
    R[:, 2, 1] = x * sin + z * y * csm
    R[:, 0, 2] = y * sin + x * z * csm
    R[:, 1, 2] = -x * sin + y * z * csm
    R[:, 2, 2] = cos + z * z * csm
    return R


def _angle_between_skew(a, b):
    """Use skew cross-product to find the angle between two unit vectors."""
    n, device, dtype = a.shape[0], a.device, a.dtype
    # cross and dot products
    crs = torch.cross(a, b)
    csp = 1.0 + (a[:, None, ...] @ b[:, ..., None])[..., 0, 0]
    eye = eye_batch(n, 3, device, dtype)

    # skew-symmetric cross-product matrix
    skew = torch.zeros(n, 3, 3, device=device, dtype=dtype)
    skew[:, 0, 1] = -crs[:, 2]
    skew[:, 0, 2] = crs[:, 1]
    skew[:, 1, 0] = crs[:, 2]
    skew[:, 1, 2] = -crs[:, 0]
    skew[:, 2, 0] = -crs[:, 1]
    skew[:, 2, 1] = crs[:, 0]
    return eye + skew + skew @ skew / csp