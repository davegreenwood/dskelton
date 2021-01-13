# pylint: disable=relative-beyond-top-level,no-member,not-callable

import torch
from .data import SKELETONS
from .geometry import (eye_batch, rotation_6d_to_matrix,
                       matrix_to_rotation_6d,  batch_vector_rotation)


class Skeleton(torch.nn.Module):
    """Base class of skeleton."""

    def __init__(self, skeleton=None, **kwargs):
        """Load the default skeleton values."""
        super().__init__()
        self.device = kwargs.get("device", torch.device("cpu"))
        self.dtype = kwargs.get("dtype", torch.float32)

        d = SKELETONS.get(skeleton, SKELETONS["UDH_UPPER"])
        self.names = d["names"]
        self.n = len(self.names)

        # there may not be a reference point set
        _xyz = d.get("xyz", None)
        if _xyz is None:
            xyz = torch.ones(
                1, self.n, 3, device=self.device, dtype=self.dtype)
        else:
            xyz = torch.tensor(
                _xyz, device=self.device, dtype=self.dtype)[None, ...]

        parent = torch.tensor(d["parent"], device=self.device)
        child = torch.tensor(d["child"], device=self.device)

        self.register_buffer("parent_idx", parent)
        self.register_buffer("child_idx", child)
        self.register_buffer("xyz", xyz)

    def _rotation2points(self, rot):
        """
        Return the XYZ locations of the landmarks, rotated by the 
        batch of rotation matrices. Requires valid reference skeleton xyz.
        """
        n = rot.shape[0]
        device = self.device
        dtype = self.dtype
        rot = rot.to(device=device, dtype=dtype)

        M = [eye_batch(n, 4, device, dtype) for _ in range(self.n)]

        def _m(r, t):
            m = eye_batch(n, 4, device, dtype) 
            m[:, :3, :3] = r
            m[:, :3, 3] = (r @ t[..., None])[..., 0]
            return m

        M[0] = _m(rot[:, 0], self.xyz[:, 0])

        for p, c in zip(self.parent_idx, self.child_idx):
            r, t = rot[:, c], self.xyz[:, c] - self.xyz[:, p]
            M[c] = M[p].clone() @ _m(r, t)

        return torch.stack(M, dim=1)[..., :3, 3]

    def _points2rotation(self, xyz):
        """
        Return the angles between each joint in 9dof rotation matrices.
        Requires a valid reference skeleton at self.xyz.
        Root rotation cannot be determined from landmarks, so is identity.
        """
        n = xyz.shape[0]
        device = self.device
        dtype = self.dtype
        xyz = xyz.to(device=device, dtype=dtype)

        # start with zero rotations
        R = [eye_batch(n, 3, device, dtype) for _ in range(self.n)]
        Rs = [eye_batch(n, 3, device, dtype) for _ in range(self.n)]

        def _m(p, c):
            t = self.xyz[:, c] - self.xyz[:, p]
            a = (Rs[p] @ t[..., None])[..., 0]
            b = xyz[:, c] - xyz[:, p]
            R[c] = batch_vector_rotation(a, b)
            Rs[c] = Rs[p] @ R[c].clone()

        for p, c in zip(self.parent_idx, self.child_idx):
            _m(p, c)
        return torch.stack(R, dim=1)

    def angles(self, xyz: torch.tensor) -> torch.Tensor:
        """
        Return the angles between each joint in 6d rotation matrices.
        xyz: torch.tensor of shape [N_batches, n_joints, 6]
        see: https://zhouyisjtu.github.io/project_rotation/rotation.html
        """
        R = self._points2rotation(xyz)
        return matrix_to_rotation_6d(R)

    def points(self, angles: torch.tensor) -> torch.Tensor:
        """
        Return the 3d XYZ point locations of the landmarks. 
        angles: torch.tensor of shape [N_batches, n_joints, 6]
        Requires valid reference skeleton xyz.
        """
        angles = rotation_6d_to_matrix(angles)
        return self._rotation2points(angles)
