# pylint: disable=relative-beyond-top-level,no-member,not-callable

import torch
from .data import SKELETONS
from .geometry import (eye_batch, rotation_6d_to_matrix,
                       matrix_to_rotation_6d,  batch_vector_rotation)


class Skeleton(torch.nn.Module):
    """Base class of skeleton."""

    def __init__(self, skeleton=None):
        """Load the default skeleton values."""
        super().__init__()
        d = SKELETONS.get(skeleton, SKELETONS["UDH_UPPER"])
        self.names = d["names"]
        self.n = len(self.names)

        # there may not be a reference point set
        _xyz = d.get("xyz", None)
        if _xyz is None:
            xyz = torch.ones(1, self.n, 3)
        else:
            xyz = torch.tensor(_xyz)[None, ...]

        self.register_buffer("parent_idx", torch.tensor(d["parent"]))
        self.register_buffer("child_idx", torch.tensor(d["child"]))
        self.register_buffer("lengths", torch.ones(self.n))
        self.register_buffer("xyz", xyz)
        self.calculate_lengths(self.xyz)

    def calculate_lengths(self, xyz: torch.tensor):
        """
        From xyz data, calculate the bone lengths.
        Data is shape: [num_examples, num_bones, 3].
        """
        diff = xyz[:, self.child_idx] - xyz[:, self.parent_idx]
        self.lengths = torch.linalg.norm(diff, dim=-1)

    def _rotation2points(self, rot):
        """
        Return the XYZ locations of the landmarks, rotated by the 
        batch of rotation matrices. Requires valid reference skeleton xyz.
        """
        n, device = rot.shape[0], rot.device
        M = [eye_batch(n, 4, device) for _ in range(self.n)]

        def _m(r, t):
            m = eye_batch(n, 4, device) 
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
        n, device = xyz.shape[0], xyz.device
        # start with zero rotations
        R = [eye_batch(n, 3, device) for _ in range(self.n)]
        Rs = [eye_batch(n, 3, device) for _ in range(self.n)]

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
