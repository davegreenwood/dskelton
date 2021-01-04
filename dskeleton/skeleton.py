import torch
from .data import SKELETONS
from .geometry import (
    rotation_6d_to_matrix, matrix_to_rotation_6d,  batch_vector_rotation)


class Skeleton(torch.nn.Module):
    """Base class of skeleton."""

    def __init__(self, skeleton=None):
        """Load the default skeleton values."""
        super().__init__()
        d = SKELETONS.get(skeleton, SKELETONS["UDH_UPPER"])
        self.names = d["names"]
        self.chain = {c:p for p, c in zip(d["parent"], d["child"])}
        self.n = len(self.names)
        xyz = torch.tensor(d.get("xyz", torch.ones(self.n, 3)))[None, ...]
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

    def _parent_chain(self, idx):
        """return the list of parent indices, in root first order."""
        p = [idx]
        while True:
            i = self.chain.get(p[-1], None)
            if i is None:
                break
            p.append(i)
        return list(reversed(p))

    def angles(self, xyz: torch.tensor) -> torch.Tensor:
        """
        Return the angles between each joint in 6d rotation matrices.
        see: https://zhouyisjtu.github.io/project_rotation/rotation.html
        For the (first) root angle, a zero rotation is prepended to the result.
        """
        n, device = xyz.shape[0], xyz.device
        R = [torch.eye(3, device=device)[None,...].repeat(n, 1, 1),]

        def _parent_rotation(idx):
            """pre multiply the parent rotation hierarchy."""
            r = torch.eye(3, device=device)[None,...].repeat(n, 1, 1)
            for i in self._parent_chain(idx):
                r = r @ R[i].clone()
            return r

        def _ref(parent, child):
            """Expand and rotate the reference vector."""
            r = _parent_rotation(parent)
            a = (self.xyz[:, child] - self.xyz[:, parent])
            a = a.repeat(n, 1)[..., None]
            a = r @ a
            return a[..., 0]

        for parent, child in zip(self.parent_idx, self.child_idx):
            a = _ref(parent, child)
            b = xyz[:, child] - xyz[:, parent]
            R.append(batch_vector_rotation(a, b))

        return matrix_to_rotation_6d(torch.stack(R, dim=1))

    def points(self, angles: torch.tensor) -> torch.Tensor:
        """
        Return the XYZ locations of the landmarks, rotated by the angles.
        Requires valid reference skeleton xyz.
        """
        angles = rotation_6d_to_matrix(angles)
        n, device = angles.shape[0], angles.device

        R0 = torch.zeros(n, 4, 4, device=device)
        R0[:, 3, 3] = 1.0
        R0[:, :3, :3] = angles[:, 0, ...]
        R0[:, :3, 3] = self.xyz[:, 0, :]
        R = [R0]

        for parent, child in zip(self.parent_idx, self.child_idx):
            M = torch.zeros(n, 4, 4, device=device)
            M[:, 3, 3] = 1.0
            M[:, :3, :3] = angles[:, child, ...]
            M[:, :3, 3] = self.xyz[:, child, :] - self.xyz[:, parent, :]
            R.append(R[parent].clone() @ M)

        return torch.stack(R, dim=1)[..., :3, 3]
