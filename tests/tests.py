import unittest
import numpy as np
import torch
import torch.nn.functional as F

from dskeleton.skeleton import Skeleton
from dskeleton.geometry import batch_vector_rotation


ZR = torch.eye(3)[None, ...].repeat(4, 1, 1)

R = torch.tensor([
    [[1, 0, 0],
     [0, 1, 0],
     [0, 0, 1]],
    [[0.7071, 0.7071, 0.0],
     [-0.7071, 0.7071, 0],
     [0, 0, 1]],
    [[0, -1, 0],
     [1, 0, 0],
     [0, 0, 1]],
    [[0.7071, -0.7071, 0],
     [0.7071, 0.7071, 0],
     [0, 0, 1]]
])

X = torch.tensor([
    [0, 0, 0],
    [0.7071, 0.7071, 0],
    [0, 1.4142, 0],
    [1.7071, 0.7071, 0]
])


class SkeletonTests(unittest.TestCase):
    """Test dskeleton."""
    pts = torch.tensor(np.load("tests/pts100.npy"))
    ang = torch.tensor(np.load("tests/ang100.npy"))

    def test_data(self):
        pts_shape = self.pts.shape
        ang_shape = self.ang.shape
        self.assertEqual(len(pts_shape), 3)
        self.assertEqual(len(ang_shape), 3)

    def test_default(self):
        """Test number of lengths is greater than 0."""
        skel = Skeleton()
        self.assertGreater(skel.n, 0)

    def test_body25(self):
        """Test body 25 has 25 points"""
        skel = Skeleton("BODY_25")
        self.assertEqual(skel.n, 25)

    def test_calculate_length(self):
        """Test sum of lengths is greater than k = self.n."""
        skel = Skeleton("UDH_UPPER")
        k = skel.lengths.sum()
        skel.calculate_lengths(self.pts)
        self.assertGreater(skel.lengths.sum(), k)
        self.assertTrue((skel.lengths > 0).all())

    def test_rotation2points1(self):
        """Test that identity does not apply rotation."""
        skel = Skeleton("TEST")
        xyz = skel._rotation2points(ZR[None, ...])
        self.assertTrue(torch.allclose(skel.xyz[0], xyz))

    def test_rotation2points2(self):
        """Test rotation retruns test points."""
        tol = 1e-5
        skel = Skeleton("TEST")
        xyz = skel._rotation2points(R[None, ...])
        result = torch.allclose(X, xyz[0, ...], atol=tol, rtol=tol)
        self.assertTrue(result)

    def test_points2rotation1(self):
        """Test that X points returns R rotations."""
        tol = 1e-5
        skel = Skeleton("TEST")
        rot = skel._points2rotation(X[None, ...])
        result = torch.allclose(R, rot, atol=tol, rtol=tol)
        self.assertTrue(result)

    def test_roundtrip(self):
        """Test that X points returns R rotations that make X points."""
        tol = 1e-5
        skel = Skeleton("TEST")
        rot = skel._points2rotation(X[None, ...])
        xyz = skel._rotation2points(rot)
        result = torch.allclose(X, xyz, atol=tol, rtol=tol)
        self.assertTrue(result)

    def test_roundtrip2(self):
        """Test that X points returns R rotations that make X points."""
        tol = 1e-5
        skel = Skeleton("TEST")
        xyz = skel._rotation2points(R[None, ...])
        _r = skel._points2rotation(xyz)
        result = torch.allclose(R, _r, atol=tol, rtol=tol)
        self.assertTrue(result)


class GeometryTests(unittest.TestCase):

    def test_rotation1(self):
        n = 1
        a = F.normalize(torch.rand(n, 3))
        b = F.normalize(torch.rand(n, 3))
        r = batch_vector_rotation(a, b)[0]
        _a = r @ a[0]
        self.assertTrue(torch.allclose(_a, b))

    def test_rotation2(self):
        n = 7
        a = F.normalize(torch.rand(n, 3))
        b = F.normalize(torch.rand(n, 3))
        r = batch_vector_rotation(a, b)
        _a = r @ a[..., None]
        _b = b[..., None]
        self.assertTrue(torch.allclose(_a, _b))

    def test_rotation3(self):
        """Test two orthoganal vectors return 90 degree rotations."""
        a = torch.tensor([[0, 1, 0.], [-1, 0, 0.]])
        b = torch.tensor([[-1, 0, 0.], [0, 1, 0.]])

        _r = torch.tensor([[[0., -1.,  0.],
                            [1.,  0.,  0.],
                            [0.,  0.,  1.]],
                           [[0.,  1.,  0.],
                            [-1.,  0.,  0.],
                            [0.,  0.,  1.]]])

        r = batch_vector_rotation(a, b)
        self.assertTrue(torch.allclose(_r, r))
