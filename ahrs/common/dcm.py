# -*- coding: utf-8 -*-
"""
Direction Cosine Matrix
=======================

Rotations are linear transformations in Euclidean three-dimensional spaces
about the origin. They have several representations. The most common to use and
easier to understand are the Direction Cosine Matrices, or Rotation Matrices.

An orientation (attitude) is the difference between two coordinate frames. This
difference is determined by the coordinates of its orthonormal vectors. For
three dimensions the discrepancy between any given frame and the base
coordinate frame is the orientation.

Three vectors :math:`r_1`, :math:`r_2` and :math:`r_3` are the unit vectors
along the principal axes :math:`X`, :math:`Y` and :math:`Z`. An orientation is
defined by a :math:`3\times 3` matrix:

.. math::

    R = \\begin{bmatrix}r_1 & r_2 & r_3\\end{bmatrix} \\in \\mathbb{R}^{3\times 3}

where :math:`r_1`, :math:`r_2` and :math:`r_3` are unit vectors stacked as
columns and represent an orthonormal frame. Thus, :math:`RR^{-1}=RR^T=R^TR=I`,
indicating that the inverse of :math:`R` is its transpose. All matrices
satisfying this condition are called _orthogonal matrices_ .

Its determinant is always +1, so its product with any vector will leave the
vector's length unchanged.

Direction cosines are cosines of angles between a vector and a base coordinate
frame. In this case, the difference between orthogonal vectors :math:`r_i` and
the base frame are describing the Direction Cosines. This orientation matrix is
commonly named the "Direction Cosine Matrix."

References
----------
.. [1] Wikipedia: Direction Cosine.
       (https://en.wikipedia.org/wiki/Direction_cosine)
.. [2] Wikipedia: Rotation Matrix
       (https://mathworld.wolfram.com/RotationMatrix.html)
.. [3] Yi Ma, Stefano Soatto, Jana Kosecka, and S. Shankar Sastry. An
       Invitation to 3-D Vision: From Images to Geometric Models. Springer
       Verlag. 2003.
       (https://www.eecis.udel.edu/~cer/arv/readings/old_mkss.pdf)
.. [4] Huynh, D.Q. Metrics for 3D Rotations: Comparison and Analysis. J Math
       Imaging Vis 35, 155–164 (2009).
.. [5] Howard D Curtis. Orbital Mechanics for Engineering Students (Third
       Edition) Butterworth-Heinemann. 2014.

"""

from typing import NoReturn, Tuple
import numpy as np
from .mathfuncs import *
from .orientation import rotation
from .orientation import rot_seq
from .orientation import shepperd
from .orientation import hughes
from .orientation import chiaverini
from .orientation import itzhack
from .orientation import sarabandi

class DCM(np.ndarray):
    """Direction Cosine Matrix in SO(3)

    """
    def __new__(subtype, array=None, *args, **kwargs):
        if array is None:
            array = np.identity(3)
            if 'q' in kwargs:
                array = DCM.from_q(DCM, kwargs.pop('q'))
            if any(x.lower() in ['x', 'y', 'z'] for x in kwargs):
                array = np.identity(3)
                array = array@rotation('x', kwargs.pop('x', 0.0))
                array = array@rotation('y', kwargs.pop('y', 0.0))
                array = array@rotation('z', kwargs.pop('z', 0.0))
            if 'rpy' in kwargs:
                array = rot_seq('zyx', kwargs.pop('rpy'))
            if 'euler' in kwargs:
                seq, angs = kwargs.pop('euler')
                array = rot_seq(seq, angs)
            if 'axang' in kwargs:
                ax, ang = kwargs.pop('axang')
                array = DCM.from_axisangle(ax, ang)
        if array.shape[-2:]!=(3, 3):
            raise ValueError("Direction Cosine Matrix must have shape (3, 3) or (N, 3, 3), got {}.".format(array.shape))
        in_SO3 = np.isclose(np.linalg.det(array), 1.0)
        in_SO3 &= np.allclose(array@array.T, np.identity(3))
        if not in_SO3:
            raise ValueError("Given attitude is not in SO(3)")
        # Create the ndarray instance of type DCM. This will call the standard
        # ndarray constructor, but return an object of type DCM.
        obj = super(DCM, subtype).__new__(subtype, array.shape, float, array)
        obj.A = array
        return obj

    # def __array_finalize__(self, obj):
    #     if obj is None: return
    #     self.A = getattr(obj, 'A', None)

    @property
    def I(self) -> np.ndarray:
        return self.A.T

    @property
    def inv(self) -> np.ndarray:
        return self.A.T

    @property
    def det(self) -> float:
        return np.linalg.det(self.A)

    @property
    def fro(self) -> float:
        return np.linalg.norm(self.A, 'fro')

    @property
    def frobenius(self) -> float:
        return np.linalg.norm(self.A, 'fro')

    @property
    def log(self) -> np.ndarray:
        """Logarithm of DCM

        Returns
        -------
        log : numpy.ndarray
            Logarithm of DCM

        """
        S = 0.5*(self.A-self.A.T)       # Skew-symmetric matrix
        y = np.array([S[2, 1], -S[2, 0], S[1, 0]])  # Axis
        if np.allclose(np.zeros(3), y):
            return np.zeros(3)
        y2 = np.linalg.norm(y)
        return np.arcsin(y2)*y/y2

    @property
    def adjugate(self) -> np.ndarray:
        return np.linalg.det(self.A)*self.A.T

    @property
    def adj(self) -> np.ndarray:
        return np.linalg.det(self.A)*self.A.T

    def to_axisangle(self) -> Tuple[np.ndarray, float]:
        """DCM from axis-angle representation

        Use Rodrigue's formula to obtain the axis-angle representation from the
        DCM.

        Parameters
        ----------
        axis : numpy.ndarray
            Axis of rotation.
        angle : float
            Angle of rotation.

        """
        angle = np.arccos((self.A.trace()-1)/2)
        axis = np.zeros(3)
        if angle!=0:
            axis = np.array([self.A[2, 1]-self.A[1, 2], self.A[0, 2]-self.A[2, 0], self.A[1, 0]-self.A[0, 1]])/(2*np.sin(angle))
        return axis, angle

    def to_axang(self) -> Tuple[np.ndarray, float]:
        """Synonym of method to_axisangle()
        """
        return self.to_axisangle()

    def from_axisangle(self, axis: np.ndarray, angle: float) -> np.ndarray:
        """DCM from axis-angle representation

        Use Rodrigue's formula to obtain the DCM from the axis-angle
        representation.

        Parameters
        ----------
        axis : numpy.ndarray
            Axis of rotation.
        angle : float
            Angle of rotation.

        """
        K = skew(axis)
        return np.identity(3) + np.sin(angle)*K + (1-np.cos(angle))*K@K

    def from_quaternion(self, q: np.ndarray) -> np.ndarray:
        """DCM from given quaternion

        The quaternion :math:`\\mathbf{q}` has the form :math:`\\mathbf{q} = (q_w, q_x, q_y, q_z)`,
        where :math:`\\mathbf{q}_v = (q_x, q_y, q_z)` is the vector part, and
        :math:`q_w` is the scalar part.

        The resulting matrix :math:`\\mathbf{R}` has the form:

        .. math::

            \\mathbf{R}(\\mathbf{q}) =
            \\begin{bmatrix}
            1 - 2(q_y^2 + q_z^2) & 2(q_xq_y - q_wq_z) & 2(q_xq_z + q_wq_y) \\\\
            2(q_xq_y + q_wq_z) & 1 - 2(q_x^2 + q_z^2) & 2(q_yq_z - q_wq_x) \\\\
            2(q_xq_z - q_wq_y) & 2(q_wq_x + q_yq_z) & 1 - 2(q_x^2 + q_y^2)
            \\end{bmatrix}

        The identity Quaternion :math:`\\mathbf{q} = (1, 0, 0, 0)`, produces a
        a :math:`3 \\times 3` Identity matrix :math:`\\mathbf{I}_3`.

        Returns
        -------
        R : numpy.ndarray
            3-by-3 direction cosine matrix R

        """
        if q is None:
            return np.identity(3)
        if q.shape[-1]!=4 or q.ndim>2:
            raise ValueError("Quaternion must be of the form (4,) or (N, 4)")
        if q.ndim>1:
            q /= np.linalg.norm(q, axis=1)[:, None]     # Normalize
            R = np.zeros((q.shape[0], 3, 3))
            R[:, 0, 0] = 1.0 - 2.0*(q[:, 2]**2 + q[:, 3]**2)
            R[:, 1, 0] = 2.0*(q[:, 1]*q[:, 2]+q[:, 0]*q[:, 3])
            R[:, 2, 0] = 2.0*(q[:, 1]*q[:, 3]-q[:, 0]*q[:, 2])
            R[:, 0, 1] = 2.0*(q[:, 1]*q[:, 2]-q[:, 0]*q[:, 3])
            R[:, 1, 1] = 1.0 - 2.0*(q[:, 1]**2 + q[:, 3]**2)
            R[:, 2, 1] = 2.0*(q[:, 0]*q[:, 1]+q[:, 2]*q[:, 3])
            R[:, 0, 2] = 2.0*(q[:, 1]*q[:, 3]+q[:, 0]*q[:, 2])
            R[:, 1, 2] = 2.0*(q[:, 2]*q[:, 3]-q[:, 0]*q[:, 1])
            R[:, 2, 2] = 1.0 - 2.0*(q[:, 1]**2 + q[:, 2]**2)
            return R
        q /= np.linalg.norm(q)
        return np.array([
            [1.0-2.0*(q[2]**2+q[3]**2), 2.0*(q[1]*q[2]-q[0]*q[3]), 2.0*(q[1]*q[3]+q[0]*q[2])],
            [2.0*(q[1]*q[2]+q[0]*q[3]), 1.0-2.0*(q[1]**2+q[3]**2), 2.0*(q[2]*q[3]-q[0]*q[1])],
            [2.0*(q[1]*q[3]-q[0]*q[2]), 2.0*(q[0]*q[1]+q[2]*q[3]), 1.0-2.0*(q[1]**2+q[2]**2)]])

    def from_q(self, q: np.ndarray) -> np.ndarray:
        """Synonym of method `from_quaternion`
        """
        return self.from_quaternion(self, q)

    def to_quaternion(self, method: str = 'shepperd', **kw) -> np.ndarray:
        """
        Set quaternion from given Direction Cosine Matrix.

        Parameters
        ----------
        dcm : NumPy array
            3-by-3 Direction Cosine Matrix.
        method : NumPy array

        """
        q = np.array([1., 0., 0., 0.])
        if method.lower()=='hughes':
            q = hughes(self.A)
        if method.lower()=='chiaverini':
            q = chiaverini(self.A)
        if method.lower()=='shepperd':
            q = shepperd(self.A)
        if method.lower()=='itzhack':
            q = itzhack(self.A, version=kw.get('version', 3))
        if method.lower()=='sarabandi':
            q = sarabandi(self.A, eta=kw.get('threshold', 0.0))
        return q/np.linalg.norm(q)

    def to_q(self, method: str = 'shepperd', **kw) -> np.ndarray:
        """Synonym of method `to_quaternion`
        """
        return self.to_quaternion(method=method, **kw)


