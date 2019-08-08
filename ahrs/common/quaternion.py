# -*- coding: utf-8 -*-
"""
Quaternion
==========

References
----------
.. [Dantam] Dantam, N. (2014) Quaternion Computation. Institute for Robotics
    and Intelligent Machines. Georgia Tech.
    (http://www.neil.dantam.name/note/dantam-quaternion.pdf)
.. [WikiQuat1] https://en.wikipedia.org/wiki/Quaternion
.. [WikiQuat2] https://en.wikipedia.org/wiki/Quaternions_and_spatial_rotation
.. [WikiQuat3] https://en.wikipedia.org/wiki/Versor
.. [WikiQuat4] https://en.wikipedia.org/wiki/Conversion_between_quaternions_and_Euler_angles
.. [WikiDCM] https://en.wikipedia.org/wiki/Rotation_matrix
.. [MWQM] Mathworks: Quaternion Multiplication.
    https://www.mathworks.com/help/aeroblks/quaternionmultiplication.html
.. [MWorks1] https://www.mathworks.com/help/robotics/ref/axang2quat.html
.. [Sarabandi] Sarabandi, S. et al. (2018) Accurate Computation of Quaternions
    from Rotation Matrices.
    (http://www.iri.upc.edu/files/scidoc/2068-Accurate-Computation-of-Quaternions-from-Rotation-Matrices.pdf)

"""

import numpy as np

class Quaternion:
    def __init__(self, q=None, *args, **kwargs):
        if q is None:
            q = [1., 0., 0., 0.]
        q = np.array(q)
        if q.ndim != 1 or q.shape[-1] not in [3, 4]:
            raise ValueError("Expected `q` to have shape (4,) or (3,), got {}.".format(q.shape))
        self.q = np.concatenate(([0.0], q)) if q.shape[-1] == 3 else q
        self.q /= np.linalg.norm(self.q)
        self.w = self.q[0]
        self.v = self.q[1:]
        self.x, self.y, self.z = self.v

    def is_pure(self):
        return self.w == 0.0

    def is_versor(self):
        return np.isclose(np.linalg.norm(self.q), 1.0)

    def __str__(self):
        return "({:-.4f} {:+.4f}i {:+.4f}j {:+.4f}k)".format(self.w, self.x, self.y, self.z)

    def conjugate(self):
        """
        Return the conjugate of a quaternion

        A quaternion, whose form is :math:`\\mathbf{q} = (q_w, q_x, q_y, q_z)`,
        has a conjugate of the form :math:`\\mathbf{q}^* = (q_w, -q_x, -q_y, -q_z)`.

        .. math::

            \\|\\mathbf{q}\\| = \\sqrt{q_w^2+q_x^2+q_y^2+q_z^2} = 1.0

        Parameters
        ----------
        q : array
            Unit quaternion or 2D array of Quaternions.

        Returns
        -------
        q* : array
            Conjugated quaternion or 2D array of conjugated Quaternions.

        Examples
        --------
        >>> q = Quaternion([0.603297, 0.749259, 0.176548, 0.20850])
        >>> q.conjugate()
        array([0.603297, -0.749259, -0.176548, -0.20850 ])

        """
        return self.q*np.array([1.0, -1.0, -1.0, -1.0])

    def product(self, q):
        """
        Product of two unit quaternions.

        Given two unit quaternions :math:`\\mathbf{p}=(p_w, \\mathbf{p}_v)` and
        :math:`\\mathbf{q} = (q_w, \\mathbf{q}_v)`, their product is defined [Dantam]_ [MWQW]_
        as:

        .. math::
    
            \\begin{eqnarray}
            \\mathbf{pq} & = & \\big( (q_w p_w - \\mathbf{q}_v \\cdot \\mathbf{p}_v) \\; ,
            \\; \\mathbf{q}_v \\times \\mathbf{p}_v + q_w \\mathbf{p}_v + p_w \\mathbf{q}_v \\big) \\\\
            & = &
            \\begin{bmatrix}
            p_w & -\\mathbf{p}_v^T \\\\ \\mathbf{p}_v & p_w \\mathbf{I}_3 + \\lfloor \\mathbf{p}_v \\rfloor
            \\end{bmatrix}
            \\begin{bmatrix} q_w \\\\ \\mathbf{q}_v \\end{bmatrix}
            \\\\
            & = &
            \\begin{bmatrix}
            p_w & -p_x & -p_y & -p_z \\\\
            p_x &  p_w & -p_z &  p_y \\\\
            p_y &  p_z &  p_w & -p_x \\\\
            p_z & -p_y &  p_x &  p_w
            \\end{bmatrix}
            \\begin{bmatrix} q_w \\\\ q_x \\\\ q_y \\\\ q_z \\end{bmatrix}
            \\\\
            & = &
            \\begin{bmatrix}
            p_w q_w - p_x q_x - p_y q_y - p_z q_z \\\\
            p_x q_w + p_w q_x - p_z q_y + p_y q_z \\\\
            p_y q_w + p_z q_x + p_w q_y - p_x q_z \\\\
            p_z q_w - p_y q_x + p_x q_y + p_w q_z
            \\end{bmatrix}
            \\end{eqnarray}

        Parameters
        ----------
        q : array, Quaternion
            Quaternion to multiply with

        Returns
        -------
        q'*q : array
            Product of quaternions

        Examples
        --------
        >>> q1 = Quaternion([0.55747131, 0.12956903, 0.5736954 , 0.58592763])

        Can multiply with a given quaternion in vector form...

        >>> q1.product([0.49753507, 0.50806522, 0.52711628, 0.4652709])
        array([-0.36348726,  0.38962514,  0.34188103,  0.77407146])
        >>> q2 = Quaternion([0.49753507, 0.50806522, 0.52711628, 0.4652709 ])

        or with a Quaternion object

        >>> q1.product(q2)
        array([-0.36348726,  0.38962514,  0.34188103,  0.77407146])

        """
        if type(q) is Quaternion:
            q = q.q.copy()
        q /= np.linalg.norm(q)
        return np.array([
            self.w*q[0] - self.x*q[1] - self.y*q[2] - self.z*q[3],
            self.w*q[1] + self.x*q[0] - self.y*q[3] + self.z*q[2],
            self.w*q[2] + self.x*q[3] + self.y*q[0] - self.z*q[1],
            self.w*q[3] - self.x*q[2] + self.y*q[1] + self.z*q[0]])

    def mult_L(self):
        """
        Matrix form of a left-sided quaternion multiplication Q.

        Returns
        -------
        Q : array
            Matrix form of the left side quaternion multiplication.

        """
        return np.array([
            [self.w, -self.x, -self.y, -self.z],
            [self.x,  self.w, -self.z,  self.y],
            [self.y,  self.z,  self.w, -self.x],
            [self.z, -self.y,  self.x,  self.w]])

    def mult_R(self):
        """
        Matrix form of a right-sided quaternion multiplication Q.

        Returns
        -------
        Q : array
            Matrix form of the right side quaternion multiplication.

        """
        return np.array([
            [self.w, -self.x, -self.y, -self.z],
            [self.x,  self.w,  self.z, -self.y],
            [self.y, -self.z,  self.w,  self.x],
            [self.z,  self.y, -self.x,  self.w]])

    def rotate(self, v):
        """
        Rotate vector :math:`\\mathbf{v}` through quaternion :math:`\\mathbf{q}`.

        It should be equal to calling `q.to_DCM()@v`.

        Parameters
        ----------
        v : array
            Vector to rotate in 3 dimensions.

        Returns
        -------
        v' : array
            Rotated vector around current quaternion.

        """
        return np.array([
            -2.0*v[0]*(self.y**2 + self.z**2 - 0.5) + 2.0*v[1]*(self.w*self.z + self.x*self.y)       - 2.0*v[2]*(self.w*self.y - self.x*self.z),
            -2.0*v[0]*(self.w*self.z - self.x*self.y)       - 2.0*v[1]*(self.x**2 + self.z**2 - 0.5) + 2.0*v[2]*(self.w*self.x + self.y*self.z),
             2.0*v[0]*(self.w*self.y + self.x*self.z)       - 2.0*v[1]*(self.w*self.x - self.y*self.z)       - 2.0*v[2]*(self.x**2 + self.y**2 - 0.5)])

    def to_axang(self):
        axis = np.asarray(self.v)
        denom = np.linalg.norm(axis)
        angle = 2.0*np.arctan2(denom, self.w)
        axis = np.array([0.0, 0.0, 0.0]) if angle == 0.0 else axis/denom
        return axis, angle

    def to_DCM(self):
        q = self.q.copy()
        return np.array([
            [1.0-2.0*(q[2]**2+q[3]**2), 2.0*(q[1]*q[2]-q[0]*q[3]), 2.0*(q[1]*q[3]+q[0]*q[2])],
            [2.0*(q[1]*q[2]+q[0]*q[3]), 1.0-2.0*(q[1]**2+q[3]**2), 2.0*(q[2]*q[3]-q[0]*q[1])],
            [2.0*(q[1]*q[3]-q[0]*q[2]), 2.0*(q[0]*q[1]+q[2]*q[3]), 1.0-2.0*(q[1]**2+q[2]**2)]])

