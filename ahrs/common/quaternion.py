# -*- coding: utf-8 -*-
"""
Quaternion
==========

References
----------
.. [Dantam] Dantam, N. (2014) Quaternion Computation. Institute for Robotics
    and Intelligent Machines. Georgia Tech. (http://www.neil.dantam.name/note/dantam-quaternion.pdf)
.. [Sarkka] Särkkä, S. (2007) Notes on Quaternions (https://users.aalto.fi/~ssarkka/pub/quat.pdf)
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
.. [Eberly] Eberly, D. (2010) Quaternion Algebra and Calculus. Geometric Tools.
    https://www.geometrictools.com/Documentation/Quaternions.pdf

"""

import numpy as np

class Quaternion:
    """
    Quaternion Class
    """
    def __init__(self, q=None):
        if q is None:
            self.q = np.array([1., 0., 0., 0.])
        else:
            q = np.array(q)
            if q.ndim != 1 or q.shape[-1] not in [3, 4]:
                raise ValueError("Expected `q` to have shape (4,) or (3,), got {}.".format(q.shape))
            self.q = np.concatenate(([0.0], q)) if q.shape[-1] == 3 else q
            self.q = self.normalize()
        self.w = self.q[0]
        self.v = self.q[1:]
        self.x, self.y, self.z = self.v

    def __str__(self):
        return "({:-.4f} {:+.4f}i {:+.4f}j {:+.4f}k)".format(self.w, self.x, self.y, self.z)

    def __pow__(self, y, *args):
        """
        Power of Quaternion
        """
        return np.e**(y*self.logarithm())

    def is_pure(self):
        return self.w == 0.0

    def is_versor(self):
        return np.isclose(np.linalg.norm(self.q), 1.0)

    def normalize(self):
        return self.q/np.linalg.norm(self.q)

    def conjugate(self):
        """
        Return the conjugate of a quaternion

        A quaternion, whose form is :math:`\\mathbf{q} = (q_w, q_x, q_y, q_z)`,
        has a conjugate of the form :math:`\\mathbf{q}^* = (q_w, -q_x, -q_y, -q_z)`.

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

    def exponential(self):
        """
        Eponential of the Quaternion
        """
        qv_norm = np.linalg.norm(self.v)
        scalar = np.cos(qv_norm)
        vector = self.v*np.sin(qv_norm)/qv_norm
        q_exp = np.concatenate(([scalar], vector))
        if self.is_pure():
            return q_exp
        return np.e**self.w * q_exp

    def logarithm(self):
        """
        Logarithm of Quaternion
        """
        qv_norm = np.linalg.norm(self.v)
        phi = np.arctan2(qv_norm, self.w)
        if self.is_pure():
            return np.concatenate(([0.0], self.v*phi/np.sin(phi)))
        return np.concatenate((np.log(np.linalg.norm(self.q)), self.v*phi/qv_norm))

    def product(self, q):
        """
        Product of two quaternions.

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
        self.q*q : array
            Product of quaternions

        Examples
        --------
        >>> q1 = Quaternion([0.55747131, 0.12956903, 0.5736954 , 0.58592763])

        Can multiply with a given quaternion in vector form...

        >>> q1.product([0.49753507, 0.50806522, 0.52711628, 0.4652709])
        array([-0.36348726,  0.38962514,  0.34188103,  0.77407146])

        or with a Quaternion object

        >>> q2 = Quaternion([0.49753507, 0.50806522, 0.52711628, 0.4652709 ])
        >>> q1.product(q2)
        array([-0.36348726,  0.38962514,  0.34188103,  0.77407146])

        It holds with the result after the cross and dot product definition

        >>> q3_w = q1.w*q2.w-np.dot(q1.v, q2.v)
        >>> q3_v = np.cross(q2.v, q1.v) + q2.w*q1.v + q1.w*q2.v
        >>> q3_w, q3_v
        (-0.36348726, array([0.38962514,  0.34188103,  0.77407146]))

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

    def rotate(self, a):
        """
        Rotate array :math:`\\mathbf{a}` through quaternion :math:`\\mathbf{q}`.

        Parameters
        ----------
        a : array
            3-by-N array to rotate in 3 dimensions, where N is the number of
            vectors to rotate.

        Returns
        -------
        a' : array
            3-by-N rotated array around current quaternion.

        Examples
        --------
        >>> q = Quaternion([-0.00085769, -0.0404217, 0.29184193, -0.47288709])
        >>> v = [0.25557699 0.74814091 0.71491841]
        >>> q.rotate(v)
        array([-0.22481078 -0.99218916 -0.31806219])
        >>> A = [[0.18029565, 0.14234782], [0.47473686, 0.38233722], [0.90000689, 0.06117298]]
        >>> q.rotate(A)
        array([[-0.10633285 -0.16347163]
               [-1.02790041 -0.23738541]
               [-0.00284403 -0.29514739]])

        """
        if type(a) != np.ndarray:
            a = np.array(a)
        if a.shape[0] != 3:
            raise ValueError("Expected `a` to have shape (3, N), got {}.".format(a.shape))
        return self.to_DCM()@a

    def to_axang(self):
        """
        Return the equivalent axis-angle rotation of the quaternion.

        Returns
        -------
        (axis, angle) : (ndarray, float)
            Axis and angle.

        Examples
        --------
        >>> q = ahrs.common.Quaternion([0.7071, 0.7071, 0.0, 0.0])
        >>> q.to_axang()
        (array([1., 0., 0.]), 1.5707963267948966)

        """
        denom = np.linalg.norm(self.v)
        angle = 2.0*np.arctan2(denom, self.w)
        axis = np.array([0.0, 0.0, 0.0]) if angle == 0.0 else self.v/denom
        return axis, angle

    def to_angles(self):
        """
        Return corresponding Euler angles of quaternion.

        Returns
        -------
        angles : array
            Euler angles of quaternion.

        """
        phi = np.arctan2(2.0*(self.w*self.x+self.y*self.z), 1.0-2.0*(self.x**2+self.y**2))
        theta = np.arcsin(2.0*(self.w*self.y-self.z*self.x))
        psi = np.arctan2(2.0*(self.w*self.z+self.x*self.y), 1.0-2.0*(self.y**2+self.z**2))
        return np.array([phi, theta, psi])

    def to_DCM(self):
        """
        Return a rotation matrix :math:`\\mathbf{R} \\in SO(3)` from a given unit
        quaternion :math:`\\mathbf{q}`.

        The given unit quaternion :math:`\\mathbf{q}` must have the form
        :math:`\\mathbf{q} = (q_w, q_x, q_y, q_z)`, where :math:`\\mathbf{q}_v = (q_x, q_y, q_z)`
        is the vector part, and :math:`q_w` is the scalar part.

        The resulting rotation matrix :math:`\\mathbf{R}` has the form  [W1]_ [W2]_:

        .. math::

            \\mathbf{R}(\\mathbf{q}) =
            \\begin{bmatrix}
            1 - 2(q_y^2 + q_z^2) & 2(q_xq_y - q_wq_z) & 2(q_xq_z + q_wq_y) \\\\
            2(q_xq_y + q_wq_z) & 1 - 2(q_x^2 + q_z^2) & 2(q_yq_z - q_wq_x) \\\\
            2(q_xq_z - q_wq_y) & 2(q_wq_x + q_yq_z) & 1 - 2(q_x^2 + q_y^2)
            \\end{bmatrix}

        The default value is the unit Quaternion :math:`\\mathbf{q} = (1, 0, 0, 0)`,
        which produces a :math:`3 \\times 3` Identity matrix :math:`\\mathbf{I}_3`.

        Returns
        -------
        R : array
            3-by-3 rotation matrix R

        """
        return np.array([
            [1.0-2.0*(self.y**2+self.z**2), 2.0*(self.x*self.y-self.w*self.z), 2.0*(self.x*self.z+self.w*self.y)],
            [2.0*(self.x*self.y+self.w*self.z), 1.0-2.0*(self.x**2+self.z**2), 2.0*(self.y*self.z-self.w*self.x)],
            [2.0*(self.x*self.z-self.w*self.y), 2.0*(self.w*self.x+self.y*self.z), 1.0-2.0*(self.x**2+self.y**2)]])

    def from_angles(self, angles):
        """
        Create a quaternion from given Euler angles.
        """
        if angles.ndim != 1 or angles.shape[0] != 3:
            raise ValueError("Expected `angles` to have shape (3,), got {}.".format(angles.shape))
        yaw, pitch, roll = angles
        cy = np.cos(0.5*yaw)
        sy = np.sin(0.5*yaw)
        cp = np.cos(0.5*pitch)
        sp = np.sin(0.5*pitch)
        cr = np.cos(0.5*roll)
        sr = np.sin(0.5*roll)
        self.q[0] = cy*cp*cr + sy*sp*sr
        self.q[1] = cy*cp*sr - sy*sp*cr
        self.q[2] = sy*cp*sr + cy*sp*cr
        self.q[3] = sy*cp*cr - cy*sp*sr
        self.normalize()
        self.w, self.x, self.y, self.z = self.q

    def derivative(self, w):
        """
        Quaternion derivative from angular velocity.

        Parameters
        ----------
        w : array
            Angular velocity, in rad/s, about X-, Y- and Z-angle.

        Returns
        -------
        dq/dt : array
            Derivative of quaternion.

        """
        if w.ndim != 1 or w.shape[0] != 3:
            raise ValueError("Expected `w` to have shape (3,), got {}.".format(w.shape))
        w = np.concatenate(([0.0], w))
        F = 0.5*np.array([
            [0.0, -w[0], -w[1], -w[2]],
            [w[0], 0.0, w[2], -w[1]],
            [w[1], -w[2], 0.0, w[0]],
            [w[2], w[1], -w[0], 0.0]])
        return F@self.q
