# -*- coding: utf-8 -*-
"""
Quaternion
==========

References
----------
.. [Sola] Solà, Joan. Quaternion kinematics for the error-state Kalman Filter.
    October 12, 2017. (http://www.iri.upc.edu/people/jsola/JoanSola/objectes/notes/kinematics.pdf)
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
.. [Itzhack] Y. Bar-Itzhack. New method for Extracting the Quaternion from a
    Rotation Matrix. Journal of Guidance, Control, and Dynamics,
    23(6):1085–1087, 2000. (https://arc.aiaa.org/doi/abs/10.2514/2.4654)
.. [Hughes] P. Hughes. Spacecraft Attitude Dynamics. 1986.
.. [Markley] F. Landis Markley. Unit Quaternion from Rotation Matrix. Journal
    of Guidance, Control, and Dynamics. Vol 31, Num 2. 2008.
    (https://arc.aiaa.org/doi/pdf/10.2514/1.31730)
.. [Curtis] H. D. Curtis. Orbital Mechanics for Engineering Students.
    (https://en.wikipedia.org/wiki/Orbital_Mechanics_for_Engineering_Students)
.. [Grosskatthoefer] K. Grosskatthoefer. Introduction into quaternions from
    spacecraft attitude representation. TU Berlin. 2012.
    (http://www.tu-berlin.de/fileadmin/fg169/miscellaneous/Quaternions.pdf)
.. [Shepperd] S.W. Shepperd. "Quaternion from rotation matrix." Journal of
    Guidance and Control, Vol. 1, No. 3, pp. 223-224, 1978.
    (https://arc.aiaa.org/doi/10.2514/3.55767b)
.. [Chiaverini] S. Chiaverini & B. Siciliano. The Unit Quaternion: A Useful
    Tool for Inverse Kinematics of Robot Manipulators. Systems Analysis
    Modelling Simulation. May 1999.
    (https://www.researchgate.net/publication/262391661)

"""

import numpy as np

def chiaverini(dcm):
    """
    Obtain a Quaternion from a Direction Cosine Matrix using Chiaverini's
    algebraic method [Chiaverini]_.

    Parameters
    ----------
    dcm : array
        3-by-3 Direction Cosine Matrix.

    Returns
    -------
    q : array
        Quaternion.
    """
    tr = dcm.trace()
    n = 0.5*np.sqrt(1.0 + tr)
    e = np.array([0.5*np.sign(dcm[2, 1]-dcm[1, 2])*np.sqrt(dcm[0, 0]-dcm[1, 1]-dcm[2, 2]+1.0),
                  0.5*np.sign(dcm[0, 2]-dcm[2, 0])*np.sqrt(dcm[1, 1]-dcm[2, 2]-dcm[0, 0]+1.0),
                  0.5*np.sign(dcm[1, 0]-dcm[0, 1])*np.sqrt(dcm[2, 2]-dcm[0, 0]-dcm[1, 1]+1.0)])
    return np.concatenate(([n], e))

def hughes(dcm):
    """
    Obtain a Quaternion from a Direction Cosine Matrix using Hughe's method [Hughes]_.

    Parameters
    ----------
    dcm : array
        3-by-3 Direction Cosine Matrix.

    Returns
    -------
    q : array
        Quaternion.
    """
    tr = dcm.trace()
    if np.isclose(tr, 3.0):
        # No rotation. DCM is identity.
        return np.array([1., 0., 0., 0.])
    n = 0.5*np.sqrt(1.0 + tr)
    if np.isclose(n, 0):    # trace = -1: q_w = 0 (Pure Quaternion)
        e = np.sqrt((1.0+np.diag(R))/2.0)
    else:
        e = 0.25*np.array([dcm[1, 2]-dcm[2, 1], dcm[2, 0]-dcm[0, 2], dcm[0, 1]-dcm[1, 0]])/n
    return np.concatenate(([n], e))

def sarabandi(dcm, eta=0.0):
    """
    Obtain a Quaternion from a Direction Cosine Matrix using Sarabandi's method
    [Sarabandi]_.

    Parameters
    ----------
    dcm : array
        3-by-3 Direction Cosine Matrix.
    eta : float
        Threshold

    Returns
    -------
    q : array
        Quaternion.
    """
    # Get elements of R
    r11, r12, r13 = dcm[0, 0], dcm[0, 1], dcm[0, 2]
    r21, r22, r23 = dcm[1, 0], dcm[1, 1], dcm[1, 2]
    r31, r32, r33 = dcm[2, 0], dcm[2, 1], dcm[2, 2]
    # Compute qw
    dw = r11+r22+r33
    if dw > eta:
        qw = 0.5*np.sqrt(1.0+dw)
    else:
        nom = (r32-r23)**2+(r13-r31)**2+(r21-r12)**2
        denom = 3.0-dw
        qw = 0.5*np.sqrt(nom/denom)
    # Compute qx
    dx = r11-r22-r33
    if dx > eta:
        qx = 0.5*np.sqrt(1.0+dx)
    else:
        nom = (r32-r23)**2+(r12+r21)**2+(r31+r13)**2
        denom = 3.0-dx
        qx = 0.5*np.sqrt(nom/denom)
    # Compute qy
    dy = -r11+r22-r33
    if dy > eta:
        qy = 0.5*np.sqrt(1.0+dy)
    else:
        nom = (r13-r31)**2+(r12+r21)**2+(r23+r32)**2
        denom = 3.0-dy
        qy = 0.5*np.sqrt(nom/denom)
    # Compute qz
    dz = -r11-r22+r33
    if dz > eta:
        qz = 0.5*np.sqrt(1.0+dz)
    else:
        nom = (r21-r12)**2+(r31+r13)**2+(r23+r32)**2
        denom = 3.0-dz
        qz = 0.5*np.sqrt(nom/denom)
    return np.array([qw, qx, qy, qz])

def itzhack(dcm, version=3):
    """
    Obtain a Quaternion from a Direction Cosine Matrix using Bar-Itzhack's
    method [Itzhack]_.

    Versions 1 and 2 are used twith orthogonal matrices (which all rotation
    matrices should be.)

    Parameters
    ----------
    dcm : array
        3-by-3 Direction Cosine Matrix.
    version : int
        Version used to compute the Quaternion. Defaults to 3.

    Returns
    -------
    q : array
        Quaternion.
    """
    is_orthogonal = np.isclose(np.linalg.det(dcm)**2, 1.0) and np.isclose(dcm@dcm.T, np.eye(3)).all()
    if is_orthogonal:
        if version == 1:
            K2 = np.array([
                [dcm[0, 0]-dcm[1, 1], dcm[1, 0]+dcm[0, 1], dcm[2, 0], -dcm[2, 1]],
                [dcm[1, 0]+dcm[0, 1], dcm[1, 1]-dcm[0, 0], dcm[2, 1], dcm[2, 0]],
                [dcm[2, 0], dcm[2, 1], -dcm[0, 0]-dcm[1, 1], dcm[0, 1]-dcm[1, 0]],
                [-dcm[2, 1], dcm[2, 0], dcm[0, 1]-dcm[1, 0], dcm[0, 0]+dcm[1, 1]]])/2.0
            eigval, eigvec = np.linalg.eig(K2)
            q = eigvec[:, np.where(np.isclose(eigval, 1.0))[0]].flatten().real
            return np.roll(q, 1)
        if version == 2:
            K3 = np.array([
                [dcm[0, 0]-dcm[1, 1]-dcm[2, 2], dcm[1, 0]+dcm[0, 1], dcm[2, 0]+dcm[0, 2], dcm[1, 2]-dcm[2, 1]],
                [dcm[1, 0]+dcm[0, 1], dcm[1, 1]-dcm[0, 0]-dcm[2, 2], dcm[2, 1]+dcm[1, 2], dcm[2, 0]-dcm[0, 2]],
                [dcm[2, 0]+dcm[0, 2], dcm[2, 1]+dcm[1, 2], dcm[2, 2]-dcm[0, 0]-dcm[1, 1], dcm[0, 1]-dcm[1, 0]],
                [dcm[1, 2]-dcm[2, 1], dcm[2, 0]-dcm[0, 2], dcm[0, 1]-dcm[1, 0], dcm[0, 0]+dcm[1, 1]+dcm[2, 2]]])/3.0
            eigval, eigvec = np.linalg.eig(K3)
            q = eigvec[:, np.where(np.isclose(eigval, 1.0))[0]].flatten().real
            return np.roll(q, 1)
    # Non-orthogonal DCM. Use version 3
    K3 = np.array([
        [dcm[0, 0]-dcm[1, 1]-dcm[2, 2], dcm[1, 0]+dcm[0, 1], dcm[2, 0]+dcm[0, 2], dcm[1, 2]-dcm[2, 1]],
        [dcm[1, 0]+dcm[0, 1], dcm[1, 1]-dcm[0, 0]-dcm[2, 2], dcm[2, 1]+dcm[1, 2], dcm[2, 0]-dcm[0, 2]],
        [dcm[2, 0]+dcm[0, 2], dcm[2, 1]+dcm[1, 2], dcm[2, 2]-dcm[0, 0]-dcm[1, 1], dcm[0, 1]-dcm[1, 0]],
        [dcm[1, 2]-dcm[2, 1], dcm[2, 0]-dcm[0, 2], dcm[0, 1]-dcm[1, 0], dcm[0, 0]+dcm[1, 1]+dcm[2, 2]]])/3.0
    eigval, eigvec = np.linalg.eig(K3)
    q = eigvec[:, eigval.argmax()]
    return np.roll(q, 1)

def shepperd(dcm):
    """
    Obtain a Quaternion from a Direction Cosine Matrix using Shepperd's
    method [Shepperd]_.

    Parameters
    ----------
    dcm : array
        3-by-3 Direction Cosine Matrix.

    Returns
    -------
    q : array
        Quaternion.
    """
    b = np.concatenate(([dcm.trace()], np.diag(dcm)))
    i = b.argmax()
    if i == 0:
        q = np.array([1.0+dcm[0, 0]+dcm[1, 1]+dcm[2, 2], dcm[1, 2]-dcm[2, 1], dcm[2, 0]-dcm[0, 2], dcm[0, 1]-dcm[1, 0]])
    elif i == 1:
        q = np.array([dcm[1, 2]-dcm[2, 1], 1.0+dcm[0, 0]-dcm[1, 1]-dcm[2, 2], dcm[1, 0]+dcm[0, 1], dcm[2, 0]+dcm[0, 2]])
    elif i == 2:
        q = np.array([dcm[2, 0]-dcm[0, 2], dcm[1, 0]+dcm[0, 1], 1.0-dcm[0, 0]+dcm[1, 1]-dcm[2, 2], dcm[2, 1]+dcm[1, 2]])
    else:
        q = np.array([dcm[0, 1]-dcm[1, 0], dcm[2, 0]+dcm[0, 2], dcm[2, 1]+dcm[1, 2], 1.0-dcm[0, 0]-dcm[1, 1]+dcm[2, 2]])
    q /= 2.0*np.sqrt(q[i])
    return q

class Quaternion:
    """
    Quaternion
    ==========

    Class to represent a quaternion. It can be built with 3- or 4-dimensional
    vectors. The quaternion objects are always normalized to represent
    rotations in 3D space.

    Attributes
    ----------
    q : ndarray
        Array with the 4 elements of quaternion of the form: q = [w, x, y, z]
    w : float
        Scalar part of the quaternion.
    v : ndarray
        Vector part of the quaternion.
    x : float
        First element of the vector part of the quaternion.
    y : float
        Second element of the vector part of the quaternion.
    z : float
        Third element of the vector part of the quaternion.

    Examples
    --------
    >>> from ahrs.common import Quaternion
    >>> q = Quaternion([1., 2., 3., 4.])
    >>> str(q)
    (0.1826 +0.3651i +0.5477j +0.7303k)
    >>> x = [1., 2., 3.]
    >>> q.rot(x)
    [1.8 2.  2.6]
    >>> R = q.to_DCM()
    >>> R@x
    [1.8 2.  2.6]

    A call to method product() will return an array of a multiplied vector.

    >>> q1 = Quaternion([1., 2., 3., 4.])
    >>> q2 = Quaternion([5., 4., 3., 2.])
    >>> q1.product(q2)
    [-0.49690399  0.1987616   0.74535599  0.3975232 ]

    But multiplication operators are overriden to return quaternions

    >>> str(q1*q2)
    '(-0.4969 +0.1988i +0.7454j +0.3975k)'
    >>> str(q1@q2)
    '(-0.4969 +0.1988i +0.7454j +0.3975k)'

    Basic operators are also overriden and return quaternions

    >>> str(q1+q2)
    '(0.4619 +0.4868i +0.5117j +0.5366k)'
    >>> str(q1-q2)
    '(-0.6976 -0.2511i +0.1954j +0.6420k)'

    Pure quaternions are built from 3-element arrays

    >>> q = Quaternion([1., 2., 3.])
    >>> str(q)
    '(0.0000 +0.2673i +0.5345j +0.8018k)'
    >>> q.is_pure()
    True

    And all basic properties can be fetched

    >>> q.conj()
    [ 0.         -0.26726124 -0.53452248 -0.80178373]
    >>> q.exponential()
    [0.54030231 0.22489258 0.44978516 0.67467774]
    >>> q.logarithm()
    [0.         0.41981298 0.83962595 1.25943893]
    >>> q.to_axang()
    (array([0.26726124, 0.53452248, 0.80178373]), 3.141592653589793)
    >>> q.to_angles()
    [ 1.24904577 -0.44291104  2.8198421 ]
    """
    q = np.array([1., 0., 0., 0.])
    def __init__(self, q=None, **kw):
        if "angles" in kw:
            self.from_angles(kw["angles"])
        else:
            if q is None:
                self.q = np.array([1., 0., 0., 0.])
            else:
                q = np.array(q)
                if q.ndim != 1 or q.shape[-1] not in [3, 4]:
                    raise ValueError("Expected `q` to have shape (4,) or (3,), got {}.".format(q.shape))
                self.q = np.concatenate(([0.0], q)) if q.shape[-1] == 3 else q
        self.normalize()
        self.w = self.q[0]
        self.v = self.q[1:]
        self.x, self.y, self.z = self.v

    def __str__(self):
        """
        Build 'printable' representation of quaternion

        Returns
        -------
        q : str
            Quaternion written as string.

        Examples
        --------
        >>> q = Quaternion([0.55747131, 0.12956903, 0.5736954 , 0.58592763])
        >>> str(q)
        '(0.5575 +0.1296i +0.5737j +0.5859k)'
        """
        return "({:-.4f} {:+.4f}i {:+.4f}j {:+.4f}k)".format(self.w, self.x, self.y, self.z)

    def __add__(self, p):
        """
        Add quaternions

        Returns
        -------
        q : Quaternion
            Normalized sum of quaternions

        Examples
        --------
        >>> q1 = Quaternion([0.55747131, 0.12956903, 0.5736954 , 0.58592763])
        >>> q2 = Quaternion([0.49753507, 0.50806522, 0.52711628, 0.4652709])
        >>> q3 = q1+q2
        >>> q3
        <ahrs.common.quaternion.Quaternion object at 0x000001F379003748>
        >>> str(q3)
        '(0.5386 +0.3255i +0.5620j +0.5367k)'
        """
        return Quaternion(self.q + p.q)

    def __sub__(self, p):
        """
        Difference of quaternions

        Returns
        -------
        q : Quaternion
            Normalized difference of quaternions

        Examples
        --------
        >>> q1 = Quaternion([0.55747131, 0.12956903, 0.5736954 , 0.58592763])
        >>> q2 = Quaternion([0.49753507, 0.50806522, 0.52711628, 0.4652709])
        >>> q3 = q1-q2
        >>> q3
        <ahrs.common.quaternion.Quaternion object at 0x000001F379003748>
        >>> str(q3)
        '(0.1482 -0.9358i +0.1152j +0.2983k)'
        """
        return Quaternion(self.q - p.q)

    def __mul__(self, q):
        """
        Product between quaternions

        Given two unit quaternions :math:`\\mathbf{p}=(p_w, p_x, p_y, p_z)` and
        :math:`\\mathbf{q} = (q_w, q_x, q_y, q_z)`, their product is obtained
        [Dantam]_ [MWQW]_ as:

        .. math::

            \\mathbf{pq} =
            \\begin{bmatrix}
            p_w q_w - p_x q_x - p_y q_y - p_z q_z \\\\
            p_x q_w + p_w q_x - p_z q_y + p_y q_z \\\\
            p_y q_w + p_z q_x + p_w q_y - p_x q_z \\\\
            p_z q_w - p_y q_x + p_x q_y + p_w q_z
            \\end{bmatrix}

        Parameters
        ----------
        q : array, Quaternion
            Second quaternion to multiply with.

        Returns
        -------
        out : Quaternion
            Product of quaternions.

        Examples
        --------
        >>> q1 = Quaternion([0.55747131, 0.12956903, 0.5736954 , 0.58592763])
        >>> q2 = Quaternion([0.49753507, 0.50806522, 0.52711628, 0.4652709])
        >>> q3 = q1*q2
        >>> q3
        <ahrs.common.quaternion.Quaternion object at 0x000001F379003748>
        >>> str(q3)
        '(-0.3635 +0.3896i +0.3419j +0.7740k)'
        """
        if not hasattr(q, 'q'):
            q = Quaternion(q)
        return Quaternion(self.product(q.q))

    def __matmul__(self, q):
        """
        Product between quaternions using @ operator

        Given two unit quaternions :math:`\\mathbf{p}=(p_w, p_x, p_y, p_z)` and
        :math:`\\mathbf{q} = (q_w, q_x, q_y, q_z)`, their product is obtained
        [Dantam]_ [MWQW]_ as:

        .. math::

            \\mathbf{pq} =
            \\begin{bmatrix}
            p_w q_w - p_x q_x - p_y q_y - p_z q_z \\\\
            p_x q_w + p_w q_x - p_z q_y + p_y q_z \\\\
            p_y q_w + p_z q_x + p_w q_y - p_x q_z \\\\
            p_z q_w - p_y q_x + p_x q_y + p_w q_z
            \\end{bmatrix}

        Parameters
        ----------
        q : array, Quaternion
            Second quaternion to multiply with.

        Returns
        -------
        out : Quaternion
            Product of quaternions.

        Examples
        --------
        >>> q1 = Quaternion([0.55747131, 0.12956903, 0.5736954 , 0.58592763])
        >>> q2 = Quaternion([0.49753507, 0.50806522, 0.52711628, 0.4652709])
        >>> q1@q2
        array([-0.36348726,  0.38962514,  0.34188103,  0.77407146])
        >>> q3 = q1@q2
        >>> q3
        <ahrs.common.quaternion.Quaternion object at 0x000001F379003748>
        >>> str(q3)
        '(-0.3635 +0.3896i +0.3419j +0.7740k)'
        """
        if not hasattr(q, 'q'):
            q = Quaternion(q)
        return Quaternion(self.product(q.q))

    def __pow__(self, a):
        """
        Returns a normalized `q` to the power of `a`

        A Quaternion power can be dfined as:

        .. math::

            \\mathbf{q}^a = e^{a \\log(\\mathbf{q})}

        Parameters
        ----------
        a : float
            Value to which to calculate quaternion power.

        Returns
        -------
        p : ndarray
            Quaternion `q`to the power of `a`
        """
        p = np.e**(a*self.logarithm())
        return p / np.linalg.norm(p)

    def is_pure(self):
        """
        Returns a bool value, where True if quaternion is pure.

        A pure quaternion has a scalar part equal to zero: :math:`\\mathbf{q} = (0 + xi + yj + zk)`

        .. math::

        \\mathrm{.is_versor}() = \\left\\{
        \\begin{array}{ll}
            \\mathrm{True} & \\: w = 0 \\\\
            \\mathrm{False} & \\: \\mathrm{otherwise}
        \\end{array}
        \\right.

        Returns
        -------
        out : bool
            Boolean equal to True if q_w is 0.
        """
        return self.w == 0.0

    def is_real(self):
        """
        Returns a bool value, where True if quaternion is real.

        A real quaternion has all elements of its vector part equal to zero:
        :math:`\\mathbf{q} = (w + 0i + 0j + 0k) = (q_w, 0)`

        .. math::

        \\mathrm{.is_versor}() = \\left\\{
        \\begin{array}{ll}
            \\mathrm{True} & \\: v = 0 \\\\
            \\mathrm{False} & \\: \\mathrm{otherwise}
        \\end{array}
        \\right.

        Returns
        -------
        out : bool
            Boolean equal to True if q_v is 0.
        """
        return not any(self.v)

    def is_versor(self):
        """
        Returns a bool value, where True if quaternion is a versor.

        A versor is a quaternion of norm equal to one:

        .. math::

        \\mathrm{.is_versor}() = \\left\\{
        \\begin{array}{ll}
            \\mathrm{True} & \\: \\|mathbf{q}\\| = 1 \\\\
            \\mathrm{False} & \\: \\mathrm{otherwise}
        \\end{array}
        \\right.

        Returns
        -------
        out : bool
            Boolean equal to True if q_v is 0.
        """
        return np.isclose(np.linalg.norm(self.q), 1.0)

    def is_identity(self):
        """
        Returns a bool value, where True if quaternion is identity quaternion.

        A quaternion is a quaternion if its scalar part is equal to 1, and the
        vector part is equal to 0:

        .. math::

        \\mathrm{.is_identity}() = \\left\\{
        \\begin{array}{ll}
            \\mathrm{True} & \\: \\mathbf{q}\\ = \\begin{bmatrix} 1 & 0 & 0 & 0 \\end{bmatrix} \\\\
            \\mathrm{False} & \\: \\mathrm{otherwise}
        \\end{array}
        \\right.

        Returns
        -------
        out : bool
            Boolean equal to True if q_v is 0.
        """
        return np.allclose(self.q, np.array([1.0, 0.0, 0.0, 0.0]))

    def normalize(self):
        """Normalize the quaternion
        """
        self.q /= np.linalg.norm(self.q)

    def conjugate(self):
        """
        Return the conjugate of a quaternion

        A quaternion, whose form is :math:`\\mathbf{q} = (q_w, q_x, q_y, q_z)`,
        has a conjugate of the form :math:`\\mathbf{q}^* = (q_w, -q_x, -q_y, -q_z)`.

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

    def conj(self):
        """Synonym to method conjugate()
        """
        return self.conjugate()

    def inverse(self):
        """Inverse Quaternion
        """
        if self.is_versor():
            return self.conjugate()
        return self.conjugate() / np.linalg.norm(self.q)

    def exponential(self):
        """Exponential of Quaternion
        """
        t = np.linalg.norm(self.v)
        u = self.v/t
        q_exp = np.concatenate(([np.cos(t)], u*np.sin(t)))
        if self.is_pure():
            return q_exp
        return np.e**self.w * q_exp

    def logarithm(self):
        """Logarithm of Quaternion
        """
        v_norm = np.linalg.norm(self.v)
        u = self.v / v_norm
        if self.is_pure():
            return np.concatenate(([0.0], u))
        t = np.arctan(v_norm/self.w)
        return np.concatenate((np.log(np.linalg.norm(self.q)), u*t))

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

        or with a Quaternion object...

        >>> q2 = Quaternion([0.49753507, 0.50806522, 0.52711628, 0.4652709 ])
        >>> q1.product(q2)
        array([-0.36348726,  0.38962514,  0.34188103,  0.77407146])

        or with a Quaternion itself to return a Quaternion

        >>> q3 = q1*q2
        >>> q3.__str__()
        (-0.3635 +0.3896i +0.3419j +0.7740k)

        It holds with the result after the cross and dot product definition

        >>> q3_w = q1.w*q2.w-np.dot(q1.v, q2.v)
        >>> q3_v = np.cross(q2.v, q1.v) + q2.w*q1.v + q1.w*q2.v
        >>> q3_w, q3_v
        (-0.36348726, array([0.38962514,  0.34188103,  0.77407146]))

        """
        if hasattr(q, 'q'):
            q = q.q.copy()
        q /= np.linalg.norm(q)
        if self.q[0] == 0.0 and q[0] == 0.0:
            return np.concatenate(([-np.dot(self.v, q[1:])], np.cross(self.v, q[1:])))
        return np.array([
            self.w*q[0] - self.x*q[1] - self.y*q[2] - self.z*q[3],
            self.w*q[1] + self.x*q[0] + self.y*q[3] - self.z*q[2],
            self.w*q[2] - self.x*q[3] + self.y*q[0] + self.z*q[1],
            self.w*q[3] + self.x*q[2] - self.y*q[1] + self.z*q[0]])

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
        a = np.array(a)
        if a.shape[0] != 3:
            raise ValueError("Expected `a` to have shape (3, N) or (3,), got {}.".format(a.shape))
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
        Return a Direction Cosine matrix :math:`\\mathbf{R} \\in SO(3)` from a
        given unit quaternion :math:`\\mathbf{q}`.

        The given unit quaternion :math:`\\mathbf{q}` must have the form
        :math:`\\mathbf{q} = (q_w, q_x, q_y, q_z)`, where :math:`\\mathbf{q}_v = (q_x, q_y, q_z)`
        is the vector part, and :math:`q_w` is the scalar part.

        The resulting matrix :math:`\\mathbf{R}` [WikiQuat1]_ [WikiQuat2]_ has
        the form:

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
        R : array
            3-by-3 direction cosine matrix R

        """
        return np.array([
            [1.0-2.0*(self.y**2+self.z**2), 2.0*(self.x*self.y-self.w*self.z), 2.0*(self.x*self.z+self.w*self.y)],
            [2.0*(self.x*self.y+self.w*self.z), 1.0-2.0*(self.x**2+self.z**2), 2.0*(self.y*self.z-self.w*self.x)],
            [2.0*(self.x*self.z-self.w*self.y), 2.0*(self.w*self.x+self.y*self.z), 1.0-2.0*(self.x**2+self.y**2)]])

    def from_DCM(self, dcm, method='shepperd', **kw):
        """
        Set quaternion from given Direction Cosine Matrix.

        Parameters
        ----------
        dcm : array
            3-by-3 Direction Cosine Matrix.

        """
        dcm = np.array(dcm)
        if dcm.shape != (3, 3):
            raise TypeError("Expected matrix of size (3, 3). Got {}".format(dcm.shape))
        q = np.array([1., 0., 0., 0.])
        if method.lower() == 'hughes':
            q = hughes(dcm)
        if method.lower() == 'chiaverini':
            q = chiaverini(dcm)
        if method.lower() == 'shepperd':
            q = shepperd(dcm)
        if method.lower() == 'itzhack':
            q = itzhack(dcm, version=kw.get('version', 3))
        if method.lower() == 'sarabandi':
            q = sarabandi(dcm, eta=kw.get('threshold', 0.0))
        if q is None:
            raise KeyError("Given method '{}' is not implemented.".format(method))
        self.q = q.copy()
        self.normalize()
        self.w, self.x, self.y, self.z = self.q

    def from_angles(self, angles):
        """
        Set quaternion from given Euler angles.

        Parameters
        ----------
        angles : array
            3 Euler angles in following order: roll -> pitch -> yaw.

        """
        angles = np.array(angles)
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

if __name__ == "__main__":
    # Test some Quaternion methods
    print("Testing methods to obtain Quaternions from DCM")
    q = Quaternion(np.random.random(4)*2.0-1.0)
    R = q.to_DCM()
    print("\n Original   = {}\n".format(q.q))
    q.from_DCM(R, method='hughes')
    print(" Hughes     = {}".format(q.q))
    q.from_DCM(R, method='chiaverini')
    print(" Chiaverini = {}".format(q.q))
    q.from_DCM(R, method='itzhack', version=1)
    print(" Itzhack(1) = {}".format(q.q))
    q.from_DCM(R, method='itzhack', version=2)
    print(" Itzhack(2) = {}".format(q.q))
    q.from_DCM(R, method='itzhack', version=3)
    print(" Itzhack(3) = {}".format(q.q))
    q.from_DCM(R, method='shepperd')
    print(" Shepperd   = {}".format(q.q))
    q.from_DCM(R, method='sarabandi')
    print(" Sarabandi  = {}".format(q.q))
