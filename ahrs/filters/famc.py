# -*- coding: utf-8 -*-
"""

Fast Accelerometer-Magnetometer Combination
===========================================

The most typical low-cost sensor system is the Accelerometer-Magnetometer
combination (AMC). It integrates the local gravity and the Earth's magnetic
field together, forming a full-attitude estimation system.

The most famous attitude determination formulation is `Wahba's Problem
<https://en.wikipedia.org/wiki/Wahba%27s_problem>`_, which sprung several
solutions using Euler Angles, Direction Cosine Matrices and Quaternions.

The matrix operations in these solutions are the main focus of attention in
this method. The operations are analytically simplified, where the accuracy is
maintained, while the time consumption is reduced, yielding the **Fast
Accelerometer-Magnetometer Combination (FAMC)**, whose main contributions are:

- Analytic eigenvalue results are given for the dynamic magnetometer reference
  vector.
- Solution as quaternion representation from a simplification of `Davenport's
  q-method <./davenport.html>`_.
- Advantageous on time consumption, compared with existing solutions to Wahba's
  problem.

The AMC relates a Direction Cosine Matrix, :math:`\\mathbf{C}`, such that:

.. math::
    \\begin{array}{c}
    ^b\\mathbf{a} = \\mathbf{C}\,^r\\mathbf{a} \\\\ \\\\
    ^b\\mathbf{m} = \\mathbf{C}\,^r\\mathbf{m}
    \\end{array}

where :math:`^b\\mathbf{a}=\\begin{bmatrix}a_x&a_y&a_z\\end{bmatrix}^T` and
:math:`^b\\mathbf{m}=\\begin{bmatrix}m_x&m_y&m_z\\end{bmatrix}^T` are
*normalized* observation vectors from the tri-axial accelerometer and
magnetometer sensors in the body frame :math:`b`, respectively.

The reference frame, :math:`r`, is chosen to follow the `North-East-Down
<https://en.wikipedia.org/wiki/Local_tangent_plane_coordinates#Local_north,_east,_down_(NED)_coordinates>`_
(NED) coordinate system.

Thus, the reference vectors are given by :math:`^r\\mathbf{a}=\\begin{bmatrix}0 & 0 & 1\\end{bmatrix}^T`
and :math:`^r\\mathbf{m}=\\begin{bmatrix}m_N & m_E & m_D\\end{bmatrix}^T`,
although this solution neglects the Easterly geomagnetic field describing it
based on the local `magnetic dip <https://en.wikipedia.org/wiki/Magnetic_dip>`_
:math:`\\theta` to form a simpler expression
:math:`^r\\mathbf{m}=\\begin{bmatrix}\\cos\\theta & 0 & -\\sin\\theta\\end{bmatrix}^T`

The most common solution to Wahba's problem relates two vector observation
pairs with weights. Here, two equivalent weights equal to :math:`0.5` yield the
loss function:

.. math::
    L(\\mathbf{C}) = \\frac{1}{2}\\Big[\\frac{1}{2}\\|^b\\mathbf{a}-\\mathbf{C}\,^r\\mathbf{a}\\|^2+\\frac{1}{2}\\|^b\\mathbf{m}-\\mathbf{C}\,^r\\mathbf{m}\\|^2\\Big]

Using `Davenport's q-method <./davenport.html>`_ we find the minimum of
:math:`L(\\mathbf{C})` by calculating the maximum eigenvalue of Davenport's
matrix :math:`\\mathbf{K}` in terms of quaternions:

.. math::
    \\begin{array}{rcl}
    \\mathbf{Kq} &=& \\lambda_{\\mathrm{max}}\\mathbf{q} \\\\
    \\begin{bmatrix}
        \\mathbf{B}+\\mathbf{B}-\\mathrm{tr}(\\mathbf{B})\\mathbf{I} & \\mathbf{z} \\\\
        \\mathbf{z}^T & \\mathrm{tr}(\\mathbf{B})
    \\end{bmatrix}
    \\begin{bmatrix}q_x\\\\q_y\\\\q_z\\\\q_w\\end{bmatrix} &=&
    \\lambda_{\\mathrm{max}}\\begin{bmatrix}q_x\\\\q_y\\\\q_z\\\\q_w\\end{bmatrix}
    \\end{array}

.. warning::
    The definition of the quaternion does NOT follow the common practice of
    setting the scalar part first, but actually putting the vector part first:
    :math:`\\mathbf{q}=\\begin{pmatrix}q_x & q_y & q_z & q_w\\end{pmatrix}`.
    Consider this for all operations. The resulting attitude is, however,
    represented following the common definition at the final step:
    :math:`\\mathbf{q}=\\begin{pmatrix}q_w & q_x & q_y & q_z\\end{pmatrix}`.

The helping arrays are:

.. math::
    \\begin{array}{rcl}
    \\mathbf{B} &=& \\frac{1}{2}\\big(\,^b\\mathbf{a}(\,^r\\mathbf{a})^T+\,^b\\mathbf{m}(^r\\mathbf{m})^T\\big) \\\\ && \\\\
    \\mathbf{z}^T &=& \\begin{bmatrix}B_{23}-B_{32}\\\\B_{31}-B_{13}\\\\B_{12}-B_{21}\\end{bmatrix}
    \\end{array}

.. note::
    Indexing is normally starting from zero, especially in computational setups,
    but the article starts it from one, and it is kept like that in this
    documentation to coincide with the original document.

After knowing that :math:`\\|\\mathbf{a}\\|=\\|\\mathbf{m}\\|=m_N^2+m_D^2=1`,
the four eigenvalues of :math:`\\mathbf{K}` are computed by:

.. math::
    \\begin{array}{rcl}
    \\lambda\\mathbf{K}_1 &=& \\frac{1}{2}\\sqrt{2m_N^2 + 2 + 2m_D^2} = 1 \\\\
    \\lambda\\mathbf{K}_2 &=& \\frac{1}{2}\\sqrt{-2m_N^2 + 2 + 2m_D^2} = \\frac{1}{2}\\sqrt{4m_D^2} = |m_D| \\\\
    \\lambda\\mathbf{K}_3 &=& \\frac{1}{2}\\sqrt{-2m_N^2 + 2 + 2m_D^2} = \\frac{1}{2}\\sqrt{4m_D^2} = -|m_D| \\\\
    \\lambda\\mathbf{K}_4 &=& \\frac{1}{2}\\sqrt{2m_N^2 + 2 + 2m_D^2} = -1
    \\end{array}

The attitude quaternion is the eigenvector of :math:`\\mathbf{K}` corresponding
to the maximum eigenvalue 1. This shows that the AMC is actually
self-constrained and does not require the outer information of the magnetic dip
angle.

.. warning::
    When the AMC is used near the poles there may be two ambiguous solutions
    corresponding to an eigenvalue equal to 1, disorienting the system and
    rendering this estimator useless on polar regions.

Finally, we must compute the eigenvector of the eigenvalule 1. We start defining:

.. math::
    \\mathbf{S} = \mathbf{K} - \\mathbf{I}

where :math:`\\mathbf{S}` can be further expanded with matrix row operations in
echelon form :math:`\\mathbf{T}=\\mathbf{\\Lambda}_1\\mathbf{\\Lambda}_2\\mathbf{\\Lambda}_3\\mathbf{S}`:

.. math::
    \\begin{array}{rcl}
    \\mathbf{\\Lambda}_1 &=& \\begin{bmatrix}Y_{11}&&&\\\\Y_{12}&1&&\\\\Y_{13}&&1&\\\\Y_{14}&&&1\\end{bmatrix} \\\\&&\\\\
    \\mathbf{\\Lambda}_2 &=& \\begin{bmatrix}1&Y_{21}&&\\\\&Y_{22}&&\\\\&Y_{23}&1&\\\\&Y_{24}&&1\\end{bmatrix} \\\\&&\\\\
    \\mathbf{\\Lambda}_3 &=& \\begin{bmatrix}1&&Y_{31}&\\\\&1&Y_{32}&\\\\&&Y_{33}&\\\\&&Y_{34}&1\\end{bmatrix}
    \\end{array}

So, :math:`\\mathbf{T}` is expanded as:

.. math::
    \\mathbf{T} = \\begin{bmatrix}1 & & & a \\\\ & 1 & & b \\\\ & & 1 & c \\\\ & & & 0\\end{bmatrix}

where:

.. math::
    \\begin{array}{rcl}
    a &=& B_{23}\\big[Y_{11}+Y_{12}(Y_{21}+Y_{23}Y_{31})+Y_{13}Y_{31}\\big] - (B_{13}-B_{31})(Y_{21}+Y_{23}Y_{31}) - Y_{31}B_{21} \\\\
    b &=& B_{23}\\big[Y_{12}(Y_{22}+Y_{23}Y_{32})+Y_{13}Y_{32}\\big] - (B_{13}-B_{31})(Y_{22}+Y_{23}Y_{32}) - Y_{32}B_{21} \\\\
    c &=& B_{23}(Y_{13}Y_{33}+Y_{12}Y_{23}Y_{33}) - Y_{33}B_{21} - Y_{23}Y_{33}(B_{13}-B_{31})
    \\end{array}

The *unnormalized scalar-wise* quaternion is given by the solution:

.. math::
    \\mathbf{q} = \\begin{pmatrix}-1 & a & b & c\\end{pmatrix}

which is easily normalized:

.. math::
    \\mathbf{q} = \\frac{1}{\\sqrt{a^2+b^2+c^2+1}}\\begin{pmatrix}-1 & a & b & c\\end{pmatrix}

References
----------
.. [Liu] Zhuohua Liu, Wei Liu, Xiangyang Gong, and Jin Wu, "Simplified Attitude
    Determination Algorithm Using Accelerometer and Magnetometer with Extremely
    Low Execution Time," Journal of Sensors, vol. 2018, Article ID 8787236,
    11 pages, 2018. https://doi.org/10.1155/2018/8787236.

"""

import numpy as np

def _assert_iterables(item, item_name: str = 'iterable'):
    if not isinstance(item, (list, tuple, np.ndarray)):
        raise TypeError(f"{item_name} must be given as an array. Got {type(item)}")

class FAMC:
    """
    Fast Accelerometer-Magnetometer Combination

    Parameters
    ----------
    acc : numpy.ndarray, default: None
        M-by-3 array with measurements of acceleration in in m/s^2
    mag : numpy.ndarray, default: None
        M-by-3 array with measurements of magnetic field in mT

    Attributes
    ----------
    acc : numpy.ndarray
        M-by-3 array with M accelerometer samples.
    mag : numpy.ndarray
        M-by-3 array with m magnetometer samples.
    Q : numpy.array, default: None
        M-by-4 Array with all estimated quaternions, where M is the number of
        samples. Equal to None when no estimation is performed.

    Raises
    ------
    ValueError
        When dimension of input arrays ``acc`` and ``mag`` are not equal.

    Examples
    --------
    >>> acc_data.shape, mag_data.shape      # NumPy arrays with sensor data
    ((1000, 3), (1000, 3))
    >>> from ahrs.filters import FAMC
    >>> famc = FAMC(acc=acc_data, mag=mag_data)
    >>> famc.Q       # Estimated attitudes as Quaternions
    array([[-0.82311077,  0.45760535, -0.33408929, -0.0383452 ],
           [-0.82522048,  0.4547043 , -0.33277675, -0.03892033],
           [-0.82463698,  0.4546915 , -0.33422422, -0.03903417],
           ...,
           [-0.82420642,  0.56217735,  0.02548005, -0.06317571],
           [-0.82364606,  0.56311099,  0.0241655 , -0.06268338],
           [-0.81844766,  0.57077781,  0.02532182, -0.06095017]])
    >>> famc.Q.shape
    (1000, 4)

    """
    def __init__(self, acc: np.ndarray = None, mag: np.ndarray = None):
        self.acc: np.ndarray = acc
        self.mag: np.ndarray = mag
        self.Q = None
        if self.acc is not None and self.mag is not None:
            self.Q = self._compute_all()

    def _compute_all(self) -> np.ndarray:
        """
        Estimate the quaternions given all data.

        Attributes ``acc`` and ``mag`` must contain data.

        Returns
        -------
        Q : numpy.ndarray
            M-by-4 Array with all estimated quaternions, where M is the number
            of samples.

        """
        _assert_iterables(self.acc, 'acc')
        _assert_iterables(self.mag, 'mag')
        if self.acc.shape != self.mag.shape:
            raise ValueError("acc and mag are not the same size")
        if self.acc.ndim < 2:
            return self.estimate(self.acc, self.mag)
        num_samples = len(self.acc)
        return np.array([self.estimate(self.acc[t], self.mag[t]) for t in range(num_samples)])

    def estimate(self, acc: np.ndarray, mag: np.ndarray) -> np.ndarray:
        """
        Attitude Estimation

        Parameters
        ----------
        a : numpy.ndarray
            Sample of tri-axial Accelerometer.
        m : numpy.ndarray
            Sample of tri-axial Magnetometer.

        Returns
        -------
        q : numpy.ndarray
            Estimated quaternion of the form :math:`\\begin{pmatrix}q_w & q_x & q_y & q_z\\end{pmatrix}`

        Examples
        --------
        >>> acc_data = np.array([4.098297, 8.663757, 2.1355896])
        >>> mag_data = np.array([-28.71550512, -25.92743566, 4.75683931])
        >>> from ahrs.filters import FAMC
        >>> famc = FAMC()
        >>> famc.estimate(acc=acc_data, mag=mag_data)   # Estimate attitude as quaternion
        array([-0.82311077,  0.45760535, -0.33408929, -0.0383452])

        """
        # Normalize measurements (eq. 10)
        a_norm = np.linalg.norm(acc)
        m_norm = np.linalg.norm(mag)
        if not a_norm > 0 or not m_norm > 0:    # handle NaN
            return None
        ax, ay, az = acc / a_norm               # A = [ax, ay, az] in body frame
        mx, my, mz = mag / m_norm               # M = [mx, my, mz] in body frame
        # Dynamic magnetometer reference vector
        m_D = ax*mx + ay*my + az*mz             # (eq. 13)
        m_N = np.sqrt(1.0 - m_D**2)
        # Parameters
        B = np.zeros((3, 3))                    # (eq. 18)
        B[:, 0] = m_N * np.array([mx, my, mz])
        B[:, 2] = m_D * np.array([mx, my, mz]) + np.array([ax, ay, az])
        B *= 0.5
        tau = B[0, 2] + B[2, 0]
        alpha = np.zeros(3)
        Y = np.zeros((3, 3))
        # First Row
        alpha[0] = B[2, 2] - B[0, 0] + 1.0
        Y[0, 0] = -1.0 / alpha[0]
        Y[0, 1] = B[1, 0] / alpha[0]
        Y[0, 2] = tau / alpha[0]
        # Second Row
        alpha[1] = -B[1, 0]**2/alpha[0] + B[0, 0] + B[2, 2] + 1
        Y[1, 0] = -B[1, 0] / (alpha[0] * alpha[1])
        Y[1, 1] = -1.0 / alpha[1]
        Y[1, 2] = (B[1, 2] + B[1, 0]*tau/alpha[0]) / alpha[1]
        # Third row
        alpha[2] = alpha[0] - 2 + tau**2/alpha[0] + Y[1, 2]**2*alpha[1]
        Y[2, 0] = (tau/alpha[0] + B[1, 0]*Y[1, 2]/alpha[0]) / alpha[2]
        Y[2, 1] = Y[1, 2] / alpha[2]
        Y[2, 2] = 1.0 / alpha[2]
        # Quaternion Elements (eq. 21)
        a = B[1, 2]*(Y[0, 0] + Y[0, 1]*(Y[1, 2]*Y[2, 0] + Y[1, 0]) + Y[0, 2]*Y[2, 0]) - (B[0, 2]-B[2, 0])*(Y[1, 2]*Y[2, 0] + Y[1, 0]) - Y[2, 0]*B[1, 0]
        b = B[1, 2]*(          Y[0, 1]*(Y[1, 2]*Y[2, 1] + Y[1, 1]) + Y[0, 2]*Y[2, 1]) - (B[0, 2]-B[2, 0])*(Y[1, 2]*Y[2, 1] + Y[1, 1]) - Y[2, 1]*B[1, 0]
        c = B[1, 2]*(          Y[0, 1]* Y[1, 2]*Y[2, 2]            + Y[0, 2]*Y[2, 2]) - (B[0, 2]-B[2, 0])*(Y[1, 2]*Y[2, 2])           - Y[2, 2]*B[1, 0]
        q = np.array([-1, a, b, c])         # (eq. 22)
        return q/np.linalg.norm(q)          # (eq. 23)
