# -*- coding: utf-8 -*-
"""
Super-fast Attitude from Accelerometer and Magnetometer
=======================================================

This novel estimator proposed by [Wu]_, offers an extremely simplified
computation of `Davenport's <../davenport.html>`_ solution to
`Wahba's problem <https://en.wikipedia.org/wiki/Wahba%27s_problem>`_, where the
full solution is reduced to a couple of floating point operations, without
losing accuracy and gaining important computation times.

The accelerometer and magnetometer have their normalized observations
:math:`^b\\mathbf{a}=\\begin{bmatrix}a_x&a_y&a_z\\end{bmatrix}^T`,
:math:`^b\\mathbf{m}=\\begin{bmatrix}m_x&m_y&m_z\\end{bmatrix}^T` in the body
frame :math:`b`.

Their corresponding normalized vectors in the
`NED <https://en.wikipedia.org/wiki/Local_tangent_plane_coordinates#Local_north,_east,_down_(NED)_coordinates>`_
reference frame :math:`^r\\mathbf{a}=\\begin{bmatrix}0&0&1\\end{bmatrix}^T` and
:math:`^b\\mathbf{a}=\\begin{bmatrix}m_N&0&m_D\\end{bmatrix}^T` are such that:

.. math::
    a_x^2+a_y^2+a_z^2 = m_x^2+m_y^2+m_z^2 = m_N^2+m_D^2 = 1

and they are related by the `direction cosine matrix <../dcm.html>`_
:math:`\\mathbf{C}\\in SO(3)` with the minimization of Wahba's problem as:

.. math::
    \\mathrm{min} \\big(w\\|\,^b\\mathbf{a}-\\mathbf{C}\,^r\\mathbf{a}\\|^2+(1-w)\\|\,^b\\mathbf{m}-\\mathbf{C}\,^r\\mathbf{m}\\|^2\\big)

where :math:`w` is the weight of the accelerometer correlation and :math:`1-w`
is the one of the magnetometer.

The solution to Wahba's problem is equivalent to finding the eigenvector of the
maximum eigenvalue of Davenport's matrix :math:`\\mathbf{K}`:

.. math::
    \\mathbf{K} =
    \\begin{bmatrix}
        \\mathbf{B}+\\mathbf{B}-\\mathrm{tr}(\\mathbf{B})\\mathbf{I}_3 & \\mathbf{z} \\\\ && \\\\
        \\mathbf{z}^T & \\mathrm{tr}(\\mathbf{B})
    \\end{bmatrix}

where :math:`\\mathrm{tr}` is the `matrix trace <https://en.wikipedia.org/wiki/Trace_(linear_algebra)>`_,
:math:`\\mathbf{I}_3` is the :math:`3\\times 3` `identity matrix <https://en.wikipedia.org/wiki/Identity_matrix>`_,
and the helper arrays are:

.. math::
    \\begin{array}{rcl}
    \\mathbf{B} &=& w\,^b\\mathbf{a}\,^r\\mathbf{a}^T + (1-w)\,^b\\mathbf{m}\,^r\\mathbf{m}^T \\\\
    \\mathbf{z} &=& \\begin{bmatrix}B_{23}-B_{32}\\\\B_{31}-B_{13}\\\\B_{12}-B_{21}\\end{bmatrix}
    \\end{array}

in which :math:`B_{ij}` stands for the element of :math:`\\mathbf{B}` in the
:math:`i`-th row and the :math:`j`-th column.

.. note::
    Indexing is normally starting from zero, especially in computational setups,
    but the article starts it from one, and it is kept like that in this
    documentation to coincide with the original document.

The eigenvalues of :math:`\\mathbf{K}` are given by:

.. math::
    \\begin{array}{rcl}
    \\lambda_{\\mathbf{K},1} &=& \\sqrt{(1-w)^2+w^2+2w(1-w)(\\alpha m_D+V)} \\\\ && \\\\
    \\lambda_{\\mathbf{K},2} &=& \\sqrt{(1-w)^2+w^2+2w(1-w)(\\alpha m_D-V)} \\\\ && \\\\
    \\lambda_{\\mathbf{K},3} &=& -\\sqrt{(1-w)^2+w^2+2w(1-w)(\\alpha m_D-V)} \\\\ && \\\\
    \\lambda_{\\mathbf{K},4} &=& -\\sqrt{(1-w)^2+w^2+2w(1-w)(\\alpha m_D+V)}
    \\end{array}

where

.. math::
    \\begin{array}{rcl}
    \\alpha &=& a_xm_x + a_ym_y + a_zm_z \\\\
    V &=& m_N\\sqrt{1-\\alpha^2}
    \\end{array}

The local `geomagnetic dip angle  <https://en.wikipedia.org/wiki/Magnetic_dip>`_
:math:`\\theta\\in[-\\frac{\\pi}{2}, \\frac{\\pi}{2}]` ensures that
:math:`m_N=\\cos\\theta>0` and :math:`\\lambda_{\\mathbf{K},1}>\\lambda_{\\mathbf{K},2}>\\lambda_{\\mathbf{K},3}>\\lambda_{\\mathbf{K},4}`.

So, the attitude quaternion should be the eigenvector associated to the
eigenvalue :math:`\\lambda_{\\mathbf{K},1}`.

The dip angle is not required in the accelerometer-magnetometer configuration,
since :math:`m_D=\\alpha` and :math:`m_N=\\sqrt{1-\\alpha^2}` always holds, and
the fundamental solution to :math:`(\\mathbf{K}-\\mathbf{I})\\mathbf{q}=0` is:

.. math::
    \\begin{array}{rcl}
    q_w &=& -a_y(m_N+m_x) + a_xm_y \\\\ && \\\\
    q_x &=& (a_z-1)(m_N+m_x) + a_x(m_D-m_z) \\\\ && \\\\
    q_y &=& (a_z-1)m_y + a_y(m_D-m_z) \\\\ && \\\\
    q_z &=& a_zm_D - a_xm_N - m_z
    \\end{array}

which shows that the weights are not even necessary. Finally, the normalized
quaternion representing the attitude is:

.. math::
    \\mathbf{q} = \\frac{1}{\\sqrt{q_w^2+q_x^2+q_y^2+q_z^2}}\\begin{pmatrix}q_w & q_x & q_y & q_z\\end{pmatrix}

This estimator is extremely short and relies solely on linear operations,
making it very suitable for low-cost and simple processors. Its accuracy is
comparable to that of QUEST and FQA, but it is one order of magnitude faster.

References
----------
.. [Wu] Jin Wu, Zebo Zhou, Hassen Fourati, Yuhua Cheng. A Super Fast Attitude
    Determination Algorithm for Consumer-Level Accelerometer and Magnetometer.
    IEEE Transactions on Con-sumer Electronics, Institute of Electrical and
    Electronics Engineers, 2018, 64 (3), pp. 375. 381.10.1109/tce.2018.2859625.
    hal-01922922
    (https://hal.inria.fr/hal-01922922/document)

"""

import numpy as np

class SAAM:
    """
    Super-fast Attitude from Accelerometer and Magnetometer

    Parameters
    ----------
    acc : numpy.ndarray, default: None
        N-by-3 array with measurements of acceleration in in m/s^2
    mag : numpy.ndarray, default: None
        N-by-3 array with measurements of magnetic field in mT

    Attributes
    ----------
    acc : numpy.ndarray
        N-by-3 array with N accelerometer samples.
    mag : numpy.ndarray
        N-by-3 array with N magnetometer samples.
    Q : numpy.ndarray, default: None
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
    >>> from ahrs.filters import SAAM
    >>> orientation = SAAM(acc=acc_data, mag=mag_data)
    >>> orientation.Q.shape                 # Estimated attitudes as Quaternions
    (1000, 4)
    >>> orientation.Q
    array([[-0.09867706, -0.33683592, -0.52706394, -0.77395607],
           [-0.10247491, -0.33710813, -0.52117549, -0.77732433],
           [-0.10082646, -0.33658091, -0.52082828, -0.77800078],
           ...,
           [-0.78760687, -0.57789515,  0.2131519,  -0.01669966],
           [-0.78683706, -0.57879487,  0.21313092, -0.02142776],
           [-0.77869223, -0.58616905,  0.22344478, -0.01080235]])

    """
    def __init__(self, acc: np.ndarray = None, mag: np.ndarray = None):
        self.acc = acc
        self.mag = mag
        self.Q = None
        if self.acc is not None and self.mag is not None:
            self.Q = self._compute_all()

    def _compute_all(self) -> np.ndarray:
        """
        Estimate the quaternions given all data.

        Attributes ``acc`` and ``mag`` must contain data. It is assumed that
        these attributes have the same shape (M, 3), where M is the number of
        observations.

        The full estimation is vectorized, to avoid the use of a time-wasting
        loop.

        Returns
        -------
        Q : numpy.ndarray
            M-by-4 Array with all estimated quaternions, where M is the number
            of samples.

        """
        if self.acc.shape != self.mag.shape:
            raise ValueError("acc and mag are not the same size")
        if self.acc.shape[-1] != 3:
            raise ValueError(f"Sensor data must be of shape (M, 3), but it got {self.acc.shape}")
        # Normalize measurements (eq. 1)
        ax, ay, az = np.transpose(self.acc/np.linalg.norm(self.acc, axis=1)[:, None])
        mx, my, mz = np.transpose(self.mag/np.linalg.norm(self.mag, axis=1)[:, None])
        # Dynamic magnetometer reference vector (eq. 12)
        mD = ax*mx + ay*my + az*mz
        mN = np.sqrt(1-mD**2)
        # Quaternion components (eq. 16)
        qw = ax*my - ay*(mN+mx)
        qx = (az-1)*(mN+mx) + ax*(mD-mz)
        qy = (az-1)*my + ay*(mD-mz)
        qz = az*mD - ax*mN-mz
        # Final quaternion (eq. 18)
        Q = np.c_[qw, qx, qy, qz]
        return Q/np.linalg.norm(Q, axis=1)[:, None]

    def estimate(self, acc: np.ndarray = None, mag: np.ndarray = None) -> np.ndarray:
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
            Estimated quaternion.

        Examples
        --------
        >>> acc_data = np.array([4.098297, 8.663757, 2.1355896])
        >>> mag_data = np.array([-28.71550512, -25.92743566, 4.75683931])
        >>> from ahrs.filters import SAAM
        >>> saam = SAAM()
        >>> saam.estimate(acc=acc_data, mag=mag_data)   # Estimate attitude as quaternion
        array([-0.09867706, -0.33683592, -0.52706394, -0.77395607])

        """
        # Normalize measurements (eq. 1)
        a_norm = np.linalg.norm(acc)
        m_norm = np.linalg.norm(mag)
        if not a_norm>0 or not m_norm>0:      # handle NaN
            return None
        ax, ay, az = acc/a_norm
        mx, my, mz = mag/m_norm
        # Dynamic magnetometer reference vector (eq. 12)
        mD = ax*mx + ay*my + az*mz
        mN = np.sqrt(1-mD**2)
        # Quaternion components (eq. 16)
        qw = ax*my - ay*(mN+mx)
        qx = (az-1)*(mN+mx) + ax*(mD-mz)
        qy = (az-1)*my + ay*(mD-mz)
        qz = az*mD - ax*mN-mz
        # Final quaternion (eq. 18)
        q = np.array([qw, qx, qy, qz])
        return q/np.linalg.norm(q)
