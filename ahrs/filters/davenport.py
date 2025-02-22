# -*- coding: utf-8 -*-
"""
In 1965 `Grace Wahba <https://en.wikipedia.org/wiki/Grace_Wahba>`_ came up with
a simple, yet very intuitive, way to describe the problem of finding a rotation
between two coordinate systems.

Given a set of :math:`N` vector measurements :math:`\\mathbf{u}` in the body
coordinate system, an optimal attitude matrix :math:`\\mathbf{A}` would
minimize the `loss function <https://en.wikipedia.org/wiki/Loss_function>`_:

.. math::
    L(\\mathbf{A}) = \\frac{1}{2}\\sum_{i=1}^Nw_i|u_i-\\mathbf{A}v_i|^2

where :math:`u_i` is the i-th vector measurement in the body frame, :math:`v_i`
is the i-th vector in the reference frame, and :math:`w_i` are a set of :math:`N`
nonnegative weights for each observation. This famous formulation is known as
`Wahba's problem <https://en.wikipedia.org/wiki/Wahba%27s_problem>`_.

A first elegant solution was proposed by :cite:p:`davenport1968` that solves
this in terms of quaternions, yielding a unique optimal solution. The
corresponding **objective function** is defined as:

.. math::
    g(\\mathbf{A}) = 1 - L(\\mathbf{A}) = \\sum_{i=1}^Nw_i\\mathbf{U}^T\\mathbf{AV}

The objective function is at maximum when the loss function :math:`L(\\mathbf{A})`
is at minimum. The goal is, then, to find the optimal attitude matrix
:math:`\\mathbf{A}`, which *maximizes* :math:`g(\\mathbf{A})`. We first notice
that:

.. math::
    \\begin{array}{rl}
    g(\\mathbf{A}) =& \\sum_{i=1}^Nw_i\\mathrm{tr}\\big(\\mathbf{U}_i^T\\mathbf{AV}_i\\big) \\\\
    =& \\mathrm{tr}(\\mathbf{AB}^T)
    \\end{array}

where :math:`\\mathrm{tr}` denotes the `trace
<https://en.wikipedia.org/wiki/Trace_(linear_algebra)>`_ of a matrix, and
:math:`\\mathbf{B}` is the *attitude profile matrix*:

.. math::
    \\mathbf{B} = \\sum_{i=1}^Nw_i\\mathbf{UV}

Now, we must parametrize the attitude matrix in terms of a quaternion
:math:`\\mathbf{q}` :cite:p:`lerner1978` :

.. math::
    \\mathbf{A}(\\mathbf{q}) = (q_w^2-\\mathbf{q}_v\\cdot\\mathbf{q}_v)\\mathbf{I}_3+2\\mathbf{q}_v\\mathbf{q}_v^T-2q_w\\lfloor\\mathbf{q}\\rfloor_\\times

where :math:`\\mathbf{I}_3` is a :math:`3\\times 3` identity matrix, and the
expression :math:`\\lfloor \\mathbf{x}\\rfloor_\\times` is the `skew-symmetric
matrix <https://en.wikipedia.org/wiki/Skew-symmetric_matrix>`_ of a vector
:math:`\\mathbf{x}`. See the `quaternion page <../quaternion.html>`_ for further
details about this representation mapping.

The objective function, in terms of quaternion, becomes:

.. math::
    g(\\mathbf{q}) = (q_w^2-\\mathbf{q}_v\\cdot\\mathbf{q}_v)\\mathrm{tr}\\mathbf{B}^T + 2\\mathrm{tr}\\big(\\mathbf{q}_v\\mathbf{q}_v^T\\mathbf{B}^T\\big) + 2q_w\\mathrm{tr}(\\lfloor\\mathbf{q}\\rfloor_\\times\\mathbf{B}^T)

A simpler expression, using helper quantities, can be a bilinear relationship
of the form:

.. math::
    g(\\mathbf{q}) = \\mathbf{q}^T\\mathbf{Kq}

where the :math:`4\\times 4` matrix :math:`\\mathbf{K}` is built with:

.. math::
    \\mathbf{K} = \\begin{bmatrix}
    \\sigma & \\mathbf{z}^T \\\\
    \\mathbf{z} & \\mathbf{S}-\\sigma\\mathbf{I}_3
    \\end{bmatrix}

using the intermediate values:

.. math::
    \\begin{array}{rcl}
    \\sigma &=& \\mathrm{tr}\\mathbf{B} \\\\
    \\mathbf{S} &=& \\mathbf{B}+\\mathbf{B}^T \\\\
    \\mathbf{z} &=& \\begin{bmatrix}B_{23}-B_{32} \\\\ B_{31}-B_{13} \\\\ B_{12}-B_{21}\\end{bmatrix}
    \\end{array}

The optimal quaternion :math:`\\hat{\\mathbf{q}}`, which parametrizes the
optimal attitude matrix, is an eigenvector of :math:`\\mathbf{K}`. With the
help of `Lagrange multipliers <https://en.wikipedia.org/wiki/Lagrange_multiplier>`_,
:math:`g(\\mathbf{q})` is maximized if the eigenvector corresponding to the
largest eigenvalue :math:`\\lambda` is chosen.

.. math::
    \\mathbf{K}\\hat{\\mathbf{q}} = \\lambda\\hat{\\mathbf{q}}

The biggest disadvantage of this method is its computational load in the last
step of computing the eigenvalues and eigenvectors to find the optimal
quaternion.

"""

import numpy as np
from ..common.constants import MUNICH_LONGITUDE
from ..common.constants import MUNICH_LATITUDE
from ..common.constants import MUNICH_HEIGHT
from ..common.mathfuncs import cosd
from ..common.mathfuncs import sind
from ..utils.core import _assert_numerical_iterable
from ..utils.core import _assert_acc_mag_inputs
from ..utils.wmm import WMM
from ..utils.wgs84 import WGS

# Reference Observations in Munich, Germany
GRAVITY = WGS().normal_gravity(MUNICH_LATITUDE, MUNICH_HEIGHT)
wmm = WMM(latitude=MUNICH_LATITUDE, longitude=MUNICH_LONGITUDE, height=MUNICH_HEIGHT)
REFERENCE_MAGNETIC_VECTOR = np.array([wmm.X, wmm.Y, wmm.Z])

class Davenport:
    """
    Davenport's q-Method for attitude estimation

    Parameters
    ----------
    acc : numpy.ndarray, default: None
        N-by-3 array with measurements of acceleration in in m/s^2
    mag : numpy.ndarray, default: None
        N-by-3 array with measurements of magnetic field in mT
    weights : array-like
        Array with two weights used in each observation.
    magnetic_dip : float
        Magnetic Inclination angle, in degrees. Defaults to magnetic dip of
        Munich, Germany.
    gravity : float
        Normal gravity, in m/s^2. Defaults to normal gravity in Munich,
        Germany.

    Attributes
    ----------
    acc : numpy.ndarray
        N-by-3 array with N accelerometer samples.
    mag : numpy.ndarray
        N-by-3 array with N magnetometer samples.
    w : numpy.ndarray
        Weights of each observation.

    Raises
    ------
    ValueError
        When dimension of input arrays ``acc`` and ``mag`` are not equal.

    """
    def __init__(self, acc: np.ndarray = None, mag: np.ndarray = None, **kw) -> None:
        self.acc: np.ndarray = acc
        self.mag: np.ndarray = mag
        self.w: np.ndarray = kw.get('weights', np.ones(2))
        # Reference measurements
        mdip: np.ndarray = kw.get('magnetic_dip')           # Magnetic dip, in degrees
        self.m_q: np.ndarray = REFERENCE_MAGNETIC_VECTOR if mdip is None else np.array([cosd(mdip), 0., sind(mdip)])
        g: float = kw.get('gravity', GRAVITY)               # Earth's normal gravity, in m/s^2
        self.g_q: np.ndarray = np.array([0.0, 0.0, g])      # Normal Gravity vector
        if self.acc is not None and self.mag is not None:
            self.Q: np.ndarray = self._compute_all()

    def _compute_all(self) -> np.ndarray:
        """
        Estimate all quaternions given all data.

        Attributes ``acc`` and ``mag`` must contain data.

        Returns
        -------
        Q : array
            M-by-4 Array with all estimated quaternions, where M is the number
            of samples.

        """
        _assert_acc_mag_inputs(self.acc, self.mag)
        self.acc = np.copy(self.acc)
        self.mag = np.copy(self.mag)
        if self.acc.ndim < 2:
            return self.estimate(self.acc, self.mag)
        num_samples = len(self.acc)
        return np.array([self.estimate(self.acc[t], self.mag[t]) for t in range(num_samples)])

    def estimate(self, acc: np.ndarray = None, mag: np.ndarray = None) -> np.ndarray:
        """
        Attitude Estimation

        Parameters
        ----------
        acc : numpy.ndarray
            Sample of tri-axial Accelerometer in m/s^2
        mag : numpy.ndarray
            Sample of tri-axial Magnetometer in T

        Returns
        -------
        q : numpy.ndarray
            Estimated attitude as a quaternion.

        """
        _assert_numerical_iterable(acc, 'Gravitational acceleration vector')
        _assert_numerical_iterable(mag, 'Geomagnetic field vector')
        B = self.w[0]*np.outer(acc, self.g_q) + self.w[1]*np.outer(mag, self.m_q)   # Attitude profile matrix
        sigma = B.trace()
        z = np.array([B[1, 2]-B[2, 1], B[2, 0]-B[0, 2], B[0, 1]-B[1, 0]])
        S = B+B.T
        K = np.zeros((4, 4))
        K[0, 0] = sigma
        K[1:, 1:] = S - sigma*np.eye(3)
        K[0, 1:] = K[1:, 0] = z
        w, v = np.linalg.eig(K)
        return v[:, np.argmax(w)]       # Eigenvector associated to largest eigenvalue is optimal quaternion
