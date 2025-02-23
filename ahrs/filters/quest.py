# -*- coding: utf-8 -*-
"""
QUaternion ESTimator as described by Shuster in :cite:p:`shuster1981` and
:cite:p:`shuster1978`.

We start to define the goal of finding an orthogonal matrix :math:`\\mathbf{A}`
that minimizes the loss function:

.. math::
    L(\\mathbf{A}) = \\frac{1}{2}\\sum_{i=1}^n a_i |\\hat{\\mathbf{W}}_i - \\mathbf{A}\\hat{\\mathbf{V}}_i|^2

where :math:`a_i` are a set of non-negative weights such that :math:`\\sum_{i=1}^na_i=1`,
:math:`\\hat{\\mathbf{V}}_i` are nonparallel **reference vectors**, and
:math:`\\hat{\\mathbf{W}}_i` are the corresponding **observation vectors**.

The gain function :math:`g(\\mathbf{A})` is defined by

.. math::
    g(\\mathbf{A}) = 1 - L(\\mathbf{A}) = \\sum_{i=1}^na_i\\,\\hat{\\mathbf{W}}_i^T\\mathbf{A}\\hat{\\mathbf{V}}_i

The loss function :math:`L(\\mathbf{A})` is at its minimum when the gain
function :math:`g(\\mathbf{A})` is at its maximum. The gain function can be
reformulated as:

.. math::
    g(\\mathbf{A}) = \\sum_{i=1}^na_i\\mathrm{tr}\\big(\\hat{\\mathbf{W}}_i^T\\mathbf{A}\\hat{\\mathbf{V}}_i\\big) = \\mathrm{tr}(\\mathbf{AB}^T)

where :math:`\\mathrm{tr}` is the `trace <https://en.wikipedia.org/wiki/Trace_(linear_algebra)>`_
of a matrix, and :math:`\\mathbf{B}` is the **attitude profile matrix**:

.. math::
    \\mathbf{B} = \\sum_{i=1}^na_i\\hat{\\mathbf{W}}_i\\hat{\\mathbf{V}}_i^T

The quaternion :math:`\\bar{\\mathbf{q}}` representing a rotation is defined by
Shuster as:

.. math::
    \\bar{\\mathbf{q}} = \\begin{bmatrix}\\mathbf{Q} \\\\ q\\end{bmatrix}
    = \\begin{bmatrix}\\hat{\\mathbf{X}}\\sin\\frac{\\theta}{2} \\\\ \\cos\\frac{\\theta}{2}\\end{bmatrix}

where :math:`\\hat{\\mathbf{X}}` is the axis of rotation, and :math:`\\theta`
is the angle of rotation about :math:`\\hat{\\mathbf{X}}`.

.. warning::
    The definition of a quaternion used by Shuster sets the vector part
    :math:`\\mathbf{Q}` followed by the scalar part :math:`q`. This module,
    however, will return the estimated quaternion with the *scalar part first*
    and followed by the vector part: :math:`\\bar{\\mathbf{q}} = \\begin{bmatrix}q
    & \\mathbf{Q}\\end{bmatrix}`

Because the quaternion works as a versor, it must satisfy:

.. math::
    \\bar{\\mathbf{q}}^T\\bar{\\mathbf{q}} = |\\mathbf{Q}|^2 + q^2 = 1

The attitude matrix :math:`\\mathbf{A}` is related to the quaternion by:

.. math::
    \\mathbf{A}(\\bar{\\mathbf{q}}) = (q^2-\\mathbf{Q}\\cdot\\mathbf{Q})\\mathbf{I} + 2\\mathbf{QQ}^T + 2q\\lfloor\\mathbf{Q}\\rfloor_\\times

where :math:`\\mathbf{I}` is the identity matrix, and :math:`\\lfloor\\mathbf{Q}\\rfloor_\\times`
is the **antisymmetric matrix** of :math:`\\mathbf{Q}`, a.k.a. the
`skew-symmetric matrix <https://en.wikipedia.org/wiki/Skew-symmetric_matrix>`_:

.. math::
    \\lfloor\\mathbf{Q}\\rfloor_\\times =
    \\begin{bmatrix}0 & Q_3 & -Q_2 \\\\ -Q_3 & 0 & Q_1 \\\\ Q_2 & -Q_1 & 0\\end{bmatrix}

Now the gain function can be rewritten again, but in terms of quaternions:

.. math::
    g(\\bar{\\mathbf{q}}) = (q^2-\\mathbf{Q}\\cdot\\mathbf{Q})\\mathrm{tr}\\mathbf{B}^T + 2\\mathrm{tr}\\big(\\mathbf{QQ}^T\\mathbf{B}^T\\big) + 2q\\mathrm{tr}\\big(\\lfloor\\mathbf{Q}\\rfloor_\\times\\mathbf{B}^T\\big)

A further simplification gives:

.. math::
    g(\\bar{\\mathbf{q}}) = \\bar{\\mathbf{q}}^T\\mathbf{K}\\bar{\\mathbf{q}}

where the :math:`4\\times 4` matrix :math:`\\mathbf{K}` is given by:

.. math::
    \\mathbf{K} = \\begin{bmatrix} \\mathbf{S} - \\sigma\\mathbf{I} & \\mathbf{Z} \\\\ \\mathbf{Z}^T & \\sigma \\end{bmatrix}

using the helper values:

.. math::
    \\begin{array}{rcl}
    \\sigma &=& \\mathrm{tr}\\mathbf{B} \\\\ && \\\\
    \\mathbf{S} &=& \\mathbf{B} + \\mathbf{B}^T \\\\ && \\\\
    \\mathbf{Z} &=& \\sum_{i=1}^na_i\\big(\\hat{\\mathbf{W}}_i\\times\\hat{\\mathbf{V}}_i\\big)
    \\end{array}

.. note::
    :math:`\\mathbf{Z}` can be also defined from :math:`\\lfloor\\mathbf{Z}\\rfloor_\\times = \\mathbf{B} - \\mathbf{B}^T`

A new gain function :math:`g'(\\bar{\\mathbf{q}})` with `Lagrange multipliers
<https://en.wikipedia.org/wiki/Lagrange_multiplier>`_ is defined:

.. math::
    g'(\\bar{\\mathbf{q}}) = \\bar{\\mathbf{q}}^T\\mathbf{K}\\bar{\\mathbf{q}} - \\lambda\\bar{\\mathbf{q}}^T\\bar{\\mathbf{q}}

It is verified that :math:`\\mathbf{K}\\bar{\\mathbf{q}}=\\lambda\\bar{\\mathbf{q}}`.
Thus, :math:`g(\\bar{\\mathbf{q}})` will be maximized if :math:`\\bar{\\mathbf{q}}_\\mathrm{opt}`
is chosen to be the eigenvector of :math:`\\mathbf{K}` belonging to the largest
eigenvalue of :math:`\\mathbf{K}`:

.. math::
    \\mathbf{K}\\bar{\\mathbf{q}}_\\mathrm{opt} = \\lambda_\\mathrm{max}\\bar{\\mathbf{q}}_\\mathrm{opt}

which is the desired result. This equation can be rearranged to read, for any
eigenvalue :math:`\\lambda`:

.. math::
    \\lambda = \\sigma + \\mathbf{Z}\\cdot\\mathbf{Y}

where :math:`\\mathbf{Y}` is the `Gibbs vector
<https://en.wikipedia.org/wiki/Rotation_formalisms_in_three_dimensions#Rodrigues_vector>`_,
a.k.a. the **Rodrigues vector**, defined as:

.. math::
    \\mathbf{Y} = \\frac{\\mathbf{Q}}{q} = \\hat{\\mathbf{X}}\\tan\\frac{\\theta}{2}

rewriting the quaternion as:

.. math::
    \\bar{\\mathbf{q}} = \\frac{1}{\\sqrt{1+|\\mathbf{Y}|^2}} = \\begin{bmatrix}\\mathbf{Y}\\\\ 1 \\end{bmatrix}

:math:`\\mathbf{Y}` and :math:`\\bar{\\mathbf{q}}` are representations of the
optimal attitude solution when :math:`\\lambda` is equal to :math:`\\lambda_\\mathrm{max}`,
leading to an equation for the eigenvalues:

.. math::
    \\lambda = \\sigma + \\mathbf{Z}^T \\frac{1}{(\\lambda+\\sigma)\\mathbf{I}-\\mathbf{S}}\\mathbf{Z}

which is equivalent to the characteristic equation of the eigenvalues of :math:`\\mathbf{K}`

With the aid of `Cayley-Hamilton theorem <https://en.wikipedia.org/wiki/Cayley%E2%80%93Hamilton_theorem>`_
we can get rid of the Gibbs vector to find a more convenient expression of the
characteristic equation:

.. math::
    \\lambda^4-(a+b)\\lambda^2-c\\lambda+(ab+c\\sigma-d)=0

where:

.. math::
    \\begin{array}{rcl}
    a &=& \\sigma^2-\\kappa \\\\ && \\\\
    b &=& \\sigma^2 + \\mathbf{Z}^T\\mathbf{Z} \\\\ && \\\\
    c &=& \\Delta + \\mathbf{Z}^T\\mathbf{SZ} \\\\ && \\\\
    d &=& \\mathbf{Z}^T\\mathbf{S}^2\\mathbf{Z} \\\\ && \\\\
    \\sigma &=& \\frac{1}{2}\\mathrm{tr}\\mathbf{S} \\\\ && \\\\
    \\kappa &=& \\mathrm{tr}\\big(\\mathrm{adj}(\\mathbf{S})\\big) \\\\ && \\\\
    \\Delta &=& \\mathrm{det}(\\mathbf{S})
    \\end{array}

To find :math:`\\lambda` we implement the `Newton-Raphson method
<https://en.wikipedia.org/wiki/Newton%27s_method>`_ using the sum of the
weights :math:`a_i` (in the beginning is constrained to be equal to 1) as a
starting value.

.. math::
    \\lambda_{t+1} \\gets \\lambda_t - \\frac{f(\\lambda)}{f'(\\lambda)}
    = \\lambda_t - \\frac{\\lambda^4-(a+b)\\lambda^2-c\\lambda+(ab+c\\sigma-d)}{4\\lambda^3-2(a+b)\\lambda-c}

For sensor accuracies better than 1 arc-min (1 degree) the accuracy of a 64-bit
word is exhausted after only one iteration.

Finally, the **optimal quaternion** describing the attitude is found as:

.. math::
    \\bar{\\mathbf{q}}_\\mathrm{opt} = \\frac{1}{\\sqrt{\\gamma^2+|\\mathbf{X}|^2}} \\begin{bmatrix}\\mathbf{X}\\\\ \\gamma \\end{bmatrix}

with:

.. math::
    \\begin{array}{rcl}
    \\mathbf{X} &=& (\\alpha\\mathbf{I} + \\beta\\mathbf{S} + \\mathbf{S}^2)\\mathbf{Z} \\\\ && \\\\
    \\gamma &=& (\\lambda + \\sigma)\\alpha - \\Delta \\\\ && \\\\
    \\alpha &=& \\lambda^2 - \\sigma^2 + \\kappa \\\\ && \\\\
    \\beta &=& \\lambda - \\sigma
    \\end{array}

This solution can still lead to an indeterminant result if both :math:`\\gamma`
and :math:`\\mathbf{X}` vanish simultaneously. :math:`\\gamma` vanishes if and
only if the angle of rotation is equal to :math:`\\pi`, even if
:math:`\\mathbf{X}` does not vanish along.

"""

from typing import Union
import numpy as np

from ..common.mathfuncs import cosd
from ..common.mathfuncs import sind
from ..common.constants import MUNICH_LATITUDE
from ..common.constants import MUNICH_LONGITUDE
from ..common.constants import MUNICH_HEIGHT
from ..utils.core import _assert_numerical_iterable
from ..utils.core import _assert_acc_mag_inputs

# Reference Observations in Munich, Germany
from ..utils.wmm import WMM
from ..utils.wgs84 import WGS
GRAVITY = WGS().normal_gravity(MUNICH_LATITUDE, MUNICH_HEIGHT)
wmm = WMM(latitude=MUNICH_LATITUDE, longitude=MUNICH_LONGITUDE, height=MUNICH_HEIGHT)
REFERENCE_MAGNETIC_VECTOR = np.array([wmm.X, wmm.Y, wmm.Z])

def _set_magnetic_field_vector(magnetic_dip: Union[int, float, list, tuple, np.ndarray]):
    """
    Set the magnetic reference vector.

    Parameters
    ----------
    magnetic_dip : float, array-like
        Magnetic dip, in degrees, or local geomagnetic reference as a
        three-dimensional vector.

    """
    if isinstance(magnetic_dip, bool):
        raise TypeError("magnetic_dip must be given as a float, list, tuple or NumPy array. Got bool")
    elif isinstance(magnetic_dip, (float, int)):
        magnetic_field = np.array([cosd(magnetic_dip), 0., sind(magnetic_dip)])
    elif isinstance(magnetic_dip, (list, tuple, np.ndarray)):
        if not all(isinstance(x, (float, int)) for x in magnetic_dip):
            raise TypeError("magnetic_dip must be an array of floats. Contains non-numeric values.")
        magnetic_field = np.copy(magnetic_dip)
    elif magnetic_dip is None:
        magnetic_field = REFERENCE_MAGNETIC_VECTOR
    else:
        raise TypeError(f"magnetic_dip must be given as a float, list, tuple or NumPy array. Got {type(magnetic_dip)}")
    if magnetic_field.shape != (3,):
        raise ValueError(f"magnetic_dip array must contain 3 elements. Got {magnetic_field.shape}")
    return magnetic_field / np.linalg.norm(magnetic_field)

class QUEST:
    """
    QUaternion ESTimator

    Parameters
    ----------
    acc : numpy.ndarray, default: None
        N-by-3 array with measurements of acceleration in in m/s^2
    mag : numpy.ndarray, default: None
        N-by-3 array with measurements of magnetic field in mT
    weights : array-like
        Array with two weights. One per sensor measurement.
    magnetic_dip : float
        Local magnetic inclination angle, in degrees.
    gravity : float
        Local normal gravity, in m/s^2.

    Attributes
    ----------
    acc : numpy.ndarray
        N-by-3 array with N accelerometer samples.
    mag : numpy.ndarray
        N-by-3 array with N magnetometer samples.
    w : numpy.ndarray
        Weights for each observation.

    Raises
    ------
    ValueError
        When dimension of input arrays ``acc`` and ``mag`` are not equal.

    """
    def __init__(self, acc: np.ndarray = None, mag: np.ndarray = None, **kw):
        self.acc: np.ndarray = acc
        self.mag: np.ndarray = mag
        self.w: np.ndarray = kw.get('weights', 0.5*np.ones(2))
        # Reference measurements
        self.m_q: np.ndarray = _set_magnetic_field_vector(kw.get('magnetic_dip'))
        self.g_q: np.ndarray = np.array([0., 0., 1.])  # Normal Gravity vector
        if self.acc is not None and self.mag is not None:
            self.Q: np.ndarray = self._compute_all()

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
        _assert_acc_mag_inputs(self.acc, self.mag)
        acc, mag = np.copy(self.acc), np.copy(self.mag)
        if acc.ndim < 2:
            return self.estimate(acc, mag)
        num_samples = len(acc)
        return np.array([self.estimate(acc[t], mag[t]) for t in range(num_samples)])

    def estimate(self, acc: np.ndarray, mag: np.ndarray) -> np.ndarray:
        """
        Attitude Estimation.

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
        acc = acc/np.linalg.norm(acc)
        mag = mag/np.linalg.norm(mag)
        B = self.w[0]*np.outer(acc, self.g_q) + self.w[1]*np.outer(mag, self.m_q)   # Attitude profile matrix
        S = B + B.T
        z = np.array([B[1, 2]-B[2, 1], B[2, 0]-B[0, 2], B[0, 1]-B[1, 0]])   # Pseudovector (Axial vector)
        # Parameters of characeristic equation (eq. 63)
        sigma = B.trace()
        Delta = np.linalg.det(S)
        adjS = Delta*np.linalg.inv(S)
        kappa = adjS.trace()
        ### Elements of characteristic polynomial (eq. 71)
        a = sigma**2 - kappa
        b = sigma**2 + z.T @ z
        c = Delta + z.T @ S @ z
        d = z.T @ S@S @ z
        # Newton-Raphson method (eq. 70)
        k = a*b + c*sigma - d
        l_max = l_old = self.w.sum()
        num_iters = 0
        while abs(l_old-l_max) > 1e-8 or num_iters <= 10:
            l_old = l_max
            phi = l_max**4 - (a+b)*l_max**2 - c*l_max + k
            phi_prime = 4.0*l_max**3 - 2.0*(a+b)*l_max - c
            l_max -= phi/phi_prime
            num_iters += 1
        # (eq. 66)
        alpha = l_max**2 - sigma**2 + kappa
        beta = l_max - sigma
        gamma = (l_max + sigma)*alpha - Delta
        Chi = (alpha*np.identity(3) + beta*S + S@S)@ z      # (eq. 68)
        # Optimal Quaternion (eq. 69)
        q = [gamma, *Chi]
        q /= np.linalg.norm(q)
        return q
