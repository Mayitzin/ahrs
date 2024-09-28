# -*- coding: utf-8 -*-
"""
Fast Linear Attitude Estimator
==============================

The Fast Linear Attitude Estimator (FLAE) obtains the attitude quaternion with
an eigenvalue-based solution as proposed by :cite:p:`wu2018`.

A symbolic solution to the corresponding characteristic polynomial is also
derived for a higher computation speed.

One-Dimensional Fusion
----------------------

We assume that we have a single observable (can be measured) frame. The sensor
outputs can be rotated with a :math:`3\\times 3` `Direction Cosine Matrix
<../dcm.html>`_ :math:`\\mathbf{C}` using:

.. math::
    \\mathbf{D}^b = \\mathbf{CD}^r

where :math:`\\mathbf{D}^b=\\begin{bmatrix}D_x^b & D_y^b & D_z^b\\end{bmatrix}^T`
is the observation vector in body frame and
:math:`\\mathbf{D}^r=\\begin{bmatrix}D_x^r & D_y^r & D_z^r\\end{bmatrix}^T` is
the observation vector in reference frame. To put it in terms of a quaternion,
we define the loss function :math:`\\mathbf{f}_D(\\mathbf{q})` as:

.. math::
    \\mathbf{f}_D(\\mathbf{q}) \\triangleq \\mathbf{CD}^r - \\mathbf{D}^b

where the quaternion :math:`\\mathbf{q}` is defined as:

.. math::
    \\begin{array}{rcl}
    \\mathbf{q}&=&\\begin{pmatrix}q_w & q_x & q_y & q_z\\end{pmatrix}^T \\\\
    &=& \\begin{pmatrix}\\cos\\frac{\\theta}{2} & n_x\\sin\\frac{\\theta}{2} & n_y\\sin\\frac{\\theta}{2} & n_z\\sin\\frac{\\theta}{2}\\end{pmatrix}^T
    \\end{array}

The purpose is to *minimize the loss function*. We start by expanding
:math:`\\mathbf{f}_D(\\mathbf{q})`:

.. math::
    \\begin{array}{rcl}
    \\mathbf{f}_D(\\mathbf{q}) &=& \\mathbf{CD}^r - \\mathbf{D}^b \\\\
    &=& \\mathbf{P}_D\\mathbf{q} - \\mathbf{D}^b \\\\
    &=& (D_x^r\\mathbf{P}_1 + D_y^r\\mathbf{P}_2 + D_z^r\\mathbf{P}_3)\\mathbf{q} - \\mathbf{D}^b \\\\
    &=& D_x^r\\mathbf{C}_1 + D_y^r\\mathbf{C}_2 + D_z^r\\mathbf{C}_3 - \\mathbf{D}^b
    \\end{array}

where :math:`\\mathbf{C}_1`, :math:`\\mathbf{C}_2` and :math:`\\mathbf{C}_3`
are the columns of :math:`\\mathbf{C}` represented as:

.. math::
    \\begin{array}{rcl}
    \\mathbf{C}_1 &=& \\mathbf{P}_1\\mathbf{q} = \\begin{bmatrix}q_w^2+q_x^2-q_y^2-q_z^2 \\\\ 2(q_xq_y + q_wq_z) \\\\ 2(q_xq_z - q_wq_y) \\end{bmatrix} \\\\ && \\\\
    \\mathbf{C}_2 &=& \\mathbf{P}_2\\mathbf{q} = \\begin{bmatrix}2(q_xq_y - q_wq_z) \\\\ q_w^2-q_x^2+q_y^2-q_z^2 \\\\2(q_wq_x + q_yq_z) \\end{bmatrix} \\\\ && \\\\
    \\mathbf{C}_3 &=& \\mathbf{P}_3\\mathbf{q} = \\begin{bmatrix}2(q_xq_z + q_wq_y) \\\\ 2(q_yq_z - q_wq_x) \\\\ q_w^2-q_x^2-q_y^2+q_z^2 \\end{bmatrix}
    \\end{array}

When :math:`\\mathbf{q}` is optimal, it satisfies:

.. math::
    \\mathbf{q} = \\mathbf{P}_D^\\dagger \\mathbf{D}^b

where :math:`\\mathbf{P}_D^\\dagger` is the `pseudo-inverse
<https://en.wikipedia.org/wiki/Moore%E2%80%93Penrose_inverse>`_ of
:math:`\\mathbf{P}_D` *if and only if* it has full rank.

.. note::
    A matrix is said to have **full rank** if its `rank
    <https://en.wikipedia.org/wiki/Rank_(linear_algebra)>`_ is equal to the
    largest possible for a matrix of the same dimensions, which is the lesser
    of the number of rows and columns.

The analytical form of any pseudo-inverse is normally difficult to obtain, but
thanks to the orthogonality of :math:`\\mathbf{P}_D` we get:

.. math::
    \\mathbf{P}_D^\\dagger = \\mathbf{P}_D^T = D_x^r\\mathbf{P}_1^T + D_y^r\\mathbf{P}_2^T + D_z^r\\mathbf{P}_3^T

The orientation :math:`\\mathbf{q}` is obtained from:

.. math::
    \\mathbf{P}_D^\\dagger\\mathbf{D}^b - \\mathbf{q} = \\mathbf{Gq}

Solving :math:`\\mathbf{Gq}=0` (*if and only if* :math:`\\mathrm{det}(\\mathbf{G})=0`)
using elementary row transformations we obtain the wanted orthonormal quaternion.

N-Dimensional Fusion
--------------------

We assume having :math:`n` observation equations, such that the error residual
vector is given by augmenting :math:`\\mathbf{f}_D(\\mathbf{q})` as:

.. math::
    \\mathbf{f}_{\\Sigma D}(\\mathbf{q}) =
    \\begin{bmatrix}
    \\sqrt{a_1}(\\mathbf{P}_{D_1}\\mathbf{q}-D_1^b) \\\\
    \\sqrt{a_2}(\\mathbf{P}_{D_2}\\mathbf{q}-D_2^b) \\\\
    \\vdots \\\\
    \\sqrt{a_n}(\\mathbf{P}_{D_n}\\mathbf{q}-D_n^b)
    \\end{bmatrix}

When :math:`\\mathbf{f}_{\\Sigma D}(\\mathbf{q})=0`, the equation satisfies:

.. math::
    \\begin{array}{rcl}
    \\mathbf{P}_{\\Sigma D}\\mathbf{q} &=& \\mathbf{D}_\\Sigma^b \\\\
    \\begin{bmatrix}
    \\sqrt{a_1}\\mathbf{P}_{D_1} \\\\
    \\sqrt{a_2}\\mathbf{P}_{D_2} \\\\
    \\vdots \\\\
    \\sqrt{a_n}\\mathbf{P}_{D_n}
    \\end{bmatrix}\\mathbf{q} &=&
    \\begin{bmatrix}
    \\sqrt{a_1}\\mathbf{D}_1^b \\\\
    \\sqrt{a_2}\\mathbf{D}_2^b \\\\
    \\vdots \\\\
    \\sqrt{a_n}\\mathbf{D}_n^b
    \\end{bmatrix}
    \\end{array}

Intuitively, we would solve it with :math:`\\mathbf{q}=\\mathbf{P}_{\\Sigma D}^\\dagger\\mathbf{D}_\\Sigma^b`,
but the pseudo-inverse of :math:`\\mathbf{P}_{\\Sigma D}` is very difficult to
compute. However, it is possible to transform the equation by the pseudo-inverse
matrices of :math:`\\mathbf{q}` and :math:`\\mathbf{D}_\\Sigma^b`:

.. math::
    \\mathbf{q}^\\dagger = (\\mathbf{D}_\\Sigma^b)^\\dagger \\mathbf{P}_{\\Sigma D}

:math:`\\mathbf{P}_{\\Sigma D}` can be further expanded into:

.. math::
    \\mathbf{P}_{\\Sigma D} = \\mathbf{U}_{D_x}\\mathbf{P}_1 + \\mathbf{U}_{D_y}\\mathbf{P}_2 + \\mathbf{U}_{D_z}\\mathbf{P}_3

where :math:`\\mathbf{P}_1`, :math:`\\mathbf{P}_2` and :math:`\\mathbf{P}_3`
are :math:`3\\times 4` matrices, and :math:`\\mathbf{U}_{D_x}`,
:math:`\\mathbf{U}_{D_y}` and :math:`\\mathbf{U}_{D_z}` are :math:`3n\\times 3`
matrices. Hence,

.. math::
    (\\mathbf{D}_\\Sigma^b)^\\dagger\\mathbf{P}_{\\Sigma D} = (\\mathbf{D}_\\Sigma^b)^\\dagger\\mathbf{U}_{D_x}\\mathbf{P}_1 + (\\mathbf{D}_\\Sigma^b)^\\dagger\\mathbf{U}_{D_y}\\mathbf{P}_2 + (\\mathbf{D}_\\Sigma^b)^\\dagger\\mathbf{U}_{D_z}\\mathbf{P}_3

The fusion equation finally arrives to:

.. math::
    \\mathbf{H}_x\\mathbf{P}_1 + \\mathbf{H}_y\\mathbf{P}_2 + \\mathbf{H}_z\\mathbf{P}_3 - \\mathbf{q}^\\dagger = \\mathbf{0}_{1\\times 4}

where :math:`\\mathbf{H}_x`, :math:`\\mathbf{H}_y` and :math:`\\mathbf{H}_z`
are :math:`1\\times 3` matrices

.. math::
    \\begin{array}{rcl}
    \\mathbf{H}_x &= (\\mathbf{D}_\\Sigma^b)^\\dagger\\mathbf{U}_{D_x} =&
    \\begin{bmatrix} \\sum_{i=1}^n a_iD_{x,i}^rD_{x,i}^b & \\sum_{i=1}^n a_iD_{x,i}^rD_{y,i}^b & \\sum_{i=1}^n a_iD_{x,i}^rD_{z,i}^b & \\end{bmatrix} \\\\ && \\\\
    \\mathbf{H}_y &= (\\mathbf{D}_\\Sigma^b)^\\dagger\\mathbf{U}_{D_y} =&
    \\begin{bmatrix} \\sum_{i=1}^n a_iD_{y,i}^rD_{x,i}^b & \\sum_{i=1}^n a_iD_{y,i}^rD_{y,i}^b & \\sum_{i=1}^n a_iD_{y,i}^rD_{z,i}^b & \\end{bmatrix} \\\\ && \\\\
    \\mathbf{H}_z &= (\\mathbf{D}_\\Sigma^b)^\\dagger\\mathbf{U}_{D_z} =&
    \\begin{bmatrix} \\sum_{i=1}^n a_iD_{z,i}^rD_{x,i}^b & \\sum_{i=1}^n a_iD_{z,i}^rD_{y,i}^b & \\sum_{i=1}^n a_iD_{z,i}^rD_{z,i}^b & \\end{bmatrix}
    \\end{array}

Refactoring the equation with a transpose operation, we obtain:

.. math::
    \\begin{array}{rcl}
    \\mathbf{P}_1^T\\mathbf{H}_x^T + \\mathbf{P}_2^T\\mathbf{H}_y^T + \\mathbf{P}_3^T\\mathbf{H}_z^T - \\mathbf{q} &=& \\mathbf{0} \\\\
    (\\mathbf{W} - \\mathbf{I})\\mathbf{q} &=& \\mathbf{0}
    \\end{array}

where the elements of :math:`\\mathbf{W}` are given by:

.. math::
    \\mathbf{W} =
    \\begin{bmatrix}
    H_{x1} + H_{y2} + H_{z3} & -H_{y3} + H_{z2} & -H_{z1} + H_{x3} & -H_{x2} + H_{y1} \\\\
    -H_{y3} + H_{z2} & H_{x1} - H_{y2} - H_{z3} & H_{x2} + H_{y1} & H_{x3} + H_{z1} \\\\
    -H_{z1} + H_{x3} & H_{x2} + H_{y1} & H_{y2} - H_{x1} - H_{z3} & H_{y3} + H_{z2} \\\\
    -H_{x2} + H_{y1} & H_{x3} + H_{z1} & H_{y3} + H_{x2} & H_{z3} - H_{y2} - H_{x1}
    \\end{bmatrix}

Eigenvector solution
--------------------

The simplest solution is to find the eigenvector corresponding to the largest
eigenvalue of :math:`\\mathbf{W}`, as used by `Davenport's <davenport.html>`_
q-method.

This has the advantage of returning a normalized and valid quaternion, which is
used to represent the attitude.

This method's main disadvantage is :math:`(\\mathbf{W}-\\mathbf{I})` suffering
from rank-deficient problems in the sensor outputs, besides its computational
cost.

Characteristic Polynomial
-------------------------

The fusion equation can be transformed by adding a small quaternion error
:math:`\\epsilon \\mathbf{q}`

.. math::
    \\mathbf{Wq} = (1+\\epsilon)\\mathbf{q}

recognizing that :math:`1+\\epsilon` is an eigenvalue of :math:`\\mathbf{W}`
the problem is now shifted to find the eigenvalue that is closest to 1.

Analytically the calculation of the eigenvalue of :math:`\\mathbf{W}` builds
first its characteristic polynomial as:

.. math::
    \\begin{array}{rcl}
    f(\\lambda) &=& \\mathrm{det}(\\mathbf{W}-\\lambda\\mathbf{I}_{4\\times 4}) \\\\
    &=& \\lambda^4 + \\tau_1\\lambda^2 + \\tau_2\\lambda + \\tau_3
    \\end{array}

where the coefficients are obtained from:

.. math::
    \\begin{array}{rcl}
    \\tau_1 &=& -2\\big(H_{x1}^2 + H_{x2}^2 + H_{x3}^2 + H_{y1}^2 + H_{y2}^2 + H_{y3}^2 + H_{z1}^2 + H_{z2}^2 + H_{z3}^2\\big) \\\\ && \\\\
    \\tau_2 &=& 8\\big(H_{x3}H_{y2}H_{z1} - H_{x2}H_{y3}H_{z1} - H_{x3}H_{y1}H_{z2} + H_{x1}H_{y3}H_{z2} + H_{x2}H_{y1}H_{z3} - H_{x1}H_{y2}H_{z3}\\big) \\\\ && \\\\
    \\tau_3 &=& \\mathrm{det}(\\mathbf{W})
    \\end{array}

Once :math:`\\lambda` is defined, the eigenvector can be obtained using
elementary row operations (Gaussian elimination).

There are two main methods to compute the optimal :math:`\\lambda`:

**1. Iterative Newton-Raphson method**

This 4th-order characteristic polynomial :math:`f(\\lambda)` can be solved with
the `Newton-Raphson's method <https://en.wikipedia.org/wiki/Newton%27s_method>`_
and the aid of its derivative, which is found to be:

.. math::
    f'(\\lambda) = 4\\lambda^3 + 2\\tau_1\\lambda + \\tau_2

The initial value for the root finding process can be set to 1, because
:math:`\\lambda` is very close to it. So, every iteration at :math:`n` updates
:math:`\\lambda` as:

.. math::
    \\lambda_{n+1} \\gets \\lambda_n - \\frac{f\\big(\\lambda_n\\big)}{f'\\big(\\lambda_n\\big)}
    = \\lambda_n - \\frac{\\lambda_n^4 + \\tau_1\\lambda_n^2 + \\tau_2\\lambda_n + \\tau_3}{4\\lambda_n^3 + 2\\tau_1\\lambda_n + \\tau_2}

The value of :math:`\\lambda` is commonly found after a couple iterations, but
the accuracy is not linear with the iteration steps and will not always achieve
good results.

**2. Symbolic method**

A more precise solution involves a symbolic approach, where four solutions to
the characteristic polynomial are obtained as follows:

.. math::
    \\begin{array}{rcl}
    \\lambda_1 &=& \\alpha \\Big(T_2 - \\sqrt{k_1 - k_2}\\Big) \\\\ && \\\\
    \\lambda_2 &=& \\alpha \\Big(T_2 + \\sqrt{k_1 - k_2}\\Big) \\\\ && \\\\
    \\lambda_3 &=& -\\alpha \\Big(T_2 + \\sqrt{k_1 + k_2}\\Big) \\\\ && \\\\
    \\lambda_4 &=& -\\alpha \\Big(T_2 - \\sqrt{k_1 + k_2}\\Big) \\\\
    \\end{array}

with the helper variables:

.. math::
    \\begin{array}{rcl}
    \\alpha &=& \\frac{1}{2\\sqrt{6}} \\\\ && \\\\
    k_1 &=& -T_2^2-12\\tau_1 \\\\ && \\\\
    k_2 &=& \\frac{12\\sqrt{6}\\tau_2}{T_2} \\\\ && \\\\
    T_0 &=& 2\\tau_1^3 + 27\\tau_2^2 - 72\\tau_1\\tau_3 \\\\ && \\\\
    T_1 &=& \\Big(T_0 + \\sqrt{-4(t_1^2+12\\tau_3)^3 + T_0^2}\\Big)^{\\frac{1}{3}} \\\\ && \\\\
    T_2 &=& \\sqrt{-4\\tau_1 + \\frac{2^{\\frac{4}{3}}(\\tau_1^2+12\\tau_3)}{T_1} + 2^{\\frac{2}{3}}T_1}
    \\end{array}

Then chose the :math:`\\lambda`, which is closest to 1. This way solving for
:math:`\\lambda` is truly shortened.

Optimal Quaternion
------------------

Having :math:`\\mathbf{N}=\\mathbf{W}-\\lambda\\mathbf{I}_{4\\times 4}`, the
matrix can be transformed via row operations to:

.. math::
    \\mathbf{N} \\to \\mathbf{N}' = \\begin{bmatrix}
    1 & 0 & 0 & \\chi \\\\ 0 & 1 & 0 & \\rho \\\\ 0 & 0 & 1 & \\upsilon \\\\ 0 & 0 & 0 & \\zeta
    \\end{bmatrix}

where :math:`\\zeta` is usually a very small number. To ensure that
:math:`(\\mathbf{W}-\\lambda\\mathbf{I}_{4\\times 4})=\\mathbf{q}=\\mathbf{0}`
has non-zero and unique solution, :math:`\\zeta` is chosen to be 0. Hence:

.. math::
    \\mathbf{N}' = \\begin{bmatrix}
    1 & 0 & 0 & \\chi \\\\ 0 & 1 & 0 & \\rho \\\\ 0 & 0 & 1 & \\upsilon \\\\ 0 & 0 & 0 & 0
    \\end{bmatrix}

Letting :math:`q_w=-1`, the solution to the optimal quaternion is obtained with:

.. math::
    \\mathbf{q} = \\begin{pmatrix}q_w\\\\q_x\\\\q_y\\\\q_z\\end{pmatrix} =
    \\begin{pmatrix}-1\\\\\\chi\\\\\\rho\\\\\\upsilon\\end{pmatrix}

Finally, the quaternion is normalized to be able to be used as a versor:

.. math::
    \\mathbf{q} = \\frac{1}{\\|\\mathbf{q}\\|} \\mathbf{q}

The decisive element of QUEST is its matrix :math:`\\mathbf{K}`, whereas for
FLAE :math:`\\mathbf{W}` plays the same essential role. Both algorithms spend
most of its computation obtaining said matrices.

FLAE has the same accuracy as other similar estimators (QUEST, SVD, etc.), but
with the advantage of being up to 47% faster than the fastest among them.

Another advantage is the symbolic formulation of the characteristic polynomial,
which does not contain any adjoint matrices, leading to a simpler (therefore
faster) calculation of the eigenvalues.

FLAE advocates for the symbolic method to calculate the eigenvalue. However,
the Newton iteration can be also used to achieve a similar performance to that
of QUEST.

"""

import numpy as np
from ..common.mathfuncs import cosd
from ..common.mathfuncs import sind
from ..common.constants import MUNICH_LATITUDE
from ..common.constants import MUNICH_LONGITUDE
from ..common.constants import MUNICH_HEIGHT
from ..utils.core import _assert_numerical_iterable

# Reference Observations in Munich, Germany
from ..utils.wmm import WMM
MAG = WMM(latitude=MUNICH_LATITUDE, longitude=MUNICH_LONGITUDE, height=MUNICH_HEIGHT).magnetic_elements

def _assert_valid_method(method : str, valid_methods : list) -> None:
    if not isinstance(method, str):
        raise TypeError(f"method must be given as a string. Got {type(method)}")
    if method not in valid_methods:
        joint_methods = "', '".join(valid_methods[:-1]) + "' or '" + valid_methods[-1]
        raise ValueError(f"Given method '{method}' is not valid. Try '{joint_methods}'")

class FLAE:
    """
    Fast Linear Attitude Estimator

    Parameters
    ----------
    acc : numpy.ndarray, default: None
        N-by-3 array with measurements of acceleration in in m/s^2
    mag : numpy.ndarray, default: None
        N-by-3 array with measurements of magnetic field in mT
    method : str, default: ``'symbolic'``
        Method used to estimate the attitude. Options are: ``'symbolic'``,
        ``'eig'``, and ``'newton'``.
    weights : np.ndarray, default: [0.5, 0.5]
        Weights used for each sensor. They must add up to 1.
    magnetic_dip : float
        Geomagnetic Inclination angle at local position, in degrees. Defaults
        to magnetic dip of Munich, Germany.

    Raises
    ------
    ValueError
        When estimation method is invalid.

    Examples
    --------
    >>> orientation = FLAE()
    >>> accelerometer = np.array([-0.2853546, 9.657394, 2.0018768])
    >>> magnetometer = np.array([12.32605, -28.825378, -26.586914])
    >>> orientation.estimate(acc=accelerometer, mag=magnetometer)
    array([-0.45447247, -0.69524546,  0.55014011, -0.08622285])

    You can set a different estimation method passing its name to parameter
    ``method``.

    >>> orientation.estimate(acc=accelerometer, mag=magnetometer, method='newton')
    array([ 0.42455176,  0.68971918, -0.58315259, -0.06305803])

    Or estimate all quaternions at once by giving the data to the constructor.
    All estimated quaternions are stored in attribute ``Q``.

    >>> orientation = FLAE(acc=acc_data, mag=mag_data, method='eig')
    >>> orientation.Q.shape
    (1000, 4)

    """
    def __init__(self, acc: np.ndarray = None, mag: np.ndarray = None, method: str = 'symbolic', **kw):
        self.acc: np.ndarray = acc
        self.mag: np.ndarray = mag
        self.method: str = method
        # Reference measurements
        mdip: float = kw.get('magnetic_dip')                       # Magnetic dip, in degrees
        mag_ref: np.ndarray = np.array([MAG['X'], MAG['Y'], MAG['Z']]) if mdip is None else np.array([cosd(mdip), 0., -sind(mdip)])
        mag_ref /= np.linalg.norm(mag_ref)
        acc_ref: np.ndarray = np.array([0.0, 0.0, 1.0])
        self.ref = np.vstack((acc_ref, mag_ref))
        # Weights of sensors
        self.a: np.ndarray = kw.get('weights', np.array([0.5, 0.5]))
        self.a /= np.sum(self.a)
        if self.acc is not None and self.mag is not None:
            self.Q = self._compute_all()

    def _compute_all(self) -> np.ndarray:
        """
        Estimate the quaternions given all data in class Data.

        Class Data must have, at least, `acc` and `mag` attributes.

        Returns
        -------
        Q : numpy.ndarray
            M-by-4 Array with all estimated quaternions, where M is the number
            of samples.

        """
        _assert_numerical_iterable(self.acc, 'Gravitational acceleration vector')
        _assert_numerical_iterable(self.mag, 'Geomagnetic field vector')
        self.acc = np.copy(self.acc)
        self.mag = np.copy(self.mag)
        if self.acc.shape != self.mag.shape:
            raise ValueError("acc and mag are not the same size")
        if self.acc.ndim < 2:
            return self.estimate(self.acc, self.mag)
        num_samples = len(self.acc)
        return np.array([self.estimate(self.acc[t], self.mag[t], method=self.method) for t in range(num_samples)])

    def _P1Hx(self, Hx: np.ndarray) -> np.ndarray:
        return np.array([
            [ Hx[0],   0.0, -Hx[2],  Hx[1]],
            [ 0.0,   Hx[0],  Hx[1],  Hx[2]],
            [-Hx[2], Hx[1], -Hx[0],    0.0],
            [ Hx[1], Hx[2],    0.0, -Hx[0]]])

    def _P2Hy(self, Hy: np.ndarray) -> np.ndarray:
        return np.array([
            [ Hy[1],  Hy[2],    0.0, -Hy[0]],
            [ Hy[2], -Hy[1],  Hy[0],    0.0],
            [ 0.0,    Hy[0],  Hy[1],  Hy[2]],
            [-Hy[0],    0.0,  Hy[2], -Hy[1]]])

    def _P3Hz(self, Hz: np.ndarray) -> np.ndarray:
        return np.array([
            [ Hz[2], -Hz[1],  Hz[0], 0.0],
            [-Hz[1], -Hz[2],    0.0, Hz[0]],
            [ Hz[0],    0.0, -Hz[2], Hz[1]],
            [ 0.0,    Hz[0],  Hz[1], Hz[2]]])

    def estimate(self, acc: np.ndarray, mag: np.ndarray, method: str = 'symbolic') -> np.ndarray:
        """
        Estimate a quaternion with the given measurements and weights.

        Parameters
        ----------
        acc : numpy.ndarray
            Sample of tri-axial Accelerometer.
        mag : numpy.ndarray
            Sample of tri-axial Magnetometer.
        method : str, default: 'symbolic'
            Method used to estimate the attitude. Options are: ``'symbolic'``,
            ``'eig'`` and ``'newton'``.

        Returns
        -------
        q : numpy.ndarray
            Estimated orienation as quaternion.

        Examples
        --------
        >>> accelerometer = np.array([-0.2853546, 9.657394, 2.0018768])
        >>> magnetometer = np.array([12.32605, -28.825378, -26.586914])
        >>> orientation = FLAE()
        >>> orientation.estimate(acc=accelerometer, mag=magnetometer)
        array([-0.45447247, -0.69524546,  0.55014011, -0.08622285])
        >>> orientation.estimate(acc=accelerometer, mag=magnetometer, method='eig')
        array([ 0.42455176,  0.68971918, -0.58315259, -0.06305803])
        >>> orientation.estimate(acc=accelerometer, mag=magnetometer, method='newton')
        array([ 0.42455176,  0.68971918, -0.58315259, -0.06305803])

        """
        _assert_valid_method(method, ['symbolic', 'eig', 'newton'])
        _assert_numerical_iterable(acc, 'Gravitational acceleration vector')
        _assert_numerical_iterable(mag, 'Geomagnetic field vector')
        if acc.size != 3:
            raise ValueError(f"Accelerometer sample must be a (3,) array. Got array of shape {acc.shape}")
        if mag.size != 3:
            raise ValueError(f"Magnetometer sample must be a (3,) array. Got array of shape {mag.shape}")
        Db = np.r_[[acc/np.linalg.norm(acc)], [mag/np.linalg.norm(mag)]]
        H = self.a * Db.T @ self.ref                                # (eq. 42)
        W = self._P1Hx(H[0]) + self._P2Hy(H[1]) + self._P3Hz(H[2])  # (eq. 44)
        if method.lower() == 'eig':
            V, D = np.linalg.eig(W)
            q = D[:, np.argmax(V)]
            return q / np.linalg.norm(q)
        # Polynomial parameters                             (eq. 49)
        t1 = -2*np.trace(H@H.T)
        t2 = -8*np.linalg.det(H.T)
        t3 = np.linalg.det(W)
        if method.lower() == 'newton':
            lam = 1.0
            lam_old = 0.0
            i = 0
            while abs(lam_old-lam) > 1e-8 and i < 6:
                lam_old = lam
                f = lam**4 + t1*lam**2 + t2*lam + t3        # (eq. 48)
                fp = 4*lam**3 + 2*t1*lam + t2               # (eq. 50)
                lam -= f/fp                                 # (eq. 51)
                i += 1
        if method.lower() == 'symbolic':
            # Parameters (eq. 53)
            T0 = 2*t1**3 + 27*t2**2 - 72*t1*t3
            T1 = np.cbrt(T0 + np.emath.sqrt(-4*(t1**2 + 12*t3)**3 + T0**2).real)
            T2 = np.sqrt(-4*t1 + np.cbrt(16)*(t1**2 + 12*t3)/T1 + np.cbrt(4)*T1)
            # Solutions to polynomial (eq. 52)
            L = np.zeros(4)
            L[0] =   T2 - np.sqrt(-T2**2 - 12*t1 - 12*np.sqrt(6)*t2/T2)
            L[1] =   T2 + np.sqrt(-T2**2 - 12*t1 - 12*np.sqrt(6)*t2/T2)
            L[2] = -(T2 + np.sqrt(-T2**2 - 12*t1 + 12*np.sqrt(6)*t2/T2))
            L[3] = -(T2 - np.sqrt(-T2**2 - 12*t1 + 12*np.sqrt(6)*t2/T2))
            L *= 1.0/(2.0*np.sqrt(6))
            lam = L[(np.abs(L-1.0)).argmin()]               # Eigenvalue closest to 1
        N = W - lam*np.identity(4)                          # (eq. 54)
        # Return identity quaternion if N is singular matrix
        if np.linalg.matrix_rank(N[1:, :-1]) != N[1:, :-1].shape[0]:
            return np.array([1., 0., 0., 0.])
        # Solve for N and get fundamental solution
        r = np.linalg.solve(N[1:, :-1], N[1:, -1])          # (eq. 55)
        q = np.array([*r, -1])                              # (eq. 58)
        return q / np.linalg.norm(q)
