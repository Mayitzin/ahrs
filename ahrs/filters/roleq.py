# -*- coding: utf-8 -*-
"""
Recursive Optimal Linear Estimator of Quaternion
================================================

This is a modified `OLEQ <./oleq.html>`_, where a recursive estimation of the
attitude is made with the measured angular velocity [Zhou2018]_. This
estimation is set as the initial value for the OLEQ estimation, simplyfing the
rotational operations.

First, the quaternion :math:`\\mathbf{q}_\\omega` is estimated from the angular
velocity, :math:`\\boldsymbol\\omega=\\begin{bmatrix}\\omega_x & \\omega_y &
\\omega_z \\end{bmatrix}^T`, measured by the gyroscopes, in rad/s, at a time
:math:`t` as:

.. math::
    \\mathbf{q}_\\omega = \\Big(\\mathbf{I}_4 + \\frac{\\Delta t}{2}\\boldsymbol\\Omega_t\\Big)\\mathbf{q}_{t-1} =
    \\begin{bmatrix}
    q_w - \\frac{\\Delta t}{2} \\omega_x q_x - \\frac{\\Delta t}{2} \\omega_y q_y - \\frac{\\Delta t}{2} \\omega_z q_z\\\\
    q_x + \\frac{\\Delta t}{2} \\omega_x q_w - \\frac{\\Delta t}{2} \\omega_y q_z + \\frac{\\Delta t}{2} \\omega_z q_y\\\\
    q_y + \\frac{\\Delta t}{2} \\omega_x q_z + \\frac{\\Delta t}{2} \\omega_y q_w - \\frac{\\Delta t}{2} \\omega_z q_x\\\\
    q_z - \\frac{\\Delta t}{2} \\omega_x q_y + \\frac{\\Delta t}{2} \\omega_y q_x + \\frac{\\Delta t}{2} \\omega_z q_w
    \\end{bmatrix}

Then, the attitude is "corrected" through OLEQ using a single multiplication of
its rotation operator:

.. math::
    \\mathbf{q}_\\mathbf{ROLEQ} = \\frac{1}{2}\\Big(\\mathbf{I}_4 + \\sum_{i=1}^na_i\\mathbf{W}_i\\Big)\\mathbf{q}_\\omega

where each :math:`\\mathbf{W}` (one for accelerations and one for magnetic
field) is built from their reference vectors, :math:`D^r`, and measurements,
:math:`D^b`, exactly as in OLEQ:

.. math::
    \\begin{array}{rcl}
    \\mathbf{W} &=& D_x^r\\mathbf{M}_1 + D_y^r\\mathbf{M}_2 + D_z^r\\mathbf{M}_3 \\\\ && \\\\
    \\mathbf{M}_1 &=&
    \\begin{bmatrix}
    D_x^b & 0 & D_z^b & -D_y^b \\\\
    0 & D_x^b & D_y^b & D_z^b \\\\
    D_z^b & D_y^b & -D_x^b & 0 \\\\
    -D_y^b & D_z^b & 0 & -D_x^b
    \\end{bmatrix} \\\\
    \\mathbf{M}_2 &=&
    \\begin{bmatrix}
    D_y^b & -D_z^b & 0 & D_x^b \\\\
    -D_z^b & -D_y^b & D_x^b & 0 \\\\
    0 & D_x^b & D_y^b & D_z^b \\\\
    D_x^b & 0 & D_z^b & -D_y^b
    \\end{bmatrix} \\\\
    \\mathbf{M}_3 &=&
    \\begin{bmatrix}
    D_z^b & D_y^b & -D_x^b & 0 \\\\
    D_y^b & -D_z^b & 0 & D_x^b \\\\
    -D_x^b & 0 & -D_z^b & D_y^b \\\\
    0 & D_x^b & D_y^b & D_z^b
    \\end{bmatrix}
    \\end{array}

It is noticeable that, for OLEQ, a random quaternion was used as a starting
value for an iterative procedure to find the optimal quaternion. Here, that
initial value is now :math:`\\mathbf{q}_\\omega` and a simple product (instead
of a large iterative product) is required.

In this way, the quaternions are recursively computed with much fewer
computations, and the accuracy is maintained.

For this case, however the three sensor data (gyroscopes, accelerometers and
magnetometers) have to be provided, along with the an initial quaternion,
:math:`\\mathbf{q}_0` from which the attitude will be built upon.

References
----------
.. [Zhou2018] Zhou, Z.; Wu, J.; Wang, J.; Fourati, H. Optimal, Recursive and
    Sub-Optimal Linear Solutions to Attitude Determination from Vector
    Observations for GNSS/Accelerometer/Magnetometer Orientation Measurement.
    Remote Sens. 2018, 10, 377.
    (https://www.mdpi.com/2072-4292/10/3/377)

"""

import numpy as np
from ..common.orientation import ecompass
from ..common.mathfuncs import cosd
from ..common.mathfuncs import sind
from ..utils.core import _assert_numerical_iterable

class ROLEQ:
    """
    Recursive Optimal Linear Estimator of Quaternion

    Uses OLEQ to estimate the initial attitude.

    Parameters
    ----------
    gyr : numpy.ndarray, default: None
        N-by-3 array with measurements of angular velocity in rad/s.
    acc : numpy.ndarray, default: None
        N-by-3 array with measurements of acceleration in in m/s^2.
    mag : numpy.ndarray, default: None
        N-by-3 array with measurements of magnetic field in mT.

    Attributes
    ----------
    gyr : numpy.ndarray
        N-by-3 array with N gyroscope samples.
    acc : numpy.ndarray
        N-by-3 array with N accelerometer samples.
    mag : numpy.ndarray
        N-by-3 array with N magnetometer samples.
    frequency : float
        Sampling frequency in Herz
    Dt : float
        Sampling step in seconds. Inverse of sampling frequency.
    Q : numpy.array, default: None
        M-by-4 Array with all estimated quaternions, where M is the number of
        samples. Equal to None when no estimation is performed.

    Raises
    ------
    ValueError
        When dimension of input arrays ``gyr``, ``acc`` or ``mag`` are not
        equal.

    Examples
    --------
    >>> gyr_data.shape, acc_data.shape, mag_data.shape      # NumPy arrays with sensor data
    ((1000, 3), (1000, 3), (1000, 3))
    >>> from ahrs.filters import ROLEQ
    >>> orientation = ROLEQ(gyr=gyr_data, acc=acc_data, mag=mag_data)
    >>> orientation.Q.shape                 # Estimated attitude
    (1000, 4)

    """
    def __init__(self,
        gyr: np.ndarray = None,
        acc: np.ndarray = None,
        mag: np.ndarray = None,
        weights: np.ndarray = None,
        magnetic_ref: np.ndarray = None,
        frame: str = 'NED',
        **kwargs
        ):
        self.gyr: np.ndarray = gyr
        self.acc: np.ndarray = acc
        self.mag: np.ndarray = mag
        self.a: np.ndarray = weights if weights is not None else np.ones(2)
        self.frequency: float = kwargs.get('frequency', 100.0)
        self.Dt: float = kwargs.get('Dt', (1.0/self.frequency) if self.frequency else 0.01)
        self.q0: np.ndarray = kwargs.get('q0')
        self.frame: str = frame
        # Reference measurements
        self._set_reference_frames(magnetic_ref, self.frame)
        self._assert_validity_of_inputs()
        # Estimate all quaternions if data is given
        if self.acc is not None and self.gyr is not None and self.mag is not None:
            self.Q = self._compute_all()

    def _set_reference_frames(self, mref: float, frame: str = 'NED'):
        if not isinstance(frame, str):
            raise TypeError(f"'frame' must be a string. Got {type(frame)}.")
        if frame.upper() not in ['NED', 'ENU']:
            raise ValueError(f"Invalid frame '{frame}'. Try 'NED' or 'ENU'")
        #### Magnetic Reference Vector ####
        if mref is None:
            # Local magnetic reference of Munich, Germany
            from ..common.constants import MUNICH_LATITUDE, MUNICH_LONGITUDE, MUNICH_HEIGHT
            from ..utils.wmm import WMM
            wmm = WMM(latitude=MUNICH_LATITUDE, longitude=MUNICH_LONGITUDE, height=MUNICH_HEIGHT)
            cd, sd = cosd(wmm.I), sind(wmm.I)
            self.m_ref = np.array([sd, 0.0, cd]) if frame.upper() == 'NED' else np.array([0.0, cd, -sd])
        elif isinstance(mref, bool):
            raise TypeError(f"'mref' must be a float or numpy.ndarray. Got {type(mref)}.")
        elif isinstance(mref, (int, float)):
            # Use given magnetic dip angle (in degrees)
            cd, sd = cosd(mref), sind(mref)
            self.m_ref = np.array([sd, 0.0, cd]) if frame.upper() == 'NED' else np.array([0.0, cd, -sd])
        elif isinstance(mref, (list, tuple, np.ndarray)):
            # Magnetic reference is given as a vector
            self.m_ref = np.copy(mref)
        else:
            raise TypeError(f"Magnetic reference must be float, int, list, tuple or numpy.ndarray. Got {type(mref)}.")
        if self.m_ref.shape != (3,):
            raise ValueError(f"Magnetic reference vector must be of shape (3,). Got {self.m_ref.shape}.")
        if not any(self.m_ref):
            raise ValueError("Magnetic reference vector must contain non-zero values.")
        self.m_ref /= np.linalg.norm(self.m_ref)
        #### Gravitational Reference Vector ####
        self.a_ref = np.array([0.0, 0.0, -1.0]) if frame.upper() == 'NED' else np.array([0.0, 0.0, 1.0])

    def _assert_validity_of_inputs(self):
        """Asserts the validity of the inputs."""
        # Assert floats
        for item in ["frequency", "Dt"]:
            if isinstance(self.__getattribute__(item), bool):
                raise TypeError(f"Parameter '{item}' must be numeric.")
            if not isinstance(self.__getattribute__(item), (int, float)):
                raise TypeError(f"Parameter '{item}' is not a non-zero number.")
            if self.__getattribute__(item) <= 0.0:
                raise ValueError(f"Parameter '{item}' must be a non-zero number.")
        # Assert arrays
        for item in ['gyr', 'acc', 'mag', 'a', 'm_ref', 'a_ref', 'q0']:
            if self.__getattribute__(item) is not None:
                if isinstance(self.__getattribute__(item), bool):
                    raise TypeError(f"Parameter '{item}' must be an array of numeric values.")
                _assert_numerical_iterable(self.__getattribute__(item), item)
                self.__setattr__(item, np.copy(self.__getattribute__(item)))
        if self.acc is not None and self.mag is None:
            raise ValueError("If 'acc' is given, 'mag' must also be given.")
        if self.mag is not None and self.acc is None:
            raise ValueError("If 'mag' is given, 'acc' must also be given.")
        if self.q0 is not None:
            if self.q0.shape != (4,):
                raise ValueError(f"Parameter 'q0' must be an array of shape (4,). It is {self.q0.shape}.")
            if not np.allclose(np.linalg.norm(self.q0), 1.0):
                raise ValueError(f"Parameter 'q0' must be a versor (norm equal to 1.0). Its norm is equal to {np.linalg.norm(self.q0)}.")
        # Assert weights
        if self.a.shape != (2,):
            raise ValueError(f"Dimension of 'weights' must be (2,). Got {self.a.shape}.")
        for item in self.a:
            if not isinstance(item, (int, float)):
                raise TypeError(f"'weights' must be an array of numeric values. Got {type(item)}.")
            if item < 0.0:
                raise ValueError(f"'weights' must be non-negative. Got {item}.")
        if not any(self.a > 0):
            raise ValueError("'weights' must contain positive values.")

    def _compute_all(self) -> np.ndarray:
        """
        Estimate the quaternions given all data.

        Attributes ``gyr``, ``acc`` and ``mag`` must contain data.

        Returns
        -------
        Q : numpy.ndarray
            M-by-4 Array with all estimated quaternions, where M is the number
            of samples.

        """
        if self.acc.shape != self.gyr.shape:
            raise ValueError("acc and gyr are not the same size")
        if self.acc.shape != self.mag.shape:
            raise ValueError("acc and mag are not the same size")
        num_samples = np.atleast_2d(self.acc).shape[0]
        if num_samples < 2:
            raise ValueError("ROLEQ needs at least 2 samples of each sensor")
        Q = np.zeros((num_samples, 4))
        Q[0] = ecompass(-self.acc[0], self.mag[0], frame=self.frame, representation='quaternion') if self.q0 is None else self.q0
        for t in range(1, num_samples):
            Q[t] = self.update(Q[t-1], self.gyr[t], self.acc[t], self.mag[t])
        return Q

    def attitude_propagation(self, q: np.ndarray, omega: np.ndarray, dt: float) -> np.ndarray:
        """
        Attitude estimation from previous quaternion and current angular velocity.

        .. math::
            \\mathbf{q}_\\omega = \\Big(\\mathbf{I}_4 + \\frac{\\Delta t}{2}\\boldsymbol\\Omega_t\\Big)\\mathbf{q}_{t-1} =
            \\begin{bmatrix}
            q_w - \\frac{\\Delta t}{2} \\omega_x q_x - \\frac{\\Delta t}{2} \\omega_y q_y - \\frac{\\Delta t}{2} \\omega_z q_z\\\\
            q_x + \\frac{\\Delta t}{2} \\omega_x q_w - \\frac{\\Delta t}{2} \\omega_y q_z + \\frac{\\Delta t}{2} \\omega_z q_y\\\\
            q_y + \\frac{\\Delta t}{2} \\omega_x q_z + \\frac{\\Delta t}{2} \\omega_y q_w - \\frac{\\Delta t}{2} \\omega_z q_x\\\\
            q_z - \\frac{\\Delta t}{2} \\omega_x q_y + \\frac{\\Delta t}{2} \\omega_y q_x + \\frac{\\Delta t}{2} \\omega_z q_w
            \\end{bmatrix}

        Parameters
        ----------
        q : numpy.ndarray
            A-priori quaternion.
        omega : numpy.ndarray
            Angular velocity, in rad/s.
        dt : float
            Time step, in seconds, between consecutive Quaternions.

        Returns
        -------
        q : numpy.ndarray
            Attitude as a quaternion.
        """
        Omega_t = np.array([
            [0.0,  -omega[0], -omega[1], -omega[2]],
            [omega[0],   0.0,  omega[2], -omega[1]],
            [omega[1], -omega[2],   0.0,  omega[0]],
            [omega[2],  omega[1], -omega[0],   0.0]])
        q_omega = (np.identity(4) + 0.5*dt*Omega_t) @ q    # (eq. 37)
        return q_omega/np.linalg.norm(q_omega)

    def WW(self, Db, Dr):
        """
        W Matrix

        .. math::
            \\mathbf{W} = D_x^r\\mathbf{M}_1 + D_y^r\\mathbf{M}_2 + D_z^r\\mathbf{M}_3

        Parameters
        ----------
        Db : numpy.ndarray
            Normalized tri-axial observations vector.
        Dr : numpy.ndarray
            Normalized tri-axial reference vector.

        Returns
        -------
        W_matrix : numpy.ndarray
            W Matrix.
        """
        bx, by, bz = Db
        rx, ry, rz = Dr
        M1 = np.array([
            [bx, 0.0, bz, -by],
            [0.0, bx, by, bz],
            [bz, by, -bx, 0.0],
            [-by, bz, 0.0, -bx]])       # (eq. 18a)
        M2 = np.array([
            [by, -bz, 0.0, bx],
            [-bz, -by, bx, 0.0],
            [0.0, bx, by, bz],
            [bx, 0.0, bz, -by]])        # (eq. 18b)
        M3 = np.array([
            [bz, by, -bx, 0.0],
            [by, -bz, 0.0, bx],
            [-bx, 0.0, -bz, by],
            [0.0, bx, by, bz]])         # (eq. 18c)
        return rx*M1 + ry*M2 + rz*M3    # (eq. 20)

    def oleq(self, acc: np.ndarray, mag: np.ndarray, q_omega: np.ndarray) -> np.ndarray:
        """
        OLEQ with a single rotation by R.

        Parameters
        ----------
        acc : numpy.ndarray
            Sample of tri-axial Accelerometer.
        mag : numpy.ndarray
            Sample of tri-axial Magnetometer.
        q_omega : numpy.ndarray
            Preceding quaternion estimated with angular velocity.

        Returns
        -------
        q : numpy.ndarray
            Final quaternion.

        """
        _assert_numerical_iterable(acc, 'Gravitational acceleration vector')
        _assert_numerical_iterable(mag, 'Geomagnetic field vector')
        a_norm = np.linalg.norm(acc)
        m_norm = np.linalg.norm(mag)
        if not a_norm > 0 or not m_norm > 0:    # handle NaN
            return q_omega
        acc = np.copy(acc) / np.linalg.norm(acc)
        mag = np.copy(mag) / np.linalg.norm(mag)
        sum_aW = self.a[0]*self.WW(acc, self.a_ref) + self.a[1]*self.WW(mag, self.m_ref)    # (eq. 31)
        R = 0.5*(np.identity(4) + sum_aW)       # (eq. 33)
        q = R @ q_omega                         # (eq. 25)
        return q / np.linalg.norm(q)

    def update(self, q: np.ndarray, gyr: np.ndarray, acc: np.ndarray, mag: np.ndarray, dt: float = None) -> np.ndarray:
        """
        Update Attitude with a Recursive OLEQ

        Parameters
        ----------
        q : numpy.ndarray
            A-priori quaternion.
        gyr : numpy.ndarray
            Sample of angular velocity in rad/s
        acc : numpy.ndarray
            Sample of tri-axial Accelerometer in m/s^2
        mag : numpy.ndarray
            Sample of tri-axial Magnetometer in mT
        dt : float, default: None
            Time step, in seconds, between consecutive Quaternions.

        Returns
        -------
        q : numpy.ndarray
            Estimated quaternion.

        """
        dt = self.Dt if dt is None else dt
        q_g = self.attitude_propagation(q, gyr, dt)     # Quaternion from previous quaternion and angular velocity
        q = self.oleq(acc, mag, q_g)                    # Second stage: Estimate with OLEQ
        return q
