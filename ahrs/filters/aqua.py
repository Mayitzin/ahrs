# -*- coding: utf-8 -*-
"""
Algebraic Quaternion Algorithm
==============================

Roberto Valenti's Algebraic Quaterion Algorithm (AQUA) [Valenti2015]_ estimates
a quaternion with the algebraic solution of a system from inertial/magnetic
observations, solving `Wahba's Problem <https://en.wikipedia.org/wiki/Wahba%27s_problem>`.

AQUA computes the "tilt" quaternion and the "heading" quaternion separately in
two sub-parts. This avoids the impact of the magnetic disturbances on the roll
and pitch components of the orientation.

AQUA can be used with a complementary filter to fuse the gyroscope data
together with accelerometer and magnetic field readings. The correction part of
the filter is based on the independently estimated quaternions and works for
both IMU (Inertial Measurement Unit) and MARG (Magnetic, Angular Rate, and
Gravity) sensors [Valenti2016]_.

Quaternion as Orientation
-------------------------

Any orientation in a three-dimensional euclidean space of a frame :math:`A`
with respect to a frame :math:`B` can be represented by a unit quaternion
(a.k.a. versor), :math:`\\mathbf{q}\\in\\mathbb{H}^4`, in Hamiltonian space
defined as:

.. math::
    ^B_A\\mathbf{q} = \\begin{bmatrix}q_w\\\\q_x\\\\q_y\\\\q_z\\end{bmatrix} =
    \\begin{bmatrix}
    \\cos\\frac{\\alpha}{2}\\\\e_x\\sin\\frac{\\alpha}{2}\\\\e_y\\sin\\frac{\\alpha}{2}\\\\e_z\\sin\\frac{\\alpha}{2}
    \\end{bmatrix}

where :math:`\\alpha` is the rotation angle and :math:`e` is the unit vector
representing the rotation axis. The conjugate quaternion is used to represent
the orientation of frame :math:`B` relative to frame :math:`A`:

.. math::
    ^B_A\\mathbf{q}^* = \,^A_B\\mathbf{q} = \\begin{bmatrix}q_w\\\\-q_x\\\\-q_y\\\\-q_z\\end{bmatrix}

The orientation quaternion after a sequence of rotations can be easily found by
quaternion multiplication:

.. math::
    ^C_A\\mathbf{q} = \,^C_B\\mathbf{q} \, ^B_A\\mathbf{q}

where the quaternion multiplication of quaternions :math:`\\mathbf{p}` and
:math:`\\mathbf{q}` is computed as:

.. math::
    \\mathbf{pq} =
    \\begin{bmatrix}
        p_w q_w - p_x q_x - p_y q_y - p_z q_z \\\\
        p_w q_x + p_x q_w + p_y q_z - p_z q_y \\\\
        p_w q_y - p_x q_z + p_y q_w + p_z q_x \\\\
        p_w q_z + p_x q_y - p_y q_x + p_z q_w
    \\end{bmatrix}

Unit quaternions in Hamiltonian spaces are often used to operate rotations of
vectors in a 3D euclidean space. A vector
:math:`^A\\mathbf{v}_q=\\begin{bmatrix}v_x & v_y & v_z\\end{bmatrix}^T`
expressed with respect to frame :math:`A` can be represented with respect to
frame :math:`B` as:

.. math::
    ^B\\mathbf{v}_q = \, ^B_A\\mathbf{q} \, ^A\\mathbf{v}_q \, ^B_A\\mathbf{q}^*

where the vector :math:`^A\\mathbf{v}` is rewritten as the pure quaternion
:math:`^A\\mathbf{v}_q=\\begin{bmatrix}0 & v_x & v_y & v_z\\end{bmatrix}^T`, so
that it can multiply with the versor and its conjugate.

The inverse rotation can be achieved by flipping the versors:

.. math::
    ^A\\mathbf{v}_q = \, ^B_A\\mathbf{q}^* \, ^B\\mathbf{v}_q \, ^B_A\\mathbf{q}
    = \, ^A_B\\mathbf{q} \, ^B\\mathbf{v}_q \, ^A_B\\mathbf{q}^*

These rotations can also be expressed within the three-dimensional euclidean
space if we express the rotation as a Direction Cosine Matrix
:math:`\\mathbf{R}\\in SO(3)`:

.. math::
    \\mathbf{R}(^B_A\\mathbf{q}) =
    \\begin{bmatrix}
    q_w^2+q_x^2-q_y^2-q_z^2 & 2(q_xq_y - q_wq_z) & 2(q_xq_z + q_wq_y) \\\\
    2(q_xq_y + q_wq_z) & q_w^2-q_x^2+q_y^2-q_z^2 & 2(q_yq_z - q_wq_x) \\\\
    2(q_xq_z - q_wq_y) & 2(q_wq_x + q_yq_z) & q_w^2-q_x^2-q_y^2+q_z^2
    \\end{bmatrix}

And, because the Direction Cosine Matrix belongs to the Special Orthogonal
Group :math:`SO(3)`, the inverse rotation is simply its transpose:

.. math::
    ^A\\mathbf{v} = \\mathbf{R}(^A_B\\mathbf{q})\,^B\\mathbf{v} = \\mathbf{R}^T(^B_A\\mathbf{q})\,^B\\mathbf{v}

.. note::
    For a more detailed explanation of quaternions and their use in spatial
    rotation see the documentation of the class `Quaternions <../quaternion.html>`_.

Given two quaternions :math:`\\mathbf{p}` and :math:`\\mathbf{q}`, the cosine
of the angle :math:`\\Omega` subtended by the arc between them is equal to the
dot product of the two quaternions.

.. math::
    \\cos\\Omega = \\mathbf{p}\\cdot\\mathbf{q} = p_wq_w + p_xq_x + p_yq_y + p_zq_z

It is easy to see that the dot product of any :math:`\\mathbf{q}=\\begin{bmatrix}q_w & q_x & q_y & q_z\\end{bmatrix}`
and the identity quaternion :math:`\\mathbf{q}_I=\\begin{bmatrix}1 & 0 & 0 & 0\\end{bmatrix}`
is equal to the :math:`q_w` component:

.. math::
    \\mathbf{q}\\cdot\\mathbf{q}_I = q_w

The simple **Linear intERPolation** (LERP) between two quaternions :math:`\\mathbf{p}`
and :math:`\\mathbf{q}` is obtained as:

.. math::
    \\overline{\\mathbf{r}} = (1-\\alpha)\\mathbf{p} + \\alpha\\mathbf{q}

where :math:`\\alpha\\in [0, 1]`. But this does not keep the unit norm, so we
must normalize the resulting interpolation:

.. math::
    \\widehat{\\mathbf{r}} = \\frac{\\overline{\\mathbf{r}}}{\\|\\overline{\\mathbf{r}}\\|}

The **Spherical Linear intERPolation** (SLERP) gives a correct evaluation of
the weighted average of two points lying on a curve. In the case of quaternions,
the points lie on the surface of the 4D sphere (hypersphere).

.. math::
    \\widehat{\\mathbf{r}} = \\frac{\\sin([1-\\alpha]\\Omega)}{\\sin\\Omega}\\mathbf{p} +
    \\frac{\\sin(\\alpha\\Omega)}{\\sin\\Omega}\\mathbf{q}

Quaternion from Earth-Field Observations
----------------------------------------

The local (sensor) frame is labeled as :math:`L`, and the global (Earth) frame
as :math:`G`. The *measured* acceleration, :math:`^L\\mathbf{a}`, and the true
Earth gravitational acceleration, :math:`^G\\mathbf{g}`, are defined as **unit
vectors** [#]_:

.. math::
    \\begin{array}{rcl}
    ^L\\mathbf{a} &=& \\begin{bmatrix}a_x & a_y & a_z\\end{bmatrix}^T \\\\ && \\\\
    ^G\\mathbf{g} &=& \\begin{bmatrix}0 & 0 & 1\\end{bmatrix}^T
    \\end{array}

The *measured* local magnetic field, :math:`^L\\mathbf{m}`, and the true
geomagnetic field, :math:`^G\\mathbf{h}`, are also unit vectors:

.. math::
    \\begin{array}{rcl}
    ^L\\mathbf{m} &=& \\begin{bmatrix}m_x & m_y & m_z\\end{bmatrix}^T \\\\ && \\\\
    ^G\\mathbf{h} &=& \\begin{bmatrix}h_x & h_y & h_z\\end{bmatrix}^T
    \\end{array}

The gyroscopes measure the angular velocity, :math:`^L\\mathbf{\\omega}`,
around the three sensor frame axes:

.. math::
    ^L\\mathbf{\\omega} = \\begin{bmatrix}\\omega_x & \\omega_y & \\omega_z\\end{bmatrix}^T

The measured angular velocities are **not** normalized, unlike the other
sensors, and are assumed to be in *radians per second*.

A straightforward way to formulate the quaternion, :math:`^L_G\\mathbf{q}`,
relating the global frame :math:`G` to the local frame :math:`L`, is through
the inverse orientation which rotates the measured quantities :math:`^L\\mathbf{a}`
and :math:`^L\\mathbf{m}` into the reference quantities :math:`^G\\mathbf{g}`
and :math:`^G\\mathbf{h}`:

.. math::
    \\begin{array}{rcl}
    \\mathbf{R}^T(^L_G\\mathbf{q})\,^L\\mathbf{a} &=& \,^G\\mathbf{g} \\\\ && \\\\
    \\mathbf{R}^T(^L_G\\mathbf{q})\,^L\\mathbf{m} &=& \,^G\\mathbf{h}
    \\end{array}

In the case of a disagreement between the gravitational and magnetometer
readings, the system will not have a solution. To address this problem a
modified equation system is built.

First, the global coordinate frame :math:`G` is aligned with the magnetic North.
The global frame's X-axis points to the same direction as the local magnetic
field (Z-axis remains vertical.)

.. warning::
    This global frame is only *fixed* in case the local magnetic field does not
    change its heading. Thus, no magnetic inference should be present.

Let :math:`^G\\Pi_{zx^+}` be the half-plane which contains all points that lie
in the global XZ-plane such that X is non-negative.

The magnetic reading, when rotated into the global frame, must lie on the
half-plane :math:`^G\\Pi_{zx^+}` to guarantee that the heading will be measured
with respect to the geomagnetic North.

.. math::
    \\begin{array}{lc}
    \\mathbf{R}^T(^L_G\\mathbf{q})\,^L\\mathbf{a} = \,^G\\mathbf{g} &\\\\ & \\\\
    \\mathbf{R}^T(^L_G\\mathbf{q})\,^L\\mathbf{m} \\in \,^G\\Pi_{zx^+} &
    \\end{array}

This way we don't need a-priori knowledge of the direction of Earth's magnetic
field :math:`^G\\mathbf{h}`.

The orientation :math:`^L_G\\mathbf{q}` is decomposed into two auxiliary
quaternions, such that:

.. math::
    \\begin{array}{rcl}
    ^L_G\\mathbf{q} &=& \\mathbf{q}_\\mathrm{acc} \\mathbf{q}_\\mathrm{mag} \\\\ && \\\\
    \\mathbf{R}(^L_G\\mathbf{q}) &=& \\mathbf{R}(\\mathbf{q}_\\mathrm{acc}) \\mathbf{R}(\\mathbf{q}_{\\mathrm{mag}})
    \\end{array}

The quaternion :math:`\\mathbf{q}_{\\mathrm{mag}}` represents a rotation around
the Z-axis to point North only:

.. math::
    \\mathbf{q}_{\\mathrm{mag}} = \\begin{bmatrix}q_{w\\mathrm{mag}} & 0 & 0 & q_{z\\mathrm{mag}}\\end{bmatrix}^T

Quaternion from Accelerometer
-----------------------------

The auxiliary quaternion :math:`\\mathbf{q}_\\mathrm{acc}` is built as a
function of :math:`^L\\mathbf{a}`. The observations of the gravity vector in
the two reference frames allows us to find the quaternion that performs the
transformation between the two representations.

.. math::
    \\begin{array}{rcl}
    \\mathbf{R}(^L_G\\mathbf{q})\,^G\\mathbf{g} &=& \, ^L\\mathbf{a} \\\\
    \\mathbf{R}(\\mathbf{q}_\\mathrm{acc})\\mathbf{R}(\\mathbf{q}_\\mathrm{mag})
    \\begin{bmatrix}0 \\\\ 0 \\\\ 1\\end{bmatrix} &=&
    \\begin{bmatrix}a_x \\\\ a_y \\\\ a_z\\end{bmatrix}
    \\end{array}

The representation of the gravity vector in the global frame :math:`G` only has
a component on the Z-axis. Thus, any rotation about this axis does not produce
any change on it.

.. math::
    \\mathbf{R}(\\mathbf{q}_\\mathrm{acc})
    \\begin{bmatrix}0 \\\\ 0 \\\\ 1\\end{bmatrix} =
    \\begin{bmatrix}a_x \\\\ a_y \\\\ a_z\\end{bmatrix}

The alignment of the gravity vector from global frame into local frame can be
achieved by infinite rotations with definite roll and pitch angles and
arbitrary yaw. To restrict the solutions to a finite number :math:`q_{z\\mathrm{acc}}=0`
is chosen.

This gives four solutions of :math:`\\mathbf{q}_\\mathrm{acc}`. Two are
discarded for having a negative norm, and from the remaining two the one with a
positive :math:`q_w` is taken.

.. math::
    \\mathbf{q}_\\mathrm{acc} =
    \\begin{bmatrix}\\sqrt{\\frac{a_z+1}{2}} & -\\frac{a_y}{\\sqrt{2(a_z+1)}} & \\frac{a_x}{\\sqrt{2(a_z+1)}} & 0\\end{bmatrix}^T

But it has a singularity at :math:`a_z=-1`. Therefore, the final solution has to
be defined as:

.. math::
    \\mathbf{q}_\\mathrm{acc} =
    \\left\\{
    \\begin{array}{ll}
        \\begin{bmatrix}\\sqrt{\\frac{a_z+1}{2}} & -\\frac{a_y}{\\sqrt{2(a_z+1)}} & \\frac{a_x}{\\sqrt{2(a_z+1)}} & 0\\end{bmatrix}^T & \\mathrm{if}\; a_z \\geq 0 \\\\
        \\begin{bmatrix}-\\frac{a_y}{\\sqrt{2(1-a_z)}} & \\sqrt{\\frac{1-a_z}{2}} & 0 & \\frac{a_x}{\\sqrt{2(1-a_z)}}\\end{bmatrix}^T & \\mathrm{if}\; a_z < 0
    \\end{array}
    \\right.

Quaternion from Magnetometer
----------------------------

The auxiliary quaternion :math:`\\mathbf{q}_\\mathrm{mag}` is derived as a
function of :math:`^L\\mathbf{m}` and :math:`\\mathbf{q}_\\mathrm{acc}`.

The quaternion :math:`\\mathbf{q}_\\mathrm{acc}` is used to rotate the magnetic
field vector :math:`^L\\mathbf{m}` into an intermediate frame whoe Z-axis is
the same as the global coordinate frame with X- and Y-axes pointing to unknown
directions due to the unknown yaw of :math:`\\mathbf{q}_\\mathrm{acc}`.

.. math::
    \\mathbf{R}^T(\\mathbf{q}_\\mathrm{acc})\,^L\\mathbf{m} = \\mathbf{l}

where :math:`\\mathbf{l}` is the rotated magnetic field vector. Then, we find
the quaternion :math:\\mathbf{q}_\\mathrm{mag} that rotates the vector
:math:`\\mathbf{l}` into the vector that lies on :math:`^G\\Pi_{zx^+}`:

.. math::
    \\mathbf{R}^T(\\mathbf{q}_\\mathrm{mag})
    \\begin{bmatrix}l_x \\\\ l_y \\\\ l_z\\end{bmatrix} =
    \\begin{bmatrix}\\sqrt{\\Gamma} \\\\ 0 \\\\ l_z\\end{bmatrix}

where :math:`\\Gamma=l_x^2+l_y^2`. This quaternion performs a rotation only
about the global Z-axis to align the X-axis of the intermediate frame with the
X-axis of the global frame without affecting the roll and pitch. Therefore, if
there is a magnetic interference, it would affect only the headin angle.

The solution to find this quaternion ensuring the shortest rotation is:

.. math::
    \\mathbf{q}_\\mathrm{mag} =
    \\begin{bmatrix}\\frac{\\sqrt{\\Gamma + l_x\\sqrt{\\Gamma}}}{\\sqrt{2\\Gamma}} & 0 & 0 &\\frac{l_y}{\\sqrt{2}\\sqrt{\\Gamma+l_x\\sqrt{\\Gamma}}}\\end{bmatrix}^T

This quaternion has a singularity too, but here it happens when :math:`l_x<0`
and :math:`l_y=0`. Eventually, a simliar solution is found restraining the
condition of :math:`l_x`:

.. math::
    \\mathbf{q}_\\mathrm{mag} =
    \\left\\{
    \\begin{array}{ll}
        \\begin{bmatrix}\\frac{\\sqrt{\\Gamma + l_x\\sqrt{\\Gamma}}}{\\sqrt{2\\Gamma}} & 0 & 0 & \\frac{l_y}{\\sqrt{2}\\sqrt{\\Gamma+l_x\\sqrt{\\Gamma}}}\\end{bmatrix}^T & \\mathrm{if}\; l_x \\geq 0 \\\\
        \\begin{bmatrix}\\frac{l_y}{\\sqrt{2}\\sqrt{\\Gamma-l_x\\sqrt{\\Gamma}}} & 0 & 0 & \\frac{\\sqrt{\\Gamma - l_x\\sqrt{\\Gamma}}}{\\sqrt{2\\Gamma}}\\end{bmatrix}^T & \\mathrm{if}\; l_x < 0
    \\end{array}
    \\right.

The generalized quaternion orientation of the global frame relative to the
local frame as the mulitplication of two quaternions :math:`\\mathbf{q}_\\mathrm{acc}`
and :math:`\\mathbf{q}_\\mathrm{mag}`:

.. math::
    ^L_G\\mathbf{q} = \\mathbf{q}_\\mathrm{acc} \, \\mathbf{q}_\\mathrm{mag}

The quaternion :math:`^L_G\\mathbf{q}` does not suffer from the discontinuity
problem of the yaw angle given by the switching formulation of
:math:`\\mathbf{q}_\\mathrm{acc}` thanks to the multiplication with
:math:`\\mathbf{q}_\\mathrm{mag}`, which performs the alignment of the
intermediate frame into the global frame.

Quaternion-Based Complementary Filter
-------------------------------------

A complementary filter fuses attitude estimation in quaternion form from
gyroscope data with accelerometer and magnetometer data in the form of a delta
quaternion.

If only IMU data is provided (gyroscopes and accelerometers), it corrects only
roll and pitch of the attitude. If magnetometer data is also provided (MARG) a
second step is added to the algorithm where a magnetic delta quaternion is
derived to correct the heading of the previous estimation by aligning the
current frame with the magnetic field.

In the **Prediction** step the measured angular velocity is used to compute a
first estimation of the orientation in quaternion form.

Most literature estimating the quaternion derivative from an angular rate
measurement is usually calculated for the one representing the orientation of
the local frame with respect to the global frame.

.. math::
    ^G_L\\dot{\\mathbf{q}}_{\\omega, t_k}=\\frac{1}{2}\,^G_L\\mathbf{q}_{t_{k-1}}\,^L\\mathbf{\\omega}_{q, t_k}

However, Valenti uses the inverse orientation, so the quaternion derivative is
computed using the inverse unit quaternion, which is simply the conjugate:

.. math::
    ^L_G\\dot{\\mathbf{q}}_{\\omega, t_k}= \,^G_L\\dot{\\mathbf{q}}_{\\omega, t_k}^*=-\\frac{1}{2}\,^L\\mathbf{\\omega}_{q, t_k}\,^L_G\\mathbf{q}_{t_{k-1}}

where :math:`^L\\mathbf{\\omega}_{q, t_k}=\\begin{bmatrix}0 & \\omega_x & \\omega_y & \\omega_z\\end{bmatrix}^T`
is the measured angular velocity, in radians per second, arranged as a pure
quaternion at time :math:`t_k`, and :math:`^L_G\\mathbf{q}_{t_{k-1}}` is the
previous estimate of the orientation.

The orientation of the global frame relative to local frame at time :math:`t_k`
can be finally computed by numerically integrating the quaternion derivative
using the sampling period :math:`\\Delta t = t_k - t_{k-1}`.

.. math::
    ^L_G\\mathbf{q}_{\\omega, t_k} = \, ^L_G\\mathbf{q}_{t_{k-1}} + \,^L_G\\dot{\\mathbf{q}}_{\\omega, t_k}\\Delta t

The **Correction** step is based on a multiplicative approach, where the
predicted quaternion :math:`^L_G\\mathbf{q}_\\omega` is corrected by means of
two *delta quaternions*:

.. math::
    \\begin{equation}
    ^L_G\\mathbf{q} = \, ^L_G\\mathbf{q}_\\omega \, \\Delta\\mathbf{q}_\\mathrm{acc} \, \\Delta\\mathbf{q}_\\mathrm{mag}
    \\end{equation}

The delta quaternions are computed and filtered independently by the high-frequency
noise. This correction is divided in two steps: correction of roll and pitch of
the predicted quaternion, and then the correction of the yaw angle if readings
of the magnetic field are provided.

Accelerometer-Based Correction
------------------------------

The inverse predicted quaternion :math:`^G_L\\mathbf{q}_\\omega` is used to
rotate the normalized body frame gravity vector :math:`^L\\mathbf{a}`,
*measured* by the accelerometer, into the global frame:

.. math::
    \\mathbf{R}(^G_L\\mathbf{q}_\\omega)\,^L\\mathbf{a} = \,^G\\mathbf{g}_p

where :math:`^G\\mathbf{g}_p=\\begin{bmatrix}g_x & g_y & g_z\\end{bmatrix}^T`
is the **predicted gravity**, which always has a small deviation from the real
gravity vector :math:`^G\\mathbf{g}=\\begin{bmatrix}0 & 0 & 1\\end{bmatrix}^T`.
We compute the delta quaternion :math:`\\Delta\\mathbf{q}_\\mathrm{acc}` to
rotate :math:`^G\\mathbf{g}` into :math:`^G\\mathbf{g}_p`:

.. math::
    \\mathbf{R}(\\Delta\\mathbf{q}_\\mathrm{acc}) \, ^G\\mathbf{g} = \,^G\\mathbf{g}_p

Similar to the auxiliary quaternions, we find a closed-form solution:

.. math::
    \\Delta\\mathbf{q}_\\mathrm{acc} =
    \\begin{bmatrix}\\sqrt{\\frac{g_z+1}{2}} & - \\frac{g_y}{\\sqrt{2(g_z+1)}} & \\frac{g_x}{\\sqrt{2(g_z+1)}} & 0\\end{bmatrix}^T

This has a singularity at :math:`g_z=-1`, but it can be ignored, because the
value of :math:`g_z` will always be closer to 1.

The delta quaternion is affected by the accelerometer's high frequency noise,
so we scale it down by using an interpolation with the identity quaternion
:math:`\\mathbf{q}_I`. As demonstrated above, the dot product with
:math:`\\mathbf{q}_I` is equal to the :math:`\\Delta q_{w\\mathrm{acc}}`
component of :math:`\\Delta\\mathbf{q}_\\mathrm{acc}`.

If :math:`\\Delta q_{w\\mathrm{acc}}>\\epsilon`, where :math:`\\epsilon` is a
threshold value (default is :math:`\\epsilon=0.9`), a simple LERP is used:

.. math::
    \\widehat{\\Delta\\mathbf{q}}_\\mathrm{acc} = \\frac{\\overline{\\Delta\\mathbf{q}}_\\mathrm{acc}}{\\|\\overline{\\Delta\\mathbf{q}}_\\mathrm{acc}\\|}

The predicted quaternion from gyroscopes is multiplied with the delta
quaternion to correct the roll and pitch components:

.. math::
    ^L_G\\mathbf{q}' = \, ^L_G\\mathbf{q}_\\omega \, \\widehat{\\Delta\\mathbf{q}}_\\mathrm{acc}

The *heading angle* predicted by the gyroscope integration is *not corrected*
in this step.

Magnetometer-Based Correction
-----------------------------

When a magnetic field measurement is provided the second step corrects the
heading component. We use the quaternion inverse of :math:`^L_G\\mathbf{q}'` to
rotate the magnetic field vector :math:`^L\\mathbf{m}` from the body frame into
the world frame.

.. math::
    \\mathbf{R}(^L_G\\mathbf{q}')\,^L\\mathbf{m} = \\mathbf{l}

The delta quaternion :math:`\\Delta\\mathbf{q}_\\mathrm{mag}` rotates the
vector :math:`\\mathbf{l}` into the vector that lies on the XZ-semiplane:

.. math::
    \\mathbf{R}^T(\\Delta\\mathbf{q}_\\mathrm{mag})
    \\begin{bmatrix}l_x \\\\ l_y \\\\ l_z\\end{bmatrix} =
    \\begin{bmatrix}\\sqrt{l_x^2+l_y^2} \\\\ 0 \\\\ l_z\\end{bmatrix}

The solution to the above ensures the shortest rotation:

.. math::
    \\Delta\\mathbf{q}_\\mathrm{mag} = \\begin{bmatrix}
    \\frac{\\sqrt{\\Gamma+l_x\\sqrt{\\Gamma}}}{\\sqrt{2\\Gamma}} & 0 & 0 &
    \\frac{l_y}{\\sqrt{2(\\Gamma+l_x\\sqrt{\\Gamma})}}
    \\end{bmatrix}^T

This delta quaternion is affected by the noise of the magnetometer, which is
also filtered like the :math:`\\Delta\\mathbf{q}_\\mathrm{acc}` switching
between LERP and SLERP according to the same criterion.

Because each delta quaternion is affected independently with different noises,
two different thresholds can be used: :math:`\\alpha` for the accelerometer and
:math:`\\beta` for the magnetometer to obtain :math:`\\widehat{\\Delta\\mathbf{q}}_\\mathrm{mag}`.

Finally, the delta quaternion is multiplied with :math:`^L_G\\mathbf{q}'` to
obtain the orientation of the global frame with respect to the local frame:

.. math::
    ^L_G\\mathbf{q} = \,^L_G\\mathbf{q}' \, \\widehat{\\Delta\\mathbf{q}}_\\mathrm{mag}

Adaptive Gain
-------------

When the vehicle moves with high acceleration, the magnitude and direction of
the total measured acceleration vector are different from gravity, and the
attitude is evaluated using a false reference.

However, the gyroscope readings are not affected by linear acceleration, thus
they can still be used to compute a relatively accurate orientation estimation.

A constant gain fusion algorithm cannot overcome the aforementioned problem if
the optimal gain has been evaluated for static conditions. An adaptive gain can
bw used to tackle this problem.

First a magnitude error :math:`e_m` is defined:

.. math::
    e_m = \\frac{|\\|\,^L\\tilde{a}\\|-g|}{g}

where :math:`\\|\,^L\\hat{a}\\|` is the norm of the measured local frame
acceleration vector before normalization and :math:`g=9.81 \, \\frac{m}{s^2}`.

From the LERP and SLERP definitions, we make the filtering gain :math:`\\alpha`
dependent on the magnitude error :math:`e_m` through the gain factor :math:`f`:

.. math::
    \\alpha = \\overline{\\alpha}f(e_m)

where :math:`\\overline{\\alpha}` is the constant value that gives the best
filtering result in static conditions and :math:`f(e_m)` is what is called the
**gain factor**, which is a piecewise continuous function of the magnitude
error.

This gain factor is equal to :math:`1` when the magnitude of the
non-gravitational acceleration is not high enough to overcome the acceleration
gravity and the value of the error magnitude does not reach the first threshold
:math:`t_1`. If the nongravitational acceleration rises and the error magnitude
exceeds that first threshold, the gain factor decreases linearly with the
increase of the magnitude error until reaching zero for error magnitude equal
to the second threshold :math:`t_2` and over.

.. math::
    f(e_m) =
    \\left\\{
    \\begin{array}{ll}
        1 & \\mathrm{if}\; e_m \\leq t_1 \\\\
        \\frac{t_2-e_m}{t_1} & \\mathrm{if}\; t_1 < e_m < t_2 \\\\
        0 & \\mathrm{if}\; e_m \\geq t_2
    \\end{array}
    \\right.

Empirically, the threshold values giving the best results are :math:`0.1` and
:math:`0.2`.

Filter Initialization
---------------------

The values of the current body-frame acceleration and magnetic field vectors
are used to produce the quaternion representing the initial orientation of the
rigid body in any configuration. Therefore, for the initialization, the filter
does not need any assumption and it is performed in one single step.

.. math::
    ^L_G\\mathbf{q}_0 = \\mathbf{q}_\\mathrm{acc} \\mathbf{q}_\\mathrm{mag}

The bias of a gyroscope's reading is a slow-varying signal considered as low
frequency noise. A low-pass filtering is applied to separate the bias from the
actual angular velocity, but only when the sensor is in a steady-state
condition to avoid filtering useful information.

If the sensor is in the steady-state condition, the bias is updated, otherwise
it is assumed to be equal to the previous step value. The estimated bias is
then subtracted from the gyroscope reading obtaining a bias-free angular
velocity measurement.

Footnotes
---------
.. [#] Any vector :math:`\\mathbf{x}` is a **unit vector** if :math:`\\|\\mathbf{x}\\|=1`.

References
----------
.. [Valenti2015] Valenti, R.G.; Dryanovski, I.; Xiao, J. Keeping a Good
    Attitude: A Quaternion-Based Orientation Filter for IMUs and MARGs. Sensors
    2015, 15, 19302-19330.
    (https://res.mdpi.com/sensors/sensors-15-19302/article_deploy/sensors-15-19302.pdf)
.. [Valenti2016] R. G. Valenti, I. Dryanovski and J. Xiao, "A Linear Kalman
    Filter for MARG Orientation Estimation Using the Algebraic Quaternion
    Algorithm," in IEEE Transactions on Instrumentation and Measurement, vol.
    65, no. 2, pp. 467-481, 2016.
    (https://ieeexplore.ieee.org/document/7345567)

"""

import numpy as np
from ..common.orientation import q_prod, q2R
from ..common.constants import MUNICH_LATITUDE, MUNICH_HEIGHT

# Reference Observations in Munich, Germany
from ..utils.wgs84 import WGS
GRAVITY = WGS().normal_gravity(MUNICH_LATITUDE, MUNICH_HEIGHT)

def _assert_iterables(item, item_name: str = 'iterable'):
    if not isinstance(item, (list, tuple, np.ndarray)):
        raise TypeError(f"{item_name} must be given as an array. Got {type(item)}")

def _assert_same_shapes(item1, item2, item_names: list = None):
    for item in [item1, item2]:
        if not isinstance(item, (list, tuple, np.ndarray)):
            raise TypeError(f"{item} must be an array. Got {type(item)}")
    if item_names is None:
        item_names = ['item1', 'item2']
    item1, tem2 = np.copy(item1), np.copy(item2)
    if item1.shape != item2.shape:
        raise ValueError(f"{item_names[0]} and {item_names[1]} must have the same shape. Got {item1.shape} and {item2.shape}")

def slerp_I(q: np.ndarray, ratio: float, t: float) -> np.ndarray:
    """
    Interpolation with identity quaternion

    Interpolate a given quaternion with the identity quaternion
    :math:`\\mathbf{q}_I=\\begin{pmatrix}1 & 0 & 0 & 0\\end{pmatrix}` to
    scale it to closest versor.

    The interpolation can be with either LERP (Linear) or SLERP (Spherical
    Linear) methods, decided by a threshold value :math:`t`, which lies
    between ``0.0`` and ``1.0``.

    .. math::
        \\mathrm{method} = \\left\\{
        \\begin{array}{ll}
            \\mathrm{LERP} & \\: q_w > t \\\\
            \\mathrm{SLERP} & \\: \\mathrm{otherwise}
        \\end{array}
        \\right.

    For LERP, a simple equation is implemented:

    .. math::
        \\hat{\\mathbf{q}} = (1-\\alpha)\\mathbf{q}_I + \\alpha\\Delta \\mathbf{q}

    where :math:`\\alpha\\in [0, 1]` is the gain characterizing the cut-off
    frequency of the filter. It basically decides how "close" to the given
    quaternion or to the identity quaternion the interpolation is.

    If the scalar part :math:`q_w` of the given quaternion is below the
    threshold :math:`t`, SLERP is used:

    .. math::
        \\hat{\\mathbf{q}} = \\frac{\\sin([1-\\alpha]\\Omega)}{\\sin\\Omega} \\mathbf{q}_I + \\frac{\\sin(\\alpha\\Omega)}{\\sin\\Omega} \\mathbf{q}

    where :math:`\\Omega=\\arccos(q_w)` is the subtended arc between the
    quaternions.

    Parameters
    ----------
    q : numpy.array
        Quaternion to inerpolate with.
    ratio : float
        Gain characterizing the cut-off frequency of the filter.
    t : float
        Threshold deciding interpolation method. LERP when qw>t, otherwise
        SLERP.

    Returns
    -------
    q : numpy.array
        Interpolated quaternion
    """
    q_I = np.array([1.0, 0.0, 0.0, 0.0])
    if q[0]>t:  # LERP
        q = (1.0-ratio)*q_I + ratio*q   # (eq. 50)
    else:       # SLERP
        angle = np.arccos(q[0])
        q = q_I*np.sin(abs(1.0-ratio)*angle)/np.sin(angle) + q*np.sin(ratio*angle)/np.sin(angle)    # (eq. 52)
    q /= np.linalg.norm(q)              # (eq. 51)
    return q

def adaptive_gain(gain: float, a_local: np.ndarray, t1: float = 0.1, t2: float = 0.2, g: float = GRAVITY) -> float:
    """
    Adaptive filter gain factor

    The estimated gain :math:`\\alpha` is dependent on the gain factor
    :math:`f(e_m)`:

    .. math::
        \\alpha = a f(e_m)

    where the magnitude error is defined by the measured acceleration
    :math:`\\mathbf{a}` before normalization and the reference gravity
    :math:`g\\approx 9.809196 \, \\frac{m}{s^2}`:

    .. math::
        e_m = \\frac{|\\|\\mathbf{a}\\|-g|}{g}

    The gain factor is constant and equal to 1 when the magnitude of the
    nongravitational acceleration is not high enough to overcome gravity.

    If nongravitational acceleration rises and :math:`e_m` exceeds the
    first threshold, the gain factor :math:`f` decreases linearly with the
    increase of the magnitude until reaching zero at the second threshold
    and above it.

    .. math::
        f(e_m) =
        \\left\\{
        \\begin{array}{ll}
            1 & \\mathrm{if}\; e_m \\leq t_1 \\\\
            \\frac{t_2-e_m}{t_1} & \\mathrm{if}\; t_1 < e_m < t_2 \\\\
            0 & \\mathrm{if}\; e_m \\geq t_2
        \\end{array}
        \\right.

    Empirically, both thresholds have been defined at ``0.1`` and ``0.2``,
    respectively. They can be, however, changed by setting the values of
    input parameters ``t1`` and ``t2``.

    Parameters
    ----------
    gain : float
        Gain yielding best results in static conditions.
    a_local : numpy.ndarray
        Measured local acceleration vector.
    t1 : float, default: 0.1
        First threshold
    t2 : float, default: 0.2
        Second threshold
    g : float, default: 9.809196
        Reference gravitational acceleration in m/s^2. The estimated gravity in
        Munich, Germany (``9.809196``) is used as default reference value.

    Returns
    -------
    alpha : float
        Gain factor

    Examples
    --------
    >>> from ahrs.filters.aqua import adaptive_gain
    >>> alpha = 0.01    # Best gain in static conditions
    >>> acc = np.array([0.0699, 9.7688, -0.2589])   # Measured acceleration. Quasi-static state.
    >>> adaptive_gain(alpha, acc)
    0.01
    >>> acc = np.array([0.8868, 10.8803, -0.4562])  # New measured acceleration. Slightly above first threshold.
    >>> adaptive_gain(alpha, acc)
    0.008615664547367627
    >>> acc = np.array([4.0892, 12.7667, -2.6047])  # New measured acceleration. Above second threshold.
    >>> adaptive_gain(alpha, acc)
    0.0
    >>> adaptive_gain(alpha, acc, t1=0.2, t2=0.5)   # Same acceleration. New thresholds.
    0.005390131074499384
    >>> adaptive_gain(alpha, acc, t1=0.2, t2=0.5, g=9.82)   # Same acceleration and thresholds. New reference gravity.
    0.005466716107480152

    """
    if t1 > t2:
        raise ValueError("The second threshold should be greater than the first threshold.")
    em = abs(np.linalg.norm(a_local)-g)/g   # Magnitude error (eq. 60)
    f = 0.0
    if t1 < em < t2:
        f = (t2-em)/t1
    if em <= t1:
        f = 1.0
    return f*gain   # Filtering gain (eq. 61)

class AQUA:
    """
    Algebraic Quaternion Algorithm

    Parameters
    ----------
    gyr : numpy.ndarray, default: None
        N-by-3 array with measurements of angular velocity in rad/s
    acc : numpy.ndarray, default: None
        N-by-3 array with measurements of acceleration in m/s^2
    mag : numpy.ndarray, default: None
        N-by-3 array with measurements of magnetic field in nT
    frequency : float, default: 100.0
        Sampling frequency in Herz
    Dt : float, default: 0.01
        Sampling step in seconds. Inverse of sampling frequency. Not required
        if ``frequency`` value is given.
    alpha : float, default: 0.01
        Gain characterizing cut-off frequency for accelerometer quaternion
    beta : float, default: 0.01
        Gain characterizing cut-off frequency for magnetometer quaternion
    threshold : float, default: 0.9
        Threshold to discriminate between LERP and SLERP interpolation
    adaptive : bool, default: False
        Whether to use an adaptive gain for each sample
    q0 : numpy.ndarray, default: None
        Initial orientation, as a versor (normalized quaternion).

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
    alpha : float
        Gain characterizing cut-off frequency for accelerometer quaternion.
    beta : float
        Gain characterizing cut-off frequency for magnetometer quaternion.
    threshold : float
        Threshold to discern between LERP and SLERP interpolation.
    adaptive : bool
        Flag indicating use of adaptive gain.
    q0 : numpy.ndarray
        Initial orientation, as a versor (normalized quaternion).

    """
    def __init__(self, gyr: np.ndarray = None, acc: np.ndarray = None, mag: np.ndarray = None, **kw):
        self.gyr: np.ndarray = gyr
        self.acc: np.ndarray = acc
        self.mag: np.ndarray = mag
        self.frequency: float = kw.get('frequency', 100.0)
        self.frame: str = kw.get('frame', 'NED')
        self.Dt: float = kw.get('Dt', 1.0/self.frequency)
        self.alpha: float = kw.get('alpha', 0.01)
        self.beta: float = kw.get('beta', 0.01)
        self.threshold: float = kw.get('threshold', 0.9)
        self.adaptive: bool = kw.get('adaptive', False)
        self.q0: np.ndarray = kw.get('q0')
        if self.acc is not None:
            self.Q = self._compute_all()

    def _compute_all(self):
        """Estimate all quaternions with given sensor values"""
        _assert_iterables(self.acc, 'Accelerometer data')
        # A single sample was given
        if self.acc.ndim < 2:
            if self.mag is None:
                return self.estimate(self.acc)
            _assert_iterables(self.mag, 'Magnetometer data')
            _assert_same_shapes(self.acc, self.mag, ['acc', 'mag'])
            return self.estimate(self.acc, self.mag)
        # Multiple samples were given
        num_samples = len(self.acc)
        Q = np.zeros((num_samples, 4))
        if self.mag is None:
            Q[0] = self.estimate(self.acc[0]) if self.q0 is None else self.q0.copy()
            if self.gyr is not None:
                _assert_iterables(self.gyr, 'Gyroscope data')
                _assert_same_shapes(self.acc, self.gyr, ['acc', 'gyr'])
                for t in range(1, num_samples):
                    Q[t] = self.updateIMU(Q[t-1], self.gyr[t], self.acc[t])
                return Q
            for t in range(1, num_samples):
                Q[t] = self.estimate(self.acc[t])
            return Q
        Q[0] = self.estimate(self.acc[0], self.mag[0]) if self.q0 is None else self.q0.copy()
        if self.gyr is not None:
            _assert_iterables(self.mag, 'Magnetometer data')
            _assert_iterables(self.gyr, 'Gyroscope data')
            _assert_same_shapes(self.acc, self.mag, ['acc', 'mag'])
            _assert_same_shapes(self.acc, self.gyr, ['acc', 'gyr'])
            for t in range(1, num_samples):
                Q[t] = self.updateMARG(Q[t-1], self.gyr[t], self.acc[t], self.mag[t])
            return Q
        for t in range(1, num_samples):
            Q[t] = self.estimate(self.acc[t], self.mag[t])
        return Q

    def Omega(self, x: np.ndarray) -> np.ndarray:
        """Omega operator.

        Given a vector :math:`\\mathbf{x}\\in\\mathbb{R}^3`, return a
        :math:`4\\times 4` matrix of the form:

        .. math::
            \\boldsymbol\\Omega(\\mathbf{x}) =
            \\begin{bmatrix}
            0 & -\\mathbf{x}^T \\\\ \\mathbf{x} & \\lfloor\\mathbf{x}\\rfloor_\\times
            \\end{bmatrix} =
            \\begin{bmatrix}
            0 & x_1 & x_2 & x_3 \\\\
            -x_1 & 0 & x_3 & -x_2 \\\\
            -x_2 & -x_3 & 0 & x_1 \\\\
            -x_3 & x_2 & -x_1 & 0
            \\end{bmatrix}

        This operator is a simplification to create a 4-by-4 matrix used for
        the product between the angular rate and a quaternion, such that:

        .. math::

            ^L_G\\dot{\\mathbf{q}}_{\\omega, t_k} = \\frac{1}{2}\\boldsymbol\\Omega(\,^L\\mathbf{\\omega}_{q, t_k})\;^L_G\\mathbf{q}_{t_{k-1}}

        .. note::
            The original definition in the article (eq. 39) has an errata
            missing the multiplication with :math:`\\frac{1}{2}`.

        Parameters
        ----------
        x : numpy.ndarray
            Three-dimensional vector representing the angular rate around the
            three axes of the local frame.

        Returns
        -------
        Omega : numpy.ndarray
            Omega matrix.
        """
        return np.array([
            [0.0,    x[0],  x[1],  x[2]],
            [-x[0],   0.0,  x[2], -x[1]],
            [-x[1], -x[2],   0.0,  x[0]],
            [-x[2],  x[1], -x[0],   0.0]])

    def estimate(self, acc: np.ndarray, mag: np.ndarray = None) -> np.ndarray:
        """
        Quaternion from Earth-Field Observations

        Algebraic estimation of a quaternion as a function of an observation of
        the Earth's gravitational and magnetic fields.

        It decomposes the quaternion :math:`\\mathbf{q}` into two auxiliary
        quaternions :math:`\\mathbf{q}_{\\mathrm{acc}}` and
        :math:`\\mathbf{q}_{\\mathrm{mag}}`, such that:

        .. math::
            \\mathbf{q} = \\mathbf{q}_{\\mathrm{acc}}\\mathbf{q}_{\\mathrm{mag}}

        Parameters
        ----------
        acc : numpy.ndarray
            Sample of tri-axial Accelerometer in m/s^2
        mag : numpy.ndarray, default: None
            Sample of tri-axial Magnetometer in mT

        Returns
        -------
        q : numpy.ndarray
            Estimated quaternion.
        """
        ax, ay, az = acc/np.linalg.norm(acc)
        # Quaternion from Accelerometer Readings (eq. 25)
        if az >= 0:
            q_acc = np.array([np.sqrt((az+1)/2), -ay/np.sqrt(2*(az+1)), ax/np.sqrt(2*(az+1)), 0.0])
        else:
            q_acc = np.array([-ay/np.sqrt(2*(1-az)), np.sqrt((1-az)/2.0), 0.0, ax/np.sqrt(2*(1-az))])
        q_acc /= np.linalg.norm(q_acc)
        if mag is not None:
            m_norm = np.linalg.norm(mag)
            if m_norm == 0:
                raise ValueError(f"Invalid geomagnetic field. Its magnitude must be greater than zero.")
            lx, ly, _ = q2R(q_acc).T @ (mag/np.linalg.norm(mag))   # (eq. 26)
            Gamma = lx**2 + ly**2                                   # (eq. 28)
            # Quaternion from Magnetometer Readings (eq. 35)
            if lx >= 0:
                q_mag = np.array([np.sqrt(Gamma+lx*np.sqrt(Gamma))/np.sqrt(2*Gamma), 0.0, 0.0, ly/np.sqrt(2)*np.sqrt(Gamma+lx*np.sqrt(Gamma))])
            else:
                q_mag = np.array([ly/np.sqrt(2)*np.sqrt(Gamma-lx*np.sqrt(Gamma)), 0.0, 0.0, np.sqrt(Gamma-lx*np.sqrt(Gamma))/np.sqrt(2*Gamma)])
            # Generalized Quaternion Orientation (eq. 36)
            q = q_prod(q_acc, q_mag)
            return q/np.linalg.norm(q)
        return q_acc

    def updateIMU(self, q: np.ndarray, gyr: np.ndarray, acc: np.ndarray, dt: float = None) -> np.ndarray:
        """
        Quaternion Estimation with a IMU architecture.

        The estimation is made in two steps: a *prediction* is done with the
        angular rate (gyroscope) to integrate and estimate the current
        orientation; then a *correction* step uses the measured accelerometer
        to infer the expected gravity vector and use it to correct the
        predicted quaternion.

        If the gyroscope data is invalid, it returns the given a-priori
        quaternion. Secondly, if the accelerometer data is invalid the
        predicted quaternion (using gyroscopes) is returned.

        Parameters
        ----------
        q : numpy.ndarray
            A-priori quaternion.
        gyr : numpy.ndarray
            Sample of tri-axial Gyroscope in rad/s.
        acc : numpy.ndarray
            Sample of tri-axial Accelerometer in m/s^2
        dt : float, default: None
            Time step, in seconds, between consecutive Quaternions.

        Returns
        -------
        q : numpy.ndarray
            Estimated quaternion.

        """
        dt = self.Dt if dt is None else dt
        if gyr is None or not np.linalg.norm(gyr) > 0:
            return q
        # PREDICTION
        qDot = 0.5*self.Omega(gyr) @ q                      # Quaternion derivative (eq. 39)
        qInt = q + qDot*dt                             # Quaternion integration (eq. 42)
        qInt /= np.linalg.norm(qInt)
        # CORRECTION
        a_norm = np.linalg.norm(acc)
        if not a_norm > 0:
            return qInt
        a = acc/a_norm
        gx, gy, gz = q2R(qInt).T@a                          # Predicted gravity (eq. 44)
        q_acc = np.array([np.sqrt((gz+1)/2.0), -gy/np.sqrt(2.0*(gz+1)), gx/np.sqrt(2.0*(gz+1)), 0.0])     # Delta Quaternion (eq. 47)
        if self.adaptive:
            self.alpha = adaptive_gain(self.alpha, acc)
        q_acc = slerp_I(q_acc, self.alpha, self.threshold)
        q_prime = q_prod(qInt, q_acc)                       # (eq. 53)
        return q_prime/np.linalg.norm(q_prime)

    def updateMARG(self, q: np.ndarray, gyr: np.ndarray, acc: np.ndarray, mag: np.ndarray, dt: float = None) -> np.ndarray:
        """
        Quaternion Estimation with a MARG architecture.

        The estimation is made in two steps: a *prediction* is done with the
        angular rate (gyroscope) to integrate and estimate the current
        orientation; then a *correction* step uses the measured accelerometer
        and magnetic field to infer the expected geodetic values. Its
        divergence is used to correct the predicted quaternion.

        If the gyroscope data is invalid, it returns the given a-priori
        quaternion. Secondly, if the accelerometer data is invalid the
        predicted quaternion (using gyroscopes) is returned. Finally, if the
        magnetometer measurements are invalid, returns a quaternion corrected
        by the accelerometer only.

        Parameters
        ----------
        q : numpy.ndarray
            A-priori quaternion.
        gyr : numpy.ndarray
            Sample of tri-axial Gyroscope in rad/s.
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
        if gyr is None or not np.linalg.norm(gyr)>0:
            return q
        # PREDICTION
        qDot = 0.5*self.Omega(gyr) @ q                      # Quaternion derivative (eq. 39)
        qInt = q + qDot*dt                             # Quaternion integration (eq. 42)
        qInt /= np.linalg.norm(qInt)
        # CORRECTION
        a_norm = np.linalg.norm(acc)
        if not a_norm > 0:
            return qInt
        a = acc/a_norm
        gx, gy, gz = q2R(qInt).T @ a                        # Predicted gravity (eq. 44)
        # Accelerometer-Based Quaternion
        q_acc = np.array([np.sqrt((gz+1.0)/2.0), -gy/np.sqrt(2.0*(gz+1.0)), gx/np.sqrt(2.0*(gz+1.0)), 0.0])     # Delta Quaternion (eq. 47)
        if self.adaptive:
            self.alpha = adaptive_gain(self.alpha, acc)
        q_acc = slerp_I(q_acc, self.alpha, self.threshold)
        q_prime = q_prod(qInt, q_acc)                       # (eq. 53)
        q_prime /= np.linalg.norm(q_prime)
        # Magnetometer-Based Quaternion
        m_norm = np.linalg.norm(mag)
        if not m_norm > 0:
            return q_prime
        lx, ly, _ = q2R(q_prime).T @ (mag/m_norm)          # World frame magnetic vector (eq. 54)
        Gamma = lx**2 + ly**2                               # (eq. 28)
        q_mag = np.array([np.sqrt(Gamma+lx*np.sqrt(Gamma))/np.sqrt(2*Gamma), 0.0, 0.0, ly/np.sqrt(2*(Gamma+lx*np.sqrt(Gamma)))])    # (eq. 58)
        q_mag = slerp_I(q_mag, self.beta, self.threshold)
        # Generalized Quaternion
        q = q_prod(q_prime, q_mag)                          # (eq. 59)
        return q/np.linalg.norm(q)

    def init_q(self, acc: np.ndarray, mag: np.ndarray = None) -> np.ndarray:
        return self.estimate(acc, mag)
