# -*- coding: utf-8 -*-
"""
TRIAD
=====

Three-Axis Attitude Determination from Vector Observations

References
----------
.. [TRIAD] M.D. Shuster et al. Three-Axis Attitude Determination from
    Vector Observations. Journal of Guidance and Control. Volume 4. Number 1.
    1981. Page 70 (http://www.malcolmdshuster.com/Pub_1981a_J_TRIAD-QUEST_scan.pdf)
.. [Shuster] M.D. Shuster. Deterministic Three-Axis Attitude Determination.
    The Journal of the Astronautical Sciences. Vol 52. Number 3. September
    2004. Pages 405-419 (http://www.malcolmdshuster.com/Pub_2004c_J_dirangs_AAS.pdf)
.. [WikiTRIAD] Triad method in Wikipedia. (https://en.wikipedia.org/wiki/Triad_method)
.. [Garcia] H. Garcia de Marina et al. UAV attitude estimation using
    Unscented Kalman Filter and TRIAD. IEE 2016. (https://arxiv.org/pdf/1609.07436.pdf)
.. [CHall4] Chris Hall. Spacecraft Attitude Dynamics and Control.
    Chapter 4: Attitude Determination. 2003.
    (http://www.dept.aoe.vt.edu/~cdhall/courses/aoe4140/attde.pdf)
.. [iitbTRIAD] IIT Bombay Student Satellite Team. Triad Algorithm.
    (https://www.aero.iitb.ac.in/satelliteWiki/index.php/Triad_Algorithm)
.. [MarkleyTRIAD] F.L. Makley et al. Fundamentals of Spacecraft Attitude
    Determination and Control. 2014. Pages 184-186.

"""

import numpy as np
from ahrs.common.orientation import triad
from ahrs.common.quaternion import sarabandi

if __name__ == "__main__":
    from ahrs.utils import plot
    data = np.genfromtxt('../../tests/repoIMU.csv', dtype=float, delimiter=';', skip_header=2)
    q_ref = data[:, 1:5]
    acc = data[:, 5:8]
    mag = data[:, 11:14]
    num_samples = data.shape[0]
    # Estimate Orientations with IMU
    q = np.tile([1., 0., 0., 0.], (num_samples, 1))
    for i in range(num_samples):
        dcm = triad(acc[i], mag[i])
        q[i] = sarabandi(dcm)
    # Compute Error
    sqe = abs(q_ref - q).sum(axis=1)**2
    # Plot results
    plot(q_ref, q, sqe,
        title="TRIAD estimation",
        subtitles=["Reference Quaternions", "Estimated Quaternions", "Squared Errors"],
        yscales=["linear", "linear", "log"],
        labels=[[], [], ["MSE = {:.3e}".format(sqe.mean())]])
