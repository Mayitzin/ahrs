#!/usr/bin/env python3
import os
import unittest
import numpy as np
import ahrs

class TestWMM(unittest.TestCase):
    """
    Test Magnetic Field estimation with provided values.

    WMM 2015 uses a CSV test file with values split with semicolons, whereas
    the WMM 2020 uses a TXT file with values split with spaces. The position of
    their values is different. The following table shows their differences:

    =====  =================  =================
    Index  CSV File (WM2015)  TXT File (WM2020)
    =====  =================  =================
    0      date               date
    1      height (km)        height (km)
    2      latitude (deg)     latitude (deg)
    3      longitude (deg)    longitude (deg)
    4      X (nT)             D (deg)
    5      Y (nT)             I (deg)
    6      Z (nT)             H (nT)
    7      H (nT)             X (nT)
    8      F (nT)             Y (nT)
    9      I (deg)            Z (nT)
    10     D (deg)            F (nT)
    11     GV (deg)           dD/dt (deg/year)
    12     Xdot (nT/yr)       dI/dt (deg/year)
    13     Ydot (nT/yr)       dH/dt (nT/year)
    14     Zdot (nT/yr)       dX/dt (nT/year)
    15     Hdot (nT/yr)       dY/dt (nT/year)
    16     Fdot (nT/yr)       dZ/dt (nT/year)
    17     dI/dt (deg/year)   dF/dt (nT/year)
    18     dD/dt (deg/year)
    =====  =================  =================

    """
    def _load_test_values(self, filename: str) -> np.ndarray:
        """Load test values from file.

        Parameters
        ----------
        filename : str
            Path to file with test values.

        Returns
        -------
        data : dict
            Dictionary with the test values.
        """
        if filename.endswith('.csv'):
            data = np.genfromtxt(filename, delimiter=';', skip_header=1)
            if data.shape[1] < 19:
                raise ValueError(f"File '{filename}' has incomplete data.")
            keys = ["date", "height", "latitude", "longitude", "X", "Y", "Z", "H", "F", "I", "D", "GV",
                "dX", "dY", "dZ", "dH", "dF", "dI", "dD"]
        elif filename.endswith('.txt'):
            data = np.genfromtxt(filename, skip_header=1, comments='#')
            if data.shape[1] < 18:
                raise ValueError(f"File '{filename}' has incomplete data.")
            keys = ["date", "height", "latitude", "longitude", "D", "I", "H", "X", "Y", "Z", "F",
                "dD", "dI", "dH", "dX", "dY", "dZ", "dF"]
        else:
            raise TypeError("File type is not supported. Try a csv or txt File.")
        return dict(zip(keys, data.T))

    def setUp(self):
        self.wmm = ahrs.utils.WMM()
        self.base_path = os.path.dirname(os.path.abspath(__file__))

    def tearDown(self) -> None:
        del self.wmm

    def test_wmm2015(self):
        test_values = self._load_test_values(os.path.join(self.base_path, "../ahrs/utils/WMM2015/WMM2015_test_values.csv"))
        num_tests = len(test_values['date'])
        for i in range(num_tests):
            self.wmm.magnetic_field(test_values['latitude'][i], test_values['longitude'][i], test_values['height'][i], date=test_values['date'][i])
            self.assertAlmostEqual(test_values['X'][i], self.wmm.X, 1, 'Expected {:.1f}, result {:.1f}'.format(test_values['X'][i], self.wmm.X))
            self.assertAlmostEqual(test_values['Y'][i], self.wmm.Y, 1, 'Expected {:.1f}, result {:.1f}'.format(test_values['Y'][i], self.wmm.Y))
            self.assertAlmostEqual(test_values['Z'][i], self.wmm.Z, 1, 'Expected {:.1f}, result {:.1f}'.format(test_values['Z'][i], self.wmm.Z))
            self.assertAlmostEqual(test_values['I'][i], self.wmm.I, 2, 'Expected {:.2f}, result {:.2f}'.format(test_values['I'][i], self.wmm.I))
            self.assertAlmostEqual(test_values['D'][i], self.wmm.D, 2, 'Expected {:.2f}, result {:.2f}'.format(test_values['D'][i], self.wmm.D))
            self.assertAlmostEqual(test_values['GV'][i], self.wmm.GV, 2, 'Expected {:.2f}, result {:.2f}'.format(test_values['GV'][i], self.wmm.GV))

    def test_wmm2020(self):
        test_values = self._load_test_values(os.path.join(self.base_path, "../ahrs/utils/WMM2020/WMM2020_TEST_VALUES.txt"))
        num_tests = len(test_values['date'])
        for i in range(num_tests):
            self.wmm.magnetic_field(test_values['latitude'][i], test_values['longitude'][i], test_values['height'][i], date=test_values['date'][i])
            self.assertAlmostEqual(test_values['X'][i], self.wmm.X, 1, 'Expected {:.1f}, result {:.1f}'.format(test_values['X'][i], self.wmm.X))
            self.assertAlmostEqual(test_values['Y'][i], self.wmm.Y, 1, 'Expected {:.1f}, result {:.1f}'.format(test_values['Y'][i], self.wmm.Y))
            self.assertAlmostEqual(test_values['Z'][i], self.wmm.Z, 1, 'Expected {:.1f}, result {:.1f}'.format(test_values['Z'][i], self.wmm.Z))
            self.assertAlmostEqual(test_values['I'][i], self.wmm.I, 2, 'Expected {:.2f}, result {:.2f}'.format(test_values['I'][i], self.wmm.I))
            self.assertAlmostEqual(test_values['D'][i], self.wmm.D, 2, 'Expected {:.2f}, result {:.2f}'.format(test_values['D'][i], self.wmm.D))

if __name__ == '__main__':
    unittest.main()
