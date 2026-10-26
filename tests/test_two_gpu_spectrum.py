import unittest
import numpy as np
from mof_heat_capacity.analysis.two_gpu_spectrum import review_matrix


class SpectrumReviewTests(unittest.TestCase):
    def test_translational_modes_and_stability_are_retained(self):
        spring = np.kron(np.array([[1.,-1.],[-1.,1.]]), np.eye(3))
        report, frequencies = review_matrix(spring, np.ones(2))
        self.assertEqual(report['imaginary_modes_below_minus_1_cm1'], 0)
        self.assertEqual(report['near_zero_modes_within_1_cm1'], 3)
        self.assertEqual(report['relative_asymmetry'], 0.)
        self.assertEqual(report['max_translation_sum_eV_A2'], 0.)
        self.assertTrue(np.isfinite(report['harmonic_cv_J_gK']).all())
        unstable, freqs = review_matrix(-spring, np.ones(2))
        self.assertEqual(unstable['imaginary_modes_below_minus_1_cm1'], 3)
        self.assertIsNone(unstable['harmonic_cv_J_gK'])
