"""Known-answer tests, independent of any favorable scientific result."""
import unittest
import numpy as np
from selector_analysis import accepted, components, score, validate


class SelectorTests(unittest.TestCase):
    def setUp(self):
        self.p = np.array([[1, 0, 0, 0, 0, 0, 0], [.5, .5, 0, 0, 0, 0, 0],
                           [1/7]*7, [.7, .2, .1, 0, 0, 0, 0]])
        self.rows = np.array([5, 3, 2, 1])

    def test_msp(self):
        np.testing.assert_allclose(score(self.p, 'msp'), [1, .5, 1/7, .7])

    def test_margin(self):
        np.testing.assert_allclose(score(self.p, 'margin'), [1, 0, 0, .5])

    def test_entropy_zeros(self):
        result = score(self.p, 'negative_entropy')
        self.assertTrue(np.isfinite(result).all())
        np.testing.assert_allclose(result[:3], [0, -np.log(2), -np.log(7)])

    def test_exact_ties_by_row(self):
        p = np.tile(self.p[1], (3, 1))
        rows = np.array([9, 4, 7])
        for name in ('msp', 'margin', 'negative_entropy'):
            self.assertEqual(rows[accepted(p, rows, name, .5)].tolist(), [4, 7])

    def test_permutation(self):
        perm = np.array([3, 0, 2, 1])
        for name in ('msp', 'margin', 'negative_entropy'):
            left = set(self.rows[accepted(self.p, self.rows, name, .5)])
            right = set(self.rows[perm][accepted(self.p[perm], self.rows[perm], name, .5)])
            self.assertEqual(left, right)

    def test_ceil_and_full(self):
        for name in ('msp', 'margin', 'negative_entropy'):
            self.assertEqual(accepted(self.p, self.rows, name, .2).sum(), 1)
            self.assertEqual(accepted(self.p, self.rows, name, .5).sum(), 2)
            self.assertTrue(accepted(self.p, self.rows, name, 1).all())

    def test_invalid_inputs(self):
        for p in (self.p*2, np.full_like(self.p, np.nan), -self.p, self.p[:, :3]):
            with self.assertRaises(ValueError):
                validate(p, self.rows)
        with self.assertRaises(ValueError):
            validate(self.p, np.array([1, 1, 2, 3]))
        for coverage in (0, -1, 1.1):
            with self.assertRaises(ValueError):
                accepted(self.p, self.rows, 'msp', coverage)
        with self.assertRaises(ValueError):
            score(self.p, 'invented')

    def test_decomposition_and_zero_support(self):
        q = np.zeros((4, 10))
        q[0, 0] = 1
        q[1, 1], q[1, 7] = .5, .5
        q[2, 8] = 1
        q[3, 0], q[3, 1] = .8, .2
        values = components(self.p, q)
        np.testing.assert_allclose(values['risk'], [0, 1, 1, .2])
        np.testing.assert_allclose(values['risk'], values['unsupported'] + values['supported_ambiguity'] + values['excess'])


if __name__ == '__main__':
    unittest.main()
