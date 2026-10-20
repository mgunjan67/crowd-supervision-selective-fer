import unittest
from ravdess_metadata import convert_row, original_accuracy


class RavdessAdapterTests(unittest.TestCase):
    def row(self):
        return ('02-01-01-01-01-01-01.mp4','VO','Speech','Neutral','Normal','Kids','R1','A1',
                2,1,1,1,1,1,1, 1,2,3,4,5,6,7,8,9,1)

    def test_canonical_and_unsupported_counts(self):
        self.assertEqual(convert_row(self.row()),[1,1,1,1,2,1,1,1,1])

    def test_bad_filename(self):
        row=list(self.row());row[0]='01-01-01-01-01-01-01.mp4'
        with self.assertRaises(ValueError):convert_row(row)

    def test_missing_and_invalid_responses(self):
        for invalid in (None,0,10,1.5):
            row=list(self.row());row[-1]=invalid
            with self.assertRaises(ValueError):convert_row(row)

    def test_original_calm_neutral_accuracy_not_vote_relabeling(self):
        self.assertEqual(original_accuracy((1,2,3,9),1),(1,1,0,0))
        self.assertEqual(original_accuracy((1,2,3,9),2),(1,1,0,0))
        self.assertEqual(original_accuracy((1,2,3,9),3),(0,0,1,0))
        self.assertEqual(convert_row(self.row()),[1,1,1,1,2,1,1,1,1])

    def test_invalid_original_accuracy_category(self):
        for target in (0,9,None):
            with self.assertRaises(ValueError):original_accuracy((1,2),target)
        with self.assertRaises(ValueError):original_accuracy((10,),1)


if __name__ == '__main__':unittest.main()
