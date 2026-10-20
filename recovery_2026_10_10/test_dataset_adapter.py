import unittest
from accessible_dataset_audit import parse_votes


class DatasetAdapterTests(unittest.TestCase):
    def test_only_visual_votes_and_canonical_mapping(self):
        text = ',A,D,F,H,N,S,fileName,numResponses,agreement\n'
        text += '100001,0,0,0,1,1,0,1001_IEO_HAP_LO,2,0.5\n'
        text += '200001,1,2,3,4,5,6,1001_IEO_HAP_LO,21,0.2857142857142857\n'
        text += '300001,0,0,0,1,1,0,1001_IEO_HAP_LO,2,0.5\n'
        clips, counts, n = parse_votes(text)
        self.assertEqual(clips.tolist(), ['1001_IEO_HAP_LO'])
        self.assertEqual(counts.tolist(), [[1,2,3,4,5,6,0]])
        self.assertEqual(n.tolist(), [21])

    def test_reject_inconsistent_counts(self):
        text = ',A,D,F,H,N,S,fileName,numResponses,agreement\n200001,1,0,0,0,0,0,1001_IEO_HAP_LO,2,1\n'
        with self.assertRaises(ValueError):
            parse_votes(text)

    def test_reject_duplicate_visual_clip(self):
        text = ',A,D,F,H,N,S,fileName,numResponses,agreement\n'
        row = '200001,1,0,0,0,0,0,1001_IEO_HAP_LO,1,1\n'
        with self.assertRaises(ValueError):
            parse_votes(text + row + row)


if __name__ == '__main__':
    unittest.main()
