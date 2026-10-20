"""Regression checks against primary response semantics, not adapter self-agreement."""
import csv
import unittest
from pathlib import Path
from annotation_adapter import RESPONSE_IDS, LABELS

HERE=Path(__file__).resolve().parent


class MappingTests(unittest.TestCase):
    def test_primary_disgust_surprise_ids(self):
        self.assertEqual(dict(zip(LABELS,RESPONSE_IDS))['disgust'],7)
        self.assertEqual(dict(zip(LABELS,RESPONSE_IDS))['surprise'],8)

    def test_primary_disgust_example(self):
        with (HERE/'video_only_speech_votes.csv').open(newline='') as stream:
            rows={r['filename']:r for r in csv.DictReader(stream)}
        row=rows['02-01-07-01-01-01-01.mp4']
        # S5 original responses: 7,7,7,6,8,7,7,7,7,7.
        self.assertEqual(int(row['count_disgust']),8)
        self.assertEqual(int(row['count_surprise']),1)
        self.assertEqual(int(row['count_fear']),1)


if __name__=='__main__':unittest.main()
