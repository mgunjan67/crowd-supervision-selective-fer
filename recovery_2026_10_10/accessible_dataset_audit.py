"""Retrieve only official CREMA-D metadata; no videos, registrations or training."""
import csv
import hashlib
import io
import json
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent / 'external_data/crema_d'
REPO = 'CheyneyComputerScience/CREMA-D'
COMMIT = '1658cd342dff90010aa843eaeebd53610a08b1dc'
CANONICAL = ('angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise')
FILES = ('LICENSE.txt', 'README.md', 'docs/README.md', 'tabulateVotesV2.r',
         'processedResults/tabulatedVotes.csv', 'SentenceFilenames.csv')


def fetch(url):
    req = urllib.request.Request(url, headers={'User-Agent': 'research-metadata-audit/1.0'})
    with urllib.request.urlopen(req, timeout=30) as response:
        data = response.read(16_000_001)
    if len(data) > 16_000_000:
        raise ValueError('Metadata download exceeded the fixed per-file size limit')
    return data


def parse_votes(text):
    source = list(csv.DictReader(io.StringIO(text)))
    clips, counts, response_counts = [], [], []
    for row in source:
        key = int(row[''])
        query_type = key // 100000
        if query_type not in (1, 2, 3):
            raise ValueError('Unrecognized modality encoding')
        votes = [int(row[c]) for c in ('A','D','F','H','N','S')]
        n = int(row['numResponses'])
        if min(votes) < 0 or sum(votes) != n or n <= 0:
            raise ValueError('Invalid annotation counts')
        if abs(max(votes)/n - float(row['agreement'])) > 1e-10:
            raise ValueError('Agreement/count mismatch')
        if query_type != 2:
            continue
        clip = row['fileName']
        if not clip or len(clip.split('_')) != 4:
            raise ValueError('Unexpected clip identifier')
        clips.append(clip)
        counts.append(votes + [0])
        response_counts.append(n)
    if len(set(clips)) != len(clips) or not clips:
        raise ValueError('Missing or duplicate face-only clips')
    votes = np.array(counts, dtype=np.int64)
    return np.array(clips), votes, np.array(response_counts)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    commit = COMMIT
    assert len(commit) == 40 and all(c in '0123456789abcdef' for c in commit)
    sources = {}
    for filename in FILES:
        url = f'https://raw.githubusercontent.com/{REPO}/{commit}/{filename}'
        data = fetch(url)
        destination = OUT / 'upstream' / filename
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(data)
        sources[filename] = dict(url=url, bytes=len(data), sha256=hashlib.sha256(data).hexdigest())
    script = (OUT / 'upstream/tabulateVotesV2.r').read_text()
    assert '100000*testResponses$queryType' in script
    assert 'faceTestResponses <- subset(testResponses,queryType==2)' in script
    clips, counts, n = parse_votes((OUT / 'upstream/processedResults/tabulatedVotes.csv').read_text())
    q7 = counts/n[:,None]
    assert np.allclose(q7.sum(1), 1) and (q7[:,-1] == 0).all()
    np.savez_compressed(OUT / 'face_only_votes.npz', clips=clips, vote_counts=counts,
                        annotation_n=n, q=q7, classes=np.array(CANONICAL))
    report = dict(dataset='CREMA-D', commit=commit, retrieved_utc=datetime.now(timezone.utc).isoformat(),
                  sources=sources, evaluation_unit='complete video clip, not an independently annotated still face',
                  modality='face-only human responses; source queryType 2',
                  face_only_clips=len(clips), actors=len(set(c.split('_')[0] for c in clips)),
                  annotation_count_min=int(n.min()), annotation_count_max=int(n.max()),
                  seven_output_mapping=CANONICAL, surprise_reference_mass=0,
                  unsupported_reference_mass=0,
                  license='Official repository: Open Database License 1.0; contents Database Contents License 1.0',
                  caveats=['No videos downloaded or predictions evaluated',
                           'Repository requests a user-registration form; none submitted',
                           'Not an independent static-image replication',
                           'Cannot test varying unsupported crowd mass: six annotation options are supported',
                           'Video integrity, exact/near-image overlap and participant independence not yet audited'],
                  annotations_sha256=hashlib.sha256((OUT/'face_only_votes.npz').read_bytes()).hexdigest())
    (OUT / 'metadata_audit.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k != 'sources'}, indent=2))


if __name__ == '__main__':
    main()
