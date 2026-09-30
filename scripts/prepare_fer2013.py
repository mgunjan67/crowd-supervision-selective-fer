"""Validate FER2013 CSV, preserve official splits, and record pixel hashes."""
import csv
import hashlib
import io
import json
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
archive = ROOT / 'data/raw/fer2013-kaggle-v1.zip'
target = ROOT / 'data/fer2013'
target.mkdir(parents=True, exist_ok=True)
with zipfile.ZipFile(archive) as z:
    files = [n for n in z.namelist() if n.endswith('fer2013.csv')]
    if len(files) != 1:
        raise ValueError(f'Expected one fer2013.csv: {files}')
    content = z.read(files[0])
classes = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
canonical = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
splits = defaultdict(lambda: {'images': [], 'labels': [], 'rows': [], 'hashes': []})
counts = defaultdict(Counter)
hash_to_splits = defaultdict(set)
manifest = []
for index, row in enumerate(csv.DictReader(io.StringIO(content.decode('utf-8')))):
    pixels = np.fromstring(row['pixels'], dtype=np.uint8, sep=' ')
    if len(pixels) != 2304:
        raise ValueError(f'Bad image at row {index}')
    usage = row['Usage']
    label = canonical.index(classes[int(row['emotion'])])
    digest = hashlib.sha256(pixels.tobytes()).hexdigest()
    splits[usage]['images'].append(pixels.reshape(48, 48))
    splits[usage]['labels'].append(label)
    splits[usage]['rows'].append(index)
    splits[usage]['hashes'].append(digest)
    counts[usage][canonical[label]] += 1
    hash_to_splits[digest].add(usage)
    manifest.append({'row': index, 'partition': usage, 'class': canonical[label], 'pixel_sha256': digest})
expected = {'Training': 28709, 'PublicTest': 3589, 'PrivateTest': 3589}
for name, size in expected.items():
    assert len(splits[name]['images']) == size
    np.savez_compressed(target / f'{name}.npz', **{k: np.asarray(v) for k,v in splits[name].items()})
overlaps = [m for m in manifest if len(hash_to_splits[m['pixel_sha256']]) > 1]
report = {
    'source': 'https://www.kaggle.com/datasets/deadskull7/fer2013',
    'source_version': 1, 'retrieved': '2026-09-20',
    'provenance': 'Community mirror, not original maintainer. Original Kaggle competition API required authentication (401).',
    'licensing': 'Mirror advertises CC0; this is not independent verification of underlying image rights. Images are excluded from release.',
    'archive_sha256': hashlib.sha256(archive.read_bytes()).hexdigest(),
    'csv_sha256': hashlib.sha256(content).hexdigest(),
    'class_names': canonical, 'counts': dict(counts),
    'cross_partition_duplicate_rows': len(overlaps),
    'cross_partition_duplicate_hashes': sum(len(v)>1 for v in hash_to_splits.values()),
    'test_rows_matching_training_pixels': sum(m['partition']=='PrivateTest' and 'Training' in hash_to_splits[m['pixel_sha256']] for m in manifest),
}
with (ROOT / 'evidence/dataset_manifest.csv').open('w', newline='', encoding='utf-8') as f:
    w = csv.DictWriter(f, fieldnames=list(manifest[0])); w.writeheader(); w.writerows(manifest)
(ROOT / 'evidence/dataset_audit.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
print(json.dumps(report, indent=2))
