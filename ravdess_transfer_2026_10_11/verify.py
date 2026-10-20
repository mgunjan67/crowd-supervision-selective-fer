"""Independent recomputation; deliberately does not import analysis/common code."""
import csv
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import openpyxl

HERE = Path(__file__).resolve().parent
PAPER = HERE.parent
LABELS = ('angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise', 'calm', 'none')
SEEDS = (17, 42, 89)
REFS = ('raw9', 'neutral_calm_merged')


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4194304), b''):
            h.update(block)
    return h.hexdigest()


def read_csv(name):
    with (HERE / 'results' / name).open(newline='') as stream:
        return list(csv.DictReader(stream))


def close(actual, expected, label, errors):
    if isinstance(expected, str):
        assert actual == expected, (label, actual, expected)
        return
    a = float(actual)
    e = float(expected)
    assert math.isfinite(a) and math.isfinite(e), label
    error = abs(a - e)
    errors.append(error)
    assert error <= 2e-10, (label, a, e, error)


def main():
    started = time.monotonic()
    freeze = json.loads((HERE / 'freeze.json').read_text())
    amendment = json.loads((HERE / 'amendment_001.json').read_text())
    assert amendment['prior_freeze_sha256'] == digest(HERE / 'freeze.json')
    for name, expected in freeze['source_hashes'].items():
        expected = amendment['source_hashes'].get(name, expected)
        assert digest(PAPER / name) == expected, name
    for name, expected in {**amendment['preserved_originals'], **amendment['source_hashes']}.items():
        assert digest(PAPER / name) == expected, name
    frozen_sha = digest(HERE / 'freeze.json')
    for item in freeze['checkpoints']:
        assert digest(PAPER / item['path']) == item['sha256']

    # Reconstruct counts directly from the primary workbook, with a different
    # implementation from the adapter. No call to the previous mapping code.
    workbook_path = PAPER / 'recovery_2026_10_10/external_data/ravdess/pone.0196391.s008.xlsx'
    book = openpyxl.load_workbook(workbook_path, read_only=True, data_only=True)
    votes = {}
    id_to_position = {5: 0, 7: 1, 6: 2, 3: 3, 1: 4, 4: 5, 8: 6, 2: 7, 9: 8}
    checked_rows = 0
    for row in book['Speech_EmoCategory'].iter_rows(min_row=4, values_only=True):
        checked_rows += 1
        if int(row[8]) != 2:
            continue
        assert row[1] == 'VO' and int(row[9]) == 1
        factors = [int(token) for token in Path(row[0]).stem.split('-')]
        assert factors == [int(v) for v in row[8:15]]
        counts = np.zeros(9, dtype=int)
        for response in row[15:25]:
            counts[id_to_position[int(response)]] += 1
        assert counts.sum() == 10
        votes[row[0]] = counts
    book.close()
    assert checked_rows == 4320 and len(votes) == 1440
    names = np.array(sorted(votes))
    actors = np.array([int(Path(name).stem.split('-')[-1]) for name in names])
    q = np.array([votes[name] for name in names], dtype=float) / 10
    merged = q.copy()
    merged[:, 4] += merged[:, 7]
    merged[:, 7] = 0
    references = {'raw9': q, 'neutral_calm_merged': merged}
    with (HERE / 'video_only_speech_votes.csv').open(newline='') as stream:
        converted = {r['filename']: r for r in csv.DictReader(stream)}
    for name, counts in votes.items():
        for i, label in enumerate(LABELS):
            assert int(converted[name]['count_' + label]) == counts[i]
            assert float(converted[name]['q_' + label]) == counts[i] / 10

    counts = np.zeros(1440, dtype=int)
    expected_frame_ids = []
    expected_frame_indices = []
    preparation_status = {}
    raw_video_bytes = 0
    for actor in range(1, 25):
        integrity = json.loads((HERE / f'media/Actor_{actor:02d}_integrity.json').read_text())
        report = json.loads((HERE / f'cache/Actor_{actor:02d}.json').read_text())
        assert report['freeze_sha256'] == frozen_sha
        assert report['source_integrity_sha256'] == digest(HERE / f'media/Actor_{actor:02d}_integrity.json')
        assert report['cache_sha256'] == digest(HERE / f'cache/Actor_{actor:02d}.npz')
        assert integrity['video_only_speech_clips'] == len(integrity['clips']) == 60
        for clip in integrity['clips']:
            path = HERE / clip['path']
            assert path.stat().st_size == clip['bytes']
            assert digest(path) == clip['sha256']
            raw_video_bytes += clip['bytes']
        by_id = {r['clip_id']: r for r in report['clips']}
        assert len(by_id) == 60 and all(r['actor'] == actor for r in by_id.values())
        for frame in report['frames']:
            preparation_status[frame['status']] = preparation_status.get(frame['status'], 0) + 1
            info = by_id[frame['clip_id']]
            n = info['frame_count']
            requested = min(n - 1, int((frame['bin_id'] + .5) * n / 8))
            assert requested == frame['frame_index']
            if frame['status'] == 'valid':
                assert frame['decoded_frame_index'] == requested and frame['detections'] == 1
                left, top, right, bottom = frame['box']
                assert 0 <= left < right <= frame['width'] and 0 <= top < bottom <= frame['height']
                counts[frame['clip_id']] += 1
                expected_frame_ids.append(frame['clip_id'])
                expected_frame_indices.append(frame['frame_index'])
        with np.load(HERE / f'cache/Actor_{actor:02d}.npz', allow_pickle=False) as cache:
            assert cache['images'].dtype == np.uint8 and cache['images'].shape[1:] == (224, 224)
            assert cache['images'].shape[0] == report['valid_frames']
        for clip_id, clip in by_id.items():
            assert counts[clip_id] == clip['valid_frames'] and clip['eligible'] == (counts[clip_id] >= 4)
    eligible = counts >= 4
    frame_ids = np.array(expected_frame_ids)
    frame_indices = np.array(expected_frame_indices)
    boot = np.load(HERE / 'bootstrap_actor_counts.npz', allow_pickle=False)['counts']
    assert boot.shape == (5000, 24)
    assert np.array_equal(boot, np.random.default_rng(11011).multinomial(24, np.full(24, 1 / 24), size=5000))

    models = {}
    aggregate_errors = []
    for item in freeze['checkpoints']:
        key = (item['backbone'], item['condition'], item['seed'], item['state'])
        stem = f'{key[0]}_{key[1]}_seed{key[2]}_{key[3]}'
        path = HERE / 'predictions' / (stem + '.npz')
        info = json.loads(path.with_suffix('.json').read_text())
        assert digest(path) == info['output_sha256'] and info['freeze_sha256'] == frozen_sha
        assert not info['pilot_only']
        with np.load(path, allow_pickle=False) as source:
            p = source['probabilities']
            frames = source['frame_probabilities']
            assert np.array_equal(source['filenames'], names) and np.array_equal(source['actors'], actors)
            assert np.array_equal(source['valid'], eligible) and np.array_equal(source['valid_frame_count'], counts)
            assert np.array_equal(source['frame_clip_ids'], frame_ids)
            assert np.array_equal(source['frame_indices'], frame_indices)
            assert frames.shape == (len(frame_ids), 7) and np.isfinite(frames).all()
            assert np.all(frames >= 0) and np.allclose(frames.sum(1), 1, atol=1e-6, rtol=0)
        sums = np.zeros((1440, 7), dtype=float)
        np.add.at(sums, frame_ids, frames.astype(float))
        recovered = sums[eligible] / counts[eligible, None]
        error = float(np.max(np.abs(p[eligible] - recovered)))
        assert error <= 1e-14 and np.isnan(p[~eligible]).all()
        aggregate_errors.append(error)
        order = np.sort(p[eligible], axis=1)
        scores = {'msp': order[:, -1], 'margin': order[:, -1] - order[:, -2]}
        entropy_terms = np.zeros_like(p[eligible])
        nonzero = p[eligible] > 0
        entropy_terms[nonzero] = p[eligible][nonzero] * np.log(p[eligible][nonzero])
        scores['negative_entropy'] = entropy_terms.sum(1)
        valid_ids = np.flatnonzero(eligible)
        ranks = {label: np.array(sorted(valid_ids, key=lambda j: (-float(value[np.searchsorted(valid_ids, j)]), str(names[j]))))
                 for label, value in scores.items()}
        choice = p[eligible].argmax(1)
        losses = {}
        for ref, annotation in references.items():
            chosen = annotation[valid_ids, choice]
            supported = annotation[eligible, :7].sum(1)
            maximum = annotation[eligible, :7].max(1)
            data = np.full((1440, 4), np.nan)
            data[eligible] = np.array([1 - chosen, 1 - supported, supported - maximum, maximum - chosen]).T
            assert np.max(np.abs(data[eligible, 0] - data[eligible, 1:].sum(1))) < 1e-14
            losses[ref] = data
        # Alternative weighted-rank calculation: clip cumulative accepted copies
        # at the target, then difference consecutive cumulative counts.
        ranked = ranks['msp']
        weights = boot[:, actors[ranked] - 1]
        cumulative = np.cumsum(weights, axis=1, dtype=np.int32)
        cumulative = np.minimum(cumulative, 1152)
        selected_copies = np.diff(cumulative, axis=1, prepend=np.zeros((5000, 1), dtype=int))
        available = weights.sum(1) >= 1152
        boot_risks = {}
        for ref in REFS:
            values = selected_copies @ losses[ref][ranked, 0] / 1152
            values[~available] = np.nan
            boot_risks[ref] = values
        models[key] = dict(ranks=ranks, losses=losses, bootstrap=boot_risks)

    errors = []
    measured = {}
    cells = read_csv('clip_results.csv')
    assert len(cells) == 936
    for row in cells:
        key = (row['backbone'], row['condition'], int(row['seed']), row['state'])
        model = models[key]
        n = int(eligible.sum()) if row['score'] == 'all_eligible' else math.ceil(float(row['requested_coverage']) * 1440)
        valid = n <= eligible.sum()
        index = np.flatnonzero(eligible) if row['score'] == 'all_eligible' else model['ranks'][row['score']][:n]
        close(row['population_n'], 1440, 'population', errors)
        close(row['eligible_n'], eligible.sum(), 'eligibility', errors)
        close(row['status'], 'available' if valid else 'unavailable_insufficient_eligible_clips', 'availability', errors)
        if valid:
            close(row['accepted_n'], n, 'accepted count', errors)
            close(row['achieved_coverage'], n / 1440, 'coverage', errors)
            values = model['losses'][row['reference']][index].mean(0)
            for field, value in zip(('risk', 'unsupported', 'supported_ambiguity', 'excess'), values):
                close(row[field], value, field, errors)
        else:
            for field in ('accepted_n', 'achieved_coverage', 'risk', 'unsupported', 'supported_ambiguity', 'excess'):
                assert row[field] == ''
        measured[(*key, row['reference'], row['score'], row['requested_coverage'])] = row
    assert len(measured) == len(cells)

    seeds = read_csv('paired_seed_contrasts.csv')
    means = read_csv('paired_mean_contrasts.csv')
    fixed = read_csv('fixed_image_comparisons.csv')
    intervals = read_csv('actor_intervals.csv')
    assert (len(seeds), len(means), len(fixed), len(intervals)) == (576, 192, 192, 16)
    values_by_group = {}
    for row in seeds:
        prefix = (row['backbone'], row['state'], row['reference'], row['score'], row['requested_coverage'], row['control'])
        soft = measured[row['backbone'], 'soft', int(row['seed']), row['state'], row['reference'], row['score'], row['requested_coverage']]
        control = measured[row['backbone'], row['control'], int(row['seed']), row['state'], row['reference'], row['score'], row['requested_coverage']]
        if soft['status'] == control['status'] == 'available':
            value = 100 * (float(soft['risk']) - float(control['risk']))
            close(row['difference_pp'], value, 'paired seed contrast', errors)
            values_by_group.setdefault(prefix, []).append(value)
        else:
            assert row['status'] == 'unavailable' and row['difference_pp'] == ''
    for row in means:
        key = tuple(row[f] for f in ('backbone', 'state', 'reference', 'score', 'requested_coverage', 'control'))
        values = values_by_group.get(key, [])
        close(row['paired_seeds'], len(values), 'paired seeds', errors)
        if len(values) == 3:
            for field, value in [('mean_difference_pp', np.mean(values)), ('minimum_seed_difference_pp', min(values)), ('maximum_seed_difference_pp', max(values))]:
                close(row[field], value, field, errors)
        else:
            assert row['mean_difference_pp'] == ''
    for row in fixed:
        prefix = (row['backbone'], int(row['seed']), row['state'])
        selector = models[prefix[0], row['selector'], prefix[1], prefix[2]]
        predictor = models[prefix[0], row['predictor'], prefix[1], prefix[2]]
        if eligible.sum() >= 1152:
            selected = selector['ranks']['msp'][:1152]
            close(row['risk_percent'], 100 * predictor['losses'][row['reference']][selected, 0].mean(), 'fixed image', errors)
            close(row['accepted_n'], 1152, 'fixed image count', errors)
        else:
            assert row['risk_percent'] == row['accepted_n'] == ''
    for row in intervals:
        group_key = (row['backbone'], row['state'], row['reference'], 'msp', '0.8', row['control'])
        point_values = values_by_group.get(group_key, [])
        if len(point_values) == 3:
            close(row['mean_difference_pp'], np.mean(point_values), 'interval point estimate', errors)
        paired = np.stack([100 * (models[row['backbone'], 'soft', seed, row['state']]['bootstrap'][row['reference']]
                                      - models[row['backbone'], row['control'], seed, row['state']]['bootstrap'][row['reference']]) for seed in SEEDS])
        available = np.isfinite(paired).all(0)
        close(row['available_replicates'], available.sum(), 'bootstrap availability', errors)
        close(row['bootstrap_replicates'], 5000, 'bootstrap count', errors)
        if available.all():
            bounds = np.percentile(paired.mean(0), [2.5, 97.5])
            close(row['lower95_pp'], bounds[0], 'lower interval', errors)
            close(row['upper95_pp'], bounds[1], 'upper interval', errors)
        else:
            assert row['lower95_pp'] == row['upper95_pp'] == ''

    # Guard earlier evidence against accidental edits by this extension.
    preserved = {
        'backbone_followup/reported_results.csv': '200a9b1df63cb267169a9f888b64a7fda3993f241494fff2e33320817bbfec3b',
        'output/recovery-2026-10-10/research-revision.pdf': '182ed2e73b4501f4e13d2c903c4f243ed51b67ce46ef69be9d6693d4bf5f1c57',
        'output/recovery-2026-10-10/Supplementary-diagnostics.pdf': 'ff1b523d2079eab72fe70d572fa1376c1103f9c06cdfc3f0e8d098fe572274d3',
    }
    for name, expected in preserved.items():
        assert digest(PAPER / name) == expected, 'Earlier edition altered: ' + name
    report = dict(status='passed', freeze_sha256=frozen_sha, independent_code=True,
                  amendment_001_sha256=digest(HERE / 'amendment_001.json'),
                  raw_workbook_rows_checked=checked_rows, video_only_votes_reconstructed=1440,
                  media_clips_rehashed=1440, raw_video_bytes=raw_video_bytes,
                  valid_clips=int(eligible.sum()), sampled_frame_statuses=preparation_status,
                  checkpoint_states=36, metric_cells=936, paired_seed_contrasts=576,
                  paired_mean_contrasts=192, fixed_image_cells=192, actor_intervals=16,
                  bootstrap_replicates_recomputed=5000,
                  maximum_numeric_error=max(errors), maximum_clip_aggregation_error=max(aggregate_errors),
                  earlier_artifacts_preserved=preserved,
                  elapsed_seconds=time.monotonic() - started,
                  caveat='Internal independent implementation verification, not external replication.')
    (HERE / 'verification_report.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
