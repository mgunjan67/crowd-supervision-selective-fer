"""Independent annotation/metric/interval replay; never changes fitted results.

This verification uses the official count CSV directly and expands bootstrap
copies rather than reusing the primary analysis's weighted-prefix calculation.
It is an internal numerical check, not an independent empirical replication.
"""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REVISION = HERE.parent
PAPER = REVISION.parent
TRAINING = REVISION / 'training'
RESULTS = TRAINING / 'results'
SEEDS = (17, 42, 89)
VOTES = ('anger', 'disgust', 'fear', 'happiness', 'neutral', 'sadness',
         'surprise', 'contempt', 'unknown', 'NF')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def csv_rows(path):
    with path.open(newline='', encoding='utf-8-sig') as handle:
        return list(csv.DictReader(handle))


def close(actual, expected):
    assert abs(float(actual) - float(expected)) < 1e-10, (actual, expected)


def main():
    assert (TRAINING / 'evaluation_complete.json').is_file()
    summary_path = RESULTS / 'summary.json'
    summary = json.loads(summary_path.read_text())
    annotations_path = PAPER / 'data/ferplus_annotations/fer2013new.csv'
    annotations = csv_rows(annotations_path)
    counts = np.array([[int(row[name]) for name in VOTES] for row in annotations], dtype=float)
    votes = counts / counts.sum(axis=1, keepdims=True)
    role_manifest = json.loads((PAPER / 'final_experiment/split_manifest.json').read_text())
    test_rows = set(role_manifest['roles']['test'])
    cache = {}
    role_files = list((TRAINING / 'predictions').glob('*/*.npz'))
    assert len(role_files) == 99
    for path in role_files:
        with np.load(path, allow_pickle=False) as archive:
            rows = archive['rows']
            assert np.array_equal(archive['q'], votes[rows]), ('Official vote mismatch', path)
            role = path.stem
            assert set(rows) == set(role_manifest['roles'][role])
            assert len(rows) == len(set(rows))

    states = csv_rows(RESULTS / 'all_states.csv')
    assert len(states) == 51
    for record in states:
        k, condition, seed, state = int(record['k']), record['condition'], int(record['seed']), record['state']
        if k == 7 and condition != 'permutation':
            family = 'evidence_revision' if condition == 'tie_uniform' else 'reviewer_followup'
            path = PAPER / family / 'predictions' / f'{condition}_seed{seed}_{state}' / 'test_full.npz'
        else:
            path = TRAINING / 'predictions' / f'k{k}_{condition}_seed{seed}_{state}' / 'test_full.npz'
        with np.load(path, allow_pickle=False) as archive:
            mask = np.array([int(row) in test_rows for row in archive['rows']])
            rows = archive['rows'][mask]
            probabilities = archive['probabilities'][mask]
            hashes = archive['hashes'][mask]
            assert np.array_equal(archive['q'][mask], votes[rows])
        assert len(rows) == 3275 and set(rows) == test_rows
        q = votes[rows]
        prediction = probabilities.argmax(axis=1)
        loss = 1 - q[np.arange(len(rows)), prediction]
        ranking = np.array(sorted(range(len(rows)), key=lambda i: (-float(probabilities[i].max()), int(rows[i]))))
        accepted = ranking[:2620]
        padded = np.zeros((len(rows), 10), dtype=float)
        padded[:, :k] = probabilities
        distance = np.square(padded - q).sum(axis=1)
        unsupported = q[:, k:].sum(axis=1)
        ambiguity = q[:, :k].sum(axis=1) - q[:, :k].max(axis=1)
        excess = q[:, :k].max(axis=1) - q[np.arange(len(rows)), prediction]
        assert np.allclose(loss, unsupported + ambiguity + excess, atol=1e-15, rtol=0)
        for field, value in [('risk_pct', 100*loss[accepted].mean()), ('distance', distance.mean()),
                             ('unsupported_pp', 100*unsupported[accepted].mean()),
                             ('ambiguity_pp', 100*ambiguity[accepted].mean()),
                             ('excess_pp', 100*excess[accepted].mean())]:
            close(record[field], value)
        cache[k, condition, seed, state] = dict(rows=rows, hashes=hashes, loss=loss, ranking=ranking)

    intervals = []
    for contrast in summary['contrasts']:
        k, state, control = contrast['k'], contrast['state'], contrast['right']
        pairs = [(cache[k, 'soft', seed, state], cache[k, control, seed, state]) for seed in SEEDS]
        base = pairs[0][0]
        names, groups = np.unique(base['hashes'], return_inverse=True)
        assert len(names) == 3268
        for pair in pairs:
            for item in pair:
                assert np.array_equal(item['rows'], base['rows'])
                assert np.array_equal(item['hashes'], base['hashes'])
        rng = np.random.default_rng(20261020)
        sampled = []
        for _ in range(5000):
            draw = rng.integers(len(names), size=len(names))
            multiplicity = np.bincount(draw, minlength=len(names))[groups]
            retained_n = int(np.ceil(.8*sum(multiplicity)))
            deltas = []
            for left, right in pairs:
                risks = []
                for item in (left, right):
                    copied = np.repeat(item['ranking'], multiplicity[item['ranking']])
                    risks.append(item['loss'][copied[:retained_n]].mean())
                deltas.append(risks[0] - risks[1])
            sampled.append(100*np.mean(deltas))
        lower, upper = np.quantile(sampled, [.025, .975])
        close(lower, contrast['lower_pp'])
        close(upper, contrast['upper_pp'])
        point = [100*(a['loss'][a['ranking'][:2620]].mean()-b['loss'][b['ranking'][:2620]].mean()) for a,b in pairs]
        for actual, expected in zip(point, contrast['seed_differences_pp']):
            close(actual, expected)
        close(np.mean(point), contrast['mean_pp'])
        close(np.std(point, ddof=1), contrast['seed_sd_pp'])
        intervals.append(dict(k=k, state=state, control=control, lower_pp=float(lower), upper_pp=float(upper)))
        print(f'Independent interval verified: K={k}, {state}, Soft minus {control}', flush=True)

    report = dict(status='passed', raw_annotation_sha256=digest(annotations_path),
                  summary_sha256=digest(summary_path), new_role_archives=99,
                  state_rows=51, complete_vote_metrics_verified=True,
                  interval_replays=intervals, replicates_per_contrast=5000,
                  implementation='Direct official counts; Python-key ranking; explicit repeat-copy bootstrap',
                  scope='Internal numerical verification, not independent empirical replication')
    destination = REVISION / 'checks/independent_completed_results.json'
    destination.write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
