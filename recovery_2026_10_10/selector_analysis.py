"""Post-hoc score sensitivity from retained predictions; never retrains models."""
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent
ROOT = OUT.parent
SEEDS = (17, 42, 89)
STATES = ('selected', 'epoch12')
SCORES = ('msp', 'margin', 'negative_entropy')
COVERAGES = (.2, .5, .8, 1.)
FAMILIES = {'swin_t': ('hard', 'soft', 'uniform', 'tie_hard', 'tie_uniform'),
            'resnet18': ('soft', 'uniform', 'tie_uniform')}
LEDGER_SHA = '200a9b1df63cb267169a9f888b64a7fda3993f241494fff2e33320817bbfec3b'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate(p, rows):
    p = np.asarray(p, dtype=float)
    rows = np.asarray(rows)
    if p.ndim != 2 or p.shape[1] != 7 or len(p) != len(rows) or len(p) == 0:
        raise ValueError('Expected nonempty N x 7 probabilities and N row identifiers')
    if rows.ndim != 1 or not np.issubdtype(rows.dtype, np.integer) or len(np.unique(rows)) != len(rows):
        raise ValueError('Row identifiers must be unique integers')
    if not np.isfinite(p).all() or (p < 0).any() or (p > 1).any():
        raise ValueError('Invalid probabilities')
    if not np.allclose(p.sum(1), 1, atol=1e-6, rtol=0):
        raise ValueError('Probability rows do not sum to one; no implicit renormalization')
    return p, rows


def score(p, name):
    if name == 'msp':
        return p.max(1)
    if name == 'margin':
        top = np.partition(p, -2, axis=1)[:, -2:]
        return top.max(1) - top.min(1)
    if name == 'negative_entropy':
        logp = np.zeros_like(p)
        np.log(p, out=logp, where=p > 0)
        return (p * logp).sum(1)
    raise ValueError(f'Unknown confidence score: {name}')


def accepted(p, rows, name, coverage):
    p, rows = validate(p, rows)
    if not 0 < coverage <= 1:
        raise ValueError('Coverage must be in (0, 1]')
    rank = np.lexsort((rows, -score(p, name)))
    keep = np.zeros(len(p), dtype=bool)
    keep[rank[:math.ceil(coverage * len(p))]] = True
    return keep


def components(p, q):
    if q.shape != (len(p), 10) or not np.isfinite(q).all() or (q < 0).any():
        raise ValueError('Expected N x 10 annotation distributions')
    if not np.allclose(q.sum(1), 1, atol=1e-6, rtol=0):
        raise ValueError('Annotation distributions do not sum to one')
    w = q[:, :7].sum(1)
    m = q[:, :7].max(1)
    chosen = q[np.arange(len(q)), p.argmax(1)]
    result = {'risk': 1 - chosen, 'unsupported': 1 - w,
              'supported_ambiguity': w - m, 'excess': m - chosen}
    assert np.allclose(result['risk'], result['unsupported'] + result['supported_ambiguity'] + result['excess'])
    return result


def prediction_path(family, condition, seed, state):
    folder = ('backbone_followup' if family == 'resnet18' else
              'evidence_revision' if condition == 'tie_uniform' else 'reviewer_followup')
    return ROOT / folder / 'predictions' / f'{condition}_seed{seed}_{state}' / 'test_full.npz'


def load(path, ids):
    with np.load(path, allow_pickle=False) as f:
        a = {k: f[k] for k in ('probabilities', 'q', 'rows', 'hashes')}
    mask = np.isin(a['rows'], ids)
    a = {k: v[mask] for k, v in a.items()}
    a['probabilities'], a['rows'] = validate(a['probabilities'], a['rows'])
    a['q'] = a['q'].astype(float)
    assert len(a['rows']) == len(ids) == 3275
    assert set(a['rows']) == set(ids)
    components(a['probabilities'], a['q'])
    return a


def write_csv(path, rows):
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    frozen = ROOT / 'backbone_followup/reported_results.csv'
    assert sha(frozen) == LEDGER_SHA, 'Historical result ledger changed'
    with frozen.open(newline='') as f:
        originals = [r for r in csv.DictReader(f) if r['record_type'] == 'common']
    manifest = ROOT / 'backbone_followup/split_manifest.json'
    ids = json.loads(manifest.read_text())['roles']['test']
    hashes = {str(manifest.relative_to(ROOT)): sha(manifest),
              str(frozen.relative_to(ROOT)): sha(frozen),
              str((OUT / 'PROTOCOL.txt').relative_to(ROOT)): sha(OUT / 'PROTOCOL.txt')}
    records = []
    canonical = None
    matched = 0
    for family, conditions in FAMILIES.items():
        for state in STATES:
            for condition in conditions:
                for seed in SEEDS:
                    path = prediction_path(family, condition, seed, state)
                    hashes[str(path.relative_to(ROOT))] = sha(path)
                    a = load(path, ids)
                    if canonical is None:
                        canonical = a
                    for key in ('rows', 'q', 'hashes'):
                        assert np.array_equal(a[key], canonical[key]), (family, condition, seed, state, key)
                    p, rows = a['probabilities'], a['rows']
                    values = components(p, a['q'])
                    for name in SCORES:
                        for coverage in COVERAGES:
                            keep = accepted(p, rows, name, coverage)
                            rec = dict(backbone=family, state=state, condition=condition,
                                       seed=seed, score=name, requested_coverage=coverage,
                                       population_n=len(rows), accepted_n=int(keep.sum()),
                                       coverage=float(keep.mean()),
                                       **{k: float(v[keep].mean()) for k, v in values.items()})
                            records.append(rec)
                            if name == 'msp':
                                match = [r for r in originals if r['backbone'] == family and r['state'] == state
                                         and r['condition'] == condition and int(r['seed']) == seed
                                         and float(r['requested_coverage']) == coverage]
                                assert len(match) == 1, (family, state, condition, seed, coverage)
                                for key in ('risk', 'unsupported', 'supported_ambiguity', 'excess'):
                                    assert abs(float(match[0][key]) - rec[key]) < 1e-12, (key, match[0], rec)
                                assert int(match[0]['accepted_n']) == rec['accepted_n']
                                matched += 1
                    full = [r for r in records if r['backbone'] == family and r['state'] == state
                            and r['condition'] == condition and r['seed'] == seed and r['requested_coverage'] == 1]
                    assert len(full) == 3 and len({r['risk'] for r in full}) == 1
    assert len(records) == 576 and matched == 192
    lookup = {(r['backbone'], r['state'], r['condition'], r['seed'], r['score'], r['requested_coverage']): r for r in records}
    contrasts, summaries = [], []
    for family in FAMILIES:
        for state in STATES:
            for name in SCORES:
                for control in ('uniform', 'tie_uniform'):
                    for coverage in COVERAGES:
                        deltas = []
                        for seed in SEEDS:
                            left = lookup[family, state, 'soft', seed, name, coverage]
                            right = lookup[family, state, control, seed, name, coverage]
                            delta = 100 * (left['risk'] - right['risk'])
                            deltas.append(delta)
                            contrasts.append(dict(backbone=family, state=state, score=name,
                                                  control=control, requested_coverage=coverage,
                                                  seed=seed, difference_pp=delta))
                        summaries.append(dict(backbone=family, state=state, score=name, control=control,
                                              requested_coverage=coverage, mean_difference_pp=float(np.mean(deltas)),
                                              minimum_seed_difference_pp=min(deltas), maximum_seed_difference_pp=max(deltas)))
    assert sha(frozen) == LEDGER_SHA
    write_csv(OUT / 'selector_results.csv', records)
    write_csv(OUT / 'selector_contrasts.csv', contrasts)
    write_csv(OUT / 'selector_summary.csv', summaries)
    report = dict(analysis='exploratory post-hoc; already-exposed test population',
                  model_fits=24, inference_states=48, new_training_runs=0,
                  input_hashes=hashes, original_ledger_unchanged=True,
                  independently_matched_msp_rows=matched, result_rows=len(records),
                  contrast_rows=len(contrasts), summary_rows=len(summaries),
                  analysis_sha256=sha(Path(__file__)),
                  output_hashes={p.name: sha(p) for p in (OUT / 'selector_results.csv', OUT / 'selector_contrasts.csv', OUT / 'selector_summary.csv')})
    (OUT / 'analysis_audit.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps([r for r in summaries if r['requested_coverage'] == .8], indent=2))


if __name__ == '__main__':
    main()
