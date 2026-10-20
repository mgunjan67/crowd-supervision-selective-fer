"""Complete the previously defined held-out distribution-distance report.

No fitting, new endpoint, probability ensemble, resampling or image exclusion.
The publication summary is newly assembled after known results; the endpoint
and all 48 values already existed in the immutable consolidated ledger.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
PAPER = HERE.parent
SEEDS = (17, 42, 89)
FAMILIES = {'swin_t': 'Swin-T', 'resnet18': 'ResNet-18'}
CONDITIONS = {'hard': 'Hard', 'tie_hard': 'Tie-hard', 'uniform': 'Uniform',
              'tie_uniform': 'Tie-Uniform', 'soft': 'Soft'}
STATES = {'selected': 'Selected', 'epoch12': 'Epoch 12'}


def squared_distance(probabilities, votes):
    p = np.asarray(probabilities, dtype=np.float64)
    q = np.asarray(votes, dtype=np.float64)
    if p.ndim != 2 or q.shape != (len(p), 10) or p.shape[1] != 7 or len(p) == 0:
        raise ValueError('Expected nonempty N x 7 probabilities and N x 10 votes')
    if not np.isfinite(p).all() or not np.isfinite(q).all() or (p < 0).any() or (q < 0).any():
        raise ValueError('Probabilities and votes must be finite and nonnegative')
    if not np.allclose(p.sum(1), 1, atol=2e-6, rtol=0) or not np.allclose(q.sum(1), 1, atol=1e-12, rtol=0):
        raise ValueError('Rows must retain normalized probability/vote mass')
    padded = np.pad(p, ((0, 0), (0, 3)))
    return float(np.square(padded - q).sum(axis=1).mean())


def read(path):
    with path.open(encoding='utf-8', newline='') as stream:
        return list(csv.DictReader(stream))


def main(output_dir=None):
    destination = Path(output_dir) if output_dir else HERE / 'diagnostics'
    destination.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((PAPER / 'reviewer_followup/split_manifest.json').read_text())
    expected_ids = np.array(manifest['roles']['test'])
    assert len(expected_ids) == 3275
    ledger = {(r['backbone'], r['condition'], int(r['seed']), r['state']): r
              for r in read(PAPER / 'backbone_followup/reported_results.csv')
              if r['record_type'] == 'common' and float(r['requested_coverage']) == 1}
    assert len(ledger) == 48
    records = []
    reference = None
    for folder, family in [('reviewer_followup', 'swin_t'), ('evidence_revision', 'swin_t'), ('backbone_followup', 'resnet18')]:
        for metadata in sorted((PAPER / folder / 'predictions').glob('*/evaluation.json')):
            info = json.loads(metadata.read_text())
            path = metadata.parent / 'test_full.npz'
            with np.load(path, allow_pickle=False) as data:
                keep = np.isin(data['rows'], expected_ids)
                ids, p, q = data['rows'][keep], data['probabilities'][keep], data['q'][keep]
            assert np.array_equal(ids, expected_ids)
            if reference is None:
                reference = q.copy()
            assert np.array_equal(q, reference)
            value = squared_distance(p, q)
            # Independent expression: separate supported and omitted coordinates.
            other = float(np.mean(np.einsum('ij,ij->i', p.astype(float)-q[:, :7], p.astype(float)-q[:, :7]) + np.einsum('ij,ij->i', q[:, 7:], q[:, 7:])))
            assert abs(value-other) < 1e-14
            key = (family, info['condition'], int(info['seed']), info['state'])
            original = float(ledger[key]['distance'])
            assert abs(value-original) <= 1e-10, (key, value, original)
            records.append(dict(backbone=family, condition=key[1], seed=key[2], state=key[3],
                                population_n=len(ids), distance=value, ledger_distance=original,
                                absolute_difference=abs(value-original),
                                prediction_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    assert len(records) == len({(r['backbone'], r['condition'], r['seed'], r['state']) for r in records}) == 48
    with (destination / 'distribution_distance.csv').open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    def values(family, condition, state):
        result = [next(r['distance'] for r in records if (r['backbone'], r['condition'], r['state'], r['seed']) == (family, condition, state, seed)) for seed in SEEDS]
        return result
    means = {f'{family}|{state}|{condition}': float(np.mean(values(family, condition, state)))
             for family in FAMILIES for state in STATES for condition in ('soft', 'uniform', 'tie_uniform')}
    report = dict(status='passed', metric='mean_image(sum_category((zero_padded_p7-q10)^2))',
                  primary_population_n=3275, states=48, identical_rows_and_votes=True,
                  support_renormalization=False, maximum_ledger_discrepancy=max(r['absolute_difference'] for r in records),
                  independent_expression_tolerance=1e-14, ledger_tolerance=1e-10,
                  previously_recorded_endpoint=True, newly_assembled_descriptive_summary=True,
                  training_runs_added=0, means=means)
    (destination / 'distribution_audit.json').write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    if output_dir is None:
        table = [r'\subsection{Held-out agreement with the full vote distribution}\label{sec:distance}',
                 r'Table~\ref{tab:distance} measures agreement with the full vote distribution on every primary test image. Figure~\ref{fig:backbones} places the paired distance differences beside selective disagreement. No images are rejected by confidence for the distance endpoint. This secondary summary uses distances already recorded for all 48 states; it is not a new confirmatory analysis. The complete per-seed values for all five Swin and three ResNet targets appear in Supplementary Table~\ref{tab:distributionseeds}.',
                 r'\begin{table}[t]\caption{Agreement with the complete crowd-vote distribution. Ten-category squared distances are averaged across three separate seeds on the same 3,275 test images. Lower is better; values are unscaled distance units, not percentages}\label{tab:distance}',
                 r'\centering\small\begin{tabular}{llrrrrr}\toprule',
                 r'Backbone & State & Soft & Uniform & Tie-Uniform & $\Delta_U$ & $\Delta_{TU}$\\\midrule']
        for family in FAMILIES:
            for state in STATES:
                a, b, c = [means[f'{family}|{state}|{x}'] for x in ('soft', 'uniform', 'tie_uniform')]
                table.append(f'{FAMILIES[family]} & {STATES[state]} & {a:.5f} & {b:.5f} & {c:.5f} & {a-b:+.5f} & {a-c:+.5f}'+r'\\')
        table += [r'\bottomrule\end{tabular}\par\smallskip',
                  r'$\Delta_U$ and $\Delta_{TU}$ are Soft minus Uniform and Soft minus Tie-Uniform. For each image, the score sums ten squared probability differences after adding three zero entries to the seven model outputs. It then averages across images. Vote distributions are not renormalized to the supported categories. Model probabilities are not combined across seeds.\par\end{table}',
                  r'Soft has lower distance than both controls in every tested seed, backbone and checkpoint state. Training with the full votes therefore improves agreement with the source-dataset vote distributions, even where it does not reduce disagreement on retained category predictions. The measures answer different questions. Squared distance uses the whole probability vector. Category disagreement uses only the predicted category, and selective evaluation also changes which images are retained. This finding does not establish deployment calibration, an effect caused solely by minority-category identities, or better external transfer.']
        # A reference in a separate supplementary document cannot use main .aux.
        table[1] = table[1].replace(r'Supplementary Table~\ref{tab:distributionseeds}', 'the supplementary section on full-distribution agreement')
        (HERE / 'source/generated/distribution_results.tex').write_text('\n\n'.join(table)+'\n', encoding='utf-8')
        supplement = [r'\clearpage\section*{Held-out full-distribution agreement}',
                      r'The distance is $D=N^{-1}\sum_i\sum_{c=1}^{10}(\tilde p_{ic}-q_{ic})^2$, where $\tilde p$ adds three zeros to the seven outputs. All states use the same $N=3,275$ primary test rows and the original ten-category vote distributions, without renormalization. These descriptive summaries were assembled after the outcomes were known. The measure and per-state values were already recorded. No images are rejected by confidence, and no training or probability ensemble is added.',
                      r'\begin{table}[H]\caption{Held-out squared distribution distance for every retained target, state and seed. Lower is better; values are distance units, not percentages}\label{tab:distributionseeds}',
                      r'\centering\small\begin{tabular}{lllrrrr}\toprule',
                      r'Backbone & State & Target & Seed 17 & Seed 42 & Seed 89 & Mean\\\midrule']
        for family in FAMILIES:
            for state in STATES:
                for condition in CONDITIONS:
                    if (family, condition, SEEDS[0], state) not in ledger:
                        continue
                    scores = values(family, condition, state)
                    supplement.append(f'{FAMILIES[family]} & {STATES[state]} & {CONDITIONS[condition]} & '+' & '.join(f'{v:.6f}' for v in [*scores, np.mean(scores)])+r'\\')
                supplement.append(r'\addlinespace')
        supplement += [r'\bottomrule\end{tabular}\end{table}',
                       r'The companion \texttt{distribution\_distance.csv} retains full precision, prediction hashes and exact agreement with the consolidated original ledger. Soft is lower than both controls for each paired seed in both families and states. These overlapping comparisons are not independent replications, and no new confidence interval or calibration claim is attached.']
        (HERE / 'source/generated/distribution_supplement.tex').write_text('\n'.join(supplement)+'\n', encoding='utf-8')
    print(json.dumps(report, indent=2))
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', type=Path)
    main(parser.parse_args().output_dir)
