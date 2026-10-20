"""Independently recompute newly plotted summaries from retained predictions."""
import hashlib, json, re
from pathlib import Path
import numpy as np

OUT = Path(__file__).resolve().parent
ROOT = OUT.parent
BASE = ROOT / 'backbone_followup'
SEEDS = (17, 42, 89)

def load(backbone, condition, seed):
    folder = ('backbone_followup' if backbone == 'resnet18' else
              'evidence_revision' if condition == 'tie_uniform' else 'reviewer_followup')
    path = ROOT / folder / 'predictions' / f'{condition}_seed{seed}_selected/test_full.npz'
    with np.load(path) as z:
        arrays = {k: z[k] for k in ('rows', 'q', 'probabilities')}
    ids = json.loads((BASE / 'split_manifest.json').read_text())['roles']['test']
    mask = np.isin(arrays['rows'], ids)
    return {k: v[mask] for k, v in arrays.items()}

def accepted(a, mask=None):
    if mask is None:
        mask = np.ones(len(a['rows']), dtype=bool)
    indices = np.flatnonzero(mask)
    order = np.lexsort((a['rows'][indices], -a['probabilities'][indices].max(axis=1)))
    return indices[order[:int(np.ceil(.8 * len(indices)))]]

def expand(path):
    return re.sub(r'\\input\{([^}]+)\}', lambda m: expand(ROOT / m[1]), path.read_text())

def main():
    plotted = json.loads((OUT / 'figure_values.json').read_text())
    checked = 0
    components = []
    for condition in ('hard', 'tie_hard', 'uniform', 'tie_uniform', 'soft'):
        per_seed = []
        strata = []
        for seed in SEEDS:
            a = load('swin_t', condition, seed)
            assert len(a['rows']) == 3275
            q = a['q']; w = q[:, :7].sum(axis=1); largest = q[:, :7].max(axis=1)
            chosen = q[np.arange(len(q)), a['probabilities'].argmax(axis=1)]
            ix = accepted(a)
            per_seed.append([100 * v[ix].mean() for v in (1-w, w-largest, largest-chosen)])
            if condition != 'tie_uniform':
                masks = [q[:, 7:].sum(axis=1) == 0,
                         (w > 0) & (q[:, 7:].sum(axis=1) > 0)]
                strata.append([100 * (1-chosen)[accepted(a, m)].mean() for m in masks])
        components.append(np.mean(per_seed, axis=0))
        if strata:
            np.testing.assert_allclose(np.mean(strata, axis=0), plotted['Fig3_within80_' + condition], atol=1e-10, rtol=0)
            checked += 2
    np.testing.assert_allclose(components, plotted['Fig3_components_pp'], atol=1e-10, rtol=0)
    checked += 15
    for backbone in ('swin_t', 'resnet18'):
        matrices = []
        for seed in SEEDS:
            arrays = {c: load(backbone, c, seed) for c in ('uniform', 'soft')}
            np.testing.assert_array_equal(arrays['uniform']['rows'], arrays['soft']['rows'])
            np.testing.assert_allclose(arrays['uniform']['q'], arrays['soft']['q'], atol=0, rtol=0)
            matrix = []
            for predictor in ('uniform', 'soft'):
                a = arrays[predictor]
                loss = 1 - a['q'][np.arange(len(a['q'])), a['probabilities'].argmax(axis=1)]
                matrix.append([100 * loss[accepted(arrays[selector])].mean() for selector in ('uniform', 'soft')])
            matrices.append(matrix)
        np.testing.assert_allclose(np.mean(matrices, axis=0), plotted['Fig4_' + backbone + '_risk_pct'], atol=1e-10, rtol=0)
        checked += 4
    text = expand(OUT / 'manuscript.tex')
    orders = {}
    for prefix in ('fig', 'tab'):
        labels = re.findall(r'\\label\{(' + prefix + r':[^}]+)\}', text)
        refs = list(dict.fromkeys(re.findall(r'\\ref\{(' + prefix + r':[^}]+)\}', text)))
        assert labels == refs, (labels, refs)
        orders[prefix] = labels
    assert hashlib.sha256((OUT / 'reported_results.csv').read_bytes()).hexdigest() == '200a9b1df63cb267169a9f888b64a7fda3993f241494fff2e33320817bbfec3b'
    report = {'status': 'passed', 'new_figure_summary_values_recomputed': checked,
              'sources': 'retained per-image prediction files; no new inference or training',
              'first_citation_order': orders, 'ledger_unchanged': True,
              'copied_figures_have_prior_backbone_audit': True}
    (OUT / 'figure_data_audit.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))

if __name__ == '__main__':
    main()
