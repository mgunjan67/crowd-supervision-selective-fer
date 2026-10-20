"""Exploratory final diagnostics; no inference, retraining, or old-file writes."""
from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
PAPER = HERE.parents[1]
SEEDS = (17, 42, 89)
STATES = ('selected', 'epoch12')
FAMILIES = {'swin_t': ('hard', 'soft', 'uniform', 'tie_hard', 'tie_uniform'),
            'resnet18': ('soft', 'uniform', 'tie_uniform')}
EXPONENTS = (1., 1.5, 2., 3., 4., 6., 8., 16.)
COVERAGES = (.2, .5, .8, 1.)
CLASSES = ('angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise')
SOURCES = {}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def record_source(path):
    SOURCES[str(path.relative_to(PAPER)).replace('\\', '/')] = sha(path)
    return path


def write_csv(name, records):
    assert records
    fields = list(dict.fromkeys(k for row in records for k in row))
    with (HERE / name).open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)


def read_csv(name):
    with (HERE / name).open(newline='', encoding='utf-8') as handle:
        return list(csv.DictReader(handle))


def prediction_path(family, condition, seed, state, role):
    folder = ('backbone_followup' if family == 'resnet18' else
              'evidence_revision' if condition == 'tie_uniform' else 'reviewer_followup')
    return PAPER / folder / 'predictions' / f'{condition}_seed{seed}_{state}' / f'{role}.npz'


def load_predictions(family, condition, seed, state, role, ids=None):
    path = record_source(prediction_path(family, condition, seed, state, role))
    with np.load(path, allow_pickle=False) as data:
        result = {k: data[k] for k in ('probabilities', 'q', 'rows', 'hashes')}
    if ids is not None:
        mask = np.isin(result['rows'], ids)
        result = {k: v[mask] for k, v in result.items()}
        assert set(result['rows']) == set(ids)
    p = result['probabilities'].astype(float)
    assert p.shape == (len(result['rows']), 7)
    assert np.isfinite(p).all() and (p >= 0).all()
    assert np.allclose(p.sum(1), 1, atol=1e-6, rtol=0)
    assert np.allclose(result['q'].sum(1), 1)
    result['probabilities'] = p
    return result


def centered_logit_score(p, exponent):
    p = np.asarray(p, dtype=float)
    logp = np.log(np.maximum(p, np.finfo(np.float32).tiny))
    z = logp - logp.mean(1, keepdims=True)
    # Scale before exponentiation to avoid overflow without changing the ratio.
    scale = np.max(np.abs(z), axis=1, keepdims=True)
    scaled = np.divide(z, scale, out=np.zeros_like(z), where=scale > 0)
    norm = np.power(np.sum(np.abs(scaled) ** exponent, axis=1), 1 / exponent)
    return np.divide(scaled.max(1), norm, out=np.zeros(len(p)), where=norm > 0)


def ordered(scores, rows):
    return np.lexsort((rows, -np.asarray(scores)))


def aurc(scores, loss, rows):
    risk = np.cumsum(loss[ordered(scores, rows)]) / np.arange(1, len(rows) + 1)
    return float(risk.mean())


def risk_at(scores, loss, rows, coverage):
    count = math.ceil(coverage * len(rows))
    return float(loss[ordered(scores, rows)[:count]].mean()), count


def loss_for(p, q):
    return 1 - q[np.arange(len(p)), p.argmax(1)]


def score_diagnostics():
    manifest = record_source(PAPER / 'backbone_followup/split_manifest.json')
    ids = json.loads(manifest.read_text())['roles']['test']
    endpoint_rows, tuning_rows, curve_rows, input_rows = [], [], [], []
    canonical = None
    for family, conditions in FAMILIES.items():
        for condition in conditions:
            for state in STATES:
                for seed in SEEDS:
                    test = load_predictions(family, condition, seed, state, 'test_full', ids)
                    calibration = load_predictions(family, condition, seed, state, 'calibration')
                    assert len(test['rows']) == 3275 and len(calibration['rows']) == 1642
                    if canonical is None:
                        canonical = test
                    for key in ('rows', 'q', 'hashes'):
                        assert np.array_equal(test[key], canonical[key])
                    base = dict(backbone=family, condition=condition, state=state, seed=seed)
                    losses = {role: loss_for(a['probabilities'], a['q'])
                              for role, a in [('test', test), ('calibration', calibration)]}
                    for role, a in [('test', test), ('calibration', calibration)]:
                        pp = a['probabilities']
                        input_rows.append(dict(**base, role=role, n=len(pp),
                                               minimum_probability=float(pp.min()),
                                               zero_entries=int((pp == 0).sum()),
                                               clipped_entries=int((pp < np.finfo(np.float32).tiny).sum())))
                    cal = calibration['probabilities']
                    candidates = [(aurc(centered_logit_score(cal, exponent), losses['calibration'],
                                        calibration['rows']), exponent) for exponent in EXPONENTS]
                    best_index = int(np.argmin([x[0] for x in candidates]))
                    best_aurc, exponent = candidates[best_index]
                    msp_cal = aurc(cal.max(1), losses['calibration'], calibration['rows'])
                    fallback = 'centered_logit_norm' if best_aurc < msp_cal else 'msp'
                    for value, candidate in candidates:
                        tuning_rows.append(dict(**base, exponent=candidate, calibration_aurc=value,
                                                selected=int(candidate == exponent),
                                                msp_calibration_aurc=msp_cal, fallback_score=fallback))
                    p = test['probabilities']
                    score_map = {'msp': p.max(1), 'centered_logit_norm': centered_logit_score(p, exponent)}
                    score_map['calibration_choice'] = score_map[fallback]
                    for name, scores in score_map.items():
                        for coverage in COVERAGES:
                            risk, count = risk_at(scores, losses['test'], test['rows'], coverage)
                            endpoint_rows.append(dict(**base, score=name, exponent=exponent,
                                                      calibration_choice=fallback, requested_coverage=coverage,
                                                      population_n=len(p), accepted_n=count, risk=risk))
                        for coverage in np.arange(1, 21) / 20:
                            risk, count = risk_at(scores, losses['test'], test['rows'], float(coverage))
                            curve_rows.append(dict(**base, score=name, requested_coverage=float(coverage),
                                                   accepted_n=count, risk=risk))
                        assert abs(risk_at(scores, losses['test'], test['rows'], 1)[0]
                                   - losses['test'].mean()) < 1e-14
    assert len(endpoint_rows) == 48 * 3 * 4
    lookup = {(r['backbone'], r['condition'], r['state'], r['seed'], r['score'], r['requested_coverage']): r
              for r in endpoint_rows}
    contrasts = []
    for family in FAMILIES:
        for state in STATES:
            for control in ('uniform', 'tie_uniform'):
                for name in ('msp', 'centered_logit_norm', 'calibration_choice'):
                    for coverage in COVERAGES:
                        values = []
                        for seed in SEEDS:
                            delta = 100 * (lookup[family, 'soft', state, seed, name, coverage]['risk'] -
                                           lookup[family, control, state, seed, name, coverage]['risk'])
                            values.append(delta)
                            contrasts.append(dict(backbone=family, state=state, control=control, score=name,
                                                  requested_coverage=coverage, seed=seed, difference_pp=delta))
                        contrasts.append(dict(backbone=family, state=state, control=control, score=name,
                                              requested_coverage=coverage, seed='mean', difference_pp=float(np.mean(values)),
                                              minimum_seed_pp=min(values), maximum_seed_pp=max(values)))
    for filename, values in [('ranking_results.csv', endpoint_rows), ('ranking_tuning.csv', tuning_rows),
                             ('ranking_curves.csv', curve_rows), ('ranking_contrasts.csv', contrasts),
                             ('probability_input_audit.csv', input_rows)]:
        write_csv(filename, values)
    unique, counts = np.unique(canonical['hashes'], return_counts=True)
    return dict(primary_images=len(canonical['rows']), pixel_clusters=len(unique),
                cluster_size_counts={str(v): int((counts == v).sum()) for v in np.unique(counts)},
                bootstrap_replicate_formula='mean_seed(risk_soft(seed, resampled_clusters) - risk_control(seed, resampled_clusters))',
                accepted_count_formula='ceil(0.8 * resampled_image_count)',
                centered_logit_states=48, zero_probability_entries=sum(r['zero_entries'] for r in input_rows),
                fallback_chooses_normalization=sum(r['selected'] == 1 and r['fallback_score'] == 'centered_logit_norm'
                                                  for r in tuning_rows))


def external_module():
    path = record_source(PAPER / 'ravdess_transfer_2026_10_11/common.py')
    spec = importlib.util.spec_from_file_location('frozen_external_common', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def aggregate_frames(frame_p, clip_ids, frame_indices, middle):
    count = int(clip_ids.max()) + 1
    result = np.empty((count, 7), dtype=float)
    for clip in range(count):
        positions = np.flatnonzero(clip_ids == clip)
        positions = positions[np.argsort(frame_indices[positions], kind='stable')]
        assert len(positions) == 8 and len(np.unique(frame_indices[positions])) == 8
        if middle:
            positions = positions[2:6]
        result[clip] = frame_p[positions].astype(float).mean(0)
    assert np.isfinite(result).all() and np.allclose(result.sum(1), 1, atol=1e-6, rtol=0)
    return result


def external_diagnostics():
    common = external_module()
    record_source(common.HERE / 'video_only_speech_votes.csv')
    names, actors, q = common.annotations()
    rows, constants, aggregation_checks = [], [], []
    for family in FAMILIES:
        for condition in ('soft', 'uniform', 'tie_uniform'):
            for state in STATES:
                for seed in SEEDS:
                    path = record_source(common.HERE / 'predictions' / f'{family}_{condition}_seed{seed}_{state}.npz')
                    with np.load(path, allow_pickle=False) as data:
                        assert np.array_equal(names, data['filenames']) and data['valid'].all()
                        full = aggregate_frames(data['frame_probabilities'], data['frame_clip_ids'], data['frame_indices'], False)
                        center = aggregate_frames(data['frame_probabilities'], data['frame_clip_ids'], data['frame_indices'], True)
                        error = float(np.abs(full - data['probabilities']).max())
                        assert error < 1e-14
                    base = dict(backbone=family, condition=condition, state=state, seed=seed)
                    aggregation_checks.append(dict(**base, full_mean_max_absolute_reconstruction_error=error,
                                                   central_full_argmax_agreement=float((full.argmax(1) == center.argmax(1)).mean())))
                    for aggregation, p in [('full8', full), ('central4', center)]:
                        score = p.max(1)
                        for reference in common.REFERENCES:
                            qq = common.reference(q, reference)
                            values = common.components(p, qq, np.ones(1440, bool))
                            for coverage in (.8, 1.):
                                accepted_n = math.ceil(coverage * len(p))
                                keep = ordered(score, np.arange(len(p)))[:accepted_n]
                                means = values[keep].mean(0)
                                rows.append(dict(**base, aggregation=aggregation, reference=reference,
                                                 requested_coverage=coverage, accepted_n=accepted_n,
                                                 **{k: float(v) for k, v in zip(('risk', 'unsupported', 'supported_ambiguity', 'excess'), means)},
                                                 neutral_prediction_fraction=float((p[keep].argmax(1) == 4).mean()),
                                                 happy_prediction_fraction=float((p[keep].argmax(1) == 3).mean())))
                                for category, index in [('neutral', 4), ('happy', 3)]:
                                    constants.append(dict(**base, aggregation=aggregation, reference=reference,
                                                          requested_coverage=coverage, accepted_n=accepted_n,
                                                          constant_category=category,
                                                          risk=float((1 - qq[keep, index]).mean())))
    assert len(rows) == 36 * 2 * 2 * 2
    lookup = {(r['backbone'], r['condition'], r['state'], r['seed'], r['aggregation'], r['reference'], r['requested_coverage']): r for r in rows}
    contrasts = []
    for family in FAMILIES:
        for state in STATES:
            for aggregation in ('full8', 'central4'):
                for ref in common.REFERENCES:
                    for coverage in (.8, 1.):
                        for control in ('uniform', 'tie_uniform'):
                            deltas = []
                            for seed in SEEDS:
                                delta = 100 * (lookup[family, 'soft', state, seed, aggregation, ref, coverage]['risk'] -
                                               lookup[family, control, state, seed, aggregation, ref, coverage]['risk'])
                                deltas.append(delta)
                                contrasts.append(dict(backbone=family, state=state, aggregation=aggregation, reference=ref,
                                                      requested_coverage=coverage, control=control, seed=seed, difference_pp=delta))
                            contrasts.append(dict(backbone=family, state=state, aggregation=aggregation, reference=ref,
                                                  requested_coverage=coverage, control=control, seed='mean', difference_pp=float(np.mean(deltas)),
                                                  minimum_seed_pp=min(deltas), maximum_seed_pp=max(deltas)))
    changes = []
    for row in rows:
        if row['aggregation'] != 'central4':
            continue
        full = lookup[row['backbone'], row['condition'], row['state'], row['seed'], 'full8', row['reference'], row['requested_coverage']]
        changes.append({k: row[k] for k in ('backbone', 'condition', 'state', 'seed', 'reference', 'requested_coverage')} |
                       dict(central_minus_full_risk_pp=100 * (row['risk'] - full['risk']),
                            central_minus_full_neutral_pp=100 * (row['neutral_prediction_fraction'] - full['neutral_prediction_fraction'])))
    for name, values in [('external_aggregation_results.csv', rows), ('external_constant_references.csv', constants),
                         ('external_aggregation_contrasts.csv', contrasts), ('external_aggregation_changes.csv', changes),
                         ('frame_reconstruction_checks.csv', aggregation_checks)]:
        write_csv(name, values)
    return dict(external_states=36, clips=1440, aggregation_reconstruction_max_error=max(r['full_mean_max_absolute_reconstruction_error'] for r in aggregation_checks))


def targets(q, condition):
    supported = q[:, :7]
    weight = supported.sum(1)
    r = np.divide(supported, weight[:, None], out=np.zeros_like(supported), where=weight[:, None] > 0)
    maximum = r.max(1)
    hard = r.argmax(1)
    tied = r == maximum[:, None]
    count = tied.sum(1)
    if condition == 'soft':
        t = r.copy()
    elif condition == 'hard':
        t = np.eye(7)[hard]
    elif condition == 'tie_hard':
        t = tied / count[:, None]
    elif condition == 'uniform':
        t = np.repeat(((1 - maximum) / 6)[:, None], 7, axis=1)
        t[np.arange(len(t)), hard] = maximum
    elif condition == 'tie_uniform':
        other = np.maximum(0, 1 - count * maximum) / np.maximum(1, 7 - count)
        t = np.where(tied, maximum[:, None], other[:, None])
        t[count == 7] = 1 / 7
    else:
        raise ValueError(condition)
    t[weight == 0] = 0
    assert np.allclose(t[weight > 0].sum(1), 1)
    return t, weight


def target_diagnostics():
    path = record_source(PAPER / 'data/ferplus_annotations/fer2013new.csv')
    with path.open(newline='', encoding='utf-8-sig') as handle:
        train = [r for r in csv.DictReader(handle) if r['Usage'] == 'Training']
    vocabulary = ('anger', 'disgust', 'fear', 'happiness', 'neutral', 'sadness', 'surprise', 'contempt', 'unknown', 'NF')
    votes = np.array([[int(r[c]) for c in vocabulary] for r in train], dtype=float)
    q = votes / votes.sum(1, keepdims=True)
    summaries, marginals = [], []
    for condition in FAMILIES['swin_t']:
        t, weight = targets(q, condition)
        positive = weight > 0
        terms = np.zeros_like(t)
        np.log(t, where=t > 0, out=terms)
        entropy = -(t * terms).sum(1)
        for weighting in ('unweighted_positive_mass', 'supervised_mass_weighted'):
            ww = np.ones(int(positive.sum())) if weighting == 'unweighted_positive_mass' else weight[positive]
            summaries.append(dict(condition=condition, weighting=weighting, training_n=len(t),
                                  positive_mass_n=int(positive.sum()), zero_mass_n=int((~positive).sum()),
                                  mean_entropy_nats=float(np.average(entropy[positive], weights=ww)),
                                  mean_maximum_probability=float(np.average(t[positive].max(1), weights=ww))))
            marginal = np.average(t[positive], axis=0, weights=ww)
            for category, mass in zip(CLASSES, marginal):
                marginals.append(dict(condition=condition, weighting=weighting, category=category, marginal=float(mass)))
    write_csv('target_entropy_summary.csv', summaries)
    write_csv('target_class_marginals.csv', marginals)
    return dict(training_images=len(train), zero_supervised_mass=int((votes[:, :7].sum(1) == 0).sum()))


def make_figures():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family': 'serif', 'font.size': 9, 'axes.spines.top': False,
                         'axes.spines.right': False, 'svg.fonttype': 'none',
                         'pdf.fonttype': 42, 'ps.fonttype': 42})
    contrasts = read_csv('ranking_contrasts.csv')
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.7), sharey=True)
    for ax, family, title in zip(axes, FAMILIES, ('Swin-T', 'ResNet-18')):
        ax.axhline(0, color='.5', lw=.7)
        for j, score in enumerate(('msp', 'centered_logit_norm', 'calibration_choice')):
            values = [r for r in contrasts if r['backbone'] == family and r['state'] == 'selected'
                      and r['control'] == 'uniform' and r['score'] == score and float(r['requested_coverage']) == .8
                      and r['seed'] != 'mean']
            for i, row in enumerate(values):
                ax.plot(j + (i - 1) * .09, float(row['difference_pp']), 'o', ms=4,
                        color=('#1b5e8a', '#b65020', '#377642')[j])
            mean = np.mean([float(r['difference_pp']) for r in values])
            ax.plot([j-.16, j+.16], [mean, mean], color='black', lw=1.5)
        ax.set(xticks=[0, 1, 2], xticklabels=['MSP', 'Centered\nlogit norm', 'Calibration\nchoice'], title=title)
        ax.grid(axis='y', alpha=.2)
    axes[0].set_ylabel('Disagreement difference (pp)')
    fig.tight_layout()
    fig.savefig(HERE / 'ranking_comparison.svg', bbox_inches='tight')
    fig.savefig(HERE / 'ranking_comparison.pdf', bbox_inches='tight')
    fig.savefig(HERE / 'ranking_comparison.png', dpi=220, bbox_inches='tight')
    plt.close(fig)
    data = read_csv('external_aggregation_results.csv')
    constants = read_csv('external_constant_references.csv')
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.8), sharey=True)
    for ax, ref, title in zip(axes, ('raw9', 'neutral_calm_merged'), ('Raw clip votes', 'Calm merged with neutral')):
        for j, condition in enumerate(('soft', 'uniform', 'tie_uniform')):
            for offset, aggregation, marker in [(-.10, 'full8', 'o'), (.10, 'central4', 's')]:
                values = [100*float(r['risk']) for r in data if r['backbone']=='swin_t' and r['state']=='selected'
                          and r['condition']==condition and r['reference']==ref and r['aggregation']==aggregation
                          and float(r['requested_coverage'])==.8]
                ax.scatter([j+offset]*len(values), values, marker=marker, s=22,
                           color='#1b5e8a' if aggregation=='full8' else '#b65020',
                           label={'full8':'Eight frames', 'central4':'Middle four'}[aggregation] if j==0 else None)
        for category, style in [('neutral','--'),('happy',':')]:
            values = [100*float(r['risk']) for r in constants if r['reference']==ref and r['constant_category']==category
                      and float(r['requested_coverage'])==1]
            assert np.ptp(values) < 1e-10
            ax.axhline(values[0], color='.4', ls=style, lw=.9, label=f'Always {category}: all clips')
        ax.set(title=title, xticks=[0,1,2], xticklabels=['Soft','Uniform','Tie-Uniform'])
        ax.grid(axis='y', alpha=.2)
    axes[0].set_ylabel('Expected vote disagreement (%)')
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=7, loc='lower center', ncol=2,
               bbox_to_anchor=(.5,-.12), frameon=False)
    fig.tight_layout()
    fig.savefig(HERE / 'external_aggregation.svg', bbox_inches='tight')
    fig.savefig(HERE / 'external_aggregation.pdf', bbox_inches='tight')
    fig.savefig(HERE / 'external_aggregation.png', dpi=220, bbox_inches='tight')
    plt.close(fig)


def integration_outputs(audit):
    ranking = read_csv('ranking_contrasts.csv')
    tuning = read_csv('ranking_tuning.csv')
    external = read_csv('external_aggregation_results.csv')
    contrasts = read_csv('external_aggregation_contrasts.csv')
    constants = read_csv('external_constant_references.csv')
    targets_summary = read_csv('target_entropy_summary.csv')
    marginals = read_csv('target_class_marginals.csv')
    primary = [r for r in ranking if r['seed']=='mean' and float(r['requested_coverage'])==.8]
    aggregation = []
    for family in FAMILIES:
        for state in STATES:
            for condition in ('soft','uniform','tie_uniform'):
                for ref in ('raw9','neutral_calm_merged'):
                    for name in ('full8','central4'):
                        values=[r for r in external if r['backbone']==family and r['state']==state
                                and r['condition']==condition and r['reference']==ref and r['aggregation']==name
                                and float(r['requested_coverage'])==.8]
                        aggregation.append(dict(backbone=family,state=state,condition=condition,reference=ref,
                                                aggregation=name,mean_risk_percent=float(np.mean([100*float(r['risk']) for r in values])),
                                                mean_neutral_prediction_percent=float(np.mean([100*float(r['neutral_prediction_fraction']) for r in values]))))
    constants_summary=[]
    for ref in ('raw9','neutral_calm_merged'):
        for category in ('neutral','happy'):
            full=[100*float(r['risk']) for r in constants if r['reference']==ref and r['constant_category']==category
                  and float(r['requested_coverage'])==1]
            assert np.ptp(full)<1e-10
            constants_summary.append(dict(reference=ref,constant_category=category,all_clips_risk_percent=full[0]))
    effects = []
    for family in FAMILIES:
        for state in STATES:
            for name in ('msp','centered_logit_norm'):
                values=[r for r in ranking if r['backbone']==family and r['state']==state and r['score']==name
                        and r['control']=='uniform' and r['seed']!='mean' and float(r['requested_coverage'])==.8]
                delta=np.array([float(r['difference_pp']) for r in values])
                effects.append(dict(backbone=family,state=state,score=name,paired_seeds=3,
                                    paired_mean_pp=float(delta.mean()),paired_sample_sd_pp=float(delta.std(ddof=1)),
                                    paired_minimum_pp=float(delta.min()),paired_maximum_pp=float(delta.max()),
                                    interpretation='Descriptive over three retained seeds; not population-of-training-runs uncertainty'))
    write_csv('seed_effect_descriptives.csv',effects)
    summary=dict(exploratory=True,scope='New diagnostic analysis of already-exposed historical predictions',
                 primary_images=audit['primary_images'],distinct_primary_pixel_clusters=audit['pixel_clusters'],
                 cluster_size_counts=audit['cluster_size_counts'],
                 bootstrap_replicate_formula=audit['bootstrap_replicate_formula'],
                 bootstrap_accepted_count_formula=audit['accepted_count_formula'],
                 norm_exponents=list(EXPONENTS),normalization_calibration_wins=audit['fallback_chooses_normalization'],
                 evaluated_states=48,normalization_zero_probability_entries=audit['zero_probability_entries'],
                 ranking_primary_contrasts=primary,external_80_means=aggregation,
                 external_80_contrasts=[r for r in contrasts if r['seed']=='mean' and float(r['requested_coverage'])==.8],
                 constant_references_all_clips=constants_summary,target_entropy_summary=targets_summary,
                 seed_effect_descriptives=effects,
                 limits=['p-grid is a declared adaptation; training and crowd-disagreement loss differ from Xia et al.',
                         'Original uncentered logits cannot be reconstructed from probabilities',
                         'Positive-probability centered-logit identity is exact algebraically but finite retained probabilities are rounded',
                         'All analyses are post hoc after prior outcomes; calibration-only parameter choice is not independent test confirmation',
                         'Central four is one fixed sensitivity; there are no frame annotations or causal transfer diagnosis'])
    (HERE/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    # Small self-contained tables; source manuscript controls placement/captions.
    lines=[r'\begin{tabular}{llrr}',r'\toprule',
           r'Backbone & State & MSP & Centered norm \\',r'\midrule']
    for family in FAMILIES:
        for state in STATES:
            vals={r['score']:float(r['difference_pp']) for r in primary if r['backbone']==family and r['state']==state and r['control']=='uniform'}
            label={'swin_t':'Swin-T','resnet18':'ResNet-18'}[family]
            lines.append(f"{label} & {state} & {vals['msp']:+.3f} & {vals['centered_logit_norm']:+.3f} \\\\")
    lines.extend([r'\bottomrule',r'\end{tabular}'])
    (HERE/'ranking_table.tex').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    lines=[r'\begin{tabular}{lrrrr}',r'\toprule',
           r' & \multicolumn{2}{c}{Raw votes} & \multicolumn{2}{c}{Calm merged} \\',
           r'Target & Eight & Middle four & Eight & Middle four \\',r'\midrule']
    for condition in ('soft','uniform','tie_uniform'):
        values=[next(r['mean_risk_percent'] for r in aggregation if r['backbone']=='swin_t' and r['state']=='selected'
                     and r['condition']==condition and r['reference']==ref and r['aggregation']==name)
                for ref in ('raw9','neutral_calm_merged') for name in ('full8','central4')]
        lines.append(condition.replace('_','-').title()+' & '+' & '.join(f'{v:.2f}' for v in values)+r' \\')
    lines.extend([r'\bottomrule',r'\end{tabular}'])
    (HERE/'external_aggregation_table.tex').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    lines=[r'\begin{tabular}{lrr}',r'\toprule',r'Target & Unweighted entropy & Weighted entropy \\',r'\midrule']
    for condition in FAMILIES['swin_t']:
        values=[float(next(r['mean_entropy_nats'] for r in targets_summary if r['condition']==condition and r['weighting']==weighting))
                for weighting in ('unweighted_positive_mass','supervised_mass_weighted')]
        lines.append(condition.replace('_','-').title()+f' & {values[0]:.4f} & {values[1]:.4f}'+r' \\')
    lines.extend([r'\bottomrule',r'\end{tabular}'])
    (HERE/'target_entropy_table.tex').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    # Largest weighted marginal difference quantifies the acknowledged confound.
    for control in ('uniform','tie_uniform'):
        soft={r['category']:float(r['marginal']) for r in marginals if r['condition']=='soft' and r['weighting']=='supervised_mass_weighted'}
        right={r['category']:float(r['marginal']) for r in marginals if r['condition']==control and r['weighting']=='supervised_mass_weighted'}
        audit[f'soft_vs_{control}_weighted_marginal_l1']=float(sum(abs(soft[k]-right[k]) for k in soft))


def main():
    protocol = HERE / 'PROTOCOL.md'
    assert protocol.is_file()
    audit = dict(status='complete', exploratory=True, prior_test_outcomes_known=True,
                 new_training_runs=0, new_inference_runs=0, cpu_only=True,
                 protocol_sha256=sha(protocol), analysis_sha256=sha(Path(__file__)))
    audit.update(score_diagnostics())
    audit.update(external_diagnostics())
    audit.update(target_diagnostics())
    make_figures()
    integration_outputs(audit)
    audit['input_hashes'] = SOURCES
    audit['output_hashes'] = {p.name: sha(p) for p in HERE.iterdir()
                              if p.suffix in ('.csv', '.png', '.svg', '.tex', '.pdf') or p.name == 'summary.json'}
    (HERE / 'analysis_audit.json').write_text(json.dumps(audit, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in audit.items() if k not in ('input_hashes','output_hashes')}, indent=2))


if __name__ == '__main__':
    main()
