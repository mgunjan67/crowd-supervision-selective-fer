"""Build a new local research edition only after verified external results exist."""
import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import subprocess
from pathlib import Path

import fitz
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageOps

HERE = Path(__file__).resolve().parent
PAPER = HERE.parent
SOURCE = HERE / 'source'
BASE = PAPER / 'recovery_2026_10_10/source'
NEW = 'ravdess_transfer_2026_10_11/source'
OLD = 'recovery_2026_10_10/source'
DEST = PAPER / 'output/ravdess-transfer-2026-10-12'
BUILD = PAPER / 'tmp/ravdess_transfer_pdf'
FAMILIES = {'swin_t': 'Swin-T', 'resnet18': 'ResNet-18'}
STATES = {'selected': 'Selected', 'epoch12': 'Epoch 12'}
REFS = {'raw9': 'Raw nine', 'neutral_calm_merged': 'Calm/neutral merged'}
SCORES = {'msp': 'MSP', 'margin': 'Margin', 'negative_entropy': 'Neg. entropy'}
CONTROLS = {'uniform': 'Uniform', 'tie_uniform': 'Tie-Uniform'}
SEEDS = (17, 42, 89)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rows(name):
    with (HERE / 'results' / name).open(newline='') as stream:
        return list(csv.DictReader(stream))


def replace_once(text, before, after):
    assert text.count(before) == 1, before
    return text.replace(before, after)


def fmt(value, places=2, sign=True):
    if value in ('', None):
        return '--'
    return format(float(value), ('+' if sign else '') + '.' + str(places) + 'f')


def source_and_assets():
    verification = json.loads((HERE / 'verification_report.json').read_text())
    assert verification['status'] == 'passed'
    qc = json.loads((HERE / 'QC_review.json').read_text())
    assert qc['status'] == 'passed' and qc['actors_reviewed'] == list(range(1,25))
    overlap = json.loads((HERE / 'overlap_review_note.json').read_text())
    assert overlap['review_complete'] and overlap['outcome_informed'] is False
    audit = json.loads((HERE / 'results/analysis_audit.json').read_text())
    assert audit['status'] == 'complete'
    SOURCE.mkdir(exist_ok=True)
    for folder in ('generated', 'figures'):
        shutil.copytree(BASE / folder, SOURCE / folder, dirs_exist_ok=True)
    for path in (SOURCE / 'generated').glob('*.tex'):
        content=path.read_text().replace(OLD, NEW)
        content=content.replace('in the addendum ledger', 'in the Online Resource 1 score ledger')
        content=content.replace('learned-selector and independent-source evaluation',
                                'learned-selector and independent static-image evaluation')
        path.write_text(content, encoding='utf-8')
    bib = (BASE / 'references.bib').read_text()
    assert 'livingstone2018ravdess' not in bib
    bib += r'''

@article{livingstone2018ravdess,
 author={Livingstone, Steven R. and Russo, Frank A.},
 title={The {Ryerson Audio-Visual Database of Emotional Speech and Song} ({RAVDESS}): A dynamic, multimodal set of facial and vocal expressions in {North American English}},
 journal={PLoS ONE}, volume={13}, number={5}, pages={e0196391}, year={2018},
 doi={10.1371/journal.pone.0196391}
}
'''
    (SOURCE / 'references.bib').write_text(bib, encoding='utf-8')
    main = (BASE / 'manuscript.tex').read_text().replace(OLD, NEW)
    main = replace_once(main,
        'in two backbone families, and how sensitive is the comparison to the confidence score?',
        'in two backbone families, how sensitive is the comparison to confidence score, and does it transfer to an external acted-video source?')
    main = replace_once(main,
        'The contribution concerns reliability interpretation, not a new architecture, loss principle or rejection algorithm.',
        'A frozen-checkpoint transfer evaluation additionally tests 1,440 RAVDESS video-only speech clips with original human judgments. The contribution concerns reliability interpretation, not a new architecture, loss principle or rejection algorithm.')
    main = replace_once(main,
        'The local prototype remains an implementation artifact rather than a communication validation study.',
        'A later local plan fixed the external clip analysis before external model outcomes were computed, but after internal findings and RAVDESS annotation summaries were known; it is not an external preregistration. The local prototype remains an implementation artifact rather than a communication validation study.')
    main = replace_once(main,
        'their value still requires independent empirical validation.',
        'their external scope is examined here using frozen transfer to acted speech videos, not a new static-image training benchmark.')
    main = replace_once(main,
        'The local prototype is separate (Online Resource 1)',
        'This diagram describes the within-source experiments; external clip transfer reuses frozen checkpoints without optimization (Sect.~\\ref{sec:external}). The local prototype is separate (Online Resource 1)')
    method = r'''
\subsection{Frozen-checkpoint external clip evaluation}\label{sec:external}
RAVDESS provides an accessible source of original human categorical judgments \cite{livingstone2018ravdess}. We use all 1,440 video-only speech clips from its 24 actors, with ten responses per clip, and no audio. The unmodified S5 workbook supplies nine response options: the seven model categories, calm, and none. The original collection used professional actors aged 21--33 from Toronto, a limited population rather than a naturalistic messaging sample. The external media were obtained from the original Zenodo record under CC BY-NC-SA 4.0; each archive passed its publisher MD5 check and each extracted clip has a retained SHA256 hash. No media are redistributed. RAVDESS differs from FER in capture conditions, actor population and temporal judgment unit; it tests cross-source clip transfer, not replication of a static-image benchmark.

The local protocol was frozen on 11 October 2026 at 18:17:50 UTC, before external model outcomes, with implementation, annotation, detector and checkpoint hashes. Internal outcomes and external annotation summaries were already known. A recorded pre-analysis amendment corrects a transposed disgust/surprise response mapping using primary descriptor/accuracy anchors; no external performance metric existed before correction. The original freeze is retained. All 36 existing Soft, Uniform and Tie-Uniform states (two families, three seeds, selected and epoch 12) are evaluated without fitting, checkpoint choice, threshold fitting or expression-peak selection on RAVDESS. Hard and Tie-hard are not part of this external contrast family. The frame protocol samples eight bin centers $\lfloor(b+0.5)N/8\rfloor$, $b=0,\ldots,7$, for a clip of $N\geq8$ frames. Each frame requires exactly one MediaPipe detection (model selection 1, minimum confidence 0.5) and a valid shared clipped crop. No expanded box, fallback detector or manual repair is used. The original grayscale-three-channel, bilinear $224\times224$, mean/std 0.5 transform is retained. At least four valid distinct frames are required; otherwise the clip receives no prediction.

Inference uses the earlier benchmark's CUDA BF16 forward and float32 softmax; arithmetic mean of valid-frame probabilities in float64 gives the clip probability before argmax. This is neither a temporal model nor the prototype's FP32 single-frame update rule. Frame probabilities, clip means, crop diagnostics and failures are retained. All 1,440 clips remain in the coverage denominator. At coverage $\gamma$, the top $\lceil\gamma\,1440\rceil$ eligible clips are accepted by MSP, with filename order breaking ties. An endpoint is unavailable if too few clips are eligible. All-eligible risk and maximum achievable coverage are reported separately. Margin and negative entropy and coverages 20\%, 50\% and 100\% are prespecified sensitivities; unavailable values are not imputed.

Complete-reference disagreement is $1-q_{i\hat c_i}$ and uses the same three-term accounting as Eq.~\ref{eq:threeparts}. The primary reference preserves all nine responses. A mandatory sensitivity merges calm into neutral while leaving none unchanged, respecting the original study's neutral/calm interchangeability convention. The raw counts are never edited. These endpoints compare predictions with human response distributions, not intended-expression labels or published RAVDESS accuracy. Soft-minus-Uniform at 80\% MSP for selected Swin is the principal contrast. Both controls, families and states, all three paired seeds, and 80\% MSP fixed-image cross-evaluations are retained irrespective of direction.

Uncertainty uses 5,000 paired actor-cluster bootstrap samples, resampling 24 whole actors with replacement (RNG seed 11011). Each replicate contains 1,440 clips because each actor has 60. It reranks the duplicated clip multiset at 80\% of all repeated clips; the same actor multiplicities apply to all seeds, conditions and references. Percentile 95\% intervals describe mean paired-seed contrasts for both controls, families and states. Replicates with insufficient eligible clips are disclosed and prohibit a complete interval. Intervals condition on these fits, actors and observed votes, not unrestricted seed, rater or demographic populations; the contrast family is not used for multiple-testing superiority claims. Source-image screening compares canonical $48\times48$ grayscale crop hashes and 63-bit AC DCT perceptual hashes with all 35,887 FER2013 images before outcome interpretation. Perceptual Hamming distance at most four is a review flag, not an automatic exclusion or proof of identity overlap. Absence of a match cannot establish person-disjointness. Online Resource 8 retains the protocol, integrity records, complete outputs and independent recomputation.
'''
    main = replace_once(main, '\\subsection{Implementation and verification boundary}', method + '\n\\subsection{Implementation and verification boundary}')
    main = replace_once(main, '\\input{' + NEW + '/generated/selector_results.tex}',
                        '\\input{' + NEW + '/generated/selector_results.tex}\n\\input{' + NEW + '/generated/external_results.tex}')
    main = replace_once(main, '\\section{Discussion}', '\\section{Discussion}\n\\input{' + NEW + '/generated/external_discussion.tex}')
    before = ('The principal limitation is one reused image source. The later confidence-score sensitivity uses exposed test predictions and cannot restore a prospective test. Its seed ranges are not population uncertainty estimates. FER+ reannotates FER2013; it is not an independent image dataset. Earlier PrivateTest outcomes informed the adaptive follow-up phases. The six-control, three-run Tie-Uniform and nine-run ResNet protocols were each fixed before execution but after preceding outcomes and reviews were known. Pixel isolation removes a specific overlap route, not identity leakage, near duplicates, dataset-selection bias, or independent-domain uncertainty. The findings are conditional within-dataset evidence rather than external confirmation.')
    after = ('Training and development remain adaptive on one image source. FER+ reannotates FER2013, so the within-source test and later exploratory score sensitivity cannot restore an untouched prospective image test. Earlier PrivateTest outcomes informed the follow-up phases. RAVDESS adds a different media source under a locally frozen external plan, but its selection followed known internal findings and annotation metadata. It is a 24-actor acted-video transfer test, not an independently acquired static-image training replication or a naturalistic messaging sample. Uniform frame aggregation changes the prediction unit. Pixel/perceptual screening checks specific overlap routes, not all identity or near-duplicate leakage. The external intervals condition on the observed actors and votes; generalization to other populations remains unresolved.')
    main = replace_once(main, before, after)
    main = replace_once(main,
        'but do not compare a learned selector, an independently acquired vote-annotated image source, or an expanded-vocabulary model.',
        'but do not compare a learned selector, an independent vote-annotated static-image source, or an expanded-vocabulary model.')
    main = replace_once(main,
        'This work reanalyzes existing benchmark images and annotations and performs software tests.',
        'This work reanalyzes existing benchmark images, video-only speech clips and annotations and performs software tests.')
    main = replace_once(main,
        'Consent documentation for every photograph in the existing benchmark was not independently verified;',
        'The RAVDESS publication reports consent and ethics review for its original collection; this is not approval of the present analysis. Consent documentation for every FER photograph was not independently verified;')
    main = replace_once(main,
        'The full ten-category annotations are retained for auditing the reported metrics.',
        'The full ten-category FER+ annotations are retained for auditing the reported metrics. RAVDESS original S5 human responses and official media are available from its primary publication and Zenodo record \\cite{livingstone2018ravdess}; Online Resource 8 retains derived nine-category counts, model probabilities and integrity records, without redistributing video or facial images.')
    main = replace_once(main,
        'The recovery addendum supplies the exploratory confidence-score protocol, code, all score/coverage summaries and input hashes.',
        'The consolidated Online Resource 1 also supplies the exploratory confidence-score protocol, code, all score/coverage summaries and input hashes.')
    main = replace_once(main,
        'The additional confidence-score analysis is local in the recovery addendum and has not been uploaded to the public repository; it must accompany any submission of this edition.',
        'The consolidated Online Resource 1 includes the additional confidence-score analysis and the current complete supplement. These updates have not been uploaded to the public repository. Online Resource 8 supplies the external transfer code and evidence and is also local only; both current archives must accompany submission.')
    main = replace_once(main,
        'with checksum manifests.',
        'with checksum manifests. Online Resource 8: frozen RAVDESS protocol, clip/frame probabilities, complete external ledgers, source hashes and independent verification; no media or duplicate weights.')
    main = replace_once(main,
        'Independent-image evaluation and an appropriately reviewed communication study remain distinct future work, rather than evidence supplied by this revision.',
        'Independent static-image evaluation and an appropriately reviewed communication study remain distinct future work; the external acted-video transfer test does not supply either.')
    (SOURCE / 'manuscript.tex').write_text(main, encoding='utf-8')
    make_results(audit, overlap)


def make_results(audit, overlap):
    intervals = rows('actor_intervals.csv')
    lookup = {(r['backbone'], r['state'], r['reference'], r['control']): r for r in intervals}
    summaries = rows('paired_mean_contrasts.csv')
    means = {(r['backbone'], r['state'], r['reference'], r['score'], float(r['requested_coverage']), r['control']): r for r in summaries}
    contrasts = rows('paired_seed_contrasts.csv')
    seeds = {(r['backbone'], r['state'], r['reference'], r['score'], float(r['requested_coverage']), r['control'], int(r['seed'])): r for r in contrasts}
    all_cells = rows('clip_results.csv')
    rates = {}
    eligible = audit['eligible_clips']
    statuses = ', '.join(f'{v:,} {k.replace("_", " ")}' for k, v in sorted(audit['frame_status_counts'].items()))
    table = [r'\begin{table}[t]',
             r'\caption{External RAVDESS transfer: absolute disagreement at 80\% MSP coverage, averaged across the three seeds (\%). Each model accepts its own ranked clips}\label{tab:external}',
             r'\centering\small\begin{tabular}{lllrrr}\toprule',
             r'Backbone & State & Reference & Soft & Uniform & Tie-Uniform\\\midrule']
    for family in FAMILIES:
        for state in STATES:
            for ref in REFS:
                values=[]
                for condition in ('soft','uniform','tie_uniform'):
                    cells=[r for r in all_cells if r['backbone']==family and r['state']==state and r['reference']==ref
                           and r['condition']==condition and r['score']=='msp' and float(r['requested_coverage'])==.8]
                    assert len(cells)==3
                    rates[family,state,ref,condition]=(100*float(np.mean([float(r['risk']) for r in cells]))
                                                      if all(r['status']=='available' for r in cells) else None)
                    values.append(fmt(100*np.mean([float(r['risk']) for r in cells]),2,False)
                                  if all(r['status']=='available' for r in cells) else '--')
                table.append(f'{FAMILIES[family]} & {STATES[state]} & {"Raw" if ref=="raw9" else "Merged"} & '
                             + ' & '.join(values) + r'\\')
    table += [r'\bottomrule\end{tabular}\par\smallskip',
              r'Both references preserve none; Merged adds calm votes to neutral. Available cells use 1,152 of all 1,440 clips; -- means unavailable. These are vote-disagreement rates, not intended-expression accuracies. Paired contrasts and actor intervals appear in Fig.~\ref{fig:external}.\par', r'\end{table}']
    selected = {ref: lookup['swin_t', 'selected', ref, 'uniform'] for ref in REFS}
    res = {ref: lookup['resnet18', 'selected', ref, 'uniform'] for ref in REFS}
    # Guard the result-specific interpretation against accidental ledger drift.
    assert float(selected['raw9']['lower95_pp'])>0
    assert float(selected['neutral_calm_merged']['lower95_pp'])<0<float(selected['neutral_calm_merged']['upper95_pp'])
    direct = [r'\subsection{External transfer to RAVDESS clips}',
              f'All 24 actor archives passed publisher checksum checks. Of 1,440 clips, {eligible:,} are eligible ({100*eligible/1440:.2f}\\% maximum achievable coverage). The frame audit records {statuses}. Failures remain in the 1,440-clip denominator; the complete ledger reports unavailable nominal endpoints explicitly.',
              f'The outcome-independent source screen found {overlap["exact_canonical_matches"]} exact canonical-pixel matches and {overlap["perceptual_flags"]} perceptual review flags. This is not verification of person-disjointness. No clip is excluded on model performance.',
              *table,
              f'Selected Swin Soft-minus-Uniform disagreement at 80\\% MSP is {fmt(selected["raw9"]["mean_difference_pp"])} pp [{fmt(selected["raw9"]["lower95_pp"])}, {fmt(selected["raw9"]["upper95_pp"])}] with the raw reference and {fmt(selected["neutral_calm_merged"]["mean_difference_pp"])} pp [{fmt(selected["neutral_calm_merged"]["lower95_pp"])}, {fmt(selected["neutral_calm_merged"]["upper95_pp"])}] after merging calm into neutral. Selected ResNet differences are {fmt(res["raw9"]["mean_difference_pp"])} and {fmt(res["neutral_calm_merged"]["mean_difference_pp"])} pp, respectively. Table~\\ref{{tab:external}} supplies absolute disagreement rates, while Fig.~\\ref{{fig:external}} retains both controls, both states and all actor intervals. These are source-transfer contrasts, not accuracy improvements against the dataset\'s intended-expression labels.',
              'The principal raw-reference interval lies above zero, favoring Uniform on this conditional comparison rather than reproducing the within-source Swin advantage. The merged-reference interval includes zero; it establishes neither a stable Soft advantage nor equivalence. Both results must be interpreted alongside the large absolute disagreement rates, not as evidence that either frozen classifier is deployment-ready.',
              r'\begin{figure}[t]\centering\includegraphics[width=\linewidth]{' + NEW + r'/figures/ExternalContrasts.pdf}',
              r'\caption{External transfer at 80\% MSP coverage. Panels distinguish the two human-vote references. Each row compares one backbone, checkpoint state and control; the symbol is the mean of three paired seed differences and the horizontal line is the 5,000-replicate actor-cluster percentile 95\% interval. The dashed zero line denotes equal own-ranking disagreement, not equivalent predictions or accepted clips. Negative favors Soft. Common axis limits allow reference comparison; no result is selected by significance}\label{fig:external}\end{figure}',
              r'All three confidence scores and four nominal coverage levels are retained in Online Resource 8. Fixed-image cross-evaluation holds the accepted clips constant at 80\% MSP; it is descriptive, not a causal decomposition. Supplementary tables report per-seed contrasts and fixed-set cells. The raw-versus-merged change alters annotation scoring only; model probabilities, ranking and chosen category are unchanged.']
    (SOURCE / 'generated/external_results.tex').write_text('\n\n'.join(direct), encoding='utf-8')
    discussion = (f'The external comparison does not support transferring the small within-source Swin advantage to these acted speech clips. At the prespecified endpoint, selected Swin Soft disagreement is {fmt(rates["swin_t","selected","raw9","soft"],2,False)}\\% under raw votes and {fmt(rates["swin_t","selected","neutral_calm_merged","soft"],2,False)}\\% under the merged reference. Those absolute rates limit the use of these frozen models as reliable message cues, independently of the smaller between-target contrasts. The raw-reference principal interval favors Uniform; the merged interval crosses zero and does not demonstrate equivalence.\n\n'
                  'The paired references score unchanged predictions and accepted clips: adding calm votes to neutral changes the reward for a neutral prediction, not the recognition computation. Reference sensitivity is therefore a metric-definition effect that must be distinguished from a training-target advantage. Actor-level intervals account for repeated clips from the same performer while conditioning on only 24 actors and their observed judgments. Across-source differences combine capture, actor, annotation-vocabulary and temporal-aggregation changes; the present design cannot identify any single factor as their cause. Neither the adverse transfer contrasts nor the small within-source contrasts refute crowd-distribution learning in other settings.\n')
    (SOURCE / 'generated/external_discussion.tex').write_text(discussion, encoding='utf-8')
    abstract = r'''\textbf{Purpose:} To test whether crowd-label detail improves selective facial-expression prediction beyond target softening and whether the comparison survives changes in model family, annotation reference and source.
\textbf{Methods:} Twenty-four matched training runs compare five Swin-T and three ResNet-18 targets across three seeds on FER2013/FER+. Own-ranking, identical-image and fixed-epoch analyses qualify supervision contrasts. Frozen transfer evaluates 36 existing states on 1,440 RAVDESS video-only speech clips with ten human judgments each. Raw votes, a calm/neutral-merged sensitivity and 5,000 paired actor-bootstrap replicates are retained.
\textbf{Results:} Within-source Soft-minus-Uniform disagreement at 80\% coverage is -0.38 percentage points for selected Swin and +0.17 for ResNet. The external selected Swin contrast is ''' + fmt(selected['raw9']['mean_difference_pp']) + ' points (95\\% actor interval [' + fmt(selected['raw9']['lower95_pp']) + ', ' + fmt(selected['raw9']['upper95_pp']) + ']) using raw votes and ' + fmt(selected['neutral_calm_merged']['mean_difference_pp']) + ' [' + fmt(selected['neutral_calm_merged']['lower95_pp']) + ', ' + fmt(selected['neutral_calm_merged']['upper95_pp']) + '] after merging calm and neutral. Its absolute Soft disagreement is ' + fmt(rates['swin_t','selected','raw9','soft'],2,False) + '\\% and ' + fmt(rates['swin_t','selected','neutral_calm_merged','soft'],2,False) + '\\%, respectively. External selected ResNet differences are ' + fmt(res['raw9']['mean_difference_pp']) + ' and ' + fmt(res['neutral_calm_merged']['mean_difference_pp']) + r''' points. All 36 nonempty within-source operating points exceed calibration targets.
\textbf{Conclusion:} The within-source Swin advantage does not establish a transferable benefit of full crowd distributions. Accepted-set composition and vocabulary choice affect interpretation. High external disagreement limits deployment claims. The prototype remains unvalidated for communication benefit; acted-video transfer does not replace independent static-image evaluation.
'''
    (SOURCE / 'generated/abstract.tex').write_text(abstract, encoding='utf-8')
    conclusion = ('Matched target-softening controls, identical-image predictor comparisons and explicit annotation references constrain claims about selective facial-expression prediction. The selected Swin advantage on FER+ is not reproduced uniformly across backbone and confidence-score choices. In the locally frozen RAVDESS transfer test, the principal raw-reference interval favors Uniform, while the calm/neutral-merged interval includes zero. This limits transfer of the within-source advantage; it does not establish equivalence or refute crowd-distribution learning generally. Absolute external disagreement is too high to support reliable deployment of these frozen classifiers. All three paired seeds, both checkpoint states, both references and actor-cluster intervals are retained with sample-level probabilities. This is conditional acted-video evidence, not a static-image replication or a new rejection method. Omitted-vocabulary accounting and calibration-target overshoot remain relevant to the sender-controlled prototype, whose communication benefit and naturalistic reliability have not been evaluated.\n')
    (SOURCE / 'generated/conclusion.tex').write_text(conclusion, encoding='utf-8')
    make_figures(lookup, seeds)
    make_supplement(audit, seeds, all_cells)


def make_figures(lookup, seeds):
    plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'DejaVu Sans'],
                         'font.size': 10, 'axes.labelsize': 10, 'axes.titlesize': 11,
                         'pdf.fonttype': 42, 'ps.fonttype': 42, 'axes.spines.top': False,
                         'axes.spines.right': False})
    combinations = [(f, s, c) for f in FAMILIES for s in STATES for c in CONTROLS]
    finite = [float(r[v]) for r in lookup.values() for v in ('mean_difference_pp', 'lower95_pp', 'upper95_pp') if r[v] not in ('', None)]
    span = max(abs(x) for x in finite) * 1.15 if finite else 1
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 4.6), sharex=True, sharey=True, layout='constrained')
    for ax, ref, panel in zip(axes, REFS, ('(a)', '(b)')):
        for y, (family, state, control) in enumerate(combinations):
            row = lookup[family, state, ref, control]
            color = '#195a8b' if family == 'swin_t' else '#9a4c17'
            if row['mean_difference_pp'] != '':
                point = float(row['mean_difference_pp'])
                if row['lower95_pp'] != '':
                    lo, hi = float(row['lower95_pp']), float(row['upper95_pp'])
                    ax.plot([lo, hi], [y, y], color=color, lw=1.2)
                    ax.plot([lo, lo], [y-.09, y+.09], color=color, lw=.8)
                    ax.plot([hi, hi], [y-.09, y+.09], color=color, lw=.8)
                ax.scatter(point, y, marker='o' if family=='swin_t' else 's', color=color, s=30, zorder=3)
        ax.axvline(0, color='.3', ls='--', lw=.8)
        ax.set_title(f'{panel} {REFS[ref]}', loc='left')
        ax.set_xlim(-span, span)
        ax.grid(axis='x', color='.9', lw=.6)
        ax.set_xlabel('Soft minus control disagreement (pp)')
    axes[0].set_yticks(range(8), [f'{FAMILIES[f]} | {STATES[s]}\nvs {CONTROLS[c]}' for f,s,c in combinations])
    axes[0].invert_yaxis()
    for extension in ('pdf', 'svg', 'eps', 'png'):
        fig.savefig(SOURCE / 'figures' / ('ExternalContrasts.' + extension), dpi=240)
    plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(7.1, 5.6), sharex=True, sharey=True, layout='constrained')
    points = []
    for ax, (state, ref) in zip(axes.flat, [(s,r) for s in STATES for r in REFS]):
        for family, color, marker, dx in [('swin_t','#195a8b','o',-.12), ('resnet18','#9a4c17','s',.12)]:
            for x, score in enumerate(SCORES):
                values = [seeds[family,state,ref,score,.8,'uniform',seed]['difference_pp'] for seed in SEEDS]
                if all(v != '' for v in values):
                    yy = np.array(values, dtype=float)
                    points.extend(yy)
                    ax.scatter(x+dx+np.array([-.035,0,.035]), yy, facecolors='white', edgecolors=color, marker=marker, s=24, lw=.8)
                    ax.scatter(x+dx, yy.mean(), color=color, marker=marker, s=50, label=FAMILIES[family] if x==0 else None)
        ax.axhline(0, color='.3', ls='--', lw=.8)
        ax.set_title(STATES[state] + ' | ' + REFS[ref], loc='left')
        ax.set_xticks(range(3), ['MSP','Margin','Negative\nentropy'])
        ax.grid(axis='y', color='.9', lw=.6)
    if points:
        pad = max(.1, .1*(max(points)-min(points)))
        axes[0,0].set_ylim(min(min(points),0)-pad, max(max(points),0)+pad)
    for ax in axes[:,0]:
        ax.set_ylabel('Soft minus Uniform disagreement (pp)')
    axes[0,0].legend(frameon=False, fontsize=9.5)
    for extension in ('pdf','svg','eps','png'):
        fig.savefig(SOURCE / 'figures' / ('ExternalScores.' + extension), dpi=240)
    plt.close(fig)


def make_supplement(audit, seeds, cells):
    sup = [r'\clearpage\section*{Frozen external transfer: RAVDESS}',
           r'The external unit is a video-only speech clip, not an image or frame judgment. All 1,440 clips and 24 actors are retained in the coverage denominator. The protocol, raw-vote adapter, integrity manifests, all frame/clip probabilities and five CSV ledgers are in Online Resource 8. No new training or test-fitted threshold is performed. Raw nine-category responses preserve calm and none; the merged sensitivity adds calm to neutral and leaves none unchanged. Actor intervals are in the main paper; seed dispersion below is descriptive.',
           r'\begin{longtable}{lllrrr}', r'\caption{External frame eligibility by actor}\\',
           r'\toprule Actor & Clips & Eligible & Valid frames & Attempted frames & Max. coverage (\%)\\\midrule\endfirsthead',
           r'\toprule Actor & Clips & Eligible & Valid frames & Attempted frames & Max. coverage (\%)\\\midrule\endhead']
    for actor in range(1,25):
        report = json.loads((HERE / f'cache/Actor_{actor:02d}.json').read_text())
        sup.append(f'{actor:02d} & 60 & {report["eligible_clips"]} & {report["valid_frames"]} & {len(report["frames"])} & {100*report["eligible_clips"]/60:.2f}' + r'\\')
    sup += [r'\bottomrule\end{longtable}', r'\clearpage', r'\begin{longtable}{lllllrrr}',
            r'\caption{External 80\% contrasts for every paired seed (pp)}\\',
            r'\toprule Model & State & Ref. & Score & Control & 17 & 42 & 89\\\midrule\endfirsthead',
            r'\toprule Model & State & Ref. & Score & Control & 17 & 42 & 89\\\midrule\endhead']
    for family in FAMILIES:
        for state in STATES:
            for ref in REFS:
                for score in SCORES:
                    for control in CONTROLS:
                        vals = ' & '.join(fmt(seeds[family,state,ref,score,.8,control,seed]['difference_pp'],3) for seed in SEEDS)
                        sup.append(f'{FAMILIES[family]} & {"Sel." if state=="selected" else "E12"} & {"Raw" if ref=="raw9" else "Merged"} & {SCORES[score]} & {CONTROLS[control]} & {vals}' + r'\\')
    sup += [r'\bottomrule\end{longtable}',
            r'\clearpage\begin{figure}[H]\centering\includegraphics[width=\linewidth]{' + NEW + r'/figures/ExternalScores.pdf}',
            r'\caption{External confidence-score sensitivity at 80\% coverage for Soft minus Uniform. Hollow marks are the three paired seeds; filled marks are their means. Panels separate checkpoint state and vote reference. All panels share the same vertical scale. These are own-ranking comparisons, not fixed-image comparisons. Seed dispersion is not an actor interval and no score is selected for deployment from this plot}\end{figure}',
            r'\clearpage\begin{longtable}{llllrrrr}',
            r'\caption{External 80\% MSP fixed-image cross-evaluation: mean risk across paired seeds (\%)}\\',
            r'\toprule Model & State & Ref. & Control & S on S & C on S & S on C & C on C\\\midrule\endfirsthead',
            r'\toprule Model & State & Ref. & Control & S on S & C on S & S on C & C on C\\\midrule\endhead']
    fixed = rows('fixed_image_comparisons.csv')
    for family in FAMILIES:
        for state in STATES:
            for ref in REFS:
                for control in CONTROLS:
                    vals=[]
                    for selector,predictor in [('soft','soft'),('soft',control),(control,'soft'),(control,control)]:
                        chosen=[r['risk_percent'] for r in fixed if r['backbone']==family and r['state']==state and r['reference']==ref and r['control']==control and r['selector']==selector and r['predictor']==predictor]
                        assert len(chosen)==3
                        vals.append(fmt(np.mean(np.array(chosen,dtype=float)),2,False) if all(v!='' for v in chosen) else '--')
                    sup.append(f'{FAMILIES[family]} & {"Sel." if state=="selected" else "E12"} & {"Raw" if ref=="raw9" else "Merged"} & {CONTROLS[control]} & ' + ' & '.join(vals) + r'\\')
    sup += [r'\bottomrule\end{longtable}',
            r'S denotes Soft and C the named control. ``S on C'' applies Soft predictions to the clips accepted by the control. Every available cell uses the same nominal count of 1,152 clips. Per-seed values are retained in the CSV; these cross-evaluations are descriptive, not causal attribution.',
            r'\begin{longtable}{lllrrrrr}',
            r'\caption{External all-eligible MSP-independent risk accounting: paired-seed means (\%)}\\',
            r'\toprule Model & State & Ref. & Risk & Unsupported & Ambiguity & Excess & Coverage\\\midrule\endfirsthead',
            r'\toprule Model & State & Ref. & Risk & Unsupported & Ambiguity & Excess & Coverage\\\midrule\endhead']
    for family in FAMILIES:
        for state in STATES:
            for ref in REFS:
                for condition in ('soft','uniform','tie_uniform'):
                    group=[r for r in cells if r['backbone']==family and r['state']==state and r['reference']==ref and r['condition']==condition and r['score']=='all_eligible']
                    assert len(group)==3
                    values=[100*np.mean([float(r[k]) for r in group]) for k in ('risk','unsupported','supported_ambiguity','excess','achieved_coverage')]
                    sup.append(f'{FAMILIES[family]} {"S" if condition=="soft" else "U" if condition=="uniform" else "TU"} & {"Sel." if state=="selected" else "E12"} & {"Raw" if ref=="raw9" else "Merged"} & ' + ' & '.join(fmt(v,2,False) for v in values) + r'\\')
    sup += [r'\bottomrule\end{longtable}',
            r'All-eligible is not relabeled 100\% when a clip fails. Risk equals unsupported plus supported ambiguity plus excess before rounding. U denotes Uniform, TU Tie-Uniform. Raw/merged values score unchanged predictions; the accepted population and all-eligible coverage do not change with reference.']
    path = SOURCE / 'generated/supplement.tex'
    text = path.read_text()
    text = text.replace('in the recovery addendum', 'in the consolidated Online Resource 1')
    text = text.replace('The image source is reused; no external-domain or communication benefit is established.',
                        'Training phases reuse one image source. A later frozen acted-video transfer analysis is reported separately; no naturalistic communication benefit is established.')
    text = replace_once(text, '\\end{document}', '\n'.join(sup) + '\n\\end{document}')
    path.write_text(text, encoding='utf-8')


def expand(path):
    return re.sub(r'\\input\{([^}]+)\}', lambda m: expand(PAPER / m.group(1)), path.read_text())


def compile_and_render():
    compiler = Path('C:/Users/mgunj/AppData/Local/Programs/MiKTeX/miktex/bin/x64')
    os.environ['PATH'] = str(compiler) + os.pathsep + os.environ['PATH']
    BUILD.mkdir(parents=True, exist_ok=True)
    DEST.mkdir(parents=True, exist_ok=True)
    text = expand(SOURCE / 'manuscript.tex')
    prohibited = (r'\b(?:TODO|FIXME|placeholder)\b', r'state[ -]of[ -]the[ -]art',
                  r'significantly improved communication', r'genuine emotion',
                  r'detects? sarcasm', r'detects? bluffs?', r'privacy-preserving', r'\bVADER\b', r'affective dissonance')
    assert not any(re.search(pattern,text,re.I) for pattern in prohibited)
    bib = (SOURCE / 'references.bib').read_text()
    keys = re.findall(r'@\w+\{([^,]+),',bib)
    cited = {key.strip() for group in re.findall(r'\\cite\{([^}]+)\}',text) for key in group.split(',')}
    assert len(keys)==len(set(keys)) and set(keys)==cited
    dois=re.findall(r'doi\s*=\s*\{([^}]+)\}',bib,re.I)
    assert len(dois)==len(set(dois))
    abstract=(SOURCE/'generated/abstract.tex').read_text()
    words=len(re.findall(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)*",re.sub(r'\\[A-Za-z]+','',abstract)))
    assert 150<=words<=250,words
    reports=[]
    for job,source,filename in [('external-manuscript',NEW+'/manuscript.tex','research-revision.pdf'),
                                ('external-supplement',NEW+'/generated/supplement.tex','Supplementary-diagnostics.pdf')]:
        command=['pdflatex','-interaction=nonstopmode','-halt-on-error','-jobname='+job,
                 '-output-directory=tmp/ravdess_transfer_pdf',source]
        for index in range(4 if job=='external-manuscript' else 3):
            proc=subprocess.run(command,cwd=PAPER,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding='utf-8',errors='replace')
            if proc.returncode:
                raise RuntimeError(proc.stdout[-18000:])
            if index==0 and job=='external-manuscript':
                subprocess.run(['bibtex','tmp/ravdess_transfer_pdf/'+job],cwd=PAPER,check=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
        log=(BUILD/(job+'.log')).read_text(errors='replace')
        warnings=[line for line in log.splitlines() if re.search(r'LaTeX Warning:|Package .* Warning:|Overfull \\[hv]box|Underfull \\[hv]box|Fatal error',line)]
        if warnings:
            raise RuntimeError('\n'.join(warnings))
        target=DEST/filename
        shutil.copy2(BUILD/(job+'.pdf'),target)
        if job=='external-manuscript':
            shutil.copy2(BUILD/(job+'.bbl'),SOURCE/'manuscript.bbl')
        doc=fitz.open(target)
        rendered=BUILD/job
        rendered.mkdir(exist_ok=True)
        for number,page in enumerate(doc):
            page.get_pixmap(matrix=fitz.Matrix(1.6,1.6)).save(rendered/f'page-{number+1:02d}.png')
            for font in page.get_fonts(full=True):
                assert font[0]>0 and doc.extract_font(font[0])[3] and font[2]!='Type3'
        contacts=BUILD/(job+'-contacts')
        contacts.mkdir(exist_ok=True)
        for start in range(0,len(doc),4):
            canvas=Image.new('RGB',(1600,2310),'#cccccc')
            draw=ImageDraw.Draw(canvas)
            for j,i in enumerate(range(start,min(start+4,len(doc)))):
                with Image.open(rendered/f'page-{i+1:02d}.png') as original:
                    im=ImageOps.contain(original.convert('RGB'),(780,1110))
                x,y=j%2*800+10,j//2*1155+30
                canvas.paste(im,(x,y))
                draw.text((x,y-20),f'{job} page {i+1}',fill='black')
            canvas.save(contacts/f'sheet-{start//4+1:02d}.png')
        reports.append(dict(path=str(target),sha256=sha(target),pages=len(doc),warnings=warnings))
    report=dict(outputs=reports,references=len(keys),abstract_words=words,
                prohibited_claim_scan='passed',new_training_runs=0,
                external_verification_sha256=sha(HERE/'verification_report.json'),
                external_result_hashes=json.loads((HERE/'results/analysis_audit.json').read_text())['output_hashes'],
                original_source_sha256=sha(BASE/'manuscript.tex'),visual_review='pending')
    (HERE/'build_report.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--compile-only',action='store_true')
    args=parser.parse_args()
    if not args.compile_only:
        source_and_assets()
    compile_and_render()
