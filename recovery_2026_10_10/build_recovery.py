"""Create a separate research revision; preserve the previous submission package."""
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
from PIL import Image, ImageOps, ImageDraw

OUT = Path(__file__).resolve().parent
ROOT = OUT.parent
BASE = ROOT / 'output/submission-journal-style-2026-10-04-scientific-diagrams/source'
DEST = ROOT / 'output/recovery-2026-10-10'
SOURCE = OUT / 'source'
BUILD = ROOT / 'tmp/recovery_2026_10_10_pdf'
NEW = 'recovery_2026_10_10/source'
OLD = 'journal_style_revision'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rows(name):
    with (OUT / name).open(newline='') as f:
        return list(csv.DictReader(f))


def replace_once(text, old, new):
    assert text.count(old) == 1, old
    return text.replace(old, new)


def create_source():
    SOURCE.mkdir(exist_ok=True)
    for sub in ('generated', 'figures'):
        shutil.copytree(BASE / OLD / sub, SOURCE / sub, dirs_exist_ok=True)
    shutil.copy2(BASE / OLD / 'references.bib', SOURCE / 'references.bib')
    bibliography = SOURCE / 'references.bib'
    bibliography.write_text(bibliography.read_text() + r'''

@article{washington2021crowdsourcing,
 author={Washington, Peter and Kalantarian, Haik and Kent, Jack and Husic, Arman and Kline, Aaron and Leblanc, Emilie and Hou, Cathy and Mutlu, Cezmi and Dunlap, Kaitlyn and Penev, Yordan and Stockham, Nate and Chrisman, Brianna and Paskov, Kelley and Jung, Jae-Yoon and Voss, Catalin and Haber, Nick and Wall, Dennis P.},
 title={Training Affective Computer Vision Models by Crowdsourcing Soft-Target Labels},
 journal={Cognitive Computation}, volume={13}, pages={1363--1373}, year={2021},
 doi={10.1007/s12559-021-09936-4}
}
''', encoding='utf-8')
    for path in (SOURCE / 'generated').glob('*.tex'):
        path.write_text(path.read_text().replace(OLD + '/', NEW + '/'), encoding='utf-8')
    main = (BASE / 'main.tex').read_text().replace(OLD + '/', NEW + '/')
    main = replace_once(main,
        'in two backbone families?',
        'in two backbone families, and how sensitive is the comparison to the confidence score?')
    main = replace_once(main,
        'The contribution concerns reliability interpretation, not a new architecture or loss principle.',
        'An additional retained-prediction analysis varies three probability-only confidence scores without changing the models. The contribution concerns reliability interpretation, not a new architecture, loss principle or rejection algorithm.')
    main = replace_once(main,
        'The local prototype remains an implementation artifact rather than a communication validation study.',
        'A subsequent confidence-score sensitivity analysis reuses the already exposed test predictions; it is exploratory, not an additional independent confirmation. The local prototype remains an implementation artifact rather than a communication validation study.')
    main = replace_once(main,
        'Its language-task scores are not comparable FER benchmarks.',
        "Crowd-Calibrator estimates crowd distributions from learned representations and uses their distance from model probabilities, optionally with predictive entropy, to decide whether to abstain. We do not implement that learned comparator; probability-only score sensitivity is a narrower control. Its language-task scores are not comparable FER benchmarks.")
    main = replace_once(main,
        'Uncertain annotations have motivated FER-specific methods such as Self-Cure Network,',
        r'Washington et al. train hard- and soft-target ResNet-152 models on CAFE human response distributions and distinguish categorical-reference performance from distributional agreement \cite{washington2021crowdsourcing}. Thus, neither learning from crowd distributions nor observing a categorical-versus-distributional trade-off is new here. Our narrower comparison adds dominant-confidence and tie-preserving controls, fixed-image selective comparisons and omitted-vocabulary accounting; their value still requires independent empirical validation.' + '\n\n' + 'Uncertain annotations have motivated FER-specific methods such as Self-Cure Network,')
    main = replace_once(main,
        'We use the simpler maximum-probability score to isolate the supervision comparison.',
        'Our primary analysis uses maximum probability; an exploratory sensitivity analysis also uses the top-two probability margin and negative predictive entropy, with the fitted models unchanged.')
    method = r'''
\subsection{Exploratory confidence-score sensitivity}
After the four training phases, an analysis plan fixed three confidence functionals before computing their new comparative summaries: MSP, the top-two margin $p_{(1)}-p_{(2)}$, and negative entropy $\sum_{c=1}^{7}p_c\log p_c$, where $p_{(1)}\geq p_{(2)}$ and $0\log0=0$. All increase with confidence. Each score ranks the same frozen probabilities; predicted categories and crowd annotations are unchanged. Original row ID resolves exact ties and $\lceil\gamma N\rceil$ images are retained at coverage $\gamma\in\{0.2,0.5,0.8,1.0\}$. The complete ledger includes all 48 states and every available target. Both controls, both families and both checkpoint states are reported at 80\% coverage. These are own-ranking comparisons, not identical-image comparisons or threshold-transfer experiments. Seed differences are descriptive; no new intervals or significance claims are added. Previously known MSP and test outcomes make this analysis exploratory even though its score set and reporting rules were fixed before computing the additional summaries. The calibration-transfer policies and prototype remain unchanged.
'''
    main = replace_once(main, '\\subsection{Implementation and verification boundary}',
                        method + '\n\\subsection{Implementation and verification boundary}')
    main = replace_once(main,
        '\\input{' + NEW + '/generated/results.tex}',
        '\\input{' + NEW + '/generated/results.tex}\n\\input{' + NEW + '/generated/selector_results.tex}')
    main = replace_once(main,
        'The principal limitation is one reused image source.',
        'The principal limitation is one reused image source. The later confidence-score sensitivity uses exposed test predictions and cannot restore a prospective test. Its seed ranges are not population uncertainty estimates.')
    main = replace_once(main,
        'supplied in Online Resource 1. Images are not redistributed.',
        'supplied in Online Resource 1. The recovery addendum supplies the exploratory confidence-score protocol, code, all score/coverage summaries and input hashes. Images are not redistributed.')
    main = replace_once(main,
        'Online Resource 1 supplies experimental, analysis, and application code, tests, environment records, predictions, and package hashes.',
        'Online Resource 1 supplies experimental, analysis, and application code, tests, environment records, predictions, and package hashes. The additional confidence-score analysis is local in the recovery addendum and has not been uploaded to the public repository; it must accompany any submission of this edition.')
    main = replace_once(main, 'The current code and evidence repository is',
                        'Earlier code and evidence are at')
    (SOURCE / 'manuscript.tex').write_text(main, encoding='utf-8')
    abstract = SOURCE / 'generated/abstract.tex'
    text = abstract.read_text()
    text = replace_once(text, 'Changing the reference on identical images substantially changes F1.',
                        'Alternative confidence scores attenuate the selected Swin advantage and reverse some seed differences. Changing the reference on identical images substantially changes F1.')
    abstract.write_text(text, encoding='utf-8')
    discussion = SOURCE / 'generated/discussion.tex'
    discussion.write_text(discussion.read_text() + r'''

The score sensitivity limits a second tempting interpretation: the MSP contrast is not an intrinsic quality difference between supervision targets independent of the acceptance rule. Margin and entropy can change the retained images even when the model probabilities and predicted categories stay fixed. Some Swin paired-seed differences reverse direction, while the reported mean directions remain family-dependent. No score was chosen for deployment from these exposed-test comparisons. A learned crowd-aware or support-aware selector remains a distinct, untested comparator.
''', encoding='utf-8')
    conclusion = SOURCE / 'generated/conclusion.tex'
    text = conclusion.read_text().replace('Unsupported-vote retention and calibration transfer constrain application interpretations.',
          'Probability-only score sensitivity also changes effect magnitudes and some seed directions. Unsupported-vote retention and calibration transfer constrain application interpretations.')
    conclusion.write_text(text, encoding='utf-8')


def numeric_assets():
    summaries = rows('selector_summary.csv')
    lookup = {(r['backbone'], r['state'], r['score'], r['control']): float(r['mean_difference_pp'])
              for r in summaries if float(r['requested_coverage']) == .8}
    names = {'msp': 'MSP', 'margin': 'Margin', 'negative_entropy': 'Neg. entropy'}
    families = {'swin_t': 'Swin-T', 'resnet18': 'ResNet-18'}
    states = {'selected': 'Selected', 'epoch12': 'Epoch 12'}
    table = [r'\begin{table}[t]',
             r'\caption{Exploratory score sensitivity at 80\% coverage: paired-seed mean Soft-minus-control disagreement (pp). Negative values favor Soft; accepted image sets can differ}\label{tab:scores}',
             r'\centering\begin{tabular}{lllrr}\toprule',
             r'Backbone & State & Score & Uniform & Tie-Uniform\\\midrule']
    for family in families:
        for state in states:
            for name in names:
                table.append(f"{families[family]} & {states[state]} & {names[name]} & {lookup[family,state,name,'uniform']:+.3f} & {lookup[family,state,name,'tie_uniform']:+.3f}" + r'\\')
    table.extend([r'\bottomrule\end{tabular}\par', r'\smallskip',
                  r'MSP reproduces the original ledger exactly. Margin and negative entropy are post hoc sensitivity analyses, not proposed rejection methods. Per-seed results are in Table~S16 and Fig.~S9.\par', r'\end{table}'])
    swin = [lookup['swin_t', 'selected', n, 'uniform'] for n in names]
    resnet = [lookup['resnet18', 'selected', n, 'uniform'] for n in names]
    result = [r'\subsection{Confidence-score sensitivity on frozen predictions}', *table,
              f'Selected Swin Soft-minus-Uniform disagreement is {swin[0]:+.3f} pp under MSP, {swin[1]:+.3f} pp under margin and {swin[2]:+.3f} pp under negative entropy. The corresponding ResNet means are {resnet[0]:+.3f}, {resnet[1]:+.3f} and {resnet[2]:+.3f} pp (Table~\\ref{{tab:scores}}). Mean directions at this endpoint remain unchanged across these three scores, but magnitudes and individual-seed directions are not stable. Some Swin margin and entropy differences favor Uniform; the per-seed table retains these reversals. These comparisons do not establish a universally preferred score or negate the need for learned-selector and independent-source evaluation. All four coverage values and all 48 states are retained in the addendum ledger; full coverage gives identical risk for all three scores, as required.']
    (SOURCE / 'generated/selector_results.tex').write_text('\n'.join(result), encoding='utf-8')
    contrasts = [r for r in rows('selector_contrasts.csv') if float(r['requested_coverage']) == .8]
    d = {(r['backbone'], r['state'], r['score'], r['control'], int(r['seed'])): float(r['difference_pp']) for r in contrasts}
    sup = [r'\clearpage\section*{Exploratory confidence-score sensitivity}',
           r'This additional analysis uses already exposed test predictions. All 24 fitted models and 48 inference states are retained. Probability-only scores never use test votes to rank images. The complete new ledgers cover 20\%, 50\%, 80\% and 100\%; the table below reports 80\%, with 2,620 retained images per model. Seed variation is descriptive, not a confidence interval. No test-selected score or threshold is deployed.',
           r'\begin{longtable}{llllrrr}',
           r'\caption{All paired seed differences for score sensitivity at 80\% coverage (pp)}\label{tab:scoreseeds}\\',
           r'\toprule Backbone & State & Score & Control & Seed 17 & Seed 42 & Seed 89\\\midrule\endfirsthead',
           r'\toprule Backbone & State & Score & Control & Seed 17 & Seed 42 & Seed 89\\\midrule\endhead']
    for family in families:
        for state in states:
            for name in names:
                for control in ('uniform', 'tie_uniform'):
                    values = ' & '.join(f'{d[family,state,name,control,seed]:+.4f}' for seed in (17,42,89))
                    sup.append(f"{families[family]} & {states[state]} & {names[name]} & {'Uniform' if control=='uniform' else 'Tie-Uniform'} & {values}" + r'\\')
    sup.extend([r'\bottomrule\end{longtable}',
                r'\clearpage\begin{figure}[H]\centering\includegraphics[width=\linewidth]{' + NEW + r'/figures/SelectorSensitivity.pdf}',
                r'\caption{Probability-only score sensitivity at 80\% own-ranking coverage for Soft minus Uniform. (a) Selected checkpoints; (b) fixed epoch 12. Small hollow symbols are the three paired seeds, horizontally offset only for visibility; large filled symbols are their mean. All panels share a vertical scale. Negative values favor Soft. There are no uncertainty bars; seed dispersion is not population uncertainty. The categorical score choices are not an ordered tuning path. Some Swin seed differences cross zero; no score was selected after observing these outcomes}\label{fig:scores}\end{figure}'])
    supplement = SOURCE / 'generated/supplement.tex'
    text = supplement.read_text().replace('All numerical summaries are generated from \\texttt{reported\\_results.csv}.',
           'Original numerical summaries are generated from \\texttt{reported\\_results.csv}. The additional score-sensitivity summaries are generated from the three \\texttt{selector} CSV ledgers in the recovery addendum.')
    text = replace_once(text, '\\end{document}', '\n'.join(sup) + '\n\\end{document}')
    supplement.write_text(text, encoding='utf-8')
    plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'DejaVu Sans'],
                         'font.size': 10, 'axes.labelsize': 10, 'axes.titlesize': 11,
                         'pdf.fonttype': 42, 'ps.fonttype': 42, 'axes.spines.top': False,
                         'axes.spines.right': False})
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.35), sharey=True, layout='constrained')
    for ax, state, panel in zip(axes, states, ('(a)', '(b)')):
        for family, color, marker, dx in [('swin_t','#1b5592','o',-.12), ('resnet18','#a04816','s',.12)]:
            for x, name in enumerate(names):
                yy = [d[family,state,name,'uniform',s] for s in (17,42,89)]
                ax.scatter(x + dx + np.array([-.035,0,.035]), yy, s=26, marker=marker,
                           facecolors='white', edgecolors=color, linewidths=.9, zorder=3)
                ax.scatter(x + dx, np.mean(yy), s=62, marker=marker, color=color, zorder=4,
                           label=families[family] if x == 0 else None)
        ax.axhline(0, color='.25', linestyle='--', linewidth=.8)
        ax.set_xticks([0,1,2], ['MSP','Margin','Negative\nentropy'])
        ax.set_xlim(-.5,2.5)
        ax.set_ylim(-.56,.88)
        ax.set_title(f'{panel} {states[state]}', loc='left')
        ax.grid(axis='y', color='.9', linewidth=.6)
        ax.set_axisbelow(True)
        ax.set_xlabel('Confidence score')
    axes[0].set_ylabel('Soft minus Uniform disagreement (pp)')
    axes[0].legend(loc='upper left', frameon=False, fontsize=9)
    fig.savefig(SOURCE / 'figures/SelectorSensitivity.pdf')
    fig.savefig(SOURCE / 'figures/SelectorSensitivity.svg')
    fig.savefig(SOURCE / 'figures/SelectorSensitivity.png', dpi=240)
    plt.close(fig)


def expand(path):
    return re.sub(r'\\input\{([^}]+)\}', lambda m: expand(ROOT / m.group(1)), path.read_text())


def run(command):
    proc = subprocess.run(command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          text=True, encoding='utf-8', errors='replace')
    if proc.returncode:
        raise RuntimeError(proc.stdout[-18000:])


def compile_and_render():
    compiler = Path('C:/Users/mgunj/AppData/Local/Programs/MiKTeX/miktex/bin/x64')
    os.environ['PATH'] = str(compiler) + os.pathsep + os.environ['PATH']
    BUILD.mkdir(parents=True, exist_ok=True)
    DEST.mkdir(parents=True, exist_ok=True)
    text = expand(SOURCE / 'manuscript.tex')
    patterns = (r'\b(?:TODO|FIXME|placeholder)\b',r'state[ -]of[ -]the[ -]art',
                r'significantly improved communication',r'genuine emotion',r'detects? sarcasm',
                r'detects? bluffs?',r'privacy-preserving',r'\bVADER\b',r'affective dissonance')
    assert not any(re.search(p, text, re.I) for p in patterns)
    bib = (SOURCE / 'references.bib').read_text()
    keys = re.findall(r'@\w+\{([^,]+),', bib)
    cited = {k.strip() for group in re.findall(r'\\cite\{([^}]+)\}',text) for k in group.split(',')}
    assert len(keys) == len(set(keys)) and set(keys) == cited
    dois = re.findall(r'doi\s*=\s*\{([^}]+)\}', bib, re.I)
    assert len(dois) == len(set(dois))
    abstract = text.split('\\abstract{',1)[1].split('\\keywords',1)[0]
    word_count = len(re.findall(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)*", re.sub(r'\\[a-zA-Z]+','',abstract)))
    assert 150 <= word_count <= 250
    for heading in ('Purpose:', 'Methods:', 'Results:', 'Conclusion:'):
        assert heading in abstract
    reports = []
    for job, source, filename in [('recovery-manuscript', NEW + '/manuscript.tex', 'research-revision.pdf'),
                                   ('recovery-supplement', NEW + '/generated/supplement.tex', 'Supplementary-diagnostics.pdf')]:
        command = ['pdflatex','-interaction=nonstopmode','-halt-on-error',f'-jobname={job}',
                   '-output-directory=tmp/recovery_2026_10_10_pdf',source]
        for i in range(4 if job == 'recovery-manuscript' else 3):
            run(command)
            if i == 0 and job == 'recovery-manuscript':
                run(['bibtex', 'tmp/recovery_2026_10_10_pdf/' + job])
        log = (BUILD / (job + '.log')).read_text(errors='replace')
        warnings = [line for line in log.splitlines() if re.search(r'LaTeX Warning:|Package .* Warning:|Overfull \\[hv]box|Underfull \\[hv]box|Fatal error', line)]
        if warnings:
            raise RuntimeError('\n'.join(warnings))
        target = DEST / filename
        shutil.copy2(BUILD / (job + '.pdf'), target)
        if job == 'recovery-manuscript':
            shutil.copy2(BUILD / (job + '.bbl'), SOURCE / 'manuscript.bbl')
        doc = fitz.open(target)
        rendered = BUILD / job
        rendered.mkdir(exist_ok=True)
        for i, page in enumerate(doc):
            page.get_pixmap(matrix=fitz.Matrix(1.6,1.6)).save(rendered / f'page-{i+1:02d}.png')
            for font in page.get_fonts(full=True):
                assert font[0] > 0 and doc.extract_font(font[0])[3] and font[2] != 'Type3', font
        contacts = BUILD / (job + '-contacts')
        contacts.mkdir(exist_ok=True)
        for start in range(0,len(doc),4):
            canvas = Image.new('RGB',(1600,2310),'#cccccc')
            draw = ImageDraw.Draw(canvas)
            for j, i in enumerate(range(start,min(start+4,len(doc)))):
                with Image.open(rendered / f'page-{i+1:02d}.png') as original:
                    im = ImageOps.contain(original.convert('RGB'),(780,1110))
                x, y = (j%2)*800+10, (j//2)*1155+30
                canvas.paste(im,(x,y))
                draw.text((x,y-20),f'{job} page {i+1}',fill='black')
            canvas.save(contacts / f'sheet-{start//4+1:02d}.png')
        reports.append({'path': str(target), 'sha256':sha(target), 'pages':len(doc), 'warnings':warnings})
    report = {'outputs':reports, 'references':len(keys), 'abstract_words':word_count,
              'unsupported_claim_scan':'passed', 'new_training_runs':0,
              'original_source_sha256':sha(BASE / 'main.tex'),
              'historical_ledger_sha256':sha(ROOT / 'backbone_followup/reported_results.csv'),
              'visual_review':'pending; rendering is not inspection'}
    (OUT / 'build_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__ == '__main__':
    create_source()
    numeric_assets()
    compile_and_render()
