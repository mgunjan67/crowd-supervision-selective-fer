"""Make the final descriptive supplement directly from the audited CSV ledgers."""
import csv
from pathlib import Path

HERE=Path(__file__).resolve().parent
OUT=HERE.parent/'source/generated/final_diagnostic_supplement.tex'
SEEDS=('17','42','89')
NAME={'soft':'Soft','uniform':'Uniform','tie_uniform':'Tie-Uniform',
      'hard':'Hard','tie_hard':'Tie-hard','permutation':'Permutation',
      'swin_t':'Swin-T','resnet18':'ResNet-18','selected':'Selected',
      'epoch12':'Epoch 12','msp':'MSP','centered_logit_norm':'Norm.',
      'full8':'Eight','central4':'Middle four',
      'raw9':'Raw','neutral_calm_merged':'Merged'}

def read(name):
    with (HERE/name).open(newline='',encoding='utf-8') as stream:
        return list(csv.DictReader(stream))

def label(item):return NAME.get(item,item.replace('_',' '))

def table(caption,columns,header,rows,key):
    return '\n'.join([
        r'\begingroup\small\setlength{\tabcolsep}{4pt}\setlength{\LTcapwidth}{\linewidth}\renewcommand{\arraystretch}{.97}',
        r'\begin{longtable}{'+columns+'}',
        r'\caption{'+caption+'}'+r'\label{'+key+r'}\\',
        r'\toprule '+header+r'\\\midrule\endfirsthead',
        r'\toprule '+header+r'\\\midrule\endhead',
        *(' & '.join(row)+r'\\' for row in rows),
        r'\bottomrule\end{longtable}\endgroup',''])

def main():
    text=[r'''\clearpage\section{Final ranking and temporal-aggregation diagnostics}
\label{supp:final-diagnostics}
The diagnostic protocol was recorded before these additional computations, after the original outcomes were known. These are exploratory reanalyses of retained outputs, not independent validation. Source and prediction hashes, all score candidates, all-prefix calibration AURC, test risk--coverage summaries, and per-state reconciliation checks are supplied with the final evidence resource.

\subsection{Centered logit normalization}
For positive saved probabilities, centered logits are reconstructed as $z_c=\log p_c-K^{-1}\sum_j\log p_j$. The candidate score is $\max_c z_c/\lVert z\rVert_a$. Exponents are chosen independently for each fitted state on calibration AURC for complete-vote disagreement, from the ordered candidates $1,1.5,2,3,4,6,8,16$. Ties retain the first candidate. All 48 calibration selections prefer MSP over their best normalized candidate. Test comparisons below nevertheless report normalization itself as a sensitivity, without selecting a score from the test results. Predictions and full-coverage risk do not change. No probability required clipping to compute its logarithm.
''']
    tuning=[r for r in read('ranking_tuning.csv') if r['selected']=='1']
    rows=[]
    for backbone in ('swin_t','resnet18'):
        for state in ('selected','epoch12'):
            conditions=('hard','tie_hard','uniform','tie_uniform','soft') if backbone=='swin_t' else ('soft','uniform','tie_uniform')
            for condition in conditions:
                records={r['seed']:r for r in tuning if r['backbone']==backbone and r['state']==state and r['condition']==condition}
                assert set(records)==set(SEEDS)
                rows.append([label(backbone),label(state),label(condition),*(f"{float(records[s]['exponent']):g}" for s in SEEDS)])
    text.append(table('Calibration-selected normalization exponents. The fallback to MSP is the same for every row and seed.', 'lllrrr',
                      'Backbone & State & Target & 17 & 42 & 89',rows,'tab:normalization-exponents'))
    contrasts=[r for r in read('ranking_contrasts.csv') if float(r['requested_coverage'])==.8]
    rows=[]
    for backbone in ('swin_t','resnet18'):
        for state in ('selected','epoch12'):
            for control in ('uniform','tie_uniform'):
                for score in ('msp','centered_logit_norm'):
                    records={r['seed']:r for r in contrasts if r['backbone']==backbone and r['state']==state and r['control']==control and r['score']==score}
                    assert set(records)==set(SEEDS)|{'mean'},(backbone,state,control,score,set(records))
                    rows.append([label(backbone),label(state),label(control),label(score),*(f"{float(records[s]['difference_pp']):+.3f}" for s in (*SEEDS,'mean'))])
    text.append(table('Soft-minus-control disagreement at 80\\% coverage (percentage points). Each model and score retains its own 2,620 images. Seed columns are paired fitted comparisons, not confidence limits.',
                      'llllrrrr','Backbone & State & Control & Score & 17 & 42 & 89 & Mean',rows,'tab:normalization-seeds'))
    text.append(r'''\subsection{Target entropy and weighted class proportions}
Entropy is computed on positive supported-mass Training rows before fitting. The weighted mean uses the same supported masses as the training loss. Class-marginal L1 distance compares the weighted target distribution with Soft within each vocabulary. Zero-mass rows have zero target loss and are excluded from unweighted entropy means. The permutation control exactly preserves each seven-output probability multiset and all maximum positions; it does not preserve aggregate class proportions. These descriptive checks do not identify a causal mechanism.
''')
    rows=[]
    for r in read('new_target_audit.csv'):
        rows.append([r['k'],label(r['condition']),r['positive_mass_images'],
                     f"{float(r['mean_entropy_positive_nats']):.4f}",f"{float(r['weighted_mean_entropy_nats']):.4f}",
                     f"{float(r['marginal_l1_from_soft']):.4f}"])
    text.append(table('Training-target properties before outcome evaluation. Entropy is in nats; all conditions have 28,709 Training images.',
                      'rlrrrr','Outputs & Target & Positive rows & Entropy & Weighted & L1',rows,'tab:final-target-audit'))
    text.append(r'''\subsection{Fixed middle-window aggregation}
The middle four chronologically sampled positions (2--5, zero-based) are averaged without selecting peak frames. Each aggregation reranks clips independently by MSP. All 1,440 clips have eight valid retained frames; the original eight-frame aggregation reproduces every stored clip probability exactly. Human votes describe whole clips rather than individual frames. The middle-window analysis cannot determine which frames express the intended or perceived category.
''')
    contrasts=[r for r in read('external_aggregation_contrasts.csv') if float(r['requested_coverage'])==.8]
    rows=[]
    for backbone in ('swin_t','resnet18'):
        for state in ('selected','epoch12'):
            for ref in ('raw9','neutral_calm_merged'):
                for agg in ('full8','central4'):
                    records={r['seed']:r for r in contrasts if r['backbone']==backbone and r['state']==state and r['reference']==ref and r['aggregation']==agg and r['control']=='uniform'}
                    assert set(records)==set(SEEDS)|{'mean'},(backbone,state,ref,agg,set(records))
                    rows.append([label(backbone),label(state),label(ref),label(agg),*(f"{float(records[s]['difference_pp']):+.2f}" for s in (*SEEDS,'mean'))])
    text.append(table('External Soft-minus-Uniform disagreement at 80\\% coverage (percentage points). Each state and aggregation accepts 1,152 clips under its own ranking.',
                      'llllrrrr','Backbone & State & Votes & Frames & 17 & 42 & 89 & Mean',rows,'tab:aggregation-seeds'))
    text.append(r'''The full ledger retains all 36 frozen external states, both reference mappings, all-eligible and 80\% populations, risk components and category proportions. Constant-category references have no confidence ranking. They are scored both on all clips and on each model's exact accepted set, so the two population types are not interchangeable. Always-neutral disagreement on all clips is 90.94\% raw and 82.18\% calm-merged; always-happy disagreement is 83.23\% for either reference. These references establish poor absolute transfer despite performance better than these constant predictions.
''')
    OUT.parent.mkdir(parents=True,exist_ok=True)
    OUT.write_text('\n'.join(text),encoding='utf-8')
    print(OUT)

if __name__=='__main__':main()
