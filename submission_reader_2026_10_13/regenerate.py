"""Regenerate finalization tables, numerical prose and descriptive figures from CSVs.

No training, inference, resampling, selector tuning or seed averaging of probabilities.
"""
import csv
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

HERE=Path(__file__).resolve().parent
PAPER=HERE.parent
SOURCE=HERE/'source'
FIG=SOURCE/'figures'
GEN=SOURCE/'generated'
REL='submission_reader_2026_10_13/source'
SEEDS=(17,42,89)
LABELS=('angry','disgust','fear','happy','neutral','sad','surprise','calm','none')
FAMILIES={'swin_t':'Swin-T','resnet18':'ResNet-18'}
STATES={'selected':'Selected','epoch12':'Epoch 12'}
REFS={'raw9':'Raw','neutral_calm_merged':'Merged'}

def read(path):
    with path.open(newline='',encoding='utf-8') as stream:return list(csv.DictReader(stream))

def subset(rows,**filters):return [r for r in rows if all(str(r[k])==str(v) for k,v in filters.items())]
def one(rows,**filters):
    rr=subset(rows,**filters);assert len(rr)==1,(filters,len(rr));return rr[0]
def mean(rows,key,**filters):
    rr=subset(rows,**filters);assert len(rr)==3,(filters,len(rr));return float(np.mean([float(r[key]) for r in rr]))
def f(v,d=2,sign=False):return format(float(v),('+' if sign else '')+f'.{d}f')
def write(name,content):(GEN/name).write_text(content+'\n',encoding='utf-8')
def save(fig,name):
    for extension in ('pdf','svg','eps'):fig.savefig(FIG/(name+'.'+extension),bbox_inches='tight',dpi=300)
    plt.close(fig)

def main():
    from legacy_figures import main as regenerate_retained_figures
    regenerate_retained_figures()
    audit=json.loads((HERE/'diagnostics/diagnostic_audit.json').read_text());assert audit['status']=='passed'
    ext=PAPER/'ravdess_transfer_2026_10_11/results'
    intervals=read(ext/'actor_intervals.csv');contrasts=read(ext/'paired_seed_contrasts.csv')
    cells=read(ext/'clip_results.csv');diag=HERE/'diagnostics'
    accounting=read(diag/'risk_accounting.csv');categories=read(diag/'predicted_categories.csv')
    temporal=read(diag/'temporal_bins.csv');matrices=read(diag/'vote_weighted_matrices.csv')
    refs=read(diag/'reference_vote_mass.csv');principal=read(diag/'principal_seed_contrasts.csv')
    selector=read(PAPER/'recovery_2026_10_10/selector_summary.csv')
    internal={family:float(one(selector,backbone=family,state='selected',score='msp',control='uniform',requested_coverage=.8)['mean_difference_pp']) for family in FAMILIES}
    p={ref:one(intervals,backbone='swin_t',state='selected',reference=ref,control='uniform') for ref in REFS}
    def ci(ref):
        r=p[ref];return f'[{f(r["lower95_pp"],2,True)}, {f(r["upper95_pp"],2,True)}]'
    def ar(family,state,ref,condition):
        return 100*mean(cells,'risk',backbone=family,state=state,reference=ref,condition=condition,score='msp',requested_coverage=.8)
    absolute=[r'\begin{table}[t]',r'\caption{External disagreement at 80\% MSP coverage (\%): means of three seed-specific risks, not an ensemble. Every model accepts its own 1,152 of 1,440 clips}\label{tab:external}',r'\centering\small\begin{tabular}{lllrrr}\toprule',r'Backbone & State & Reference & Soft & Uniform & Tie-Uniform\\\midrule']
    for family in FAMILIES:
        for state in STATES:
            for ref in REFS:
                absolute.append(f'{FAMILIES[family]} & {STATES[state]} & {REFS[ref]} & '+ ' & '.join(f(ar(family,state,ref,c)) for c in ('soft','uniform','tie_uniform'))+r'\\')
    absolute += [r'\bottomrule\end{tabular}\par\smallskip',r'Raw retains nine human responses; Merged adds calm to neutral and retains none. Neither is intended-expression accuracy. The model predictions and accepted IDs are unchanged between references.\par',r'\end{table}']
    seed_table=[r'\begin{table}[t]',r'\caption{Principal external comparison: selected Swin Soft minus Uniform at 80\% MSP coverage (percentage points). Negative favors Soft}\label{tab:externalseeds}',r'\centering\small\begin{tabular}{lrrrrl}\toprule',r'Reference & Seed 17 & Seed 42 & Seed 89 & Mean & Actor interval\\\midrule']
    for ref in REFS:
        values=[float(one(principal,reference=ref,seed=s)['difference_pp']) for s in SEEDS]
        assert abs(np.mean(values)-float(p[ref]['mean_difference_pp']))<1e-10
        seed_table.append(REFS[ref]+' & '+' & '.join(f(v,3,True) for v in values)+f' & {f(np.mean(values),3,True)} & {ci(ref)}'+r'\\')
    seed_table += [r'\bottomrule\end{tabular}\par\smallskip',r'Percentile 95\% actor-cluster intervals use shared actor multiplicities and reranking within 5,000 replicates, conditional on these three fitted pairs and observed votes. They do not quantify seed-population, new-rater or unrestricted demographic uncertainty.\par',r'\end{table}']
    table=[r'\begin{table}[t]',r'\caption{Principal external risk accounting: selected Swin, means across three seeds (\%). Components are evaluated on the same population as total risk}\label{tab:externalaccounting}',r'\centering\small\begin{tabular}{lllrrrr}\toprule',r'Population & Target & Ref. & Risk & Unsup. & Ambig. & Excess\\\midrule']
    means={}
    for pop in ('all_eligible','msp80'):
        for ref in REFS:
            for c in ('soft','uniform'):
                values=[100*mean(accounting,k,backbone='swin_t',state='selected',population=pop,reference=ref,condition=c) for k in ('risk','unsupported','supported_ambiguity','excess')]
                means[pop,ref,c]=values
                assert abs(values[0]-sum(values[1:]))<1e-10
                table.append(('All eligible' if pop=='all_eligible' else r'80\% MSP')+f' & {c.title()} & {REFS[ref]} & '+' & '.join(f(v) for v in values)+r'\\')
    table += [r'\bottomrule\end{tabular}\par\smallskip',r'All eligible: 1,440 clips per state. 80\% MSP: 1,152 independently ranked clips per state. Unsup. is unsupported vote mass; Ambig. is supported ambiguity. All-eligible components are not substituted for accepted-set components. Rounding can prevent displayed components from summing exactly.\par',r'\end{table}']
    neutral={c:[100*float(one(categories,backbone='swin_t',condition=c,state='selected',seed=s,population='msp80',predicted_category='neutral')['proportion']) for s in SEEDS] for c in ('soft','uniform')}
    vv=means['all_eligible','raw9','soft'];aa=means['msp80','raw9','soft']
    direct=[r'\subsection{External transfer and post hoc diagnostics}',
            r'All 24 actor archives passed publisher checksums. All 1,440 clips and 11,520 sampled frames are valid, giving 100\% maximum achievable coverage. No model-outcome filtering is applied. Canonical-pixel and perceptual source screening found no matches or review flags, without establishing person-disjointness.',
            '\n'.join(absolute),'\n'.join(seed_table),
            f'The principal mean under the raw reference is {f(p["raw9"]["mean_difference_pp"],3,True)} pp. One larger positive contrast (seed 42) outweighs two negative contrasts (seeds 17 and 89). The direction therefore does not reverse consistently across seeds. The raw actor interval is above zero, but it describes actor-sampling sensitivity for these fitted pairs only. The merged interval crosses zero. This does not establish equivalence or a stable Soft advantage. The within-source advantage is not established reliably in this transfer setting.',
            r'\begin{figure}[tp]\centering\includegraphics[width=\linewidth]{'+REL+r'/figures/ExternalContrasts.pdf}',
            r'\caption{External disagreement differences at 80\% MSP coverage, using separate model rankings. Hollow symbols show all three paired seed contrasts; the diamond and bar show their mean and conditional 95\% actor-cluster interval. Vertical offsets separate the seeds. Sel.: selected checkpoint; E12: fixed-epoch checkpoint; U: Uniform; TU: Tie-Uniform. Both panels share the horizontal scale; negative favors Soft. Intervals describe actor-sampling sensitivity for these fitted pairs and votes, not uncertainty over new seeds, raters or populations. Zero denotes equal mean disagreement, not equivalent predictions or accepted sets}\label{fig:external}\end{figure}',
            '\n'.join(table),
            fr'For selected Swin Soft across all eligible clips, raw disagreement is {f(vv[0])}\%: {f(vv[1])} points from unsupported mass, {f(vv[2])} from supported ambiguity and {f(vv[3])} from excess. At 80\% MSP the corresponding values are {f(aa[0])}\%, {f(aa[1])}, {f(aa[2])} and {f(aa[3])} points. Excess is the largest component in both populations. Missing output categories alone therefore cannot explain the poor transfer. This breakdown describes disagreement; it does not establish its cause.',
            fr'The post hoc outputs show neutral-dominated accepted predictions in all three selected Swin seeds: Soft {f(min(neutral["soft"]),1)}--{f(max(neutral["soft"]),1)}\% and Uniform {f(min(neutral["uniform"]),1)}--{f(max(neutral["uniform"]),1)}\%. Supplementary Figs.~S11--S12 show all six principal vote-weighted matrices and all eight temporal bins. Neutral probability is highest at the first bin and falls toward the middle; frame-to-clip argmax agreement also varies by bin. This is compatible with temporal aggregation difficulty but cannot identify its cause, and is not frame accuracy.',
            'All 36 states, category frequencies, raw reference vote mass, raw/merged matrices and temporal summaries are retained in Online Resource 8. The original score/coverage and fixed-image contrasts remain in the supplement and full ledgers. Reference changes never modify predictions or accepted IDs.']
    write('external_results.tex','\n\n'.join(direct))
    abstract=(r'\textbf{Purpose:} To evaluate crowd-target supervision for selective facial-expression prediction while separating target softening, accepted-image composition and annotation vocabulary.\n'
        r'\textbf{Methods:} Twenty-four matched training runs compare five Swin-T and three ResNet-18 targets across three seeds on FER2013/FER+. Own-ranking, identical-image, fixed-epoch and confidence-score analyses reuse retained outputs. Frozen transfer evaluates 36 existing states on 1,440 RAVDESS video-only speech clips with ten judgments each; raw and calm/neutral-merged references score unchanged predictions.\n'
        f'\\textbf{{Results:}} Within-source selected Soft-minus-Uniform disagreement at 80\\% coverage is {f(internal["swin_t"])} percentage points for Swin and {f(internal["resnet18"],2,True)} for ResNet. The external selected Swin raw-reference mean is {f(p["raw9"]["mean_difference_pp"],2,True)} points (95\\% actor interval {ci("raw9")}), but two seed contrasts are negative and one is positive. The merged mean is {f(p["neutral_calm_merged"]["mean_difference_pp"],2,True)} points {ci("neutral_calm_merged")}. Intervals condition on the three fitted pairs and observed votes, not seed-population or new-rater uncertainty. External Soft disagreement is {f(ar("swin_t","selected","raw9","soft"))}\\%; excess above the available-category annotation floor is the largest accounting component. All 36 nonempty within-source empirical operating points exceed calibration targets.\n'
        r'\textbf{Conclusion:} Matched softening controls and fixed-image comparisons limit claims of a general benefit from full crowd distributions. The within-source advantage is not established reliably in external transfer. Vocabulary-aware accounting and achieved-risk reporting qualify reliability claims; the prototype remains unvalidated for communication benefit.').replace('\\n','\n')
    write('abstract.tex',abstract)
    write('external_discussion.tex',f'The external analysis provides adverse transfer evidence rather than a stable reversal favoring Uniform. Its positive raw mean combines two negative contrasts with a larger positive one; the actor interval describes conditional sampling sensitivity, not confidence over training seeds. Absolute Soft disagreement of {f(ar("swin_t","selected","raw9","soft"))}\\% (raw) and {f(ar("swin_t","selected","neutral_calm_merged","soft"))}\\% (merged) limits deployment claims independently of the target contrast.\n\nThe raw-to-merged comparison rewards an unchanged neutral prediction differently. It therefore changes metric definition, not recognition computation. The accounting and post hoc matrices show large excess as well as unsupported vote mass; neither a restricted vocabulary nor mean-frame aggregation uniquely explains failure. Capture, population, judgment unit and annotation vocabulary change together across sources. The temporal summaries diagnose variation without supplying frame-level labels or a causal explanation.')
    write('conclusion.tex',f'Matched softening controls, identical-image comparisons and explicit annotation vocabularies qualify selective facial-expression comparisons. The selected Swin within-source Soft-minus-Uniform difference is {f(internal["swin_t"],3)} pp, whereas ResNet is {f(internal["resnet18"],3,True)} pp; the fixed-image predictor comparison depends on the retained set. External selected Swin has a positive raw mean but two negative seed contrasts and one larger positive contrast. Its actor interval is conditional on those fitted pairs and observed votes, not seed-population, new-rater or demographic uncertainty. The within-source advantage is therefore not established reliably in this transfer setting. High external disagreement, with excess larger than unsupported mass, and calibration-target overshoot limit deployment claims. The findings concern the tested targets and evaluation conditions; they establish neither general superiority or equivalence of targets nor a new rejection method or a communication benefit.')
    # All scientific graphics are generated from complete, verified tabular data.
    # Main source is narrower than the plotting canvas: use >=11-pt lettering
    # here so the final embedded external forest plot remains >=8 pt.
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.titlesize':12,'axes.labelsize':11,'xtick.labelsize':11,'ytick.labelsize':11,'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False})
    rows=[(b,s,c) for b in FAMILIES for s in STATES for c in ('uniform','tie_uniform')]
    fig,axes=plt.subplots(2,1,figsize=(6.7,6.7),sharex=True,sharey=True,layout='constrained')
    allvalues=[float(r['difference_pp']) for r in contrasts if r['score']=='msp' and float(r['requested_coverage'])==.8]
    lower=min(allvalues+[float(r['lower95_pp']) for r in intervals])-.45
    upper=max(allvalues+[float(r['upper95_pp']) for r in intervals])+.45
    for ax,(ref,label) in zip(axes,REFS.items()):
        ax.axvline(0,color='.4',lw=.8,ls='--');ax.grid(axis='x',color='.91',lw=.6)
        for y,(family,state,control) in enumerate(rows):
            r=one(intervals,backbone=family,state=state,reference=ref,control=control)
            m,lo,hi=[float(r[k]) for k in ('mean_difference_pp','lower95_pp','upper95_pp')]
            for offset,seed,marker in zip((-.17,0,.17),SEEDS,('o','s','^')):
                value=float(one(contrasts,backbone=family,state=state,reference=ref,control=control,score='msp',requested_coverage=.8,seed=seed)['difference_pp'])
                ax.plot(value,y+offset,marker=marker,ms=4.5,mfc='white',mec='#466b81',linestyle='none',label=f'Seed {seed}' if y==0 else None,zorder=3)
            ax.plot([lo,hi],[y,y],color='#1b3d52',lw=1.5,zorder=4)
            ax.plot(m,y,'D',color='#1b3d52',ms=4,zorder=5,label='Mean / actor interval' if y==0 else None)
        ax.set_title(label+' human-vote reference');ax.set_xlabel('Soft minus control disagreement (pp)');ax.set_xlim(lower,upper)
        ax.set_yticks(range(8));ax.set_yticklabels([f'{FAMILIES[b]} | {"Sel." if s=="selected" else "E12"} | {"U" if c=="uniform" else "TU"}' for b,s,c in rows]);ax.set_ylim(7.6,-.6)
    axes[0].legend(loc='lower center',bbox_to_anchor=(.5,1.14),fontsize=11,frameon=False,ncol=2)
    save(fig,'ExternalContrasts')
    fig,axes=plt.subplots(2,3,figsize=(7.0,6.1),layout='constrained')
    for row,c in enumerate(('soft','uniform')):
        for col,seed in enumerate(SEEDS):
            ax=axes[row,col]
            mm=np.array([[float(one(matrices,backbone='swin_t',state='selected',condition=c,seed=seed,population='msp80',reference='raw9',reference_category=a,predicted_category=b)['row_normalized']) for b in LABELS[:7]] for a in LABELS])
            im=ax.imshow(mm,vmin=0,vmax=1,cmap='Blues',aspect='auto',interpolation='nearest')
            ax.set_xticks(range(7),LABELS[:7],rotation=60,ha='right',fontsize=9);ax.set_yticks(range(9),LABELS,fontsize=9)
            ax.set_title(f'{c.title()} | seed {seed}\nn=1,152',fontsize=9.5);ax.set_xlabel('Predicted category',fontsize=9)
            if col==0:ax.set_ylabel('Human-vote reference category',fontsize=9)
    fig.colorbar(im,ax=axes,shrink=.67,pad=.015,label='Fraction of reference-row vote mass')
    save(fig,'ExternalVoteMatrices')
    fig,axes=plt.subplots(1,2,figsize=(6.1,3.8),layout='constrained')
    for c,color,style in (('soft','#215e80','-'),('uniform','#9a541d','--')):
        for seed,marker in zip(SEEDS,('o','s','^')):
            rr=sorted(subset(temporal,backbone='swin_t',condition=c,state='selected',seed=seed,predicted_category='neutral'),key=lambda r:int(r['bin']))
            for ax,key in zip(axes,('mean_neutral_probability','frame_clip_argmax_agreement')):
                ax.plot(range(1,9),[100*float(r[key]) for r in rr],color=color,ls=style,marker=marker,ms=3.5,lw=1,label=f'{c.title()}, seed {seed}')
    for ax in axes:
        ax.set_xticks(range(1,9));ax.set_xlabel('Temporal bin',fontsize=10);ax.set_ylim(0,100);ax.grid(color='.91',lw=.6);ax.tick_params(labelsize=9.5)
    axes[0].set_ylabel('Mean neutral probability (%)',fontsize=10);axes[1].set_ylabel('Frame-to-clip argmax agreement (%)',fontsize=10)
    fig.legend(*axes[0].get_legend_handles_labels(),fontsize=9,ncol=3,loc='lower center',bbox_to_anchor=(.5,-.13),frameon=False)
    save(fig,'ExternalTemporal')
    # A compact principal category table displays every category and all seeds.
    category_table=[r'\begin{table}[H]',r'\caption{Principal selected Swin predicted-category proportions at 80\% MSP (\%). Columns retain every seed; each contains 1,152 clips}\label{tab:externalcategories}',r'\centering\small\begin{tabular}{lrrrrrr}\toprule',r' & \multicolumn{3}{c}{Soft} & \multicolumn{3}{c}{Uniform}\\',r'Category & 17 & 42 & 89 & 17 & 42 & 89\\\midrule']
    for label in LABELS[:7]:
        values=[100*float(one(categories,backbone='swin_t',condition=c,state='selected',seed=s,population='msp80',predicted_category=label)['proportion']) for c in ('soft','uniform') for s in SEEDS]
        category_table.append(label.title()+' & '+' & '.join(f(v) for v in values)+r'\\')
    category_table += [r'\bottomrule\end{tabular}\end{table}']
    masses=[100*float(one(refs,backbone='swin_t',condition='soft',state='selected',seed=17,population='all_eligible',reference='raw9',reference_category=l)['proportion']) for l in LABELS]
    sup=[r'\clearpage\section*{Post hoc external output diagnostics}',
         'The 12 October 2026 addendum was written before these computations, but after the external outcomes were known. No fitting, peak selection or new inferential test was performed. Exports include all 36 frozen states and both populations. The displays below show all three selected Swin Soft/Uniform seeds, not only favorable ones.',
         '\n'.join(category_table),
         r'Across all 1,440 clips, raw human-vote mass (\%) is '+', '.join(f'{l}: {f(v)}' for l,v in zip(LABELS,masses))+'. Accepted-set vote masses are exported separately for every state and seed; model counts and reference vote mass are distinct quantities.',
         r'\begin{figure}[H]\centering\includegraphics[width=\linewidth]{'+REL+r'/figures/ExternalVoteMatrices.pdf}',
         r'\caption{Human-vote reference versus predicted category for all principal selected Swin Soft/Uniform seeds at 80\% MSP. Each cell divides the vote mass assigned to that predicted column by the row total; it is not single-label confusion or frame accuracy. Calm and none rows remain although the model has no corresponding outputs. Every panel contains 1,152 clips. All 36 states, both populations, raw contributions, row masses and both references are in Online Resource 8. Rows with no vote mass are undefined, not zero. The common scale is 0--1}\end{figure}',
         r'\clearpage\begin{figure}[H]\centering\includegraphics[width=\linewidth]{'+REL+r'/figures/ExternalTemporal.pdf}',
         r'\caption{Predictions across all eight temporal bins for selected Swin Soft/Uniform and all three seeds. Left: mean neutral probability. Right: agreement between frame-level and mean-probability clip predictions from the same model. Each bin has 1,440 valid frames from 1,440 clips; each state has 11,520 frames. Human judgments are clip-level, so agreement is not frame accuracy. Exports include all categories and all 36 states. Temporal patterns do not establish why external disagreement occurs}\end{figure}',
         'Acquisition checks verify video-only speech IDs, ten responses per clip, primary response mapping, frame-bin indices, shared crop validity and canonical category order. Every clip has eight valid frames. Independently summing retained float32 frame probabilities in float64 reproduces the clip mean exactly (maximum absolute difference zero). Raw/merged scoring preserves predictions and accepted IDs. Diagnostic identities use tolerance $10^{-10}$ for vote-mass summation and $10^{-14}$ for clip aggregation; probability normalization tolerance is $10^{-6}$.']
    write('posthoc_supplement.tex','\n\n'.join(sup))
    numeric={'internal_primary_pp':internal,'external_primary':p,'selected_swin_accounting':{'|'.join(k):v for k,v in means.items()},'neutral_accepted_percentages':neutral,'probabilities_ensembled':False,'states':36,'training_runs_added':0}
    (HERE/'checks/generated_numeric_claims.json').write_text(json.dumps(numeric,indent=2))
    from distribution_results import main as complete_distribution_report
    complete_distribution_report()
    from final_text import main as finalize_scientific_narrative
    finalize_scientific_narrative()
    from visual_figures import main as explain_targets_and_transfer
    explain_targets_and_transfer()
    from apply_editorial import main as restore_reviewed_academic_narrative
    restore_reviewed_academic_narrative()
    print(json.dumps({'status':'passed','generated_figures':['ExternalContrasts','ExternalVoteMatrices','ExternalTemporal','TargetControls','ExternalPipeline','Fig7'],'seed_rows':6,'accounting_rows':8,'new_training_runs':0},indent=2))

if __name__=='__main__':main()
