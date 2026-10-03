"""Non-mutating extension of the retained asset builder, using the merged ledger."""
import csv,hashlib,importlib.util,json,re,shutil
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
OUT=Path(__file__).resolve().parent;ROOT=OUT.parent
CONDITIONS=('hard','tie_hard','uniform','tie_uniform','soft')
LABELS={'hard':'Hard','tie_hard':'Tie-hard','uniform':'Uniform','tie_uniform':'Tie-Uniform','soft':'Soft'}
COLORS={'hard':'#235B83','tie_hard':'#6A5484','uniform':'#447651','tie_uniform':'#B48C29','soft':'#A34227'}
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    assert json.loads((OUT/'audit.json').read_text())['status']=='passed'
    # Copy—not modify—the historical generator. Its four-condition outputs remain
    # explicitly historical comparisons; new five-condition tables are additional.
    old_builder=(ROOT/'reviewer_followup/build_assets.py').read_text()
    old_builder=old_builder.replace('reviewer_followup/','evidence_revision/').replace('ambiguity-preserving','dominant-confidence-preserving').replace("'build_assets.py'","'retained_assets.py'")
    (OUT/'retained_assets.py').write_text(old_builder)
    source=(ROOT/'output/submission-github-2026-09-30/source/main.tex').read_text()
    source=source.replace('reviewer_followup/','evidence_revision/').replace('ambiguity-preserving','dominant-confidence-preserving')
    (OUT/'manuscript.tex').write_text(source)
    bibliography=(ROOT/'reviewer_followup/references.bib').read_text()+r'''
@inproceedings{khurana2024crowd,
 author={Khurana, Urja and Nalisnick, Eric and Fokkens, Antske and Swayamdipta, Swabha},
 title={{Crowd-Calibrator}: Can Annotator Disagreement Inform Calibration in Subjective Tasks?},
 booktitle={Conference on Language Modeling}, year={2024},
 url={https://openreview.net/forum?id=VWWzO3ewMS}
}
@misc{vats2023swin,
 author={Vats, Arpita and Chadha, Aman},
 title={Facial Expression Recognition using Squeeze and Excitation-powered Swin Transformers},
 year={2023}, howpublished={arXiv:2301.10906v7}, note={Unpublished preprint},
 doi={10.48550/arXiv.2301.10906}, url={https://arxiv.org/abs/2301.10906}
}
@misc{wang2026disagreement,
 author={Wang, Yiming and Li, Frederick W. B. and Wang, Jingyun},
 title={Predicting Human Disagreement for Calibrated Dynamic Facial Expression Recognition},
 year={2026}, howpublished={arXiv:2609.17130v1}, note={Unpublished preprint},
 url={https://arxiv.org/abs/2609.17130}
}
'''
    (OUT/'references.bib').write_text(bibliography)
    spec=importlib.util.spec_from_file_location('retained_assets',OUT/'retained_assets.py');builder=importlib.util.module_from_spec(spec);spec.loader.exec_module(builder);builder.main()
    with (OUT/'reported_results.csv').open(newline='') as f:ledger=list(csv.DictReader(f))
    def rows(kind,**criteria):return [r for r in ledger if r['record_type']==kind and all(r[k]==str(v) for k,v in criteria.items())]
    def avg(rr,key,scale=1):return np.mean([float(r[key])*scale for r in rr])
    def table(name,caption,label,columns,head,body,note=''):
        content='\\begin{table}[t]\n\\caption{'+caption+'}\\label{'+label+'}\n\\centering\\begin{tabular}{'+columns+'}\\toprule\n'+head+r'\\\midrule'+'\n'
        content+='\n'.join(' & '.join(row)+r'\\' for row in body)+'\n'+r'\bottomrule\end{tabular}'+'\n'+note+'\n\\end{table}\n'
        (OUT/'generated'/name).write_text(content)
    cross=[]
    for state in ('selected','epoch12'):
        for predictor in ('uniform','soft'):
            cross.append([state,LABELS[predictor]]+[f'{avg(rows("cross_evaluation",state=state,predictor=predictor,selector=selector),"risk",100):.3f}' for selector in ('uniform','soft')])
    table('cross.tex',r'Cross-evaluation at 80\% coverage: mean complete-vote disagreement (\%) across seeds. Rows hold predictions fixed; columns hold accepted images fixed','tab:cross','llrr','State & Predictor & Uniform set & Soft set',cross)
    f1=[];retention=[];epochs=[]
    for c in CONDITIONS:
        f1.append([LABELS[c]]+[f'{avg(rows("same_population_f1",condition=c,state="selected",population=pop,reference=ref),"weighted_f1"):.4f}' for pop,ref in [('all','original'),('majority_subset','original'),('majority_subset','crowd_majority')]])
        retention.append([LABELS[c]]+[', '.join(str(int(float(r['accepted_n']))) for r in sorted(rows('retention',condition=c,stratum=st),key=lambda r:int(r['seed']))) for st in ('full','partial','zero')])
        epochs.append([LABELS[c]]+[str(int(float(r['epoch']))) for r in sorted(rows('selected_epoch',condition=c),key=lambda r:int(r['seed']))])
    table('same_f1.tex','Selected-model weighted F1 with population and reference separated; means across three seeds','tab:samef1','lrrr','Targets & Original/all & Original/subset & Crowd/subset',f1,'All: 3,275 images; subset: the same 2,484 supported-majority images in the last two columns. No retraining or relabeling is performed.')
    table('retention.tex',r'Accepted image counts at global 80\% coverage, seeds 17, 42, 89 in that order. Population sizes: full 1,732; partial 1,527; zero 16','tab:retention','lrrr','Targets & Full mass & Partial mass & Zero mass',retention,'Each seed retains 2,620 images in total. Zero-mass images have disagreement one under every supported prediction.')
    table('epochs.tex','Selected epochs under the common distribution-distance criterion','tab:epochs','lrrr','Targets & Seed 17 & Seed 42 & Seed 89',epochs)
    tie=[]
    for state in ('selected','epoch12'):
        rr=rows('contrast',left='soft',right='tie_uniform',state=state)[0]
        tie.append([state,f'{avg(rows("common",condition="tie_uniform",state=state,requested_coverage=.8),"risk",100):.3f}',f'{avg(rows("common",condition="soft",state=state,requested_coverage=.8),"risk",100):.3f}',f'{float(rr["point_difference_pp"]):+.3f}',f'[{float(rr["lower_pp"]):+.3f}, {float(rr["upper_pp"]):+.3f}]'])
    table('tie_control.tex',r'Secondary matched Tie-Uniform comparison at 80\% coverage. Risks are percentages; differences and conditional 95\% intervals are percentage points','tab:tieuniform','lrrrr','State & Tie-Uniform & Soft & Difference & Interval',tie)
    table('phases.tex','Adaptive experimental phases on the same source; every phase uses seeds 17, 42, 89 and selected/fixed-epoch evaluation','tab:phases','llrl','Phase & Target conditions & Runs & Prior outcomes known',[
        ['1','Hard, Soft','6','Historical results'],['2','Uniform, Tie-hard','6','Phase 1'],['3','Tie-Uniform','3','Phases 1 and 2']],'All phases retain the same data roles and matched within-seed optimization. These are not independent-source replications.')
    primary_rows=[];sensitivity=[]
    for c in CONDITIONS:
        rr=rows('common',condition=c,state='selected',requested_coverage=.8);v=np.array([100*float(r['risk']) for r in rr])
        primary_rows.append([LABELS[c],f'${v.mean():.2f} \\pm {v.std(ddof=1):.2f}$']+[f'{avg(rr,key,100):.2f}' for key in ('unsupported','supported_ambiguity','excess')])
        sensitivity.append([LABELS[c],f'{v.mean():.3f}',f'{avg(rows("common",condition=c,state="epoch12",requested_coverage=.8),"risk",100):.3f}'])
    table('primary_table.tex',r'Selected-checkpoint risk at 80\% coverage (2,620 of 3,275 images). Risk is mean $\pm$ sample SD; components are mean percentage points','tab:primary','lrrrr',r'Targets & Risk (\%) & Unsupported & Ambiguity & Excess',primary_rows,'Accepted identities differ; components provide accounting, not causal attribution.')
    table('sensitivity_table.tex',r'Selected versus fixed epoch-12 complete-vote risk at 80\% coverage (\%); means across three paired seeds','tab:sensitivity','lrr','Targets & Selected & Epoch 12',sensitivity)
    new_text=r'''\subsection{Fixed-image comparisons and reference effects}
Table~\ref{tab:cross} holds the accepted image set fixed within each column.
\input{evidence_revision/generated/cross.tex}
'''
    u_u=avg(rows('cross_evaluation',state='selected',predictor='uniform',selector='uniform'),'risk',100)
    s_u=avg(rows('cross_evaluation',state='selected',predictor='soft',selector='uniform'),'risk',100)
    u_s=avg(rows('cross_evaluation',state='selected',predictor='uniform',selector='soft'),'risk',100)
    s_s=avg(rows('cross_evaluation',state='selected',predictor='soft',selector='soft'),'risk',100)
    overlap=rows('accepted_overlap',state='selected')
    new_text+=f'On Uniform-selected images, Soft minus Uniform is {s_u-u_u:+.3f} pp; on Soft-selected images it is {s_s-u_s:+.3f} pp. Within a column, annotations and the floor are identical, so the risk difference is also a category-choice excess difference on that image set. Holding the Uniform predictor fixed, changing from its own set to the Soft set changes risk by {u_s-u_u:+.3f} pp; for the Soft predictor the corresponding change is {s_s-s_u:+.3f} pp. Thus, the own-set contrast is not a consistent category-choice improvement on a fixed image set. Predictor and selector interact; these cross-evaluations are not a unique causal decomposition. The mean Jaccard overlap is {avg(overlap,"jaccard"):.4f}; individual intersections and overlap fractions are released.\n'
    new_text+=r'''\input{evidence_revision/generated/same_f1.tex}
In Table~\ref{tab:samef1}, the last two columns hold the image population fixed while changing the categorical reference. The first two hold the original-label reference fixed while changing the population. Their differences therefore disentangle two sources of the superficially large original-versus-crowd F1 contrast. The reference confusion matrix is supplied in the supplement; neither reference is treated as emotional truth.
\begin{figure}[t]\centering\includegraphics[width=\linewidth]{evidence_revision/figures/RiskDifference.pdf}
\caption{Soft-minus-Uniform complete-vote disagreement versus common coverage. Thin curves are paired seeds; the thick curve is their mean. Curves use each model's own ranking, not identical image sets. The vertical marker identifies the primary 80\% endpoint; this descriptive curve has no simultaneous uncertainty band}\label{fig:difference}\end{figure}
\subsection{Tie-preserving dominant-confidence control}
Table~\ref{tab:tieuniform} reports both checkpoint states for the new secondary comparison.
\input{evidence_revision/generated/tie_control.tex}
'''
    for state in ('selected','epoch12'):
        rr=rows('contrast',left='soft',right='tie_uniform',state=state)[0];seedrows=sorted(rows('seed_contrast',left='soft',right='tie_uniform',state=state),key=lambda r:int(r['seed']))
        new_text+=state.capitalize()+': paired Soft-minus-Tie-Uniform differences are '+', '.join(f'{float(r["difference_pp"]):+.3f}' for r in seedrows)+' pp for seeds 17, 42, and 89. '
        lo,hi=float(rr['lower_pp']),float(rr['upper_pp'])
        new_text+=('The conditional interval lies below zero. ' if hi<0 else 'The conditional interval lies above zero. ' if lo>0 else 'The conditional interval crosses zero; this does not establish equivalence. ')
    new_text+='This secondary comparison removes canonical tie asymmetry from Uniform but still changes target entropy and class marginals relative to Soft. It cannot isolate a unique minority-category identity mechanism.\n\\input{evidence_revision/generated/epochs.tex}\nTraining-loss and selection-distance trajectories are supplied for every run in the supplement. Loss scales differ by target construction and are not directly comparable evidence of better fitting.\n'
    new_text+=r'''\subsection{Retention of unsupported categories}
Table~\ref{tab:retention} retains individual seed counts rather than only a mean ordering.
\input{evidence_revision/generated/retention.tex}
'''
    new_text+=f'Soft retains {avg(rows("retention",condition="soft",stratum="zero"),"accepted_n"):.2f} of the 16 zero-supported-mass images on average, versus {avg(rows("retention",condition="uniform",stratum="zero"),"accepted_n"):.2f} for Uniform and {avg(rows("retention",condition="tie_uniform",stratum="zero"),"accepted_n"):.2f} for Tie-Uniform. Better aggregate risk therefore does not imply better rejection of wholly unsupported annotations. These small counts are descriptive, not an estimated deployment failure rate. Contempt, unknown and not-a-face vote-mass breakdowns are supplied separately rather than interpreted as one semantic category.\n'
    for pop in ('calibration','test'):
        values=[avg(rows('unsupported_categories',condition='hard',population=pop,scope='all',category=cat),'mass',100) for cat in ('contempt','unknown','not_a_face')]
        new_text+=f'Before confidence selection, {pop} vote mass is {values[0]:.3f}\\% contempt, {values[1]:.3f}\\% unknown, and {values[2]:.3f}\\% not-a-face. '
    new_text+='These are vote fractions, not counts of images; the accepted-set breakdown in the supplement conditions on each ranking.\n'
    (OUT/'generated/revision_results.tex').write_text(new_text)
    results=(OUT/'generated/results.tex').read_text().replace('all four target representations','all five target representations').replace('Figure~\\ref{fig:contrasts} shows risk--coverage curves','Figure~\\ref{fig:contrasts} shows original-four-condition risk--coverage curves').replace('(a) Mean selected-checkpoint risk--coverage curves across three seeds;','(a) Original four conditions: mean selected-checkpoint risk--coverage curves across three seeds;')
    old_contrast_figure=re.search(r'\\begin\{figure\}\[t\].*?\\label\{fig:contrasts\}.*?\\end\{figure\}',results,re.S).group(0)
    results=results.replace(old_contrast_figure,'')
    results=results.replace('Figure~\\ref{fig:contrasts} shows original-four-condition risk--coverage curves and selected/fixed-epoch contrasts.','Original-four-condition risk--coverage curves and selected/fixed-epoch contrast intervals are retained in the supplement; the direct primary difference curve appears below.')
    results=results.replace('The majority-only F1 column concerns a restricted reference task, not an interchangeable performance improvement over original-label F1.','Reference-specific classification outcomes are separated by both image population and label reference below.')
    results=results.replace('\\subsection{Calibration transfer and finite-sample diagnostics}',new_text+'\n\\subsection{Calibration transfer and finite-sample diagnostics}')
    results=results.replace('Across four conditions,','Across the original four conditions,')
    results=results.replace('(a) Three-way mean risk accounting','Original four conditions: (a) three-way mean risk accounting').replace('are reported separately in Online Resource 1','are reported in Table~\\ref{tab:retention} and Online Resource 1')
    (OUT/'generated/results.tex').write_text(results)
    # Keep old four-condition panels, clearly scoped; add the requested direct curve.
    plt.rcParams.update({'font.size':10,'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
    fig,ax=plt.subplots(figsize=(6.2,3.3));curves=[]
    for seed in (17,42,89):
        rr=sorted(rows('risk_difference_curve',seed=seed),key=lambda r:int(r['accepted_n']));x=np.array([float(r['coverage']) for r in rr]);y=np.array([float(r['difference_pp']) for r in rr]);curves.append(y);ax.plot(x[x>=.1]*100,y[x>=.1],lw=.9,color={17:'#235B83',42:'#B48C29',89:'#447651'}[seed],label=f'Seed {seed}')
    ax.plot(x[x>=.1]*100,np.mean(curves,axis=0)[x>=.1],color='#A34227',lw=2,label='Mean');ax.axhline(0,color='gray',lw=.7);ax.axvline(80,color='gray',ls=':',lw=.8);ax.set(xlim=(10,100),xlabel='Coverage (%)',ylabel='Soft - Uniform risk (pp)');ax.legend(ncol=2,fontsize=9);builder.save(fig,'RiskDifference')
    fig,axes=plt.subplots(5,2,figsize=(6.2,9),sharex=True)
    for i,c in enumerate(CONDITIONS):
        for seed in (17,42,89):
            rr=sorted(rows('learning_curve',condition=c,seed=seed),key=lambda r:int(r['epoch']));xx=[int(r['epoch']) for r in rr]
            for j,key in enumerate(('train_loss','selection_distance')):axes[i,j].plot(xx,[float(r[key]) for r in rr],label=f'Seed {seed}',lw=.8)
        axes[i,0].set_ylabel(LABELS[c],fontsize=9)
    axes[0,0].set_title('Training loss');axes[0,1].set_title('Selection distance');axes[-1,0].set_xlabel('Epoch');axes[-1,1].set_xlabel('Epoch');axes[0,1].legend(fontsize=8);fig.tight_layout();builder.save(fig,'LearningCurves')
    mat=np.zeros((7,7),int);classes=('angry','disgust','fear','happy','neutral','sad','surprise')
    for r in rows('reference_confusion'):mat[classes.index(r['original_class']),classes.index(r['crowd_class'])]=int(r['count'])
    fig,ax=plt.subplots(figsize=(6.2,4.5));im=ax.imshow(mat,cmap='Blues');ax.set_xticks(range(7),classes,rotation=35,ha='right');ax.set_yticks(range(7),classes);ax.set(xlabel='Crowd majority label',ylabel='Original FER2013 label')
    for i in range(7):
        for j in range(7):ax.text(j,i,str(mat[i,j]),ha='center',va='center',fontsize=9,color='white' if mat[i,j]>mat.max()/2 else 'black')
    fig.colorbar(im,ax=ax,label='Images');fig.tight_layout();builder.save(fig,'ReferenceConfusion')
    # Retain the old application figure in supplement; replace the main flow.
    shutil.copy2(OUT/'figures/Fig1.pdf',OUT/'figures/ApplicationFlow.pdf')
    fig,ax=plt.subplots(figsize=(6.2,3.3));ax.axis('off');ax.set(xlim=(0,1),ylim=(0,1))
    boxes=[(.02,.65,.28,.22,'Training: 28,709\n5 target conditions\n3 matched seeds'),(.36,.65,.28,.22,'Selection: 1,667\n12-epoch budget\nCommon criterion'),(.70,.65,.28,.22,'15 selected +\n15 epoch-12 models\nFrozen by phase'),(.70,.17,.28,.24,'Calibration: 1,642\nPolicies for original\nfour conditions'),(.36,.17,.28,.24,'Test: 3,275\nSame-coverage +\nfixed-image checks'),(.02,.17,.28,.24,'Risk accounting\nReference effects\nUnsupported votes')]
    for x,y,w,h,t in boxes:ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.005',facecolor='#F1F4F6',edgecolor='#243849'));ax.text(x+w/2,y+h/2,t,ha='center',va='center',fontsize=9)
    for start,end in [((.30,.76),(.35,.76)),((.64,.76),(.69,.76)),((.84,.65),(.84,.42)),((.70,.29),(.65,.29)),((.36,.29),(.31,.29))]:ax.annotate('',xy=end,xytext=start,arrowprops={'arrowstyle':'->'})
    ax.text(.5,.035,'Exact-pixel isolation; adaptive phases; not an independent dataset',ha='center',fontsize=9);builder.save(fig,'Fig1')
    # Main text: compact system boundary, new control, honest adaptive scope.
    start=source.index('\\subsection{Local prototype and verification boundary}');end=source.index('\\section{Results}',start)
    prototype=source[start:end].split('AI assistance (OpenAI Codex)',1)[0]
    compact=r'''\subsection{Implementation and verification boundary}
The sender-controlled local prototype is an implementation artifact, not the validation setting. It uses MediaPipe detection \cite{lugaresi2019mediapipe} and the shared image transform, class order and model head; invalid, stale or unapproved cues are suppressed. Detailed runtime, local-processing and transport boundaries are in the supplement. Code checks cover numerical metrics, checkpoint contracts and failure states. AI assistance (OpenAI Codex) was used for literature checking, experimental and analysis code, manuscript drafting, and technical verification. The authors retain responsibility for methods, results, source accuracy and final submission.

'''
    source=source[:start]+compact+source[end:]
    source=source.replace('Twelve matched training runs','Fifteen matched training runs').replace('three matched groups of four conditions','three matched groups of five conditions').replace('all four conditions. The latter','all five conditions. The latter')
    source=source.replace('The initial six hard/soft runs preceded the review; the six control runs and follow-up endpoints were specified after those results were known and before the controls were trained.','The initial six hard/soft runs preceded review; six Uniform/Tie-hard control runs followed their known outcomes; three Tie-Uniform runs followed the second review. Each adaptive phase was specified before its training and frozen before its calibration/test inference.')
    source=source.replace('\\input{evidence_revision/generated/roles.tex}','\\input{evidence_revision/generated/roles.tex}\n\\input{evidence_revision/generated/phases.tex}')
    source=source.replace('RQ1: does the detailed allocation of minority votes add selective-prediction value beyond image-specific target softening, and does arbitrary tie resolution explain the hard/soft contrast? RQ2: how much selected disagreement comes from unsupported votes, supported-category ambiguity, and excess category-choice error?','RQ1: how do detailed minority allocation and tie-preserving dominant-confidence softening compare under matched training? RQ2: do own-set risk differences persist when predictions are compared on identical accepted images, and how do label reference and unsupported votes affect interpretation?')
    source=source.replace('Separate experimental and application flows. Checkpoint selection does not use the calibration subset; calibrated thresholds are evaluated offline and are not installed as a validated runtime guarantee. The application ends in a local preview, without recipient transport','Offline experimental flow. Calibration transfer analyses retain the original four conditions; Tie-Uniform is a targeted secondary control. Checkpoint selection does not use calibration; all phases are adaptive rather than independent confirmation')
    sentence='These comparisons do not test discarding ambiguous images, manual relabeling, or training an expanded output vocabulary.'
    newcontrol=r'''A third control, Tie-Uniform, preserves every tied maximum. With $k_i=|T_i|$, its target is
\begin{equation}
t_{ic}^{\mathrm{TU}}=\begin{cases}m_i,&c\in T_i,\\(1-k_i m_i)/(7-k_i),&c\notin T_i,\ k_i<7.\end{cases}
\end{equation}
When $k_i=7$, all targets are $1/7$. Zero-supported-mass rows retain zero loss. This equals Uniform on unique maxima but avoids privileging one tied category. It preserves dominant confidence, not entropy or aggregate class marginals. Three additional matched runs test this secondary representation contrast, not a uniquely isolated mechanism.
'''
    source=source.replace(sentence,sentence+'\n\n'+newcontrol)
    source=source.replace('Each six-run phase is frozen','Each training phase is frozen').replace('Only Soft-minus-Uniform is the primary follow-up contrast','Soft-minus-Tie-Uniform is a prespecified secondary contrast in the final adaptive phase. Only Soft-minus-Uniform is the primary follow-up contrast')
    source=source.replace('The six-control protocol was fixed before its execution but after the hard/soft results and reviewer suggestions were known.','The six-control and later three-run Tie-Uniform protocols were each fixed before execution but after the preceding outcomes and reviews were known.')
    source=source.replace('Earlier PrivateTest outcomes informed both experimental phases.','Earlier PrivateTest outcomes informed the adaptive follow-up phases.')
    source=source.replace('The implemented system preserves sender choice and suppresses invalid or stale observations, while stopping at a local conversation preview. Its scientific relevance is the concrete setting in which annotation-sensitive reliability matters. The present evidence does not establish what a sender feels or whether attaching a cue improves a conversation.','')
    related='Crowd-Calibrator connects annotation disagreement with calibration and selective prediction in subjective language tasks \\cite{khurana2024crowd}. Our contribution is not that conceptual connection; it is the matched static-image target comparison and the fixed-set, label-reference and omitted-vocabulary diagnostics in this seven-output FER setting. Its language-task scores are not comparable FER benchmarks.\n\n'
    source=source.replace('Selective classification examines',related+'Selective classification examines')
    source=source.replace('(unpublished preprint, 2023, \\url{https://arxiv.org/abs/2301.10906})','(unpublished preprint, 2023) \\cite{vats2023swin}').replace('(2026, \\url{https://arxiv.org/abs/2609.17130})','(2026) \\cite{wang2026disagreement}')
    source=source.replace('Online Resources 2--5 supply selected and epoch-12 inference checkpoints for Hard, Soft, Uniform, and Tie-hard, respectively.','Online Resources 2--6 supply selected and epoch-12 inference checkpoints for Hard, Soft, Uniform, Tie-hard, and Tie-Uniform, respectively.').replace('/releases/tag/v1.0.0','/releases/tag/v1.1.0')
    source=source.replace('code, both protocols,','code, all three adaptive protocols,').replace('Online Resources 2--5: all 24','Online Resources 2--6: all 30')
    (OUT/'manuscript.tex').write_text(source)
    # Replace the old abstract/interpretation rather than add stronger claims.
    primary=rows('contrast',left='soft',right='uniform',state='selected')[0]
    tu=rows('contrast',left='soft',right='tie_uniform',state='selected')[0]
    abstract=r'''\textbf{Purpose:} Crowd-vote supervision can alter confidence ranking without consistently improving category choice. We examine this distinction when a facial-expression predictor omits categories in the annotation vocabulary.
\textbf{Methods:} Fifteen matched Swin-T runs compare hard, tie-aware hard, dominant-confidence-preserving uniform-minority, tie-preserving uniform-minority, and full soft targets on FER2013 images with FER+ votes. Exact-pixel-isolated selection, calibration and testing support common-coverage, fixed-image and fixed-epoch comparisons. Complete-vote disagreement is separated into unsupported mass, supported ambiguity and category-choice excess. Three adaptive phases reuse one image source; they are not independent confirmation.
'''
    abstract+=f'\\textbf{{Results:}} At 80\\% coverage, selected-model Soft-minus-Uniform disagreement is {float(primary["point_difference_pp"]):+.2f} percentage points; Soft-minus-Tie-Uniform is {float(tu["point_difference_pp"]):+.2f}. On Uniform-selected images, Soft instead differs by {s_u-u_u:+.2f} points. Changing the reference on identical majority-subset images substantially changes F1. Soft retains more zero-supported-mass images than Uniform. All 36 nonempty empirical test operating points for the original four conditions exceed their calibration targets.\n'
    abstract+=r'''\textbf{Conclusion:} Target representation, accepted-image composition, categorical reference and vocabulary mismatch jointly shape apparent reliability. The observed within-source contrasts support diagnostic reporting, not a novel architecture, universal superiority, deployment risk guarantee or validated messaging benefit.'''
    (OUT/'generated/abstract.tex').write_text(abstract)
    discussion=r'''The largest original Hard-to-Soft contrast primarily reflects changing the target representation rather than evidence that detailed minority identities alone are essential. Uniform is a strong control, and Tie-Uniform removes its canonical tie asymmetry. Neither controls entropy or aggregate class marginals. The cross-evaluation matrix additionally shows that predictor comparisons depend on the accepted image set. An own-ranking advantage should not be promoted to a general claim of improved category choice.

The matched-reference table makes another reporting distinction concrete: changing image eligibility and changing labels both alter F1. Full-vote disagreement avoids silently excluding non-majority images, but remains conditional on the finite crowd panel. The zero-supported-mass retention counts are an adverse result that aggregate selected risk conceals. Contempt, unknown and not-a-face have different meanings; a seven-output model cannot resolve them simply by reporting high MSP.
'''
    hard_risk=avg(rows('common',condition='hard',state='selected',requested_coverage=.8),'risk',100)
    tie_risk=avg(rows('common',condition='tie_uniform',state='selected',requested_coverage=.8),'risk',100)
    tie_interval=rows('contrast',left='soft',right='tie_uniform',state='selected')[0]
    interpretation=('The residual own-ranking advantage persists under this tie-preserving control within the conditional interval.' if float(tie_interval['upper_pp'])<0 else 'The tie-preserving control does not support a clear residual Soft advantage under this conditional interval; a zero-crossing interval is not equivalence.' if float(tie_interval['lower_pp'])<=0 else 'The tie-preserving control has lower own-ranking risk than Soft under this conditional interval, so the earlier advantage is control-dependent.')
    discussion+=f'Tie-Uniform selected risk differs from Uniform by {tie_risk-u_u:+.3f} pp. {interpretation} Because this adaptive check still uses the same source, three seeds and one model family, it is a representation sensitivity rather than independent replication.\n'
    discussion+=f'For scale, Uniform lowers selected-checkpoint risk relative to Hard by {hard_risk-u_u:.2f} pp, compared with {hard_risk-s_s:.2f} pp for Soft. The additional {u_u-s_s:.2f} pp at 2,620 retained images corresponds to about {(u_u-s_s)*26.2:.1f} fewer expected disagreements with an empirical annotation draw, not that many corrected images: accepted identities differ and losses are fractional. This arithmetic is not a causal fraction of improvement.\n'
    (OUT/'generated/discussion.tex').write_text(discussion)
    conclusion=r'''This matched, adaptive within-source study finds that conclusions about crowd supervision depend on target softening, tie handling, accepted-image composition and reference labels. Detailed votes are not shown to be universally necessary, and lower own-set risk is not a uniform fixed-image prediction improvement. Unsupported-vote retention and calibration transfer failures constrain runtime interpretations. Releasing all selected/fixed-epoch weights, predictions and diagnostics makes these bounded findings auditable; independent-image and naturalistic messaging evaluation remain absent.'''
    (OUT/'generated/conclusion.tex').write_text(conclusion)
    sup=(OUT/'generated/supplement.tex').read_text().replace('all four conditions','all five conditions').replace('both protocols','all three adaptive protocols').replace('Online Resources 2--5','Online Resources 2--6')
    sup=sup.replace('Every paired seed contrast at 80\\% coverage','Original four-condition paired seed contrasts at 80\\% coverage').replace('Selected epochs from the common squared-distance criterion.','Original four conditions: selected epochs from the common squared-distance criterion.').replace('Individual paired seed differences at the common-coverage endpoint:','Original four conditions: individual paired seed differences at the common-coverage endpoint:').replace('Supported-mass strata: means over three selected-checkpoint seeds.','Original four conditions, supported-mass strata: means over three selected-checkpoint seeds.').replace('Calibration half-split holdout-minus-fit risk gaps.','Original four conditions, calibration half-split holdout-minus-fit risk gaps.')
    appendix='\\clearpage\n\\section*{Prototype implementation boundary}\n'+prototype.replace('\\subsection{Local prototype and verification boundary}','').replace('\\cite{lugaresi2019mediapipe}','(MediaPipe Face Detection; main reference list)')+'\n'
    appendix+='\\clearpage\n'+old_contrast_figure.replace('\\label{fig:contrasts}','')+'\n'
    for fig,cap in [('ApplicationFlow','Historical four-condition offline flow and local application boundary; the main manuscript now isolates the experimental flow.'),('LearningCurves','Training loss and selection-distance trajectories for all fifteen runs. The target-dependent loss scales are not comparable performance endpoints.'),('ReferenceConfusion','Original-label versus supported crowd-majority reference counts on exactly the same 2,484 primary-test images. This matrix compares references, not predictions.')]:
        appendix+='\\clearpage\n\\begin{figure}[ht]\\centering\\includegraphics[width=.94\\linewidth]{evidence_revision/figures/'+fig+'.pdf}\\caption{'+cap+'}\\end{figure}\n'
    appendix+='\\clearpage\\section*{Unsupported category vote mass}\n\\begin{longtable}{lllrrr}\\toprule Targets & Population & Scope & Contempt & Unknown & Not-a-face\\\\\\midrule\\endhead\n'
    for c in CONDITIONS:
        for pop in ('calibration','test'):
            for scope in ('all','global80'):
                appendix+=LABELS[c]+' & '+pop+' & '+scope+' & '+' & '.join(f'{avg(rows("unsupported_categories",condition=c,population=pop,scope=scope,category=cat),"mass",100):.3f}' for cat in ('contempt','unknown','not_a_face'))+r'\\'+'\n'
    appendix+='\\bottomrule\\end{longtable}\nEntries are mean vote-mass percentages across three seeds, not proportions of images with that label. All-image masses are model-independent; global80 conditions on each model\'s ranking. The three categories do not share one semantic interpretation.\n'
    appendix+='\\input{evidence_revision/generated/tie_control.tex}\n\\input{evidence_revision/generated/epochs.tex}\n'
    appendix+='\\clearpage\\section*{Per-seed fixed-image cross-evaluation}\n\\begin{longtable}{llrrr}\\toprule State & Predictor & Seed & Uniform set & Soft set\\\\\\midrule\\endhead\n'
    for state in ('selected','epoch12'):
        for seed in (17,42,89):
            for predictor in ('uniform','soft'):
                appendix+=state+' & '+LABELS[predictor]+f' & {seed} & '+' & '.join(f'{avg(rows("cross_evaluation",state=state,seed=seed,predictor=predictor,selector=selector),"risk",100):.4f}' for selector in ('uniform','soft'))+r'\\'+'\n'
    appendix+='\\bottomrule\\end{longtable}\nRisk entries are complete-vote disagreement percentages at exactly 2,620 accepted images.\n'
    appendix+='\\begin{longtable}{llrrrr}\\toprule State & Seed & Intersection & Union & Jaccard & Overlap fraction\\\\\\midrule\\endhead\n'
    for state in ('selected','epoch12'):
        for seed in (17,42,89):
            rr=rows('accepted_overlap',state=state,seed=seed)[0]
            appendix+=state+f' & {seed} & '+str(rr['intersection_n'])+' & '+str(rr['union_n'])+' & '+f'{float(rr["jaccard"]):.4f} & {float(rr["overlap_fraction"]):.4f}'+r'\\'+'\n'
    appendix+='\\bottomrule\\end{longtable}\nOverlap fraction divides the intersection by 2,620, whereas Jaccard divides by the union. The two accepted sets can share most images while their differing tails change mean risk.\n'
    sup=sup.replace('\\end{document}',appendix+'\\end{document}');(OUT/'generated/supplement.tex').write_text(sup)
    sources={name:sha(OUT/name) for name in ('reported_results.csv','split_manifest.json','build_revision.py','retained_assets.py','manuscript.tex')}
    (OUT/'asset_sources.json').write_text(json.dumps(sources,indent=2))
    generated=list((OUT/'generated').glob('*.tex'))+list((OUT/'figures').glob('*.pdf'))+list((OUT/'figures').glob('*.eps'))+[OUT/'references.bib']
    (OUT/'generated_sha256.json').write_text(json.dumps({p.relative_to(OUT).as_posix():sha(p) for p in generated},indent=2))
    print('Generated bounded evidence revision without altering previous edition')
if __name__=='__main__':main()
