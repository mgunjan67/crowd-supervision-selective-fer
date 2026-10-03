"""Generate every numerical manuscript claim and result figure from one ledger."""
import csv,hashlib,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
OUT=Path(__file__).resolve().parent;ROOT=OUT.parent
GEN=OUT/'generated';FIG=OUT/'figures'
CONDITIONS=('hard','tie_hard','uniform','soft')
LABELS={'hard':'Hard','tie_hard':'Tie-hard','uniform':'Uniform','soft':'Soft'}
COLORS={'hard':'#235B83','tie_hard':'#6A5484','uniform':'#447651','soft':'#A34227'}
STYLES={'hard':'-','tie_hard':':','uniform':'-.','soft':'--'}
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.labelsize':10,'legend.fontsize':10,
    'xtick.labelsize':10,'ytick.labelsize':10,'pdf.fonttype':42,'ps.fonttype':42,
    'axes.spines.top':False,'axes.spines.right':False})

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(name,text):(GEN/name).write_text(text,encoding='ascii')
def save(fig,name):
    for extension in ('pdf','eps','png'):fig.savefig(FIG/f'{name}.{extension}',bbox_inches='tight',dpi=180)
    plt.close(fig)

def static():
    GEN.mkdir(exist_ok=True);FIG.mkdir(exist_ok=True)
    counts=json.loads((OUT/'split_manifest.json').read_text())['counts']
    role_text=r'''\begin{table}[t]
\caption{Operational roles after exact-pixel isolation. Counts are images, not people}\label{tab:roles}
\centering\begin{tabular}{lrl}\toprule
Source and role & Images & Use\\\midrule
'''
    for label,key,use in [('Training','train','Optimization'),('PublicTest: selection','selection','Checkpoint choice'),
        ('PublicTest: calibration','calibration','Threshold choice'),('PrivateTest: primary','test','Main evaluation'),
        ('PrivateTest: full','test_full','Prediction archive')]:
        role_text+=f'{label} & {counts[key]:,} & {use}'+r'\\'+'\n'
    write('roles.tex',role_text+r'\bottomrule\end{tabular}\end{table}'+'\n')
    fig,ax=plt.subplots(figsize=(6.2,4.4));ax.set(xlim=(0,1),ylim=(0,1));ax.axis('off')
    def box(x,y,w,h,text):
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.007,rounding_size=.01',
            facecolor='#F1F4F6',edgecolor='#243849',lw=.9))
        ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=10)
    def arrow(a,b):ax.annotate('',xy=b,xytext=a,arrowprops={'arrowstyle':'->','lw':1})
    ax.text(.015,.98,'a  Matched offline experiment',weight='bold',va='top',fontsize=10)
    box(.02,.74,.28,.17,'Training: 28,709\n4 target conditions\n3 matched seeds')
    box(.37,.74,.26,.17,'Selection: 1,667\n12-epoch budget\nMatched inputs')
    box(.70,.74,.28,.17,'12 selected\n12 final-epoch\nFrozen by phase')
    arrow((.30,.825),(.36,.825));arrow((.63,.825),(.69,.825))
    box(.70,.47,.28,.14,'Calibration: 1,642\nPolicies + checks')
    box(.37,.47,.26,.14,'Test: 3,275\nFixed coverage\nPolicy transfer')
    box(.02,.47,.28,.14,'Risk accounting\nUnsupported mass\nAmbiguity + excess')
    arrow((.84,.74),(.84,.62));arrow((.70,.54),(.64,.54));arrow((.37,.54),(.31,.54))
    ax.text(.50,.40,'Exact-pixel separation; not verified person-disjoint',ha='center',fontsize=10)
    ax.plot([.015,.985],[.35,.35],color='#65707A',lw=.8)
    ax.text(.015,.32,'b  Secondary local prototype',weight='bold',va='top',fontsize=10)
    box(.02,.12,.28,.12,'Camera crop\nOne eligible face')
    box(.37,.12,.26,.12,'Shared transform\nSwin-T + one gate')
    box(.70,.12,.28,.12,'Fresh + opt-in\nLocal preview only')
    arrow((.30,.18),(.36,.18));arrow((.63,.18),(.69,.18))
    ax.text(.50,.035,'No recipient transport or evaluated communication benefit',ha='center',fontsize=10)
    save(fig,'Fig1')
    fig,ax=plt.subplots(figsize=(6.2,3.1));ax.set(xlim=(0,1),ylim=(0,.88));ax.axis('off')
    box(.02,.54,.20,.24,'Input\n'+r'$3\times224\times224$')
    box(.28,.54,.27,.24,'Swin-T backbone\nPool + flatten\n768 features')
    box(.26,.14,.40,.22,'Bias-free gate\n768 -> 48 -> 768\nReLU, then sigmoid')
    box(.80,.54,.18,.24,'Linear\n768 -> 7\nwith bias')
    ax.text(.72,.66,r'$\odot$',ha='center',va='center',fontsize=17)
    arrow((.22,.66),(.28,.66));arrow((.55,.66),(.69,.66));arrow((.75,.66),(.80,.66))
    ax.plot([.57,.57,.39],[.66,.44,.44],color='#243849',lw=1);arrow((.39,.44),(.39,.36))
    ax.plot([.66,.72],[.25,.25],color='#243849',lw=1);arrow((.72,.25),(.72,.60))
    ax.text(.5,.025,'One gate after pooling; 73,728 additional parameters',ha='center',fontsize=10)
    save(fig,'FigS1')
    fig,ax=plt.subplots(figsize=(6.2,3.3));ax.set(xlim=(0,1),ylim=(0,1));ax.axis('off')
    ax.text(.02,.97,'Local expression-cue prototype: illustrative controls',weight='bold',va='top',fontsize=11)
    ax.text(.02,.86,'Synthetic text and score; not a participant record or measured prediction',fontsize=10)
    box(.02,.35,.40,.42,'Local camera preview\n\nNo facial image shown\n\nExample: neutral\nScore: 0.62 (uncalibrated)')
    box(.48,.35,.50,.42,'')
    ax.text(.51,.71,'Compose a message',weight='bold',fontsize=10)
    box(.51,.54,.44,.11,'I will send the draft tomorrow.')
    ax.add_patch(plt.Rectangle((.51,.47),.023,.04,fill=False,lw=1))
    ax.text(.55,.48,'Attach current expression cue',fontsize=10)
    ax.text(.51,.39,'Add to local preview',fontsize=10,color='#235B83')
    ax.text(.02,.23,'Default: off; camera off or invalid/stale observation: omit the cue',fontsize=10)
    ax.text(.02,.13,'Opt-in resets after each message. No face does not mean neutral.',fontsize=10)
    ax.text(.02,.03,'Messages remain local; no recipient-side transport is implemented.',fontsize=10)
    save(fig,'FigS2')

def main():
    assert json.loads((OUT/'audit.json').read_text())['status']=='passed'
    static()
    with (OUT/'reported_results.csv').open(newline='') as f:raw=list(csv.DictReader(f))
    def rows(kind,**criteria):return [r for r in raw if r['record_type']==kind and all(r[k]==str(v) for k,v in criteria.items())]
    def mean(data,key,scale=1):
        a=[scale*float(r[key]) for r in data if r[key]!=''];return float(np.mean(a)) if a else None
    def fmt(data,key,scale=1,digits=2):
        a=[scale*float(r[key]) for r in data if r[key]!='']
        if not a:return '--'
        return f'${np.mean(a):.{digits}f}'+(f' \\pm {np.std(a,ddof=1):.{digits}f}' if len(a)>1 else '')+'$'
    def common(c,state='selected',coverage=.8):return rows('common',condition=c,state=state,requested_coverage=coverage)
    def contrast(left,right,state='selected'):return rows('contrast',left=left,right=right,state=state)[0]
    def ctext(row):return f"{float(row['point_difference_pp']):+.2f} pp (conditional 95\\% interval [{float(row['lower_pp']):+.2f}, {float(row['upper_pp']):+.2f}])"
    primary=contrast('soft','uniform');old=contrast('soft','hard');fixed=contrast('soft','uniform','epoch12')
    assert int(primary['test_n'])==3275 and int(primary['accepted_n'])==2620
    delta,lo,hi=[float(primary[k]) for k in ('point_difference_pp','lower_pp','upper_pp')]
    if hi<0:
        conclusion='Full soft targets yielded lower selective disagreement than the dominant-confidence-preserving uniform-minority control under this matched protocol.'
    elif lo>0:
        conclusion='The dominant-confidence-preserving control had lower primary disagreement than full soft targets; detailed minority votes did not provide an advantage in this comparison.'
    else:
        conclusion='The primary contrast did not clearly separate detailed minority-vote allocation from dominant-confidence-preserving softening; this does not establish equivalence.'
    write('macros.tex','% All numerical claims are expanded into generated fragments.\n')
    abstract=r'\textbf{Purpose:} To test whether detailed crowd-vote distributions improve selective facial-expression prediction beyond simpler target softening, and diagnose reliability when prediction and annotation vocabularies differ. '+r'\textbf{Methods:} Twelve channel-gated Swin-T runs compare hard, tie-aware hard, dominant-confidence-preserving uniform-minority, and full soft targets across three matched seeds. Exact-pixel isolation separates training, checkpoint selection, calibration, and testing. Selected and fixed-epoch checkpoints are evaluated. Complete-vote disagreement is partitioned into unsupported mass, supported ambiguity, and excess error; calibration half-splits and a conservative fixed-grid comparator examine threshold transfer. '+r'\textbf{Results:} '+f'At 80\\% coverage on 3,275 test images, mean disagreement is {mean(common("hard"),"risk",100):.2f}\\% (hard), {mean(common("tie_hard"),"risk",100):.2f}\\% (tie-aware), {mean(common("uniform"),"risk",100):.2f}\\% (uniform-minority), and {mean(common("soft"),"risk",100):.2f}\\% (soft). Soft minus uniform-minority is {delta:+.2f} percentage points (conditional 95\\% pixel-cluster interval [{lo:+.2f}, {hi:+.2f}]). '+r'\textbf{Conclusion:} '+conclusion+' Confidence interpretation depends on the vote vocabulary and retained population. Findings concern one reused image source, not independent-domain reliability or communication benefit.\n'
    ca=rows('population',condition='hard',state='selected',population='calibration')
    te=rows('population',condition='hard',state='selected',population='test')
    composition=f'The test annotation floor exceeds calibration\'s by {mean(te,"floor",100)-mean(ca,"floor",100):.2f} percentage points, including {mean(te,"unsupported",100)-mean(ca,"unsupported",100):.2f} points of unsupported mass. '
    abstract=abstract.replace(r'\textbf{Conclusion:}',composition+r'\textbf{Conclusion:}')
    write('abstract.tex',abstract)
    # Table 2: explicitly three-way risk accounting, with all four controls.
    table=r'''\begin{table}[t]
\caption{Selected-checkpoint risk at 80\% common coverage (2,620 of 3,275 images). Risk is mean $\pm$ sample SD across three seeds; components are means in percentage points}\label{tab:primary}
\centering\begin{tabular}{lrrrr}\toprule
Targets & Risk (\%) & Unsupported & Ambiguity & Excess\\\midrule
'''
    for c in CONDITIONS:
        rr=common(c);table+=LABELS[c]+' & '+fmt(rr,'risk',100)+' & '+' & '.join(f'{mean(rr,k,100):.2f}' for k in ('unsupported','supported_ambiguity','excess'))+r'\\'+'\n'
    table+=r'''\bottomrule\end{tabular}
\smallskip\par\footnotesize The three components sum to risk, up to rounding. Accepted identities can differ between models; this is accounting, not causal attribution. Uniform preserves the dominant fraction but spreads remaining supported mass uniformly.
\end{table}
''';write('primary_table.tex',table)
    table=r'''\begin{table}[t]
\caption{Fixed-budget sensitivity and reference-specific metrics. Risk is at 80\% coverage; F1 is at full primary coverage using selected checkpoints. Values are means across three seeds}\label{tab:sensitivity}
\centering\begin{tabular}{lrrrr}\toprule
Targets & Selected & Epoch 12 & Original F1 & Majority F1\\\midrule
'''
    for c in CONDITIONS:
        rr=rows('classification',condition=c,state='selected')
        table+=LABELS[c]+f' & {mean(common(c),"risk",100):.2f} & {mean(common(c,"epoch12"),"risk",100):.2f} & {mean(rr,"original_weighted_f1"):.4f} & {mean(rr,"majority_weighted_f1"):.4f}'+r'\\'+'\n'
    rr=rows('classification',condition='soft',state='selected')[0]
    table+=r'\bottomrule\end{tabular}'+'\n'+r'\smallskip\par\footnotesize '+f'Risk is a percentage. Both F1 columns are weighted F1, but original labels cover 3,275 images whereas strict crowd-majority labels cover {int(rr["majority_n"]):,}. Original and crowd-majority labels disagree on {100*float(rr["reference_disagreement"]):.2f}\\% of that subset. These F1 values are not interchangeable FER scores. All per-seed values, macro F1, distances, and selected epochs are released.\n'+r'\end{table}'+'\n';write('sensitivity_table.tex',table)
    table=r'''\begin{table}[t]
\caption{Calibration-fitted policies applied unchanged to primary testing. Entries are mean coverage / risk (\%); coverage includes empty seeds, risk averages nonempty seeds}\label{tab:policies}
\centering\begin{tabular}{llcc}\toprule
Target & Targets & Empirical & Fixed-grid bound\\\midrule
'''
    for target in (.1,.2,.3):
        for c in CONDITIONS:
            text=[]
            for policy in ('empirical','hoeffding_grid'):
                rr=rows('policy',condition=c,state='selected',target=target,policy=policy,population='test')
                cov=mean(rr,'coverage',100);risk=mean(rr,'risk',100);nonempty=sum(r['risk']!='' for r in rr)
                text.append(f'{cov:.2f} / '+(f'{risk:.2f}' if risk is not None else '--')+f' ({nonempty}/3)')
            table+=f'{100*target:.0f}\\% & {LABELS[c]} & '+' & '.join(text)+r'\\'+'\n'
        if target!=.3:table+=r'\addlinespace'+'\n'
    table+=r'''\bottomrule\end{tabular}
\smallskip\par\footnotesize Parentheses give nonempty test seeds. An empty policy has undefined risk. Neither a nominal target nor the bound's unverified independence/unchanged-population assumptions establish a deployment guarantee. Different achieved risks are not an equal-risk coverage comparison.
\end{table}
''';write('policy_table.tex',table)
    # Figure 2: four curves and selected/final contrast checks.
    fig,(ax,bx)=plt.subplots(1,2,figsize=(6.2,3.0),gridspec_kw={'width_ratios':[1.12,1]},layout='constrained')
    for c in CONDITIONS:
        curves=[]
        for s in (17,42,89):
            rr=rows('curve',condition=c,seed=s,state='selected');xx=np.array([100*float(r['coverage']) for r in rr]);yy=np.array([100*float(r['risk']) for r in rr]);curves.append(yy)
        ax.plot(xx,np.mean(curves,axis=0),color=COLORS[c],ls=STYLES[c],label=LABELS[c],lw=1.6)
    ax.axvline(80,color='#777777',ls=':',lw=.8);ax.set(xlim=(10,100),ylim=(0,40),xlabel='Coverage (%)',ylabel='Vote disagreement (%)');ax.legend(frameon=False,loc='upper left')
    pairs=[('soft','uniform'),('soft','hard'),('tie_hard','hard'),('soft','tie_hard')]
    for state,marker,offset,color in [('selected','o',.10,'#253B50'),('epoch12','s',-.10,'#A34227')]:
        for y,(left,right) in enumerate(pairs):
            rr=contrast(left,right,state);v,l,h=[float(rr[k]) for k in ('point_difference_pp','lower_pp','upper_pp')]
            bx.hlines(y+offset,l,h,color=color,lw=1.2);bx.plot(v,y+offset,marker=marker,color=color,ms=4,label=state if y==0 else None)
    bx.axvline(0,color='#777777',ls=':',lw=.8);bx.set(yticks=range(4),yticklabels=['Soft - Uniform','Soft - Hard','Tie-hard - Hard','Soft - Tie-hard'],xlabel='Risk difference (pp)',ylim=(-.6,4.3));bx.invert_yaxis();bx.legend(frameon=False,loc='lower left',fontsize=9,ncol=2,columnspacing=.8,handlelength=1.2)
    for panel,axis in zip('ab',(ax,bx)):axis.text(-.17,1.02,panel,transform=axis.transAxes,weight='bold')
    save(fig,'Fig2')
    # Figure 3: actual supported-mass strata and additive global accounting.
    fig,(ax,bx)=plt.subplots(1,2,figsize=(6.2,3.0),layout='constrained')
    x=np.arange(4);bottom=np.zeros(4)
    for key,label,color,hatch in [('unsupported','Unsupported','#FFFFFF','///'),('supported_ambiguity','Ambiguity','#BAC8D3',''),('excess','Excess','#566775','xx')]:
        v=np.array([mean(common(c),key,100) for c in CONDITIONS]);ax.bar(x,v,bottom=bottom,label=label,color=color,hatch=hatch,edgecolor='#253B50',width=.65);bottom+=v
    ax.set(xticks=x,xticklabels=[LABELS[c] for c in CONDITIONS],ylabel='Disagreement at 80% (%)',ylim=(0,max(bottom)*1.28));ax.tick_params(axis='x',labelrotation=20)
    for j,c in enumerate(CONDITIONS):
        ys=[mean(rows('stratum',condition=c,state='selected',stratum=st,scope='within80'),'risk',100) for st in ('full','partial')]
        bx.plot([0,1],ys,ls=STYLES[c],marker=['o','s','^','D'][j],color=COLORS[c],label=LABELS[c],lw=1.3)
    bx.set(xticks=[0,1],xticklabels=['All votes\nsupported','Partial supported\nvote mass'],ylabel='Within-stratum 80% risk (%)',xlim=(-.2,1.2));bx.legend(frameon=False,loc='upper left')
    handles,labels=ax.get_legend_handles_labels();fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.5,1.1),ncol=3,frameon=False)
    for panel,axis in zip('ab',(ax,bx)):axis.text(-.17,1.02,panel,transform=axis.transAxes,weight='bold')
    save(fig,'Fig3')
    # Figure 4: accepted test-calibration component differences at each target.
    fig,axes=plt.subplots(1,3,figsize=(6.2,2.65),sharey=True,layout='constrained')
    for ax,target,panel in zip(axes,(.1,.2,.3),'abc'):
        for j,(key,label,color,marker) in enumerate([('unsupported','Unsupported','#235B83','o'),('supported_ambiguity','Ambiguity','#447651','s'),('excess','Excess','#A34227','^')]):
            gaps=[]
            for c in CONDITIONS:
                tr=rows('policy',condition=c,state='selected',target=target,policy='empirical',population='test')
                ca=rows('policy',condition=c,state='selected',target=target,policy='empirical',population='calibration')
                pairs_by_seed=[100*(float(t[key])-float(a[key])) for t,a in zip(tr,ca) if t[key]!='' and a[key]!='']
                gaps.append(np.mean(pairs_by_seed) if pairs_by_seed else np.nan)
            ax.plot(np.arange(4)+(j-1)*.1,gaps,marker=marker,ls='none',color=color,label=label,ms=4)
        ax.axhline(0,color='#777777',lw=.8);ax.set(xticks=range(4),xticklabels=['H','T','U','S'],title=f'{100*target:.0f}% target',xlabel='Supervision');ax.text(-.16,1.03,panel,transform=ax.transAxes,weight='bold')
    axes[0].set_ylabel('Test minus calibration (pp)');handles,labels=axes[0].get_legend_handles_labels();fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.5,1.14),ncol=3,frameon=False);save(fig,'Fig4')
    # Per-seed detail is intentionally visible, not hidden behind interval averages.
    fig,axes=plt.subplots(1,2,figsize=(6.2,2.7),layout='constrained')
    for ax,state,panel in zip(axes,('selected','epoch12'),'ab'):
        for j,(left,right) in enumerate(pairs):
            rr=rows('seed_contrast',left=left,right=right,state=state)
            ax.scatter([float(r['difference_pp']) for r in rr],[j]*3,marker=['o','s','^','D'][j],s=25)
        ax.axvline(0,color='#777777',ls=':',lw=.8);ax.set(yticks=range(4),yticklabels=['S - U','S - H','T - H','S - T'],title=state,xlabel='Paired difference (pp)',ylim=(-.5,3.5));ax.invert_yaxis()
        ax.text(-.16,1.03,panel,transform=ax.transAxes,weight='bold')
    save(fig,'FigS3')
    seed_values=rows('seed_contrast',left='soft',right='uniform',state='selected')
    seed_text=', '.join(f'{float(r["difference_pp"]):+.2f}' for r in seed_values)
    results=r'''\subsection{Minority allocation, softening, and tied maxima}
Table~\ref{tab:primary} compares all four target representations at the same retained count. Figure~\ref{fig:contrasts} shows risk--coverage curves and selected/fixed-epoch contrasts.
\input{evidence_revision/generated/primary_table.tex}
'''+f'The primary Soft-minus-Uniform difference is {ctext(primary)}. Paired differences for seeds 17, 42, and 89 are {seed_text} pp. {conclusion} The retained original Soft-minus-Hard contrast is {ctext(old)}. Tie-hard minus Hard is {ctext(contrast("tie_hard","hard"))}; Soft minus Tie-hard is {ctext(contrast("soft","tie_hard"))}. These latter intervals are descriptive sensitivities, not additional primary hypothesis tests.\n\n'+r'''\begin{figure}[t]
\centering\includegraphics[width=\linewidth]{evidence_revision/figures/Fig2.pdf}
\caption{(a) Mean selected-checkpoint risk--coverage curves across three seeds; the displayed range begins at 10\%, and every prefix is released. (b) Mean paired contrasts and conditional 95\% pixel-cluster intervals, for selected and epoch-12 weights. Negative differences favor the first-named target. Intervals do not measure seed-population or external-domain uncertainty}\label{fig:contrasts}
\end{figure}
\subsection{Selection sensitivity and label references}
\input{evidence_revision/generated/sensitivity_table.tex}
'''+f'At fixed epoch 12, Soft minus Uniform is {ctext(fixed)}, and Soft minus Hard is {ctext(contrast("soft","hard","epoch12"))}. Table~\\ref{{tab:sensitivity}} keeps both evaluation states visible. This test changes the stopping decision, not the budget or training objective. The majority-only F1 column concerns a restricted reference task, not an interchangeable performance improvement over original-label F1.\n\n'+r'''\subsection{Supported mass and the meaning of confidence}
Figure~\ref{fig:mass} separates unsupported mass from ambiguity and excess on each accepted set. Its second panel changes the selection population explicitly rather than silently dropping unsupported votes.
\begin{figure}[t]
\centering\includegraphics[width=\linewidth]{evidence_revision/figures/Fig3.pdf}
\caption{(a) Three-way mean risk accounting on global 80\% accepted sets. (b) Independent 80\% selection within full- and partial-supported-mass strata; these populations and vertical scales differ from panel (a). Zero-supported-mass images necessarily have complete-vote risk one and are reported separately in Online Resource 1. Neither panel evaluates an expanded-vocabulary model}\label{fig:mass}
\end{figure}
'''
    strata={st:rows('stratum',condition='soft',state='selected',stratum=st,scope='all')[0] for st in ('full','partial','zero')}
    results+=f'The primary population contains {int(strata["full"]["population_n"]):,} fully supported, {int(strata["partial"]["population_n"]):,} partially supported, and {int(strata["zero"]["population_n"]):,} zero-supported-mass images. '
    for st,label in [('full','fully supported'),('partial','partially supported')]:
        risksoft=mean(rows('stratum',condition='soft',state='selected',stratum=st,scope='within80'),'risk',100)
        riskuniform=mean(rows('stratum',condition='uniform',state='selected',stratum=st,scope='within80'),'risk',100)
        results+=f'Within the {label} stratum, 80\\% risk is {risksoft:.2f}\\% for Soft and {riskuniform:.2f}\\% for Uniform. '
    results+='These descriptive comparisons address whether the primary contrast is confined to images with unsupported votes; they do not establish why a model assigned a particular score.\n\n'
    results+=r'''\subsection{Calibration transfer and finite-sample diagnostics}
\input{evidence_revision/generated/policy_table.tex}
Table~\ref{tab:policies} includes every target and empty-policy outcome. Figure~\ref{fig:transfer} partitions the empirical policy's accepted-population risk change.
\begin{figure}[t]
\centering\includegraphics[width=\linewidth]{evidence_revision/figures/Fig4.pdf}
\caption{Mean test-minus-calibration differences in unsupported mass, supported ambiguity, and excess error for empirical thresholds. Their sum is the risk gap, not a causal attribution. H: Hard; T: Tie-hard; U: Uniform; S: Soft. The three target panels share a vertical scale. Means use paired nonempty calibration/test selections}\label{fig:transfer}
\end{figure}
'''
    empirical=rows('policy',state='selected',policy='empirical',population='test');nonempty=[r for r in empirical if r['risk']!=''];over=sum(float(r['risk'])>float(r['target'])+1e-12 for r in nonempty)
    results+=f'Across four conditions, {over} of {len(nonempty)} nonempty empirical test operating points exceed their nominal targets. These overlapping outcomes are not independent Bernoulli trials. '
    for c in ('hard','soft'):
        rr=rows('policy',condition=c,state='selected',target=.1,policy='empirical',population='test')
        results+=f'At the 10\\% target, {LABELS[c]} has {mean(rr,"coverage",100):.2f}\\% mean coverage and {mean(rr,"risk",100):.2f}\\% mean achieved risk. '
    results+='Different achieved risks prevent interpreting this as greater coverage at equal risk.\n\n'
    calpop=rows('population',condition='hard',state='selected',population='calibration')
    testpop=rows('population',condition='hard',state='selected',population='test')
    results+=f'Before confidence selection, the annotation floor is {mean(calpop,"floor",100):.2f}\\% in calibration and {mean(testpop,"floor",100):.2f}\\% in testing: a {mean(testpop,"floor",100)-mean(calpop,"floor",100):+.2f} pp difference. Unsupported mass changes from {mean(calpop,"unsupported",100):.2f}\\% to {mean(testpop,"unsupported",100):.2f}\\%; the remaining floor difference is supported-category ambiguity. Thus, these populations differ in annotation composition even before a model supplies a ranking.\n\n'
    rrtest=rows('policy',condition='soft',state='selected',target=.2,policy='empirical',population='test')
    rrcal=rows('policy',condition='soft',state='selected',target=.2,policy='empirical',population='calibration')
    gaps={k:mean(rrtest,k,100)-mean(rrcal,k,100) for k in ('risk','unsupported','supported_ambiguity','excess')}
    results+=f'For Soft at the nominal 20\\% target, the accepted test-minus-calibration risk gap is {gaps["risk"]:+.2f} pp, comprising {gaps["unsupported"]:+.2f} pp unsupported mass, {gaps["supported_ambiguity"]:+.2f} pp supported ambiguity, and {gaps["excess"]:+.2f} pp excess. The figure and ledger supply the analogous accounting for every condition and target; this arithmetic does not identify independent causal contributions.\n\n'
    for c in CONDITIONS:
        rr=rows('calibration_resample',condition=c,state='selected',target=.2)
        valid=[r for r in rr if r['optimism_gap']!=''];gaps=np.array([100*float(r['optimism_gap']) for r in valid])
        if len(gaps):results+=f'For {LABELS[c]}, the 20\\% calibration-half-split holdout-minus-fit risk gap averages {gaps.mean():+.2f} pp over {len(gaps)} nonempty seed--split outcomes (descriptive 5th--95th percentiles {np.quantile(gaps,.05):+.2f} to {np.quantile(gaps,.95):+.2f} pp). '
    results+='The complete target-specific resamples are released. These diagnostics reveal within-calibration variability and accepted-population differences, but cannot uniquely separate distribution shift, finite-sample fluctuation, and threshold-search effects.\n'
    write('results.tex',results)
    disc=conclusion+' This is a test of whether minority allocation supplies added selective value beyond dominant-confidence-preserving softening, not a claim that soft-label learning itself is new. The tie-aware condition directly probes a second alternative explanation rather than adding model capacity. The selected and final-epoch results delimit the dependence on the stopping rule; neither warrants universal superiority.\n\n'
    disc+='The vocabulary diagnostics explain why a high supported-class confidence cannot by definition encode all complete-vote risk. Unsupported mass and supported ambiguity are separately observable in the held-out annotations, while MSP omits the former even at the ideal supported-distribution optimum. Nevertheless, observed risk gaps also contain excess category-choice error. The decomposition is useful precisely because it prevents attributing a total change to a single mechanism without evidence.\n\n'
    disc+='The conservative grid comparator adds a specified finite-sample margin, at a potential coverage cost. Its assumptions have not been established for deployment or person-level independence in this benchmark, so empirical compliance is not proof of a general guarantee. Calibration half-splits are similarly diagnostic rather than a new validation cohort. The practical recommendation is to report achieved risk, coverage, annotation vocabulary, and accepted-set composition together, instead of treating a nominal confidence policy or majority-only F1 as sufficient reliability evidence.\n\n'
    uniform_gain=mean(common('hard'),'risk',100)-mean(common('uniform'),'risk',100)
    soft_gain=mean(common('hard'),'risk',100)-mean(common('soft'),'risk',100)
    interpretation=f'The size of the control contrast matters: Uniform lowers selected-checkpoint disagreement relative to Hard by {uniform_gain:.2f} pp, compared with {soft_gain:.2f} pp for Soft. Thus, the simpler target reproduces much of the observed hard-to-soft difference; detailed minority allocation adds a modest residual advantage under this protocol. This is an endpoint comparison, not a percentage of improvement causally explained by softening.\\n\\n'
    disc=disc.split('\n\n',1)[0]+'\n\n'+interpretation.replace('\\n','\n')+disc.split('\n\n',1)[1]
    bounded=rows('policy',state='selected',policy='hoeffding_grid',population='test')
    bounded_nonempty=[r for r in bounded if r['risk']!='']
    bounded_over=sum(float(r['risk'])>float(r['target'])+1e-12 for r in bounded_nonempty)
    results+=f'The fixed-grid comparator yields {len(bounded_nonempty)} nonempty test operating points, of which {bounded_over} exceed their nominal targets. It abstains completely at the 10\\% target for every condition and seed. Its observed conservatism therefore has a concrete coverage cost; these outcomes do not verify its population assumptions.\n'
    write('results.tex',results)
    write('discussion.tex',disc)
    write('conclusion.tex',conclusion+f' The primary Soft-minus-Uniform contrast at 80\\% coverage was {delta:+.2f} pp; its fixed-epoch counterpart was {float(fixed["point_difference_pp"]):+.2f} pp. The four-condition controls, three-part disagreement accounting, and calibration diagnostics make the limited empirical result interpretable without claiming a new architecture or an operational risk guarantee.\n\n')
    # Reviewer-facing supplementary summaries from the same ledger.
    report=['# Supplementary diagnostic summary','', 'All numbers below derive from reported_results.csv. Strata do not constitute independent image datasets.','',
        '## Per-seed contrasts','', '| State | Contrast | Seed | Difference (pp) |','|---|---|---:|---:|']
    for r in rows('seed_contrast'):report.append(f'| {r["state"]} | {r["left"]} - {r["right"]} | {r["seed"]} | {float(r["difference_pp"]):+.6f} |')
    report+=['','## Population composition','', '| Population | Unsupported mass (%) | Supported ambiguity (%) | Floor (%) |','|---|---:|---:|---:|']
    for population in ('calibration','test'):
        rr=rows('population',condition='hard',state='selected',population=population)
        report.append('| '+population+' | '+' | '.join(f'{mean(rr,k,100):.6f}' for k in ('unsupported','supported_ambiguity','floor'))+' |')
    report+=['','## Supported-mass strata','', '| Condition | Stratum | Selection | N | Accepted mean | Risk (%) | Supported-only risk (%) |','|---|---|---|---:|---:|---:|---:|']
    for c in CONDITIONS:
        for st in ('full','partial','zero'):
            for scope in ('all','global80','within80'):
                rr=rows('stratum',condition=c,state='selected',stratum=st,scope=scope)
                def val(key,scale=1):
                    x=mean(rr,key,scale);return '--' if x is None else f'{x:.6f}'
                report.append(f'| {LABELS[c]} | {st} | {scope} | {rr[0]["population_n"]} | {val("accepted_n")} | {val("risk",100)} | {val("supported_only_risk",100)} |')
    (OUT/'SUPPLEMENTARY_DIAGNOSTICS.md').write_text('\n'.join(report)+'\n',encoding='utf-8')
    supplement=r'''\documentclass[10pt]{article}
\usepackage[a4paper,margin=23mm]{geometry}
\usepackage{graphicx,booktabs,longtable,amsmath,url,microtype}
\renewcommand{\thetable}{S\arabic{table}}
\renewcommand{\thefigure}{S\arabic{figure}}
\begin{document}
\begin{center}
{\Large Supplementary diagnostics and implementation figures}\par\medskip
Crowd Supervision and Vocabulary-Mismatched Selective Facial-Expression Prediction\par\medskip
SN Computer Science\par\smallskip
Gunjan Kumar Mishra, Bijaya Ghimire, Badri Raj Lamichhane, Alan Shah\par\smallskip
Corresponding author: Badri Raj Lamichhane\par
School of Information, Computer and Communication Technology,\par
Sirindhorn International Institute of Technology, Thammasat University, Thailand\par
\texttt{d6622300231@g.siit.tu.ac.th}
\end{center}
\section*{Interpretation and evidence boundaries}
All numerical summaries are generated from \texttt{reported\_results.csv}.
S denotes Soft, U Uniform, H Hard, and T Tie-hard. Differences are percentage points.
Three seeds are fixed, not samples supporting broad seed-population inference.
The image source is reused; no external-domain or communication benefit is established.
Global 80\% refers to restricting a global accepted set to a stratum, whereas within-stratum 80\% reranks that stratum independently.
Supported-only risk conditions the vote reference on supported categories and is undefined at zero supported mass.
\begin{longtable}{llrrr}
\caption{Every paired seed contrast at 80\% coverage}\label{tab:seeds}\\
\toprule State & Contrast & Seed 17 & Seed 42 & Seed 89\\\midrule\endfirsthead
\toprule State & Contrast & Seed 17 & Seed 42 & Seed 89\\\midrule\endhead
'''
    for state in ('selected','epoch12'):
        for left,right in pairs:
            rr=rows('seed_contrast',left=left,right=right,state=state)
            supplement+=state+' & '+LABELS[left]+' - '+LABELS[right]+' & '+' & '.join(f'{float(r["difference_pp"]):+.4f}' for r in rr)+r'\\'+'\n'
    supplement+=r'''\bottomrule\end{longtable}
\begin{table}[ht]
\caption{Selected epochs from the common squared-distance criterion. Every fixed-epoch sensitivity checkpoint is epoch 12}
\centering\begin{tabular}{lrrr}\toprule
Targets & Seed 17 & Seed 42 & Seed 89\\\midrule
'''
    for c in CONDITIONS:
        rr=rows('checkpoint',condition=c,state='selected')
        supplement+=LABELS[c]+' & '+' & '.join(r['epoch'] for r in rr)+r'\\'+'\n'
    supplement+=r'''\bottomrule\end{tabular}\end{table}
\begin{figure}[ht]
\centering\includegraphics[width=.94\linewidth]{evidence_revision/figures/FigS1.pdf}
\caption{Fixed model head used in all four conditions: one channel gate after pooled Swin-T features, followed by a seven-category linear classifier. This is not a new architecture claim}
\end{figure}
\clearpage
\begin{figure}[ht]
\centering\includegraphics[width=.88\linewidth]{evidence_revision/figures/FigS2.pdf}
\caption{Synthetic illustration of the local sender-controlled interface. Text and category presentation are illustrative, not participant observations or a measured prediction. No recipient transport is implemented}
\end{figure}
\begin{figure}[ht]
\centering\includegraphics[width=.94\linewidth]{evidence_revision/figures/FigS3.pdf}
\caption{Individual paired seed differences at the common-coverage endpoint: (a) selected and (b) fixed epoch-12 weights. Exact values appear in Table~\ref{tab:seeds}; points can overlap}
\end{figure}
\clearpage
\begin{longtable}{lllrrrr}
\caption{Supported-mass strata: means over three selected-checkpoint seeds. Risk columns are percentages; accepted count can vary by seed}\\
\toprule Targets & Stratum & Selection & N & Accepted & Risk & Supported-only\\\midrule\endfirsthead
\toprule Targets & Stratum & Selection & N & Accepted & Risk & Supported-only\\\midrule\endhead
'''
    for c in CONDITIONS:
        for st in ('full','partial','zero'):
            for scope in ('all','global80','within80'):
                rr=rows('stratum',condition=c,state='selected',stratum=st,scope=scope)
                values=[]
                for key,scale in [('accepted_n',1),('risk',100),('supported_only_risk',100)]:
                    v=mean(rr,key,scale);values.append('--' if v is None else f'{v:.2f}')
                supplement+=LABELS[c]+' & '+st+' & '+scope+f' & {rr[0]["population_n"]} & '+' & '.join(values)+r'\\'+'\n'
        supplement+=r'\addlinespace'+'\n'
    supplement+=r'''\bottomrule\end{longtable}
Full means all votes are supported; partial means positive supported and unsupported mass; zero means no supported votes. A risk of one in the zero stratum follows from the vocabulary, not a special model failure. Global selection may leave some strata empty; risk then remains undefined.
\clearpage
\begin{longtable}{llrrrr}
\caption{Calibration half-split holdout-minus-fit risk gaps. Means and 5th/95th percentiles are descriptive percentage points over nonempty seed--split outcomes, not confidence intervals}\\
\toprule Targets & Target & Nonempty & Mean gap & 5th pct. & 95th pct.\\\midrule\endfirsthead
\toprule Targets & Target & Nonempty & Mean gap & 5th pct. & 95th pct.\\\midrule\endhead
'''
    for c in CONDITIONS:
        for target in (.1,.2,.3):
            rr=rows('calibration_resample',condition=c,state='selected',target=target)
            gap=np.array([100*float(r['optimism_gap']) for r in rr if r['optimism_gap']!=''])
            data=[f'{gap.mean():+.3f}',f'{np.quantile(gap,.05):+.3f}',f'{np.quantile(gap,.95):+.3f}'] if len(gap) else ['--']*3
            supplement+=LABELS[c]+f' & {100*target:.0f}\\% & {len(gap)}/300 & '+' & '.join(data)+r'\\'+'\n'
    supplement+=r'''\bottomrule\end{longtable}
Each of 100 shared pixel-cluster splits uses disjoint calibration halves for fitting and checking the empirical threshold. Splits overlap one another. Minimum fitted accepted count is 100 and score ties are never split. The full ledger includes fit/holdout counts, risks, thresholds, and holdout coverage for every split, including empty policies.
\section*{Unconditional population composition}
'''
    for pop in ('calibration','test'):
        rr=rows('population',condition='hard',state='selected',population=pop)
        supplement+=f'{pop.title()}: unsupported mass {mean(rr,"unsupported",100):.4f}\\%, supported ambiguity {mean(rr,"supported_ambiguity",100):.4f}\\%, annotation floor {mean(rr,"floor",100):.4f}\\%.\\par\n'
    supplement+=r'''These annotation-only quantities do not depend on the fitted model. Accepted-set differences in the main figure condition on each model's chosen threshold and can differ from these unconditional differences. None uniquely identifies a causal source of overshoot.
\section*{Reproduction map}
\texttt{REPRODUCIBILITY.md} provides CPU numerical-audit and optional image-dependent inference commands. Online Resource 1 includes predictions, source hashes, category maps, original-row joins, both protocols and tests. Online Resources 2--5 contain all selected and epoch-12 inference weights. No face images are redistributed. Internal automated verification is not external replication.
\end{document}
'''
    write('supplement.tex',supplement)
    sources={name:sha(OUT/name) for name in ('reported_results.csv','split_manifest.json','retained_assets.py','manuscript.tex')}
    (OUT/'asset_sources.json').write_text(json.dumps(sources,indent=2),encoding='utf-8')
    generated=list(GEN.glob('*.tex'))+list(FIG.glob('*.pdf'))+list(FIG.glob('*.eps'))+[OUT/'references.bib']
    (OUT/'generated_sha256.json').write_text(json.dumps({p.relative_to(OUT).as_posix():sha(p) for p in generated},indent=2),encoding='utf-8')
    print(json.dumps({'primary':dict(delta=delta,lower=lo,upper=hi),'figures':4,'conclusion':conclusion},indent=2))

if __name__=='__main__':main()
