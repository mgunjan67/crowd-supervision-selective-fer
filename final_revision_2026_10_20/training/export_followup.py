"""Produce manuscript artwork/tables from completed, audited follow-up results.

Outcome interpretation is deliberately not generated here. The final narrative
must be written after reviewing every reported condition, including adverse ones.
"""
import csv
import hashlib
import json
from pathlib import Path
import shutil

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

HERE=Path(__file__).resolve().parent
SOURCE=HERE.parent/'source'
RESULTS=HERE/'results'
SEEDS=(17,42,89)
TARGET={'soft':'Soft','uniform':'Uniform','tie_uniform':'Tie-Uniform','permutation':'Permutation'}
STATE={'selected':'Distance-selected','epoch12':'Epoch 12','risk_selected':'Risk-selected'}

plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10.5,
    'axes.labelsize':10.5,'xtick.labelsize':9.5,'ytick.labelsize':9.5,
    'pdf.fonttype':42,'ps.fonttype':42,'axes.spines.top':False,
    'axes.spines.right':False,'savefig.facecolor':'white'})

def read(name):
    with (RESULTS/name).open(newline='',encoding='utf-8') as stream:
        return list(csv.DictReader(stream))

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def table(caption,columns,header,rows,key):
    return '\n'.join([
        r'\begingroup\small\setlength{\tabcolsep}{4pt}\setlength{\LTcapwidth}{\linewidth}',
        r'\begin{longtable}{'+columns+'}',
        r'\caption{'+caption+r'}\label{'+key+r'}\\',
        r'\toprule '+header+r'\\\midrule\endfirsthead',
        r'\toprule '+header+r'\\\midrule\endhead',
        *(' & '.join(row)+r'\\' for row in rows),
        r'\bottomrule\end{longtable}\endgroup',''])

def mean(records,key):return float(np.mean([float(r[key]) for r in records]))

def format_mean(records,key,places=2):
    if not records or any(r[key] in ('',None) for r in records):return '--'
    return f'{mean(records,key):.{places}f}'

def validate_contrasts(contrasts):
    expected={(7,state,'permutation') for state in ('selected','epoch12')}
    expected.update((8,state,control) for state in ('selected','epoch12','risk_selected')
                    for control in ('uniform','tie_uniform'))
    keys=[(r['k'],r['state'],r['right']) for r in contrasts]
    assert len(keys)==8 and len(set(keys))==8 and set(keys)==expected
    for row in contrasts:
        assert row['left']=='soft' and len(row['seed_differences_pp'])==3
        values=[row[key] for key in ('mean_pp','lower_pp','upper_pp','seed_sd_pp')]
        assert np.isfinite(values).all() and np.isfinite(row['seed_differences_pp']).all()
        assert row['lower_pp']<=row['upper_pp'] and row['seed_sd_pp']>=0
        assert abs(row['mean_pp']-np.mean(row['seed_differences_pp']))<1e-10

def validate_state_rows(rows):
    expected={(7,'permutation',seed,state) for seed in SEEDS for state in ('selected','epoch12')}
    expected.update((8,condition,seed,state) for condition in ('soft','uniform','tie_uniform')
                    for seed in SEEDS for state in ('selected','epoch12','risk_selected'))
    historical={(7,condition,seed,state) for condition in ('soft','uniform','tie_uniform')
                for seed in SEEDS for state in ('selected','epoch12')}
    keys=[(int(r['k']),r['condition'],int(r['seed']),r['state']) for r in rows]
    assert len(keys)==51 and len(set(keys))==51 and set(keys)==expected|historical
    assert all(int(row['n'])==3275 and int(row['accepted_n'])==2620 for row in rows)
    return [row for row,key in zip(rows,keys) if key in expected]

def figure(contrasts):
    fig,axes=plt.subplots(2,1,figsize=(6.4,5.3),gridspec_kw={'height_ratios':[1,2.55]},sharex=True)
    maximum=max(abs(r[k]) for r in contrasts for k in ('lower_pp','upper_pp'))
    maximum=max(maximum,max(abs(x) for r in contrasts for x in r['seed_differences_pp']))
    span=max(.25,1.15*maximum)
    for ax,k,title in zip(axes,(7,8),('(a) Seven outputs: Soft minus Permutation','(b) Eight outputs: Soft minus softening controls')):
        records=[r for r in contrasts if r['k']==k]
        for i,r in enumerate(records):
            y=len(records)-i-1
            lo,hi=r['lower_pp'],r['upper_pp']
            ax.plot([lo,hi],[y,y],color='#214f70',linewidth=1.6,zorder=2)
            ax.plot([lo,lo],[y-.06,y+.06],color='#214f70',linewidth=1)
            ax.plot([hi,hi],[y-.06,y+.06],color='#214f70',linewidth=1)
            ax.scatter(r['seed_differences_pp'],y+np.array([-.10,0,.10]),
                       s=21,facecolors='white',edgecolors='#777777',linewidths=.9,zorder=3)
            ax.scatter([r['mean_pp']],[y],s=29,color='#214f70',marker='D',zorder=4)
        labels=[STATE[r['state']] if k==7 else
                ('Uniform' if r['right']=='uniform' else 'Tie-Uniform')+' / '+STATE[r['state']]
                for r in records]
        ax.set_yticks(range(len(records)-1,-1,-1),labels)
        ax.set_ylim(-.5,len(records)-.5)
        ax.axvline(0,color='#333333',linestyle='--',linewidth=.8,zorder=1)
        ax.set_title(title,loc='left',fontsize=10.5,pad=9)
        ax.grid(axis='x',color='#dddddd',linewidth=.5)
        ax.set_xlim(-span,span)
    axes[1].set_xlabel('Soft-minus-control disagreement at 80% coverage (pp)')
    from matplotlib.lines import Line2D
    handles=[Line2D([0],[0],marker='o',markerfacecolor='white',color='none',markeredgecolor='#777777',label='Paired seed'),
             Line2D([0],[0],marker='D',color='#214f70',linewidth=1.6,label='Mean; conditional 95% interval')]
    fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.5,.005),ncol=2,frameon=False,fontsize=9)
    fig.subplots_adjust(left=.36,right=.98,top=.94,bottom=.16,hspace=.65)
    for ext in ('pdf','svg','png'):
        fig.savefig(RESULTS/f'FollowupContrasts.{ext}',dpi=220)
    plt.close(fig)
    for ext in ('pdf','svg'):
        shutil.copy2(RESULTS/f'FollowupContrasts.{ext}',SOURCE/f'figures/FollowupContrasts.{ext}')

def main():
    report=json.loads((RESULTS/'summary.json').read_text())
    assert report['training_runs']==12 and report['states_new']==33
    assert report['primary_test_images']==3275 and report['pixel_clusters']==3268
    contrasts=report['contrasts']
    validate_contrasts(contrasts)
    for folder in (SOURCE/'generated',SOURCE/'figures'):folder.mkdir(parents=True,exist_ok=True)
    figure(contrasts)
    rows=read('all_states.csv')
    new=validate_state_rows(rows)
    text=[r'''\clearpage\section{New target, vocabulary and stopping-rule comparisons}
\label{supp:new-followup}
The final protocol defines twelve additional Swin-T runs on the same previously exposed source. Seven-output Permutation uses the three existing seeds; eight-output Soft, Uniform and Tie-Uniform use the same three seeds and budget. All twelve runs were completed and hashed before new calibration/test inference. There are 33 nominal inference states: six for Permutation and 27 for the eight-output conditions. Different policies can select the same epoch and therefore do not necessarily represent distinct weights. These comparisons remain adaptive within-source evidence.

All risks retain ten annotation categories. Each model selects its own 2,620 primary-test images at 80\% MSP coverage. Squared distance uses all 3,275 images. Intervals below use 5,000 shared exact-pixel-cluster replicates, rerank each resampled multiset, and average the three seed contrasts within each replicate. They condition on fitted models and observed annotations, rather than estimating variation over future seeds or people.
''']
    t=[]
    for r in new:
        t.append([r['k'],TARGET[r['condition']],STATE[r['state']],r['seed'],
                  f"{float(r['risk_pct']):.3f}",f"{float(r['distance']):.5f}"])
    text.append(table('Every new fitted state: complete-vote disagreement at 80\\% coverage (\\%) and full-coverage ten-category squared distance.',
                      'rlllrr','Outputs & Target & State & Seed & Risk & Distance',t,'tab:new-all-states'))
    t=[]
    for r in contrasts:
        compact_state={'selected':'Distance','epoch12':'Epoch 12','risk_selected':'Risk'}[r['state']]
        t.append([str(r['k']),TARGET[r['right']],compact_state,
                  *(f'{x:+.3f}' for x in r['seed_differences_pp']),f"{r['mean_pp']:+.3f}",
                  f"{r['seed_sd_pp']:.3f}",f"[{r['lower_pp']:+.3f}, {r['upper_pp']:+.3f}]"])
    text.append(r'\Needspace{15\baselineskip}')
    text.append(table('Soft-minus-control seed differences, means, sample standard deviations and conditional 95\\% intervals (all percentage points). Negative favors Soft. Distance and Risk denote the selection objectives.',
                      'rllrrrrrl','K & Control & State & 17 & 42 & 89 & Mean & SD & Interval',t,'tab:new-all-contrasts'))
    text.append(r'''\subsection{Fixed-image cross-evaluation}
The following averages apply each predictor to both selectors' accepted images. They distinguish prediction changes on fixed images from the changes in image composition inherent in own-ranking comparisons. They are descriptive cross-evaluations, not causal decomposition.
''')
    fixed=read('fixed_sets.csv');t=[]
    for r in contrasts:
        base=[x for x in fixed if int(x['k'])==r['k'] and x['state']==r['state'] and x['control']==r['right']]
        values=[]
        for selector,predictor in [('soft','soft'),('soft',r['right']),(r['right'],'soft'),(r['right'],r['right'])]:
            subset=[x for x in base if x['selector']==selector and x['predictor']==predictor]
            assert len(subset)==3
            assert {int(x['seed']) for x in subset}==set(SEEDS)
            assert all(int(x['accepted_n'])==2620 for x in subset)
            values.append(f"{mean(subset,'risk_pct'):.3f}")
        t.append([str(r['k']),TARGET[r['right']],STATE[r['state']],*values])
    text.append(table('Fixed-set risks (\\%). S is Soft and C is the named control; predictor appears before selector in each heading. All cells retain 2,620 identical images within a selector and seed.',
                      'rllrrrr','K & Control & State & S on S & C on S & S on C & C on C',t,'tab:new-fixed-sets'))
    text.append(r'''\subsection{Selection objectives and supported-mass strata}
Distance selection minimizes ten-category squared distance; the risk rule minimizes complete-vote disagreement at 80\% MSP coverage on the selection subset. Both retain the earliest minimizing epoch. The risk rule is reported only for the eight-output family and is not retrospectively reconstructed for the original runs.
''')
    t=[]
    for k,condition in [(7,'permutation'),(8,'soft'),(8,'uniform'),(8,'tie_uniform')]:
        for seed in SEEDS:
            folder=HERE/'runs'/f'k{k}_{condition}_seed{seed}'
            logs=[json.loads((folder/f'epoch-{epoch:02d}.json').read_text()) for epoch in range(1,13)]
            selected=min(logs,key=lambda x:x['distance'])['epoch']
            risk=min(logs,key=lambda x:x['risk80'])['epoch'] if k==8 else '--'
            t.append([str(k),TARGET[condition],str(seed),str(selected),str(risk),'12'])
    text.append(table('Epochs selected under the declared policies. A dash indicates a policy not reported for that family.',
                      'rllrrr','Outputs & Target & Seed & Distance & Risk & Fixed',t,'tab:new-selected-epochs'))
    strata=read('unsupported_mass_strata.csv');t=[]
    for condition in ('soft','uniform','tie_uniform'):
        for stratum in ('full','partial','zero'):
            for scope in ('all','global80','within80'):
                subset=[r for r in strata if int(r['k'])==8 and r['condition']==condition and r['state']=='selected' and r['stratum']==stratum and r['scope']==scope]
                assert len(subset)==3
                assert {int(x['seed']) for x in subset}==set(SEEDS)
                assert len({r['stratum_n'] for r in subset})==1
                t.append([TARGET[condition],stratum,scope,subset[0]['stratum_n'],
                          format_mean(subset,'accepted_n',1),format_mean(subset,'risk_pct'),
                          format_mean(subset,'unsupported_pp'),format_mean(subset,'excess_pp')])
    text.append(r'\Needspace{10\baselineskip}')
    text.append(table('Eight-output distance-selected supported-mass strata: means over seeds. Risk is a percentage; components are percentage-point contributions. Accepted counts can vary by seed. The full CSV retains every state and scope.',
                      'lllrrrrr','Target & Stratum & Scope & N & Kept & Risk & Unsupp. & Excess',t,'tab:new-strata'))
    text.append(r'''Full and partial denote supported vote mass one and strictly between zero and one; zero denotes no supported votes. Global 80\% restricts the full-population accepted set; within-stratum 80\% reranks within the stratum. Supported ambiguity is risk minus unsupported mass minus excess before rounding. A dash in a mean denotes an undefined value for at least one seed, rather than averaging only nonempty seeds. Empty or unsupported-only conditional risks are not replaced with zero. Every 20\%, 50\%, 80\% and 100\% summary, all risk--coverage prefixes and fixed-image per-seed cells are retained in the result ledgers.
''')
    (SOURCE/'generated/new_followup_supplement.tex').write_text('\n'.join(text),encoding='utf-8')
    input_files=[RESULTS/name for name in ('summary.json','all_states.csv','fixed_sets.csv','unsupported_mass_strata.csv')]
    outputs=[SOURCE/'generated/new_followup_supplement.tex',SOURCE/'figures/FollowupContrasts.pdf',
             SOURCE/'figures/FollowupContrasts.svg']
    (RESULTS/'publication_export.json').write_text(json.dumps(dict(
        input_sha256={str(p.relative_to(HERE)):sha(p) for p in input_files},
        output_sha256={str(p.relative_to(SOURCE)):sha(p) for p in outputs},
        narrative_interpretation='Requires separate outcome-aware author/editor review'),indent=2),encoding='utf-8')
    print('Exported all 33 new states and eight declared contrasts.')

if __name__=='__main__':main()
