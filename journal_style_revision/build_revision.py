"""Presentation-only edition; frozen experiments and reported numbers remain unchanged."""
import csv, hashlib, json, re, shutil
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from fix_figure_layout import improve_supplement
from scientific_diagrams import draw_methodology, draw_model
OUT=Path(__file__).resolve().parent; ROOT=OUT.parent; BASE=ROOT/'backbone_followup'
BLUE,ORANGE,INK='#235B83','#A34227','#25313D'
CONDS=['hard','tie_hard','uniform','tie_uniform','soft']; LABELS=['Hard','Tie-hard','Uniform','Tie-Uniform','Soft']
COLORS=['#25313D','#707070',BLUE,'#386B53',ORANGE]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(fig,n):
    fig.canvas.draw()
    renderer=fig.canvas.get_renderer()
    legends=[]
    for ax in fig.axes:
        legend=ax.get_legend()
        if legend is not None:
            bounds=legend.get_window_extent(renderer)
            assert not bounds.overlaps(ax.get_window_extent(renderer)),f'Figure {n}: legend intersects plot'
            for text in ax.texts:
                assert not bounds.overlaps(text.get_window_extent(renderer)),f'Figure {n}: legend intersects panel heading'
            legends.append(bounds)
    for i,a in enumerate(legends):
        for b in legends[i+1:]:assert not a.overlaps(b),f'Figure {n}: legends intersect'
    for ext in ('pdf','eps'):fig.savefig(OUT/'figures'/f'Fig{n}.{ext}',bbox_inches='tight',pad_inches=.045)
    plt.close(fig)
def box(ax,x,y,w,h,text,color=BLUE,size=10.5):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.008,rounding_size=0.015',edgecolor=color,facecolor='white',linewidth=1,clip_on=False))
    ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=size,color=INK)
def arrow(ax,a,b):ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=10,linewidth=.9,color=INK))
def panel(ax,letter,label):
    ax.text(0,1.04,letter,transform=ax.transAxes,fontweight='bold',fontsize=11)
    ax.text(.085,1.04,label,transform=ax.transAxes,fontsize=10.5)
def main():
    assert json.loads((BASE/'audit.json').read_text())['status']=='passed'
    assert sha(BASE/'reported_results.csv')=='200a9b1df63cb267169a9f888b64a7fda3993f241494fff2e33320817bbfec3b'
    for folder in ('generated','figures'):
        (OUT/folder).mkdir(exist_ok=True)
        for p in (BASE/folder).iterdir():
            if p.is_file():shutil.copy2(p,OUT/folder/p.name)
    for name in ('reported_results.csv','references.bib'):shutil.copy2(BASE/name,OUT/name)
    for p in (OUT/'generated').glob('*.tex'):p.write_text(p.read_text().replace('backbone_followup/','journal_style_revision/'))
    for name in ('Fig1','Fig2','Fig3','Fig4'):
        for ext in ('pdf','eps'):shutil.copy2(BASE/'figures'/f'{name}.{ext}',OUT/'figures'/f'Historical{name}.{ext}')
    rows=list(csv.DictReader((OUT/'reported_results.csv').open(newline='')))
    def get(kind,**criteria):return [r for r in rows if r['record_type']==kind and all(r[k]==str(v) for k,v in criteria.items())]
    def avg(rr,key,scale=1):
        assert len(rr)==3,(key,len(rr));return float(np.mean([float(r[key])*scale for r in rr]))
    records={}
    plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],'font.size':11,'axes.labelsize':11,'xtick.labelsize':10.5,'ytick.labelsize':10.5,'axes.linewidth':.7,'pdf.fonttype':42,'ps.fonttype':42,'text.color':INK,'axes.labelcolor':INK})
    draw_methodology(OUT)
    draw_model(OUT)
    fig,axes=plt.subplots(1,2,figsize=(6.5,3.9),gridspec_kw={'width_ratios':[1.1,1]})
    vals=np.array([[avg(get('common',backbone='swin_t',condition=c,state='selected',requested_coverage=.8),k,100) for k in ('unsupported','supported_ambiguity','excess')] for c in CONDS]);records['Fig3_components_pp']=vals.tolist();bottom=np.zeros(5)
    for j,(label,col,hatch) in enumerate(zip(('Unsupported','Ambiguity','Excess'),('white','#B9CEDC','#607E95'),('///','','xx'))):
        axes[0].bar(range(5),vals[:,j],bottom=bottom,color=col,edgecolor=INK,hatch=hatch,width=.65,lw=.65,label=label);bottom+=vals[:,j]
    axes[0].set_xticks(range(5),LABELS,rotation=45,ha='right');axes[0].set_ylabel('Global 80% disagreement (%)');axes[0].set_ylim(0,31);axes[0].legend(frameon=False,fontsize=10,loc='lower left',bbox_to_anchor=(-.12,1.18),ncol=3,columnspacing=.8,handlelength=1.4);panel(axes[0],'a','Global accepted sets')
    for c,label,col,marker in zip(CONDS,LABELS,COLORS,('o','s','^','D','v')):
        if c=='tie_uniform':continue
        values=[avg(get('stratum',backbone='swin_t',condition=c,state='selected',stratum=s,scope='within80'),'risk',100) for s in ('full','partial')]
        records['Fig3_within80_'+c]=values;axes[1].plot([0,1],values,marker=marker,color=col,label=label,ls='--' if c=='tie_hard' else '-',lw=1)
    axes[1].set_xticks([0,1],['Full supported\nmass','Partial supported\nmass']);axes[1].set_ylabel('Within-stratum 80% risk (%)');axes[1].set_ylim(0,47);axes[1].set_xlim(-.25,1.25);axes[1].legend(frameon=False,fontsize=10,loc='lower left',bbox_to_anchor=(0,1.18),ncol=2,columnspacing=.8,handlelength=1.4);panel(axes[1],'b','Original four targets');fig.tight_layout(w_pad=1.1);save(fig,3)
    fig,axes=plt.subplots(1,2,figsize=(6.5,3.6));matrices=[]
    for b in ('swin_t','resnet18'):
        mat=[[avg(get('cross_evaluation',backbone=b,state='selected',predictor=p,selector=s,**({'comparison':'uniform'} if b=='resnet18' else {})),'risk',100) for s in ('uniform','soft')] for p in ('uniform','soft')];matrices.append(np.array(mat));records['Fig4_'+b+'_risk_pct']=mat
    vmin=min(m.min() for m in matrices);vmax=max(m.max() for m in matrices)
    for j,(ax,b,m) in enumerate(zip(axes,('Swin-T','ResNet-18'),matrices)):
        im=ax.imshow(m,cmap='Blues',vmin=vmin,vmax=vmax)
        for y,x in np.ndindex(2,2):ax.text(x,y,f'{m[y,x]:.3f}',ha='center',va='center',fontsize=12,color='white' if m[y,x]>(vmin+vmax)/2 else INK)
        ax.set_xticks([0,1],['Uniform set','Soft set']);ax.set_yticks([0,1],['Uniform','Soft']);ax.set_xlabel('Accepted image set');ax.set_ylabel('Predictor');panel(ax,chr(97+j),b);d=m[1]-m[0];ax.text(.5,-.56,f'Soft - Uniform (pp)\nUniform set: {d[0]:+.3f} | Soft set: {d[1]:+.3f}',transform=ax.transAxes,ha='center',va='top',fontsize=10.5)
    fig.subplots_adjust(left=.095,right=.88,bottom=.40,top=.87,wspace=.55);fig.colorbar(im,cax=fig.add_axes([.91,.41,.024,.45]),label='Disagreement (%)');save(fig,4)
    fig,ax=plt.subplots(figsize=(6.5,2.8));curves=[]
    for seed,col,style in zip((17,42,89),('#537895','#707070','#558165'),('-','--',':')):
        rr=sorted(get('risk_difference_curve',backbone='swin_t',state='selected',seed=seed),key=lambda r:int(r['accepted_n']));assert len(rr)==3275
        x=np.array([float(r['coverage'])*100 for r in rr]);y=np.array([float(r['difference_pp']) for r in rr]);keep=x>=10;curves.append(y);ax.plot(x[keep],y[keep],color=col,ls=style,lw=.8,label=f'Seed {seed}')
    ax.plot(x[keep],np.mean(curves,axis=0)[keep],color=ORANGE,lw=1.6,label='Mean');ax.axhline(0,color=INK,lw=.7);ax.axvline(80,color=INK,ls=':',lw=.9);ax.set(xlim=(10,100),xlabel='Coverage (%)',ylabel='Soft - Uniform risk (pp)');ax.legend(frameon=False,fontsize=10.5,ncol=4,loc='lower center',bbox_to_anchor=(.5,1.02));fig.tight_layout();save(fig,5)
    for ext in ('pdf','eps'):shutil.copy2(BASE/'figures'/f'Fig4.{ext}',OUT/'figures'/f'Fig6.{ext}')
    fig,axes=plt.subplots(2,1,figsize=(6.5,4.5),sharex=True)
    for ax,state,letter in zip(axes,('selected','epoch12'),('a','b')):
        items=[(b,c,get('contrast',backbone=b,state=state,left='soft',right=c)[0]) for b in ('swin_t','resnet18') for c in ('uniform','tie_uniform')]
        for i,(b,c,r) in enumerate(items):
            pt,lo,hi=[float(r[k]) for k in ('point_difference_pp','lower_pp','upper_pp')];col=BLUE if b=='swin_t' else ORANGE
            ax.hlines(i,lo,hi,color=col,lw=1.1);ax.vlines([lo,hi],i-.06,i+.06,color=col,lw=1.1);ax.plot(pt,i,'o' if b=='swin_t' else 's',color=col,ms=5)
        ax.set_yticks(range(4),[(('Swin-T' if b=='swin_t' else 'ResNet-18')+' / '+('Uniform' if c=='uniform' else 'Tie-Uniform')) for b,c,_ in items]);ax.invert_yaxis();ax.axvline(0,color=INK,lw=.8);ax.grid(axis='x',lw=.4,color='#DDDDDD');panel(ax,letter,'Selected checkpoint' if state=='selected' else 'Fixed epoch 12');ax.set_xlabel('Soft - control disagreement (pp)')
    fig.tight_layout(h_pad=1.1);save(fig,7)
    improve_supplement(rows)
    text=(BASE/'manuscript.tex').read_text().replace('backbone_followup/','journal_style_revision/')
    a=text.index('Crowd annotations make facial-expression recognition');b=text.index('\\section{Related work}')
    text=text[:a]+(OUT/'introduction.tex').read_text()+text[b:]
    text=text.replace('\\section{Related work}\n','\\section{Related work}\n\\subsection{Facial-expression supervision}\n').replace('Crowd-Calibrator connects','\\subsection{Selective prediction and annotation disagreement}\nCrowd-Calibrator connects')
    old=r'\caption{Retained Swin experimental flow. The ResNet robustness phase reuses the same isolated roles. Calibration transfer analyses retain the original four Swin conditions; Tie-Uniform is a secondary control. Checkpoint selection does not use calibration; all phases are adaptive rather than independent confirmation}'
    assert old in text;text=text.replace(old,r'\caption{Experimental dependencies. (a) Target $t_i$ and supported mass $w_i$ determine the training loss. $J_e$ is the selection-set mean ten-category squared distance after padding $p_e$ with three zeros; selection chooses $\theta_{e^*}$, while $\theta_{12}$ is retained independently. (b) Frozen models produce test probabilities and maximum-probability scores. Own-ranking compares equal counts; fixed-image comparisons hold the accepted images constant. Test votes score predictions, not acceptance decisions. (c) Only calibration-transfer analyses fit a threshold $\tau$ using calibration votes and apply it unchanged to test scores. This branch concerns the original four Swin conditions. All roles are exact-pixel isolated; all four adaptive phases reuse one image source. The local prototype is separate (Online Resource 1)}')
    old='A dimensioned head schematic and an illustrative composition interface are supplied in Online Resource 1.'
    assert old in text;text=text.replace(old,r'Figure~\ref{fig:head} specifies the Swin-T feature hierarchy and the implemented post-pooling head. The same head structure is used with ResNet-18 at its own pooled dimension; the illustrative interface remains in the supplement.'+'\n'+(OUT/'head_figure.tex').read_text())
    text=text.replace('Both sets of weights, logs, and probabilities are released.','Both sets of weights, logs, and probabilities are retained in the supplementary package.')
    text=re.sub(r'Fig\.~\\ref\{fig:transfer\}', 'the calibration-transfer results', text)
    text=text.replace(r', with versioned archives and inference checkpoints at \url{https://github.com/mgunjan67/crowd-supervision-selective-fer/releases/tag/v1.2.0}',r'. Existing Swin checkpoint archives remain at \url{https://github.com/mgunjan67/crowd-supervision-selective-fer/releases}; ResNet checkpoints are supplied in Online Resource 7')
    (OUT/'manuscript.tex').write_text(text)
    results=(OUT/'generated/results.tex').read_text().replace('journal_style_revision/figures/RiskDifference.pdf','journal_style_revision/figures/Fig5.pdf').replace('journal_style_revision/figures/BackboneContrasts.pdf','journal_style_revision/figures/Fig7.pdf')
    results=results.replace('Original-four-condition risk--coverage curves and selected/fixed-epoch contrast intervals are retained in the supplement; the direct primary difference curve appears below.','Original-four-condition risk--coverage curves remain in the supplement; the primary difference curve and two-backbone intervals appear later.')
    results=results.replace('\\begin{figure}[t]\\centering\\includegraphics[width=\\linewidth]{journal_style_revision/figures/Fig5.pdf}',r'Figure~\ref{fig:difference} shows how the Soft-minus-Uniform contrast changes with coverage; it is descriptive, not an additional confirmatory test.'+'\n'+r'\begin{figure}[t]\centering\includegraphics[width=\linewidth]{journal_style_revision/figures/Fig5.pdf}')
    results=results.replace('Original four conditions: (a) three-way mean risk accounting on global 80\\% accepted sets. (b) Independent','(a) Five Swin conditions: mean risk accounting on global 80\\% accepted sets. (b) Original four conditions: independent')
    results=results.replace(r'reported in Table~\ref{tab:retention} and Online Resource 1',r'reported in the later retention diagnostics and Online Resource 1')
    old='Table~\\ref{tab:cross} holds the accepted image set fixed within each column.\n\\input{journal_style_revision/generated/cross.tex}'
    assert old in results;results=results.replace(old,(OUT/'cross_figure.tex').read_text())
    results=results.replace('journal_style_revision/figures/Fig4.pdf}\n\\caption{Mean test-minus-calibration','journal_style_revision/figures/Fig6.pdf}\n\\caption{Mean test-minus-calibration')
    results=results.replace('The nine new ResNet runs keep the supervision contrast matched within each seed. Table~\\ref{tab:backbones} reports','The secondary ResNet checks do not confirm the selected Swin advantage: all four conditional intervals cross zero. Figure~\\ref{fig:backbones} summarizes both families; Table~\\ref{tab:backbones} reports')
    results=results.replace('Soft-minus-control risk at 80\\% own-ranking coverage in two backbone families.','Soft-minus-control disagreement at 80\\% own-ranking coverage in two backbone families: (a) selected checkpoints and (b) fixed epoch 12.')
    a=results.index('For Hard, the 20\\% calibration-half-split');b=results.index('The fixed-grid comparator yields',a)
    results=results[:a]+r'''At the 20\% target, mean calibration-half-split holdout-minus-fit gaps range from +0.05 to +0.19 pp across the original four conditions, with broad descriptive 5th--95th-percentile ranges of roughly -2.3 to +2.8 pp. Complete per-condition summaries and target-specific resamples are supplied in the supplement and ledger. They show within-calibration variability but do not uniquely separate distribution shift, finite-sample fluctuation and threshold-search effects.
''' +results[b:]
    (OUT/'generated/results.tex').write_text(results)
    shutil.copy2(OUT/'abstract.tex',OUT/'generated/abstract.tex')
    p=OUT/'generated/conclusion.tex';p.write_text(p.read_text().replace('are released','are retained in the supplementary package'))
    for p in (OUT/'generated').glob('*.tex'):
        if p.name!='supplement.tex':p.write_text('\n'.join(line.rstrip() for line in p.read_text().replace('\\bottomrule\\end{tabular}\n','\\bottomrule\\end{tabular}\\par\n').splitlines())+'\n')
    p=OUT/'generated/supplement.tex';sup=p.read_text()
    for name in ('Fig1','Fig2','Fig3','Fig4'):sup=sup.replace(f'figures/{name}.pdf',f'figures/Historical{name}.pdf')
    sup=re.sub(r'\\clearpage\s*\\begin\{figure\}\[ht\]\\centering\\includegraphics\[width=\.94\\linewidth\]\{journal_style_revision/figures/ApplicationFlow.pdf\}.*?\\end\{figure\}', '',sup,flags=re.S)
    for spec,header,caption in [('lllrrr','Targets & Population','Unsupported category vote mass by population and accepted-set scope'),('llrrr','State & Predictor','Swin-T per-seed fixed-image cross-evaluation'),('llrrrr','State & Seed','Swin-T accepted-set overlap for Soft and Uniform'),('llrrr','State & Targets','ResNet-18 reference-specific weighted F1')]:
        old='\\begin{longtable}{'+spec+'}\\toprule '+header;new='\\begin{longtable}{'+spec+'}\\caption{'+caption+'}\\\\\\toprule '+header;assert old in sup;sup=sup.replace(old,new)
    for state in ('selected','epoch12'):
        old='\\subsection*{'+state+' fixed-image comparisons}\\begin{longtable}{lllrrr}\\toprule';new='\\subsection*{'+('Selected-checkpoint' if state=='selected' else 'Fixed-epoch')+' fixed-image comparisons}\\begin{longtable}{lllrrr}\\caption{ResNet-18 '+state+' fixed-image cross-evaluation}\\\\\\toprule';assert old in sup;sup=sup.replace(old,new)
    # Keep short supplementary tables beside their source position; floating them
    # above a numbered longtable otherwise reverses S5/S6/S7 on the page.
    sup=re.sub(r'\\input\{([^}]+)\}',lambda m:(ROOT/m[1]).read_text(),sup)
    sup=sup.replace(r'\usepackage{graphicx,booktabs,longtable,amsmath,url,microtype}',r'\usepackage{graphicx,booktabs,longtable,amsmath,url,microtype,float}')
    sup=re.sub(r'\\begin\{table\}\[[^]]*\]',lambda m:r'\begin{table}[H]',sup)
    sup=sup.replace('Exact values appear in Table~\\ref{tab:seeds}; points can overlap','Exact values appear in Table~\\ref{tab:seeds}; within-row vertical offsets separate seeds without changing their difference values')
    p.write_text(sup)
    (OUT/'figure_values.json').write_text(json.dumps(records,indent=2))
    assert sha(OUT/'reported_results.csv')==sha(BASE/'reported_results.csv')
    (OUT/'presentation_audit.json').write_text(json.dumps({'status':'passed','experiment_changes':False,'new_training_runs':0,'ledger_sha256':sha(OUT/'reported_results.csv'),'copied_third_party_artwork':False,'prospective_release_url_removed':True,'presentation_only':True},indent=2))
    (OUT/'asset_sources.json').write_text(json.dumps({p.name:sha(p) for p in (OUT/'reported_results.csv',OUT/'build_revision.py',OUT/'scientific_diagrams.py',OUT/'manuscript.tex')},indent=2))
    assets=list((OUT/'generated').glob('*.tex'))+list((OUT/'figures').glob('*.pdf'))+[OUT/'references.bib']
    (OUT/'generated_sha256.json').write_text(json.dumps({p.relative_to(OUT).as_posix():sha(p) for p in assets},indent=2))
    print('Prepared journal-style presentation edition; empirical ledger unchanged')
if __name__=='__main__':main()
