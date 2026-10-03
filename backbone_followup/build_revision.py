"""Build a separate edition from the verified v1.1.0 assets and merged ledger."""
import csv,hashlib,json,re,shutil
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
OUT=Path(__file__).resolve().parent;ROOT=OUT.parent;OLD=ROOT/'evidence_revision'
LABELS={'soft':'Soft','uniform':'Uniform','tie_uniform':'Tie-Uniform'}
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    assert json.loads((OUT/'audit.json').read_text())['status']=='passed'
    for folder in ('generated','figures'):
        (OUT/folder).mkdir(exist_ok=True)
        for p in (OLD/folder).iterdir():
            if p.is_file():shutil.copy2(p,OUT/folder/p.name)
    for p in (OUT/'generated').glob('*.tex'):p.write_text(p.read_text().replace('evidence_revision/','backbone_followup/'))
    rows=list(csv.DictReader((OUT/'reported_results.csv').open(newline='')))
    def get(kind,**criteria):return [r for r in rows if r['record_type']==kind and all(r[k]==str(v) for k,v in criteria.items())]
    def avg(rr,k,scale=1):return np.mean([float(r[k])*scale for r in rr])
    def table(file,caption,label,columns,head,body,note=''):
        tex='\\begin{table}[t]\n\\caption{'+caption+'}\\label{'+label+'}\n\\centering\\begin{tabular}{'+columns+'}\\toprule\n'+head+r'\\\midrule'+'\n'
        tex+='\n'.join(' & '.join(r)+r'\\' for r in body)+'\n'+r'\bottomrule\end{tabular}\par'+'\n'+('\\smallskip\n'+note+'\\par\n' if note else '')+'\\end{table}\n'
        (OUT/'generated'/file).write_text(tex)
    comparison=[];body=[];seedbody=[];crossbody=[];retention=[]
    for backbone in ('swin_t','resnet18'):
        for state in ('selected','epoch12'):
            for right in ('uniform','tie_uniform'):
                r=get('contrast',backbone=backbone,state=state,left='soft',right=right)[0]
                comparison.append(('Swin-T' if backbone=='swin_t' else 'ResNet-18',state,right,r))
                body.append(['Swin-T' if backbone=='swin_t' else 'ResNet-18',state,LABELS[right],f'{float(r["point_difference_pp"]):+.3f}',f'[{float(r["lower_pp"]):+.3f}, {float(r["upper_pp"]):+.3f}]'])
    table('backbone_contrasts.tex',r'Within-backbone Soft-minus-control disagreement at 80\% coverage (percentage points). Intervals condition on fitted models, three seeds, observed pixels and votes','tab:backbones','lllrr','Backbone & State & Control & Difference & 95\\% interval',body,'ResNet comparisons are adaptive secondary robustness checks, not independent-image confirmation or architecture comparisons.')
    riskbody=[]
    for c in LABELS:
        riskbody.append([LABELS[c]]+[f'{avg(get("common",backbone="resnet18",condition=c,state=state,requested_coverage=.8),"risk",100):.3f}' for state in ('selected','epoch12')])
        retention.append([LABELS[c]]+[', '.join(r['accepted_n'] for r in sorted(get('retention',backbone='resnet18',condition=c,stratum=st),key=lambda r:int(r['seed']))) for st in ('full','partial','zero')])
    table('resnet_risks.tex',r'ResNet-18 complete-vote risk (\%) at 80\% coverage, means across paired seeds','tab:resnetrisk','lrr','Targets & Selected & Epoch 12',riskbody)
    table('resnet_retention.tex',r'ResNet-18 global 80\% accepted counts in full/partial/zero-supported-mass strata. Entries list seeds 17, 42, 89','tab:resnetretention','lrrr','Targets & Full & Partial & Zero',retention,'Each seed retains 2,620 of 3,275 test images. Zero-mass population size is 16.')
    plt.rcParams.update({'font.size':11,'pdf.fonttype':42,'ps.fonttype':42})
    fig,axes=plt.subplots(2,1,figsize=(6.3,5.2),sharex=True)
    for ax,state in zip(axes,('selected','epoch12')):
        items=[v for v in comparison if v[1]==state]
        for i,(b,_,c,r) in enumerate(items):
            pt,lo,hi=[float(r[k]) for k in ('point_difference_pp','lower_pp','upper_pp')]
            color='#235B83' if b=='Swin-T' else '#A34227'
            ax.hlines(i,lo,hi,color=color,lw=1.2);ax.vlines([lo,hi],i-.07,i+.07,color=color,lw=1.2);ax.plot(pt,i,'o',color=color)
        ax.axvline(0,color='0.5',lw=.8);ax.set_yticks(range(len(items)),[b+' / '+LABELS[c] for b,_,c,_ in items]);ax.invert_yaxis();ax.set_title('Selected' if state=='selected' else 'Epoch 12');ax.set_xlabel('Soft minus control (pp)');ax.grid(axis='x',alpha=.2)
    fig.tight_layout();fig.savefig(OUT/'figures/BackboneContrasts.pdf');fig.savefig(OUT/'figures/BackboneContrasts.eps');plt.close(fig)
    fig,axes=plt.subplots(3,3,figsize=(7.2,6.7),sharex=True)
    for i,c in enumerate(LABELS):
        for j,s in enumerate((17,42,89)):
            rr=sorted(get('learning_curve',backbone='resnet18',condition=c,seed=s),key=lambda r:int(r['epoch']))
            ax=axes[i,j];ax.plot(range(1,13),[float(r['selection_distance']) for r in rr],color='#235B83')
            ax.set_title(LABELS[c]+f', seed {s}');ax.set_ylabel('Selection distance');ax.set_xlabel('Epoch');ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(OUT/'figures/ResNetLearning.pdf');plt.close(fig)
    fig,axes=plt.subplots(2,1,figsize=(6.3,5.1),sharex=True)
    for ax,right in zip(axes,('uniform','tie_uniform')):
        values=[]
        for seed in (17,42,89):
            rr=sorted(get('risk_difference_curve',backbone='resnet18',comparison=right,seed=seed),key=lambda r:int(r['accepted_n']))
            x=np.array([float(r['coverage']) for r in rr]);y=np.array([float(r['difference_pp']) for r in rr]);visible=x>=.2;values.append(y);ax.plot(x[visible],y[visible],color='#A34227',alpha=.4,lw=.65)
        ax.plot(x[visible],np.mean(values,axis=0)[visible],color='#A34227',lw=1.7);ax.axhline(0,color='0.5',lw=.7);ax.axvline(.8,color='0.5',ls=':',lw=.7);ax.set_xlim(.2,1);ax.set_ylabel('Soft minus '+LABELS[right]+' (pp)');ax.set_xlabel('Coverage');ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(OUT/'figures/ResNetDifferences.pdf');plt.close(fig)
    text=(OLD/'manuscript.tex').read_text().replace('evidence_revision/','backbone_followup/')
    text=text.replace(r'{\vskip12pt}',r'{\vskip0pt}')
    text=text.replace('\\input{backbone_followup/generated/phases.tex}','Table~\\ref{tab:phases} summarizes the adaptive phases.\n\\input{backbone_followup/generated/phases.tex}')
    text=text.replace('Fifteen matched training runs','Twenty-four matched training runs across Swin-T and ResNet-18')
    text=text.replace('under matched training?','under matched training across two backbone families?')
    text=text.replace('Offline experimental flow. Calibration transfer analyses retain the original four conditions; Tie-Uniform is a targeted secondary control.','Retained Swin experimental flow. The ResNet robustness phase reuses the same isolated roles. Calibration transfer analyses retain the original four Swin conditions; Tie-Uniform is a secondary control.')
    text=text.replace('All analyses reuse the same image source and are not independent confirmation.','A fourth phase adds nine ResNet-18 runs after the preceding outcomes and final review were known. All phases reuse the same image source.')
    text=text.replace('The architecture is held fixed so that supervision is the experimental variable.','Architecture is fixed within each backbone so that supervision is the experimental variable.')
    text=text.replace('All conditions use torchvision Swin-T','The five original conditions use torchvision Swin-T')
    text=text.replace('Seeds 17, 42, and 89 produce three matched groups of five conditions','For Swin-T, seeds 17, 42, and 89 produce three matched groups of five conditions')
    text=text.replace('Vats and Chadha have also described Swin with squeeze-and-excitation for FER (unpublished preprint, 2023) \\cite{vats2023swin}.','Vats and Chadha also describe Swin with squeeze-and-excitation for FER (unpublished preprint, 2023; \\url{https://arxiv.org/abs/2301.10906}).')
    text=text.replace('A closely related unpublished preprint by Wang, Li, and Wang combines vote-count modeling, predicted disagreement, and rejection for dynamic expressions (2026) \\cite{wang2026disagreement}.','Wang, Li, and Wang combine vote-count modeling, predicted disagreement, and rejection for dynamic expressions (unpublished preprint, 2026; \\url{https://arxiv.org/abs/2609.17130}).')
    methods=r'''\subsection{Second-backbone robustness check}
The final adaptive phase trains ResNet-18 \cite{he2016resnet} with torchvision ImageNet-1K V1 initialization. The pooled 512-vector receives one bias-free $512\!\to\!32\!\to\!512$ ReLU/sigmoid gate (32,768 parameters) and a biased seven-logit classifier. Soft, Uniform and Tie-Uniform each use seeds 17, 42 and 89. Within a seed the three conditions share initialization, image order and augmentation. Data roles, transforms, losses, optimization, budget and selection criterion remain as above; BatchNorm uses standard training updates. Zero-mass training images remain in batches and can affect BatchNorm statistics despite contributing no direct supervised loss. The shared transform is deliberately retained rather than using ResNet's default ImageNet transform. Backbone dynamics and gate dimensions differ, so this is a supervision sensitivity check, not an optimized architecture comparison. All nine runs finish and are hashed before calibration/test inference. Selected and epoch-12 states, both 80\% contrasts, fixed-set cross-evaluation and vocabulary/reference diagnostics are reported irrespective of direction.

'''
    text=text.replace('\\subsection{Disagreement, annotation floor, and classification metrics}',methods+'\\subsection{Disagreement, annotation floor, and classification metrics}')
    text=text.replace('Each contrast is reported for all three paired seeds.','Each within-backbone contrast is reported for all three paired seeds.')
    text=text.replace('Only Soft-minus-Uniform is the primary follow-up contrast;','Only the retained Swin-T Soft-minus-Uniform is the primary follow-up contrast; the ResNet comparisons are adaptive secondary robustness checks;')
    text=text.replace('The six-control and later three-run Tie-Uniform protocols were each fixed before execution but after the preceding outcomes and reviews were known.','The six-control, three-run Tie-Uniform and nine-run ResNet protocols were each fixed before execution but after preceding outcomes and reviews were known.')
    text=text.replace('We do not compare a learned selector, another backbone, a second independently acquired vote-annotated image source, or an expanded-vocabulary model.','We test two backbone families, but do not compare a learned selector, an independently acquired vote-annotated image source, or an expanded-vocabulary model.')
    text=text.replace('Online Resources 2--6 supply selected and epoch-12 inference checkpoints for Hard, Soft, Uniform, Tie-hard, and Tie-Uniform, respectively.','Online Resources 2--6 supply the 30 Swin-T selected and epoch-12 checkpoints; Online Resource 7 supplies all 18 ResNet-18 checkpoints.')
    text=text.replace('Online Resources 2--6: all 30 selected and fixed-epoch inference checkpoints, grouped by supervision condition, with checksum manifests.','Online Resources 2--7: all 48 selected and fixed-epoch inference checkpoints, grouped by backbone and supervision condition, with checksum manifests.')
    text=text.replace('/releases/tag/v1.1.0','/releases/tag/v1.2.0')
    text=text.replace('all three adaptive protocols','all four adaptive protocols')
    text=text.replace('Three paired seeds and a fixed 12-epoch budget','Three paired seeds per backbone and a fixed 12-epoch budget')
    text=text.replace('The sender-controlled local prototype is an implementation artifact, not the validation setting.','The sender-controlled local prototype retains its Swin checkpoint contract; ResNet is only a research robustness check. It is an implementation artifact, not the validation setting.')
    # Give the three diagnostic contributions positively before scope qualifications.
    old='The contribution is a bounded empirical investigation of these alternative explanations, not a new architecture, loss principle, or decomposition theorem.'
    new='The study contributes a matched comparison of target representations, fixed-image tests separating predictor from accepted-set effects, and annotation-vocabulary diagnostics connecting reported risk to its reference. These address reliability interpretation rather than architecture or loss novelty.'
    assert old in text;text=text.replace(old,new)
    (OUT/'manuscript.tex').write_text(text)
    bib=(OLD/'references.bib').read_text()
    for key in ('vats2023swin','wang2026disagreement'):bib=re.sub(r'@misc\{'+key+r',[\s\S]*?\n\}', '',bib)
    bib+=r'''
@inproceedings{he2016resnet,
 author={He, Kaiming and Zhang, Xiangyu and Ren, Shaoqing and Sun, Jian},
 title={Deep Residual Learning for Image Recognition},
 booktitle={2016 IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
 year={2016}, pages={770--778}, doi={10.1109/CVPR.2016.90}
}
'''
    (OUT/'references.bib').write_text(bib)
    secondary=r'''\subsection{Robustness across two backbone families}
The nine new ResNet runs keep the supervision contrast matched within each seed. Table~\ref{tab:backbones} reports both selected and epoch-12 contrasts; absolute ResNet risks are in the supplement.
\input{backbone_followup/generated/backbone_contrasts.tex}
'''
    for state in ('selected','epoch12'):
        for right in ('uniform','tie_uniform'):
            rr=get('seed_contrast',backbone='resnet18',state=state,left='soft',right=right)
            secondary+=('Selected' if state=='selected' else 'Epoch-12')+' ResNet Soft-minus-'+LABELS[right]+' differences for seeds 17, 42 and 89 are '+', '.join(f'{float(r["difference_pp"]):+.3f}' for r in sorted(rr,key=lambda r:int(r['seed'])))+' pp. '
    secondary+='\n'
    for right in ('uniform','tie_uniform'):
        values=[]
        for selector in (right,'soft'):
            u=avg(get('cross_evaluation',backbone='resnet18',state='selected',comparison=right,predictor=right,selector=selector),'risk',100)
            s=avg(get('cross_evaluation',backbone='resnet18',state='selected',comparison=right,predictor='soft',selector=selector),'risk',100)
            values.append(s-u)
        secondary+=f'On {LABELS[right]}-selected images the ResNet fixed-set difference is {values[0]:+.3f} pp; on Soft-selected images it is {values[1]:+.3f} pp. '
    secondary+=r'''The full fixed-set matrices, overlaps, reference F1, retention and learning trajectories are provided in the supplement and ledger. These comparisons extend architecture sensitivity but retain the same images, annotations and adaptive research history.
\begin{figure}[t]\centering\includegraphics[width=\linewidth]{backbone_followup/figures/BackboneContrasts.pdf}
\caption{Soft-minus-control risk at 80\% own-ranking coverage in two backbone families. Conditional pixel-cluster intervals do not represent seed-population uncertainty or external-domain confirmation. Negative values favor Soft; the figure does not rank architectures}\label{fig:backbones}\end{figure}
'''
    results=(OUT/'generated/results.tex').read_text()+ '\n'+secondary
    results=results.replace('Table~\\ref{tab:tieuniform} reports both checkpoint states for the new secondary comparison.','The tie-preserving comparison is evaluated at both checkpoint states; the later cross-backbone table also reports its contrast intervals.')
    results=results.replace('\\input{backbone_followup/generated/tie_control.tex}','').replace('\\input{backbone_followup/generated/epochs.tex}','')
    results=results.replace('Training-loss and selection-distance trajectories are supplied for every run in the supplement.','Selected epochs and complete training-loss/selection-distance trajectories are supplied in the supplement and ledger.')
    (OUT/'generated/results.tex').write_text(results)
    discussion=(OUT/'generated/discussion.tex').read_text().replace('three seeds and one model family','three seeds per family')
    discussion=discussion.replace('Tie-Uniform selected risk differs from Uniform by +0.210 pp. The residual own-ranking advantage persists under this tie-preserving control within the conditional interval. Because this adaptive check still uses the same source, three seeds per family, it is a representation sensitivity rather than independent replication.','In Swin-T, Tie-Uniform selected risk differs from Uniform by +0.210 pp. The residual Soft own-ranking advantage persists under this tie-preserving control within the conditional interval. This third adaptive phase reuses the same images and three Swin seeds; it is a representation sensitivity, not independent replication.')
    res=get('contrast',backbone='resnet18',state='selected',left='soft',right='uniform')[0]
    res2=get('contrast',backbone='resnet18',state='selected',left='soft',right='tie_uniform')[0]
    fixed=get('contrast',backbone='resnet18',state='epoch12',left='soft',right='uniform')[0]
    fixed2=get('contrast',backbone='resnet18',state='epoch12',left='soft',right='tie_uniform')[0]
    same=float(res['point_difference_pp'])<0 and float(res2['point_difference_pp'])<0
    conclusion='The selected own-ranking differences share the Swin direction in ResNet-18, but direction consistency is not a deployment benefit or external validation.' if same else 'ResNet-18 does not reproduce the selected Swin own-ranking advantage. Its conditional intervals cross zero; this neither establishes equivalence nor supports universal Soft superiority. The comparison is sensitive to the model family and shared-budget training configuration, not an independently isolated architecture effect.'
    discussion+='\n'+conclusion+' The magnitude and seed patterns must be interpreted alongside fixed-epoch and fixed-set results. Neither family tests a learned support-aware selector or an entropy-matched mechanism.\n'
    (OUT/'generated/discussion.tex').write_text(discussion)
    (OUT/'generated/conclusion.tex').write_text('The matched within-source study shows why supervision comparisons need target-softening controls, identical-image predictor comparisons and explicit annotation references. '+conclusion+' Unsupported-vote retention and calibration transfer constrain application interpretations. All selected/fixed-epoch weights and sample-level diagnostics are released. Independent-source and naturalistic messaging evaluation remain absent.\n')
    abstract=r'''\textbf{Purpose:} Crowd supervision can change confidence ranking without consistently improving category choice. We examine this distinction under an incomplete facial-expression output vocabulary.
\textbf{Methods:} Twenty-four matched runs on FER2013 images with FER+ votes compare five Swin-T target conditions and three ResNet-18 conditions across three seeds. Pixel-isolated selection, calibration and testing support own-ranking, fixed-image and fixed-epoch comparisons. Complete-vote disagreement is separated into unsupported mass, supported ambiguity and category-choice excess. Four adaptive phases reuse one image source.
\textbf{Results:} At 80\% coverage, selected-model Swin-T Soft-minus-Uniform disagreement is -0.38 percentage points; Soft-minus-Tie-Uniform is -0.59. ResNet-18 selected differences are RESNET1 and RESNET2 points; fixed-epoch differences are RESNET3 and RESNET4, with all four conditional intervals crossing zero. On Swin Uniform-selected images, Soft instead differs by +0.09 points. Changing the reference on identical images substantially changes F1. Soft retains more wholly unsupported images in Swin. All 36 nonempty empirical test operating points for the original four conditions exceed calibration targets.
\textbf{Conclusion:} Target representation, accepted-image composition, categorical reference and vocabulary mismatch shape apparent reliability. Cross-backbone checks extend sensitivity evidence, not independent-domain confirmation. These results support diagnostic reporting rather than universal superiority, deployment risk guarantees or validated messaging benefit.
'''
    abstract=abstract.replace('RESNET1',f'{float(res["point_difference_pp"]):+.2f}').replace('RESNET2',f'{float(res2["point_difference_pp"]):+.2f}').replace('RESNET3',f'{float(fixed["point_difference_pp"]):+.2f}').replace('RESNET4',f'{float(fixed2["point_difference_pp"]):+.2f}')
    (OUT/'generated/abstract.tex').write_text(abstract)
    # Phase table is retained historically but extended explicitly.
    p=OUT/'generated/phases.tex';t=p.read_text().replace('Phase & Target conditions & Runs & Prior outcomes known','Phase & Target conditions & Runs & Prior outcomes known').replace('\\bottomrule','4 & ResNet soft controls & 9 & Phases 1--3\\\\\n\\bottomrule').replace('All phases retain the same data roles and matched within-seed optimization.','Phase 4 compares Soft, Uniform and Tie-Uniform. All phases retain the same roles and matched within-backbone, within-seed optimization.');p.write_text(t)
    for name in ('backbone_contrasts','resnet_risks','resnet_retention'):
        p=OUT/'generated'/f'{name}.tex';(OUT/'generated'/f'supp_{name}.tex').write_text(p.read_text().replace('\\begin{table}[t]','\\begin{table}[htbp]'))
    sup=OUT/'generated/supplement.tex';t=sup.read_text().replace('Training loss and selection-distance trajectories for all fifteen runs.','Training loss and selection-distance trajectories for the fifteen retained Swin runs. ResNet trajectories are supplied separately below.')
    t=t.replace('all three adaptive','all four adaptive').replace('Online Resources 2--6','Online Resources 2--7').replace('Fixed model head used in all five conditions','Fixed model head used in the five Swin conditions')
    appendix='\\clearpage\\section*{Second-backbone robustness diagnostics}\n\\input{backbone_followup/generated/supp_backbone_contrasts.tex}\n\\input{backbone_followup/generated/supp_resnet_risks.tex}\n\\input{backbone_followup/generated/supp_resnet_retention.tex}\n'
    for state in ('selected','epoch12'):
        appendix+='\\clearpage\\subsection*{'+state+' fixed-image comparisons}\\begin{longtable}{lllrrr}\\toprule Control & Predictor & Selector & Seed 17 & Seed 42 & Seed 89\\\\\\midrule\\endhead\n'
        for right in ('uniform','tie_uniform'):
            for pred in ('soft',right):
                for sel in ('soft',right):appendix+=LABELS[right]+' & '+LABELS[pred]+' & '+LABELS[sel]+' & '+' & '.join(f'{avg(get("cross_evaluation",backbone="resnet18",state=state,comparison=right,predictor=pred,selector=sel,seed=s),"risk",100):.4f}' for s in (17,42,89))+r'\\'+'\n'
        appendix+='\\bottomrule\\end{longtable}\nEntries are complete-vote disagreement percentages at 2,620 retained images.\n'
    appendix+='\\clearpage\\begin{figure}[ht]\\centering\\includegraphics[width=\\linewidth]{backbone_followup/figures/ResNetLearning.pdf}\\caption{Selection-distance trajectories for the nine ResNet runs. Selected and epoch-12 models are both evaluated; these curves do not establish convergence}\\end{figure}\n'
    appendix+='\\clearpage\\begin{figure}[ht]\\centering\\includegraphics[width=\\linewidth]{backbone_followup/figures/ResNetDifferences.pdf}\\caption{ResNet selected-model own-ranking risk differences at 20--100\\% common coverage. Thin curves are paired seeds and thick curves their mean. The dotted marker identifies the prespecified 80\\% secondary endpoint. These descriptive curves have no simultaneous uncertainty bands and do not compare predictions on identical accepted images}\\end{figure}\n'
    appendix+='\\clearpage\\section*{ResNet reference-specific weighted F1}\n\\begin{longtable}{llrrr}\\toprule State & Targets & Original/all & Original/subset & Crowd/subset\\\\\\midrule\\endhead\n'
    for state in ('selected','epoch12'):
        for c in LABELS:appendix+=state+' & '+LABELS[c]+' & '+' & '.join(f'{avg(get("same_population_f1",backbone="resnet18",state=state,condition=c,population=pop,reference=ref),"weighted_f1"):.4f}' for pop,ref in [('all','original'),('majority_subset','original'),('majority_subset','crowd_majority')])+r'\\'+'\n'
    appendix+='\\bottomrule\\end{longtable}\nAll images: 3,275; identical majority subset in the last two columns: 2,484. No extra relabeling is performed. Per-seed unsupported masses and overlaps are supplied in the CSV ledger.\n'
    sup.write_text(t.replace('\\end{document}',appendix+'\\end{document}'))
    sources={p.name:sha(p) for p in (OUT/'reported_results.csv',OUT/'build_revision.py',OUT/'manuscript.tex')};(OUT/'asset_sources.json').write_text(json.dumps(sources,indent=2))
    generated=list((OUT/'generated').glob('*.tex'))+list((OUT/'figures').glob('*.pdf'))+[OUT/'references.bib']
    (OUT/'generated_sha256.json').write_text(json.dumps({p.relative_to(OUT).as_posix():sha(p) for p in generated},indent=2))
    print('Prepared two-backbone revision with all outcomes reported')
if __name__=='__main__':main()
