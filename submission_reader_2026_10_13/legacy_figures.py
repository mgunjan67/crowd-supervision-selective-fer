"""Regenerate retained scientific figures without mutating historical editions.

Original editable plotting routines run only in a new staging directory. For the
older mixed manuscript/plot builder, the recorded AST span contains plotting
statements only; manuscript-rewrite statements are not executed.
"""
import ast
import csv
import hashlib
import importlib.util
import json
import shutil
import sys
from pathlib import Path

HERE=Path(__file__).resolve().parent;PAPER=HERE.parent

def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module
def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def span(path,function,start,end,namespace,skip=()):
    tree=ast.parse(path.read_text());body=next(n.body for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==function)
    nodes=[n for n in body if n.lineno>=start and n.end_lineno<=end and n.lineno not in skip]
    assert nodes and nodes[0].lineno==start,(path,start)
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[])),str(path),'exec'),namespace)

def main():
    stage=HERE/'checks/figure_staging';stage.mkdir(exist_ok=True)
    for name in ('original','journal','external'):
        (stage/name/'figures').mkdir(parents=True,exist_ok=True)
        (stage/name/'generated').mkdir(exist_ok=True)
    target=HERE/'source/figures'
    old=PAPER/'evidence_revision';out=stage/'original'
    path=old/'retained_assets.py'
    assert digest(path)=='a981b4a24b0a68ea66ff3e1ac0303d9d439d4b03f7c1ffd75734fc084b166234'
    retained=load('retained_plot_only',path)
    for name in ('audit.json','reported_results.csv','split_manifest.json','manuscript.tex','retained_assets.py','references.bib'):shutil.copy2(old/name,out/name)
    retained.OUT=out;retained.GEN=out/'generated';retained.FIG=out/'figures';retained.ROOT=PAPER
    retained.main()
    for oldname,newname in [('Fig2','HistoricalFig2'),('Fig4','Fig6')]:
        for ext in ('pdf','eps'):shutil.copy2(out/'figures'/f'{oldname}.{ext}',target/f'{newname}.{ext}')
    style=PAPER/'journal_style_revision'
    sys.path.insert(0,str(style))
    module=load('journal_plots_only',style/'build_revision.py');module.OUT=stage/'journal'
    shutil.copy2(PAPER/'backbone_followup/reported_results.csv',module.OUT/'reported_results.csv')
    import fix_figure_layout
    fix_figure_layout.OUT=module.OUT
    # Exact immutable historical plotting span; no document editing is invoked.
    assert digest(style/'build_revision.py')=='dcc20e30b329a9b838f734d5e0e58ccda956e90c2fecd3dbc8ef0b532aa689b9'
    # The skipped statement only copies an old calibration figure; the figure
    # was already regenerated above from the verified four-condition ledger.
    span(style/'build_revision.py','main',50,91,module.__dict__,skip=(82,))
    names=('Fig1','Fig2','Fig3','Fig4','Fig5','Fig7','FigS2','FigS3','LearningCurves','ResNetLearning','ResNetDifferences')
    for name in names:
        for ext in ('pdf','eps'):shutil.copy2(module.OUT/'figures'/f'{name}.{ext}',target/f'{name}.{ext}')
    # Reference-to-reference counts, rather than prediction confusion.
    evidence=load('reference_plot_only',old/'build_revision.py')
    ledger=list(csv.DictReader((old/'reported_results.csv').open(newline='')))
    def rows(kind,**criteria):return [r for r in ledger if r['record_type']==kind and all(r[k]==str(v) for k,v in criteria.items())]
    evidence.rows=rows;evidence.builder=retained
    span(old/'build_revision.py','main',136,141,evidence.__dict__)
    for ext in ('pdf','eps'):shutil.copy2(out/'figures'/f'ReferenceConfusion.{ext}',target/f'ReferenceConfusion.{ext}')
    # Existing external score sensitivity from the complete frozen contrast ledger.
    external=load('external_plots_only',PAPER/'ravdess_transfer_2026_10_11/build_paper.py');external.SOURCE=stage/'external'
    intervals=external.rows('actor_intervals.csv');contrasts=external.rows('paired_seed_contrasts.csv')
    lookup={(r['backbone'],r['state'],r['reference'],r['control']):r for r in intervals}
    seeds={(r['backbone'],r['state'],r['reference'],r['score'],float(r['requested_coverage']),r['control'],int(r['seed'])):r for r in contrasts}
    external.make_figures(lookup,seeds)
    for ext in ('pdf','eps','svg'):shutil.copy2(external.SOURCE/'figures'/f'ExternalScores.{ext}',target/f'ExternalScores.{ext}')
    # Within-source score sensitivity, all three seeds, both states and families.
    import matplotlib.pyplot as plt
    import numpy as np
    rr=list(csv.DictReader((PAPER/'recovery_2026_10_10/selector_contrasts.csv').open(newline='')))
    fig,axes=plt.subplots(1,2,figsize=(7.1,3.35),sharey=True,layout='constrained')
    for ax,state in zip(axes,('selected','epoch12')):
        for family,color,marker,dx in [('swin_t','#1b5592','o',-.12),('resnet18','#a04816','s',.12)]:
            for x,score in enumerate(('msp','margin','negative_entropy')):
                data=sorted([r for r in rr if r['backbone']==family and r['state']==state and r['score']==score and r['control']=='uniform' and float(r['requested_coverage'])==.8],key=lambda r:int(r['seed']))
                assert len(data)==3;yy=[float(r['difference_pp']) for r in data]
                ax.scatter(x+dx+np.array([-.035,0,.035]),yy,s=26,marker=marker,facecolors='white',edgecolors=color,linewidths=.9,zorder=3)
                ax.scatter(x+dx,np.mean(yy),s=62,marker=marker,color=color,zorder=4,label=('Swin-T' if family=='swin_t' else 'ResNet-18') if x==0 else None)
        ax.axhline(0,color='.25',ls='--',lw=.8);ax.set_xticks([0,1,2],['MSP','Margin','Negative\nentropy']);ax.set_xlim(-.5,2.5);ax.set_ylim(-.56,.88);ax.set_title('Selected' if state=='selected' else 'Epoch 12',loc='left');ax.grid(axis='y',color='.9',lw=.6);ax.set_xlabel('Confidence score')
    axes[0].set_ylabel('Soft minus Uniform disagreement (pp)');axes[0].legend(loc='upper left',frameon=False,fontsize=9)
    for ext in ('pdf','eps','svg'):fig.savefig(target/f'SelectorSensitivity.{ext}')
    plt.close(fig)
    report={'status':'passed','historical_editions_modified':False,'regenerated_figures':list(names)+['HistoricalFig2','Fig6','ReferenceConfusion','ExternalScores','SelectorSensitivity'],'ledger_sha256':digest(PAPER/'backbone_followup/reported_results.csv'),'plotting_source_spans':{'journal_style_revision/build_revision.py':[50,91],'evidence_revision/build_revision.py':[136,141]}}
    (HERE/'checks/legacy_figure_regeneration.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))

if __name__=='__main__':main()
