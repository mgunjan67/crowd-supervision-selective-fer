"""Training-only target intervention audit and explanatory scientific figure.

Executes the exact frozen target function without importing torch/CUDA or
loading image arrays. No model or held-out outcomes are accessed.
"""
import ast
import csv
import hashlib
import json
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
PAPER=HERE.parents[1]
TRAIN=HERE.parent/'training'
CLASSES=('angry','disgust','fear','happy','neutral','sad','surprise','contempt')
ABBREVIATIONS=('Ang','Dis','Fea','Hap','Neu','Sad','Sur','Con')
OLD_CONDITIONS=('hard','tie_hard','soft','uniform','tie_uniform')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def frozen_target_function():
    path=TRAIN/'train_followup.py'
    freeze=json.loads((TRAIN/'freeze.json').read_text())
    expected=freeze['source_sha256'][str(path.relative_to(PAPER)).replace('\\','/')]
    assert sha(path)==expected,'Frozen training target source changed'
    tree=ast.parse(path.read_text())
    functions=[node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name=='targets']
    assert len(functions)==1
    namespace={'np':np,'hashlib':hashlib}
    exec(compile(ast.Module(body=functions,type_ignores=[]),str(path),'exec'),namespace)
    return namespace['targets'],expected


def entropy(t):
    log=np.zeros_like(t)
    np.log(t,out=log,where=t>0)
    return -(t*log).sum(1)


def write_csv(name,records):
    with (HERE/name).open('w',newline='',encoding='utf-8') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(records[0]))
        writer.writeheader();writer.writerows(records)


def main():
    targets,target_source_sha=frozen_target_function()
    path=PAPER/'data/ferplus_annotations/fer2013new.csv'
    source_manifest=json.loads((PAPER/'final_experiment/split_manifest.json').read_text())
    assert sha(path)==source_manifest['source_hashes']['data/ferplus_annotations/fer2013new.csv']
    with path.open(newline='',encoding='utf-8-sig') as handle:
        allrows=list(csv.DictReader(handle))
    rows=np.array([i for i,r in enumerate(allrows) if r['Usage']=='Training'])
    vocab=('anger','disgust','fear','happiness','neutral','sadness','surprise','contempt','unknown','NF')
    votes=np.array([[int(allrows[i][key]) for key in vocab] for i in rows],dtype=float)
    q=votes/votes.sum(1,keepdims=True)
    data={};records=[];marginals=[]
    for k,conditions in [(7,('soft','uniform','tie_uniform','permutation')),(8,('soft','uniform','tie_uniform'))]:
        for condition in conditions:
            t,w=targets(q,rows,k,condition)
            positive=w>0
            h=entropy(t)
            marginal=(w[:,None]*t).sum(0)/w.sum()
            data[k,condition]=dict(t=t,w=w,h=h,marginal=marginal)
            soft=data[k,'soft']
            changed=np.any(t!=soft['t'],axis=1)&positive
            record=dict(k=k,condition=condition,training_images=len(t),positive_mass_images=int(positive.sum()),
                        zero_mass_images=int((~positive).sum()),changed_rows_from_soft=int(changed.sum()),
                        changed_fraction_all=float(changed.mean()),
                        changed_fraction_positive=float(changed.sum()/positive.sum()),
                        mean_entropy_positive_nats=float(h[positive].mean()),
                        weighted_mean_entropy_nats=float(np.dot(w,h)/w.sum()),
                        marginal_l1_from_soft=float(np.abs(marginal-soft['marginal']).sum()),
                        maximum_sorted_target_difference=float(np.max(np.abs(np.sort(t,axis=1)-np.sort(soft['t'],axis=1)))),
                        maximum_entropy_difference_from_soft=float(np.abs(h-soft['h']).max()))
            records.append(record)
            for index,category in enumerate(CLASSES[:k]):
                marginals.append(dict(k=k,condition=condition,category=category,
                                      weighted_class_marginal=float(marginal[index]),
                                      difference_from_soft_pp=float(100*(marginal[index]-soft['marginal'][index]))))
    # Reconcile the three overlapping conditions against the independent old-target
    # diagnostic, not against hand-entered expected statistics.
    with (HERE/'target_entropy_summary.csv').open(newline='') as handle:
        old=list(csv.DictReader(handle))
    reconciled=0
    for condition in ('soft','uniform','tie_uniform'):
        result=next(r for r in records if r['k']==7 and r['condition']==condition)
        for weighting,key in [('unweighted_positive_mass','mean_entropy_positive_nats'),
                              ('supervised_mass_weighted','weighted_mean_entropy_nats')]:
            previous=next(r for r in old if r['condition']==condition and r['weighting']==weighting)
            assert abs(result[key]-float(previous['mean_entropy_nats']))<1e-13
            assert result['positive_mass_images']==int(previous['positive_mass_n'])
            reconciled+=1
    seven=data[7,'soft'];permutation=data[7,'permutation']
    maxima=seven['t']==seven['t'].max(1,keepdims=True)
    assert np.array_equal(seven['t'][maxima],permutation['t'][maxima])
    assert np.array_equal(np.sort(seven['t'],axis=1),np.sort(permutation['t'],axis=1))
    assert np.max(np.abs(seven['h']-permutation['h']))<2e-15
    write_csv('new_target_audit.csv',records)
    write_csv('new_target_marginals.csv',marginals)

    # Two constructed, non-data examples retain all original target types and
    # add the actual frozen permutation intervention, using declared row IDs.
    example_q=np.array([[.6,.2,.1,.05,.03,.02,0,0,0,0],
                        [.3,.3,.15,.1,.05,.05,.05,0,0,0]],dtype=float)
    example_rows=np.array([1,2])
    labels=['Soft','Hard','Tie-hard','Uniform','Tie-Uniform','Permutation']
    examples=[]
    for name in labels:
        condition=name.lower().replace('-','_')
        if condition=='hard':
            tt=np.eye(7)[example_q[:,:7].argmax(1)]
        elif condition=='tie_hard':
            maximum=example_q[:,:7].max(1,keepdims=True)
            tied=example_q[:,:7]==maximum
            tt=tied/tied.sum(1,keepdims=True)
        else:
            tt,_=targets(example_q,example_rows,7,condition)
        examples.append(tt)
    examples=np.array(examples)
    example_records=[]
    for j in range(2):
        for i,name in enumerate(labels):
            for c,category in enumerate(CLASSES[:7]):
                example_records.append(dict(example='unique_maximum' if j==0 else 'tied_maxima',
                                            constructed_row_id=int(example_rows[j]),condition=name,
                                            category=category,target_probability=float(examples[i,j,c])))
    write_csv('constructed_target_examples.csv',example_records)
    make_figure(examples,labels,marginals)
    audit=dict(status='passed',training_only=True,new_model_inference=False,gpu_used=False,
               frozen_target_source_sha256=target_source_sha,annotation_sha256=sha(path),
               old_entropy_rows_reconciled=reconciled,permutation_maximum_positions_preserved=True,
               permutation_target_multiset_exactly_preserved=True,
               permutation_maximum_entropy_error=float(np.abs(seven['h']-permutation['h']).max()),
               rows=records,class_order=list(CLASSES),constructed_examples_not_observations=True,
               analysis_sha256=sha(Path(__file__)),
               output_hashes={name:sha(HERE/name) for name in ('new_target_audit.csv','new_target_marginals.csv',
                             'constructed_target_examples.csv','TargetIntervention.pdf','TargetIntervention.svg','TargetIntervention.png')})
    (HERE/'new_target_audit.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in audit.items() if k!='output_hashes'},indent=2))


def make_figure(examples,labels,marginals):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'serif','font.size':9,'pdf.fonttype':42,
                         'ps.fonttype':42,'svg.fonttype':'none'})
    fig=plt.figure(figsize=(7.1,4.8))
    grid=fig.add_gridspec(2,2,height_ratios=[1.38,1.0],hspace=.75,wspace=.15)
    axes=[fig.add_subplot(grid[0,j]) for j in range(2)]
    for j,ax in enumerate(axes):
        values=examples[:,j,:]
        ax.imshow(values,vmin=0,vmax=1,cmap='Blues',aspect='auto')
        ax.set_xticks(np.arange(7),ABBREVIATIONS[:7])
        ax.set_yticks(np.arange(6),labels if j==0 else ['']*6)
        ax.tick_params(length=0,labelsize=8)
        ax.set_title('(a) Unique maximum\nConstructed, row ID 1' if j==0 else
                     '(b) Tied maxima\nConstructed, row ID 2',fontsize=9.5,pad=9)
        for y in range(6):
            for x in range(7):
                value=values[y,x]
                text='0' if value==0 else f'{value:.2f}'.lstrip('0')
                ax.text(x,y,text,ha='center',va='center',fontsize=7.3,
                        color='white' if value>.55 else 'black')
        for spine in ax.spines.values():spine.set_visible(False)
        ax.set_xticks(np.arange(-.5,7,.99999999),minor=True)
        ax.set_yticks(np.arange(-.5,6,.99999999),minor=True)
        ax.grid(which='minor',color='white',lw=.7)
        ax.tick_params(which='minor',length=0)
    bottom=fig.add_subplot(grid[1,:])
    x=np.arange(7)
    for i,(condition,label,color) in enumerate([('uniform','Uniform','#3b78a4'),
                                               ('tie_uniform','Tie-Uniform','#699b72'),
                                               ('permutation','Permutation','#ad632a')]):
        values=[next(r['difference_from_soft_pp'] for r in marginals if r['k']==7
                     and r['condition']==condition and r['category']==category) for category in CLASSES[:7]]
        bottom.bar(x+(i-1)*.23,values,width=.23,label=label,color=color)
    bottom.axhline(0,color='.3',lw=.7)
    bottom.set_xticks(x,ABBREVIATIONS[:7])
    bottom.set_ylabel('Marginal difference (pp)',fontsize=9)
    bottom.set_title('(c) Observed training targets relative to Soft',fontsize=10,pad=9)
    bottom.spines[['top','right']].set_visible(False)
    bottom.grid(axis='y',alpha=.2)
    bottom.set_axisbelow(True)
    bottom.legend(loc='upper center',bbox_to_anchor=(.5,-.21),ncol=3,frameon=False,fontsize=8)
    fig.subplots_adjust(left=.13,right=.985,top=.92,bottom=.14)
    for suffix in ('pdf','svg','png'):
        fig.savefig(HERE/f'TargetIntervention.{suffix}',bbox_inches='tight',pad_inches=.12,
                    **({'dpi':240} if suffix=='png' else {}))
    plt.close(fig)


if __name__=='__main__':main()
