"""Fixed CPU ranking extension; all new inference must finish before analysis."""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys
import numpy as np

HERE=Path(__file__).resolve().parent
PAPER=HERE.parents[1]
sys.path.insert(0,str(HERE))
from contracts_followup import sha,validate_evaluation

RESULTS=HERE/'results'
SEEDS=(17,42,89)
EXPONENTS=(1.,1.5,2.,3.,4.,6.,8.,16.)
COVERAGES=(.2,.5,.8,1.)
SCORES=('msp','margin','negative_entropy','centered_logit_norm','calibration_choice')


def planned_keys():
    new=[(7,'permutation',s,state) for s in SEEDS for state in ('selected','epoch12')]
    new.extend((8,c,s,state) for s in SEEDS for c in ('soft','uniform','tie_uniform')
               for state in ('selected','epoch12','risk_selected'))
    old=[(7,'soft',s,state) for s in SEEDS for state in ('selected','epoch12')]
    return new,old


def contrast_groups():
    return [(7,state,'permutation') for state in ('selected','epoch12')]+[
        (8,state,c) for state in ('selected','epoch12','risk_selected') for c in ('uniform','tie_uniform')]


def legacy_path(seed,state,role):
    return PAPER/'reviewer_followup/predictions'/f'soft_seed{seed}_{state}'/f'{role}.npz'


def source_paths():
    paths=[HERE/'RANKING_EXTENSION.md',Path(__file__),HERE/'test_ranking_followup.py',
           HERE/'contracts_followup.py',PAPER/'final_experiment/split_manifest.json',
           PAPER/'final_revision_2026_10_20/diagnostics/ranking_tuning.csv',
           PAPER/'final_revision_2026_10_20/diagnostics/ranking_results.csv',
           PAPER/'recovery_2026_10_10/selector_results.csv']
    paths.extend(legacy_path(seed,state,role) for seed in SEEDS for state in ('selected','epoch12')
                 for role in ('calibration','test_full'))
    return paths


def frozen_record():
    new,old=planned_keys()
    return dict(new_states=new,historical_soft_comparison_states=old,exponent_grid=EXPONENTS,
                scores=SCORES,coverages=COVERAGES,
                source_sha256={str(p.relative_to(PAPER)).replace('\\','/'):sha(p) for p in source_paths()})


def freeze():
    destination=HERE/'ranking_extension_freeze.json'
    current=json.loads(json.dumps(frozen_record()))
    if destination.exists():
        saved=json.loads(destination.read_text())
        assert saved['plan']==current,'Ranking extension sources or plan changed'
        return saved
    assert not (HERE/'evaluation_freeze.json').exists(),'Freeze must precede new inference'
    assert not (HERE/'evaluation_complete.json').exists()
    assert not any((HERE/'predictions').glob('*/calibration.npz'))
    assert not any((HERE/'predictions').glob('*/test_full.npz'))
    saved=dict(created_utc=datetime.now(timezone.utc).isoformat(),
               new_calibration_or_test_available_at_freeze=False,
               prior_historical_results_known=True,plan=current)
    destination.write_text(json.dumps(saved,indent=2),encoding='utf-8')
    return saved


def require_freeze():
    saved=json.loads((HERE/'ranking_extension_freeze.json').read_text())
    assert saved['new_calibration_or_test_available_at_freeze'] is False
    assert saved['plan']==json.loads(json.dumps(frozen_record()))
    return saved


def write_csv(path,records):
    assert records
    with path.open('w',newline='',encoding='utf-8') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)


def read_csv(path):
    with path.open(newline='',encoding='utf-8') as handle:return list(csv.DictReader(handle))


def normalized_score(p,exponent):
    logp=np.log(np.maximum(np.asarray(p,dtype=float),np.finfo(np.float32).tiny))
    z=logp-logp.mean(1,keepdims=True)
    scale=np.max(np.abs(z),axis=1,keepdims=True)
    zz=np.divide(z,scale,out=np.zeros_like(z),where=scale>0)
    norm=np.power(np.sum(np.abs(zz)**exponent,axis=1),1/exponent)
    return np.divide(zz.max(1),norm,out=np.zeros(len(p)),where=norm>0)


def ordinary_score(p,name):
    if name=='msp':return p.max(1)
    if name=='margin':
        top=np.partition(p,-2,axis=1)[:,-2:]
        return top.max(1)-top.min(1)
    if name=='negative_entropy':
        log=np.zeros_like(p);np.log(p,out=log,where=p>0)
        return (p*log).sum(1)
    raise ValueError(name)


def rank(score,rows):return np.lexsort((rows,-score))


def aurc(score,loss,rows):
    return float((np.cumsum(loss[rank(score,rows)])/np.arange(1,len(rows)+1)).mean())


def selected_risk(score,loss,rows,coverage):
    n=math.ceil(coverage*len(rows))
    return float(loss[rank(score,rows)[:n]].mean()),n


def loss(a):
    p,q=a['probabilities'],a['q']
    return 1-q[np.arange(len(p)),p.argmax(1)]


def validate_arrays(a,k):
    p,q,rows=a['probabilities'],a['q'],a['rows']
    assert p.shape==(len(rows),k) and q.shape==(len(rows),10)
    assert len(rows)>0 and len(np.unique(rows))==len(rows)
    assert np.isfinite(p).all() and (p>=0).all() and (p<=1).all()
    assert np.isfinite(q).all() and (q>=0).all()
    assert np.allclose(p.sum(1),1,atol=1e-6,rtol=0) and np.allclose(q.sum(1),1)


def analyze_models(models):
    expected=set(sum(planned_keys(),[]))
    assert set(models)==expected and len(models)==39
    outcomes=[];tuning=[];candidates=[];inputs=[]
    for key,model in models.items():
        k,condition,seed,state=key
        base=dict(k=k,condition=condition,seed=seed,state=state,
                  origin='historical_comparator' if k==7 and condition=='soft' else 'followup')
        calibration,test=model['calibration'],model['test']
        for role,a in model.items():
            validate_arrays(a,k)
            p=a['probabilities']
            inputs.append(dict(**base,role=role,n=len(p),minimum_probability=float(p.min()),
                               zero_entries=int((p==0).sum()),
                               clipped_entries=int((p<np.finfo(np.float32).tiny).sum())))
        cal_loss,test_loss=loss(calibration),loss(test)
        cp,tp=calibration['probabilities'],test['probabilities']
        values=[aurc(normalized_score(cp,e),cal_loss,calibration['rows']) for e in EXPONENTS]
        chosen=int(np.argmin(values));exponent=EXPONENTS[chosen]
        msp_cal=aurc(cp.max(1),cal_loss,calibration['rows'])
        fallback='centered_logit_norm' if values[chosen]<msp_cal else 'msp'
        for index,e in enumerate(EXPONENTS):
            test_score=normalized_score(tp,e)
            test_auc=aurc(test_score,test_loss,test['rows'])
            tuning.append(dict(**base,exponent=e,calibration_aurc=values[index],
                               selected=int(index==chosen),msp_calibration_aurc=msp_cal,
                               fallback_score=fallback,test_aurc=test_auc))
            for coverage in COVERAGES:
                risk,n=selected_risk(test_score,test_loss,test['rows'],coverage)
                candidates.append(dict(**base,exponent=e,selected=int(index==chosen),
                                       requested_coverage=coverage,accepted_n=n,risk=risk,
                                       calibration_aurc=values[index],test_aurc=test_auc))
        scores={name:ordinary_score(tp,name) for name in ('msp','margin','negative_entropy')}
        scores['centered_logit_norm']=normalized_score(tp,exponent)
        scores['calibration_choice']=scores[fallback]
        for name,score in scores.items():
            for coverage in COVERAGES:
                risk,n=selected_risk(score,test_loss,test['rows'],coverage)
                if coverage==1:assert abs(risk-test_loss.mean())<1e-13
                outcomes.append(dict(**base,score=name,exponent=exponent,calibration_choice=fallback,
                                     requested_coverage=coverage,population_n=len(tp),accepted_n=n,risk=risk))
    lookup={(r['k'],r['condition'],r['seed'],r['state'],r['score'],r['requested_coverage']):r for r in outcomes}
    paired=[];means=[]
    for k,state,control in contrast_groups():
        for name in SCORES:
            for coverage in COVERAGES:
                values=[]
                for seed in SEEDS:
                    delta=100*(lookup[k,'soft',seed,state,name,coverage]['risk']-
                               lookup[k,control,seed,state,name,coverage]['risk'])
                    values.append(delta)
                    paired.append(dict(k=k,state=state,control=control,score=name,
                                       requested_coverage=coverage,seed=seed,difference_pp=delta))
                means.append(dict(k=k,state=state,control=control,score=name,requested_coverage=coverage,
                                  paired_seeds=3,mean_difference_pp=float(np.mean(values)),
                                  sample_seed_sd_pp=float(np.std(values,ddof=1)),
                                  minimum_seed_pp=min(values),maximum_seed_pp=max(values)))
    assert len(outcomes)==780 and len(tuning)==312 and len(candidates)==1248
    assert len(paired)==480 and len(means)==160
    return dict(results=outcomes,tuning=tuning,candidates=candidates,paired=paired,means=means,inputs=inputs)


def load_models():
    manifest=json.loads((PAPER/'final_experiment/split_manifest.json').read_text())
    model_runs=json.loads((HERE/'freeze.json').read_text())['runs']
    integrity=validate_evaluation(HERE,PAPER,model_runs,portable=True)
    models={};input_hashes={};canonical={}
    new,old=planned_keys()
    for key in new+old:
        k,condition,seed,state=key;model={}
        for role,filename in [('calibration','calibration'),('test','test_full')]:
            path=(legacy_path(seed,state,filename) if key in old else
                  HERE/'predictions'/f'k{k}_{condition}_seed{seed}_{state}'/f'{filename}.npz')
            input_hashes[str(path.relative_to(PAPER)).replace('\\','/')]=sha(path)
            with np.load(path,allow_pickle=False) as data:
                a={name:data[name] for name in ('probabilities','q','rows','hashes')}
            ids=manifest['roles']['calibration' if role=='calibration' else 'test']
            keep=np.isin(a['rows'],ids);a={name:value[keep] for name,value in a.items()}
            a['probabilities']=a['probabilities'].astype(float)
            assert set(a['rows'])==set(ids) and len(a['rows'])==(1642 if role=='calibration' else 3275)
            validate_arrays(a,k)
            if role not in canonical:canonical[role]=a
            for name in ('rows','q','hashes'):assert np.array_equal(a[name],canonical[role][name])
            model[role]=a
        models[key]=model
    return models,integrity,input_hashes


def reconcile(records):
    earlier=PAPER/'final_revision_2026_10_20/diagnostics'
    old_tuning=[r for r in read_csv(earlier/'ranking_tuning.csv') if r['backbone']=='swin_t' and r['condition']=='soft']
    old_results=[r for r in read_csv(earlier/'ranking_results.csv') if r['backbone']=='swin_t' and r['condition']=='soft']
    old_scores=[r for r in read_csv(PAPER/'recovery_2026_10_10/selector_results.csv') if r['backbone']=='swin_t' and r['condition']=='soft']
    tuning={(r['seed'],r['state'],float(r['exponent'])):r for r in old_tuning}
    counts=dict(legacy_candidate_tuning=0,legacy_normalization_results=0,legacy_ordinary_scores=0,new_msp_primary_rows=0)
    for row in records['tuning']:
        if row['origin']!='historical_comparator':continue
        prior=tuning[str(row['seed']),row['state'],row['exponent']]
        assert abs(row['calibration_aurc']-float(prior['calibration_aurc']))<1e-13
        assert row['selected']==int(prior['selected']) and row['fallback_score']==prior['fallback_score']
        counts['legacy_candidate_tuning']+=1
    lookup={(r['seed'],r['state'],r['score'],float(r['requested_coverage'])):r for r in old_results}
    ordinary={(r['seed'],r['state'],r['score'],float(r['requested_coverage'])):r for r in old_scores}
    primary=read_csv(RESULTS/'all_states.csv')
    new_lookup={(int(r['k']),r['condition'],int(r['seed']),r['state']):r for r in primary}
    for row in records['results']:
        if row['origin']=='historical_comparator':
            key=(str(row['seed']),row['state'],row['score'],row['requested_coverage'])
            if row['score'] in ('msp','centered_logit_norm','calibration_choice'):
                assert abs(row['risk']-float(lookup[key]['risk']))<1e-13
                counts['legacy_normalization_results']+=1
            if row['score'] in ('msp','margin','negative_entropy'):
                assert abs(row['risk']-float(ordinary[key]['risk']))<1e-13
                counts['legacy_ordinary_scores']+=1
        elif row['score']=='msp' and row['requested_coverage']==.8:
            prior=new_lookup[row['k'],row['condition'],row['seed'],row['state']]
            assert abs(100*row['risk']-float(prior['risk_pct']))<1e-11
            counts['new_msp_primary_rows']+=1
    assert counts==dict(legacy_candidate_tuning=48,legacy_normalization_results=72,legacy_ordinary_scores=72,new_msp_primary_rows=33)
    return counts


def export_table(means):
    primary=[row for row in means if float(row['requested_coverage'])==.8]
    lines=[r'\begingroup\small\setlength{\tabcolsep}{3pt}',r'\begin{longtable}{rllrrrrr}',
           r'\caption{Fixed ranking sensitivities for all new paired comparisons at 80\% coverage. Values are means of three paired seed differences, in percentage points; negative favors Soft. Norm. uses a calibration-selected exponent, and Choice compares only MSP with Norm. on calibration AURC. These are adaptive descriptive sensitivities, not new hypothesis tests.}\label{tab:new-ranking-extension}\\',
           r'\toprule K & Control & State & MSP & Margin & Neg. entropy & Norm. & Choice\\\midrule\endfirsthead',
           r'\toprule K & Control & State & MSP & Margin & Neg. entropy & Norm. & Choice\\\midrule\endhead']
    labels={'permutation':'Permutation','uniform':'Uniform','tie_uniform':'Tie-Uniform'}
    states={'selected':'Distance','epoch12':'Epoch 12','risk_selected':'Risk'}
    for k,state,control in contrast_groups():
        selected=[row for row in primary if int(row['k'])==k and row['state']==state and row['control']==control]
        assert len(selected)==5 and {row['score'] for row in selected}==set(SCORES)
        lookup={row['score']:float(row['mean_difference_pp']) for row in selected}
        lines.append(f'{k} & {labels[control]} & {states[state]} & '+
                     ' & '.join(f'{lookup[name]:+.3f}' for name in SCORES)+r'\\')
    lines.extend([r'\bottomrule\end{longtable}\endgroup',''])
    destination=RESULTS/'new_ranking_table.tex';destination.write_text('\n'.join(lines),encoding='utf-8')
    return destination


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--freeze',action='store_true')
    parser.add_argument('--export-tex',action='store_true');args=parser.parse_args()
    if args.freeze:
        saved=freeze();print(json.dumps(dict(freeze_sha256=sha(HERE/'ranking_extension_freeze.json'),
                                            new_states=len(saved['plan']['new_states']),legacy_states=6),indent=2));return
    require_freeze()
    if args.export_tex:
        print(export_table(read_csv(RESULTS/'new_ranking_means.csv')));return
    assert (HERE/'evaluation_complete.json').exists(),'All new inference must finish before the extension'
    assert (RESULTS/'summary.json').exists(),'Run the primary frozen analysis first'
    models,integrity,input_hashes=load_models();records=analyze_models(models)
    reconciled=reconcile(records)
    for name,rows in records.items():write_csv(RESULTS/f'new_ranking_{name}.csv',rows)
    export_table(records['means'])
    report=dict(status='complete',exploratory=True,nominal_states=39,new_states=33,historical_comparator_states=6,
                scores=list(SCORES),exponents=list(EXPONENTS),coverages=list(COVERAGES),
                no_new_training=True,no_new_inference=True,gpu_used=False,
                extension_freeze_sha256=sha(HERE/'ranking_extension_freeze.json'),
                evaluation_integrity=integrity,input_sha256=input_hashes,reconciliation=reconciled,
                normalized_fallback_selected_new=sum(r['origin']=='followup' and r['selected']==1 and r['fallback_score']=='centered_logit_norm' for r in records['tuning']),
                zero_probability_entries=sum(r['zero_entries'] for r in records['inputs']),
                output_sha256={p.name:sha(p) for p in RESULTS.glob('new_ranking_*') if p.suffix in ('.csv','.tex')})
    (RESULTS/'new_ranking_summary.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({key:value for key,value in report.items() if key not in ('evaluation_integrity','input_sha256','output_sha256')},indent=2))


if __name__=='__main__':main()
