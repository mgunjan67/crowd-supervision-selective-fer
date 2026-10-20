"""Prespecified clip-level analyses; only runs after all 36 states complete."""
import json
import time
from pathlib import Path
import numpy as np
from common import (HERE,PAPER,CLASSES,REFERENCES,SEEDS,CONDITIONS,STATES,SCORES,COVERAGES,
                    annotations,reference,components,select,ranked_valid,bootstrap_risk,write_csv,require_frozen,sha)


def main():
    started=time.monotonic();frozen=require_frozen()
    complete=json.loads((HERE/'inference_complete.json').read_text())
    assert complete['checkpoint_count']==36 and len(complete['actors'])==24
    # The overlap audit must be completed before interpreting any scientific outcomes.
    overlap=json.loads((HERE/'overlap_audit.json').read_text())
    assert overlap['status']=='completed' and overlap['model_outcomes_used'] is False
    names,actors,q=annotations();models={};source_hashes={};records=[]
    for item in frozen['checkpoints']:
        key=(item['backbone'],item['condition'],item['seed'],item['state'])
        stem=f"{key[0]}_{key[1]}_seed{key[2]}_{key[3]}"
        path=HERE/'predictions'/(stem+'.npz');info=json.loads(path.with_suffix('.json').read_text())
        assert sha(path)==info['output_sha256'] and info['freeze_sha256']==sha(HERE/'freeze.json')
        source_hashes[str(path.relative_to(HERE))]=sha(path)
        with np.load(path,allow_pickle=False) as data:
            p=data['probabilities'];valid=data['valid'];counts=data['valid_frame_count']
            assert np.array_equal(data['filenames'],names) and np.array_equal(data['actors'],actors)
            assert np.array_equal(valid,counts>=4) and np.isnan(p[~valid]).all()
        if models:assert np.array_equal(valid,next(iter(models.values()))['valid'])
        values={r:components(p,reference(q,r),valid) for r in REFERENCES}
        selections={}
        for score in SCORES:
            for coverage in COVERAGES:
                keep=select(p,valid,score,coverage);selections[score,coverage]=keep
                for ref in REFERENCES:
                    metrics=np.mean(values[ref][keep],axis=0) if keep is not None else np.full(4,np.nan)
                    records.append(dict(backbone=key[0],condition=key[1],seed=key[2],state=key[3],
                                        reference=ref,score=score,requested_coverage=coverage,
                                        population_n=1440,eligible_n=int(valid.sum()),
                                        accepted_n=len(keep) if keep is not None else '',
                                        achieved_coverage=len(keep)/1440 if keep is not None else '',
                                        status='available' if keep is not None else 'unavailable_insufficient_eligible_clips',
                                        **{k:float(v) if np.isfinite(v) else '' for k,v in zip(('risk','unsupported','supported_ambiguity','excess'),metrics)}))
        for ref in REFERENCES:
            metrics=np.mean(values[ref][valid],axis=0)
            records.append(dict(backbone=key[0],condition=key[1],seed=key[2],state=key[3],
                                reference=ref,score='all_eligible',requested_coverage='',population_n=1440,
                                eligible_n=int(valid.sum()),accepted_n=int(valid.sum()),achieved_coverage=float(valid.mean()),
                                status='available',**{k:float(v) for k,v in zip(('risk','unsupported','supported_ambiguity','excess'),metrics)}))
        models[key]=dict(p=p,valid=valid,values=values,selections=selections)
    assert len(records)==936 and len(models)==36
    lookup={(r['backbone'],r['condition'],r['seed'],r['state'],r['reference'],r['score'],r['requested_coverage']):r for r in records}
    contrasts=[];summaries=[];fixed=[];intervals=[]
    for family in ('swin_t','resnet18'):
        for state in STATES:
            for ref in REFERENCES:
                for score in SCORES:
                    for coverage in COVERAGES:
                        for control in ('uniform','tie_uniform'):
                            values=[]
                            for seed in SEEDS:
                                left=lookup[family,'soft',seed,state,ref,score,coverage]
                                right=lookup[family,control,seed,state,ref,score,coverage]
                                available=left['status']==right['status']=='available'
                                delta=100*(left['risk']-right['risk']) if available else None
                                if delta is not None:values.append(delta)
                                contrasts.append(dict(backbone=family,state=state,reference=ref,score=score,
                                                      requested_coverage=coverage,control=control,seed=seed,
                                                      status='available' if available else 'unavailable',
                                                      difference_pp=delta if delta is not None else ''))
                            summaries.append(dict(backbone=family,state=state,reference=ref,score=score,
                                                  requested_coverage=coverage,control=control,paired_seeds=len(values),
                                                  mean_difference_pp=float(np.mean(values)) if len(values)==3 else '',
                                                  minimum_seed_difference_pp=min(values) if values else '',
                                                  maximum_seed_difference_pp=max(values) if values else ''))
                for control in ('uniform','tie_uniform'):
                    for seed in SEEDS:
                        for selector in ('soft',control):
                            keep=models[family,selector,seed,state]['selections']['msp',.8]
                            for predictor in ('soft',control):
                                risk=(100*float(models[family,predictor,seed,state]['values'][ref][keep,0].mean())
                                      if keep is not None else '')
                                fixed.append(dict(backbone=family,state=state,reference=ref,control=control,
                                                  seed=seed,selector=selector,predictor=predictor,
                                                  accepted_n=len(keep) if keep is not None else '',risk_percent=risk))
    with np.load(HERE/'bootstrap_actor_counts.npz',allow_pickle=False) as f:boot_counts=f['counts']
    assert boot_counts.shape==(5000,24) and np.all(boot_counts.sum(1)==24)
    boot_risks={}
    for family in ('swin_t','resnet18'):
        for state in STATES:
            for condition in CONDITIONS:
                for seed in SEEDS:
                    model=models[family,condition,seed,state]
                    order=ranked_valid(model['p'],model['valid'],'msp')
                    for ref in REFERENCES:
                        boot_risks[family,state,condition,seed,ref]=bootstrap_risk(order,model['values'][ref][:,0],actors,boot_counts)
            print('BOOTSTRAP '+family+' '+state,flush=True)
            for ref in REFERENCES:
                for control in ('uniform','tie_uniform'):
                    paired=np.stack([100*(boot_risks[family,state,'soft',seed,ref]-boot_risks[family,state,control,seed,ref]) for seed in SEEDS])
                    all_available=np.isfinite(paired).all(0)
                    means=paired[:,all_available].mean(0)
                    point=next(r['mean_difference_pp'] for r in summaries if r['backbone']==family and r['state']==state
                               and r['reference']==ref and r['score']=='msp' and r['requested_coverage']==.8 and r['control']==control)
                    # Do not give an apparently complete CI if any replicate endpoint is unavailable.
                    ci=np.quantile(means,[.025,.975]) if all_available.all() and point!='' else [None,None]
                    intervals.append(dict(backbone=family,state=state,reference=ref,score='msp',requested_coverage=.8,
                                          control=control,mean_difference_pp=point,lower95_pp=ci[0],upper95_pp=ci[1],
                                          bootstrap_replicates=5000,available_replicates=int(all_available.sum()),
                                          status='available' if ci[0] is not None else 'unavailable_or_incomplete',
                                          uncertainty='actor-cluster percentile; frozen fits and observed votes'))
    assert len(contrasts)==576 and len(summaries)==192 and len(fixed)==192 and len(intervals)==16
    output=HERE/'results';output.mkdir(exist_ok=True)
    for filename,values in [('clip_results.csv',records),('paired_seed_contrasts.csv',contrasts),
                            ('paired_mean_contrasts.csv',summaries),('fixed_image_comparisons.csv',fixed),
                            ('actor_intervals.csv',intervals)]:write_csv(output/filename,values)
    failure_counts={};prepared=[]
    for actor in range(1,25):
        report=json.loads((HERE/f'cache/Actor_{actor:02d}.json').read_text());prepared.append(report)
        for frame in report['frames']:failure_counts[frame['status']]=failure_counts.get(frame['status'],0)+1
    result=dict(status='complete',freeze_sha256=sha(HERE/'freeze.json'),input_hashes=source_hashes,
                output_hashes={path.name:sha(path) for path in output.glob('*.csv')},
                population_clips=1440,actors=24,human_responses_per_clip=10,inference_states=36,new_training_runs=0,
                eligible_clips=int(next(iter(models.values()))['valid'].sum()),frame_status_counts=failure_counts,
                actor_bootstrap_replicates=5000,elapsed_analysis_seconds=time.monotonic()-started,
                external_unit='acted video-only speech clip with frozen static-model probability aggregation',
                identity_disjointness_verified=False,overlap_audit_sha256=sha(HERE/'overlap_audit.json'),
                limits=['External source was chosen after known FER results','No temporal model or static-image replication',
                        'No learned-selector comparison','Only 24 actors; annotation counts conditioned on',
                        'No communication benefit or broad demographic validity established'])
    (output/'analysis_audit.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(dict(audit=result,primary_and_sensitivity_intervals=[r for r in intervals if r['backbone']=='swin_t' and r['state']=='selected' and r['control']=='uniform']),indent=2))


if __name__=='__main__':main()
