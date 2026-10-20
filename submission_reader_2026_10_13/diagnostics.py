"""Post hoc descriptive diagnostics. No fitting, filtering, new ranking or ensemble."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
PAPER=HERE.parent
EXTERNAL=PAPER/'ravdess_transfer_2026_10_11'
LABELS=('angry','disgust','fear','happy','neutral','sad','surprise','calm','none')

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(4194304),b''):h.update(block)
    return h.hexdigest()

def save(path,rows):
    with path.open('w',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output-dir',type=Path,default=HERE/'diagnostics')
    args=parser.parse_args();out=args.output_dir;out.mkdir(parents=True,exist_ok=True)
    with (EXTERNAL/'video_only_speech_votes.csv').open(newline='') as stream:
        records=sorted(csv.DictReader(stream),key=lambda r:r['filename'])
    names=np.array([r['filename'] for r in records]);actors=np.array([int(r['actor']) for r in records])
    assert len(records)==len(set(names))==1440
    counts=np.array([[int(r['count_'+c]) for c in LABELS] for r in records])
    assert np.all(counts.sum(1)==10)
    q=counts/10
    assert all(names[i].startswith('02-01-') and int(names[i][-6:-4])==actors[i] for i in range(1440))
    assert all(np.sum(actors==a)==60 for a in range(1,25))
    refs={'raw9':q,'neutral_calm_merged':q.copy()}
    refs['neutral_calm_merged'][:,4]+=q[:,7];refs['neutral_calm_merged'][:,7]=0
    frames=[];clips={}
    for a in range(1,25):
        report=json.loads((EXTERNAL/f'cache/Actor_{a:02d}.json').read_text())
        for c in report['clips']:
            assert c['actor']==a and c['filename']==names[c['clip_id']]
            clips[c['clip_id']]=c
        for f in report['frames']:
            n=clips[f['clip_id']]['frame_count']
            assert f['frame_index']==min(n-1,math.floor((f['bin_id']+.5)*n/8))
            if f['status']=='valid':
                assert f['detections']==1 and f['decoded_frame_index']==f['frame_index']
                l,t,r,b=f['box'];assert 0<=l<r<=f['width'] and 0<=t<b<=f['height']
                frames.append(f)
    assert len(clips)==1440 and len(frames)==11520
    frame_ids=np.array([f['clip_id'] for f in frames]);frame_indices=np.array([f['frame_index'] for f in frames])
    bins=np.array([f['bin_id'] for f in frames])
    valid_counts=np.bincount(frame_ids,minlength=1440);valid=valid_counts>=4
    assert np.all(valid_counts==8)
    freeze=json.loads((EXTERNAL/'freeze.json').read_text());assert len(freeze['checkpoints'])==36
    predicted=[];reference_mass=[];matrices=[];temporal=[];accounting=[];accepted_ids=[]
    models={};inputs={};aggregate_errors=[]
    for item in freeze['checkpoints']:
        family,condition,seed,state=(item[k] for k in ('backbone','condition','seed','state'))
        key=(family,condition,seed,state)
        stem=f'{family}_{condition}_seed{seed}_{state}'
        path=EXTERNAL/'predictions'/(stem+'.npz');meta=json.loads(path.with_suffix('.json').read_text())
        assert meta['sha256']==item['sha256'] and meta['state']==state
        assert sha(path)==meta['output_sha256'];inputs[str(path.relative_to(PAPER))]=sha(path)
        with np.load(path,allow_pickle=False) as z:
            p=z['probabilities'];fp=z['frame_probabilities']
            assert np.array_equal(z['filenames'],names) and np.array_equal(z['actors'],actors)
            assert np.array_equal(z['valid'],valid) and np.array_equal(z['valid_frame_count'],valid_counts)
            assert np.array_equal(z['frame_clip_ids'],frame_ids) and np.array_equal(z['frame_indices'],frame_indices)
        assert p.shape==(1440,7) and fp.shape==(11520,7)
        assert np.isfinite(p[valid]).all() and np.isfinite(fp).all() and (fp>=0).all() and (p[valid]>=0).all()
        assert np.allclose(fp.sum(1),1,atol=1e-6,rtol=0) and np.allclose(p[valid].sum(1),1,atol=1e-6,rtol=0)
        sums=np.zeros((1440,7));np.add.at(sums,frame_ids,fp.astype(float))
        recovered=sums[valid]/valid_counts[valid,None]
        err=float(np.max(np.abs(recovered-p[valid])));aggregate_errors.append(err);assert err<=1e-14
        pred=p.argmax(1);frame_pred=fp.argmax(1)
        ranking=sorted(np.flatnonzero(valid),key=lambda i:(-float(p[i].max()),str(names[i])))
        sets={'all_eligible':np.flatnonzero(valid),'msp80':np.array(ranking[:math.ceil(.8*1440)])}
        models[key]={}
        identity=dict(backbone=family,condition=condition,seed=seed,state=state)
        for population,keep in sets.items():
            n=len(keep);assert n==(1440 if population=='all_eligible' else 1152)
            idhash=hashlib.sha256('\n'.join(names[np.sort(keep)]).encode()).hexdigest()
            accepted_ids.append(dict(**identity,population=population,accepted_n=n,ids_sha256=idhash))
            for b in range(7):
                number=int(np.sum(pred[keep]==b))
                predicted.append(dict(**identity,population=population,population_n=1440,accepted_n=n,
                                      predicted_category=LABELS[b],count=number,proportion=number/n))
            for ref,qq in refs.items():
                support=qq[:,:7].sum(1);mx=qq[:,:7].max(1);chosen=qq[np.arange(1440),pred]
                values=np.column_stack((1-chosen,1-support,support-mx,mx-chosen))
                assert np.max(np.abs(values[:,0]-values[:,1:].sum(1)))<1e-14
                means=values[keep].mean(0)
                accounting.append(dict(**identity,population=population,reference=ref,accepted_n=n,
                                       **dict(zip(('risk','unsupported','supported_ambiguity','excess'),map(float,means)))))
                models[key][population,ref]=float(means[0])
                # Independent indicator-matrix construction, not a single-label confusion.
                indicators=np.eye(7,dtype=float)[pred[keep]]
                matrix=qq[keep].T@indicators;mass=qq[keep].sum(0)
                assert np.allclose(matrix.sum(1),mass,atol=1e-10,rtol=0)
                assert abs(matrix.sum()-n)<1e-10
                for a in range(9):
                    reference_mass.append(dict(**identity,population=population,reference=ref,accepted_n=n,
                                               reference_category=LABELS[a],vote_mass=float(mass[a]),proportion=float(mass[a]/n)))
                    for b in range(7):
                        matrices.append(dict(**identity,population=population,reference=ref,accepted_n=n,
                                             reference_category=LABELS[a],predicted_category=LABELS[b],
                                             contribution=float(matrix[a,b]),row_vote_mass=float(mass[a]),
                                             row_normalized=float(matrix[a,b]/mass[a]) if mass[a]>0 else '',
                                             row_status='defined' if mass[a]>0 else 'undefined_zero_mass'))
        for bin_id in range(8):
            mask=bins==bin_id;n=int(mask.sum());assert n==1440
            for b in range(7):
                number=int(np.sum(frame_pred[mask]==b))
                temporal.append(dict(**identity,bin=bin_id+1,valid_frames=n,clips_with_valid_bin=int(len(set(frame_ids[mask]))),
                                     predicted_category=LABELS[b],count=number,proportion=number/n,
                                     mean_neutral_probability=float(fp[mask,4].mean(dtype=float)),
                                     frame_clip_argmax_agreement=float(np.mean(frame_pred[mask]==pred[frame_ids[mask]]))))
    principal=[]
    for ref in refs:
        for seed in (17,42,89):
            soft=models['swin_t','soft',seed,'selected']['msp80',ref]
            uniform=models['swin_t','uniform',seed,'selected']['msp80',ref]
            principal.append(dict(reference=ref,seed=seed,soft_risk=soft,uniform_risk=uniform,difference_pp=100*(soft-uniform)))
    outputs={'predicted_categories.csv':predicted,'reference_vote_mass.csv':reference_mass,
             'vote_weighted_matrices.csv':matrices,'temporal_bins.csv':temporal,
             'risk_accounting.csv':accounting,'accepted_id_hashes.csv':accepted_ids,'principal_seed_contrasts.csv':principal}
    for filename,rows in outputs.items():save(out/filename,rows)
    summary=dict(status='passed',post_hoc=True,training_runs_added=0,states=36,clips=1440,actors=24,frames=11520,
                 bins=8,probability_tolerance=1e-6,aggregation_tolerance=1e-14,maximum_aggregation_error=max(aggregate_errors),
                 input_prediction_hashes=inputs,output_hashes={name:sha(out/name) for name in outputs},
                 rows={name:len(rows) for name,rows in outputs.items()},principal_seed_contrasts=principal,
                 note='Descriptive clip-level vote matrices and frame summaries; no frame-level human accuracy or causal diagnosis.')
    (out/'diagnostic_audit.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps({k:v for k,v in summary.items() if k not in ('input_prediction_hashes','output_hashes')},indent=2))

if __name__=='__main__':main()
