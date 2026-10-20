"""Portable CPU-only independent arithmetic checks on retained probabilities.

No torch, images, videos, credentials or weights are needed. This is internal
verification, not external replication. The separately recorded full workspace
audits additionally rehash media/weights and reproduce the bootstrap intervals.
"""
import csv
import hashlib
import json
import math
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
PAPER=HERE.parent
EXT=PAPER/'ravdess_transfer_2026_10_11'
SEEDS=(17,42,89)
LABELS=('angry','disgust','fear','happy','neutral','sad','surprise','calm','none')

def read(path):
    with path.open(newline='',encoding='utf-8') as stream:return list(csv.DictReader(stream))
def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(4194304),b''):h.update(block)
    return h.hexdigest()
def same(actual,expected,tol=1e-10):assert abs(float(actual)-float(expected))<=tol,(actual,expected)
def select(p,ids,coverage=.8):return np.lexsort((ids,-p.max(1)))[:math.ceil(coverage*len(p))]

def main():
    manifest=json.loads((PAPER/'reviewer_followup/split_manifest.json').read_text())
    source=read(PAPER/'evidence/dataset_manifest.csv');votes=read(PAPER/'data/ferplus_annotations/fer2013new.csv')
    vote_names=('anger','disgust','fear','happiness','neutral','sadness','surprise','contempt','unknown','NF')
    q=np.array([[int(r[c]) for c in vote_names] for r in votes],float);q/=q.sum(1,keepdims=True)
    hashes=np.array([r['pixel_sha256'] for r in source]);classes=LABELS[:7]
    labels=np.array([classes.index(r['class']) for r in source])
    sets=[set(hashes[manifest['roles'][r]]) for r in ('train','selection','calibration','test')]
    assert all(not a&b for i,a in enumerate(sets) for b in sets[i+1:])
    predictions={};files=0;calibration={}
    folders={'reviewer_followup':'swin_t','evidence_revision':'swin_t','backbone_followup':'resnet18'}
    for folder,family in folders.items():
        for metadata in sorted((PAPER/folder/'predictions').glob('*/evaluation.json')):
            info=json.loads(metadata.read_text());condition,seed,state=info['condition'],info['seed'],info['state']
            for role,rec in info['roles'].items():
                path=metadata.parent/(role+'.npz');assert sha(path)==rec['sha256']
                with np.load(path,allow_pickle=False) as z:a={k:z[k] for k in z.files}
                ids=a['rows'];assert ids.tolist()==manifest['roles'][role]
                for key,expected in [('q',q[ids]),('labels',labels[ids]),('hashes',hashes[ids])]:assert np.array_equal(a[key],expected),key
                p=a['probabilities'];assert p.shape==(len(ids),7) and np.isfinite(p).all() and (p>=0).all()
                np.testing.assert_allclose(p.sum(1),1,atol=2e-6,rtol=0)
                if role=='test_full':
                    m=np.isin(ids,manifest['roles']['test']);predictions[family,condition,seed,state]={k:v[m] for k,v in a.items()}
                if role=='calibration' and folder=='reviewer_followup':calibration[condition,seed,state]=a
                files+=1
    assert len(predictions)==48 and files==144,(len(predictions),files)
    # Independently derive primary risks and identical-image contrasts.
    contrasts={};fixed={}
    for family in ('swin_t','resnet18'):
        pairs=[]
        for seed in SEEDS:
            a=predictions[family,'soft',seed,'selected'];b=predictions[family,'uniform',seed,'selected']
            ia=select(a['probabilities'],a['rows']);ib=select(b['probabilities'],b['rows']);assert len(ia)==len(ib)==2620
            da=1-a['q'][np.arange(len(a['rows'])),a['probabilities'].argmax(1)]
            db=1-b['q'][np.arange(len(b['rows'])),b['probabilities'].argmax(1)]
            pairs.append(100*(da[ia].mean()-db[ib].mean()))
            for name,ii in [('soft',ia),('uniform',ib)]:fixed.setdefault((family,name),[]).append(100*(da[ii].mean()-db[ii].mean()))
        contrasts[family]=pairs
    ledger=read(PAPER/'recovery_2026_10_10/selector_summary.csv')
    for family,pairs in contrasts.items():
        r=next(r for r in ledger if r['backbone']==family and r['state']=='selected' and r['score']=='msp' and r['control']=='uniform' and float(r['requested_coverage'])==.8)
        same(np.mean(pairs),r['mean_difference_pp'])
    # Recreate empirical and fixed-grid policies from calibration only, no test fitting.
    overshoots={'empirical':0,'conservative':0};nonempty={'empirical':0,'conservative':0}
    for (condition,seed,state),a in calibration.items():
        if state!='selected':continue
        p=a['probabilities'];scores=p.max(1);loss=1-a['q'][np.arange(len(p)),p.argmax(1)]
        test=predictions['swin_t',condition,seed,state];ts=test['probabilities'].max(1)
        td=1-test['q'][np.arange(len(ts)),test['probabilities'].argmax(1)]
        for policy in nonempty:
            grid=np.unique(scores) if policy=='empirical' else np.arange(101)/100
            for target in (.1,.2,.3):
                threshold=None
                for t in grid:
                    keep=scores>=t;n=int(keep.sum())
                    if n<100:continue
                    bound=loss[keep].mean()
                    if policy=='conservative':bound=min(1,bound+np.sqrt(np.log(2020)/(2*n)))
                    if bound<=target+1e-12:threshold=t;break
                if threshold is None:continue
                keep=ts>=threshold
                if not keep.any():continue
                nonempty[policy]+=1;overshoots[policy]+=int(td[keep].mean()>target)
    assert nonempty=={'empirical':36,'conservative':24}
    assert overshoots=={'empirical':36,'conservative':0}
    # New exports checked with per-category masked sums (not matrix multiplication).
    dr=HERE/'diagnostics'
    cc=read(dr/'predicted_categories.csv');mm=read(dr/'vote_weighted_matrices.csv');tt=read(dr/'temporal_bins.csv');rr=read(dr/'risk_accounting.csv')
    count_index={(r['backbone'],r['condition'],int(r['seed']),r['state'],r['population'],r['predicted_category']):r for r in cc}
    matrix_index={(r['backbone'],r['condition'],int(r['seed']),r['state'],r['population'],r['reference'],r['reference_category'],r['predicted_category']):r for r in mm}
    temporal_index={(r['backbone'],r['condition'],int(r['seed']),r['state'],int(r['bin']),r['predicted_category']):r for r in tt}
    risk_index={(r['backbone'],r['condition'],int(r['seed']),r['state'],r['population'],r['reference']):r for r in rr}
    v=read(EXT/'video_only_speech_votes.csv');names=np.array([r['filename'] for r in v]);raw=np.array([[int(r['count_'+c])/10 for c in LABELS] for r in v])
    merged=raw.copy();merged[:,4]+=merged[:,7];merged[:,7]=0
    bins=np.array([f['bin_id']+1 for actor in range(1,25) for f in json.loads((EXT/f'cache/Actor_{actor:02d}.json').read_text())['frames'] if f['status']=='valid'])
    states=0;external_pairs={};max_error=0.0
    for path in sorted((EXT/'predictions').glob('*.npz')):
        meta=json.loads(path.with_suffix('.json').read_text());assert sha(path)==meta['output_sha256']
        key=(meta['backbone'],meta['condition'],int(meta['seed']),meta['state'])
        with np.load(path,allow_pickle=False) as z:p=z['probabilities'];fp=z['frame_probabilities'];frame_ids=z['frame_clip_ids'];assert np.array_equal(z['filenames'],names)
        predicted=p.argmax(1);frame_pred=fp.argmax(1)
        # Deliberately express aggregation as a clip loop, independent of add.at.
        for i in range(1440):
            mask=frame_ids==i;assert mask.sum()==8
            error=float(abs(fp[mask].astype(float).mean(0)-p[i]).max());max_error=max(max_error,error);assert error<=1e-14
        accepted=select(p,names)
        for pop,ids in [('all_eligible',np.arange(1440)),('msp80',accepted)]:
            for b,label in enumerate(LABELS[:7]):
                r=count_index[(*key,pop,label)];n=int(np.sum(predicted[ids]==b));assert n==int(r['count']);same(n/len(ids),r['proportion'])
            for ref,qref in [('raw9',raw),('neutral_calm_merged',merged)]:
                chosen=qref[np.arange(1440),predicted];w=qref[:,:7].sum(1);mx=qref[:,:7].max(1)
                vals={'risk':1-chosen,'unsupported':1-w,'supported_ambiguity':w-mx,'excess':mx-chosen}
                for k,x in vals.items():same(x[ids].mean(),risk_index[(*key,pop,ref)][k])
                if key[0]=='swin_t' and key[3]=='selected' and pop=='msp80':external_pairs[key[1],key[2],ref]=vals['risk'][ids].mean()
                for a,al in enumerate(LABELS):
                    mass=qref[ids,a].sum()
                    for b,bl in enumerate(LABELS[:7]):
                        val=qref[ids,a][predicted[ids]==b].sum();r=matrix_index[(*key,pop,ref,al,bl)]
                        same(val,r['contribution']);same(mass,r['row_vote_mass'])
                        if mass: same(val/mass,r['row_normalized'])
                        else:assert r['row_normalized']=='' and r['row_status']=='undefined_zero_mass'
        for bin_id in range(1,9):
            mask=bins==bin_id;assert mask.sum()==1440
            for b,label in enumerate(LABELS[:7]):
                r=temporal_index[(*key,bin_id,label)]
                assert int(r['count'])==int((frame_pred[mask]==b).sum())
                same(fp[mask,4].astype(float).mean(),r['mean_neutral_probability'])
                same((frame_pred[mask]==predicted[frame_ids[mask]]).mean(),r['frame_clip_argmax_agreement'])
        states+=1
    assert states==36
    principal=read(dr/'principal_seed_contrasts.csv')
    for r in principal:
        s=int(r['seed']);ref=r['reference'];same(100*(external_pairs['soft',s,ref]-external_pairs['uniform',s,ref]),r['difference_pp'])
    report={'status':'passed','independent_arithmetic':True,'internal_not_external_replication':True,'within_source_states':len(predictions),'within_source_prediction_files':files,'source_row_joins':35887,'external_states':states,'vote_matrix_cells_checked':len(mm),'temporal_rows_checked':len(tt),'category_rows_checked':len(cc),'maximum_aggregation_error':max_error,'numeric_tolerance':1e-10,'clip_aggregation_tolerance':1e-14,'primary_internal_seed_differences_pp':contrasts,'fixed_image_mean_differences_pp':{'|'.join(k):float(np.mean(v)) for k,v in fixed.items()},'calibration_nonempty':nonempty,'calibration_overshoot':overshoots,'new_training_runs':0}
    (HERE/'checks').mkdir(exist_ok=True)
    (HERE/'checks/portable_numerical_audit.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))

if __name__=='__main__':main()
