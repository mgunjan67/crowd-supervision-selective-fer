"""Synthetic-data checks: these never inspect new experimental predictions."""
import importlib.util
from pathlib import Path
import numpy as np

spec=importlib.util.spec_from_file_location('bounded_analysis',Path(__file__).with_name('analysis.py'))
analysis=importlib.util.module_from_spec(spec);spec.loader.exec_module(analysis)

def sample():
    p=np.full((5,7),.05);p[:,0]=.7
    q=np.zeros((5,10));q[:,0]=[1,.6,.4,.2,0];q[:,1]=[0,.2,.4,.2,0]
    q[:,9]=1-q.sum(1)
    return dict(probabilities=p,q=q,labels=np.array([0,1,0,1,2]),
                rows=np.array([4,2,1,3,0]),hashes=np.array(['a','b','c','d','e']))

def test_tied_confidence_uses_original_row_order():
    a=sample()
    assert analysis.rank(a).tolist()==[4,2,1,3,0]
    assert np.flatnonzero(analysis.common_keep(a,.8)).tolist()==[1,2,3,4]

def test_common_coverage_uses_ceiling_and_keeps_every_row_at_one():
    a=sample()
    assert analysis.common_keep(a,.5).sum()==3
    assert analysis.common_keep(a,1).all()

def test_threshold_never_splits_score_ties():
    score=np.array([.9,.8,.8,.7]);loss=np.array([0.,0.,1.,0.])
    assert analysis.threshold(score,loss,.2,minimum=1)==.9
    assert analysis.threshold(score,loss,.25,minimum=1)==.7
    assert analysis.threshold(score,loss,.2,minimum=2) is None

def test_all_disagreement_includes_out_of_scope_votes():
    a=sample();v=analysis.per_image(a)
    np.testing.assert_allclose(v['vote_risk'],[0,.4,.6,.8,1])
    np.testing.assert_allclose(v['oracle_floor']+v['excess_risk'],v['vote_risk'])
    assert np.all(v['oracle_floor']>=v['out_of_scope_mass']-1e-12)
    assert np.all(v['excess_risk']>=0)

def test_joint_error_decomposition_uses_only_majority_subset():
    a=sample();a['probabilities'][0]=np.array([.05,.7,.05,.05,.05,.05,.05])
    report=analysis.metric_summary(a,np.ones(5,bool))
    assert report['accepted_majority_n']==2
    assert report['joint_error']==1
    assert report['reference_disagreement']==.5
    assert report['shared_reference_error_contribution']==.5

def test_empty_selection_has_missing_risk_not_zero_error():
    result=analysis.metric_summary(sample(),np.zeros(5,bool))
    assert result['coverage']==0 and result['vote_risk'] is None
    assert result['joint_error'] is None

def test_cluster_bootstrap_identical_models_have_exact_zero_contrast():
    a=sample();a['hashes'][1]='a'
    arrays={(condition,seed):a for condition in analysis.CONDITIONS for seed in analysis.SEEDS}
    result=analysis.bootstrap_primary(arrays,repetitions=40)
    assert result['pixel_clusters']==4 and result['accepted_n']==4
    assert result['point_difference_pp']==result['lower_pp']==result['upper_pp']==0

def test_cluster_bootstrap_direction_for_unambiguously_better_model():
    a=sample();a['q'][:]=0;a['q'][:,0]=1
    b={k:v.copy() for k,v in a.items()};b['probabilities'][:,[0,1]]=b['probabilities'][:,[1,0]]
    arrays={(condition,seed):(a if condition=='soft' else b)
            for condition in analysis.CONDITIONS for seed in analysis.SEEDS}
    result=analysis.bootstrap_primary(arrays,repetitions=40)
    assert result['point_difference_pp']==result['lower_pp']==result['upper_pp']==-100

def test_selection_distance_and_expected_brier_differ_only_by_vote_constant():
    rng=np.random.default_rng(73);q=rng.dirichlet(np.ones(10),size=20)
    p=np.column_stack((rng.dirichlet(np.ones(7),size=20),np.zeros((20,3))))
    expected=np.sum(q[:,:,None]*np.square(p[:,None,:]-np.eye(10)[None,:,:]),axis=(1,2))
    distance=np.square(p-q).sum(1)
    np.testing.assert_allclose(expected,distance+1-np.square(q).sum(1),rtol=0,atol=1e-14)
