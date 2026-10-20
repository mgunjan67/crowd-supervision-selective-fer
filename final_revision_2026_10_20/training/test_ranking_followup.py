"""CPU invariants; optional fixtures are synthetic and confined to checks/."""
import argparse
import json
from pathlib import Path
import numpy as np
import pytest
import ranking_followup as extension


def probabilities(z):
    e=np.exp(z-z.max(1,keepdims=True));return e/e.sum(1,keepdims=True)


def synthetic_models():
    rng=np.random.default_rng(904)
    annotation={}
    for role,n in [('calibration',21),('test',31)]:
        votes=rng.multinomial(10,np.full(10,.1),size=n)
        annotation[role]=dict(q=votes/10,rows=rng.permutation(np.arange(n)),hashes=np.array([f'{i:064x}' for i in range(n)]))
    result={}
    for key in sum(extension.planned_keys(),[]):
        k=key[0];model={}
        for role,a in annotation.items():
            p=probabilities(rng.normal(size=(len(a['rows']),k))).astype(np.float32).astype(float)
            model[role]={**a,'probabilities':p}
        result[key]=model
    return result


def test_centering_and_ties_work_for_both_vocabularies():
    rng=np.random.default_rng(10)
    for k in (7,8):
        z=rng.normal(size=(12,k));centered=z-z.mean(1,keepdims=True)
        for exponent in extension.EXPONENTS:
            expected=centered.max(1)/np.power((np.abs(centered)**exponent).sum(1),1/exponent)
            np.testing.assert_allclose(extension.normalized_score(probabilities(z),exponent),expected,atol=1e-13)
            np.testing.assert_allclose(extension.normalized_score(probabilities(2*z+11),exponent),expected,atol=1e-13)
        np.testing.assert_equal(extension.normalized_score(np.full((2,k),1/k),2),np.zeros(2))
    np.testing.assert_equal(extension.rank(np.ones(3),np.array([9,1,5])),[1,2,0])


def test_all_39_states_and_candidate_outcomes_are_retained():
    data=extension.analyze_models(synthetic_models())
    assert {key:len(data[key]) for key in data}==dict(results=780,tuning=312,candidates=1248,paired=480,means=160,inputs=78)
    for key in sum(extension.planned_keys(),[]):
        k,c,s,state=key
        full=[r['risk'] for r in data['results'] if (r['k'],r['condition'],r['seed'],r['state'])==key and r['requested_coverage']==1]
        assert len(full)==5 and np.ptp(full)<1e-13
        candidates=[r['risk'] for r in data['candidates'] if (r['k'],r['condition'],r['seed'],r['state'])==key and r['requested_coverage']==1]
        assert len(candidates)==8 and np.ptp(candidates)<1e-13
    assert all(row['accepted_n']==25 for row in data['results'] if row['requested_coverage']==.8)


def test_test_outcomes_cannot_choose_exponents_or_fallback():
    models=synthetic_models();first=extension.analyze_models(models)
    for model in models.values():model['test']['q']=np.roll(model['test']['q'],1,axis=1)
    second=extension.analyze_models(models)
    fields=('k','condition','seed','state','exponent','selected','calibration_aurc','fallback_score')
    assert [tuple(r[f] for f in fields) for r in first['tuning']]==[tuple(r[f] for f in fields) for r in second['tuning']]
    assert any(a['test_aurc']!=b['test_aurc'] for a,b in zip(first['tuning'],second['tuning']))


def test_all_eight_paired_groups_have_correct_percentage_point_scale():
    models=synthetic_models();data=extension.analyze_models(models)
    assert len(extension.contrast_groups())==8
    for k,state,control in extension.contrast_groups():
        values=[]
        for seed in extension.SEEDS:
            risks=[]
            for condition in ('soft',control):
                a=models[k,condition,seed,state]['test']
                risks.append(extension.selected_risk(a['probabilities'].max(1),extension.loss(a),a['rows'],.8)[0])
            expected=100*(risks[0]-risks[1]);values.append(expected)
            row=next(r for r in data['paired'] if (r['k'],r['state'],r['control'],r['seed'],r['score'],r['requested_coverage'])==(k,state,control,seed,'msp',.8))
            assert abs(row['difference_pp']-expected)<1e-13
        mean=next(r for r in data['means'] if (r['k'],r['state'],r['control'],r['score'],r['requested_coverage'])==(k,state,control,'msp',.8))
        assert abs(mean['mean_difference_pp']-np.mean(values))<1e-13


def test_missing_state_or_wrong_vocabulary_is_rejected():
    models=synthetic_models();models.pop(next(iter(models)))
    with pytest.raises(AssertionError):extension.analyze_models(models)
    models=synthetic_models();key=next(k for k in models if k[0]==8)
    models[key]['test']['probabilities']=models[key]['test']['probabilities'][:,:7]
    with pytest.raises(AssertionError):extension.analyze_models(models)


def make_fixture(directory):
    checks=Path(__file__).resolve().parent.parent/'checks'
    assert directory.resolve().is_relative_to(checks.resolve()),'Synthetic fixtures must stay below checks/'
    directory.mkdir(parents=True,exist_ok=True)
    records=extension.analyze_models(synthetic_models())
    for name,rows in records.items():extension.write_csv(directory/f'synthetic_new_ranking_{name}.csv',rows)
    original=extension.RESULTS
    try:
        extension.RESULTS=directory;table=extension.export_table(records['means'])
        table.write_text(r'\textbf{SYNTHETIC LAYOUT FIXTURE -- NOT EMPIRICAL.}'+'\n'+table.read_text(),encoding='utf-8')
    finally:extension.RESULTS=original
    (directory/'SYNTHETIC_FIXTURE.json').write_text(json.dumps(dict(synthetic=True,empirical_results=False,
        calibration_images=21,test_images=31,new_states=33,historical_comparator_states=6),indent=2),encoding='utf-8')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--synthetic-fixture',type=Path,required=True)
    args=parser.parse_args();make_fixture(args.synthetic_fixture)
