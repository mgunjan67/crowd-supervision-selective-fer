import numpy as np
from analysis import parts, conservative, summary

def example(q,p):
    return {'q':np.array(q,float),'probabilities':np.array(p,float)}

def test_vocabulary_identity_and_same_msp():
    p=[[.9,.1,0,0,0,0,0]]*2
    q=[[.9,.1,0,0,0,0,0,0,0,0],[.45,.05,0,0,0,0,0,.5,0,0]]
    v=parts(example(q,p))
    np.testing.assert_allclose(v['risk'],[.1,.55])
    np.testing.assert_allclose(v['supported_only_risk'],[.1,.1])
    np.testing.assert_allclose(v['risk'],v['unsupported']+v['supported_ambiguity']+v['excess'])

def test_zero_mass_is_not_zero_risk():
    a=example([[0,0,0,0,0,0,0,0,1,0]],[[1,0,0,0,0,0,0]])
    s=summary(a,np.array([True]))
    assert s['risk']==1 and s['unsupported']==1 and s['supported_only_risk'] is None
    assert s['positive_mass_accepted_n']==0

def test_empty_selection_risk_undefined():
    a=example([[1,0,0,0,0,0,0,0,0,0]],[[1,0,0,0,0,0,0]])
    s=summary(a,np.array([False]))
    assert s['coverage']==0 and s['risk'] is None

def test_grid_bound_can_abstain():
    assert conservative(np.ones(99),np.zeros(99),.3)==(None,None)
    assert conservative(np.ones(100),np.zeros(100),.1)==(None,None)

def test_grid_uses_simultaneous_margin():
    t,u=conservative(np.full(1000,.8),np.zeros(1000),.1)
    assert t==0
    np.testing.assert_allclose(u,np.sqrt(np.log(101/.05)/2000))

def test_quadratic_constrained_optimum():
    q=np.array([.3,.1,.05,.05,0,0,0,.2,.2,.1])
    p=q[:7]+(1-q[:7].sum())/7
    assert np.all(p>=0) and np.isclose(p.sum(),1)
    gradient=2*(p-q[:7]);np.testing.assert_allclose(gradient,np.full(7,gradient[0]))
    r=q[:7]/q[:7].sum()
    assert np.square(p-q[:7]).sum()<np.square(r-q[:7]).sum()
