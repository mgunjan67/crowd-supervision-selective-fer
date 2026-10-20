"""Independent calculation invariants for new diagnostics."""
import math
import numpy as np
import analyze as a


def softmax(z):
    x = np.exp(z-z.max(1,keepdims=True))
    return x/x.sum(1,keepdims=True)


def test_centered_normalization_matches_logits_and_offset_invariance():
    z = np.random.default_rng(7).normal(size=(100,7))
    for exponent in a.EXPONENTS:
        centered = z-z.mean(1,keepdims=True)
        expected = centered.max(1)/np.power(np.sum(np.abs(centered)**exponent,axis=1),1/exponent)
        got = a.centered_logit_score(softmax(z),exponent)
        np.testing.assert_allclose(got,expected,atol=1e-13)
        np.testing.assert_allclose(got,a.centered_logit_score(softmax(z+33),exponent),atol=1e-13)
        np.testing.assert_allclose(got,a.centered_logit_score(softmax(z*2),exponent),atol=1e-13)
    np.testing.assert_equal(a.centered_logit_score(np.full((3,7),1/7),2),np.zeros(3))


def test_zero_probability_is_finite_and_clipping_preserves_winner():
    p=np.array([[0,.2,.3,0,0,.5,0]])
    assert np.isfinite(a.centered_logit_score(p,2)).all()
    assert p.argmax(1)[0] == np.maximum(p,np.finfo(np.float32).tiny).argmax(1)[0]


def test_coverage_rounding_ties_and_full_score_invariance():
    rows=np.array([8,4,9,1,3,2,5])
    loss=np.arange(7)/7
    zero=np.zeros(7)
    expected=np.argsort(rows)
    np.testing.assert_array_equal(a.ordered(zero,rows),expected)
    risk,count=a.risk_at(zero,loss,rows,.8)
    assert count==math.ceil(.8*7)==6
    assert risk==loss[expected[:6]].mean()
    assert a.risk_at(zero,loss,rows,1)[0] == a.risk_at(rows,loss,rows,1)[0]


def test_chronological_middle_aggregation_not_input_order():
    rng=np.random.default_rng(3)
    p=softmax(rng.normal(size=(16,7)))
    ids=np.repeat([0,1],8)
    times=np.tile(np.arange(8)*10,2)
    order=rng.permutation(16)
    full=a.aggregate_frames(p[order],ids[order],times[order],False)
    middle=a.aggregate_frames(p[order],ids[order],times[order],True)
    np.testing.assert_allclose(full,p.reshape(2,8,7).mean(1),atol=1e-14)
    np.testing.assert_allclose(middle,p.reshape(2,8,7)[:,2:6].mean(1),atol=1e-14)


def test_target_controls_preserve_required_properties():
    q=np.zeros((4,10));q[0,:3]=[.6,.3,.1];q[1,:4]=[.3,.3,.2,.1];q[1,7]=.1
    q[2,8]=1;q[3,:7]=1/7
    for condition in a.FAMILIES['swin_t']:
        t,w=a.targets(q,condition)
        np.testing.assert_allclose(t[w>0].sum(1),1)
        assert np.all(t[w==0]==0)
    r,_=a.targets(q,'soft');u,_=a.targets(q,'uniform');tu,_=a.targets(q,'tie_uniform')
    np.testing.assert_allclose(r.max(1),u.max(1))
    np.testing.assert_allclose(r.max(1),tu.max(1))
    np.testing.assert_array_equal(r==r.max(1,keepdims=True),tu==tu.max(1,keepdims=True))


def test_external_components_and_constant_identity():
    common=a.external_module()
    q=np.array([[.1,.1,.1,.1,.1,.1,.1,.2,.1]])
    p=np.eye(7)[[4]]
    parts=common.components(p,q,np.ones(1,bool))
    assert np.isclose(parts[0,0],1-q[0,4])
    assert np.isclose(parts[0,0],parts[0,1:].sum())
