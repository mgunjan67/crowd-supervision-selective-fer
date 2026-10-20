"""Tests guard the distinct new scientific interventions."""
import numpy as np
import train_followup as experiment

def sample():
    q = np.array([[.5,.2,.1,.1,.05,.03,.02,0,0,0],
                  [.3,.3,.15,.1,.05,.05,.05,0,0,0],
                  [0,0,0,0,0,0,0,1,0,0],
                  [0,0,0,0,0,0,0,0,.5,.5]], dtype=float)
    return q, np.array([1,2,3,4])

def test_permutation_preserves_entropy_all_maxima_and_multiset():
    q, rows = sample()
    t, w = experiment.targets(q, rows, 7, 'permutation')
    r = np.divide(q[:,:7], w[:,None], out=np.zeros_like(t), where=w[:,None]>0)
    assert np.array_equal(np.sort(t, axis=1), np.sort(r, axis=1))
    mask = r == r.max(1, keepdims=True)
    assert np.array_equal(t[mask], r[mask])
    assert not np.array_equal(t[0], r[0])
    assert np.array_equal(t, experiment.targets(q, rows, 7, 'permutation')[0])

def test_eighth_expression_is_supervised_but_unknown_is_not():
    q, rows = sample()
    t7, w7 = experiment.targets(q, rows, 7, 'soft')
    t8, w8 = experiment.targets(q, rows, 8, 'soft')
    assert w7[2] == 0 and w8[2] == 1 and t8[2,7] == 1
    assert w8[3] == 0 and not t8[3].any()

def test_tie_uniform_generalizes_and_preserves_tied_maxima():
    q, rows = sample()
    for k in (7,8):
        t, w = experiment.targets(q, rows, k, 'tie_uniform')
        assert np.allclose(t[w>0].sum(1),1)
        assert t[1,0] == t[1,1] == .3
        assert np.all(t[w==0] == 0)
        u, _ = experiment.targets(q, rows, k, 'uniform')
        assert np.array_equal(t[0],u[0])
