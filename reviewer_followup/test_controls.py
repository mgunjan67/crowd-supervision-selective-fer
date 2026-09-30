import torch
from train_controls import targets, loss, base

def test_uniform_preserves_dominant():
    q = torch.tensor([[.5,.2,.1,0,0,0,0,.2,0,0]])
    t,w = targets(q,'uniform')
    assert torch.allclose(w,torch.tensor([.8]))
    assert torch.allclose(t[:,0],torch.tensor([.625]))
    assert torch.allclose(t[:,1:],torch.full((1,6),.0625))
    assert torch.allclose(t.sum(1),torch.ones(1))

def test_ties_are_symmetric():
    q = torch.tensor([[.3,.3,.1,0,0,0,0,.3,0,0]])
    t,w = targets(q,'tie_hard')
    assert torch.equal(t,torch.tensor([[.5,.5,0,0,0,0,0]]))
    assert torch.allclose(w,torch.tensor([.7]))

def test_zero_rows_finite_zero_gradient():
    z = torch.randn(2,7,requires_grad=True)
    q = torch.zeros(2,10); q[:,8]=1
    for condition in ('uniform','tie_hard'):
        value=loss(z,q,condition)
        assert value.item()==0 and torch.isfinite(value)
        gradient=torch.autograd.grad(value,z,retain_graph=True)[0]
        assert torch.equal(gradient,torch.zeros_like(z))

def test_one_hot_all_losses_agree():
    q=torch.zeros(3,10);q[0,0]=1;q[1,3]=1;q[2,6]=1
    z=torch.randn(3,7)
    reference=base.supervised_loss(z,q,'hard')
    for condition in ('uniform','tie_hard'):
        assert torch.allclose(loss(z,q,condition),reference)

def test_unique_max_tie_hard_matches_hard():
    q=torch.tensor([[.2,.5,.1,0,0,0,0,.2,0,0]])
    z=torch.randn(1,7)
    assert torch.allclose(loss(z,q,'tie_hard'),base.supervised_loss(z,q,'hard'))
