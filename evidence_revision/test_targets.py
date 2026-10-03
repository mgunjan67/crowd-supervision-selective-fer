import torch
from train_tie_uniform import targets,loss

def test_unique_maximum_matches_original_uniform():
    q=torch.tensor([[.6,.2,.1,.1,0,0,0,0,0,0]])
    t,w=targets(q)
    assert torch.allclose(t,torch.tensor([[.6,*([.4/6]*6)]])) and w.item()==1

def test_tied_maxima_preserved():
    t,_=targets(torch.tensor([[.4,.4,.1,.1,0,0,0,0,0,0]]))
    assert torch.allclose(t,torch.tensor([[.4,.4,*([.2/5]*5)]]))

def test_all_tied_and_zero_supported():
    q=torch.tensor([[1/7]*7+[0]*3,[0]*7+[1,0,0]])
    t,w=targets(q)
    assert torch.allclose(t.sum(1),torch.ones(2)) and torch.allclose(t[0],torch.full((7,),1/7))
    logits=torch.randn(1,7,requires_grad=True);value=loss(logits,q[1:],'tie_uniform');value.backward()
    assert value.item()==0 and torch.count_nonzero(logits.grad)==0

def test_partial_mass_and_permutation_symmetry():
    q=torch.tensor([[.3,.3,.1,.1,0,0,0,.2,0,0]]);t,w=targets(q)
    assert torch.allclose(w,torch.tensor([.8]))
    perm=torch.tensor([6,1,4,3,0,5,2]);changed=torch.cat([q[:,:7][:,perm],q[:,7:]],1)
    shifted,_=targets(changed)
    assert torch.allclose(shifted,t[:,perm]) and torch.allclose(t.sum(1),torch.ones(1))
