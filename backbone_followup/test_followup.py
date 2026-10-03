import sys
from pathlib import Path
import pytest,torch
sys.path.insert(0,str(Path(__file__).resolve().parent))
import model,train
def test_shape_and_gate():
    torch.set_num_threads(4);m=model.build_model().eval()
    assert isinstance(m.backbone.fc,torch.nn.Identity)
    assert sum(p.numel() for p in m.gate.parameters())==32768
    with torch.no_grad():assert m(torch.zeros(2,3,224,224)).shape==(2,7)
@pytest.mark.parametrize('condition',['soft','uniform','tie_uniform'])
def test_zero_mass(condition):
    logits=torch.zeros(2,7,requires_grad=True);q=torch.zeros(2,10);q[:,7]=1
    loss=train.supervised_loss(logits,q,condition);assert loss.item()==0
    loss.backward();assert torch.equal(logits.grad,torch.zeros_like(logits))
def test_ties():
    q=torch.tensor([[.4,.4,.2,0,0,0,0,0,0,0],[.6,.2,.2,0,0,0,0,0,0,0]])
    t,w=train.targets(q,'tie_uniform');u,_=train.targets(q,'uniform')
    assert torch.allclose(t.sum(1),torch.ones(2))
    assert torch.allclose(t[0,:2],torch.tensor([.4,.4]))
    assert torch.allclose(t[1],u[1]);assert torch.allclose(w,torch.ones(2))
