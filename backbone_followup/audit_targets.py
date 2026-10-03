"""Full-source target check against a separate NumPy construction."""
import json
from pathlib import Path
import numpy as np,torch
import train
OUT=Path(__file__).resolve().parent
def main():
    q=train.data.load_role('train')['q'];s=q[:,:7];w=s.sum(1);r=np.divide(s,w[:,None],out=np.zeros_like(s),where=w[:,None]>0)
    maximum=r.max(1);ties=s==s.max(1,keepdims=True);counts=ties.sum(1);other=np.divide(1-counts*maximum,7-counts,out=np.zeros(len(q)),where=counts<7)
    ref=np.where(ties,maximum[:,None],other[:,None]);ref[counts==7]=1/7
    t,ww=train.targets(torch.tensor(q,dtype=torch.float32),'tie_uniform')
    assert np.allclose(t.numpy()[w>0],ref[w>0],atol=2e-7) and np.allclose(ww.numpy(),w)
    soft,sw=train.targets(torch.tensor(q,dtype=torch.float32),'soft')
    assert np.allclose((sw[:,None]*soft).numpy(),s,atol=2e-7)
    report={'status':'passed','all_training_rows':len(q),'positive_mass_tied_rows':int(((counts>1)&(w>0)).sum()),'zero_mass_rows':int((w==0).sum()),'implementation':'independent NumPy target construction, not external replication'}
    (OUT/'target_audit.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
if __name__=='__main__':main()
