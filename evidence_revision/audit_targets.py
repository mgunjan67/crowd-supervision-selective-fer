"""Check target construction on every actual training vote vector, without images."""
import csv,json
from pathlib import Path
import numpy as np
import torch
from train_tie_uniform import targets
OUT=Path(__file__).resolve().parent;ROOT=OUT.parent
def main():
    with (ROOT/'data/ferplus_annotations/fer2013new.csv').open(newline='',encoding='utf-8-sig') as f:rr=[r for r in csv.DictReader(f) if r['Usage']=='Training']
    names=('anger','disgust','fear','happiness','neutral','sadness','surprise','contempt','unknown','NF')
    votes=np.array([[int(r[k]) for k in names] for r in rr],float);q=votes/votes.sum(1,keepdims=True)
    tq=torch.tensor(q,dtype=torch.float32);actual,w=targets(tq);positive=w.numpy()>0
    supported=q[:,:7];r=np.divide(supported,supported.sum(1,keepdims=True),out=np.zeros_like(supported),where=supported.sum(1,keepdims=True)>0)
    maxima=supported==supported.max(1,keepdims=True);k=maxima.sum(1);m=r.max(1)
    expected=np.zeros((len(r),7))
    for i in np.flatnonzero(positive):
        expected[i]=1/7 if k[i]==7 else np.where(maxima[i],m[i],(1-k[i]*m[i])/(7-k[i]))
    assert np.allclose(actual.numpy()[positive],expected[positive],atol=2e-7)
    assert np.allclose(actual.numpy()[positive].sum(1),1,atol=2e-7) and (actual>=0).all()
    assert len(rr)==28709 and (~positive).sum()==152 and ((k>1)&positive).sum()==1360
    report={'status':'passed','training_rows':len(rr),'positive_mass_tied_rows':int(((k>1)&positive).sum()),'zero_mass_rows':int((~positive).sum()),'independent_numpy_formula_verified':True}
    (OUT/'target_audit.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
if __name__=='__main__':main()
