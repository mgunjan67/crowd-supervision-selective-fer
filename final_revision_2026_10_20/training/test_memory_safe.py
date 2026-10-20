"""Actual persistent-worker comparison protects sample-order equivalence."""
import json
from pathlib import Path
import torch
from torch.utils.data import Dataset,DataLoader
from run_memory_safe import OrderedMemorySafeLoader

class Indices(Dataset):
    def __init__(self):self.epoch=torch.zeros((),dtype=torch.int64).share_memory_()
    def __len__(self):return 65
    def __getitem__(self,i):return i

def test_same_actual_persistent_order():
    dataset=Indices()
    g1,g2=torch.Generator(),torch.Generator()
    old=DataLoader(dataset,batch_size=8,shuffle=True,generator=g1,num_workers=2,persistent_workers=True)
    new=OrderedMemorySafeLoader(dataset,batch_size=8,shuffle=True,generator=g2)
    results=[]
    for epoch in (1,2,3):
        dataset.epoch.fill_(epoch)
        g1.manual_seed(17+epoch*100003);g2.manual_seed(17+epoch*100003)
        a=torch.cat(list(old));b=torch.cat(list(new))
        assert torch.equal(a,b)
        results.append(dict(epoch=epoch,samples=len(a),order_identical=True))
    # Avoid persistent worker imports surviving this short proof.
    old._iterator._shutdown_workers()
    path=Path(__file__).resolve().parent/'sample_order_proof.json'
    path.write_text(json.dumps(dict(passed=True,epochs=results),indent=2),encoding='utf-8')
