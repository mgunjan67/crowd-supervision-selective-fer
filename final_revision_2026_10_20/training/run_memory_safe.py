"""Execute the frozen plan with sample-order-equivalent sequential loading."""
import argparse
import json
import math
from pathlib import Path
import torch
from torch.utils.data import DataLoader as TorchDataLoader
import train_followup as experiment

HERE=Path(__file__).resolve().parent

class OrderedMemorySafeLoader:
    def __init__(self,dataset,batch_size,shuffle=False,generator=None,num_workers=0,
                 pin_memory=False,persistent_workers=False,**kwargs):
        assert not kwargs,kwargs
        self.dataset=dataset
        self.batch_size=batch_size
        self.shuffle=shuffle
        self.generator=generator
        self.pin_memory=pin_memory
    def __len__(self):
        return math.ceil(len(self.dataset)/self.batch_size)
    def __iter__(self):
        if self.shuffle:
            assert self.generator is not None
            epoch=int(self.dataset.epoch)
            if epoch==1:
                torch.empty((),dtype=torch.int64).random_(generator=self.generator)
            indices=torch.randperm(len(self.dataset),generator=self.generator).tolist()
        else:
            indices=range(len(self.dataset))
        return iter(TorchDataLoader(self.dataset,batch_size=self.batch_size,sampler=indices,
                    num_workers=0,pin_memory=self.pin_memory,
                    generator=torch.Generator().manual_seed(20261020)))

def verify_amendment():
    paths=[HERE/'AMENDMENT_001.md',Path(__file__),HERE/'test_memory_safe.py']
    record=dict(source_sha256={p.name:experiment.base.data.sha(p) for p in paths},
                repair='sequential workers preserving uninterrupted persistent-loader order',
                original_freeze_sha256=experiment.base.data.sha(HERE/'freeze.json'))
    destination=HERE/'amendment_001_freeze.json'
    if destination.exists():
        assert json.loads(destination.read_text())==record
    else:
        destination.write_text(json.dumps(record,indent=2),encoding='utf-8')
    return record

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--all',action='store_true')
    parser.add_argument('--k',type=int,default=7)
    parser.add_argument('--condition',default='permutation')
    parser.add_argument('--seed',type=int,default=17)
    args=parser.parse_args()
    record=verify_amendment()
    experiment.DataLoader=OrderedMemorySafeLoader
    for k,condition,seed in experiment.RUNS if args.all else [(args.k,args.condition,args.seed)]:
        experiment.run(k,condition,seed)
        folder=HERE/'runs'/f'k{k}_{condition}_seed{seed}'
        (folder/'execution_amendment.json').write_text(json.dumps(record,indent=2),encoding='utf-8')

if __name__=='__main__':
    main()
