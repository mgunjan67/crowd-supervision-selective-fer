"""Bounded single-thread prefetch without changing the frozen sample order."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import torch
from run_memory_safe import OrderedMemorySafeLoader,verify_amendment
import train_followup as experiment

HERE=Path(__file__).resolve().parent

class PrefetchedLoader(OrderedMemorySafeLoader):
    def __iter__(self):
        iterator=super().__iter__()
        exhausted=object()
        def fetch():
            return next(iterator,exhausted)
        with ThreadPoolExecutor(max_workers=1,thread_name_prefix='fer-input') as pool:
            pending=pool.submit(fetch)
            while True:
                batch=pending.result()
                if batch is exhausted:
                    return
                pending=pool.submit(fetch)
                yield batch

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--all',action='store_true')
    args=parser.parse_args()
    verify_amendment()
    paths=[HERE/'AMENDMENT_002.md',Path(__file__),HERE/'test_prefetched.py']
    record=dict(source_sha256={p.name:experiment.base.data.sha(p) for p in paths},
                original_freeze_sha256=experiment.base.data.sha(HERE/'freeze.json'),
                first_amendment_freeze_sha256=experiment.base.data.sha(HERE/'amendment_001_freeze.json'))
    path=HERE/'amendment_002_freeze.json'
    if path.exists():
        assert json.loads(path.read_text())==record
    else:
        path.write_text(json.dumps(record,indent=2),encoding='utf-8')
    experiment.DataLoader=PrefetchedLoader
    original_seed=experiment.base.seed_all
    def seed_with_single_cpu_thread(seed):
        original_seed(seed)
        torch.set_num_threads(1)
    experiment.base.seed_all=seed_with_single_cpu_thread
    for k,condition,seed in experiment.RUNS:
        folder=HERE/'runs'/f'k{k}_{condition}_seed{seed}'
        folder.mkdir(parents=True,exist_ok=True)
        execution=folder/'execution_amendment_002.json'
        if not execution.exists():
            epochs=sorted(folder.glob('epoch-??.json'))
            prior=int(epochs[-1].stem.split('-')[1]) if epochs else 0
            execution.write_text(json.dumps(dict(**record,applied_after_committed_epoch=prior),indent=2),encoding='utf-8')
        experiment.run(k,condition,seed)
        first=folder/'execution_amendment.json'
        if not first.exists():
            first.write_text(json.dumps(verify_amendment(),indent=2),encoding='utf-8')

if __name__=='__main__':
    main()
