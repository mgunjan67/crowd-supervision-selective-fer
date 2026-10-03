"""Frozen nine-run driver, no calibration/test inference during training."""
import json
from pathlib import Path
import train as base
OUT=Path(__file__).resolve().parent;ROOT=OUT.parent
SEEDS=(17,42,89);CONDITIONS=('soft','uniform','tie_uniform')
def verify_freeze():
    for name,digest in json.loads((OUT/'freeze.json').read_text())['sha256'].items():
        assert base.data.sha(ROOT/name)==digest,name
def main():
    verify_freeze()
    for seed in SEEDS:
        reference=None
        for condition in CONDITIONS:
            base.run(condition,seed)
            directory=OUT/'runs'/f'{condition}_seed{seed}'
            current=json.loads((directory/'configuration.json').read_text())
            match={k:v for k,v in current.items() if k!='condition'}
            if reference is None:reference=match
            assert reference==match,'Within-seed matching failed'
            (directory/'matched_design.json').write_text(json.dumps({'passed':True,'reference':'soft','seed':seed},indent=2))
    (OUT/'training_complete.json').write_text(json.dumps({'runs':9,'seeds':SEEDS,'conditions':CONDITIONS,'finished_utc':base.now(),'test_evaluated':False},indent=2))
if __name__=='__main__':main()
