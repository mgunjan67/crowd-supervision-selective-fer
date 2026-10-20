"""Read-only source checks and arithmetic precision audit; no GPU operations."""
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
TRAIN=HERE.parent/'training'
PAPER=HERE.parents[1]
sys.path.insert(0,str(TRAIN))
import train_followup as experiment


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    manifest=json.loads((PAPER/'final_experiment/split_manifest.json').read_text())
    source_records=[]
    for name,expected in manifest['source_hashes'].items():
        observed=sha(PAPER/name)
        assert observed==expected,name
        source_records.append(dict(path=name,expected=expected,observed=observed,match=True))
    freeze=json.loads((TRAIN/'freeze.json').read_text())
    for name,expected in freeze['source_sha256'].items():
        assert sha(PAPER/name)==expected,name
    q=experiment.base.data.load_role('train')
    r,w=experiment.targets(q['q'],q['rows'],7,'soft')
    effective=(r.astype(np.float32)*w.astype(np.float32)[:,None]).astype(float)
    direct=q['q'][:,:7].astype(np.float32).astype(float)
    difference=np.abs(effective-direct)
    report=dict(status='passed',dataset_source_hashes=source_records,
                frozen_training_source_hashes_verified=len(freeze['source_sha256']),
                planned_run_list_matches=freeze['runs']==[list(x) for x in experiment.RUNS],
                gpu_used=False,precision_comparison='New supported-mass times normalized float32 target versus historical direct float32 Soft counts',
                effective_target_max_absolute_difference=float(difference.max()),
                effective_target_mean_absolute_difference=float(difference.mean()),
                nonidentical_effective_target_rows=int((difference>0).any(1).sum()),
                limitation='Tiny rounding differences preclude a claim of bitwise identical target arithmetic against retained seven-output fits; within the new eight-output family arithmetic is shared.')
    assert report['planned_run_list_matches']
    (TRAIN/'input_data_audit.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
