import hashlib,json,shutil
from datetime import datetime,timezone
from pathlib import Path
OUT=Path(__file__).resolve().parent;ROOT=OUT.parent

def main():
    assert not (OUT/'freeze.json').exists(),'Preserve frozen protocol'
    for name in ('data.py','split_manifest.json'):shutil.copy2(ROOT/'reviewer_followup'/name,OUT/name)
    paths=[OUT/name for name in ('PROTOCOL.md','train_tie_uniform.py','test_targets.py','freeze.py','data.py','split_manifest.json')]
    paths += [ROOT/'final_experiment/train.py',ROOT/'final_experiment/data.py']+list((ROOT/'code/emotion_cue').glob('*.py'))
    paths += [ROOT/'data/fer2013'/f'{name}.npz' for name in ('Training','PublicTest','PrivateTest')]
    report={'frozen_utc':datetime.now(timezone.utc).isoformat(),'prior_test_results_known':True,'sha256':{p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}}
    (OUT/'freeze.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print('Frozen',len(paths),'sources')

if __name__=='__main__':main()
