"""Create immutable source manifest and exact mechanical data-helper copies."""
import hashlib,json,shutil
from pathlib import Path
from datetime import datetime,timezone
OUT=Path(__file__).resolve().parent;ROOT=OUT.parent

def main():
    if (OUT/'freeze.json').exists(): raise RuntimeError('Already frozen; do not overwrite')
    for name in ('data.py','split_manifest.json'):
        shutil.copy2(ROOT/'final_experiment'/name,OUT/name)
    files=[OUT/name for name in ('PROTOCOL.md','train_controls.py','run_all.py','test_controls.py','freeze.py','data.py','split_manifest.json')]
    files += [ROOT/'final_experiment/train.py']+list((ROOT/'code/emotion_cue').glob('*.py'))
    hashes={p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    (OUT/'freeze.json').write_text(json.dumps({'frozen_utc':datetime.now(timezone.utc).isoformat(),
        'prior_test_results_known':True,'sha256':hashes},indent=2),encoding='utf-8')

if __name__=='__main__': main()
