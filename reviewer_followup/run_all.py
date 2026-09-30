"""Bounded six-run supervisor. Safe rerun resumes committed epochs."""
import json, subprocess, sys
from pathlib import Path
from datetime import datetime, timezone
OUT=Path(__file__).resolve().parent

def main():
    for seed in (17,42,89):
        for condition in ('uniform','tie_hard'):
            status={'active':f'{condition}_seed{seed}','updated_utc':datetime.now(timezone.utc).isoformat()}
            (OUT/'run_status.json').write_text(json.dumps(status,indent=2),encoding='utf-8')
            subprocess.run([sys.executable,str(OUT/'train_controls.py'),'--condition',condition,'--seed',str(seed)],check=True)
    (OUT/'run_status.json').write_text(json.dumps({'status':'all_six_training_complete',
        'finished_utc':datetime.now(timezone.utc).isoformat()},indent=2),encoding='utf-8')

if __name__=='__main__': main()
