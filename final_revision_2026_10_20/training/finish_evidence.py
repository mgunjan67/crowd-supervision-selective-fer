"""Run the locked evaluation and analysis only after all declared training finishes.

This is a one-off continuation of the current experiment, not a scheduled task.
It neither adds training runs nor changes any scientific protocol.
"""
import datetime as dt
import json
import os
from pathlib import Path
import subprocess
import sys
import time

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
EXPECTED=[(7,'permutation',s) for s in (17,42,89)]+[
    (8,c,s) for c in ('soft','uniform','tie_uniform') for s in (17,42,89)]

def status(stage,**items):
    record=dict(stage=stage,utc=dt.datetime.now(dt.timezone.utc).isoformat(),**items)
    (HERE/'pipeline_status.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
    print(json.dumps(record),flush=True)

def main():
    environment=dict(os.environ)
    environment['OMP_NUM_THREADS']='1'
    environment['MKL_NUM_THREADS']='1'
    old=None
    while True:
        finished=[f'k{k}_{c}_seed{s}' for k,c,s in EXPECTED
                  if (HERE/'runs'/f'k{k}_{c}_seed{s}'/'training_complete.json').exists()]
        if finished!=old:
            status('waiting_for_all_training',completed=len(finished),total=len(EXPECTED),runs=finished)
            old=finished
        if len(finished)==len(EXPECTED):break
        time.sleep(30)
    for script in ('evaluate_followup.py','analyze_followup.py','export_followup.py'):
        status('running',script=script)
        result=subprocess.run([sys.executable,'-u',str(HERE/script)],cwd=ROOT,env=environment)
        if result.returncode:
            status('failed',script=script,exit_code=result.returncode)
            raise SystemExit(result.returncode)
    status('evidence_complete',training_runs=12,nominal_inference_states=33,
           manuscript_integration='pending outcome-aware review')

if __name__=='__main__':main()
