"""Continue the authorized local evidence pipeline after the bounded training phase."""
import json,subprocess,sys,time
from datetime import datetime,timezone
from pathlib import Path
OUT=Path(__file__).resolve().parent

def status(phase):
    (OUT/'phase_status.json').write_text(json.dumps({'phase':phase,'updated_utc':datetime.now(timezone.utc).isoformat()},indent=2),encoding='utf-8')
    print(phase,flush=True)

def main():
    status('waiting_for_all_six_control_runs')
    started=time.monotonic()
    while True:
        try:record=json.loads((OUT/'run_status.json').read_text())
        except (FileNotFoundError,json.JSONDecodeError):record={}
        if record.get('status')=='all_six_training_complete':break
        if time.monotonic()-started>8*3600:raise TimeoutError('Training phase did not finish within the safety window')
        time.sleep(30)
    for script in ('evaluate.py','analysis.py','audit.py','check_checkpoint_contracts.py','build_assets.py'):
        status('running_'+script)
        subprocess.run([sys.executable,'-u',str(OUT/script)],check=True)
    status('evidence_ready_for_pdf_compilation_and_visual_review')

if __name__=='__main__':main()
