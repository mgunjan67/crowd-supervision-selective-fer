import hashlib,json
from datetime import datetime,timezone
from pathlib import Path
OUT=Path(__file__).resolve().parent;ROOT=OUT.parent
def main():
    if (OUT/'freeze.json').exists():
        assert (OUT/'freeze-before-import-repair.json').exists()
        assert not list((OUT/'runs').glob('*/epoch-*.json'))
    files=[OUT/n for n in ('PROTOCOL.md','model.py','train.py','driver.py','data.py','split_manifest.json','test_followup.py','freeze.py','TECHNICAL_REPAIRS.md')]
    files+=list((ROOT/'code/emotion_cue').glob('*.py'))
    files+=[ROOT/'data/fer2013'/f'{n}.npz' for n in ('Training','PublicTest','PrivateTest')]
    files+=[ROOT/'data/ferplus_annotations/fer2013new.csv',ROOT/'cache/torch/hub/checkpoints/resnet18-f37072fd.pth']
    report={'frozen_utc':datetime.now(timezone.utc).isoformat(),'prior_test_results_known':True,'planned_runs':9,'sha256':{p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in files}}
    (OUT/'freeze.json').write_text(json.dumps(report,indent=2));print(json.dumps({'frozen_utc':report['frozen_utc'],'sources':len(files),'runs':9}))
if __name__=='__main__':main()
