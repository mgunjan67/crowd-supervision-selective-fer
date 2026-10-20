"""Execute an unchanged prior audit, redirecting its report to this new edition."""
import argparse
import json
import runpy
import sys
from pathlib import Path

HERE=Path(__file__).resolve().parent
PAPER=HERE.parent

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('audit',choices=('reviewer_followup/audit.py','reviewer_followup/verify_raw_source.py',
        'evidence_revision/audit.py','evidence_revision/audit_targets.py','backbone_followup/audit.py',
        'backbone_followup/audit_targets.py','ravdess_transfer_2026_10_11/verify.py',
        'ravdess_transfer_2026_10_11/audit_runtime.py'))
    args=parser.parse_args();entry=PAPER/args.audit
    destination=HERE/'checks';destination.mkdir(exist_ok=True)
    original=Path.write_text;captured=[]
    def redirected(path,data,*pos,**kw):
        resolved=path.resolve()
        if resolved.parent==entry.parent.resolve() and resolved.suffix=='.json':
            new=destination/(entry.parent.name+'_'+resolved.name)
            captured.append(str(new));return original(new,data,*pos,**kw)
        raise RuntimeError('Unexpected write by preserved audit: '+str(path))
    Path.write_text=redirected
    sys.path[:0]=[str(entry.parent),str(PAPER/'code')]
    sys.argv=[str(entry)]
    try:runpy.run_path(str(entry),run_name='__main__')
    finally:Path.write_text=original
    print(json.dumps({'unchanged_audit':str(entry),'reports_redirected':captured,'original_reports_overwritten':False}))

if __name__=='__main__':main()
