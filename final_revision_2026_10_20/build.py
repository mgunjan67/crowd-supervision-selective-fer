"""Compile and render the revision; final builds require completed new evidence."""
import argparse
import importlib.util
import json
import re
from pathlib import Path

HERE=Path(__file__).resolve().parent
PAPER=HERE.parent

def active_source(path):
    """Expand inputs for final checks without permitting a silent omission."""
    content=path.read_text(encoding='utf-8')
    if '\\IfFileExists' in content:
        raise AssertionError(f'Final source may not silently omit evidence: {path}')
    def include(match):
        item=match[1]
        child=PAPER/(item if item.endswith('.tex') else item+'.tex')
        return active_source(child)
    return re.sub(r'\\input\{([^}]+)\}',include,content)

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--draft',action='store_true')
    args=parser.parse_args()
    if not args.draft:
        complete=json.loads((HERE/'training/evaluation_complete.json').read_text())
        assert complete['training_runs']==12 and complete['states']==33
        assert (HERE/'training/results/summary.json').exists()
        assert (HERE/'source/generated/new_followup.tex').exists()
        active_source(HERE/'source/manuscript.tex')
        active_source(HERE/'source/generated/supplement.tex')
    spec=importlib.util.spec_from_file_location('final_revision_build_engine',PAPER/'submission_visual_2026_10_19/build.py')
    engine=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(engine)
    engine.HERE=HERE
    engine.PAPER=PAPER
    engine.SOURCE=HERE/'source'
    engine.DEST=PAPER/('tmp/final_revision_draft/output' if args.draft else 'output/submission-final-2026-10-20')
    engine.BUILD=PAPER/'tmp/final_revision_pdf'
    engine.NEW='final_revision_2026_10_20/source'
    original_run=engine.subprocess.run
    def run_current(argv,*pos,**kw):
        if isinstance(argv,list):
            argv=[str(v).replace('tmp/submission_visual_pdf','tmp/final_revision_pdf') for v in argv]
        return original_run(argv,*pos,**kw)
    engine.subprocess.run=run_current
    if args.draft:
        def draft_expand(path):
            if not path.exists() and path.name=='new_followup.tex':
                return ''
            text=path.read_text(encoding='utf-8')
            return re.sub(r'\\input\{([^}]+)\}',lambda m:draft_expand(
                PAPER/(m[1] if m[1].endswith('.tex') else m[1]+'.tex')),text)
        engine.expand=draft_expand
    (HERE/'checks').mkdir(exist_ok=True)
    engine.main()
    path=HERE/'checks/build_report.json'
    report=json.loads(path.read_text())
    ignored_errors=[]
    for job in ('final-manuscript','final-supplement'):
        log=(engine.BUILD/(job+'.log')).read_text(errors='replace')
        ignored_errors.extend((job,line) for line in log.splitlines()
                              if re.search(r'ignored error:|Infinite glue shrinkage',line,re.I))
    if ignored_errors:
        report.update(status='failed',ignored_typesetting_errors=ignored_errors)
        path.write_text(json.dumps(report,indent=2),encoding='utf-8')
        raise AssertionError(ignored_errors)
    # The inherited engine records its original argv; record the executed paths.
    for output in report['outputs']:
        for command in output.get('commands',[]):
            command['argv']=[str(v).replace('tmp/submission_visual_pdf',
                                            'tmp/final_revision_pdf')
                             for v in command['argv']]
    report.update(draft=args.draft,training_runs_added=0 if args.draft else 12,
                  scientific_evidence='pending new training' if args.draft else 'new outputs integrated')
    path.write_text(json.dumps(report,indent=2),encoding='utf-8')

if __name__=='__main__':
    main()
