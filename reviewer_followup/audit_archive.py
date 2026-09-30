"""Verify exactly the files delivered, including fresh source rebuild and CPU audits."""
import hashlib,json,os,re,subprocess,sys,tempfile,zipfile
from pathlib import Path
import fitz
OUT=Path(__file__).resolve().parent;ROOT=OUT.parent;DEST=ROOT/'output/submission-reviewer-final'

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1048576),b''):h.update(chunk)
    return h.hexdigest()
def run(command,cwd):
    env=dict(os.environ,OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
    r=subprocess.run(command,cwd=cwd,capture_output=True,text=True,encoding='utf-8',errors='replace',env=env)
    if r.returncode:raise RuntimeError(r.stdout[-16000:]+r.stderr[-16000:])
    return r.stdout
def compare(a,b):
    original=fitz.open(a);rebuilt=fitz.open(b);assert len(original)==len(rebuilt)
    for p,q in zip(original,rebuilt):
        assert p.get_text()==q.get_text(),'Extracted-source text mismatch'
        assert p.get_pixmap().samples==q.get_pixmap().samples,'Extracted-source rendering mismatch'
    return len(original)

def main():
    location=Path(tempfile.mkdtemp(prefix='reviewer-final-archive-',dir=ROOT/'tmp')).resolve()
    packages=[]
    for record in json.loads((DEST/'package_manifest.json').read_text())['packages']:
        path=DEST/record['name'];assert sha(path)==record['sha256']
        with zipfile.ZipFile(path) as z:
            manifest=json.loads(z.read('SHA256_MANIFEST.json'))
            assert set(z.namelist())==set(manifest)|{'SHA256_MANIFEST.json'}
            for name,digest in manifest.items():
                target=(location/name).resolve();assert target.is_relative_to(location)
                h=hashlib.sha256()
                with z.open(name) as f:
                    for chunk in iter(lambda:f.read(1048576),b''):h.update(chunk)
                assert h.hexdigest()==digest,name
                if target.exists():assert sha(target)==digest,'Overlapping file differs: '+name
                else:z.extract(name,location)
        packages.append({'name':path.name,'sha256':record['sha256'],'files':len(manifest)})
        print('ARCHIVE VERIFIED '+path.name,flush=True)
    for step in range(4):
        run(['pdflatex','-interaction=nonstopmode','-halt-on-error','main.tex'],location)
        if step==0:run(['bibtex','main'],location)
    for _ in range(3):run(['pdflatex','-interaction=nonstopmode','-halt-on-error','-jobname=supplementary-diagnostics',
        'reviewer_followup/generated/supplement.tex'],location)
    for name in ('main.log','supplementary-diagnostics.log'):
        assert not re.search(r'LaTeX Warning:|Package .* Warning:|Overfull \\[hv]box|Underfull \\[hv]box|Fatal error',(location/name).read_text(errors='replace'))
    pages=compare(DEST/'crowd-supervision-final-revision.pdf',location/'main.pdf')
    suppages=compare(DEST/'Supplementary-diagnostics.pdf',location/'supplementary-diagnostics.pdf')
    print('ALL MAIN AND SUPPLEMENT PAGES REPRODUCED',flush=True)
    numerical=json.loads(run([sys.executable,'reviewer_followup/audit.py'],location))
    contract=json.loads(run([sys.executable,'reviewer_followup/check_checkpoint_contracts.py'],location))
    source=json.loads(run([sys.executable,'reviewer_followup/build.py','--source-check-only'],location))
    tests=run([sys.executable,'-m','pytest','code/tests','final_experiment/test_final.py','final_experiment/test_analysis.py',
        '-c','code/pytest.ini','-p','no:cacheprovider','--basetemp','test-temporary-core','-q'],location)
    followup=run([sys.executable,'-m','pytest','reviewer_followup/test_controls.py','reviewer_followup/test_diagnostics.py',
        '-c','code/pytest.ini','-p','no:cacheprovider','--basetemp','test-temporary-followup','-q'],location)
    report={'status':'passed','packages':packages,'pages':pages,'supplement_pages':suppages,'all_page_renders_identical':True,
        'numerical_audit':numerical,'checkpoint_contract':contract,'source_check':source,
        'core_test_output':tests.strip(),'followup_test_output':followup.strip(),'scratch_directory':str(location),
        'scope':'Internal extracted-package verification, not external replication'}
    for path in (OUT/'archive_audit.json',DEST/'archive_audit.json'):
        path.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
