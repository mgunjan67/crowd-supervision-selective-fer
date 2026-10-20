"""Render final PDFs with Poppler; actual visual review is recorded by finish.py."""
import hashlib
import json
import subprocess
from pathlib import Path
from PIL import Image
import fitz

HERE=Path(__file__).resolve().parent;PAPER=HERE.parent
DEST=PAPER/'output/submission-reader-2026-10-13'
POPPLER=Path('C:/Users/mgunj/.cache/codex-runtimes/codex-primary-runtime/dependencies/native/poppler/Library/bin/pdftoppm.exe')
def main():
    records=[]
    for filename,stem in [('research-revision.pdf','main'),('Supplementary-diagnostics.pdf','supplement')]:
        directory=PAPER/'tmp/submission_reader_pdf/poppler'/stem;directory.mkdir(parents=True,exist_ok=True)
        argv=[str(POPPLER),'-r','90','-png',str((DEST/filename).resolve()),str((directory/'page').resolve())]
        r=subprocess.run(argv,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,encoding='utf-8',errors='replace');assert r.returncode==0,r.stdout
        pages=sorted(directory.glob('page-*.png'));doc=fitz.open(DEST/filename);assert len(pages)==len(doc)
        for p in pages:
            with Image.open(p) as im:assert im.width>600 and im.height>900
        records.append({'pdf':filename,'sha256':hashlib.sha256((DEST/filename).read_bytes()).hexdigest(),'pages_rendered':len(pages),'command':argv,'exit_code':r.returncode,'renderer':'Poppler','png_directory':str(directory)})
    cover_dir=PAPER/'tmp/submission_reader_pdf/poppler/cover';cover_dir.mkdir(parents=True,exist_ok=True)
    cover_argv=[str(POPPLER),'-r','120','-singlefile','-png',str((DEST/'Cover-letter.pdf').resolve()),str((cover_dir/'page-01').resolve())]
    r=subprocess.run(cover_argv,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,encoding='utf-8',errors='replace');assert r.returncode==0,r.stdout
    assert len(fitz.open(DEST/'Cover-letter.pdf'))==1 and (cover_dir/'page-01.png').is_file()
    report={'status':'passed','outputs':records,'cover_letter':{'pdf':'Cover-letter.pdf','sha256':hashlib.sha256((DEST/'Cover-letter.pdf').read_bytes()).hexdigest(),'pages_rendered':1,'command':cover_argv,'exit_code':r.returncode,'png':str(cover_dir/'page-01.png')},'visual_review':'Pending manual review; this report confirms rendering only.'}
    (HERE/'checks/poppler_render.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
if __name__=='__main__':main()
