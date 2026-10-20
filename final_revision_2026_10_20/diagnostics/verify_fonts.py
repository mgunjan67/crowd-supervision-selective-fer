"""Verify all new vector figures use embedded non-Type3 fonts."""
import hashlib
import json
from pathlib import Path
import fitz

HERE=Path(__file__).resolve().parent


def main():
    reports=[]
    files=sorted(HERE.glob('*.pdf'))
    files.extend(HERE.parent/'source/figures'/name for name in ('ExternalAggregation.pdf','TargetIntervention.pdf'))
    for path in files:
        with fitz.open(path) as document:
            pages=[]
            for index,page in enumerate(document):
                fonts=page.get_fonts(full=True)
                assert fonts,f'No fonts found in {path.name}'
                assert all(font[2]!='Type3' for font in fonts),(path.name,index,fonts)
                pages.append(dict(page=index+1,fonts=[dict(xref=f[0],extension=f[1],type=f[2],base_font=f[3]) for f in fonts]))
            reports.append(dict(path=str(path),pages=pages,
                                sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    result=dict(status='passed',type3_fonts=0,pdf_files_checked=len(files),reports=reports)
    (HERE/'font_audit.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(dict(status='passed',type3_fonts=0,pdf_files_checked=len(files)),indent=2))


if __name__=='__main__':main()
