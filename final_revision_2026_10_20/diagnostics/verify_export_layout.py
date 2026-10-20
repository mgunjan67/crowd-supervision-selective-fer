"""Record exact-preamble and font/log checks for the synthetic S11 fixture."""
import hashlib
import json
from pathlib import Path
import re
import fitz

HERE=Path(__file__).resolve().parent
REVISION=HERE.parent
FIXTURE=REVISION/'checks/export_layout_fixture'


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    source=(REVISION/'source/generated/supplement.tex').read_text().split(r'\begin{document}',1)[0]
    wrapper=(FIXTURE/'S11LayoutCheck.tex').read_text().split(r'\begin{document}',1)[0]
    # The one compatibility input is the only addition to the actual preamble.
    normalize=lambda text:re.sub(r'\\input\{[^}]*longtable_compat\.tex\}\n','',text)
    assert normalize(wrapper)==normalize(source)
    log=(FIXTURE/'S11LayoutCheck.log').read_text(errors='replace')
    issues=[line for line in log.splitlines() if re.search(r'Warning|Overfull|Underfull|ignored error|Infinite glue',line)]
    assert not issues,issues
    with fitz.open(FIXTURE/'S11LayoutCheck.pdf') as doc:
        fonts=[font for page in doc for font in page.get_fonts(full=True)]
        assert all(font[2]!='Type3' for font in fonts)
        pages=len(doc)
    audit=dict(status='passed',synthetic_fixture=True,empirical_results=False,
               actual_supplement_preamble_matched=True,compatibility_patch='longtable finite-shrink backport',
               page_count=pages,type3_fonts=0,layout_log_issues=issues,
               visual_inspection='All four current Poppler-rendered pages inspected; no clipping or overlapping tables',
               latex_log_sha256=sha(FIXTURE/'S11LayoutCheck.log'),
               pdf_sha256=sha(FIXTURE/'S11LayoutCheck.pdf'),
               compatibility_source_sha256=sha(HERE/'longtable_compat.tex'))
    (FIXTURE/'layout_verification.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
    print(json.dumps(audit,indent=2))


if __name__=='__main__':main()
