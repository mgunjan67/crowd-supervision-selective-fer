"""Compile the packaged source independently and check physical caption order."""
import hashlib, json, os, re, subprocess, zipfile
from pathlib import Path
import fitz

OUT = Path(__file__).resolve().parent
ROOT = OUT.parent
DEST = ROOT/'output/submission-journal-style-2026-10-04-scientific-diagrams'

def main():
    source = DEST/'source'
    qa = ROOT/'tmp/journal-style-packaged-source'
    qa.mkdir(parents=True, exist_ok=True)
    compiler = Path('C:/Users/mgunj/AppData/Local/Programs/MiKTeX/miktex/bin/x64')
    env = dict(os.environ); env['PATH'] = str(compiler)+os.pathsep+env['PATH']
    def run(command):
        p = subprocess.run(command,cwd=source,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding='utf-8',errors='replace')
        assert p.returncode == 0, p.stdout[-12000:]
    with zipfile.ZipFile(DEST/'Manuscript-source.zip') as z:
        manifest = json.loads(z.read('SHA256_MANIFEST.json'))
        for name, h in manifest.items():
            assert hashlib.sha256((source/name).read_bytes()).hexdigest() == h
    command = [str(compiler/'pdflatex.exe'),'-interaction=nonstopmode','-halt-on-error',
               '-jobname=package-check','-output-directory='+str(qa),'main.tex']
    for i in range(4):
        run(command)
        if i == 0:
            run([str(compiler/'bibtex.exe'),str(qa/'package-check')])
    log = (qa/'package-check.log').read_text(errors='replace')
    assert not re.search(r'LaTeX Warning:|Package .* Warning:|Overfull \\[hv]box|Underfull \\[hv]box',log,re.I)
    main_doc = fitz.open(DEST/'crowd-supervision-journal-style.pdf')
    built_doc = fitz.open(qa/'package-check.pdf')
    assert len(main_doc) == len(built_doc)
    assert [p.get_text() for p in main_doc] == [p.get_text() for p in built_doc]
    all_main_text = '\n'.join(p.get_text() for p in main_doc)
    main_figures = [int(n) for n in re.findall(r'Fig\.\s+(\d+)\s',all_main_text)]
    assert main_figures == list(range(1,8)), main_figures
    diagram_fonts={}
    diagram_pages={}
    for name,needle in (('methodology','Matched optimization'),('model','Swin-T encoder:')):
        page=next(p for p in main_doc if needle in p.get_text())
        spans=[s for b in page.get_text('dict')['blocks'] if 'lines' in b for l in b['lines'] for s in l['spans'] if s['font'].startswith('Arial')]
        assert spans and min(s['size'] for s in spans)>=8, name+' normal lettering too small at final size'
        diagram_fonts[name]=min(s['size'] for s in spans)
        diagram_pages[name]=page.number+1
    sup = fitz.open(DEST/'Supplementary-diagnostics.pdf')
    alltext = '\n'.join(p.get_text() for p in sup)
    tables = [int(n) for n in re.findall(r'(?m)^Table S(\d+):',alltext)]
    figures = [int(n) for n in re.findall(r'(?m)^Figure S(\d+):',alltext)]
    assert tables == list(range(1,16)), tables
    assert figures == list(range(1,9)), figures
    report = {'status':'passed','archived_source_hashes_match':True,
              'packaged_source_compiles':True,'packaged_pdf_text_matches_main':True,
              'packaged_source_warnings':0,'main_figure_order':main_figures,'diagram_pages':diagram_pages,
              'diagram_minimum_normal_font_pt':diagram_fonts,'supplement_table_order':tables,
              'supplement_figure_order':figures,'main_pages':len(main_doc),'supplement_pages':len(sup)}
    (DEST/'package_verification.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))

if __name__ == '__main__':
    main()
