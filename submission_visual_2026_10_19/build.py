"""Compile the edited source, reject unresolved references/layout warnings, render all pages."""
import hashlib
import argparse
import json
import os
import re
import shutil
import subprocess
from pathlib import Path
import fitz
from PIL import Image,ImageDraw,ImageOps

HERE=Path(__file__).resolve().parent
PAPER=HERE.parent
SOURCE=HERE/'source'
DEST=PAPER/'output/submission-visual-2026-10-19'
BUILD=PAPER/'tmp/submission_visual_pdf'
NEW='submission_visual_2026_10_19/source'

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def expand(path):
    content=path.read_text(encoding='utf-8')
    return re.sub(r'\\input\{([^}]+)\}',lambda m:expand(PAPER/(m[1] if m[1].endswith('.tex') else m[1]+'.tex')),content)

def main(supplement_only=False):
    DEST.mkdir(parents=True,exist_ok=True);BUILD.mkdir(parents=True,exist_ok=True)
    compiler=Path('C:/Users/mgunj/AppData/Local/Programs/MiKTeX/miktex/bin/x64')
    if compiler.exists():os.environ['PATH']=str(compiler)+os.pathsep+os.environ['PATH']
    text=expand(SOURCE/'manuscript.tex')
    prohibited=('state of the art','state-of-the-art','significantly improved communication','genuine emotion','detects sarcasm','detects bluffs','privacy-preserving','TODO','PLACEHOLDER')
    for claim in prohibited:assert claim.lower() not in text.lower(),claim
    bib=(SOURCE/'references.bib').read_text(encoding='utf-8')
    keys=re.findall(r'@\w+\{([^,]+),',bib)
    cited=set(k.strip() for block in re.findall(r'\\cite\w*\{([^}]+)\}',text) for k in block.split(','))
    assert len(keys)==len(set(keys)) and set(keys)==cited,(set(keys)-cited,cited-set(keys))
    dois=re.findall(r'\bdoi\s*=\s*\{([^}]+)\}',bib,flags=re.I);assert len(dois)==len(set(x.lower() for x in dois))
    words=len(re.findall(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)*",re.sub(r'\\[A-Za-z]+','',(SOURCE/'generated/abstract.tex').read_text())))
    assert 150<=words<=250,words
    reports=[]
    jobs=[('final-manuscript',NEW+'/manuscript.tex','research-revision.pdf'),('final-supplement',NEW+'/generated/supplement.tex','Supplementary-diagnostics.pdf')]
    if (HERE/'Cover-letter.tex').exists():
        jobs.append(('final-cover','submission_visual_2026_10_19/Cover-letter.tex','Cover-letter.pdf'))
    if supplement_only:
        previous=json.loads((HERE/'checks/build_report.json').read_text())
        manuscript=previous['outputs'][0]
        assert manuscript['path']==str(DEST/'research-revision.pdf')
        assert manuscript['sha256']==sha(Path(manuscript['path']))
        reports.append(manuscript)
        jobs=jobs[1:]
    for job,entry,filename in jobs:
        cmd=['pdflatex','-interaction=nonstopmode','-halt-on-error','-jobname='+job,'-output-directory=tmp/submission_visual_pdf',entry]
        commands=[]
        for i in range(4 if job=='final-manuscript' else 3):
            r=subprocess.run(cmd,cwd=PAPER,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,encoding='utf-8',errors='replace');commands.append({'argv':cmd,'exit_code':r.returncode})
            if r.returncode:raise RuntimeError(r.stdout[-18000:])
            if i==0 and job=='final-manuscript':
                b=['bibtex','tmp/submission_visual_pdf/'+job];r=subprocess.run(b,cwd=PAPER,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,encoding='utf-8',errors='replace');commands.append({'argv':b,'exit_code':r.returncode});assert r.returncode==0,r.stdout
        log=(BUILD/(job+'.log')).read_text(errors='replace')
        warnings=[line for line in log.splitlines() if re.search(r'LaTeX Warning:|Package .* Warning:|Overfull \\[hv]box|Underfull \\[hv]box|Fatal error',line)]
        assert not warnings,warnings
        target=DEST/filename;shutil.copy2(BUILD/(job+'.pdf'),target)
        if job=='final-cover':shutil.copy2(target,HERE/filename)
        if job=='final-manuscript':shutil.copy2(BUILD/(job+'.bbl'),SOURCE/'manuscript.bbl')
        doc=fitz.open(target);rendered=BUILD/job;rendered.mkdir(exist_ok=True)
        for n,page in enumerate(doc):
            page.get_pixmap(matrix=fitz.Matrix(1.6,1.6)).save(rendered/f'page-{n+1:02d}.png')
            for font in page.get_fonts(full=True):assert font[0]>0 and doc.extract_font(font[0])[3] and font[2]!='Type3',font
        contacts=BUILD/(job+'-contacts');contacts.mkdir(exist_ok=True)
        for start in range(0,len(doc),4):
            canvas=Image.new('RGB',(1600,2310),'#cccccc');draw=ImageDraw.Draw(canvas)
            for j,i in enumerate(range(start,min(start+4,len(doc)))):
                with Image.open(rendered/f'page-{i+1:02d}.png') as original:im=ImageOps.contain(original.convert('RGB'),(780,1110))
                x,y=j%2*800+10,j//2*1155+30;canvas.paste(im,(x,y));draw.text((x,y-20),f'{job} page {i+1}',fill='black')
            canvas.save(contacts/f'sheet-{start//4+1:02d}.png')
        reports.append({'path':str(target),'sha256':sha(target),'pages':len(doc),'warnings':warnings,'commands':commands})
    report={'status':'passed','outputs':reports,'references':len(keys),'abstract_words':words,'prohibited_claim_scan':'passed','training_runs_added':0,'visual_review':'pending'}
    (HERE/'checks/build_report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--supplement-only',action='store_true',help='Rebuild the supplement while preserving the verified main PDF.')
    main(parser.parse_args().supplement_only)
