"""Compile and render the multi-file Springer revision without changing v1.0.0."""
import hashlib,json,re,shutil,subprocess
from pathlib import Path
import fitz
OUT=Path(__file__).resolve().parent;ROOT=OUT.parent
BUILD=ROOT/'tmp/evidence_revision_pdf';PDF=ROOT/'output/pdf/crowd-supervision-evidence-revision.pdf'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def expand(p):return re.sub(r'\\input\{([^}]+)\}',lambda m:expand(ROOT/m.group(1)),p.read_text())
def run(cmd):
    p=subprocess.run(cmd,cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding='utf-8',errors='replace')
    if p.returncode:raise RuntimeError(p.stdout[-16000:])
def main():
    text=expand(OUT/'manuscript.tex');bib=(OUT/'references.bib').read_text()
    for pattern in (r'\b(?:TODO|FIXME|placeholder)\b',r'state[ -]of[ -]the[ -]art',r'significantly improved communication',r'genuine emotion',r'detects? sarcasm',r'detects? bluffs?',r'privacy-preserving',r'\bVADER\b',r'affective dissonance',r'ambiguity-preserving'):
        assert not re.search(pattern,text,re.I),pattern
    keys=re.findall(r'@\w+\{([^,]+),',bib);cited={k.strip() for group in re.findall(r'\\cite\{([^}]+)\}',text) for k in group.split(',')}
    assert len(keys)==len(set(keys)) and set(keys)==cited,(set(keys)-cited,cited-set(keys))
    dois=re.findall(r'doi\s*=\s*\{([^}]+)\}',bib,re.I);assert len(dois)==len(set(dois))
    abstract=text.split('\\abstract{',1)[1].split('\\keywords',1)[0];words=re.findall(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)*",re.sub(r'\\[a-zA-Z]+','',abstract));assert 150<=len(words)<=250,len(words)
    for heading in ('Purpose:','Methods:','Results:','Conclusion:'):assert heading in abstract
    for file in ('asset_sources.json','generated_sha256.json'):
        for p,h in json.loads((OUT/file).read_text()).items():assert sha(OUT/p)==h,p
    BUILD.mkdir(parents=True,exist_ok=True)
    command=['pdflatex','-interaction=nonstopmode','-halt-on-error','-jobname=evidence-manuscript','-output-directory=tmp/evidence_revision_pdf','evidence_revision/manuscript.tex']
    for i in range(4):
        run(command)
        if i==0:run(['bibtex','tmp/evidence_revision_pdf/evidence-manuscript'])
    patterns=(r'LaTeX Warning:',r'Package .* Warning:',r'Overfull \\[hv]box',r'Underfull \\[hv]box',r'undefined citations?',r'undefined references?',r'Fatal error')
    logs=(BUILD/'evidence-manuscript.log').read_text(errors='replace');problems=[p for p in patterns if re.search(p,logs,re.I)]
    assert not problems,problems
    shutil.copy2(BUILD/'evidence-manuscript.bbl',OUT/'manuscript.bbl');PDF.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(BUILD/'evidence-manuscript.pdf',PDF)
    command=['pdflatex','-interaction=nonstopmode','-halt-on-error','-jobname=evidence-supplement','-output-directory=tmp/evidence_revision_pdf','evidence_revision/generated/supplement.tex']
    for _ in range(3):run(command)
    logs=(BUILD/'evidence-supplement.log').read_text(errors='replace');assert not any(re.search(p,logs,re.I) for p in patterns),'Supplement warnings'
    sup=OUT/'Supplementary-diagnostics.pdf';shutil.copy2(BUILD/'evidence-supplement.pdf',sup)
    counts=[]
    for file,name in ((PDF,'render'),(sup,'supplement-render')):
        doc=fitz.open(file);folder=BUILD/name;folder.mkdir(exist_ok=True)
        for i,page in enumerate(doc):page.get_pixmap(matrix=fitz.Matrix(1.6,1.6)).save(folder/f'page-{i+1:02d}.png')
        for page in doc:
            for font in page.get_fonts(full=True):assert font[0]>0 and doc.extract_font(font[0])[3] and font[2]!='Type3',font
        counts.append(len(doc))
    report={'pdf':str(PDF),'pdf_sha256':sha(PDF),'supplement_pdf_sha256':sha(sup),'pages':counts[0],'supplement_pages':counts[1],'warnings':0,'supplement_warnings':0,'abstract_words':len(words),'references':len(keys),'unused_references':0,'unsupported_claim_scan':'passed','visual_review':'required; rendering is not inspection'}
    (OUT/'build_report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
if __name__=='__main__':main()
