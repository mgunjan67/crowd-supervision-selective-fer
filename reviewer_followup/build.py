"""Compile, validate and render the reviewer-follow-up edition only."""
import argparse,hashlib,json,re,shutil,subprocess
from pathlib import Path
import fitz
OUT=Path(__file__).resolve().parent;ROOT=OUT.parent
BUILD=ROOT/'tmp/reviewer_followup_pdf'
PDF=ROOT/'output/pdf/crowd-supervision-final-revision.pdf'

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def run(command,cwd=ROOT):
    result=subprocess.run(command,cwd=cwd,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding='utf-8',errors='replace')
    if result.returncode:raise RuntimeError(result.stdout[-18000:])
    return result.stdout

def expand(path):
    text=path.read_text(encoding='ascii')
    return re.sub(r'\\input\{([^}]+)\}',lambda m:expand(ROOT/m.group(1)),text)

def source_check():
    text=expand(OUT/'manuscript.tex');bib=(OUT/'references.bib').read_text(encoding='ascii')
    assert not re.search(r'\b(?:TODO|FIXME|pending|placeholder)\b',text,re.I)
    for pattern in (r'state[ -]of[ -]the[ -]art',r'significantly improved communication',r'genuine emotion',r'detects? sarcasm',r'detects? bluffs?',r'privacy-preserving',r'\bVADER\b',r'affective dissonance'):
        assert not re.search(pattern,text,re.I),pattern
    keys=re.findall(r'@\w+\{([^,]+),',bib)
    cited={k.strip() for group in re.findall(r'\\cite\{([^}]+)\}',text) for k in group.split(',')}
    assert len(keys)==len(set(keys)) and set(keys)==cited,(set(keys)-cited,cited-set(keys))
    dois=re.findall(r'doi\s*=\s*\{([^}]+)\}',bib,re.I);assert len(dois)==len(set(dois))
    abstract=text.split('\\abstract{',1)[1].split('\\keywords',1)[0]
    words=re.findall(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)*",re.sub(r'\\[a-zA-Z]+','',abstract))
    assert 150<=len(words)<=250,('Abstract words',len(words))
    for heading in ('Purpose:','Methods:','Results:','Conclusion:'):assert heading in abstract
    for required in ('shahalan0202@gmail.com','d6622300231@g.siit.tu.ac.th','No institutional ethics approval','OpenAI Codex','Online Resources 2--5'):
        assert required in text,required
    for file in ('asset_sources.json','generated_sha256.json'):
        for p,digest in json.loads((OUT/file).read_text()).items():assert sha(OUT/p)==digest,p
    return {'abstract_words':len(words),'references':len(keys),'unused_references':0,'unsupported_claim_scan':'passed'}

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--source-check-only',action='store_true');args=parser.parse_args()
    checks=source_check()
    if args.source_check_only:print(json.dumps(checks,indent=2));return
    BUILD.mkdir(parents=True,exist_ok=True)
    command=['pdflatex','-interaction=nonstopmode','-halt-on-error','-jobname=reviewer-manuscript',
        '-output-directory=tmp/reviewer_followup_pdf','reviewer_followup/manuscript.tex']
    for step in range(4):
        run(command)
        if step==0:run(['bibtex','tmp/reviewer_followup_pdf/reviewer-manuscript'])
    log=(BUILD/'reviewer-manuscript.log').read_text(errors='replace')
    patterns=(r'LaTeX Warning:',r'Package .* Warning:',r'Overfull \\[hv]box',r'Underfull \\[hv]box',r'undefined citations?',r'undefined references?',r'Fatal error')
    problems=[p for p in patterns if re.search(p,log,re.I)]
    assert not problems,problems
    assert '???' not in (BUILD/'reviewer-manuscript.bbl').read_text()
    PDF.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(BUILD/'reviewer-manuscript.pdf',PDF)
    shutil.copy2(BUILD/'reviewer-manuscript.bbl',OUT/'manuscript.bbl')
    doc=fitz.open(PDF);render=BUILD/'render';render.mkdir(exist_ok=True)
    fonts={f[0]:f for p in doc for f in p.get_fonts(full=True)}
    for xref,font in fonts.items():assert xref>0 and doc.extract_font(xref)[3] and font[2]!='Type3',font
    for i,page in enumerate(doc):page.get_pixmap(matrix=fitz.Matrix(1.6,1.6)).save(render/f'page-{i+1:02d}.png')
    supplementary_command=['pdflatex','-interaction=nonstopmode','-halt-on-error','-jobname=supplementary-diagnostics',
        '-output-directory=tmp/reviewer_followup_pdf','reviewer_followup/generated/supplement.tex']
    for _ in range(3):run(supplementary_command)
    supplement_log=(BUILD/'supplementary-diagnostics.log').read_text(errors='replace')
    assert not any(re.search(p,supplement_log,re.I) for p in patterns),'Supplement layout/reference warning'
    supplement_pdf=OUT/'Supplementary-diagnostics.pdf'
    shutil.copy2(BUILD/'supplementary-diagnostics.pdf',supplement_pdf)
    sup=fitz.open(supplement_pdf);sup_render=BUILD/'supplement-render';sup_render.mkdir(exist_ok=True)
    for i,page in enumerate(sup):page.get_pixmap(matrix=fitz.Matrix(1.6,1.6)).save(sup_render/f'page-{i+1:02d}.png')
    report={**checks,'pdf':str(PDF),'pdf_sha256':sha(PDF),'pages':len(doc),'rendered_pages':len(doc),'warnings':0,
        'embedded_fonts':len(fonts),'source_sha256':sha(OUT/'manuscript.tex'),'visual_review':'required, not implied by rendering',
        'supplement_pdf_sha256':sha(supplement_pdf),'supplement_pages':len(sup),'supplement_warnings':0}
    (OUT/'build_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report,indent=2))

if __name__=='__main__':main()
