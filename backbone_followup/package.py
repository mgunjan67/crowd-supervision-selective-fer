"""New handoff; preserve prior packages/releases and never redistribute images."""
import hashlib,json,re,shutil,zipfile
from pathlib import Path
OUT=Path(__file__).resolve().parent;ROOT=OUT.parent
DEST=ROOT/'output/submission-backbone-2026-10-03';PREVIOUS=ROOT/'output/submission-evidence-2026-10-03'
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()
def archive(path,files,stored=False):
    assert not path.exists(),f'Preserve existing archive {path}'
    checks={name:sha(p) for name,p in sorted(files.items())}
    with zipfile.ZipFile(path,'w',zipfile.ZIP_STORED if stored else zipfile.ZIP_DEFLATED,allowZip64=True) as z:
        for name,p in sorted(files.items()):z.write(p,name)
        z.writestr('SHA256_MANIFEST.json',json.dumps(checks,indent=2))
    with zipfile.ZipFile(path) as z:
        assert len(z.namelist())==len(set(z.namelist()))
        for name,h in checks.items():assert hashlib.sha256(z.read(name)).hexdigest()==h,name
    return {'name':path.name,'sha256':sha(path),'bytes':path.stat().st_size,'files':len(files),'archive_hash_audit':'passed'}
def main():
    audit=json.loads((OUT/'audit.json').read_text());build=json.loads((OUT/'build_report.json').read_text());visual=json.loads((OUT/'visual_review.json').read_text())
    assert audit['status']=='passed' and audit['new_inference_checkpoints_verified']==18
    assert audit['ledger_sha256']==sha(OUT/'reported_results.csv')
    pdf=Path(build['pdf']);sup=OUT/'Supplementary-diagnostics.pdf'
    assert build['warnings']==build['supplement_warnings']==0
    assert visual['all_pages_inspected'] and visual['all_supplement_pages_inspected']
    assert sha(pdf)==build['pdf_sha256']==visual['pdf_sha256'] and sha(sup)==visual['supplement_pdf_sha256']
    DEST.mkdir(parents=True,exist_ok=True)
    source={'main.tex':OUT/'manuscript.tex','main.bbl':OUT/'manuscript.bbl','sn-jnl.cls':ROOT/'sn-jnl.cls','sn-mathphys-num.bst':ROOT/'sn-mathphys-num.bst','BUILD_README.txt':OUT/'BUILD_README.txt'}
    for p in [OUT/'references.bib']+list((OUT/'generated').glob('*.tex'))+list((OUT/'figures').glob('*.pdf'))+list((OUT/'figures').glob('*.eps')):source[p.relative_to(ROOT).as_posix()]=p
    for name,p in source.items():t=DEST/'source'/name;t.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,t)
    records=[archive(DEST/'Manuscript-source.zip',source)]
    merged=DEST/'evidence-files';merged.mkdir(exist_ok=True)
    with zipfile.ZipFile(PREVIOUS/'OnlineResource1.zip') as z:
        checks=json.loads(z.read('SHA256_MANIFEST.json'))
        for name,h in checks.items():
            blob=z.read(name);assert hashlib.sha256(blob).hexdigest()==h
            t=(merged/name).resolve();assert t.is_relative_to(merged.resolve());t.parent.mkdir(parents=True,exist_ok=True);t.write_bytes(blob)
    extras=[p for p in OUT.iterdir() if p.is_file() and p.suffix in ('.py','.json','.md','.csv','.bib','.tex','.bbl','.pdf','.xml') and p.name!='progress.json']
    extras+=list((OUT/'predictions').glob('*/*.npz'))+list((OUT/'predictions').glob('*/*.json'))+list((OUT/'generated').glob('*.tex'))+list((OUT/'figures').glob('*.pdf'))
    for run in (OUT/'runs').iterdir():extras+=[p for p in run.iterdir() if p.is_file() and (p.suffix=='.json' or p.name in ('environment.txt','best_selection_predictions.npz'))]
    for name,p in source.items():t=merged/name;t.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,t)
    for p in extras:t=merged/p.relative_to(ROOT);t.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,t)
    shutil.copy2(OUT/'REPRODUCIBILITY.md',merged/'ONLINE_RESOURCE_1_README.md')
    files={p.relative_to(merged).as_posix():p for p in merged.rglob('*') if p.is_file()}
    assert not any(p.suffix=='.pt' for p in files.values())
    assert not any('data/raw/' in n or 'data/fer2013/' in n for n in files)
    records.append(archive(DEST/'OnlineResource1.zip',files))
    weights={p.relative_to(ROOT).as_posix():p for p in (OUT/'weights').glob('*.pt')};assert len(weights)==18
    weights['ONLINE_RESOURCE_7_REPRODUCIBILITY.md']=OUT/'REPRODUCIBILITY.md';records.append(archive(DEST/'OnlineResource7.zip',weights,stored=True))
    old=json.loads((PREVIOUS/'package_manifest.json').read_text())
    for rec in old['packages']:
        if rec['name'].startswith('OnlineResource') and rec['name']!='OnlineResource1.zip':
            records.append({'name':rec['name'],'sha256':rec['sha256'],'bytes':rec['bytes'],'unchanged_from':'v1.1.0' if rec['name']=='OnlineResource6.zip' else 'v1.0.0','local_path':str(PREVIOUS/rec['name'])})
    shutil.copy2(pdf,DEST/pdf.name);shutil.copy2(sup,DEST/sup.name)
    for name in ('REVIEW_RESPONSE.md','REPRODUCIBILITY.md','REFERENCE_AUDIT.md','COVER_LETTER.txt','SUBMISSION_CHECKLIST.md','START_HERE.md','AUTHOR_APPROVAL.md'):shutil.copy2(OUT/name,DEST/name)
    meta=json.loads((PREVIOUS/'PORTAL_METADATA.json').read_text());meta['revision']='v1.2.0 two-backbone robustness revision';meta['all_four_authors_approve_v1_1_results']=True;meta['v1_1_confirmation_date']='2026-10-03';meta['all_four_authors_approve_v1_2_results']=None
    meta['confirmation_source']='User confirmed all four approved v1.1.0 and authorized extension; later new results need final review'
    meta['abstract_source']='source/backbone_followup/generated/abstract.tex';meta['abstract']=re.sub(r'\\textbf\{([^}]+)\}',r'\1',(OUT/'generated/abstract.tex').read_text()).replace('\\%','%').strip();meta['artifact_release']='https://github.com/mgunjan67/crowd-supervision-selective-fer/releases/tag/v1.2.0';meta['public_deposit_performed']=False
    (DEST/'PORTAL_METADATA.json').write_text(json.dumps(meta,indent=2))
    report={'version':'v1.2.0','pdf_sha256':sha(pdf),'supplement_pdf_sha256':sha(sup),'packages':records,'journal_submission_performed':False,'v1_1_author_approval_confirmed':True,'v1_2_results_author_review':'required before submission','ethics':'Truthful no-approval/no-exemption disclosure retained'}
    (DEST/'package_manifest.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
if __name__=='__main__':main()
