"""Separate v1.1.0 handoff; previous archives and releases are not overwritten."""
import hashlib,json,re,shutil,zipfile
from pathlib import Path
OUT=Path(__file__).resolve().parent;ROOT=OUT.parent
DEST=ROOT/'output/submission-evidence-2026-10-03';PREVIOUS=ROOT/'output/submission-github-2026-09-30'
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()
def archive(path,files,stored=False):
    assert not path.exists(),f'Preserve existing archive: {path}'
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
    assert audit['status']=='passed' and audit['new_inference_checkpoints_verified']==6
    pdf=Path(build['pdf']);sup=OUT/'Supplementary-diagnostics.pdf'
    assert build['warnings']==build['supplement_warnings']==0
    assert visual['all_pages_inspected'] and visual['all_supplement_pages_inspected']
    assert sha(pdf)==build['pdf_sha256']==visual['pdf_sha256'] and sha(sup)==visual['supplement_pdf_sha256']
    assert audit['ledger_sha256']==sha(OUT/'reported_results.csv')
    DEST.mkdir(parents=True,exist_ok=True)
    source={'main.tex':OUT/'manuscript.tex','main.bbl':OUT/'manuscript.bbl','sn-jnl.cls':ROOT/'sn-jnl.cls','sn-mathphys-num.bst':ROOT/'sn-mathphys-num.bst'}
    source['BUILD_README.txt']=OUT/'BUILD_README.txt'
    for p in [OUT/'references.bib']+list((OUT/'generated').glob('*.tex'))+list((OUT/'figures').glob('*.pdf')):source[p.relative_to(ROOT).as_posix()]=p
    sourcefolder=DEST/'source';sourcefolder.mkdir(exist_ok=True)
    for name,p in source.items():t=sourcefolder/name;t.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,t)
    records=[archive(DEST/'Manuscript-source.zip',source)]
    # Extend the already verified old evidence archive without images or weights.
    merged=DEST/'evidence-files';merged.mkdir(exist_ok=True)
    old=PREVIOUS/'OnlineResource1.zip'
    with zipfile.ZipFile(old) as z:
        hashes=json.loads(z.read('SHA256_MANIFEST.json'))
        for name,h in hashes.items():
            assert hashlib.sha256(z.read(name)).hexdigest()==h
            target=(merged/name).resolve();assert target.is_relative_to(merged.resolve())
            target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(z.read(name))
    extras=[p for p in OUT.iterdir() if p.is_file() and p.suffix in ('.py','.json','.md','.csv','.bib','.tex','.bbl','.pdf','.xml') and p.name not in ('progress.json',)]
    extras+=list((OUT/'predictions').glob('*/*.npz'))+list((OUT/'predictions').glob('*/*.json'))
    extras+=list((OUT/'generated').glob('*.tex'))+list((OUT/'figures').glob('*.pdf'))
    for run in (OUT/'runs').glob('*'):
        extras += [p for p in run.iterdir() if p.is_file() and (p.suffix=='.json' or p.name in ('environment.txt','best_selection_predictions.npz'))]
    for name,p in source.items():target=merged/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,target)
    for p in extras:target=merged/p.relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,target)
    shutil.copy2(OUT/'REPRODUCIBILITY.md',merged/'ONLINE_RESOURCE_1_README.md')
    files={p.relative_to(merged).as_posix():p for p in merged.rglob('*') if p.is_file()}
    assert not any(p.suffix=='.pt' for p in files.values())
    assert not any('data/raw/' in n or 'data/fer2013/' in n for n in files)
    records.append(archive(DEST/'OnlineResource1.zip',files))
    weights={p.relative_to(ROOT).as_posix():p for p in (OUT/'weights').glob('*.pt')};assert len(weights)==6
    weights['ONLINE_RESOURCE_6_REPRODUCIBILITY.md']=OUT/'REPRODUCIBILITY.md';records.append(archive(DEST/'OnlineResource6.zip',weights,stored=True))
    for n in ('OnlineResource2.zip','OnlineResource3.zip','OnlineResource4.zip','OnlineResource5.zip'):
        shutil.copy2(PREVIOUS/n,DEST/n);records.append({'name':n,'sha256':sha(DEST/n),'bytes':(DEST/n).stat().st_size,'unchanged_from':'v1.0.0'})
    shutil.copy2(pdf,DEST/pdf.name);shutil.copy2(sup,DEST/sup.name)
    for n in ('REVIEW_RESPONSE.md','REPRODUCIBILITY.md','REFERENCE_AUDIT.md','COVER_LETTER.txt','SUBMISSION_CHECKLIST.md','START_HERE.md'):
        shutil.copy2(OUT/n,DEST/n)
    metadata=json.loads((PREVIOUS/'PORTAL_METADATA.json').read_text())
    metadata['revision']='v1.1.0 evidence revision; final approval of these later results must be obtained before submission'
    metadata['abstract_source']='source/evidence_revision/generated/abstract.tex'
    metadata['abstract']=re.sub(r'\\textbf\{([^}]+)\}',r'\1',(OUT/'generated/abstract.tex').read_text()).replace('\\%','%').strip()
    metadata['prior_revision_author_confirmation_date']=metadata.pop('author_confirmation_date')
    metadata['prior_revision_approval']=metadata.pop('all_four_authors_approve_final_manuscript_and_new_results')
    metadata['all_four_authors_approve_v1_1_results']=None
    metadata['confirmation_source']='Prior author approval covered v1.0.0; later evidence revision needs final author review'
    metadata['artifact_release']='https://github.com/mgunjan67/crowd-supervision-selective-fer/releases/tag/v1.1.0'
    metadata['public_deposit_performed']=False
    (DEST/'PORTAL_METADATA.json').write_text(json.dumps(metadata,indent=2))
    report={'version':'v1.1.0','pdf_sha256':sha(pdf),'supplement_pdf_sha256':sha(sup),'packages':records,'journal_submission_performed':False,'new_revision_author_approval':'required before journal submission','ethics':'No institutional approval or exemption obtained; truthful disclosure retained'}
    (DEST/'package_manifest.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
if __name__=='__main__':main()
