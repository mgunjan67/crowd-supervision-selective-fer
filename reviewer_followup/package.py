"""Build a separate release after exact-PDF visual approval; preserve earlier editions."""
import hashlib,json,shutil,zipfile
from pathlib import Path
OUT=Path(__file__).resolve().parent;ROOT=OUT.parent
DEST=ROOT/'output/submission-reviewer-final'
PDF=ROOT/'output/pdf/crowd-supervision-final-revision.pdf'

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1048576),b''):h.update(chunk)
    return h.hexdigest()
def mapping(paths):return {p.relative_to(ROOT).as_posix():p for p in paths}
def archive(name,files,weights=False):
    path=DEST/name
    if path.exists():raise RuntimeError('Preserve existing release archive; choose an explicit new edition before replacing it')
    manifest={name:sha(p) for name,p in sorted(files.items())}
    with zipfile.ZipFile(path,'w',zipfile.ZIP_STORED if weights else zipfile.ZIP_DEFLATED,allowZip64=True) as z:
        for name,p in sorted(files.items()):z.write(p,name)
        z.writestr('SHA256_MANIFEST.json',json.dumps(manifest,indent=2))
    return {'name':path.name,'sha256':sha(path),'files':len(files),'bytes':path.stat().st_size}

def main():
    audit=json.loads((OUT/'audit.json').read_text());build=json.loads((OUT/'build_report.json').read_text())
    visual=json.loads((OUT/'visual_review.json').read_text());contract=json.loads((OUT/'checkpoint_contract_audit.json').read_text())
    assert audit['status']==contract['status']=='passed' and audit['inference_checkpoints_verified']==24
    assert build['warnings']==build['supplement_warnings']==0 and build['pdf_sha256']==sha(PDF)
    assert visual['all_pages_inspected'] and visual['all_supplement_pages_inspected']
    assert visual['pdf_sha256']==sha(PDF) and visual['supplement_pdf_sha256']==sha(OUT/'Supplementary-diagnostics.pdf')
    assert audit['ledger_sha256']==sha(OUT/'reported_results.csv')
    DEST.mkdir(parents=True,exist_ok=True)
    source=mapping([ROOT/'sn-jnl.cls',ROOT/'sn-mathphys-num.bst',OUT/'references.bib'])
    source.update(mapping(list((OUT/'generated').glob('*.tex'))+list((OUT/'figures').glob('*.pdf'))+list((OUT/'figures').glob('*.eps'))))
    source.update({'main.tex':OUT/'manuscript.tex','main.bbl':OUT/'manuscript.bbl','BUILD_README.txt':OUT/'BUILD_README.txt'})
    excluded={'__pycache__','.pytest_cache'}
    supplement=mapping([p for p in (ROOT/'code').rglob('*') if p.is_file() and not excluded.intersection(p.parts) and p.suffix!='.pyc' and p.name!='README.md'])
    supplement.update(source)
    new_names=['PROTOCOL.md','freeze.json','data.py','split_manifest.json','train_controls.py','run_all.py','freeze.py','evaluate.py',
        'analysis.py','audit.py','build_assets.py','build.py','package.py','audit_archive.py','write_handoff.py','retrain.py','reproduce_inference.py','finish_after_training.py',
        'check_checkpoint_contracts.py','verify_raw_source.py','raw_source_audit.json','DATA_ALIGNMENT.md','test_controls.py','test_diagnostics.py','reported_results.csv','contrasts.json','analysis_sources.json',
        'asset_sources.json','generated_sha256.json','audit.json','evaluation_freeze.json','evaluation_complete.json','checkpoint_contract_audit.json',
        'build_report.json','visual_review.json','test_results.xml','core_test_results.xml','manuscript.tex','REPRODUCIBILITY.md','MODEL_CARD.md',
        'REFERENCE_AUDIT.md','ETHICS_SUBMISSION_RISK.md','SUPPLEMENTARY_DIAGNOSTICS.md','Supplementary-diagnostics.pdf']
    supplement.update(mapping([OUT/n for n in new_names]))
    old_names=['PROTOCOL.md','data.py','train.py','analysis.py','test_final.py','test_analysis.py','split_manifest.json','split_manifest.csv',
        'REFERENCE_AUDIT.md','HISTORICAL_PROVENANCE.md','app_smoke_test.json','browser_smoke_test.json','inference_reproduction.json']
    supplement.update(mapping([ROOT/'final_experiment'/n for n in old_names]))
    supplement.update(mapping([ROOT/n for n in ['evidence/dataset_audit.json','evidence/dataset_manifest.csv',
        'reported_results.csv','data/ferplus_annotations/fer2013new.csv','data/ferplus_annotations/LICENSE.md',
        'data/ferplus_annotations/README.md','data/ferplus_annotations/source.json','scripts/prepare_fer2013.py']]))
    supplement['ONLINE_RESOURCE_1_README.md']=OUT/'REPRODUCIBILITY.md'
    supplement['code/README.md']=OUT/'CODE_README.md'
    for c in ('hard','soft','uniform','tie_hard'):
        for s in (17,42,89):
            directory=(ROOT/'final_experiment' if c in ('hard','soft') else OUT)/'runs'/f'{c}_seed{s}'
            files=[p for p in directory.iterdir() if p.suffix in ('.json','.npz') or p.name=='environment.txt']
            supplement.update(mapping(files))
    supplement.update(mapping(list((OUT/'predictions').glob('*/*.json'))+list((OUT/'predictions').glob('*/*.npz'))))
    for p in list(source.values())+list(supplement.values()):assert p.is_file(),p
    records=[archive('Manuscript-source.zip',source),archive('OnlineResource1.zip',supplement)]
    for i,c in enumerate(('hard','soft','uniform','tie_hard'),2):
        files=mapping(sorted((OUT/'weights').glob(f'{c}_seed*.pt')));assert len(files)==6
        files[f'ONLINE_RESOURCE_{i}_MODEL_CARD.md']=OUT/'MODEL_CARD.md'
        records.append(archive(f'OnlineResource{i}.zip',files,weights=True))
    for name in ('START_HERE.md','REVIEW_RESPONSE.md','SUBMISSION_CHECKLIST.md','REPRODUCIBILITY.md','COVER_LETTER.txt','FINAL_REVIEW.md',
                 'ETHICS_SUBMISSION_RISK.md','Supplementary-diagnostics.pdf'):
        shutil.copy2(OUT/name,DEST/name)
    shutil.copy2(PDF,DEST/PDF.name)
    report={'pdf_sha256':sha(PDF),'supplement_pdf_sha256':sha(OUT/'Supplementary-diagnostics.pdf'),'packages':records,
        'public_upload_performed':False,'submission_performed':False,
        'unresolved_ethics_risk':'No institutional approval or exemption determination; authors explicitly declined to seek one',
        'technical_audit':'See archive_audit.json for extracted-package verification; no acceptance guarantee'}
    (DEST/'package_manifest.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report,indent=2))

if __name__=='__main__':main()
