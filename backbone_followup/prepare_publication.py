"""Prepare additive v1.2.0 checkout; external publication remains separate."""
import hashlib,json,re,shutil,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parent.parent;PACKAGE=ROOT/'output/submission-backbone-2026-10-03';DEST=ROOT/'github-release/crowd-supervision-selective-fer'
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()
def main():
    assert (DEST/'.git').exists()
    report=json.loads((PACKAGE/'package_manifest.json').read_text());expected={r['name']:r['sha256'] for r in report['packages']};archive=PACKAGE/'OnlineResource1.zip';assert sha(archive)==expected[archive.name]
    with zipfile.ZipFile(archive) as z:
        checks=json.loads(z.read('SHA256_MANIFEST.json'))
        for name,h in checks.items():
            t=(DEST/name).resolve();assert t.is_relative_to(DEST.resolve()) and '.git' not in Path(name).parts
            blob=z.read(name);assert hashlib.sha256(blob).hexdigest()==h;t.parent.mkdir(parents=True,exist_ok=True);t.write_bytes(blob)
    for name in ('crowd-supervision-backbone-revision.pdf','Supplementary-diagnostics.pdf'):shutil.copy2(PACKAGE/name,DEST/'manuscript'/name)
    shutil.copy2(PACKAGE/'Manuscript-source.zip',DEST/'manuscript/Manuscript-source-v1.2.0.zip')
    readme=(DEST/'README.md').read_text(encoding='utf-8')
    assert '# Two-backbone revision v1.2.0' not in readme,'Do not duplicate current overview'
    readme='''# Two-backbone revision v1.2.0

The current manuscript includes 24 matched runs: 15 Swin-T and nine ResNet-18, with all selected/epoch-12 weights (48) available. This tests supervision sensitivity across model families, not independent-image generalization or architecture superiority. Detailed favorable/adverse outcomes are in backbone_followup/REVIEW_RESPONSE.md. The messaging prototype retains the Swin checkpoint contract.

Current PDF: manuscript/crowd-supervision-backbone-revision.pdf. Current editable source: manuscript/Manuscript-source-v1.2.0.zip. Current supplement: manuscript/Supplementary-diagnostics.pdf. Versioned assets: https://github.com/mgunjan67/crowd-supervision-selective-fer/releases/tag/v1.2.0.

The previous v1.0.0 and v1.1.0 releases remain unchanged. The user confirmed all-author v1.1.0 approval; authors should review the later v1.2.0 results before journal submission. No journal submission or TechRxiv update has occurred. The following older overviews remain historical and are not the current result set.

'''+readme
    (DEST/'README.md').write_text(readme,encoding='utf-8')
    ignored=(DEST/'.gitignore').read_text()+'\nbackbone_followup/weights/\nbackbone_followup/runs/*/*.pt\n';(DEST/'.gitignore').write_text(ignored)
    pattern=re.compile(r'(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,}|AKIA[A-Z0-9]{16}|-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----)')
    for p in DEST.rglob('*'):
        if not p.is_file() or '.git' in p.parts:continue
        assert p.name not in {'.env','.netrc','id_rsa','id_ed25519','credentials','hosts.yml'} and p.suffix!='.pt' and p.stat().st_size<90_000_000,p
        if p.suffix in ('.py','.txt','.md','.json','.toml','.tex','.bib','.yaml','.yml'):assert not pattern.search(p.read_text(errors='replace')),p
    (PACKAGE/'publication_preflight.json').write_text(json.dumps({'status':'passed','images_or_weights_in_git':False,'credential_scan':'passed','prior_releases_preserved':True},indent=2))
    names=('crowd-supervision-backbone-revision.pdf','Supplementary-diagnostics.pdf','Manuscript-source.zip','OnlineResource1.zip','OnlineResource7.zip','COVER_LETTER.txt','REVIEW_RESPONSE.md')
    assets={n:{'sha256':sha(PACKAGE/n),'bytes':(PACKAGE/n).stat().st_size} for n in names}
    (PACKAGE/'release-artifacts-v1.2.0.json').write_text(json.dumps(assets,indent=2));shutil.copy2(PACKAGE/'release-artifacts-v1.2.0.json',DEST/'release-artifacts-v1.2.0.json')
    notes='''## Two-backbone robustness revision

Nine new ResNet-18 runs compare Soft, Uniform and Tie-Uniform targets across three matched seeds. The paper reports all own-ranking, fixed-image and selected/fixed-epoch contrasts regardless of direction. Together with the retained Swin evidence, this is 24 runs and 48 inference weights, but still one reused image source and four adaptive phases. It is not an accepted article, external-domain validation, optimized architecture leaderboard or validated messaging intervention.

OnlineResource1.zip contains the merged auditable evidence/code/manuscript. OnlineResource7.zip contains all 18 ResNet inference weights. Original images and credentials are excluded. Unchanged Swin weights remain in preserved earlier releases:

'''
    for n,label in ((2,'Hard'),(3,'Soft'),(4,'Uniform'),(5,'Tie-hard'),(6,'Tie-Uniform')):
        version='v1.1.0' if n==6 else 'v1.0.0';notes+=f'- [Swin {label}: Online Resource {n}](https://github.com/mgunjan67/crowd-supervision-selective-fer/releases/download/{version}/OnlineResource{n}.zip)\n'
    notes+='\nAll authors approved v1.1.0; the later results need final author review before journal upload. No journal submission or preprint update has occurred.\n';(PACKAGE/'GITHUB_RELEASE_NOTES.md').write_text(notes)
    print('Prepared additive checkout; not pushed or released')
if __name__=='__main__':main()
