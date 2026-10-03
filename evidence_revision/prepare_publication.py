"""Prepare a checked local Git checkout; pushing/uploading is a separate action."""
import hashlib,json,re,shutil,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parent.parent
PACKAGE=ROOT/'output/submission-evidence-2026-10-03'
DEST=ROOT/'github-release/crowd-supervision-selective-fer'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    assert (DEST/'.git').exists()
    report=json.loads((PACKAGE/'package_manifest.json').read_text());expected={r['name']:r['sha256'] for r in report['packages']}
    archive=PACKAGE/'OnlineResource1.zip';assert sha(archive)==expected[archive.name]
    with zipfile.ZipFile(archive) as z:
        checks=json.loads(z.read('SHA256_MANIFEST.json'))
        for name,h in checks.items():
            target=(DEST/name).resolve();assert target.is_relative_to(DEST.resolve()) and '.git' not in Path(name).parts
            blob=z.read(name);assert hashlib.sha256(blob).hexdigest()==h
            target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(blob)
    for name in ('crowd-supervision-evidence-revision.pdf','Supplementary-diagnostics.pdf'):
        shutil.copy2(PACKAGE/name,DEST/'manuscript'/name)
    shutil.copy2(PACKAGE/'Manuscript-source.zip',DEST/'manuscript/Manuscript-source-v1.1.0.zip')
    readme=(DEST/'README.md').read_text(encoding='utf-8')
    readme='# Evidence revision v1.1.0\n\nFifteen matched runs now include a tie-preserving dominant-confidence Uniform control. The new manuscript adds fixed-image cross-evaluation, identical-population F1 comparisons, full learning curves and unsupported-category retention. These diagnostics narrow—not strengthen without qualification—the interpretation of the earlier aggregate benefit. See the versioned manuscript and evidence below.\n\nThe following overview preserves the original v1.0.0 twelve-run results. Use **Current revision files** below for the fifteen-run evidence revision.\n\n'+readme
    readme=readme.replace('releases/tag/v1.0.0','releases/tag/v1.1.0')
    readme+='\n## Current revision files\n\n- `manuscript/crowd-supervision-evidence-revision.pdf` is the current v1.1.0 manuscript.\n- `manuscript/Supplementary-diagnostics.pdf` contains detailed runtime and diagnostic information.\n- `manuscript/Manuscript-source-v1.1.0.zip` supplies editable Springer source.\n- `evidence_revision/REVIEW_RESPONSE.md` maps the latest supplied criticisms to changes and unresolved scientific boundaries.\n- Online Resource 6 adds six Tie-Uniform inference checkpoints; Online Resources 2–5 remain byte-identical to v1.0.0.\n\nThe old v1.0.0 release is preserved. This is not an accepted article or a journal submission. Final author approval of these later results is required before journal submission.\n'
    (DEST/'README.md').write_text(readme,encoding='utf-8')
    ignored=(DEST/'.gitignore').read_text()+'\nevidence_revision/weights/\nevidence_revision/runs/*/*.pt\n'
    (DEST/'.gitignore').write_text(ignored)
    forbidden={'.env','.netrc','id_rsa','id_ed25519','credentials','hosts.yml'}
    pattern=re.compile(r'(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,}|AKIA[A-Z0-9]{16}|-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----)')
    for p in DEST.rglob('*'):
        if not p.is_file() or '.git' in p.parts:continue
        assert p.name not in forbidden and p.suffix!='.pt' and p.stat().st_size<90_000_000,p
        if p.suffix in ('.py','.txt','.md','.json','.toml','.tex','.bib','.yaml','.yml'):assert not pattern.search(p.read_text(errors='replace')),p
    (PACKAGE/'publication_preflight.json').write_text(json.dumps({'status':'passed','images_or_weights_in_git':False,'credential_scan':'passed','prior_release_preserved':True},indent=2))
    asset_names=('crowd-supervision-evidence-revision.pdf','Supplementary-diagnostics.pdf','Manuscript-source.zip','OnlineResource1.zip','OnlineResource6.zip','COVER_LETTER.txt','REVIEW_RESPONSE.md')
    assets={name:{'sha256':sha(PACKAGE/name),'bytes':(PACKAGE/name).stat().st_size} for name in asset_names}
    (PACKAGE/'release-artifacts-v1.1.0.json').write_text(json.dumps(assets,indent=2))
    shutil.copy2(PACKAGE/'release-artifacts-v1.1.0.json',DEST/'release-artifacts-v1.1.0.json')
    notes='''## Bounded evidence revision

Fifteen matched training runs now include three Tie-Uniform runs that preserve tied maxima. The revision adds fixed-image predictor/selector cross-evaluation, identical-population label-reference F1, per-seed unsupported-image retention, separate unsupported-vote categories and complete learning curves. The paper is a bounded within-source empirical study, not an accepted journal article, external validation or demonstrated communication intervention. The original v1.0.0 remains unchanged. Final author review of the later results is needed before journal submission.

OnlineResource1.zip contains the merged auditable evidence and source. OnlineResource6.zip supplies all six new selected/epoch-12 Tie-Uniform inference weights. Original face images and credentials are excluded. Original weights below are byte-identical to the prior release and remain available there:

'''
    for index,label in ((2,'Hard'),(3,'Soft'),(4,'Uniform'),(5,'Tie-hard')):
        notes+=f'- [{label}: Online Resource {index}](https://github.com/mgunjan67/crowd-supervision-selective-fer/releases/download/v1.0.0/OnlineResource{index}.zip)\n'
    notes+='\nThe source, predictions and checkpoint hashes are supplied for reproducibility. Internal checks are not external replication. No journal submission or TechRxiv update has been made.\n'
    (PACKAGE/'GITHUB_RELEASE_NOTES.md').write_text(notes)
    print('Public checkout prepared; no push or release upload performed')
if __name__=='__main__':main()
