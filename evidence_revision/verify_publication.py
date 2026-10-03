"""Check all new public assets and preserved old weight links before promotion."""
import hashlib,json,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parent.parent;DEST=ROOT/'output/submission-evidence-2026-10-03'
REPO='mgunjan67/crowd-supervision-selective-fer'
def gh(*args):return json.loads(subprocess.check_output(['gh',*args],text=True,encoding='utf-8'))
def main():
    release=gh('api',f'repos/{REPO}/releases/tags/v1.1.0');old=gh('api',f'repos/{REPO}/releases/tags/v1.0.0');commit=gh('api',f'repos/{REPO}/commits/main')
    expected=json.loads((DEST/'release-artifacts-v1.1.0.json').read_text());p=DEST/'release-artifacts-v1.1.0.json'
    expected[p.name]={'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size}
    actual={a['name']:a for a in release['assets']};assert set(actual)==set(expected)
    for name,v in expected.items():assert actual[name]['state']=='uploaded' and actual[name]['size']==v['bytes'] and actual[name]['digest']=='sha256:'+v['sha256'],name
    original={a['name']:a for a in old['assets']};manifest=json.loads((DEST/'package_manifest.json').read_text())
    for rec in manifest['packages']:
        if rec.get('unchanged_from')!='v1.0.0':continue
        assert original[rec['name']]['digest']=='sha256:'+rec['sha256'] and original[rec['name']]['size']==rec['bytes']
    report={'status':'passed','release':release['html_url'],'draft':release['draft'],'new_assets_verified':len(expected),'unchanged_prior_weight_archives':4,'main_commit':commit['sha'],'journal_submitted':False,'new_revision_author_approval':'required before journal submission'}
    (DEST/'GITHUB_PUBLICATION_VERIFICATION.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
    if not release['draft']:
        metadata=json.loads((DEST/'PORTAL_METADATA.json').read_text());metadata['public_deposit_performed']=True
        (DEST/'PORTAL_METADATA.json').write_text(json.dumps(metadata,indent=2))
        (ROOT/'CURRENT_SUBMISSION.md').write_text(f'''# Current evidence revision

Use **output/submission-evidence-2026-10-03/** for the revised SN Computer Science manuscript.

Main PDF: crowd-supervision-evidence-revision.pdf.
Editable source: source/main.tex or Manuscript-source.zip.
Supplement: Supplementary-diagnostics.pdf.
Public release: {release['html_url']}
Verified commit: {commit['sha']}

All eight new assets and four preserved prior weight archives have verified server-reported SHA256 digests and sizes. The v1.0.0 release is preserved. Thirty inference weights cover all fifteen runs at selected/fixed-epoch states. No original face images or credentials are published.

Final approval of this later manuscript/results is needed from all four authors before actual journal submission. Prior approval covered v1.0.0. Scientific scope remains one reused image source, one backbone and three paired seeds; no assurance of acceptance is made. Truthful ethics and data-provenance disclosures remain unchanged. No journal submission or TechRxiv update has occurred.
''')
if __name__=='__main__':main()
