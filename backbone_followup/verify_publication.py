import hashlib,json,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parent.parent;DEST=ROOT/'output/submission-backbone-2026-10-03';REPO='mgunjan67/crowd-supervision-selective-fer'
def gh(*args):return json.loads(subprocess.check_output(['gh',*args],text=True,encoding='utf-8'))
def main():
    releases=gh('api',f'repos/{REPO}/releases');release=next(r for r in releases if r['tag_name']=='v1.2.0')
    expected=json.loads((DEST/'release-artifacts-v1.2.0.json').read_text());p=DEST/'release-artifacts-v1.2.0.json';expected[p.name]={'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
    actual={a['name']:a for a in release['assets']};assert set(actual)==set(expected)
    for name,v in expected.items():assert actual[name]['state']=='uploaded' and actual[name]['size']==v['bytes'] and actual[name]['digest']=='sha256:'+v['sha256'],name
    manifest=json.loads((DEST/'package_manifest.json').read_text())
    for rec in manifest['packages']:
        if not rec.get('unchanged_from'):continue
        old=next(r for r in releases if r['tag_name']==rec['unchanged_from']);asset=next(a for a in old['assets'] if a['name']==rec['name'])
        assert asset['digest']=='sha256:'+rec['sha256'] and asset['size']==rec['bytes']
    commit=gh('api',f'repos/{REPO}/commits/main')['sha'];assert release['target_commitish']==commit
    if not release['draft']:assert gh('api',f'repos/{REPO}/commits/v1.2.0')['sha']==commit
    report={'status':'passed','release':f'https://github.com/{REPO}/releases/tag/v1.2.0','draft':release['draft'],'new_assets_verified':8,'unchanged_prior_weight_archives':5,'main_commit':commit,'journal_submitted':False}
    (DEST/'GITHUB_PUBLICATION_VERIFICATION.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
    if not release['draft']:
        metadata=json.loads((DEST/'PORTAL_METADATA.json').read_text());metadata['public_deposit_performed']=True;(DEST/'PORTAL_METADATA.json').write_text(json.dumps(metadata,indent=2))
        (ROOT/'CURRENT_SUBMISSION.md').write_text(f'''# Current two-backbone evidence revision

Use output/submission-backbone-2026-10-03/ for SN Computer Science. Main PDF: crowd-supervision-backbone-revision.pdf. Supplement: Supplementary-diagnostics.pdf. Source: source/main.tex and Manuscript-source.zip.

Public release: {report['release']}
Verified commit: {commit}

24 training runs, two backbone families, 48 selected/fixed-epoch inference weights. All eight new assets and five unchanged prior weight archives are verified by server SHA256/size. Old releases remain unchanged. No original images or credentials are published. The app retains its Swin inference contract.

User confirmed all-author approval of v1.1.0 and authorized this extension; authors should review later v1.2.0 results before submission. One image source, adaptive reuse and ethics/data-provenance limits remain. No acceptance guarantee, journal submission or TechRxiv update is implied.
''')
if __name__=='__main__':main()
