"""Verify existing checkpoint ZIP bytes and every member; never duplicate weights."""
import csv
import hashlib
import json
import zipfile
from pathlib import Path

HERE=Path(__file__).resolve().parent; PAPER=HERE.parent
EXPECTED={
2:'8cfdaa686e626761f02451715d02246518f2607ae24cf380307c123504683a13',
3:'b743e11bb7c6746ab4c550bbaddbdba18464f4628b0ad399f0d3214f351b354c',
4:'d4f81c40cb81434577fb4feb90df3cdfc1ee96818c485fdf74e3784472ba4a52',
5:'ab63d8af45b6c7c40be40994fea79783f68edd959b9ed9423472547b023e9d80',
6:'06161c88eda77a509feab9c08dc80c36aef5f821b9ae62852d298396481a2cbc',
7:'3220efc92a2edabbb0814f0c680f695b41742420ea24ff6ef1f4f6afcd30a641'}
def digest(stream):
    h=hashlib.sha256()
    for block in iter(lambda:stream.read(4194304),b''):h.update(block)
    return h.hexdigest()
def sha(path):
    with path.open('rb') as stream:return digest(stream)
def main():
    wanted={}
    for folder in ('reviewer_followup','evidence_revision','backbone_followup'):
        for p in sorted((PAPER/folder/'predictions').glob('*/evaluation.json')):
            r=json.loads(p.read_text());member=f"{folder}/weights/{r['condition']}_seed{r['seed']}_{r['state']}.pt"
            wanted[member]={'backbone':'resnet18' if folder=='backbone_followup' else 'swin_t','condition':r['condition'],'seed':r['seed'],'state':r['state'],'selected_epoch':r['epoch'],'sha256':r['weight_sha256']}
    assert len(wanted)==48
    rows=[];archives=[]
    for n,expected in EXPECTED.items():
        path=PAPER/('output/submission-backbone-2026-10-03' if n==7 else 'output/submission-evidence-2026-10-03')/f'OnlineResource{n}.zip'
        assert sha(path)==expected,path
        with zipfile.ZipFile(path) as z:
            manifest=json.loads(z.read('SHA256_MANIFEST.json'));assert set(z.namelist())==set(manifest)|{'SHA256_MANIFEST.json'}
            for name,member_sha in manifest.items():
                assert not name.startswith('/') and '..' not in Path(name).parts
                with z.open(name) as stream:assert digest(stream)==member_sha,(path,name)
                if name.endswith('.pt'):
                    r=wanted.pop(name);assert r['sha256']==member_sha
                    # Actual local bytes were independently checked by the full audits;
                    # recheck here to join those bytes to the delivered archive member.
                    assert sha(PAPER/name)==member_sha
                    rows.append(dict(r,online_resource=n,archive=str(path.resolve()),archive_member=name,bytes=z.getinfo(name).file_size,availability='verified_existing_local_archive'))
        archives.append({'online_resource':n,'path':str(path.resolve()),'bytes':path.stat().st_size,'sha256':expected,'member_hashes':'passed'})
        print(f'Online Resource {n}: archive and all member hashes passed',flush=True)
    assert not wanted and len(rows)==48
    with (HERE/'CHECKPOINT_AVAILABILITY.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    report={'status':'passed','states':48,'training_runs':24,'weights_not_duplicated':True,'archives':archives}
    (HERE/'checks/checkpoint_archives.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
if __name__=='__main__':main()
