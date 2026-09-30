"""Independent original-CSV emotion-ID/pixel/row audit; requires the acquired image archive."""
import csv,hashlib,io,json,zipfile
from collections import Counter,defaultdict
from pathlib import Path
OUT=Path(__file__).resolve().parent;ROOT=OUT.parent

def digest_stream(stream):
    h=hashlib.sha256()
    for block in iter(lambda:stream.read(1048576),b''):h.update(block)
    return h.hexdigest()

def main():
    import numpy as np
    archive=ROOT/'data/raw/fer2013-kaggle-v1.zip'
    previous=json.loads((ROOT/'evidence/dataset_audit.json').read_text())
    with archive.open('rb') as f:archive_hash=digest_stream(f)
    assert archive_hash==previous['archive_sha256']
    with (ROOT/'evidence/dataset_manifest.csv').open(newline='') as f:manifest=list(csv.DictReader(f))
    with (ROOT/'data/ferplus_annotations/fer2013new.csv').open(newline='',encoding='utf-8-sig') as f:annotations=list(csv.DictReader(f))
    original_order=('angry','disgust','fear','happy','sad','surprise','neutral')
    canonical_order=('angry','disgust','fear','happy','neutral','sad','surprise')
    counts=defaultdict(Counter);n=0
    with zipfile.ZipFile(archive) as z:
        candidates=[name for name in z.namelist() if name.lower().endswith('fer2013.csv')]
        assert len(candidates)==1
        with z.open(candidates[0]) as f:csv_hash=digest_stream(f)
        assert csv_hash==previous['csv_sha256']
        with z.open(candidates[0]) as stream:
            for i,row in enumerate(csv.DictReader(io.TextIOWrapper(stream,encoding='utf-8-sig',newline=''))):
                expected=manifest[i];label=original_order[int(row['emotion'])]
                pixels=np.fromstring(row['pixels'],sep=' ',dtype=np.int16)
                assert pixels.shape==(2304,) and np.all((pixels>=0)&(pixels<=255))
                assert int(expected['row'])==i and expected['partition']==row['Usage']==annotations[i]['Usage']
                assert expected['class']==label
                assert expected['pixel_sha256']==hashlib.sha256(pixels.astype(np.uint8).tobytes()).hexdigest()
                counts[row['Usage']][label]+=1;n+=1
    assert n==len(manifest)==len(annotations)==35887
    assert {k:dict(v) for k,v in counts.items()}==previous['counts']
    result={'status':'passed','rows_checked':n,'archive_sha256':archive_hash,'csv_sha256':csv_hash,
        'audit_code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'original_numeric_class_order':original_order,'canonical_class_order':canonical_order,
        'original_to_canonical_ids':[canonical_order.index(c) for c in original_order],
        'checks':['raw original emotion IDs','original pixel bytes','original row index','FER+ partition alignment','all class counts'],
        'scope':'Internal consistency with retained community-mirror archive, not independent proof of image authenticity or rights'}
    (OUT/'raw_source_audit.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result,indent=2))

if __name__=='__main__':main()
