"""Authorized public RAVDESS speech acquisition; no audio/song or restricted data."""
import argparse
import concurrent.futures
import csv
import hashlib
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
PAPER = HERE.parent
MEDIA = HERE/'media'
API = 'https://zenodo.org/api/records/1188976'


def now():
    return datetime.now(timezone.utc).isoformat()


def digest(path, name='sha256'):
    h=hashlib.new(name)
    with path.open('rb') as f:
        while block:=f.read(4*1024*1024):
            h.update(block)
    return h.hexdigest()


def request(url, start=0):
    assert urllib.parse.urlparse(url).hostname == 'zenodo.org'
    headers={'User-Agent':'RAVDESS-research-transfer/1.0 (public academic data; no authentication)'}
    if start:
        headers['Range']=f'bytes={start}-'
    return urllib.request.urlopen(urllib.request.Request(url,headers=headers),timeout=60)


def prepare_manifest():
    HERE.mkdir(parents=True,exist_ok=True)
    path=HERE/'zenodo_record.json'
    if not path.exists():
        with request(API) as response:
            data=response.read(4_000_001)
        if len(data)>4_000_000:
            raise ValueError('Record metadata exceeded expected bounded size')
        parsed=json.loads(data)
        assert int(parsed['id'])==1188976
        path.write_bytes(data)
    record=json.loads(path.read_bytes())
    files={f['key']:f for f in record['files']}
    chosen=[]
    for actor in range(1,25):
        filename=f'Video_Speech_Actor_{actor:02d}.zip'
        item=files[filename]
        assert item['checksum'].startswith('md5:')
        chosen.append(dict(actor=actor,filename=filename,bytes=int(item['size']),
                           md5=item['checksum'][4:],
                           url=f'https://zenodo.org/records/1188976/files/{filename}?download=1'))
    total=sum(f['bytes'] for f in chosen)
    assert 12_000_000_000 < total < 14_000_000_000
    output=dict(record_url='https://zenodo.org/records/1188976',metadata_sha256=digest(path),
                checked_utc=now(),client_date='2026-10-11',license=record['metadata'].get('license'),
                total_download_bytes=total,actor_archives=chosen,
                authorization='User approved the proposed frozen-checkpoint evaluation, public download and up to 30 GB temporary storage.',
                selection='All speech actor archives; extract video-only speech, never audio-only or song')
    manifest=HERE/'download_manifest.json'
    if manifest.exists():
        prior=json.loads(manifest.read_text())
        assert prior['metadata_sha256']==output['metadata_sha256'] and prior['actor_archives']==chosen
    else:
        manifest.write_text(json.dumps(output,indent=2),encoding='utf-8')
    return output


def download(item):
    target=MEDIA/'archives'/item['filename']
    target.parent.mkdir(parents=True,exist_ok=True)
    if target.exists():
        assert target.stat().st_size==item['bytes'] and digest(target,'md5')==item['md5']
        return target
    partial=target.with_suffix('.zip.partial')
    for attempt in range(4):
        start=partial.stat().st_size if partial.exists() else 0
        assert start<=item['bytes']
        if start==item['bytes']:
            break
        try:
            with request(item['url'],start) as response:
                resumed=start and response.status==206
                if resumed:
                    assert response.headers['Content-Range'].startswith(f'bytes {start}-')
                else:
                    start=0
                mode='ab' if resumed else 'wb'
                last=time.monotonic()
                with partial.open(mode) as f:
                    while block:=response.read(1024*1024):
                        f.write(block)
                        if f.tell()>item['bytes']:
                            raise ValueError('Download exceeds recorded archive size')
                        if time.monotonic()-last>=15:
                            print(f"DOWNLOAD actor {item['actor']:02d}: {f.tell()}/{item['bytes']} bytes",flush=True)
                            last=time.monotonic()
            break
        except (urllib.error.URLError,TimeoutError,ConnectionError,OSError) as e:
            print(f"RETRY actor {item['actor']:02d}, attempt {attempt+1}: {type(e).__name__}",flush=True)
            if attempt==3:
                raise
    assert partial.stat().st_size==item['bytes']
    assert digest(partial,'md5')==item['md5'],'Publisher archive checksum mismatch; preserve partial for investigation'
    os.replace(partial,target)
    print('CHECKSUM OK '+target.name,flush=True)
    return target


def extract_video_only(archive,item):
    metadata=PAPER/'recovery_2026_10_10/external_data/ravdess/video_only_speech_votes.csv'
    with metadata.open(newline='') as f:
        wanted={r['filename'] for r in csv.DictReader(f) if int(r['actor'])==item['actor']}
    assert len(wanted)==60
    destination=(MEDIA/f"video_only_speech/Actor_{item['actor']:02d}").resolve()
    destination.mkdir(parents=True,exist_ok=True)
    records=[]
    with zipfile.ZipFile(archive) as z:
        entries={Path(info.filename).name:info for info in z.infolist()
                 if not info.is_dir() and Path(info.filename).name.startswith('02-01-') and info.filename.endswith('.mp4')}
        assert set(entries)==wanted,(len(entries),len(wanted))
        for filename in sorted(wanted):
            target=destination/filename
            assert target.resolve().is_relative_to(destination)
            info=entries[filename]
            if not target.exists():
                # Full reads verify the ZIP member CRC; only selected VO files written.
                data=z.read(info)
                assert len(data)==info.file_size
                target.write_bytes(data)
            assert target.stat().st_size==info.file_size
            records.append(dict(filename=filename,actor=item['actor'],bytes=info.file_size,
                                sha256=digest(target),zip_crc32=info.CRC,path=str(target.relative_to(HERE))))
    report=HERE/f"media/Actor_{item['actor']:02d}_integrity.json"
    value=dict(actor=item['actor'],archive_sha256=digest(archive),archive_md5=item['md5'],
               video_only_speech_clips=60,completed_utc=now(),clips=records)
    report.write_text(json.dumps(value,indent=2),encoding='utf-8')
    print(f"EXTRACTED actor {item['actor']:02d}: 60 VO speech clips",flush=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--actors',type=int,nargs='*',default=[])
    parser.add_argument('--all',action='store_true')
    parser.add_argument('--workers',type=int,default=1)
    args=parser.parse_args()
    record=prepare_manifest()
    actors=range(1,25) if args.all else args.actors
    assert all(1<=actor<=24 for actor in actors)
    print(f"MANIFEST: {record['total_download_bytes']} bytes, 24 speech archives",flush=True)
    chosen=[item for item in record['actor_archives'] if item['actor'] in actors]
    def acquire_one(item):
        archive=download(item)
        extract_video_only(archive,item)
    assert 1<=args.workers<=4
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        list(pool.map(acquire_one,chosen))


if __name__=='__main__':
    main()
