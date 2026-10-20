"""Outcome-independent exact/canonical and perceptual-hash screening."""
import json
import hashlib
from pathlib import Path
import cv2
import numpy as np
from PIL import Image,ImageDraw
from common import HERE,PAPER,sha,require_frozen
from prepare import phash


def main():
    require_frozen();images=[];source_rows=[];source_hashes=[];source_files={}
    for partition in ('Training','PublicTest','PrivateTest'):
        path=PAPER/f'data/fer2013/{partition}.npz';source_files[str(path.relative_to(PAPER))]=sha(path)
        with np.load(path,allow_pickle=False) as f:
            images.extend(f['images']);source_rows.extend(f['rows']);source_hashes.extend(f['hashes'])
    images=np.array(images);source_rows=np.array(source_rows);source_hashes=np.array(source_hashes)
    assert len(images)==35887
    hashes={str(h):i for i,h in enumerate(source_hashes)}
    for i in range(len(images)):
        assert hashlib.sha256(images[i].tobytes()).hexdigest()==str(source_hashes[i])
    source_phash=np.array([phash(Image.fromarray(image)) for image in images],dtype=np.uint64)
    frames=[]
    for actor in range(1,25):
        report=json.loads((HERE/f'cache/Actor_{actor:02d}.json').read_text())
        frames.extend([dict(frame,actor=actor) for frame in report['frames'] if frame['status']=='valid'])
    queries=np.array([frame['crop_phash63'] for frame in frames],dtype=np.uint64)
    table=np.array([int(i).bit_count() for i in range(256)],dtype=np.uint8)
    matches=[];minimum=[];review=[]
    for start in range(0,len(queries),32):
        xor=np.ascontiguousarray(queries[start:start+32,None]^source_phash[None,:])
        distance=table[xor.view(np.uint8).reshape(*xor.shape,8)].sum(-1)
        best=distance.argmin(1);score=distance[np.arange(len(best)),best]
        for j,(source_index,d) in enumerate(zip(best,score)):
            frame=frames[start+j];minimum.append(int(d))
            exact_index=hashes.get(frame['crop48_sha256'])
            if exact_index is not None:
                matches.append(dict(filename=frame['filename'],frame_index=frame['frame_index'],source_row=int(source_rows[exact_index])))
            if d<=4:
                review.append(dict(filename=frame['filename'],actor=frame['actor'],frame_index=frame['frame_index'],
                                   cache_index=frame['cache_index'],source_index=int(source_index),source_row=int(source_rows[source_index]),distance=int(d)))
        if start%1024==0:print(f'OVERLAP SCREEN {start}/{len(queries)} valid crops',flush=True)
    output=dict(status='completed',model_outcomes_used=False,original_source_images=len(images),
                screened_valid_frames=len(frames),exact_canonical_matches=matches,perceptual_flags=review,
                perceptual_flag_count=len(review),threshold_hamming63=4,
                minimum_distance=min(minimum),median_nearest_distance=float(np.median(minimum)),
                source_files=source_files,method='48x48 canonical grayscale crop SHA; 63 AC DCT bits on 32x32 gray, median threshold',
                identity_disjointness_verified=False,
                interpretation='Screening cannot establish person-disjointness or absence of all near duplicates. Flags require visual review; no automatic exclusion.')
    (HERE/'overlap_audit.json').write_text(json.dumps(output,indent=2),encoding='utf-8')
    if review:
        # Only a fixed first 20 flags, sorted without model outcomes, are shown for internal QA.
        chosen=sorted(review,key=lambda r:(r['distance'],r['filename'],r['frame_index']))[:20]
        canvas=Image.new('RGB',(640,len(chosen)*128),'white');draw=ImageDraw.Draw(canvas)
        actor_cache={}
        for row,item in enumerate(chosen):
            actor=item['actor']
            if actor not in actor_cache:
                with np.load(HERE/f'cache/Actor_{actor:02d}.npz',allow_pickle=False) as f:actor_cache[actor]=f['images']
            crop=Image.fromarray(actor_cache[actor][item['cache_index']]).resize((112,112))
            source=Image.fromarray(images[item['source_index']]).resize((112,112))
            y=row*128;canvas.paste(crop,(0,y));canvas.paste(source,(120,y))
            draw.text((240,y+10),item['filename'],fill='black')
            draw.text((240,y+32),f"frame {item['frame_index']}, FER row {item['source_row']}, distance {item['distance']}",fill='black')
        canvas.save(HERE/'overlap_review.png')
    print(json.dumps({k:v for k,v in output.items() if k not in ('source_files','perceptual_flags')},indent=2))


if __name__=='__main__':main()
