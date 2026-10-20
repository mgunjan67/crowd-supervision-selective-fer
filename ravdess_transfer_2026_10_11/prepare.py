"""Extract predetermined frames with the released single-face detector/crop rule."""
import argparse
import concurrent.futures
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from PIL import Image,ImageDraw,ImageOps

from common import HERE,PAPER,annotations,sha,require_frozen
sys.path.insert(0,str(PAPER/'code'))
from emotion_cue.runtime import clipped_box


def phash(image):
    small=np.asarray(image.convert('L').resize((32,32),Image.Resampling.BILINEAR),dtype=np.float32)
    values=cv2.dct(small)[:8,:8].reshape(-1)[1:]
    bits=values>np.median(values)
    value=0
    for bit in bits:value=(value<<1)|int(bit)
    return value


def sample_indices(n):
    if n<8:raise ValueError('At least eight decoded frames required')
    return [min(n-1,int(math.floor((b+.5)*n/8))) for b in range(8)]


def prepare_actor(actor):
    started=time.monotonic();frozen=require_frozen()
    names,actors,_=annotations();indices=np.flatnonzero(actors==actor)
    directory=HERE/'cache';directory.mkdir(exist_ok=True)
    output=directory/f'Actor_{actor:02d}.npz';report_path=directory/f'Actor_{actor:02d}.json'
    if output.exists() and report_path.exists():
        report=json.loads(report_path.read_text());assert sha(output)==report['cache_sha256']
        assert report['freeze_sha256']==sha(HERE/'freeze.json')
        print(f'CACHED actor {actor:02d}',flush=True);return report
    integrity=json.loads((HERE/f'media/Actor_{actor:02d}_integrity.json').read_text())
    sources={r['filename']:r for r in integrity['clips']}
    images=[];clip_ids=[];frame_ids=[];records=[];clips=[];qc=[]
    # Read only the raw clips and detector results; no response counts or model scores.
    with mp.solutions.face_detection.FaceDetection(model_selection=1,min_detection_confidence=.5) as detector:
        for position,clip_id in enumerate(indices):
            filename=names[clip_id];entry=sources[str(filename)]
            path=HERE/entry['path'];assert sha(path)==entry['sha256']
            cap=cv2.VideoCapture(str(path))
            n=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));fps=float(cap.get(cv2.CAP_PROP_FPS))
            valid_count=0;frame_status='valid'
            if not cap.isOpened() or n<8 or not math.isfinite(fps) or fps<=0:
                frame_status='invalid_video';wanted=[]
            else:wanted=sample_indices(n)
            for bin_id,frame_id in enumerate(wanted):
                item=dict(clip_id=int(clip_id),filename=str(filename),actor=actor,
                          bin_id=bin_id,frame_index=frame_id,status='decode_failure')
                cap.set(cv2.CAP_PROP_POS_FRAMES,frame_id)
                ok,bgr=cap.read()
                if ok and bgr is not None:
                    item['decoded_frame_index']=int(round(cap.get(cv2.CAP_PROP_POS_FRAMES)))-1
                    assert item['decoded_frame_index']==frame_id,'Decoder did not return the requested frame'
                    rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)
                    height,width=rgb.shape[:2]
                    item.update(width=width,height=height)
                    detections=detector.process(rgb).detections or []
                    item['detections']=len(detections)
                    item['status']='no_face' if not detections else 'multiple_faces' if len(detections)>1 else 'invalid_crop'
                    if len(detections)==1:
                        box=clipped_box(detections[0].location_data.relative_bounding_box,width,height)
                        item['detection_score']=float(detections[0].score[0])
                        if box is not None:
                            left,top,right,bottom=box
                            crop=Image.fromarray(rgb[top:bottom,left:right]).convert('L')
                            resized=crop.resize((224,224),Image.Resampling.BILINEAR)
                            item.update(status='valid',box=box,cache_index=len(images),
                                        crop48_sha256=hashlib.sha256(crop.resize((48,48),Image.Resampling.BILINEAR).tobytes()).hexdigest(),
                                        crop_phash63=phash(crop))
                            images.append(np.asarray(resized,dtype=np.uint8));clip_ids.append(clip_id);frame_ids.append(frame_id)
                            valid_count+=1
                            if position in (0,10,20,30,40,50) and bin_id==4:
                                preview=Image.fromarray(rgb).resize((384,216))
                                draw=ImageDraw.Draw(preview)
                                draw.rectangle((left/width*384,top/height*216,right/width*384,bottom/height*216),outline='#ffcc00',width=2)
                                qc.append((str(filename),preview))
                records.append(item)
            cap.release()
            clips.append(dict(clip_id=int(clip_id),filename=str(filename),actor=actor,frame_count=n,fps=fps,
                              valid_frames=valid_count,eligible=valid_count>=4,
                              status='eligible' if valid_count>=4 else frame_status if frame_status!='valid' else 'insufficient_valid_frames',
                              sha256=entry['sha256']))
            if (position+1)%10==0:print(f'PREPARED actor {actor:02d}: {position+1}/60 clips',flush=True)
    array=np.stack(images) if images else np.empty((0,224,224),dtype=np.uint8)
    np.savez_compressed(output,images=array,clip_ids=np.array(clip_ids,dtype=np.int32),frame_indices=np.array(frame_ids,dtype=np.int32))
    report=dict(actor=actor,freeze_sha256=sha(HERE/'freeze.json'),source_integrity_sha256=sha(HERE/f'media/Actor_{actor:02d}_integrity.json'),
                cache_sha256=sha(output),elapsed_seconds=time.monotonic()-started,
                clips=clips,frames=records,eligible_clips=sum(c['eligible'] for c in clips),valid_frames=len(images))
    report_path.write_text(json.dumps(report,indent=2),encoding='utf-8')
    if qc:
        canvas=Image.new('RGB',(768,3*246),'white');draw=ImageDraw.Draw(canvas)
        for i,(filename,picture) in enumerate(qc):
            x,y=i%2*384,i//2*246;canvas.paste(picture,(x,y));draw.text((x+3,y+218),filename,fill='black')
        canvas.save(directory/f'Actor_{actor:02d}_QC.png')
    print(f"PREPARATION DONE actor {actor:02d}: {report['eligible_clips']}/60 eligible, {len(images)} valid frames, {report['elapsed_seconds']:.1f}s",flush=True)
    return report


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--actors',type=int,nargs='*',default=[])
    parser.add_argument('--all',action='store_true');parser.add_argument('--workers',type=int,default=1)
    args=parser.parse_args();actors=list(range(1,25)) if args.all else args.actors
    assert actors and len(set(actors))==len(actors) and all(1<=a<=24 for a in actors)
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        reports=list(pool.map(prepare_actor,actors))
    print(json.dumps(dict(actors=actors,eligible_clips=sum(r['eligible_clips'] for r in reports),
                          valid_frames=sum(r['valid_frames'] for r in reports),new_training_runs=0),indent=2))


if __name__=='__main__':main()
