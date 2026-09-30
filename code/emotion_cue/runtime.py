"""Frame inference and fail-closed cue attachment shared by app and tests."""
from dataclasses import dataclass
import math
import time
import torch
from PIL import Image
from .data import build_transform

MAX_CUE_AGE_SECONDS = 2.5

@dataclass(frozen=True)
class Cue:
    category: str | None = None
    score: float = 0.0
    observed_at: float = 0.0
    status: str = 'waiting'

def clipped_box(box, width, height):
    """Clip original corners; never shift a partially out-of-frame box."""
    if not all(math.isfinite(v) for v in (box.xmin,box.ymin,box.width,box.height)) or box.width<=0 or box.height<=0:
        return None
    left=max(0,int(box.xmin*width)); top=max(0,int(box.ymin*height))
    right=min(width,int((box.xmin+box.width)*width)); bottom=min(height,int((box.ymin+box.height)*height))
    return (left,top,right,bottom) if right>left and bottom>top else None

def attached_category(cue, *, opted_in, camera_active, now):
    if not opted_in or not camera_active or cue.status!='valid' or not 0<=now-cue.observed_at<=MAX_CUE_AGE_SECONDS:
        return None
    return cue.category

class FramePredictor:
    def __init__(self, model, detector, device, class_names):
        self.model,self.detector,self.device,self.class_names=model,detector,device,class_names
        self.transform=build_transform('FER2013',runtime=True)

    @torch.inference_mode()
    def predict(self, frame_rgb, now=None):
        now=time.monotonic() if now is None else now
        detections=self.detector.process(frame_rgb).detections or []
        if len(detections)!=1:
            return Cue(observed_at=now,status='no_face' if not detections else 'multiple_faces')
        height,width=frame_rgb.shape[:2]
        box=clipped_box(detections[0].location_data.relative_bounding_box,width,height)
        if box is None: return Cue(observed_at=now,status='invalid_crop')
        left,top,right,bottom=box
        tensor=self.transform(Image.fromarray(frame_rgb[top:bottom,left:right])).unsqueeze(0).to(self.device)
        logits=self.model(tensor).float()
        if not torch.isfinite(logits).all(): return Cue(observed_at=now,status='invalid_output')
        probabilities=logits.softmax(1)[0]; index=int(probabilities.argmax())
        return Cue(self.class_names[index],float(probabilities[index]),now,'valid')
