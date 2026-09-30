from types import SimpleNamespace as NS
import numpy as np
import torch
import pytest
from emotion_cue.constants import CLASS_NAMES
from emotion_cue.runtime import Cue, FramePredictor, clipped_box, attached_category
from emotion_cue.metrics import classification_metrics
from emotion_cue.data_contract import deterministic_split_indices

def box(x=0,y=0,w=1,h=1):
    return NS(xmin=x,ymin=y,width=w,height=h)

def test_partial_box_clips_without_shifting_right_edge():
    assert clipped_box(box(-.2,-.1,.5,.5),100,100)==(0,0,30,40)

@pytest.mark.parametrize('b',[box(2),box(w=-1),box(x=float('nan'))])
def test_invalid_box_rejected(b):
    assert clipped_box(b,100,100) is None

@pytest.mark.parametrize('opted,active,now,status',[
    (False,True,10,'valid'),(True,False,10,'valid'),(True,True,13,'valid'),
    (True,True,9,'valid'),(True,True,10,'no_face'),(True,True,10,'multiple_faces')])
def test_invalid_or_unapproved_cues_cannot_attach(opted,active,now,status):
    assert attached_category(Cue('happy',.8,10,status),opted_in=opted,camera_active=active,now=now) is None

def test_current_approved_cue_can_attach():
    assert attached_category(Cue('happy',.8,10,'valid'),opted_in=True,camera_active=True,now=11)=='happy'

class Model(torch.nn.Module):
    def forward(self,x):
        assert x.shape==(1,3,224,224)
        assert torch.equal(x[:,0],x[:,1])
        return torch.tensor([[0.,0.,0.,3.,0.,0.,0.]])

def predictor(detections):
    return FramePredictor(Model(),NS(process=lambda image:NS(detections=detections)),torch.device('cpu'),CLASS_NAMES)

def test_frame_prediction_uses_shared_preprocessing():
    det=NS(location_data=NS(relative_bounding_box=box()))
    cue=predictor([det]).predict(np.zeros((48,48,3),np.uint8),now=10)
    assert cue.category=='happy' and cue.status=='valid'

def test_missing_or_multiple_faces_are_not_neutral():
    image=np.zeros((48,48,3),np.uint8)
    assert predictor([]).predict(image).category is None
    assert predictor([None,None]).predict(image).status=='multiple_faces'

def test_macro_f1_includes_absent_configured_classes():
    metrics=classification_metrics([0,0],[0,0],CLASS_NAMES)
    assert metrics['macro_f1']==pytest.approx(1/7)
    assert metrics['weighted_f1']==1

def test_extreme_validation_fraction_keeps_training_nonempty():
    training,validation=deterministic_split_indices(2,.99,17)
    assert len(training)==len(validation)==1
