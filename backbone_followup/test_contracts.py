import pytest
import sys
from pathlib import Path
import numpy as np
from emotion_cue.checkpoint import validate_checkpoint_metadata
from emotion_cue.constants import CLASS_NAMES
def test_runtime_rejects_research_backbone_checkpoint():
    checkpoint={'state_dict':{},'dataset':'FER2013','variant':'resnet18_gated','class_names':CLASS_NAMES}
    with pytest.raises(ValueError,match='Unknown checkpoint variant'):
        validate_checkpoint_metadata(checkpoint)
def test_independent_f1_matches_reference():
    sys.path.insert(0,str(Path(__file__).resolve().parent))
    from audit import f1
    from sklearn.metrics import f1_score
    y=np.array([0,0,2,2,4,6]);p=np.array([0,2,2,4,4,6])
    weighted,macro=f1(y,p)
    assert np.isclose(weighted,f1_score(y,p,labels=range(7),average='weighted',zero_division=0))
    assert np.isclose(macro,f1_score(y,p,labels=range(7),average='macro',zero_division=0))
def test_expanded_cluster_bootstrap_identical_models_zero():
    sys.path.insert(0,str(Path(__file__).resolve().parent))
    from audit import bootstrap_explicit
    q=np.zeros((4,10));q[:,0]=[.8,.6,.4,.9];q[:,1]=1-q[:,0]
    a={'rows':np.arange(4),'hashes':np.array(['a','a','b','c']),'q':q,'probabilities':np.tile([.7,.3,0,0,0,0,0],(4,1))}
    arrays={(c,s,'selected','test_full'):a for c in ('soft','uniform') for s in (17,42,89)}
    assert np.array_equal(bootstrap_explicit(arrays,'selected','uniform',100),[0,0])
