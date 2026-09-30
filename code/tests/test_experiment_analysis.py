"""Analytical checks do not substitute for the six experimental observations."""
import importlib.util
from pathlib import Path
import numpy as np
import pytest
import torch
from emotion_cue.experiment import FERPartition, make_loader, probability_metrics
from emotion_cue.constants import CLASS_NAMES

def test_probability_metrics_perfect_predictions():
    labels=np.arange(7); probabilities=np.eye(7,dtype=np.float32)
    metrics=probability_metrics(labels,probabilities)
    assert metrics['accuracy']==metrics['macro_f1']==metrics['weighted_f1']==1
    assert metrics['brier_score']==metrics['ece_15_bins']==0
    assert metrics['confusion_matrix']==np.eye(7,dtype=int).tolist()

def test_probability_metrics_uniform_scores():
    labels=np.arange(7); probabilities=np.full((7,7),1/7,dtype=np.float64)
    metrics=probability_metrics(labels,probabilities)
    assert metrics['accuracy']==pytest.approx(1/7)
    assert metrics['brier_score']==pytest.approx(6/7)
    assert metrics['ece_15_bins']==pytest.approx(0)

def test_official_partition_rows_are_preserved(tmp_path):
    images=np.stack([np.full((48,48),i*20,dtype=np.uint8) for i in range(7)])
    np.savez(tmp_path/'PublicTest.npz',images=images,labels=np.arange(7),rows=np.arange(100,107),hashes=np.array([str(i) for i in range(7)]))
    data=FERPartition(tmp_path,'PublicTest')
    assert data.rows.tolist()==list(range(100,107))
    image,label=data[3]
    assert image.shape==(3,224,224) and label==3
    assert torch.equal(image[0],image[1]) and torch.equal(image[1],image[2])
    assert float(image.mean())==pytest.approx(2*60/255-1,abs=1e-6)

def test_loader_shuffle_is_seeded():
    dataset=torch.utils.data.TensorDataset(torch.arange(30),torch.arange(30))
    def order(seed):
        return torch.cat([y for _,y in make_loader(dataset,4,seed,0,True)]).tolist()
    assert order(17)==order(17)
    assert order(17)!=order(42)

def test_evaluation_cli_contract():
    path=Path(__file__).parents[1]/'evaluate.py'
    spec=importlib.util.spec_from_file_location('paper_evaluate',path)
    module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    args=module.parse_args(['--checkpoint','model.pt','--dataset','FER2013','--output-json','metrics.json'])
    assert args.batch_size==32 and args.dataset=='FER2013'
    with pytest.raises(SystemExit): module.parse_args(['--checkpoint','model.pt','--dataset','RAF_DB_BASIC','--output-json','x.json'])

def test_noncanonical_checkpoint_class_order_rejected():
    from emotion_cue.checkpoint import validate_checkpoint_metadata
    cp={'state_dict':{},'dataset':'FER2013','variant':'swin','class_names':list(reversed(CLASS_NAMES))}
    with pytest.raises(ValueError,match='class order'): validate_checkpoint_metadata(cp)
