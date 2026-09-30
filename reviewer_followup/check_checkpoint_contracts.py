"""Validate the 24 released weights against the unchanged app's model contract."""
import json
from pathlib import Path
import torch
from train_controls import base
from emotion_cue.checkpoint import validate_checkpoint_metadata
OUT=Path(__file__).resolve().parent

def main():
    reports=[]
    for path in sorted((OUT/'weights').glob('*.pt')):
        cp=torch.load(path,weights_only=True,map_location='cpu');classes=validate_checkpoint_metadata(cp,expected_dataset='FER2013')
        assert cp['variant']=='gated' and cp['preprocessing']=='grayscale3_resize224_bilinear_mean0.5_std0.5'
        assert cp['release_protocol_sha256']==base.data.sha(OUT/'PROTOCOL.md')
        state=cp['state_dict']
        assert {k for k in state if k.startswith('channel_gate.')}=={'channel_gate.gate.0.weight','channel_gate.gate.2.weight'}
        assert tuple(state['channel_gate.gate.0.weight'].shape)==(48,768)
        assert tuple(state['channel_gate.gate.2.weight'].shape)==(768,48)
        assert tuple(state['classifier.weight'].shape)==(7,768) and tuple(state['classifier.bias'].shape)==(7,)
        assert not any(k.startswith('backbone.head.') for k in state)
        assert all(torch.isfinite(value).all() for value in state.values())
        reports.append({'file':path.name,'sha256':base.data.sha(path),'class_names':list(classes),'epoch':cp['epoch']})
    assert len(reports)==24
    result={'status':'passed','checkpoints':reports,'app_code_sha256':base.data.sha(OUT.parent/'code/app.py'),
        'scope':'Actual weights match the unchanged app/model class order, preprocessing and single pooled-vector gate'}
    (OUT/'checkpoint_contract_audit.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result,indent=2))

if __name__=='__main__':main()
