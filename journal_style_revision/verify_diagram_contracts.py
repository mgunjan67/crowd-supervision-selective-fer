"""Check figure architecture against the actual release models; no training/downloads."""
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
import torch
from torch import nn

OUT=Path(__file__).resolve().parent
ROOT=OUT.parent
sys.path.insert(0,str(ROOT/'code'))
from emotion_cue.model import build_model


def main():
    torch.set_num_threads(2)
    torch.manual_seed(17)
    swin=build_model('gated',7,pretrained=False).eval()
    spec=importlib.util.spec_from_file_location('figure_resnet_model',ROOT/'backbone_followup/model.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    resnet=module.build_model(pretrained=False).eval()
    shapes={}
    hooks=[]
    for i in (1,3,5,7):
        hooks.append(swin.backbone.features[i].register_forward_hook(
            lambda module,args,output,i=i:shapes.update({str(i):list(output.shape)})))
    with torch.inference_mode():
        x=torch.zeros(1,3,224,224)
        assert swin(x).shape==resnet(x).shape==(1,7)
    for hook in hooks:hook.remove()
    assert list(shapes.values())==[[1,56,56,96],[1,28,28,192],[1,14,14,384],[1,7,7,768]]
    assert [len(swin.backbone.features[i]) for i in (1,3,5,7)]==[2,2,6,2]
    assert isinstance(swin.backbone.head,nn.Identity) and isinstance(resnet.backbone.fc,nn.Identity)
    for d,hidden,gate,classifier in ((768,48,swin.channel_gate.gate,swin.classifier),
                                     (512,32,resnet.gate,resnet.classifier)):
        assert [type(m) for m in gate]==[nn.Linear,nn.ReLU,nn.Linear,nn.Sigmoid]
        assert (gate[0].in_features,gate[0].out_features)==(d,hidden)
        assert (gate[2].in_features,gate[2].out_features)==(hidden,d)
        assert gate[0].bias is gate[2].bias is None
        assert (classifier.in_features,classifier.out_features)==(d,7)
        assert classifier.bias is not None
    ledger=hashlib.sha256((ROOT/'backbone_followup/reported_results.csv').read_bytes()).hexdigest()
    assert ledger=='200a9b1df63cb267169a9f888b64a7fda3993f241494fff2e33320817bbfec3b'
    report={'status':'passed','swin_stage_shapes':shapes,'swin_stage_depths':[2,2,6,2],
            'both_logits_shapes':[1,7],'single_post_pooling_gate':True,
            'bias_contract_checked':True,'pretrained_weights_downloaded':False,
            'new_training_runs':0,'ledger_sha256':ledger,
            'source_hashes':{str(p.relative_to(ROOT)).replace('\\','/'):
                hashlib.sha256(p.read_bytes()).hexdigest() for p in
                (ROOT/'code/emotion_cue/model.py',ROOT/'backbone_followup/model.py')}}
    (OUT/'diagram_contract_audit.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
