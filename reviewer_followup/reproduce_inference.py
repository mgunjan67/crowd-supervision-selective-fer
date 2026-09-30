"""Verify released inference weights against all retained probabilities (images required)."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from train_controls import base
from emotion_cue.checkpoint import validate_checkpoint_metadata
OUT=Path(__file__).resolve().parent

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--require-bitwise',action='store_true')
    parser.add_argument('--output-json',type=Path,required=True);args=parser.parse_args()
    if args.output_json.exists():raise RuntimeError('Choose a new report path; retained evidence is not overwritten')
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu');reports=[]
    for c in ('hard','soft','uniform','tie_hard'):
        for s in (17,42,89):
            for state in ('selected','epoch12'):
                base.seed_all(s);name=f'{c}_seed{s}_{state}';path=OUT/'weights'/f'{name}.pt'
                cp=torch.load(path,weights_only=True,map_location='cpu');validate_checkpoint_metadata(cp,expected_dataset='FER2013')
                model=base.build_model('gated',7,pretrained=False).to(device);model.load_state_dict(cp['state_dict'])
                for role in ('selection','calibration','test_full'):
                    dataset=base.Images(role,s);loader=DataLoader(dataset,batch_size=32,shuffle=False,num_workers=0,pin_memory=device.type=='cuda')
                    p=base.predict(model,loader,device)
                    with np.load(OUT/'predictions'/name/f'{role}.npz',allow_pickle=False) as f:
                        np.testing.assert_array_equal(f['rows'],dataset.arrays['rows']);reference=f['probabilities']
                    exact=np.array_equal(reference,p)
                    if args.require_bitwise:assert exact,name+'/'+role
                    reports.append({'model':name,'role':role,'weight_sha256':base.data.sha(path),'bitwise_equal':exact,
                        'maximum_absolute_probability_difference':float(np.abs(reference-p).max())})
                del model,cp
                print('REPLAYED '+name,flush=True)
    args.output_json.write_text(json.dumps({'device':str(device),'require_bitwise':args.require_bitwise,
        'all_bitwise_equal':all(r['bitwise_equal'] for r in reports),'comparisons':reports},indent=2),encoding='utf-8')

if __name__=='__main__':main()
