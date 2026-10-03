"""Reload public inference checkpoints without archived optimizer states."""
import argparse,json
from pathlib import Path
import numpy as np,torch
from torch.utils.data import DataLoader
import train as base
OUT=Path(__file__).resolve().parent
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output-dir',type=Path,required=True);parser.add_argument('--device',choices=['cpu','cuda'],default='cuda');args=parser.parse_args()
    assert not args.output_dir.exists(),'Use a fresh output directory'
    args.output_dir.mkdir(parents=True);device=torch.device(args.device)
    for path in sorted((OUT/'weights').glob('*.pt')):
        ck=torch.load(path,weights_only=True,map_location='cpu');seed=ck['seed'];base.seed_all(seed)
        m=base.build_model(pretrained=False).to(device);m.load_state_dict(ck['state_dict']);folder=args.output_dir/path.stem;folder.mkdir()
        for role in ('selection','calibration','test_full'):
            ds=base.Images(role,seed);loader=DataLoader(ds,batch_size=32,shuffle=False,num_workers=0,pin_memory=device.type=='cuda')
            p=base.predict(m,loader,device);a=ds.arrays
            np.savez_compressed(folder/f'{role}.npz',probabilities=p,rows=a['rows'],labels=a['labels'],q=a['q'],hashes=a['hashes'])
        print('Reproduced '+path.stem,flush=True)
if __name__=='__main__':main()
