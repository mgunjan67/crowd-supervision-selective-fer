"""Rerun any target condition into a new directory, preserving released evidence."""
import argparse,shutil
from pathlib import Path
from train_controls import base,loss,verify_freeze
OUT=Path(__file__).resolve().parent;ROOT=OUT.parent

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--condition',choices=['hard','soft','uniform','tie_hard'],required=True)
    parser.add_argument('--seed',type=int,choices=[17,42,89],required=True)
    parser.add_argument('--output-dir',type=Path,required=True);args=parser.parse_args()
    verify_freeze();destination=args.output_dir.resolve()
    if not destination.is_relative_to(ROOT) or destination in (ROOT,OUT,ROOT/'final_experiment'):
        raise ValueError('Use a new child directory under paper_revision, outside the released experiment directories')
    if destination.is_relative_to(OUT) or destination.is_relative_to(ROOT/'final_experiment'):
        raise ValueError('Do not write reruns inside retained evidence')
    destination.mkdir(parents=True,exist_ok=True)
    for name in ('PROTOCOL.md','data.py','split_manifest.json'):
        target=destination/name
        if target.exists():assert base.data.sha(target)==base.data.sha(OUT/name)
        else:shutil.copy2(OUT/name,target)
    base.OUT=destination
    if args.condition in ('uniform','tie_hard'):base.supervised_loss=loss
    base.run(args.condition,args.seed)

if __name__=='__main__':main()
