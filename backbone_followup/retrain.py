"""Fresh output prevents archived configurations being overwritten."""
import argparse,shutil
from pathlib import Path
import train as base
OUT=Path(__file__).resolve().parent
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output-dir',type=Path,required=True);parser.add_argument('--condition',choices=['soft','uniform','tie_uniform'],required=True);parser.add_argument('--seed',type=int,required=True);args=parser.parse_args()
    assert not args.output_dir.exists(),'Use a fresh output directory'
    args.output_dir=args.output_dir.resolve()
    assert args.output_dir.is_relative_to(OUT.parent),'Fresh output must stay inside paper_revision for immutable source-relative paths'
    args.output_dir.mkdir(parents=True)
    for name in ('PROTOCOL.md','data.py','split_manifest.json','model.py','driver.py'):shutil.copy2(OUT/name,args.output_dir/name)
    # Files in the fresh root satisfy the retained run-source checks. Data keeps
    # the archive's immutable roles and separate acquisition location.
    base.OUT=args.output_dir
    base.run(args.condition,args.seed)
if __name__=='__main__':main()
