"""Optional independent rerun into a fresh directory; never overwrite evidence."""
import argparse,json,shutil
from pathlib import Path
from train_tie_uniform import base,loss,verify_freeze
OUT=Path(__file__).resolve().parent;ROOT=OUT.parent
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--seed',type=int,choices=(17,42,89),required=True);parser.add_argument('--output-dir',type=Path,required=True);args=parser.parse_args()
    verify_freeze();target=args.output_dir.resolve()
    excluded=(OUT,ROOT/'final_experiment',ROOT/'reviewer_followup',ROOT/'code',ROOT/'data')
    if target==ROOT or not target.is_relative_to(ROOT) or any(target==p or target.is_relative_to(p) for p in excluded):raise ValueError('Choose a fresh child directory outside retained evidence, code and data')
    if target.exists() and any(target.iterdir()):raise ValueError('Rerun destination must be empty')
    target.mkdir(parents=True,exist_ok=True)
    for name in ('PROTOCOL.md','data.py','split_manifest.json'):shutil.copy2(OUT/name,target/name)
    (target/'rerun_provenance.json').write_text(json.dumps({'target_source_sha256':base.data.sha(OUT/'train_tie_uniform.py'),'retained_freeze_sha256':base.data.sha(OUT/'freeze.json'),'boundary':'Independent rerun; numerical identity depends on actual hardware and environment'},indent=2))
    base.OUT=target;base.supervised_loss=loss;base.run('tie_uniform',args.seed)
if __name__=='__main__':main()
