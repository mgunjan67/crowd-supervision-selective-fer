"""Record actual source checks without starting inference or loading a model."""
import importlib.util
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
TRAIN=HERE.parent/'training'
PAPER=HERE.parents[1]
spec=importlib.util.spec_from_file_location('runtime_contracts',TRAIN/'contracts_followup.py')
contracts=importlib.util.module_from_spec(spec);spec.loader.exec_module(contracts)


def main():
    verified=contracts.validate_static_sources(TRAIN,PAPER)
    runs=verified['original']['runs']
    completions=[TRAIN/'runs'/f'k{k}_{condition}_seed{seed}'/'training_complete.json' for k,condition,seed in runs]
    report=dict(status='sources_verified',gpu_used=False,new_inference=False,
                original_frozen_sources=len(verified['original']['source_sha256']),
                original_data_source_records=len(verified['split']['source_hashes']),
                amendment_001_sources=len(verified['first_amendment']['source_sha256']),
                amendment_002_sources=len(verified['second_amendment']['source_sha256']),
                original_freeze_sha256=verified['original_freeze_sha256'],
                amendment_002_freeze_sha256=verified['second_amendment_sha256'],
                declared_training_runs=len(runs),planned_inference_states=33,
                completed_training_runs_at_audit=sum(path.exists() for path in completions),
                note='Actual final state validity is checked only after all twelve training runs; no inference is permitted before completeness.')
    (HERE/'runtime_source_audit.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
