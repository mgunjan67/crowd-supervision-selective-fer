"""Record the single explicit pre-analysis semantics repair; never rewrite freeze."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

HERE=Path(__file__).resolve().parent
PAPER=HERE.parent


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    target=HERE/'amendment_001.json'
    assert not target.exists(),'An amendment is immutable once recorded'
    assert not (HERE/'results/clip_results.csv').exists()
    assert not (HERE/'inference_complete.json').exists()
    original=json.loads((HERE/'freeze.json').read_text())
    key=str((HERE/'common.py').relative_to(PAPER))
    prior=HERE/'frozen_original_common.py'
    assert sha(prior)==original['source_hashes'][key]
    paths=['common.py','annotation_adapter.py','test_annotation_mapping.py',
           'video_only_speech_votes.csv','annotation_mapping_audit.json','AMENDMENT_001.txt',
           'record_amendment.py']
    value=dict(amendment='001',recorded_utc=datetime.now(timezone.utc).isoformat(),
               prior_freeze_sha256=sha(HERE/'freeze.json'),
               scientific_external_results_computed_before_amendment=False,
               technical_actor1_probabilities_existed=True,
               full_inference_started=False,
               source_hashes={str((HERE/name).relative_to(PAPER)):sha(HERE/name) for name in paths},
               preserved_originals={str(prior.relative_to(PAPER)):sha(prior)},
               defect='Prior metadata mapping transposed Disgust/Surprise response IDs',
               correction='S5 anchored ID7 Disgust, ID8 Surprise; raw human responses unchanged',
               unaffected=['clip IDs/order','media','frame extraction','crop caches','checkpoints',
                           'inference probabilities','scoring/selection/reference/bootstrap rules'])
    target.write_text(json.dumps(value,indent=2))
    print(json.dumps(value,indent=2))


if __name__=='__main__':main()
