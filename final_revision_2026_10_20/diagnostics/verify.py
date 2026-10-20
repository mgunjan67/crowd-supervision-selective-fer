"""Cross-check new rows against retained ledgers without importing analysis code."""
import csv
import hashlib
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
PAPER=HERE.parents[1]


def read(path):
    with path.open(newline='',encoding='utf-8') as handle:
        return list(csv.DictReader(handle))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    new=read(HERE/'ranking_results.csv')
    historical=read(PAPER/'recovery_2026_10_10/selector_results.csv')
    key=lambda r:(r['backbone'],r['condition'],r['state'],r['seed'],r['requested_coverage'])
    lookup={key(r):r for r in historical if r['score']=='msp'}
    matched=0
    for row in new:
        if row['score']!='msp':continue
        old=lookup[key(row)]
        assert int(row['accepted_n'])==int(old['accepted_n'])
        assert abs(float(row['risk'])-float(old['risk']))<1e-13
        matched+=1
    assert matched==192
    external=read(HERE/'external_aggregation_results.csv')
    prior=read(PAPER/'ravdess_transfer_2026_10_11/results/clip_results.csv')
    key=lambda r:(r['backbone'],r['condition'],r['state'],r['seed'],r['reference'],r['requested_coverage'])
    lookup={key(r):r for r in prior if r['score']=='msp'}
    external_matched=0
    for row in external:
        if row['aggregation']!='full8':continue
        old=lookup[key(row)]
        assert int(row['accepted_n'])==int(old['accepted_n'])
        for metric in ('risk','unsupported','supported_ambiguity','excess'):
            assert abs(float(row[metric])-float(old[metric]))<1e-12
        external_matched+=1
    assert external_matched==144
    audit=json.loads((HERE/'analysis_audit.json').read_text())
    for name,digest in audit['input_hashes'].items():
        assert sha(PAPER/name)==digest,name
    for name,digest in audit['output_hashes'].items():
        assert sha(HERE/name)==digest,name
    assert audit['analysis_sha256']==sha(HERE/'analyze.py')
    assert audit['protocol_sha256']==sha(HERE/'PROTOCOL.md')
    result=dict(status='passed',historical_primary_msp_rows_matched=matched,
                historical_external_full8_rows_matched=external_matched,
                input_hashes_verified=len(audit['input_hashes']),output_hashes_verified=len(audit['output_hashes']),
                source_predictions_unchanged=True)
    (HERE/'verification.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
