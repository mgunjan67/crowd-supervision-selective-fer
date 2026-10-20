"""Portable, no-media numerical replay of the unchanged frozen analysis.

Only immutable-manifest path separators are normalized. No statistical or
selection function is replaced. Raw video preparation and CUDA inference are
separate steps requiring the original media and checkpoint prerequisites.
"""
import json
from pathlib import Path
import common
import analyze


def checked_manifest():
    frozen=json.loads((common.HERE/'freeze.json').read_text())
    amendment=json.loads((common.HERE/'amendment_001.json').read_text())
    assert amendment['prior_freeze_sha256']==common.sha(common.HERE/'freeze.json')
    for name,old_hash in frozen['source_hashes'].items():
        expected=amendment['source_hashes'].get(name,old_hash)
        assert common.sha(common.PAPER/name.replace('\\','/'))==expected,name
    for name,expected in {**amendment['preserved_originals'],**amendment['source_hashes']}.items():
        assert common.sha(common.PAPER/name.replace('\\','/'))==expected,name
    return frozen


if __name__=='__main__':
    analyze.require_frozen=checked_manifest
    analyze.main()
