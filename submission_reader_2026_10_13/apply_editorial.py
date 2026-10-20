"""Apply the reviewed prose layer only to the unchanged, locked evidence.

The underlying generators remain available. This layer restores the academic
organisation and prose after they run. It refuses changed numerical inputs;
it must not silently apply an old narrative to new experimental results.
"""
import hashlib
import json
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
PAPER = HERE.parent
EDITORIAL = HERE / 'editorial'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    lock = json.loads((EDITORIAL / 'NUMERIC_LOCK.json').read_text(encoding='utf-8'))
    current = json.loads((HERE / 'checks/generated_numeric_claims.json').read_text(encoding='utf-8'))
    assert current == lock['numeric_claims'], 'The numerical claim ledger changed: review the prose before applying it.'
    for relative, expected in lock['evidence_sha256'].items():
        path = PAPER / relative
        assert sha(path) == expected, ('Evidence changed: editorial layer refused', relative)
    copied = []
    for relative, expected in lock['template_sha256'].items():
        template = EDITORIAL / relative
        assert sha(template) == expected, ('Editorial template differs from its reviewed lock', relative)
        destination = HERE / 'source' / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(template, destination)
        copied.append(relative)
    print(json.dumps({'status': 'passed', 'editorial_templates_applied': copied,
                      'evidence_files_locked': len(lock['evidence_sha256']),
                      'new_empirical_computation': False}, indent=2))


if __name__ == '__main__':
    main()
