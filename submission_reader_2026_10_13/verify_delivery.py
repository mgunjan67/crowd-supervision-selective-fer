"""Validate clean editorial archives and compile their exact LaTeX sources.

This writing-only edition deliberately does not repeat scientific analyses,
training, inference, bootstrap calculations or figure generation.
"""
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path
import fitz

HERE = Path(__file__).resolve().parent
PAPER = HERE.parent
DEST = PAPER / 'output/submission-reader-2026-10-13'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def extract(archive, target):
    with zipfile.ZipFile(archive) as package:
        for name in package.namelist():
            resolved = (target / name).resolve()
            assert resolved.is_relative_to(target.resolve()), name
            if resolved.exists() and name != 'SHA256SUMS.json':
                assert resolved.read_bytes() == package.read(name), name
        package.extractall(target)
        hashes = json.loads(package.read('SHA256SUMS.json'))
        for name, expected in hashes.items():
            assert sha(target / name) == expected, name
    return {'archive': archive.name, 'members_verified': len(hashes), 'sha256': sha(archive)}


def main():
    work = Path(tempfile.mkdtemp(prefix='academic-delivery-', dir=PAPER / 'tmp'))
    evidence = work / 'evidence'
    evidence.mkdir()
    source = work / 'source'
    source.mkdir()
    archives = [extract(DEST / name, evidence if 'OnlineResource' in name else source)
                for name in ('OnlineResource1.zip', 'OnlineResource8-external-transfer.zip',
                             'Manuscript-source.zip')]
    commands = []

    def run(argv, cwd):
        print('Executing: ' + ' '.join(str(item) for item in argv), flush=True)
        result = subprocess.run([str(item) for item in argv], cwd=cwd,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                encoding='utf-8', errors='replace')
        record = {'argv': [str(item) for item in argv], 'cwd': str(cwd),
                  'exit_code': result.returncode}
        commands.append(record)
        (work / f'command-{len(commands):02d}.log').write_text(result.stdout, encoding='utf-8')
        if result.returncode:
            (HERE / 'checks/delivery_attempt.json').write_text(json.dumps({
                'commands': commands, 'workspace': str(work), 'failure': result.stdout[-16000:]
            }, indent=2), encoding='utf-8')
            raise RuntimeError(result.stdout[-16000:])

    current = evidence / 'paper_revision/submission_reader_2026_10_13'
    assert (current / 'apply_editorial.py').is_file()
    assert (current / 'editorial/NUMERIC_LOCK.json').is_file()
    assert (current / 'editorial').is_dir()
    before_tex = {path.relative_to(current / 'source').as_posix(): path.read_bytes()
                  for path in (current / 'source').rglob('*.tex')}
    # The overlay validates input locks before applying canonical prose.
    # Idempotence establishes that the packaged sources are that exact edition.
    run([sys.executable, 'paper_revision/submission_reader_2026_10_13/apply_editorial.py'], evidence)
    for name, before in before_tex.items():
        assert (current / 'source' / name).read_bytes() == before, name
        assert (HERE / 'source' / name).read_bytes() == before, name
    for path in (source / 'submission_reader_2026_10_13/source').rglob('*'):
        if path.is_file():
            relative = path.relative_to(source / 'submission_reader_2026_10_13/source')
            assert path.read_bytes() == (HERE / 'source' / relative).read_bytes(), str(relative)

    diagnostics = []
    for path in sorted((current / 'diagnostics').glob('*.csv')):
        assert path.read_bytes() == (HERE / 'diagnostics' / path.name).read_bytes(), path.name
        diagnostics.append({'name': path.name, 'sha256': sha(path), 'byte_identical': True})
    assert len(diagnostics) == 8, len(diagnostics)
    figures = []
    for path in sorted((current / 'source/figures').glob('*.pdf')):
        assert path.read_bytes() == (HERE / 'source/figures' / path.name).read_bytes(), path.name
        figures.append(path.name)
    assert len(figures) == 21, (len(figures), figures)

    compiler = Path('C:/Users/mgunj/AppData/Local/Programs/MiKTeX/miktex/bin/x64')
    os.environ['PATH'] = str(compiler) + os.pathsep + os.environ['PATH']
    pdfs = []
    for stem, name in [('main', 'research-revision.pdf'),
                       ('Supplementary-diagnostics', 'Supplementary-diagnostics.pdf')]:
        for iteration in range(4 if stem == 'main' else 3):
            run(['pdflatex', '-interaction=nonstopmode', '-halt-on-error', stem + '.tex'], source)
            if iteration == 0 and stem == 'main':
                run(['bibtex', stem], source)
        log = (source / (stem + '.log')).read_text(errors='replace')
        warnings = [line for line in log.splitlines()
                    if re.search(r'Warning:|Overfull \\[hv]box|Underfull \\[hv]box', line)]
        assert not warnings, warnings
        with fitz.open(source / (stem + '.pdf')) as built, fitz.open(DEST / name) as delivered:
            assert len(built) == len(delivered), (name, len(built), len(delivered))
            assert all(left.get_text() == right.get_text()
                       for left, right in zip(built, delivered)), name
            for left, right in zip(built, delivered):
                assert left.rect == right.rect, name
                first = left.get_pixmap(matrix=fitz.Matrix(1, 1), alpha=False)
                second = right.get_pixmap(matrix=fitz.Matrix(1, 1), alpha=False)
                assert (first.width, first.height, first.n, first.samples) == (
                    second.width, second.height, second.n, second.samples), (name, left.number + 1)
                for font in left.get_fonts(full=True):
                    assert font[0] > 0 and built.extract_font(font[0])[3] and font[2] != 'Type3', font
            pdfs.append({'name': name, 'pages': len(built), 'page_text_identical': True,
                         'rendered_page_pixels_identical': True, 'layout_warnings': 0,
                         'embedded_fonts': True})
    report = {
        'status': 'passed', 'operation': 'clean editorial archive and independent LaTeX verification',
        'isolated_workspace': str(work), 'commands': commands, 'archive_member_hash_checks': archives,
        'editorial_overlay_verified': True, 'editorial_overlay_idempotent': True,
        'numeric_lock_sha256': sha(current / 'editorial/NUMERIC_LOCK.json'),
        'diagnostic_csvs_preserved': diagnostics, 'figure_assets_byte_identical': figures,
        'packaged_tex_byte_identical': sorted(before_tex), 'independent_source_builds': pdfs,
        'source_archive_sha256': sha(DEST / 'Manuscript-source.zip'),
        'within_archive_sha256': sha(DEST / 'OnlineResource1.zip'),
        'external_archive_sha256': sha(DEST / 'OnlineResource8-external-transfer.zip'),
        'training_or_inference': False, 'empirical_recomputation': False,
        'figure_regeneration': False,
        'caveat': 'Fresh delivery/layout checks only. Previous empirical tests and replay remain inherited, not rerun or independently replicated.'
    }
    for destination in (HERE / 'checks/delivery_verification.json', DEST / 'delivery_verification.json'):
        destination.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
