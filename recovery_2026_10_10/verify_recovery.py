"""Independent numerical checks; does not import the new analysis or adapters."""
import csv
import hashlib
import json
import math
import re
import subprocess
import sys
import zipfile
from pathlib import Path

import fitz
import numpy as np
try:
    import openpyxl
except ModuleNotFoundError:
    # Reuse the app's existing pure-Python spreadsheet reader; no installation.
    sys.path.append('C:/Users/mgunj/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/Lib/site-packages')
    import openpyxl

HERE = Path(__file__).resolve().parent
PAPER = HERE.parent
OLD = PAPER / 'output/submission-journal-style-2026-10-04-scientific-diagrams'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_csv(path):
    with path.open(newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def main():
    expected_old = {
        PAPER / 'backbone_followup/reported_results.csv': '200a9b1df63cb267169a9f888b64a7fda3993f241494fff2e33320817bbfec3b',
        OLD / 'crowd-supervision-journal-style.pdf': '1ca1ed4399f17b422a423f8eb4d67edae55b33e1accd120c068cb6226a532421',
        OLD / 'Supplementary-diagnostics.pdf': 'fa36498b39822e954c42f1441c66ecaa5e6bdd862f32822d32f26fd7b8350973',
        OLD / 'source/main.tex': 'b301b2bce431a7ec288ba40ffa89e093ba354b36fb11e944dcd7684689b36645',
    }
    for path, expected in expected_old.items():
        assert digest(path) == expected, path
    audit = json.loads((HERE / 'analysis_audit.json').read_text())
    for relative, expected in audit['input_hashes'].items():
        assert digest(PAPER / relative) == expected, relative
    for name, expected in audit['output_hashes'].items():
        assert digest(HERE / name) == expected, name
    assert digest(HERE / 'selector_analysis.py') == audit['analysis_sha256']
    manifest = json.loads((PAPER / 'backbone_followup/split_manifest.json').read_text())
    ids = set(manifest['roles']['test'])
    assert len(ids) == 3275
    records = read_csv(HERE / 'selector_results.csv')
    assert len(records) == 576
    recomputed = {}
    max_error = 0.
    cache = {}
    for row in records:
        key = (row['backbone'], row['state'], row['condition'], int(row['seed']))
        if key not in cache:
            family, state, condition, seed = key
            folder = ('backbone_followup' if family == 'resnet18' else
                      'evidence_revision' if condition == 'tie_uniform' else 'reviewer_followup')
            path = PAPER / folder / 'predictions' / f'{condition}_seed{seed}_{state}/test_full.npz'
            with np.load(path, allow_pickle=False) as a:
                selected = [i for i, rid in enumerate(a['rows']) if int(rid) in ids]
                p = a['probabilities'][selected].astype(np.float64)
                q = a['q'][selected].astype(np.float64)
                original_rows = a['rows'][selected]
            assert len(p) == 3275 and set(map(int, original_rows)) == ids
            assert np.allclose(p.sum(1), 1, atol=1e-6, rtol=0)
            assert np.allclose(q.sum(1), 1, atol=1e-6, rtol=0)
            pred = np.argmax(p, axis=1)
            chosen = q[np.arange(len(p)), pred]
            supported = np.sum(q[:, 0:7], axis=1)
            best = np.max(q[:, 0:7], axis=1)
            values = (1-chosen, np.sum(q[:, 7:10], axis=1), supported-best, best-chosen)
            # Sort all seven columns independently of the analysis's partition code.
            ordered = np.sort(p, axis=1)
            entropy_terms = np.zeros_like(p)
            nonzero = p > 0
            entropy_terms[nonzero] = p[nonzero] * np.log(p[nonzero])
            scores = dict(msp=ordered[:, -1], margin=ordered[:, -1]-ordered[:, -2],
                          negative_entropy=np.sum(entropy_terms, axis=1))
            cache[key] = (original_rows, scores, values)
        original_rows, scores, values = cache[key]
        score = scores[row['score']]
        n = math.ceil(float(row['requested_coverage']) * len(original_rows))
        # Python tuple ordering is independent of the analysis's np.lexsort path.
        kept = sorted(range(len(original_rows)), key=lambda i: (-float(score[i]), int(original_rows[i])))[:n]
        assert int(row['accepted_n']) == n and int(row['population_n']) == len(ids)
        errors = [abs(float(row[field])-float(value[kept].mean()))
                  for field, value in zip(('risk','unsupported','supported_ambiguity','excess'),values)]
        assert max(errors) < 1e-12, (key, errors)
        max_error = max(max_error, *errors)
        recomputed[(*key, row['score'], float(row['requested_coverage']))] = float(values[0][kept].mean())
    assert len(cache) == 48
    contrasts = read_csv(HERE / 'selector_contrasts.csv')
    assert len(contrasts) == 288
    groups = {}
    for row in contrasts:
        family, state, score, control = (row[k] for k in ('backbone','state','score','control'))
        seed, coverage = int(row['seed']), float(row['requested_coverage'])
        delta = 100 * (recomputed[family,state,'soft',seed,score,coverage] -
                       recomputed[family,state,control,seed,score,coverage])
        assert abs(delta-float(row['difference_pp'])) < 1e-12
        groups.setdefault((family,state,score,control,coverage), []).append(delta)
    summaries = read_csv(HERE / 'selector_summary.csv')
    assert len(summaries) == len(groups) == 96
    for row in summaries:
        key = tuple(row[k] for k in ('backbone','state','score','control')) + (float(row['requested_coverage']),)
        values = groups[key]
        assert len(values) == 3
        assert abs(sum(values)/3-float(row['mean_difference_pp'])) < 1e-12
        assert abs(min(values)-float(row['minimum_seed_difference_pp'])) < 1e-12
        assert abs(max(values)-float(row['maximum_seed_difference_pp'])) < 1e-12
    main_table = (HERE/'source/generated/selector_results.tex').read_text()
    sup_table = (HERE/'source/generated/supplement.tex').read_text().split('Exploratory confidence-score sensitivity',1)[1]
    family_labels = {'swin_t':'Swin-T', 'resnet18':'ResNet-18'}
    state_labels = {'selected':'Selected','epoch12':'Epoch 12'}
    score_labels = {'msp':'MSP','margin':'Margin','negative_entropy':'Neg. entropy'}
    for family in family_labels:
        for state in state_labels:
            for score in score_labels:
                values = [sum(groups[family,state,score,c,.8])/3 for c in ('uniform','tie_uniform')]
                text = f'{family_labels[family]} & {state_labels[state]} & {score_labels[score]} & {values[0]:+.3f} & {values[1]:+.3f}'
                assert text in main_table, text
                for control in ('uniform','tie_uniform'):
                    prefix = f"{family_labels[family]} & {state_labels[state]} & {score_labels[score]} & {'Uniform' if control=='uniform' else 'Tie-Uniform'} & "
                    values = groups[family,state,score,control,.8]
                    assert prefix + ' & '.join(f'{value:+.4f}' for value in values) in sup_table
    build = json.loads((HERE/'build_report.json').read_text())
    pdf_checks = []
    for item in build['outputs']:
        path = Path(item['path'])
        assert digest(path) == item['sha256'] and not item['warnings']
        with fitz.open(path) as doc:
            assert len(doc) == item['pages']
            text = '\n'.join(page.get_text() for page in doc).replace('\u2212','-')
            assert '\ufffd' not in text and '??' not in text
            if path.name == 'research-revision.pdf':
                assert 'Table 9' in text and 'Washington' in text
                values = [sum(v)/3 for k,v in groups.items() if k[-1] == .8]
                assert all(f'{v:+.3f}' in text for v in values)
            else:
                assert 'Table S16' in text and 'Figure S9' in text
                values = [v for k,vs in groups.items() if k[-1] == .8 for v in vs]
                assert all(f'{v:+.4f}' in text for v in values)
            pdf_checks.append(dict(name=path.name,pages=len(doc),sha256=digest(path),numeric_text_match=True))
    crema = HERE/'external_data/crema_d'
    cr_audit = json.loads((crema/'metadata_audit.json').read_text())
    for name, item in cr_audit['sources'].items():
        assert digest(crema/'upstream'/name) == item['sha256']
    cr_rows = read_csv(crema/'upstream/processedResults/tabulatedVotes.csv')
    assert len(cr_rows) == 22326
    for row in cr_rows:
        votes = [int(row[k]) for k in ('A','D','F','H','N','S')]
        assert min(votes) >= 0 and sum(votes) == int(row['numResponses']) > 0
        assert abs(max(votes)/sum(votes)-float(row['agreement'])) < 1e-10
        assert int(row[''])//100000 in (1,2,3)
    with np.load(crema/'face_only_votes.npz',allow_pickle=False) as derived:
        visual = [r for r in cr_rows if 200000 <= int(r['']) < 300000]
        assert len(visual) == 7442
        for i, row in enumerate(visual):
            votes = [int(row[k]) for k in ('A','D','F','H','N','S')] + [0]
            assert derived['clips'][i] == row['fileName']
            assert np.array_equal(votes,derived['vote_counts'][i])
            assert sum(votes) == int(row['numResponses']) == int(derived['annotation_n'][i])
            assert np.allclose(np.array(votes)/sum(votes),derived['q'][i],atol=0,rtol=0)
        assert digest(crema/'face_only_votes.npz') == cr_audit['annotations_sha256']
    rav = HERE/'external_data/ravdess'
    rv_audit = json.loads((rav/'metadata_audit.json').read_text())
    assert digest(rav/'pone.0196391.s008.xlsx') == rv_audit['source_sha256']
    book = openpyxl.load_workbook(rav/'pone.0196391.s008.xlsx',read_only=True,data_only=True)
    original = list(book['Speech_EmoCategory'].iter_rows(min_row=4,values_only=True))
    accuracy = {r[0]:r for r in book['Speech_RawAccuracy'].iter_rows(min_row=4,values_only=True)}
    visual = [r for r in original if r[1] == 'VO']
    assert len(original) == 4320 and len(visual) == 1440
    response_ids = (5,8,6,3,1,4,7,2,9)
    classes = ('angry','disgust','fear','happy','neutral','sad','surprise','calm','none')
    rv_rows = read_csv(rav/'video_only_speech_votes.csv')
    assert len(rv_rows) == len(visual)
    for row in original:
        assert list(map(int,Path(row[0]).stem.split('-'))) == list(map(int,row[8:15]))
        expected = tuple(int(response == row[10] or (row[10] in (1,2) and response in (1,2))) for response in row[15:25])
        assert accuracy[row[0]][15:25] == expected
    for source, row in zip(visual,rv_rows):
        assert source[0] == row['filename'] and int(source[14]) == int(row['actor'])
        votes = [source[15:25].count(k) for k in response_ids]
        assert sum(votes) == 10
        for category,count in zip(classes,votes):
            assert int(row[f'count_{category}']) == count and float(row[f'q_{category}']) == count/10
    book.close()
    metadata_python = Path('C:/Users/mgunj/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe')
    test_python = str(metadata_python) if metadata_python.exists() else sys.executable
    test = subprocess.run([test_python,'-m','unittest','discover','-s',str(HERE),'-p','test_*.py','-v'],
                          text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    assert test.returncode == 0 and 'Ran 16 tests' in test.stdout, test.stdout
    pack = Path('C:/Users/mgunj/Downloads/Codex_Research_Paper_Recovery_Pack.zip')
    with zipfile.ZipFile(pack) as z:
        checksums = json.loads(z.read('codex_review_pack/SHA256SUMS.json'))
        assert len(checksums) == 24 and z.testzip() is None
        for name, expected in checksums.items():
            assert hashlib.sha256(z.read('codex_review_pack/'+name)).hexdigest() == expected
    report = dict(status='passed',independently_recomputed_result_cells=576,
                  numeric_components_per_cell=4,maximum_absolute_component_error=max_error,
                  independently_verified_contrasts=288,independently_verified_summaries=96,
                  inference_states=48,test_population=3275,all_recorded_input_output_hashes_match=True,
                  preserved_files={str(p.relative_to(PAPER)):s for p,s in expected_old.items()},
                  main_and_supplement_table_values_match=True,pdf_checks=pdf_checks,
                  test_suite=dict(tests=16,failures=0,output=test.stdout),
                  original_metadata_checks=dict(crema_d_rows=22326,crema_visual_clips=7442,
                                                ravdess_category_accuracy_rows=4320,ravdess_visual_clips=1440),
                  reference_zip=dict(path=str(pack),sha256=digest(pack),verified_members=24,agent_template_installed=False),
                  visual_inspection=dict(status='passed',main_pages='1-25',supplement_pages='1-20',
                                         enlarged_views=['main page 19/Table 9','supplement page 19/Table S16','supplement page 20/Figure S9'],
                                         findings='No observed label overlap, cropped content or illegible new table/figure; all final pages inspected.'),
                  bibliography_scope='Structural audit of 23 entries; primary-source authorship/DOI/content check for added Washington 2021. Not a fresh full-text audit of every inherited reference.',
                  new_training_runs=0,external_videos_downloaded=0,external_model_evaluation=False,
                  q1_q2_readiness=False,external_confirmation=False)
    (HERE/'verification_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k not in ('test_suite','preserved_files')},indent=2))


if __name__ == '__main__':
    main()
