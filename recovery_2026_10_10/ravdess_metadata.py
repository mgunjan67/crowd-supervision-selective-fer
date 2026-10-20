"""Read the unmodified primary-publication S5 raw-validity workbook."""
import hashlib
import json
import csv
import urllib.request
from pathlib import Path
import openpyxl

OUT = Path(__file__).resolve().parent / 'external_data/ravdess'
URL = 'https://journals.plos.org/plosone/article/file?id=10.1371/journal.pone.0196391.s008&type=supplementary'
CLASSES = ('angry','disgust','fear','happy','neutral','sad','surprise','calm','none')
RESPONSE_IDS = (5,8,6,3,1,4,7,2,9)


def original_accuracy(responses, target):
    """Published accuracy convention; does not merge raw response counts."""
    if target not in range(1,9):
        raise ValueError('Invalid intended expression category')
    if any(r not in range(1,10) for r in responses):
        raise ValueError('Invalid categorical response')
    return tuple(int(r == target or (r in (1,2) and target in (1,2))) for r in responses)


def convert_row(row):
    if len(row) != 25:
        raise ValueError('Expected 15 descriptor columns and 10 categorical responses')
    name = row[0]
    tokens = Path(name).stem.split('-')
    if len(tokens) != 7 or list(map(int,tokens)) != list(map(int,row[8:15])):
        raise ValueError('Filename/factor ID mismatch')
    responses = row[15:25]
    if any(not isinstance(r,(int,float)) or int(r) != r or not 1 <= r <= 9 for r in responses):
        raise ValueError('Invalid or missing emotion-category response')
    counts = [sum(r == category for r in responses) for category in RESPONSE_IDS]
    assert sum(counts) == 10
    return counts


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    target = OUT / 'pone.0196391.s008.xlsx'
    if not target.exists():
        request = urllib.request.Request(URL, headers={'User-Agent': 'research-metadata-audit/1.0'})
        with urllib.request.urlopen(request, timeout=30) as response:
            data = response.read(16_000_001)
        if len(data) > 16_000_000 or not data.startswith(b'PK'):
            raise ValueError('Expected a bounded XLSX metadata download')
        target.write_bytes(data)
    book = openpyxl.load_workbook(target, read_only=True, data_only=True)
    result = dict(source=URL, sha256=hashlib.sha256(target.read_bytes()).hexdigest(), sheets=[])
    for sheet in book:
        first = list(sheet.iter_rows(min_row=1, max_row=7, values_only=True))
        result['sheets'].append(dict(name=sheet.title, rows=sheet.max_row, columns=sheet.max_column, first_rows=first))
    sheet = book['Speech_EmoCategory']
    accuracy = {r[0]: r for r in book['Speech_RawAccuracy'].iter_rows(min_row=4,values_only=True)}
    records = []
    for row in sheet.iter_rows(min_row=4,values_only=True):
        counts = convert_row(row)
        raw_accuracy = accuracy[row[0]]
        if raw_accuracy[:15] != row[:15]:
            raise ValueError('Accuracy/category metadata mismatch')
        # The original publication accepts neutral/calm interchangeably for accuracy.
        expected = original_accuracy(row[15:25], row[10])
        if raw_accuracy[15:25] != expected:
            raise ValueError('Raw accuracy does not match categorical responses')
        if int(row[8]) != 2:
            continue
        if row[1] != 'VO' or int(row[9]) != 1:
            raise ValueError('Expected video-only speech')
        records.append(dict(filename=row[0], actor=int(row[14]), annotation_n=10,
                            **{f'count_{c}':n for c,n in zip(CLASSES,counts)},
                            **{f'q_{c}':n/10 for c,n in zip(CLASSES,counts)}))
    assert len(records) == len({r['filename'] for r in records}) == 1440
    with (OUT/'video_only_speech_votes.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
    metadata=dict(dataset='RAVDESS',source=URL,source_sha256=result['sha256'],
                  clip_count=len(records),actors=len({r['actor'] for r in records}),annotations_per_clip=10,
                  original_workbook_unchanged=True,category_to_accuracy_checks=4320,
                  evaluation_unit='video-only speech clip, not a still-image judgment',
                  classes=CLASSES,unsupported_categories=['calm','none'],
                  unsupported_votes=sum(r['count_calm']+r['count_none'] for r in records),
                  clips_with_unsupported_votes=sum(r['count_calm']+r['count_none']>0 for r in records),
                  mean_annotation_floor=sum(1-max(r[f'q_{c}'] for c in CLASSES[:7]) for r in records)/len(records),
                  clip_license='Zenodo: CC BY-NC-SA 4.0; noncommercial use only; no clips downloaded',
                  supplemental_workbook_license='PLOS article supporting information; preserve attribution',
                  model_results='none; metadata only',
                  caveats=['Calm is not merged with neutral',
                           'Original benchmark accuracy accepts calm and neutral interchangeably; any new evaluation must also report that coarsened-reference sensitivity',
                           'None is not silently recoded as an incorrect expression',
                           'R1-R10 are within-stimulus response slots, not verified global rater identities',
                           'Acted adult speech-video domain differs from FER2013 stills',
                           'Actor clustering, frame/crop protocol and cross-source overlap require audit before inference'])
    (OUT/'metadata_audit.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    book.close()
    (OUT / 'workbook_structure.json').write_text(json.dumps(result, indent=2, default=str), encoding='utf-8')
    print(json.dumps(metadata, indent=2, default=str))


if __name__ == '__main__':
    main()
