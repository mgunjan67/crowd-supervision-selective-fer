"""Bind the final page inspection record to PDFs and Poppler-rendered pixels."""
import argparse
import hashlib
import json
from pathlib import Path

import fitz
from PIL import Image, ImageChops

REVISION = Path(__file__).resolve().parents[1]
PAPER = REVISION.parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--record-reviewed', action='store_true',
        help='Record the operator-completed visual inspection; not an automated visual verdict')
    args = parser.parse_args()
    documents = []
    for slug, filename, expected in [('main', 'research-revision.pdf', 27),
                                      ('supplement', 'Supplementary-diagnostics.pdf', 30)]:
        pdf = PAPER / 'output/submission-final-2026-10-20' / filename
        rendered = PAPER / f'tmp/final-poppler-{slug}-sealed'
        previous = PAPER / f'tmp/final-poppler-{slug}'
        with fitz.open(pdf) as document:
            assert len(document) == expected
            assert len(list(rendered.glob('page-*.png'))) == expected
            font_records = {}
            for page in document:
                for font in page.get_fonts(full=True):
                    xref, extension, kind, name, *_ = font
                    assert kind != 'Type3', ('Type3 font', slug, name)
                    if xref not in font_records:
                        embedded = document.extract_font(xref)[3]
                        assert embedded, ('Nonembedded font', slug, name)
                        font_records[xref] = dict(name=name, type=kind, embedded_bytes=len(embedded))
            pages = []
            for i, page in enumerate(document, 1):
                png = rendered / f'page-{i:02d}.png'
                old = previous / png.name
                image = Image.open(png).convert('RGB')
                changed = True
                if old.is_file():
                    baseline = Image.open(old).convert('RGB')
                    changed = baseline.size != image.size or ImageChops.difference(baseline, image).getbbox() is not None
                assert page.get_text().strip(), ('Empty page', slug, i)
                assert all(x0 >= -1 and y0 >= -1 and x1 <= page.rect.width+1 and y1 <= page.rect.height+1
                           for x0, y0, x1, y1, *_ in page.get_text('words')), ('Text outside media', slug, i)
                pages.append(dict(page=i, png_sha256=sha(png), changed_from_inspected_first_render=changed))
        documents.append(dict(file=filename, pdf_sha256=sha(pdf), pages=pages,
                              total_pages=expected, all_pages_rendered=True,
                              embedded_fonts=list(font_records.values()), type3_fonts=0))
    report = dict(status='visual_review_recorded' if args.record_reviewed else 'render_comparison_only',
        renderer='Poppler pdftoppm, 110 dpi; accompanying MuPDF contact sheets',
        inspection='Codex visual inspection, not a human proof approval',
        visual_review_completed=bool(args.record_reviewed), documents=documents,
        reviewed_elements=['title/abstract', 'paragraphs/equations', 'tables and continuation headers',
                           'all figures/captions/legends', 'bibliography/URLs', 'page breaks and margins'],
        corrections=['provenance spillover removed', 'eight-row contrast table kept together',
                     'eight-row ranking table kept together', 'stratum table no longer starts with an orphan row'])
    destination = REVISION / 'checks/final_visual_review.json'
    destination.write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({row['file']:dict(pages=row['total_pages'], changed=[p['page'] for p in row['pages']
        if p['changed_from_inspected_first_render']]) for row in documents}, indent=2))
    if args.record_reviewed:
        build = REVISION / 'checks/build_report.json'
        record = json.loads(build.read_text())
        record['visual_review'] = 'passed; see final_visual_review.json for delivered-PDF hashes and inspected pages'
        record['visual_review_record_sha256'] = sha(destination)
        build.write_text(json.dumps(record, indent=2)+'\n', encoding='utf-8')


if __name__ == '__main__':
    main()
