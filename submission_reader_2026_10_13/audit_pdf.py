"""Check source-derived headlines, PDF text, archive hygiene and preserved history."""
import csv
import hashlib
import json
import re
import zipfile
from pathlib import Path
import fitz

HERE=Path(__file__).resolve().parent;PAPER=HERE.parent;DEST=PAPER/'output/submission-reader-2026-10-13'
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def clean(text):return text.replace('−','-').replace('\u2013','-').replace('\u2212','-')
def main():
    claims=json.loads((HERE/'checks/generated_numeric_claims.json').read_text())
    build=json.loads((HERE/'checks/build_report.json').read_text());assert build['status']=='passed' and 150<=build['abstract_words']<=250
    docs={n:fitz.open(DEST/n) for n in ('research-revision.pdf','Supplementary-diagnostics.pdf')}
    text={n:clean('\n'.join(p.get_text() for p in doc)) for n,doc in docs.items()}
    main=text['research-revision.pdf'];supp=text['Supplementary-diagnostics.pdf']
    main_figure_numbers=[int(n) for n in re.findall(r'(?m)^Fig\.\s+(\d+)\s',main)]
    supplement_figure_numbers=[int(n) for n in re.findall(r'(?m)^Figure\s+S(\d+):',supp)]
    assert main_figure_numbers==list(range(1,10)),main_figure_numbers
    assert supplement_figure_numbers==list(range(1,13)),supplement_figure_numbers
    selector_source=(HERE/'source/generated/selector_results.tex').read_text(encoding='utf-8')
    assert 'Per-seed results are in Table~S14 and Fig.~S7.' in selector_source
    assert 'Table~S16' not in selector_source
    assert re.search(r'Per-seed results are in Table\s*S14 and Fig\.\s*S7\.',re.sub(r'\s+',' ',main))
    # A longtable continuation repeats its table number; remove adjacent repeats.
    table_numbers=[int(n) for n in re.findall(r'(?m)^Table\s+S(\d+):',supp)]
    unique_adjacent=[n for i,n in enumerate(table_numbers) if i==0 or n!=table_numbers[i-1]]
    assert unique_adjacent==list(range(1,21)),table_numbers
    reference_pages=[p.number for p in docs['research-revision.pdf'] if re.search(r'(?m)^References$',p.get_text())]
    assert len(reference_pages)==1,reference_pages
    figure_pages=[p.number for p in docs['research-revision.pdf'] if re.search(r'(?m)^Fig\.\s+\d+\s',p.get_text())]
    assert max(figure_pages)<reference_pages[0],('Figure follows References',figure_pages,reference_pages)
    cover=clean((HERE/'Cover-letter.md').read_text(encoding='utf-8'))
    # Decimal strings are generated from validated full precision, not audit anchors.
    raw=claims['external_primary']['raw9'];merged=claims['external_primary']['neutral_calm_merged']
    required_main=[f'{claims["internal_primary_pp"]["swin_t"]:.3f}',f'{claims["internal_primary_pp"]["resnet18"]:+.3f}',f'{float(raw["mean_difference_pp"]):+.3f}',f'{float(merged["mean_difference_pp"]):+.3f}']
    with (HERE/'diagnostics/principal_seed_contrasts.csv').open(newline='') as f:seedrows=list(csv.DictReader(f))
    for r in seedrows:
        token=f'{float(r["difference_pp"]):+.3f}';assert token in main and token in supp,token
    for token in required_main:assert token in main,token
    for token in [f'{float(raw[k]):+.3f}' for k in ('mean_difference_pp','lower95_pp','upper95_pp')]:assert token in cover,token
    for r in seedrows:
        if r['reference']=='raw9':assert f'{float(r["difference_pp"]):+.3f}' in cover,r
    accounting=claims['selected_swin_accounting']
    for population in ('all_eligible','msp80'):
        for v in accounting[population+'|raw9|soft']:assert f'{v:.2f}' in main,f'{population} {v}'
    for name,txt in text.items():
        for prohibited in ('state of the art','state-of-the-art','significantly improved communication','genuine emotion','detects sarcasm','detects bluffs','privacy-preserving','TODO','PLACEHOLDER','??'):
            assert prohibited.lower() not in txt.lower(),(name,prohibited)
    assert 'No institutional ethics approval or exemption determination was obtained.' in re.sub(r'\s+',' ',main)
    assert 'AI assistance (OpenAI Codex)' in re.sub(r'\s+',' ',main)
    assert 'seed-population' in main and 'rather than frame-accuracy' in main
    assert 'REPRODUCIBILITY.md' not in supp
    assert 'finalization' not in supp.lower(), 'Stale revision-directory instruction'
    # The embedded monospaced underscore extracts as a space in this PDF font.
    # Check the rendered words and the exact editable-source spelling together.
    assert 'submissionreader20261013/' in re.sub(r'[\s_]+','',supp), 'Current reproduction directory missing'
    assert r'submission\_reader\_2026\_10\_13/' in (HERE/'source/generated/supplement.tex').read_text(encoding='utf-8')
    distribution=json.loads((HERE/'diagnostics/distribution_audit.json').read_text())
    assert distribution['status']=='passed' and distribution['states']==48
    for key,value in distribution['means'].items():
        assert f'{value:.5f}' in main,(key,value)
        assert f'{value:.6f}' in supp,(key,value)
    with (HERE/'diagnostics/distribution_distance.csv').open(newline='') as stream:
        distance_rows=list(csv.DictReader(stream))
    assert len(distance_rows)==48
    for row in distance_rows:
        assert f'{float(row["distance"]):.6f}' in supp,row
    for value in claims['external_epoch12_merged_swin_seed_differences_pp']:
        assert f'{value:+.3f}' in supp,value
    # Freshly inserted scientific figures must have readable lettering after embedding.
    final_font_sizes={}
    for name,keyword in [('research-revision.pdf','Raw human-vote reference'),('Supplementary-diagnostics.pdf','Soft | seed 17'),('Supplementary-diagnostics.pdf','Mean neutral probability')]:
        pages=[p for p in docs[name] if keyword in p.get_text()]
        assert pages,(name,keyword)
        sizes=[s['size'] for p in pages for b in p.get_text('dict')['blocks'] if 'lines' in b for l in b['lines'] for s in l['spans'] if 'DejaVu' in s['font']]
        assert sizes and min(sizes)>=7.95,(keyword,min(sizes))
        final_font_sizes[keyword]={'minimum_points':min(sizes),'maximum_points':max(sizes),'pages':[p.number+1 for p in pages]}
    for keyword in ('Illustrative supported-vote distributions', 'Frozen clip prediction', 'Full-distribution agreement'):
        pages=[p for p in docs['research-revision.pdf'] if keyword in p.get_text()]
        assert pages, ('New explanatory/results figure missing',keyword)
        # Base scientific lettering must remain readable after LaTeX scaling.
        # Math super/subscripts are intentionally smaller, so check word spans.
        sizes=[s['size'] for p in pages for b in p.get_text('dict')['blocks'] if 'lines' in b
               for l in b['lines'] for s in l['spans']
               if any(font in s['font'] for font in ('Arial','DejaVu','LiberationSans'))
               and re.search(r'[A-Za-z]{3,}',s['text'])]
        assert sizes and min(sizes)>=7.95,(keyword,min(sizes) if sizes else 'missing vector fonts')
        final_font_sizes[keyword]={'minimum_word_lettering_points':min(sizes),'maximum_points':max(sizes),'pages':[p.number+1 for p in pages]}
    page_bounds=[]
    for filename,doc in docs.items():
        violations=[]
        for page in doc:
            bounds=fitz.Rect(page.rect);bounds.x0-=2;bounds.y0-=2;bounds.x1+=2;bounds.y1+=2
            for block in page.get_text('dict')['blocks']:
                for line in block.get('lines',[]):
                    for span in line['spans']:
                        if span['text'].strip() and not bounds.contains(fitz.Rect(span['bbox'])):
                            violations.append({'page':page.number+1,'text':span['text'],'bbox':span['bbox']})
        assert not violations,(filename,violations)
        page_bounds.append({'document':filename,'pages':len(doc),'text_spans_outside_page':0})
    old={'research-revision.pdf':'eb05de127b8ff35104edf73667d4990bf2196a444d8df9363baf8ff6e0419faa','Supplementary-diagnostics.pdf':'cc5959ebfdaa0278fe13f16e9c89b181e39e2684cc8653d1e677ad3a554d19b8'}
    for n,h in old.items():assert sha(PAPER/'output/ravdess-transfer-2026-10-12'/n)==h,n
    assert sha(PAPER/'ravdess_transfer_2026_10_11/freeze.json')=='6b892e3658c67059f13604ec261e3a19f3743620ddc06e4b6f88ae0b7ef05b50'
    assert sha(PAPER/'ravdess_transfer_2026_10_11/amendment_001.json')=='f683055b11dbdbb30f970c2fcd48c4e2dc7f6507262f58f56cca3fe02aa155db'
    hygiene={}
    for n in ('OnlineResource1.zip','OnlineResource8-external-transfer.zip','Manuscript-source.zip','Figures.zip'):
        with zipfile.ZipFile(DEST/n) as z:
            for member in z.namelist():
                lower=member.lower();base=Path(member).name
                assert not any(x in lower for x in ('author_approval','submission_checklist','work_log','cover-letter','finalization_report','author_actions','issue_ledger','readiness.txt','start_here','write_handoff','progress_log')),member
                assert not lower.endswith(('.jpg','.jpeg','.png','.mp4','.wav','.avi','.pt','.pyc')),member
                assert base not in ('.env','credentials.json','token.json'),member
            hygiene[n]={'members':len(z.namelist()),'sha256':sha(DEST/n),'no_face_media_weights_or_internal_author_logs':True}
    report={'status':'passed','operation':'PDF/layout and preserved-value check; no empirical recomputation','source_derived_headlines_present':required_main,'principal_seed_rows_checked':6,'all_eligible_and_accepted_accounting_checked':True,'main_figure_caption_sequence':main_figure_numbers,'supplement_figure_caption_sequence':supplement_figure_numbers,'supplement_table_caption_sequence':unique_adjacent,'selector_Table_S14_Figure_S7_pointer_present':True,'all_main_figures_before_references':True,'final_figure_fonts':final_font_sizes,'page_text_bounds':page_bounds,'old_approved_pdf_hashes_unchanged':old,'freeze_and_amendment_unchanged':True,'archive_hygiene':hygiene,'references':build['references'],'abstract_words':build['abstract_words'],'no_material_numerical_correction':'The editorial preservation check compares the unchanged full-precision ledgers. This PDF check does not recompute observations.'}
    (HERE/'checks/pdf_numeric_audit.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
if __name__=='__main__':main()
