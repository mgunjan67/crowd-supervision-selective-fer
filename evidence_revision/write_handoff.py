"""Author-facing handoff documents grounded in the final revised ledger."""
import json
from pathlib import Path
OUT=Path(__file__).resolve().parent;ROOT=OUT.parent
def main():
    build=json.loads((OUT/'build_report.json').read_text());audit=json.loads((OUT/'audit.json').read_text())
    assert audit['status']=='passed'
    previous=ROOT/'output/submission-github-2026-09-30'
    replay=(ROOT/'reviewer_followup/reproduce_inference.py').read_text().replace('from train_controls import base','from train_tie_uniform import base').replace("('hard','soft','uniform','tie_hard')","('tie_uniform',)")
    (OUT/'reproduce_inference.py').write_text(replay)
    (OUT/'BUILD_README.txt').write_text('''Extract Manuscript-source.zip to an otherwise empty folder. Keep evidence_revision/generated and evidence_revision/figures next to main.tex. With an existing LaTeX installation, run pdflatex main.tex, bibtex main, then pdflatex main.tex twice. The precompiled main.bbl is included. Supplementary source is evidence_revision/generated/supplement.tex; run pdflatex on that path from the extraction root. This is a multi-file Springer project, not a standalone .tex document. No image dataset is needed to compile the supplied source. Numeric regeneration instead uses Online Resource 1 and its reproducibility instructions.\n''')
    cover=(previous/'COVER_LETTER.txt').read_text(encoding='utf-8').replace('Twelve matched Swin-T training runs compare four supervision conditions','Fifteen matched Swin-T training runs compare five supervision conditions').replace('/releases/tag/v1.0.0','/releases/tag/v1.1.0')
    cover=cover.replace('All four authors approve this final manuscript and its new results. ','')
    paragraph='The latest evidence revision adds a tie-preserving Uniform control, fixed-image predictor/selector cross-evaluation, same-population categorical-reference comparisons, and complete learning trajectories. These analyses show that the own-ranking advantage is not a consistent fixed-image category-choice improvement, and reveal adverse retention of wholly unsupported annotations. The paper reports those limits rather than attributing a unique minority-vote mechanism.\n\n'
    cover=cover.replace('The study fits',paragraph+'The study fits')
    (OUT/'COVER_LETTER.txt').write_text(cover,encoding='utf-8')
    (OUT/'SUBMISSION_CHECKLIST.md').write_text('''# Final evidence-revision checklist

- Target: SN Computer Science, Original Research, subscription route. No acceptance or review-speed guarantee.
- Main manuscript and supplementary PDF compiled with zero warnings and every page visually checked; consult build/visual audit hashes.
- Editable Springer LaTeX, 23-source bibliography, generated tables and figures supplied.
- All 15 training runs and both checkpoint states are reported; 30 inference weights are available across Online Resources 2–6.
- No original face images, participant-study evidence, unverifiable historical scores or credentials are redistributed.
- Both supplied reviews have an evidence-backed response matrix. Independent dataset/backbone and learned supported-mass selector remain outside this revision; do not claim they were resolved.
- All four authors must review and approve this v1.1.0 manuscript and later results before the corresponding author confirms the portal declaration. The September approval covered the prior revision only.
- Reconfirm no concurrent journal submission at actual upload; the prior IEEE Transactions on Affective Computing submission ended.
- Affiliations, authorship order, corresponding author, funding and competing-interest statements remain as author-confirmed.
- Keep the truthful no-institutional-approval/no-exemption ethics statement and photograph-consent/provenance limits. Their editorial acceptability remains a risk.
- Disclose the TechRxiv v1 preprint, DOI 10.36227/techrxiv.175416003.30236370/v1. No preprint update has been posted automatically.
- Check portal artifact size limits; do not omit access to weights if direct upload fails. Use the versioned public release links and journal instructions.
- No journal submission has been made by this assistant.
''',encoding='utf-8')
    (OUT/'START_HERE.md').write_text(f'''# Evidence revision v1.1.0

Target remains SN Computer Science. This is a strengthened, bounded empirical manuscript, not evidence of assured acceptance.

Main PDF: crowd-supervision-evidence-revision.pdf ({build['pages']} pages).
Supplement: Supplementary-diagnostics.pdf ({build['supplement_pages']} pages).
Editable source: Manuscript-source.zip and source/main.tex.
Numerical evidence: OnlineResource1.zip; unchanged prior weights: OnlineResource2–5.zip; new tie-preserving weights: OnlineResource6.zip.

Read REVIEW_RESPONSE.md for what changed and what remains scientifically unresolved. Obtain final coauthor approval of the later results before actual journal submission. The old v1.0.0 release is preserved; this revision does not rewrite the old findings or imply independent confirmation.
''',encoding='utf-8')
    (OUT/'REFERENCE_AUDIT.md').write_text((ROOT/'reviewer_followup/REFERENCE_AUDIT.md').read_text(encoding='utf-8')+'''

## Added primary source in v1.1.0

Khurana, Urja; Nalisnick, Eric; Fokkens, Antske; Swayamdipta, Swabha. Crowd-Calibrator: Can Annotator Disagreement Inform Calibration in Subjective Tasks? Conference on Language Modeling, 2024. Primary publication: https://openreview.net/forum?id=VWWzO3ewMS ; primary PDF: https://openreview.net/pdf?id=VWWzO3ewMS . No unverified DOI is supplied. The citation supports the prior conceptual connection between crowd disagreement, calibration and abstention in subjective language tasks; it does not supply a comparable FER score.

Two preprints already discussed in the older main text are now formal numbered references instead of prose-only URLs. Primary arXiv pages were checked on 3 October 2026: Vats and Chadha, Facial Expression Recognition using Squeeze and Excitation-powered Swin Transformers, arXiv:2301.10906v7 (2023), DOI 10.48550/arXiv.2301.10906; Wang, Yiming; Li, Frederick W. B.; Wang, Jingyun, Predicting Human Disagreement for Calibrated Dynamic Facial Expression Recognition, arXiv:2609.17130v1 (2026). Both are explicitly labelled unpublished preprints, not peer-reviewed confirmations. The latter's DataCite DOI registration was marked pending, so the verified primary URL is supplied rather than an asserted registered DOI. No benchmark superiority or numerical comparisons are derived from either. All 23 bibliography keys are cited; duplicate and unused-key checks are part of build.py.
''',encoding='utf-8')
    print('Handoff documents generated; final author review remains required')
if __name__=='__main__':main()
