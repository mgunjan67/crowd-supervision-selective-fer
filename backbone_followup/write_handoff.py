"""Human-readable bounded revision record generated after audited results."""
import csv,json,re
from pathlib import Path
OUT=Path(__file__).resolve().parent;ROOT=OUT.parent
def main():
    audit=json.loads((OUT/'audit.json').read_text());build=json.loads((OUT/'build_report.json').read_text())
    assert audit['status']=='passed' and build['warnings']==0
    contrasts=json.loads((OUT/'contrasts.json').read_text())
    effect='; '.join(f'{r["state"]} Soft-minus-{r["right"]}: {r["point_difference_pp"]:+.3f} pp [{r["lower_pp"]:+.3f}, {r["upper_pp"]:+.3f}]' for r in contrasts)
    docs={
    'START_HERE.md':f'''# Two-backbone evidence revision v1.2.0

Target: SN Computer Science, Original Research, subscription route.
Main PDF: crowd-supervision-backbone-revision.pdf ({build['pages']} pages).
Supplement: Supplementary-diagnostics.pdf ({build['supplement_pages']} pages).
Editable source: source/main.tex and Manuscript-source.zip.
OnlineResource1.zip contains code, probabilities, logs, source records and merged results.
OnlineResource7.zip supplies 18 ResNet inference weights. Unchanged Swin weights remain OnlineResources2–5 from v1.0.0 and OnlineResource6 from v1.1.0.

Nine new matched runs add one second family, not a new image source. All adverse and favorable outcomes are retained. No journal submission, preprint update, new participant study, or acceptance guarantee is implied. Read REVIEW_RESPONSE.md and SUBMISSION_CHECKLIST.md. The user confirmed all-author v1.1.0 approval; authors should review the later v1.2.0 results before submitting.
''',
    'REVIEW_RESPONSE.md':f'''# Final bounded robustness response

Preserve the earlier review-response matrix in evidence_revision/. This follow-up addresses the latest requested improvement, not every possible research limitation.

| Concern | Action and boundary |
| --- | --- |
| Only one backbone | Nine ResNet-18 runs: Soft, Uniform, Tie-Uniform, seeds 17/42/89. Same roles/transform/training budget and matched within-seed state. Two families, still one image source. No architecture leaderboard. |
| Selective improvement conflates accepted images | Both control comparisons cross-evaluated on fixed image sets; selected and epoch-12 matrices retained. |
| Selective reporting / stopping | Frozen pre-training protocol; all nine 12-epoch runs, both states, all paired seeds/contrasts, logs and technical failure record included. |
| Small effect / weak practical meaning | Absolute disagreement changes retained; no claim of corrected-image count or messaging utility. |
| Unpublished bibliography entries | Vats–Chadha and Wang–Li–Wang kept as explicitly unpublished in-text links, removed from numbered list; published ResNet CVPR source added. |
| Contribution unclear | Introduction identifies matched target controls, fixed-image comparisons and annotation-reference/vocabulary diagnostics. |
| Independent dataset / adaptive reuse | Not solved. FER+ reannotates FER2013. Four adaptive phases do not establish external generalization. |
| Learned supported-mass selector / entropy mechanism | Not implemented and not claimed. |
| Ethics / photograph rights | Truthful disclosures retained; no original images redistributed. |

ResNet reported effects (conditional descriptive intervals): {effect}.
Internal checks are not independent external replication. Author approval of v1.1.0 is confirmed; later results need final review before actual upload.
''',
    'REPRODUCIBILITY.md':'''# Reproducing the second-backbone sensitivity

Python 3.12, PyTorch 2.8.0+cu128, torchvision 0.23.0+cu128, NVIDIA RTX 3070 Laptop GPU. See each run environment.txt for complete pins. This is an adaptive sensitivity experiment, not optimized ResNet accuracy or independent-source confirmation.

Acquire the same FER2013 source and official FER+ annotations using retained provenance records; images are not supplied. Restore data/fer2013 NPZ partitions and data/ferplus_annotations/fer2013new.csv using the retained conversion/acquisition code and hashes. Pretrained source is https://download.pytorch.org/models/resnet18-f37072fd.pth. Shared grayscale and normalization differ from ImageNet default inference preprocessing by design.

Extract OnlineResource1.zip into a working folder. Extract OnlineResource7.zip into that same folder for ResNet inference weights. Swin archives 2–6 remain unchanged at the previous versioned releases. The public reproduction helper reloads the 18 weights and produces all 54 role-specific prediction arrays; CPU reproduction is permitted but is not promised bitwise equal to matched CUDA/BF16 inference. Evaluation.py records original full training checkpoint hashes; those full optimizer states are retained locally, not distributed in inference archives. audit.py checks those original full-file hashes only when the source files are available, and always checks the supplied inference weights and prediction-file hashes. Do not run the original training driver over the archived configurations expecting it to overwrite evidence. Fresh retraining uses retrain.py with a new output folder inside the extracted study root; it retains the declared fixed budget.

From the extracted root, use:

    python backbone_followup/reproduce_inference.py --output-dir reproduced-resnet --device cuda
    python backbone_followup/audit.py
    python backbone_followup/retrain.py --output-dir fresh-resnet-soft17 --condition soft --seed 17

Inference and fresh retraining require acquired images; arithmetic auditing requires only the released arrays and inference weights. The train-only target audit also requires images/source data. Set PYTHONPATH to code/ when running the combined pytest suite. Current-result ledger: backbone_followup/reported_results.csv; earlier root ledgers remain historical provenance.

The merged reported_results.csv labels every retained row by backbone. audit.py recomputes reference F1, common-coverage risk, strata, unsupported mass and fixed-set comparisons through a separate arithmetic implementation. This is internal verification, not independent human replication. Freeze files and per-run logs retain a pre-epoch Windows worker-import repair; no calibration/test evaluation preceded the nine completed runs.

Unit tests cover the 512-dimensional pooled gate, targets and zero-mass gradients, as well as the retained study tests. The messaging app retains its Swin-only inference contract; do not supply a ResNet research checkpoint to that app. No original face images, credentials or participant data are supplied. Scope and ethics limitations remain in the manuscript.
''',
    'SUBMISSION_CHECKLIST.md':'''# Submission checklist — v1.2.0

- Target SN Computer Science, Original Research, subscription route; no speed/acceptance guarantee.
- Use the new main PDF, editable source and supplement. Preserve earlier v1.0.0/v1.1.0 releases.
- Nine additional runs complete, all 24 runs reported, 48 selected/fixed-epoch inference weights retained/released.
- Read REVIEW_RESPONSE.md: two families still share one image source. No learned support-aware selector, entropy-matched mechanism or human-study benefits are claimed.
- The user confirmed all four authors approved v1.1.0. Obtain their final review of the later v1.2.0 manuscript/results before submission.
- Reconfirm no concurrent journal consideration at actual submission. Prior IEEE TAC rejection and TechRxiv preprint are disclosed.
- Keep author-confirmed affiliations/order/funding/conflicts; do not invent institutional ethics approval or verified photo consent.
- Check portal file-size limits. Supply code/evidence and stable links to all versioned weight archives if direct upload is not supported.
- No journal submission or TechRxiv update has been performed.
''',
    'REFERENCE_AUDIT.md':'''# Reference handling for the two-backbone revision

The retained primary-publication reference audit remains in evidence_revision/. Remove the two unpublished preprints from the numbered bibliography under SN Computer Science's rule: https://link.springer.com/journal/42979/submission-guidelines. Retain their author-attributed, unpublished status and arXiv links in Related work; do not conceal the closest conceptual prior.

Added published primary source: Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun. Deep Residual Learning for Image Recognition. CVPR 2016, 770–778. DOI https://doi.org/10.1109/CVPR.2016.90. Primary author manuscript: https://arxiv.org/abs/1512.03385. Model contract/weights confirmed against local torchvision source and official torchvision ResNet-18 documentation. No new architecture novelty or source benchmark-score comparison is asserted.

Automated compilation enforces complete cited-key matching, no duplicate keys/DOIs, resolved references and no layout warnings. LLM assistance remains disclosed in Methods.
''',
    'BUILD_README.txt':'''Compile main.tex with pdfLaTeX + BibTeX + pdfLaTeX twice. Keep sn-jnl.cls, sn-mathphys-num.bst and backbone_followup/ generated/figures/references intact. Main bibliography uses published/accepted sources; the two unpublished preprints are clearly labelled in-text URLs. The study is multi-file and uses the existing local compiler, not the standalone editor compiler.
'''}
    for name,text in docs.items():(OUT/name).write_text(text,encoding='utf-8')
    cover=(ROOT/'output/submission-evidence-2026-10-03/COVER_LETTER.txt').read_text(encoding='utf-8')
    cover=cover.replace('Fifteen matched Swin-T training runs compare five supervision conditions across three seeds','Twenty-four matched training runs compare five Swin-T conditions and three ResNet-18 conditions across three seeds')
    cover=cover.replace('The latest evidence revision adds a tie-preserving Uniform control,','The evidence revisions add a second-backbone matched robustness check, a tie-preserving Uniform control,')
    cover=cover.replace('/releases/tag/v1.1.0','/releases/tag/v1.2.0')
    cover=cover.replace('Its conclusions are limited to one reused benchmark image source;','Its conclusions now cover two model families but remain limited to one reused benchmark image source;')
    cover=cover.replace('The study fits the journal',f'The ResNet robustness results, including both selected and fixed-epoch contrasts, are reported without changing the retained Swin primary endpoint: {effect}. These conditional intervals are descriptive sensitivity measures, not claims of external validation.\n\nThe study fits the journal')
    (OUT/'COVER_LETTER.txt').write_text(cover,encoding='utf-8')
if __name__=='__main__':main()
