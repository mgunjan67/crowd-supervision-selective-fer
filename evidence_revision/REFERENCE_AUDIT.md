# Reference and claim audit: reviewer-follow-up edition

The 19 existing references retain the primary-source checks documented in `final_experiment/REFERENCE_AUDIT.md`; that audit is included in Online Resource 1. No benchmark performance is borrowed from those publications for a mismatched numerical comparison. All manuscript citations must match a unique bibliography key; the build rejects unused references, duplicate keys/DOIs and unresolved rendered references.

The added reference is Wassily Hoeffding, “Probability Inequalities for Sums of Bounded Random Variables,” Journal of the American Statistical Association 58(301), 13–30 (1963), DOI https://doi.org/10.1080/01621459.1963.10500830. Publisher metadata was checked on 22 September 2026. It supports the bounded-independent-variable concentration inequality, not a guarantee for these images or deployment. The fixed-grid union-bound application and its assumptions are explicitly derived in the manuscript. The publisher page did not provide accessible full text through the browsing tool; no direct quotation or claim of reading that full text is made.

Primary publisher locator: https://www.tandfonline.com/doi/abs/10.1080/01621459.1963.10500830

## Journal requirements rechecked

- https://link.springer.com/journal/42979/submission-guidelines
- https://link.springer.com/journal/42979/how-to-publish-with-us

Checked 22 September 2026: mathematical LaTeX accepted; structured abstract 150–250 words; 4–6 keywords; numeric references; declarations and substantive AI-use disclosure. Supplementary artifacts can be collected in ZIP files; readable supplemental text/figures are additionally supplied as PDF. The actual upload portal's limits for checkpoint archives have not been verified. The subscription route is distinct from optional paid open access.

Ethics guidance was read, but it is not used to invent an institutional determination. The user explicitly declined to seek one; the unresolved risk is disclosed in the manuscript and handoff documents.

## Claim-to-source boundaries

FER+ supports prior distribution-label learning, not novelty of soft targets. Swin and squeeze-and-excitation supply established components. Selective-classification and calibration papers motivate rejection and confidence limitations, not validated guarantees for this prototype. The floor decomposition is elementary accounting, not a cited new theorem. EmotionPush, EmoChat and messaging/self-awareness work establish application context, not evidence of benefit from the present system. The Barrett review supports caution about mapping expressions to internal states. Neither the Swin-SE nor dynamic-disagreement preprint is described as a peer-reviewed benchmark result.

The FER+ author manuscript (https://arxiv.org/pdf/1608.01041) and pinned official conversion script were rechecked on 22 September 2026. The former confirms the established majority/distribution comparison and eight-output setting; the latter joins by row index and advances the index even for blank output names. No published performance number is substituted for the current experiment. See DATA_ALIGNMENT.md and raw_source_audit.json for the separate direct original-CSV mapping verification.


## Added primary source in v1.1.0

Khurana, Urja; Nalisnick, Eric; Fokkens, Antske; Swayamdipta, Swabha. Crowd-Calibrator: Can Annotator Disagreement Inform Calibration in Subjective Tasks? Conference on Language Modeling, 2024. Primary publication: https://openreview.net/forum?id=VWWzO3ewMS ; primary PDF: https://openreview.net/pdf?id=VWWzO3ewMS . No unverified DOI is supplied. The citation supports the prior conceptual connection between crowd disagreement, calibration and abstention in subjective language tasks; it does not supply a comparable FER score.

Two preprints already discussed in the older main text are now formal numbered references instead of prose-only URLs. Primary arXiv pages were checked on 3 October 2026: Vats and Chadha, Facial Expression Recognition using Squeeze and Excitation-powered Swin Transformers, arXiv:2301.10906v7 (2023), DOI 10.48550/arXiv.2301.10906; Wang, Yiming; Li, Frederick W. B.; Wang, Jingyun, Predicting Human Disagreement for Calibrated Dynamic Facial Expression Recognition, arXiv:2609.17130v1 (2026). Both are explicitly labelled unpublished preprints, not peer-reviewed confirmations. The latter's DataCite DOI registration was marked pending, so the verified primary URL is supplied rather than an asserted registered DOI. No benchmark superiority or numerical comparisons are derived from either. All 23 bibliography keys are cited; duplicate and unused-key checks are part of build.py.
