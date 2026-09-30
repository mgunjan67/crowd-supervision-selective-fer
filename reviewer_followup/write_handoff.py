"""Write author handoff documents only from completed audits and actual results."""
import csv,json
from pathlib import Path
OUT=Path(__file__).resolve().parent;ROOT=OUT.parent
TITLE='Crowd Supervision and Vocabulary-Mismatched Selective Facial-Expression Prediction'
IDENTITY=TITLE+'\nSN Computer Science\nGunjan Kumar Mishra; Bijaya Ghimire; Badri Raj Lamichhane; Alan Shah\nCorresponding author: Badri Raj Lamichhane, School of Information, Computer and Communication Technology, SIIT, Thammasat University, Thailand; d6622300231@g.siit.tu.ac.th\n'

def write(name,text):(OUT/name).write_text(text,encoding='utf-8')

def main():
    audit=json.loads((OUT/'audit.json').read_text());build=json.loads((OUT/'build_report.json').read_text())
    contract=json.loads((OUT/'checkpoint_contract_audit.json').read_text())
    assert audit['status']==contract['status']=='passed' and build['warnings']==0 and audit['inference_checkpoints_verified']==24
    with (OUT/'reported_results.csv').open(newline='') as f:rows=list(csv.DictReader(f))
    primary=next(r for r in rows if r['record_type']=='contrast' and r['left']=='soft' and r['right']=='uniform' and r['state']=='selected')
    result=f"Soft minus Uniform at 80% coverage: {float(primary['point_difference_pp']):+.4f} percentage points; conditional 95% pixel-cluster interval [{float(primary['lower_pp']):+.4f}, {float(primary['upper_pp']):+.4f}]."
    write('START_HERE.md',f'''# Final reviewer-follow-up submission package

{TITLE}

Target: SN Computer Science, subscription route. This is the new edition; earlier PDFs and submission ZIPs remain preserved and should not be mixed into this upload.

## Read first

1. `crowd-supervision-final-revision.pdf`: main manuscript, {build['pages']} pages.
2. `Supplementary-diagnostics.pdf`: reviewer-readable supplemental figures and diagnostics, {build['supplement_pages']} pages.
3. `REVIEW_RESPONSE.md`: every supplied criticism mapped to the response and evidence.
4. `FINAL_REVIEW.md` and `ETHICS_SUBMISSION_RISK.md`: remaining limitations and the acknowledged unresolved ethics risk.
5. `SUBMISSION_CHECKLIST.md`: author-only checks before uploading.

{result}

## Upload artifacts

- Main PDF and `Manuscript-source.zip` (editable Springer LaTeX, bibliography, figures).
- `OnlineResource1.zip`: code, both protocols, tests, predictions, diagnostics, provenance and audits; includes the supplementary PDF.
- `OnlineResource2.zip` through `OnlineResource5.zip`: Hard, Soft, Uniform and Tie-hard weights respectively, six selected/final files per condition.
- `COVER_LETTER.txt`: factual preprint and revision disclosure.

Large checkpoint files are separated from the small source/evidence archives. Check the live portal's file limits; repository deposition may be needed if its limits are lower. No public deposit or submission has been performed. Do not claim a repository DOI exists.

The technical package does not imply ethics clearance, external replication, acceptance, or publication. The authors explicitly chose not to seek an institutional determination; the truthful declaration and this unresolved submission risk are retained.
''')
    write('REVIEW_RESPONSE.md',f'''# Point-by-point response to the supplied review

This is a response to the user's supplied review, not an assertion that SN Computer Science has reviewed or invited the paper. Numbering follows that review. "Addressed" identifies a concrete revision or test, not a guarantee that a reviewer cannot object.

| Review point | Revision and evidence | Status / boundary |
|---|---|---|
| 1. Journal fit does not establish contribution or readiness | Retained the sound-science empirical framing; no acceptance odds or timetable claimed. Main title and Introduction lead with the research question. | Addressed by scope; editorial judgment remains. |
| 2. Preserve matched design and interpretive strengths | The same model, source images, weights, seed-specific initialization, augmentation and budget are used across four conditions. `matched_design.json`, frozen sources and prediction audits document this. | Verified controls retained; person-level independence not established. |
| 3. Soft labels, Swin plus gating and the identity are not novel | Related work explicitly attributes these established ideas. Introduction defines a bounded investigation of minority allocation, ties and vocabulary-sensitive reliability. The accounting identity is not presented as a theorem. | Modest empirical contribution, not a new method. |
| 4. Missing ambiguity-preserving softening control | Added Uniform: same dominant supported fraction, other six targets uniform, same supported-mass weight. Three matched seeds, no added data or compute per run. Main Table 2, Fig. 2, all per-seed contrasts and selected/final sensitivity. {result} | Directly tested; interpretation follows the observed outcome, not assumed superiority. |
| 4. Canonical tie handling on 1,360 training rows | Added three Tie-hard runs: uniform target over tied maxima, identical to Hard on unique maxima. New tests verify symmetry and exact objective behavior. | Directly tested; not a claim that all hard-label schemes are inferior. |
| 5. MSP omits supported mass | Methods derive the idealized 1-w*MSP relation; actual unsupported mass, supported ambiguity and excess are separated in the ledger, Fig. 3 and Fig. 4. Full/partial/zero strata and supported-only risk have explicit denominators. | Diagnosed, not claimed as the sole causal failure mechanism. |
| 5. CE and squared-distance optima differ | Methods derive both constrained optima. All four conditions are also evaluated at fixed epoch 12. Full-supported strata remove the omitted-mass distinction in that subgroup. | Meaningful sensitivity; not an alternative-metric replay or a trained expanded-vocabulary model. |
| 6. All-target overshoot underexplained | Added unconditional calibration/test annotation composition; accepted-population three-component risk gaps; 100 shared pixel-cluster calibration half-splits per model; fixed-grid Hoeffding-margin comparator with explicit assumptions and abstention reporting. Main Table 4, Fig. 4 and supplemental tables. | Investigated with multiple diagnostics; relative causal contributions remain unidentified. |
| 6. Greater coverage at unequal achieved risk | Explicitly states the 10% target comparison has different achieved risks. Common-coverage contrast remains the controlled endpoint; no equal-risk superiority claim. | Corrected interpretation. |
| 7. Reused images, three seeds and conditional CI | Both protocol phases and prior-outcome knowledge are disclosed. Primary follow-up contrast specified before new training; all six new controls finish before new test inference. All seed differences remain visible. | Controls and final-epoch sensitivity address alternatives; independent-image validation remains absent. |
| 8. Messaging title overpromises application evidence | Research-dominant title and Introduction. Prototype retained as motivation/implementation, with local-only backend condition, per-message opt-in, freshness and failure suppression. No recipient transport or communication-benefit claim. | Scope corrected; no user study is invented. |
| 9. F1 tasks and category/row mappings | Original-label and majority-subset F1 have explicit references and denominators. Independent audit compares every probability-file row to the official vote CSV and original class/hash manifest. No SOTA or accuracy/F1 comparison table. | Verified mappings; F1 discrepancy not called a performance gain. |
| 10. New evidence needs actual reproduction | All 24 exported weights are used to generate retained predictions. The original selected models additionally reproduce 18 old role-level arrays bitwise. Independent arithmetic audit checks mappings, decomposition, F1, policies, resamples and bootstrap. Extracted-archive tests and PDF rebuild are separately recorded. | Internal automated verification, not external independent replication. |
| 10. Ethics cannot be resolved by wording | Disclosure remains: no institutional approval or exemption determination obtained. The authors explicitly declined to seek a determination on 22 September 2026. `ETHICS_SUBMISSION_RISK.md` and checklist flag this. | **Unresolved, author-acknowledged submission risk. No exemption claimed.** |
| 11. Focused revision rather than endless rebuilding | Exactly six new runs, two controls, fixed-epoch sensitivity and predefined reliability diagnostics. No architecture search or outcome-dependent extra runs. All values retained in one ledger; journal-formatted source, PDFs, figures and evidence archives supplied. | Defined technical revision completed; optional external dataset/backbone remains future work. |

## Important boundaries

The review asked for a justified vocabulary sensitivity; this revision supplies analytical optima, mass stratification and a changed-reference diagnostic, not a newly trained ten-output predictor. It does not call that missing experiment completed. Similarly, fixed-epoch evaluation is not selection under a different validation metric. No wording can supply the omitted institutional determination or independent-domain/communication evidence.
''')
    write('FINAL_REVIEW.md',f'''# Final strict assessment

## What is now materially stronger

The central claim is no longer supported by hard-versus-soft alone. Uniform tests whether detailed minority allocation adds value beyond image-dependent softening; Tie-hard probes arbitrary tie choices. The same four conditions are checked at a fixed final epoch. Reliability analysis distinguishes unsupported votes, supported ambiguity and excess, with actual population-composition measurements, threshold half-splits and an assumption-qualified conservative comparator.

{result}

Every result is tied to retained predictions and exported inference weights. Main and supplemental tables are generated from one CSV. Independent arithmetic and extracted-package audits must be read alongside the manuscript; they are internal checks, not external replication.

## What a strict reviewer can still legitimately criticize

- One reused image source, three seeds, one backbone and one fixed training budget; no person-disjoint or external-domain validation.
- Established architecture and learning principles. This is a focused empirical contribution, not a methodological breakthrough.
- Supported-mass stratification is not an expanded-vocabulary training experiment; fixed-epoch evaluation is not exhaustive selection-criterion sensitivity.
- No causal identification of the relative mechanisms behind overshoot. The bound's independence and deployment-population assumptions are not established.
- No messaging benefit, naturalistic evaluation or runtime risk guarantee. The prototype remains secondary.
- **No institutional ethics approval or exemption determination. The authors chose not to seek one; this remains an unresolved submission risk.**

## Submission judgment

This revision addresses the actionable technical criticisms with actual controls and diagnostics. It is a more defensible empirical submission, not a flawless or publication-approved paper. The technical package can be prepared for submission; the corresponding author must decide whether to proceed with the explicitly unresolved ethics risk and obtain all coauthors' approval of the final new findings. No acceptance probability or fast-decision promise is justified.
''')
    write('SUBMISSION_CHECKLIST.md',f'''# Final submission checklist

Target: SN Computer Science. Main title: {TITLE}

## Technical completion evidence

- [x] Twelve training runs, three seeds per condition; no additional outcome-driven runs.
- [x] Selected and fixed epoch-12 weights and 72 role-level prediction files retained.
- [x] Independent numerical audit passed; source/label/hash joins checked.
- [x] Checkpoint class order, preprocessing and single-gate shapes verified against the unchanged app.
- [x] Structured abstract ({build['abstract_words']} words), numeric references and required declarations.
- [x] Main PDF compiled with no layout/reference warnings; every page rendered.
- [ ] Confirm `visual_review.json` and `archive_audit.json` both record a pass for this exact release hash (packaging gate checks this).

## Author-only actions before pressing Submit

- [ ] All four authors read and approve these new control results and the final manuscript; prior authorship consent is not silently treated as approval of unseen results.
- [ ] Confirm affiliations, active corresponding email, contribution statements and any ORCIDs.
- [ ] Confirm no simultaneous journal submission.
- [ ] Read and knowingly accept the unresolved ethics/data-provenance risk. No approval/exemption exists and the authors have declined to seek a determination. Answer portal questions truthfully; do not invent identifiers or select a false exemption.
- [ ] Confirm that the available dataset terms permit this use and planned artifacts. No source facial images are redistributed; a mirror's license is not proof of rights to every photograph.
- [ ] Select the subscription route if avoiding the optional open-access APC. Check current portal choices and funding requirements.
- [ ] Upload the editable source, main PDF, Online Resource 1 and weights 2--5. Check actual portal file limits; arrange a legitimate repository deposit if required. No deposit DOI is supplied or invented here.
- [ ] Disclose TechRxiv DOI 10.36227/techrxiv.175416003.30236370/v1 and AI assistance as written.

The unresolved institutional determination is not marked complete. No submission or public upload has been made.
''')
    write('COVER_LETTER.txt',f'''Dear Editors of SN Computer Science,

Please consider our manuscript, "{TITLE}", as an original research article.

The study investigates whether detailed crowd-label distributions add selective-prediction value beyond dominant-confidence-preserving target softening, and examines the interpretation of reliability when a seven-output expression model is evaluated against ten-category annotations. Four target conditions share data, supported-vote weights, initialization, augmentation, architecture and training budget across three paired seeds. Selected and fixed-epoch evaluations are accompanied by supported-mass diagnostics and calibration-transfer analysis. The contribution is a controlled empirical investigation, not a new architecture or a claim of improved communication.

{result}

An earlier version appeared on TechRxiv as "Typing with Emotions: Facial Expression Recognition and Embedding in Text Messages", version 1, DOI 10.36227/techrxiv.175416003.30236370/v1. The present revision removes participant findings and unverifiable historical scores, corrects the implementation description, and supplies newly documented experiments with retained weights, sample-level predictions and auditable source. The local messaging prototype is retained as a secondary implementation artifact. This submission does not claim to reproduce the historical manual relabeling procedure.

The manuscript openly states that no institutional ethics approval or exemption determination was obtained for this existing-benchmark analysis. No new participants were recruited and no identifiable facial photographs are reproduced. We do not equate public availability with a formal exemption or independently established consent for all source photographs. We acknowledge this unresolved issue for editorial assessment.

All four authors have confirmed the authorship arrangement, no funding and no competing interests. Badri Raj Lamichhane is the supervisor and corresponding author; Alan Shah contributed to literature review and code testing. The manuscript discloses use of OpenAI Codex for literature checking, code, drafting and technical verification, with responsibility retained by the named authors. Complete source and evidence archives accompany the manuscript. We request consideration through the subscription publishing route.

Sincerely,
Badri Raj Lamichhane
Corresponding author, on behalf of Gunjan Kumar Mishra, Bijaya Ghimire and Alan Shah
School of Information, Computer and Communication Technology
Sirindhorn International Institute of Technology, Thammasat University, Thailand
d6622300231@g.siit.tu.ac.th
''')
    write('MODEL_CARD.md',f'''# Released inference checkpoints

{IDENTITY}

24 checkpoints: four supervision conditions (Hard, Soft, Uniform, Tie-hard), seeds 17/42/89, selected or fixed epoch 12. Each has the same seven-output grayscale3/224/bilinear/mean0.5/std0.5 Swin-T plus one pooled-vector channel gate. Class order: angry, disgust, fear, happy, neutral, sad, surprise. The export removes optimizer/RNG states, not model tensors, and adds the release protocol hash and evaluation state. The inherited `protocol` field names the historical shared training implementation; `configuration.condition`, `release_protocol` and the frozen follow-up protocol identify the actual condition and provenance.

Selected/final inference on the supplied path uses BF16 CUDA autocast; the prototype uses FP32. Cross-hardware bitwise equality is not assumed. No checkpoint predicts internal feelings, intention or sincerity. No runtime demographic fairness, out-of-domain accuracy, formal privacy guarantee or communication benefit is established. Do not use for consequential decisions about people.

No source face images are distributed. Dataset provenance and rights limitations are stated in the manuscript. Use the weights with the packaged code and inspect SHA256_MANIFEST.json. The fixed-grid policy is an offline assumption-qualified comparator, not an installed safeguard.
''')
    write('REPRODUCIBILITY.md',f'''# Reproducibility entry point

{IDENTITY}

## Extracting the evidence

Extract OnlineResource1.zip into a fresh folder. For weight checks or inference, also extract OnlineResource2.zip through OnlineResource5.zip into that same folder, preserving paths. No source images are included. `SHA256_MANIFEST.json` inside each archive lists every packaged file. Outer archive hashes are in `package_manifest.json` supplied alongside the archives. A table-only audit needs the small Online Resource 1 archive, not the multi-gigabyte weights: use `python reviewer_followup/audit.py --skip-weights`; its report explicitly records that weights were not checked.

The manuscript source is self-contained and can be compiled without Python or data; see BUILD_README.txt. Supplementary-diagnostics.pdf makes the additional tables and figures directly readable without code.

The sole current numerical ledger is `reviewer_followup/reported_results.csv`. The root-level `reported_results.csv` is explicitly historical, unverified preprint provenance retained only for backward-compatible regression tests; none of its five values contributes to this manuscript. Read `final_experiment/HISTORICAL_PROVENANCE.md` for that boundary.

## CPU-only numerical checks (no source images needed)

Install the recorded Python dependencies from the packaged code requirements/environment records. From the extraction root:

```
python reviewer_followup/audit.py
python reviewer_followup/check_checkpoint_contracts.py
python -m pytest code/tests final_experiment/test_final.py final_experiment/test_analysis.py -c code/pytest.ini -p no:cacheprovider -q
python -m pytest reviewer_followup/test_controls.py reviewer_followup/test_diagnostics.py -c code/pytest.ini -p no:cacheprovider -q
python reviewer_followup/analysis.py
python reviewer_followup/build_assets.py
```

The audit independently recomputes vote/label/hash row joins, complete and supported-only risks, all three decomposition terms, confusion-matrix F1, threshold choices, calibration half-splits and 5,000 paired pixel-cluster resamples. It does not import the follow-up analysis module. `DATA_ALIGNMENT.md` supplies the explicit category mapping. `raw_source_audit.json` additionally records direct verification of original emotion IDs and pixel bytes against the retained raw archive. Regenerating the ledger uses the documented implementation separately. Conditional bootstrap uncertainty is not external validity or seed-population uncertainty.

## Image-dependent inference replay

Acquire FER2013 legitimately and prepare the exact source using `scripts/prepare_fer2013.py` and the acquisition/hash manifests. Keep the prepared Training/PublicTest/PrivateTest .npz files under `data/fer2013`. FER+ annotation CSV and its license are supplied. See `reviewer_followup/DATA_ALIGNMENT.md`: annotation row identifiers, not potentially shifted filename strings, determine the join. With the retained raw archive under `data/raw`, `python reviewer_followup/verify_raw_source.py` independently rechecks original numeric class IDs and every pixel hash.

```
python reviewer_followup/reproduce_inference.py --require-bitwise --output-json replay-report.json
```

This checks all 72 role-level arrays against the 24 packaged exported weights. Exact replay is expected only on the recorded deterministic hardware/software path. Without `--require-bitwise`, the report records exact-match status and maximum probability differences; it does not silently pass a cross-platform difference as bitwise reproduction. CPU evaluation uses FP32 and can differ from recorded BF16 inference.

## Training reruns without overwriting evidence

The original six hard/soft runs and later six reviewer-motivated controls were separate phases. Earlier test outcomes were known before the second protocol. `freeze.json` records control sources before control training. The image source is reused, not an independent confirmation set.

```
python reviewer_followup/retrain.py --condition uniform --seed 17 --output-dir reproduced-training
```

The output directory must be a new child of the extraction root, outside the retained `reviewer_followup` and `final_experiment` directories. Repeat for each of hard/soft/uniform/tie_hard and seeds 17/42/89. The original deterministic training implementation is reused; no new best-of-many search is permitted. The 12-epoch budget is not a convergence claim. Exact numerical repetition depends on the recorded environment.

## Prototype continuity

The app source is unchanged and its existing headless/browser verification reports are included. All released weights pass its class-order, preprocessing and model-head contract. To keep the demonstration checkpoint choice independent of new test performance, the previous Soft seed-17 selected checkpoint remains the example:

```
python -m streamlit run code/app.py --server.address 127.0.0.1 --server.headless true --browser.gatherUsageStats false -- --checkpoint reviewer_followup/weights/soft_seed17_selected.pt
```

The explicit launch flags bind to loopback and disable usage statistics even when launched from the extraction root rather than the code directory. Per-message opt-in, one eligible face and a fresh cue are necessary; this is a local preview without recipient transport. Do not infer that benchmark policies or safeguards are validated in live messaging.

## Limits

Internal automated audits are not external replication. No new human study, ethics determination, independent image source, alternate backbone or expanded-vocabulary training is supplied. The institutional determination remains unresolved by explicit author choice. See ETHICS_SUBMISSION_RISK.md.
''')

if __name__=='__main__':main()
