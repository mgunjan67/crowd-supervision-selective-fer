# Evidence revision v1.1.0

Fifteen matched runs now include a tie-preserving dominant-confidence Uniform control. The new manuscript adds fixed-image cross-evaluation, identical-population F1 comparisons, full learning curves and unsupported-category retention. These diagnostics narrow—not strengthen without qualification—the interpretation of the earlier aggregate benefit. See the versioned manuscript and evidence below.

The following overview preserves the original v1.0.0 twelve-run results. Use **Current revision files** below for the fifteen-run evidence revision.

# Crowd Supervision and Selective Facial-Expression Prediction

Research code and evidence for **Crowd Supervision and Vocabulary-Mismatched Selective Facial-Expression Prediction**.

Authors: Gunjan Kumar Mishra, Bijaya Ghimire, Badri Raj Lamichhane, and Alan Shah. Badri Raj Lamichhane is the supervisor and corresponding author. Alan Shah contributed to literature review and code testing. This is a research manuscript prepared for journal submission; it is not an accepted journal article.

## What this study tests

Twelve matched channel-gated Swin-T runs compare Hard, Tie-hard, Uniform-minority and full Soft supervision across seeds 17, 42 and 89. Training, checkpoint selection, calibration and testing are isolated by exact pixel hashes. Selected checkpoints and fixed epoch-12 checkpoints are evaluated. FER+ reannotates FER2013: these are one image source, not two independent image datasets.

At 80% coverage on 3,275 test images, mean complete-vote disagreement is 26.30% (Hard), 25.82% (Tie-hard), 24.06% (Uniform) and 23.68% (Soft). Soft minus Uniform is -0.38 percentage points, with a conditional 95% pixel-cluster interval of [-0.64, -0.07]. Uniform reproduces much of the hard-to-soft difference; the residual benefit is modest. The study also examines vocabulary mismatch and calibration-target transfer. It does not establish external-domain reliability or communication benefit.

## Get the exact release

Use the [v1.0.0 release](https://github.com/mgunjan67/crowd-supervision-selective-fer/releases/tag/v1.1.0). Repository source is readable directly; release assets preserve the audited submission archives. `OnlineResource1.zip` contains code, protocols, votes, predictions and audits. `OnlineResource2.zip` through `OnlineResource5.zip` contain Hard, Soft, Uniform and Tie-hard checkpoints, respectively. Each checkpoint archive contains six selected/final weights and is approximately 664 MB. Extract these archives at the repository root, preserving their paths, to run weight checks or inference. Original facial images are not distributed.

`release-artifacts.json` records the expected archive SHA256 values and sizes. `SHA256_MANIFEST.json` verifies the original Online Resource 1 contents; it intentionally does not include newly added repository documentation. `RELEASE_PROVENANCE.json` records the release-preparation checks.

## Read the paper and reproduce

- [Current submission manuscript PDF with repository links](manuscript/submission-with-repository-links.pdf)
- [Editable source for the current submission edition](manuscript/submission-source-with-repository-links.zip)
- [Original audited manuscript PDF](manuscript/crowd-supervision-final-revision.pdf)
- [Supplementary diagnostics PDF](manuscript/Supplementary-diagnostics.pdf)
- [Complete reproduction instructions](REPRODUCIBILITY.md)
- [Data alignment](reviewer_followup/DATA_ALIGNMENT.md)
- [Protocols and limitations](reviewer_followup/PROTOCOL.md)

Install Python dependencies from `code/requirements.txt` and consult the recorded environment files for exact versions. From the repository root:

```bash
python -m pip install -r code/requirements.txt
python reviewer_followup/audit.py --skip-weights
python -m pytest code/tests final_experiment/test_final.py final_experiment/test_analysis.py -c code/pytest.ini -p no:cacheprovider -q
python -m pytest reviewer_followup/test_controls.py reviewer_followup/test_diagnostics.py -c code/pytest.ini -p no:cacheprovider -q
```

The table-only audit needs no source images or weights and explicitly reports that checkpoint checks were skipped. Full inference reproduction requires the released weights and legitimately acquired FER2013 images. Commands that regenerate audit files or results change the working tree; preserve the release commit or use a separate checkout for those runs.

Train into a new output directory using the controlled study entry point:

```bash
python reviewer_followup/retrain.py --condition uniform --seed 17 --output-dir reproduced-training
```

The sole current numerical ledger is `reviewer_followup/reported_results.csv`. The root-level `reported_results.csv` and historical helper files are retained only for provenance and regression tests. They do not supply the current manuscript's results. The older generic `code/train.py` does not define the controlled paper experiment. The VADER/affective-dissonance rewrite is excluded.

## Local expression-cue prototype

With the released Soft seed-17 selected checkpoint:

```bash
python -m streamlit run code/app.py --server.address 127.0.0.1 --server.headless true --browser.gatherUsageStats false -- --checkpoint reviewer_followup/weights/soft_seed17_selected.pt
```

The backend must run on the sender's machine for local processing. The prototype implements per-message opt-in and stale/invalid-cue suppression, and ends in a local conversation preview. It does not implement recipient transport. A facial-expression category does not establish internal emotion or intent, and an attached cue may be sensitive. Offline confidence policies have not been validated as runtime safeguards.

## Provenance and permissions

The earlier manuscript appeared on [TechRxiv](https://doi.org/10.36227/techrxiv.175416003.30236370/v1). The current work supplies new traceable experiments and excludes participant findings and unverifiable historical scores. See `final_experiment/HISTORICAL_PROVENANCE.md`.

FER+ annotations retain Microsoft's copyright and MIT license in `data/ferplus_annotations/LICENSE.md`. Dataset provenance and underlying photograph-rights limitations are disclosed; images are excluded. Third-party class/style and other supplied files retain their original notices. No blanket software or checkpoint license is assigned by this release. Public availability does not grant rights beyond applicable licenses and permissions.

No institutional ethics approval or exemption determination was obtained. No new human-participant study is reported. Internal audits are not external replication. See the manuscript's declarations and `reviewer_followup/ETHICS_SUBMISSION_RISK.md`.

When referring to this work, cite the current manuscript title and all four authors, and identify the exact release/commit. Do not cite it as an accepted SN Computer Science paper or invent a journal DOI.

## Current revision files

- `manuscript/crowd-supervision-evidence-revision.pdf` is the current v1.1.0 manuscript.
- `manuscript/Supplementary-diagnostics.pdf` contains detailed runtime and diagnostic information.
- `manuscript/Manuscript-source-v1.1.0.zip` supplies editable Springer source.
- `evidence_revision/REVIEW_RESPONSE.md` maps the latest supplied criticisms to changes and unresolved scientific boundaries.
- Online Resource 6 adds six Tie-Uniform inference checkpoints; Online Resources 2–5 remain byte-identical to v1.0.0.

The old v1.0.0 release is preserved. This is not an accepted article or a journal submission. Final author approval of these later results is required before journal submission.
