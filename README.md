# Evaluating Crowd-Label Supervision for Selective Facial-Expression Recognition

Research code, protocols and retained results for revision **v1.3.0**, prepared
for SN Computer Science. This is a research manuscript, not an accepted journal
article. Authors: Gunjan Kumar Mishra, Bijaya Ghimire, Badri Raj Lamichhane and
Alan Shah. Badri Raj Lamichhane is the corresponding author.

## Study and results

The study compares hard and distributional supervision for facial-expression
recognition when a classifier withholds less-confident predictions. It measures
agreement with the complete annotation vocabulary and compares predictors on
identical accepted images. FER2013 and FER+ share one image source.

The complete study contains 36 matched training runs and 81 named checkpoint-rule states:
the original 24 runs / 48 states plus twelve explanatory follow-up runs / 33
states. The new runs test an eight-expression vocabulary and a minority-allocation
control that preserves each image's target probability values. The original
36 frozen external states are evaluated on 1,440 RAVDESS clips. External temporal
aggregation and confidence-ranking sensitivities use retained predictions.
Different checkpoint rules can select the same weights; these counts are not
counts of independent fits or necessarily distinct parameter sets.

| Outputs | Checkpoint | Comparison | Difference (pp) | Conditional 95% interval |
|---|---|---|---:|---|
| 7 | Distance-selected | Soft – permutation | -0.794 | [-1.071, -0.474] |
| 7 | Epoch 12 | Soft – permutation | -0.548 | [-0.855, -0.261] |
| 8 | Distance-selected | Soft – uniform | -0.836 | [-1.152, -0.509] |
| 8 | Distance-selected | Soft – tie-uniform | -0.052 | [-0.314, +0.212] |
| 8 | Epoch 12 | Soft – uniform | -0.385 | [-0.672, -0.071] |
| 8 | Epoch 12 | Soft – tie-uniform | -0.130 | [-0.388, +0.141] |
| 8 | Risk-selected | Soft – uniform | -0.585 | [-0.892, -0.282] |
| 8 | Risk-selected | Soft – tie-uniform | -0.104 | [-0.357, +0.168] |

Differences above are Soft minus the named control in expected annotation
disagreement at 80% coverage, in percentage points. Intervals condition on three
fitted seed pairs and observed annotations. Previous outcomes informed later
phases; the follow-ups do not turn the reused image source into independent
confirmation. Read the paper for matched-set comparisons, adverse findings,
checkpoint sensitivities, and the limits of these results.

Current follow-up numerical ledgers are under
`final_revision_2026_10_20/training/results/` and
`final_revision_2026_10_20/diagnostics/`. Earlier phase ledgers remain in their
original folders. The root `reported_results.csv` is historical provenance and
does not supply the current manuscript's results.

## Exact paper and resources

- [Current main manuscript](manuscript/research-revision-v1.3.0.pdf)
- [Current supplement](manuscript/Supplementary-diagnostics-v1.3.0.pdf)
- [Portable editable LaTeX](manuscript/Manuscript-source-v1.3.0.zip)
- [Versioned release](https://github.com/mgunjan67/crowd-supervision-selective-fer/releases/tag/v1.3.0)
- [Exact resource index](RESOURCE_INDEX.json) and [archive quick-start](QUICKSTART.md)
- [Follow-up reproduction](FOLLOWUP_REPRODUCTION.md)

All 13 scientific resource archives now have verified public URLs, sizes and
SHA-256 release metadata. See the [delivery receipt](PUBLIC_DELIVERY_RECEIPT.json)
for the exact verification scope. The v1.3.0 scientific tag remains fixed on
`012521045c0f68203f9a5160cc5308d9cbaa9b99`; this later update changes delivery
documentation only. No journal submission or acceptance is claimed.

The repository root corresponds to the `paper_revision/` subtree used by the
portable resource extraction. Old releases remain unchanged. Historical README
and root instruction files are preserved under `HISTORY/pre-v1.3.0/`; older
edition-specific documents and root historical manifests describe those earlier
releases. Use this README and `REPRODUCIBILITY.md` for current commands.

## Recalculate results

Install the recorded numerical dependencies in a separate environment. New
follow-up replay imports PyTorch helpers but does not train or infer images:

```text
python final_revision_2026_10_20/training/analyze_followup.py
python final_revision_2026_10_20/training/ranking_followup.py
python final_revision_2026_10_20/training/export_followup.py
python final_revision_2026_10_20/diagnostics/analyze.py
python final_revision_2026_10_20/diagnostics/export_supplement.py
```

The [reproduction guide](REPRODUCIBILITY.md) separates retained-output replay,
optional licensed-data inference, and manuscript compilation. Checkpoint resources
are release assets, not Git objects. The current source manifest identifies the
files staged for this edition; it does not replace the original archive manifests.

## Local prototype and boundaries

`code/app.py` is an opt-in local expression-cue preview. Its Python backend must
run on the sender's machine for local processing. It does not implement recipient
transport, deploy the studied calibrated policies, or establish communication
benefits. Expression categories and crowd annotations do not establish internal
emotional states.

The earlier [TechRxiv preprint](https://doi.org/10.36227/techrxiv.175416003.30236370/v1)
describes a different experimental foundation. Its participant findings and
unverifiable historical curation/performance claims are excluded here. No new
participants or ratings were collected. The article states that no institutional
ethics approval or exemption determination was obtained for this secondary analysis.

No source face photographs, crops, audio, or video are distributed. Dataset access
and applicable original terms remain necessary for image-based reproduction.
FER+ annotations and supplied third-party files retain their original notices.
Public access does not assign a blanket software, model, or source-image license.
