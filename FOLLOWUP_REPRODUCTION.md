# Reproducing the October 20 follow-up

Use the combined extraction produced by `extract_resources.py`, and run all
commands below from its root. The original 24 training runs / 48 states and 36
external states remain unchanged. The new family is twelve training runs and
33 nominal checkpoint-rule states: six seven-class permutation states and 27 eight-class
Soft/Uniform/Tie-Uniform states. These are adaptive explanatory follow-ups on
the previously exposed image source, not independent confirmation.
Different rules may select identical weights; the full study has 81 named
checkpoint-rule states from 36 training runs, not necessarily 81 distinct weight sets.

Use a fresh `--purpose replay` extraction as a working copy. Keep the separate
`--purpose immutable-staging` extraction unchanged for publication/source staging.
Recalculation can regenerate summary/audit timestamps or ledger formatting;
compare scientific values and identifiers, not those execution timestamps. Do
not stage replay-mutated outputs, rewrite frozen member hashes or rebuild an
immutable scientific archive solely because replay metadata changed.

## Numerical replay without source photographs or GPU inference

Resource 9 contains the follow-up code, frozen protocols, amendments, training
records, selected-epoch predictions, inference probabilities, calculation ledgers,
and diagnostics. With Python 3.12, NumPy 1.26.4, Matplotlib, Pillow and the recorded
PyTorch environment available, run:

```text
python paper_revision/final_revision_2026_10_20/training/analyze_followup.py
python paper_revision/final_revision_2026_10_20/training/ranking_followup.py
python paper_revision/final_revision_2026_10_20/training/export_followup.py
python paper_revision/final_revision_2026_10_20/diagnostics/analyze.py
python paper_revision/final_revision_2026_10_20/diagnostics/export_supplement.py
```

The first command rebuilds new-state metrics, all paired contrasts and conditional
bootstrap intervals, accepted-set comparisons, target statistics, risk/coverage
curves, the summary ledger, and its LaTeX table from retained probabilities.
Its import of shared model/training helpers requires the recorded PyTorch software,
but the calculation does not train a model or run image inference. The second
command applies the separately frozen ranking extension to all 33 new nominal
states and six reused historical Soft comparators. It reports MSP, margin,
negative entropy, centered normalization and the calibration-only MSP/normalization
choice for every declared contrast. Its own protocol/freeze precedes the new
inference; earlier historical results were already known. No score novelty or
independent confirmation is claimed. The third command regenerates follow-up
artwork and supplementary tables from the completed actual result ledgers.

The diagnostics command replays the earlier centered logit-normalization sensitivity, calibration
selection of exponents, constant-category references, fixed temporal-aggregation
sensitivity, entropy/marginal statistics, and associated figures. All 48 historical
within-source states and all 36 historical external states remain represented.
The final command generates the new supplementary diagnostic section from the
verified result CSVs. It does not collect new annotations or create a new test set.

The protocol files, execution reports and final verification reports distinguish
fresh execution from inherited checks. An archive hash or numerical replay is
internal verification, not independent replication by another laboratory.

## Unit checks

```text
python -m pytest paper_revision/final_revision_2026_10_20/training paper_revision/final_revision_2026_10_20/diagnostics -q -p no:cacheprovider
```

These checks cover target invariants and output shape contracts, preservation of
the frozen sampler/augmentation order by the documented memory repair, and
diagnostic numerical identities. The retained test reports identify the tests
actually exercised for the current edition.

The independent numerical check can also be replayed without source images:

```text
python paper_revision/final_revision_2026_10_20/diagnostics/verify_completed_results.py
```

It reconstructs the ten-category fractions directly from the official annotation
CSV, checks all 99 new role archives and 51 result rows, and verifies all eight
5,000-replicate intervals using explicit resampled image copies rather than the
primary analysis's weighted-prefix calculation. This is an internal independent
implementation check, not an independent experiment or external replication.

The ranking extension retains `RANKING_EXTENSION.md`, `ranking_followup.py`,
`test_ranking_followup.py` and `ranking_extension_freeze.json`. Its outputs are
`results/new_ranking_{results,tuning,candidates,paired,means,inputs}.csv`,
`new_ranking_summary.json` and `new_ranking_table.tex`. The compact table can be
regenerated alone with `python paper_revision/final_revision_2026_10_20/training/ranking_followup.py --export-tex`. All 312 exponent
calibration rows and 1,248 candidate test outcomes are retained; test outcomes
never choose the exponent. The extension reconciles 33 MSP rows against the raw
primary analysis and does not replace that primary summary. Its 39 nominal states
overlap the earlier 48-state ranking analysis by six historical comparators;
together they cover 81 named checkpoint-rule states, not 87 independent states.
Generated synthetic layout/ranking fixtures are excluded from public evidence.

## Optional training and inference with authorized FER2013 access

Images are not redistributed. Obtain the exact FER2013 source identified in
`paper_revision/evidence/dataset_audit.json` through an access route you are
entitled to use. Place that archive at
`paper_revision/data/raw/fer2013-kaggle-v1.zip`. The official FER+ annotation CSV
is retained in Resource 1. Reconstruct the arrays and verify the existing role
manifest without replacing its source hashes:

```text
python paper_revision/scripts/prepare_fer2013.py
python paper_revision/final_experiment/data.py
```

The scripts expect the recorded row order, image pixels and annotation revision.
A hash mismatch must be investigated; do not rewrite the freeze to suppress it.
The recorded inference/training environment is Python 3.12, PyTorch 2.8.0 with
CUDA 12.8, torchvision 0.23.0, and NumPy 1.26.4. Follow the shared requirements
and source model-weight terms. A CUDA-capable device is required by these entry
points; CPU numerical replay above does not use source images.

```text
python paper_revision/final_revision_2026_10_20/training/run_prefetched.py --all
python paper_revision/final_revision_2026_10_20/training/evaluate_followup.py
python paper_revision/final_revision_2026_10_20/training/analyze_followup.py
python paper_revision/final_revision_2026_10_20/training/ranking_followup.py
python paper_revision/final_revision_2026_10_20/training/export_followup.py
```

`run_prefetched.py` executes the frozen `train_followup.py` plan with the two
documented loader repairs. The original worker-memory failure, subsequent
bounded prefetch change, sampler-order proofs, and each run's last committed
epoch at amendment application are retained. Model objectives, batches, model
initialization checks, splits, augmentation decisions, training budget and
checkpoint selection rules are unchanged by those execution repairs. The
unamended training script remains preserved with its original freeze.

All twelve runs must complete before evaluation exports any calibration/test
predictions. Eight-class models have distance-selected, selection-risk-selected
and epoch-12 states; the seven-class permutation has distance-selected and
epoch-12 states. The exported inference files and each probability array are
hashed in their per-state evaluation records. Stored training records distinguish
selection-only activity from later calibration/test inference.

Full epoch checkpoints remain local and are individually hashed in
`training/ALL_CHECKPOINT_HASHES.json`. Only the 33 documented inference states
are supplied in Resources 10–13; the package does not claim that every epoch's
weights or optimizer-resume state was released. Cross-device bitwise equality
is not guaranteed. No new naturalistic evaluation, frame-level ground truth,
participant study or validated communication benefit is claimed.
