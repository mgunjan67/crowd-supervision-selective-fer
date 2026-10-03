# Evidence revision v1.1.0

The original two experimental phases remain unchanged in `final_experiment/` and `reviewer_followup/`. The third adaptive phase lives in `evidence_revision/`. Prior outcomes were known; this is not preregistration or independent confirmation.

## CPU diagnostics without face images

From the extracted package root:

```
python evidence_revision/diagnostics.py
python evidence_revision/audit_targets.py
python evidence_revision/analysis.py
python evidence_revision/audit.py
python evidence_revision/build_revision.py
```

The merged `evidence_revision/reported_results.csv` is the sole numeric input to the manuscript generators. The ledger includes unchanged earlier calibration diagnostics plus the new targeted control and fixed-image/reference diagnostics. Checkpoint auditing requires PyTorch and all six Tie-Uniform inference weights from Online Resource 6. Face images are not needed for arithmetic regeneration.

Full optimizer-bearing training checkpoints are retained on the original workspace, not redistributed. The public audit always verifies inference weights and predictions; it recomputes original training-checkpoint hashes only when those optional files exist and explicitly records the number checked. Recorded source hashes are provenance metadata, not a claim of public access to those files.

## Image-dependent training and inference

The immutable freeze includes acquired data hashes; acquire the same lawful archive and official FER+ annotations, then follow earlier preparation instructions. Dataset licensing and access are separate from this code. No face images are in the release.

`python evidence_revision/audit_existing.py` additionally verifies original pixel hashes, original labels and named votes against the acquired raw FER2013 archive. It is therefore image/archive-dependent, unlike the prediction-only arithmetic regeneration above.

```
python evidence_revision/retrain.py --seed 17 --output-dir rerun-tie-uniform-17
python evidence_revision/reproduce_inference.py --output-json new-replay-report.json
```

Repeat the fresh-directory training command for seeds 42 and 89 if desired. It rejects nonempty destinations and preserves released configurations. The original-workspace `train_tie_uniform.py` wrapper trained exactly three 12-epoch runs and checked matched initialization/configuration against retained Uniform controls; it is a resume driver, not a command for overwriting the released run directories. Its original `evaluate.py` requires full training checkpoints and all three completed runs. The public `reproduce_inference.py` instead reloads released inference weights and reports numeric differences against all eighteen retained prediction files without needing optimizer checkpoints. Exact numerical replay is hardware/environment dependent; recorded BF16 GPU predictions are the released evidence, while the prototype uses FP32.

Selected and epoch-12 weights retain the seven-class single-gate app contract. Original Online Resources 2--5 contain 24 weights; Online Resource 6 adds six. No new threshold policy is installed in the app and the local prototype's scope is unchanged.

## PDF build

Run `python evidence_revision/build.py` from a machine with pdflatex and bibtex available. It compiles the multi-file Springer source and supplementary document and renders every page. Automated success does not itself establish visual inspection, scientific validity, external replication or journal acceptance.

All fifteen runs, all states and all diagnostic directions must be retained. The supplementary package supplies source/hash audits, per-run logs, probability files and checksum manifests. Approval of the prior manuscript does not establish approval of these later results; final author review is needed before actual journal submission.
