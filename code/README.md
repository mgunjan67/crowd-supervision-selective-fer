# Shared model and prototype code

Start with `../ONLINE_RESOURCE_1_README.md` and `../reviewer_followup/REPRODUCIBILITY.md`.

The current manuscript's numeric ledger is `../reviewer_followup/reported_results.csv`.
The root `reported_results.csv` is explicitly historical, unverified provenance needed by legacy regression tests; it supplies none of the current paper's results.

The twelve-run study uses the frozen `../final_experiment/train.py` mechanics with the additional target definitions in `../reviewer_followup/train_controls.py`. Use `../reviewer_followup/retrain.py` to reproduce any condition into a separate directory. The older generic `train.py` in this code folder does not define the paper's controlled experiment.

`emotion_cue/` provides the common model, preprocessing, canonical class order, metadata contract, metrics and runtime safeguards. `app.py` is unchanged: local preview, not recipient messaging; no validated offline threshold is installed. The 24 inference checkpoints in Online Resources 2--5 satisfy its model contract. The reproduction guide deliberately retains the previous Soft seed-17 checkpoint as the example, independent of the new test results.

Historical configuration and label-quality helpers remain for regression tests, not as extra claimed experiments. The unvalidated VADER/affective-dissonance rewrite is not included in this paper release.
