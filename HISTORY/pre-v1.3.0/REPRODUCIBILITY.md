# Reproducibility entry point

Crowd Supervision and Vocabulary-Mismatched Selective Facial-Expression Prediction
SN Computer Science
Gunjan Kumar Mishra; Bijaya Ghimire; Badri Raj Lamichhane; Alan Shah
Corresponding author: Badri Raj Lamichhane, School of Information, Computer and Communication Technology, SIIT, Thammasat University, Thailand; d6622300231@g.siit.tu.ac.th


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
