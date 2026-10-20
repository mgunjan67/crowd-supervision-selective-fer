# Exploratory diagnostic follow-up

The protocol was saved before these new calculations; historical scientific
outcomes were already known. No model training or inference occurs here.

From the repository root:

```powershell
.venv-paper/Scripts/python.exe -m pytest paper_revision/final_revision_2026_10_20/diagnostics/test_diagnostics.py -q -p no:cacheprovider
.venv-paper/Scripts/python.exe paper_revision/final_revision_2026_10_20/diagnostics/analyze.py
.venv-paper/Scripts/python.exe paper_revision/final_revision_2026_10_20/diagnostics/verify.py
```

Requires Python, NumPy and Matplotlib; pytest is only needed for invariant tests.
All historical prediction archives, annotations and manifests must be restored
to their original `paper_revision/` paths. Scientific calculations and plots are
fully regenerated from those inputs, and their hashes are audited.

`summary.json` is the manuscript-integration digest. The CSV ledgers retain every
tested state and seed; compact TeX tables and SVG/PNG scientific plots derive
from those ledgers. `analysis_audit.json` records inputs, outputs, methods and
cluster counts. `verification.json` independently reconciles the original MSP
and eight-frame results against historical ledgers. `tests.xml` records tests.

The centered-logit formula is Xia et al. (ICLR 2025), equations 16 and 28:
https://arxiv.org/html/2403.14715 . Its probability reconstruction follows from
mean-centering log-softmax; finite stored probabilities have rounding error.
This uses a declared small exponent grid and crowd-disagreement AURC on the
calibration role, rather than reproducing the paper's original training study.
An uncentered-logit normalization cannot be reconstructed from probabilities.

All outputs remain exploratory after repeated exposure of the original test
source. Frame aggregation is a fixed sensitivity without frame-level human
annotations. Class entropy and marginal statistics document differences, not a
causal attribution. Image-cluster uncertainty is conditional on fitted seeds,
annotation counts and the reused source; it is not uncertainty over new people,
raters, datasets or future training runs. The 3,275 test images comprise 3,268
exact-pixel clusters: 3,261 singletons and seven pairs. In each historical
bootstrap replicate, all models share the sampled cluster multiplicities,
coverage uses ceil(0.8 * resampled image count), and three paired-seed
differences are averaged before taking the interval across replicates.
