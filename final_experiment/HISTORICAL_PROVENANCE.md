# Historical preprint values are not final experimental results

The TechRxiv version reported weighted-F1 values 0.7123 and 0.7677 on FER2013,
and 0.8474, 0.8971, and 0.9228 on RAF-DB. These five values remain in the root
`reported_results.csv` as historical records, because their source checkpoints,
per-image predictions, and manual curation manifests were not retained.

They are excluded from the final manuscript's performance claims, figures,
tables, averages, and uncertainty intervals. Earlier inconsistent per-class
averages, cross-paper accuracy/F1 comparisons, compound-expression results,
and participant findings are not reconstructed or endorsed.

The earlier preprint describes manual relabeling, exclusions, and targeted
augmentation. That account cannot be independently reconstructed from the
retained artifacts. The new experiment uses public FER+ votes with complete
row-level alignment; it is not a fabricated recovery of the historical manual
changes.

The final result source is **`final_experiment/reported_results.csv`**, generated
from six new hard/soft matched runs. Earlier nine-run exploratory results remain
preserved locally in their original editions but are not pooled with this study.
