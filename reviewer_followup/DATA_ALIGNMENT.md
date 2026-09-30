# Current data and category alignment

All current conditions use every Training row with the same supported-vote weight. They do not apply the majority-only relabeling rule from an earlier exploratory experiment.

| Original FER2013 integer | Original name | Canonical model index | Named FER+ vote column |
|---:|---|---:|---|
| 0 | angry | 0 | anger |
| 1 | disgust | 1 | disgust |
| 2 | fear | 2 | fear |
| 3 | happy | 3 | happiness |
| 4 | sad | 5 | sadness |
| 5 | surprise | 6 | surprise |
| 6 | neutral | 4 | neutral |

FER+ votes are read by column name, not by incidental CSV column position. The ten-category evaluation order is anger, disgust, fear, happiness, neutral, sadness, surprise, contempt, unknown, NF. The final three have no corresponding output neuron. Every vote row is normalized by its total recorded votes.

The official FER+ README and pinned `src/generate_training_data.py` align the FER2013 and FER+ CSVs by original row position. Image-name strings in the annotation file include blanks and duplicates and are not used as join keys. Partition membership, row IDs, original class labels and pixel hashes are checked independently. The original-to-canonical mapping above is also checked directly against the retained raw FER2013 CSV by `verify_raw_source.py`; that script requires the legally acquired image archive, which is not redistributed.

The official conversion script advances its row index even when an output filename is blank; blank names suppress PNG writing, not index advancement. This was rechecked directly against the pinned official source on 22 September 2026. Our controlled study retains all Training rows and assigns zero supervised loss to zero-supported-mass rows instead of adopting a separate filename-based exclusion.

Pinned official conversion source: https://raw.githubusercontent.com/microsoft/FERPlus/ae2128abf776409c93e50cf7c9d87180673314e6/src/generate_training_data.py

`audit.py` independently compares all 72 released prediction arrays against the original class/hash manifest and official vote CSV. A majority-only F1 subset is an evaluation reference, not a training-data exclusion. Its label-disagreement percentage does not prove that either reference is emotional truth.

The FER2013 file came from a community mirror. Internal hash/row consistency does not independently authenticate every underlying photograph or establish its rights. No source images are redistributed.
