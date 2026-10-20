# Reproduction and manuscript build — 20 October 2026

This is the authoritative quick-start for the October 20 revision of
“Evaluating Crowd-Label Supervision for Selective Facial-Expression Recognition.”
Instructions retained inside historical evidence or checkpoint archives describe
those earlier editions. Use the filenames and assembly steps below for this edition.

## 1. Compile the current article and supplement

Extract **Manuscript-source.zip into a separate empty directory**. It includes
the Springer class, numeric bibliography style, figures, bibliography, and two
entry files: **main.tex** and **supplement.tex**. From that directory run:

```text
pdflatex -interaction=nonstopmode -halt-on-error main.tex
bibtex main
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error supplement.tex
pdflatex -interaction=nonstopmode -halt-on-error supplement.tex
```

The main PDF is `main.pdf`; the supplement is `supplement.pdf`. These commands
do not execute training or analysis. Build logs and the clean-source verification
report record the tested compiler and whether the delivered pages match.

## 2. Verify and assemble scientific resources

Download the resource ZIPs listed in **RESOURCE_INDEX.json** into one directory.
Resources 1 and 8 contain retained predictions and the preceding scientific replay
code. Resource 9 contains the October 20 follow-up code, protocols, retained
outputs, and diagnostic results. Resources 2–7 and any separately listed new
checkpoint resources are optional for numerical replay.

Place `extract_resources.py` and `RESOURCE_INDEX.json` beside the ZIPs and run:

```text
python extract_resources.py --destination evidence --purpose replay
```

For licensed-data inference or training reproduction, also obtain the checkpoint
resources and use:

```text
python extract_resources.py --destination evidence --include-weights --purpose replay
```

The utility verifies the archive SHA-256 and **every member hash**, rejects path
traversal, refuses conflicting files, and stores one separately named manifest
per resource in `evidence/resource-manifests/`. It never overwrites historical
manifests. Do not merge archives manually: historical Resources 2–7 share a root
manifest filename and need the prefix mapping in this utility.

Numerical replay uses this mutable working copy. Regeneration may change audit
timestamps or ledger formatting without changing scientific values. Keep a second
fresh extraction made with `--purpose immutable-staging` if byte-identical source
staging is needed; never use the replay working tree for repository publication.
The installer records the purpose, and repository staging rejects replay roots.

`--download` retrieves only resources with an actually verified public URL in
the index. A proposed release URL is not treated as an available resource.
Without that flag the utility does not access the network.

## 3. Recalculate reported numerical results

After assembly, run from `evidence/`:

```text
python paper_revision/submission_reader_2026_10_13/audit.py
python paper_revision/submission_reader_2026_10_13/distribution_results.py --output-dir paper_revision/tmp/distribution-replay
python paper_revision/submission_reader_2026_10_13/diagnostics.py --output-dir paper_revision/tmp/diagnostics-replay
python paper_revision/recovery_2026_10_10/selector_analysis.py
python paper_revision/ravdess_transfer_2026_10_11/recompute_results.py
```

These historical result replays use retained probabilities and annotation vectors;
they require neither photographs nor neural-network inference. Python 3.12 and
NumPy 1.26.4 are the recorded numerical environment. October 20 additions and
their exact entry commands are listed in `FOLLOWUP_REPRODUCTION.md` alongside
this quick-start. Its verification status distinguishes new execution from
inherited checks. Optional plotting requires Matplotlib; PDF verification uses
PyMuPDF and Poppler.

## 4. Scope and data access

The resource index separates local existence, verified public delivery, and
submission-portal delivery. The old public repository alone does not contain
all evidence for the current article. No face photographs, crops, audio, or
videos are redistributed. Media acquisition must follow the original source
terms; numerical replay does not obtain those media. The historical weights
remain byte-identical and are referred to in place rather than copied again.

The TechRxiv preprint describes an earlier study. Its historical participant
results and unverifiable curation/performance claims are not evidence for the
current manuscript. The present analyses concern annotation distributions and
expression categories, not proof of internal emotion or validated messaging
benefits.
