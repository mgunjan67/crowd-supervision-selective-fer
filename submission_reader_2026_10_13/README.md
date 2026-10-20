# Reviewer reproducibility package

**Evaluating Crowd-Label Supervision for Selective Facial-Expression Recognition**

Gunjan Kumar Mishra, Bijaya Ghimire, Badri Raj Lamichhane, Alan Shah.  
Corresponding author: Badri Raj Lamichhane, d6622300231@g.siit.tu.ac.th.  
SN Computer Science; reader-focused editorial edition, 13 October 2026.

This writing-only edition applies the two supplied version-4 editorial packs.
It explains crowd votes, selective prediction and the study comparison before
introducing the detailed analysis history. The four-paragraph abstract, new
Introduction, Methods overview, constructed metric example and reference-specific
F1 paragraph improve the first-reader explanation. The preceding academic
edition's seven Results themes and Discussion are retained. The experiments,
numeric table contents, equations, figure assets, bibliography, declarations
and scientific limitations are unchanged from that sealed 13 October edition.
No training, inference, bootstrap calculation, empirical analysis or figure
generation was repeated. The supplement changes only its title and current
reproduction paths. EDITORIAL_CHANGELOG.md records the completed changes.

This package reports an empirical reliability study, not a novel architecture
or rejection method. There are 24 training runs, 48 selected/epoch-12 states,
and 36 external states. No training is required for numerical replay. All
adverse results and seeds are retained. `SCIENTIFIC_PROVENANCE.md` distinguishes
the adaptive phases and historical preprint from current evidence.

## Extraction and optional retained-science replay

Extract OnlineResource1.zip and OnlineResource8-external-transfer.zip into the
same **fresh directory**. Both contain a `paper_revision/` tree; any common
immutable source members are byte-identical. Do not extract over a research
working tree. The following retained-science replay commands are available from
the fresh directory. They were exercised in preceding editions; they were
**not rerun for this writing-only revision**:

```text
python paper_revision/submission_reader_2026_10_13/audit.py
python paper_revision/submission_reader_2026_10_13/distribution_results.py --output-dir paper_revision/tmp/distribution-replay
python paper_revision/submission_reader_2026_10_13/diagnostics.py --output-dir paper_revision/tmp/diagnostics-replay
python paper_revision/recovery_2026_10_10/selector_analysis.py
python paper_revision/ravdess_transfer_2026_10_11/recompute_results.py
```

The first command independently expresses the primary within-source/fixed-set
contrasts, source joins, calibration policies and the retained diagnostic matrices
and temporal summaries. It requires Python 3.12 and NumPy 1.26.4, no torch,
weights, photographs or videos. The final command reproduces the original
external ledgers and 5,000 actor-bootstrap replicates with the frozen numerical
implementation; the selector command reproduces all within-source score CSVs.
They write derived outputs in this isolated extraction only. Hash/byte equality
and independent arithmetic are distinct checks, not external replication.

The distribution command recomputes all 48 previously recorded ten-category
squared-distance values on the identical 3,275 primary test images. It sums
squared coordinate differences after padding seven outputs with three zeros,
then averages over images. Full votes are not support-renormalized. The
tables are descriptive reports of this existing endpoint, not
new confirmatory experiments or evidence of calibrated deployment confidence.

## Figures and numerical manuscript fragments

With Matplotlib and PyMuPDF/Pillow available, run:

```text
python paper_revision/submission_reader_2026_10_13/regenerate.py
```

This optional command regenerates every used figure from the verified
ledgers/retained outputs and editable plotting code, plus the numerical tables.
It then applies the locked academic prose layer in `editorial/` through
`apply_editorial.py`. The layer refuses changed numerical claim ledgers or
input CSVs so that it cannot silently attach the old narrative to new results.
Historical plot generators run in a staging directory and do not edit the old
research editions. The internal tables retain their verified source ledgers;
`CLAIM_ARTIFACT_COMMAND_MAP.csv` records each display's evidence and command.
Raw/merged references never change predictions or accepted IDs. Mean risks
are averages of seed-specific risks, not an ensemble of probabilities.

## Editable source and PDF build

Extract Manuscript-source.zip into a separate fresh directory. It contains
`main.tex`, `Supplementary-diagnostics.tex`, the complete nested editable source,
bibliography, figures, Springer class and numerical bibliography style. From
that source directory run, in order:

```text
pdflatex -interaction=nonstopmode -halt-on-error main.tex
bibtex main
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error Supplementary-diagnostics.tex
pdflatex -interaction=nonstopmode -halt-on-error Supplementary-diagnostics.tex
pdflatex -interaction=nonstopmode -halt-on-error Supplementary-diagnostics.tex
```

Alternatively, in the combined evidence extraction use the installed LaTeX
compiler on PATH and run `python paper_revision/submission_reader_2026_10_13/build.py`.
That command checks the abstract, bibliography, prohibited claims, layout
warnings and embedded fonts, and renders every page for human inspection.
Byte-identical PDF hashes are not expected across build dates/toolchains;
the current clean-source delivery test checks page count, extracted page text
and rendered page pixels as well as warnings on the recorded toolchain.

The fresh checks for this edition validate archive member hashes, exact
preservation of the evidence and figure bytes, idempotent application of the
locked prose layer, and independent compilation of the source ZIP. Only
`apply_editorial.py`, LaTeX and BibTeX are executed in that clean-source test;
no retained-science replay command above is executed. Inherited numerical
audits and tests are identified in `checks/INHERITED_VALIDATION.json`; they are
internal verification from preceding editions, not new tests or independent
replication of this editorial revision.

## Optional licensed-data inference — not needed for numerical replay

The scientifically checked predictions and all 48 released inference weights
are retained. Extract Online Resources 2–7 into `paper_revision/` under the
combined extraction, preserving their relative paths. The unchanged checkpoint
ZIPs retain their earlier model cards/reproduction texts as provenance; older
release locations and extraction-root instructions in those texts are not the
current delivery instructions. Use this README and REVIEWER_RESOURCE_MAP.md for
the accompanying edition. Acquisition is not performed by the CPU audit.
FER2013 was obtained through a community mirror;
review its lawful access/terms yourself. FER+ vote annotations come from the
official Microsoft repository. No mirror license is treated as proof of rights
to all underlying photographs. RAVDESS media are acquired from the original
[Zenodo record](https://zenodo.org/records/1188976) under its source terms; no
media are redistributed here.

On the recorded Windows/CUDA software stack and after separately authorized
media acquisition:

```text
python paper_revision/ravdess_transfer_2026_10_11/acquire.py --all --workers 4
python paper_revision/ravdess_transfer_2026_10_11/prepare.py --all --workers 2
python paper_revision/ravdess_transfer_2026_10_11/infer.py
python paper_revision/ravdess_transfer_2026_10_11/verify.py
```

The frozen protocol's platform/source/checkpoint checks deliberately fail if
prerequisites differ; do not rewrite its freeze or amendment to suppress a
failure. Original paths are Windows-native; numerical replay normalizes them
without replacing statistical functions. CPU inference is not guaranteed to
match CUDA BF16 bitwise. No new data acquisition, training or inference was
executed in this revision. The optional path and its runtime/media/checkpoint
prerequisites were audited in preceding editions. It was **not rerun from the
clean extraction** because that would reacquire media/duplicate inference.

## Evidence and boundaries

`SHA256SUMS.json` in each newly built ZIP covers every other member.
`CHECKPOINT_AVAILABILITY.csv` maps all 48 weights to existing verified archive
members. `REVIEWER_RESOURCE_MAP.md` gives the portable assembly map; the separate
author-facing `ONLINE_RESOURCE_MAP.md` records verified local paths and sizes. The full
preceding workspace audits additionally verified primary workbook response
mapping, 1,440 media hashes, all weights and independent bootstrap recomputation.
Their machine-readable reports are included as inherited internal verification,
not newly executed checks for this edition.

The raw source workbook's response slots are not verified persistent rater IDs.
The primary external interval is conditional on three fitted seed pairs and
observed votes; two seed contrasts are negative and one larger positive. Raw
and calm/neutral-merged references differ only in scoring. Frame summaries are
descriptive, not frame accuracy. Institutional approval/exemption was not
obtained; no authority's determination is manufactured by this package.
