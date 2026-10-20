# Historical academic revision map — retained for provenance

This map describes the preceding academic edition's changes. For the current
reader-focused revision, use `EDITORIAL_CHANGELOG.md` and
`EDITORIAL_PLAN_AND_LOCATION_MAP.md`. No unique main-text method or result was
moved to the supplement in the current pass. The seven Results themes,
Discussion and supplementary section order established below remain unchanged.

This edition responds to the supplied academic-writing review. It changes the
argument, paragraph construction, organisation and presentation, not the research.
The preceding sealed visual edition remains intact. No training, inference,
bootstrap, selector search or additional empirical analysis is part of this edit.

## Old-to-new section map

| Previous location | Revised location and editorial treatment |
| --- | --- |
| Abstract | Replaced as a complete structured abstract: problem and comparison logic precede run accounting; the primary selective comparison precedes secondary distributional/backbone findings. |
| Introduction | Evaluation problem → supervision-control rationale → unchanged three research questions → bounded contribution → brief application motivation. |
| Introduction chronology and dispersed Methods history | Methods 3.1, Study design and analysis hierarchy. Adaptive phases, known earlier outcomes, primary/secondary roles, exploratory analyses and the external plan remain explicit. |
| Related work 2.1–2.2 | Synthesised by supervision, selection and calibration questions. Prior contributions are stated positively; non-comparability is explained where relevant, without repeated novelty denials. |
| Main acquisition details, raw-source bookkeeping, exact freeze timestamp and diagnostic-addendum date | Supplementary S9, with explicit main Methods pointers. Material provenance and analysis timing remain in main Methods. |
| Results 4.1, 4.2, 4.5 | New 4.1: effects of supervision targets and checkpoint selection. |
| Results 4.8, 4.9 | New 4.2: distributional agreement and backbone sensitivity; these retain their secondary/adaptive status. |
| Results 4.4 | New 4.3: accepted-image sets and annotation references. |
| Results 4.3, 4.6 | New 4.4: effects of unsupported annotation categories. |
| Results 4.10 | New 4.5: sensitivity to confidence scores; exploratory status retained. |
| Results 4.7 | New 4.6: calibration-to-test transfer. |
| Results 4.11 | New 4.7: external clip evaluation and diagnostic analyses. |
| Duplicated external Discussion opening and reporting checklist | A connected Discussion synthesises the distinction between distributional and selective outcomes, relates it to prior literature and develops evidence-linked implications. |
| Dispersed scope denials | Necessary limits consolidated into the appropriate Methods definitions, uncertainty captions, implementation boundary and Limitations. No causal, calibration or deployment claim is strengthened. |
| Supplement architecture/prototype figures interrupting empirical diagnostics | The supplement mirrors the main Results sequence, then groups architecture/prototype material and execution provenance. A navigation page identifies the sections and displays. |

## Terminology and presentation

- `Validation-selected Swin-T` identifies the model and checkpoint condition.
- `Epoch-12 checkpoint` replaces compressed prose labels; original immutable
  ledger identifiers such as `epoch12` remain unchanged in data and code.
- `Accepted images` and `accepted set` replace interchangeable informal terms.
- Supported-category training distributions are distinguished from the complete
  annotation distributions used in evaluation.
- Original FER2013 labels, crowd-majority references and raw/merged RAVDESS
  responses are named explicitly rather than called an undifferentiated reference.
- Risk targets and supervision conditions have distinct table headings.
- The paired-endpoint caption retains populations, units, directions and the
  conditional bootstrap interpretation without repeating the Discussion.
- The external accounting table permits in-text placement instead of a top-only
  float; final pagination is checked after compilation without shrinking type.

## Numerical information moved out of repetitive prose

The following remain available in existing tables, figures or released ledgers.
They are not removed from the evidence or replaced by new estimates:

- Individual primary Swin-T seed contrasts and all Tie-Uniform seed contrasts.
- Exhaustive ResNet-18 selected/epoch-12 seed contrasts and cross-set values.
- Category-specific all-image unsupported vote fractions.
- Confidence-score mean contrasts already printed in the score table.
- External epoch-12 merged-reference seed contrasts; the main text still states
  that all three favor Soft, while the principal external mixed-seed pattern
  remains explicit in the main table and prose.
- External accounting components already present in the accounting table.

Original contrast confidence intervals previously present only in prose are
retained in the first Results subsection. Numerical table contents, their signs,
ordered columns and denominators are checked against the preceding edition.

## Preservation and release checks

The fresh `editorial_preservation.json` checks exact displayed equations,
ordered numerical table contents, all figure-asset bytes, eight diagnostic CSVs,
the full-precision numerical claim ledger, bibliography, numeric macros,
declarations, unchanged research questions and the prior sealed files.
Manual comparison additionally checks architectures, preprocessing, data roles,
seed/checkpoint definitions, uncertainty scope and analysis designations.
This is an editorial preservation check, not new research or independent
replication. Automated keyword checks are not a proof of semantic equivalence.

The editable source package is independently compiled after extraction. The final
PDFs are rendered and visually inspected for overflow, overlapping content,
stranded headings, display placement, readable lettering and consistent numbering.
The package retains the Springer submission template rather than imitating
published journal branding or forcing a two-column production layout.
