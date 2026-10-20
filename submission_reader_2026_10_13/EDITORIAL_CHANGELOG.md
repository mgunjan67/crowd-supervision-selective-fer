# Completed reader-focused revision — 13 October 2026

Baseline: the sealed 33-page academic edition in `submission-academic-2026-10-13`.
This separate edition applies both supplied version-4 editorial packs and the
accompanying first-reader review. It is a writing-only revision. The existing
seven Results themes and Discussion are preserved.

| Request | Completed change | Source location |
| --- | --- | --- |
| Clearer optional title | Used “Evaluating Crowd-Label Supervision for Selective Facial-Expression Recognition,” the second pack's option; synchronized main, supplement and cover. | `source/manuscript.tex`; `source/generated/supplement.tex`; cover files |
| Abstract assumes terminology | Four separately readable Purpose/Methods/Results/Conclusion paragraphs, 233 words. Defines the primary disagreement endpoint; retains adverse backbone/transfer findings. Methods wording shortened to fit the title page without smaller type. | `source/generated/abstract.tex` |
| Introduction reaches the study too soon | Replaced with crowd-vote context, coverage/disagreement definitions, control rationale, unsupported categories, study scope and brief prototype motivation. Largest target probability is explicitly distinguished from fitted-model confidence. | `source/generated/introduction.tex` |
| Preserve research questions | All three original questions retained verbatim, following the explanatory foundation. | Introduction |
| Avoid duplicated literature exposition | Shortened the first Related Work paragraph while retaining FER+, Washington/CAFE, seven-output distinction and existing citations. BibTeX regenerated all numbering. | Related Work 2.1 |
| Explain experiment before chronology | Added the experiment-oriented overview before the unchanged adaptive history, including shared conditions, separate image roles, two endpoints, fixed-image comparisons and frozen external evaluation. | Methods 3.1 |
| Explain disagreement concretely | Added the explicitly constructed six-happy/four-neutral example, with disagreements 0.4 and 0.6. States why better probability allocation need not change category disagreement. | Methods 3.6 |
| Make F1 reference/population changes explicit | Replaced the requested paragraph using the existing Soft row: 0.6685 versus 0.9270 on the same 2,484 images; 0.5722 versus 0.6685 under the original reference across full/subset populations. | Results 4.3, Table 7 |
| Reduce ledger narration | Consolidated the 24-run/48-state inventory in Methods 3.1; all-score coverage accounting remains in Methods 3.8. Results 4.2 and 4.5 retain findings and direct Supplement S2/S14/Fig. S7 pointers. | Methods 3.1/3.8; Results 4.2/4.5 |
| Keep definitions and explanations together | Float barriers and same-page blocks keep the data-role paragraph, target definitions/objectives and checkpoint-optimum explanation intact. Early tables permit in-text placement. | Methods 3.2–3.4; `phases.tex`; `roles.tex` |
| Verify actual supplement | Its full expanded content is unchanged except title and current reproduction paths. All 20 table and 12 figure labels retained; main-text S2, S14, S7 and S11 references checked against it. | `source/generated/supplement.tex` |
| Deliver actual source, not only replacement prose | New source ZIP, compiled main/supplement, cover, source diff, preservation record and this change log. Prior sealed files remain untouched. | Delivery directory |

No method or unique result was moved from the main article to the supplement in
this pass. The unchanged Figure 1 summary paragraph moves before the Table 2/
Figure 1 floats within Methods 3.2, avoiding a stranded paragraph after them.
Omitted abstract details remain in their existing Results tables and
supplementary analyses. The only consolidated content is repeated inventory
narration, with destinations specified above. The detailed preservation/location
plan made before editing is `EDITORIAL_PLAN_AND_LOCATION_MAP.md`.

The source comparison checks 8 displayed equation blocks, 101 inline-math
occurrences, ordered numeric contents of 13 main and 20 supplementary tables,
unchanged figure bytes, eight diagnostic CSVs, full-precision claims, bibliography,
numeric macros, declarations, research questions, adaptive history and protected
Methods passages. It also verifies the F1 prose against the original table's
condition, columns and populations. Automated checks support the editorial
comparison; they do not prove semantic equivalence or publication quality.

Fresh compilation, archive validation, clean-source PDF comparison and actual
visual review are recorded separately in the final preservation report. No
training, inference, bootstrap, statistical reanalysis, selector search, figure
regeneration or new literature verification was performed. Earlier empirical
audits and 126 tests are inherited, not fresh executions. This bounded revision
does not establish acceptance likelihood or certify ethics/data compliance.
