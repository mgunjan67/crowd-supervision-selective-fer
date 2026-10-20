# Reproduction for repository revision v1.3.0

The repository root is the paper_revision subtree. The portable archive QUICKSTART uses an outer evidence directory. Use its exact extraction instructions for archived numerical replay.

From this repository root, after installing the recorded environment:

```text
python final_revision_2026_10_20/training/analyze_followup.py
python final_revision_2026_10_20/training/ranking_followup.py
python final_revision_2026_10_20/training/export_followup.py
python final_revision_2026_10_20/diagnostics/analyze.py
python final_revision_2026_10_20/diagnostics/export_supplement.py
```

The analysis validates retained prediction hashes. Source images and unreleased epoch checkpoints are absent; their retained hashes are identified as provenance rather than byte verification. Released checkpoints are optional for this replay.

Run numerical replay in a separate working copy: regeneration can change audit timestamps and ledger formatting. Do not stage regenerated outputs or rebuild immutable scientific archives solely for these metadata changes.

Compile main.tex with pdflatex, bibtex main, and two further pdflatex passes. Compile supplement.tex twice. The portable source ZIP includes these same entry names and dependencies.

See FOLLOWUP_REPRODUCTION.md for the complete training/inference commands, frozen protocols, source-image access requirements, loader amendments and retained/released checkpoint distinction. Its commands include the outer paper_revision/ prefix used by archive assembly; remove that prefix when running from this checkout root.

Earlier root instructions and manifests are historical; their original bytes are preserved under HISTORY/pre-v1.3.0/. No accepted journal article or journal DOI is claimed.
