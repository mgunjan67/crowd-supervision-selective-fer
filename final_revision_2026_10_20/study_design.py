"""Extend the native study schematic with the frozen explanatory follow-up.

The figure contains design counts only, never model outcomes. Historical figure
sources and exports are not modified. Coordinates are in inches.
"""
from pathlib import Path
import hashlib
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle
import fitz

HERE = Path(__file__).resolve().parent
OUT = HERE / "source" / "figures"
INK = "#222222"


class Drawing:
    def __init__(self):
        self.fig = plt.figure(figsize=(6.5, 7.85))
        self.ax = self.fig.add_axes([0, 0, 1, 1])
        self.ax.set(xlim=(0, 6.5), ylim=(0, 7.85))
        self.ax.axis("off")
        self.texts = []
        self.containers = []

    def text(self, x, y, label, ha="center", bold=False, size=11.5):
        text = self.ax.text(
            x, y, label, fontsize=size, ha=ha, va="center", color=INK,
            fontweight="bold" if bold else "normal", linespacing=1.20)
        self.texts.append(text)
        return text

    def node(self, x, y, width, height, label, size=11.5):
        box = Rectangle((x, y), width, height, ec=INK, fc="white", lw=.8)
        self.ax.add_patch(box)
        text = self.text(x + width / 2, y + height / 2, label, size=size)
        self.containers.append((text, box))

    def arrow(self, points):
        for left, right in zip(points[:-2], points[1:-1]):
            self.ax.plot([left[0], right[0]], [left[1], right[1]],
                         color=INK, lw=.8)
        self.ax.add_patch(FancyArrowPatch(
            points[-2], points[-1], arrowstyle="-|>", mutation_scale=9,
            color=INK, lw=.8))

    def panel(self, y, letter, title):
        self.text(.12, y, letter, ha="left", bold=True)
        self.text(.37, y, title, ha="left")

    def audit(self):
        self.fig.canvas.draw()
        renderer = self.fig.canvas.get_renderer()
        overlaps = []
        for index, left in enumerate(self.texts):
            for right in self.texts[index + 1:]:
                if left.get_window_extent(renderer).overlaps(
                        right.get_window_extent(renderer)):
                    overlaps.append([left.get_text(), right.get_text()])
        overflow = []
        for text, box in self.containers:
            tb = text.get_window_extent(renderer)
            bb = box.get_window_extent(renderer)
            if not (bb.contains(tb.x0 - 2, tb.y0 - 2)
                    and bb.contains(tb.x1 + 2, tb.y1 + 2)):
                overflow.append(text.get_text())
        assert not overlaps and not overflow, (overlaps, overflow)
        return dict(text_overlap_pairs=overlaps, container_overflows=overflow)


def main():
    plt.rcParams.update({"font.family": "Arial", "pdf.fonttype": 42,
                         "ps.fonttype": 42})
    OUT.mkdir(parents=True, exist_ok=True)
    d = Drawing()
    d.panel(7.66, "a", "Experiment families; paired seeds 17, 42 and 89")
    d.node(.12, 6.45, 1.94, .96,
           "Original phases\nSwin-T + ResNet-18\n24 runs; 48 states")
    d.node(2.26, 6.45, 1.84, .96,
           "7-output Swin-T\nPermutation\n3 runs; 6 states")
    d.node(4.30, 6.45, 2.04, .96,
           "8-output Swin-T\nSoft / Uniform /\nTie-Uniform\n9 runs; 27 states")
    d.text(3.25, 6.20,
           "Final follow-up: 12 runs; earlier outcomes were known.")
    d.text(3.25, 5.98,
           "States are checkpoint rules, not independent training fits.")

    d.panel(5.62, "b", "Matched optimization and checkpoint selection")
    d.node(.12, 4.69, 1.62, .70, "Training images\nand FER+ votes")
    d.node(2.04, 4.61, 2.18, .86,
           r"$\ell_i=-w_i\sum_{c=1}^{K}t_{ic}\log p_{ic}$"
           + "\nAdamW; 12 epochs")
    d.node(4.62, 4.69, 1.72, .70,
           r"$\theta_1,\ldots,\theta_{12}$" + "\nCandidate states")
    d.arrow([(1.74, 5.04), (2.04, 5.04)])
    d.arrow([(4.22, 5.04), (4.62, 5.04)])
    d.text(.14, 4.385,
           "Within backbone, vocabulary and seed:\n"
           "same initialization, sample order and flips",
           ha="left", size=10.5)
    d.node(.12, 3.34, 1.62, .80, "Selection images\nand FER+ votes")
    d.node(2.04, 3.33, 2.18, .83,
           r"Minimize distance $D_e$" + "\n"
           + r"$K=8$: also risk $R_{80,e}$", size=11)
    d.node(4.62, 3.33, 1.72, .83,
           "Distance / epoch 12\nK=8: also risk rule", size=10.7)
    d.arrow([(1.74, 3.74), (2.04, 3.74)])
    d.arrow([(4.22, 3.74), (4.62, 3.74)])
    d.arrow([(5.48, 4.69), (5.48, 4.57), (4.37, 4.57),
             (4.37, 4.07), (4.22, 4.07)])

    d.panel(2.96, "c", "Frozen ranking and fixed-image comparisons")
    d.node(.12, 1.95, 1.62, .66, "Test images\nFrozen weights")
    d.node(2.04, 1.95, 1.58, .66,
           r"$p_i=f_\theta(x_i)$" + "\n" + r"$s_i=\max_c p_{ic}$")
    d.node(4.12, 2.28, 2.22, .68,
           "Own-ranking sets\nEqual accepted count")
    d.node(4.12, 1.42, 2.22, .68,
           "Fixed-image sets\nCompare predictors")
    d.arrow([(1.74, 2.28), (2.04, 2.28)])
    d.arrow([(3.62, 2.43), (3.83, 2.43), (3.83, 2.62), (4.12, 2.62)])
    d.arrow([(3.62, 2.13), (3.83, 2.13), (3.83, 1.76), (4.12, 1.76)])
    d.text(.14, 1.57, r"Test votes: $d_i=1-q_{i\hat c_i}$", ha="left")
    d.text(.14, 1.29, "Annotation-based scoring only", ha="left")

    d.panel(.96, "d", "Threshold transfer (original four Swin targets only)")
    d.node(.12, .14, 1.62, .58, "Calibration images\nVotes; frozen weights", size=10.7)
    d.node(2.04, .14, 1.58, .58,
           r"Fit $\tau$ offline" + "\nFreeze threshold")
    d.node(4.12, .14, 2.22, .58,
           r"Test: accept if $s_i\geq\tau$" + "\nReport risk and coverage")
    d.arrow([(1.74, .43), (2.04, .43)])
    d.arrow([(3.62, .43), (4.12, .43)])

    audit = d.audit()
    outputs = {}
    for ext in ("pdf", "svg", "png"):
        path = OUT / f"StudyDesign.{ext}"
        d.fig.savefig(path, bbox_inches="tight", pad_inches=.055,
                      **({"dpi": 240} if ext == "png" else {}))
        outputs[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    plt.close(d.fig)
    with fitz.open(OUT / "StudyDesign.pdf") as doc:
        fonts = [font for page in doc for font in page.get_fonts(full=True)]
        assert fonts and all(font[0] > 0 and font[2] != "Type3" for font in fonts)
        assert all(doc.extract_font(font[0])[3] for font in fonts)
    audit.update(status="passed", source="native scientific vector schematic",
                 original_training_runs=24, followup_training_runs=12,
                 original_named_states=48, followup_named_states=33,
                 model_outcome_values=False, type3_fonts=0, output_sha256=outputs,
                 selection_risk_rule="Eight-output models only; selection subset",
                 calibration_branch="Original Hard/Tie-hard/Uniform/Soft only")
    (HERE / "checks" / "study_design_audit.json").write_text(
        json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
