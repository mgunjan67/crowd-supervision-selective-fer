"""Reproducible explanatory diagrams and endpoint comparison for the visual edition.

The supervision examples are illustrative mathematics, not additional observations.
The endpoint figure reuses retained CSV values; no models, predictions, intervals,
or scientific experiments are newly fitted here.
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, Rectangle
from matplotlib.text import Text
import numpy as np

HERE = Path(__file__).resolve().parent
PAPER = HERE.parent
FIG = HERE / "source" / "figures"
CHECKS = HERE / "checks"
PREVIEWS = CHECKS / "visual_figure_previews"
SEEDS = (17, 42, 89)
CLASSES = ("angry", "disgust", "fear", "happy", "neutral", "sad", "surprise")
TARGETS = ("Hard", "Soft", "Uniform", "Tie-hard", "Tie-Uniform")
INK = "#202830"
BLUE = "#245b7d"
ORANGE = "#99603b"
LIGHT = "#eef3f6"
REPORT = {"status": "pending", "new_training_runs": 0, "new_inference": False,
          "illustrative_targets_explicitly_marked": True, "figures": {}}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def configure():
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
        "font.size": 9.5, "axes.labelsize": 9.5, "axes.titlesize": 10,
        "xtick.labelsize": 9, "ytick.labelsize": 9, "axes.linewidth": .65,
        "text.color": INK, "axes.labelcolor": INK, "xtick.color": INK,
        "ytick.color": INK, "pdf.fonttype": 42, "ps.fonttype": 42,
        "svg.fonttype": "none", "savefig.facecolor": "white",
        "figure.facecolor": "white", "mathtext.fontset": "dejavusans",
    })


def text_inventory(fig):
    """Check all visible, nonempty text bounds and pairwise intersections.

    Only actual text artists are checked (not surrounding boxes or data marks).
    A tiny tolerance avoids edge-touching false alarms. The renderer checks the
    same physical canvas exported to PDF/EPS/SVG, without tight-bbox cropping.
    """
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    canvas = fig.bbox
    records = []
    outside = []
    for artist in fig.findobj(match=Text):
        if not artist.get_visible() or not artist.get_text().strip():
            continue
        rect = artist.get_window_extent(renderer)
        if rect.width == 0 or rect.height == 0:
            continue
        item = {"text": artist.get_text(), "bbox_px": list(map(float, rect.bounds)),
                "font_pt": float(artist.get_fontsize())}
        records.append(item)
        if (rect.x0 < canvas.x0 - .5 or rect.x1 > canvas.x1 + .5 or
                rect.y0 < canvas.y0 - .5 or rect.y1 > canvas.y1 + .5):
            outside.append(item)
    overlap = []
    for n, left in enumerate(records):
        x, y, w, h = left["bbox_px"]
        for right in records[n + 1:]:
            a, b, c, d = right["bbox_px"]
            dx = min(x + w, a + c) - max(x, a)
            dy = min(y + h, b + d) - max(y, b)
            if dx > .75 and dy > .75:
                overlap.append({"left": left["text"], "right": right["text"],
                                "intersection_px": [dx, dy]})
    return {"text_count": len(records), "minimum_font_pt": min(r["font_pt"] for r in records),
            "out_of_canvas": outside, "text_overlaps": overlap,
            "text_bounds": records}


def save(fig, name, evidence):
    # The main manuscript embeds 7.1-inch artwork at approximately 5.15 inches.
    # Enforce at least 8.56-point lettering after that reduction, including ticks.
    for artist in fig.findobj(match=Text):
        if artist.get_fontsize() < 11.8:
            artist.set_fontsize(11.8)
    bounds = text_inventory(fig)
    assert not bounds["out_of_canvas"], (name, bounds["out_of_canvas"])
    assert not bounds["text_overlaps"], (name, bounds["text_overlaps"])
    for extension in ("pdf", "eps", "svg"):
        kwargs = {"metadata": {"CreationDate": None, "ModDate": None}} if extension == "pdf" else {}
        fig.savefig(FIG / f"{name}.{extension}", **kwargs)
    fig.savefig(PREVIEWS / f"{name}.png", dpi=200)
    REPORT["figures"][name] = {"width_in": float(fig.get_size_inches()[0]),
                              "height_in": float(fig.get_size_inches()[1]),
                              "embedded_width_in": 5.15,
                              "embedded_minimum_font_pt": bounds["minimum_font_pt"] * 5.15 / 7.1,
                              "layout": bounds, "evidence": evidence,
                              "assets": [f"source/figures/{name}.{ext}" for ext in ("pdf", "eps", "svg")]}
    plt.close(fig)


def target_vectors(r):
    r = np.array(r, dtype=np.float64)
    assert np.all(r >= 0) and abs(r.sum() - 1) < 1e-12
    highest = float(r.max())
    tied = np.flatnonzero(r == highest)
    hard = np.eye(7)[int(tied[0])]
    uniform = np.full(7, (1 - highest) / 6)
    uniform[int(tied[0])] = highest
    tie_hard = np.zeros(7)
    tie_hard[tied] = 1 / len(tied)
    tie_uniform = np.full(7, (1 - len(tied) * highest) / (7 - len(tied))) if len(tied) < 7 else np.full(7, 1 / 7)
    tie_uniform[tied] = highest
    values = np.array([hard, r, uniform, tie_hard, tie_uniform])
    assert np.allclose(values.sum(axis=1), 1) and np.all(values >= 0)
    return values


def probability_label(value):
    if value in (0, 1):
        return str(int(value))
    return f"{value:.2f}".rstrip("0").rstrip(".").removeprefix("0")


def target_controls():
    examples = [("a", "Unique maximum", [.5, .2, .1, .1, .05, .03, .02]),
                ("b", "Two tied maxima", [.3, .3, .15, .1, .05, .05, .05])]
    fig = plt.figure(figsize=(7.1, 6.6))
    fig.text(.5, .975, "Illustrative supported-vote distributions (not dataset observations)",
             ha="center", va="top", fontsize=10)
    figure_evidence = []
    for panel, (letter, title, r) in enumerate(examples):
        left = .17 + panel * .485
        width = .32
        fig.text(left, .919, f"({letter})  {title}", fontsize=10, weight="bold", va="top")
        vectors = np.vstack([r, target_vectors(r)])
        for row, values in enumerate(vectors):
            bottom = .77 - row * .111
            ax = fig.add_axes([left, bottom, width, .078])
            label = "Example r" if row == 0 else TARGETS[row - 1]
            color = BLUE if label == "Soft" else "#8296a3" if row == 0 else "#b9c4cb"
            ax.bar(np.arange(7), values, color=color, edgecolor=INK, linewidth=.4, width=.7)
            for x, value in enumerate(values):
                ax.text(x, float(value) + .065, probability_label(float(value)),
                        ha="center", va="bottom", fontsize=8.8)
            ax.set_ylim(0, 1.32)
            ax.set_xlim(-.55, 6.55)
            ax.set_yticks([])
            ax.set_xticks(np.arange(7), ["A", "D", "F", "H", "N", "S", "Su"] if row == 5 else [])
            ax.tick_params(axis="x", length=0, pad=3)
            for spine in ("top", "left", "right"):
                ax.spines[spine].set_visible(False)
            ax.spines["bottom"].set_color("#abb6bf")
            ax.text(-.11, .4, label, transform=ax.transAxes, ha="right", va="center", fontsize=9)
        figure_evidence.append({"case": title, "supported_distribution_r": r,
                                "target_order": list(TARGETS), "target_vectors": target_vectors(r).tolist(),
                                "fixed_tie_break_class": CLASSES[int(np.argmax(r))]})
    fig.text(.5, .14, "A angry; D disgust; F fear; H happy; N neutral; S sad; Su surprise.",
             ha="center", fontsize=9)
    fig.text(.5, .092, "Same weighted loss: w × cross-entropy(t, p), with r = q / w.",
             ha="center", fontsize=10)
    fig.text(.5, .035, "Bars share a probability scale (0 to 1); labels are rounded.\nZero supported mass gives zero loss.",
             ha="center", va="center", fontsize=9, linespacing=1.35)
    save(fig, "TargetControls", {"kind": "illustrative_target_construction", "class_order": list(CLASSES),
                                  "examples": figure_evidence, "empirical_observations": False})


def label(ax, x, y, content, fontsize=9.5, **kwargs):
    return ax.text(x, y, content, ha="center", va="center", fontsize=fontsize, **kwargs)


def box(ax, x, y, w, h, heading, body, fontsize=9.4):
    ax.add_patch(Rectangle((x, y), w, h, facecolor="white", edgecolor=INK, linewidth=.8))
    ax.add_patch(Rectangle((x, y + h - .049), w, .049, facecolor=LIGHT, edgecolor="none"))
    ax.plot([x, x + w], [y + h - .049] * 2, color=INK, lw=.5)
    label(ax, x + w / 2, y + h - .0245, heading, fontsize=9.6, weight="bold")
    label(ax, x + w / 2, y + (h - .049) / 2, body, fontsize=fontsize, linespacing=1.35)


def arrow(ax, points, style="-", color=INK):
    for first, second in zip(points[:-2], points[1:-1]):
        ax.plot([first[0], second[0]], [first[1], second[1]], color=color, lw=.9, linestyle=style)
    ax.add_patch(FancyArrowPatch(points[-2], points[-1], arrowstyle="-|>", mutation_scale=9,
                                linewidth=.9, color=color, linestyle=style,
                                shrinkA=0, shrinkB=0))


def external_pipeline():
    fig = plt.figure(figsize=(7.1, 8.2))
    ax = fig.add_axes([0, 0, 1, 1]); ax.set(xlim=(0, 1), ylim=(0, 1)); ax.axis("off")
    ax.text(.035, .98, "(a)  Frozen clip prediction and selection", fontsize=10.3, weight="bold", va="top")
    label(ax, .505, .94, r"Eight bin centers: $\lfloor(b+0.5)N/8\rfloor$, $b=0,\ldots,7$", fontsize=10)
    ax.plot([.1, .9], [.895, .895], color=INK, lw=.9)
    for b in range(8):
        x = .1 + .8 * (b + .5) / 8
        ax.plot([x, x], [.887, .903], color=INK, lw=.9)
        label(ax, x, .87, f"b = {b}", fontsize=9)
    label(ax, .042, .895, "Start", fontsize=9)
    label(ax, .952, .895, "End", fontsize=9)
    # A collector bracket groups all eight centers, rather than implying that
    # only the bin immediately above the arrow enters the detector.
    ax.plot([.15, .15, .85, .85], [.857, .851, .851, .857], color=INK, lw=.8)
    arrow(ax, [(.25, .851), (.25, .835)])
    box(ax, .035, .69, .43, .145, "Face detection and crop",
        "Exactly one MediaPipe face\nValid, image-clipped crop\nNo fallback or peak selection")
    box(ax, .535, .69, .43, .145, "Shared preprocessing",
        "Grayscale repeated to 3 channels\nBilinear resize to 224 × 224\nNormalize: mean / std = 0.5", fontsize=9.1)
    box(ax, .035, .49, .43, .145, "Frozen frame classifier",
        "Swin-T or ResNet-18\nSame retained checkpoint\nSeven probabilities per frame")
    box(ax, .535, .49, .43, .145, "Clip probabilities",
        "Arithmetic mean over valid frames\nAt least 4 distinct valid frames\nNo temporal model", fontsize=9.1)
    box(ax, .035, .29, .43, .145, "Confidence-only selection",
        "MSP = largest clip probability\nEach model ranks its own clips\n80%: retain 1,152 of 1,440", fontsize=9.1)
    box(ax, .535, .29, .43, .145, "Score accepted predictions",
        "Category disagreement\nVocabulary-aware accounting\nSame predictions for both references", fontsize=8.9)
    arrow(ax, [(.465, .7625), (.535, .7625)])
    arrow(ax, [(.75, .69), (.75, .663), (.25, .663), (.25, .635)])
    arrow(ax, [(.465, .5625), (.535, .5625)])
    arrow(ax, [(.75, .49), (.75, .463), (.25, .463), (.25, .435)])
    arrow(ax, [(.465, .3625), (.535, .3625)])
    ax.plot([.035, .965], [.25, .25], color="#b0bbc3", lw=.6)
    ax.text(.035, .226, "(b)  Human-response references: scoring only", fontsize=10.3, weight="bold", va="top")
    box(ax, .035, .055, .29, .14, "Human judgments",
        "Ten judgments per clip\nRaw counts retained", fontsize=9.3)
    box(ax, .355, .055, .29, .14, "Raw response reference",
        "Seven model categories\nPlus calm and none", fontsize=9.3)
    box(ax, .675, .055, .29, .14, "Reference sensitivity",
        "Raw nine responses, or\ncalm merged into neutral\nNone remains unsupported", fontsize=9.1)
    arrow(ax, [(.325, .125), (.355, .125)])
    arrow(ax, [(.645, .125), (.675, .125)])
    arrow(ax, [(.82, .195), (.82, .29)])
    label(ax, .5, .023, "External votes never train models, choose checkpoints, or fit thresholds.", fontsize=9.4)
    save(fig, "ExternalPipeline", {
        "kind": "existing_frozen_external_protocol", "frame_bins": 8,
        "minimum_distinct_valid_frames": 4, "detector": "MediaPipe exactly one detection",
        "model_outputs": 7, "reference_responses": 9, "clip_count": 1440,
        "msp80_accepted_count": 1152, "new_estimation": False,
        "reference_changes_predictions": False, "external_vote_training": False})


def endpoint_comparison():
    distance_path = HERE / "diagnostics" / "distribution_distance.csv"
    selective_path = PAPER / "recovery_2026_10_10" / "selector_contrasts.csv"
    interval_path = PAPER / "backbone_followup" / "reported_results.csv"
    distances = read(distance_path); selectors = read(selective_path); intervals = read(interval_path)
    lookup = {(r["backbone"], r["state"], r["condition"], int(r["seed"])): float(r["distance"]) for r in distances}
    slookup = {(r["backbone"], r["state"], r["control"], int(r["seed"])): float(r["difference_pp"])
               for r in selectors if r["score"] == "msp" and float(r["requested_coverage"]) == .8}
    ilookup = {(r["backbone"], r["state"], r["right"]): r for r in intervals
               if r["record_type"] == "contrast" and r["left"] == "soft" and r["right"] in ("uniform", "tie_uniform")}
    rows = [(b, s, c) for b in ("swin_t", "resnet18") for s in ("selected", "epoch12")
            for c in ("uniform", "tie_uniform")]
    fig = plt.figure(figsize=(7.1, 6.6))
    axes = [fig.add_axes([.27, .25, .303, .565]), fig.add_axes([.66, .25, .315, .565])]
    markers = ("o", "s", "^")
    offsets = (-.27, -.09, .09)
    mean_offset = .27
    evidence_rows = []
    for y, (backbone, state, control) in enumerate(rows):
        full = [lookup[backbone, state, "soft", seed] - lookup[backbone, state, control, seed] for seed in SEEDS]
        selective = [slookup[backbone, state, control, seed] for seed in SEEDS]
        interval = ilookup[backbone, state, control]
        mean_full, mean_selective = float(np.mean(full)), float(np.mean(selective))
        assert abs(mean_selective - float(interval["point_difference_pp"])) < 1e-10
        color = BLUE if backbone == "swin_t" else ORANGE
        for ax, values, average in zip(axes, (full, selective), (mean_full, mean_selective)):
            for marker, offset, value in zip(markers, offsets, values):
                ax.scatter(value, y + offset, marker=marker, facecolors="white", edgecolors=color,
                           s=24, linewidths=.9, zorder=3)
            ax.scatter(average, y + mean_offset, marker="D", s=35, color=INK, zorder=4)
        lo, hi = float(interval["lower_pp"]), float(interval["upper_pp"])
        axes[1].hlines(y + mean_offset, lo, hi, color="#777777", linewidth=.9, zorder=1)
        axes[1].vlines([lo, hi], y + mean_offset - .07, y + mean_offset + .07, color="#777777", linewidth=.9, zorder=1)
        evidence_rows.append({"backbone": backbone, "state": state, "control": control,
                              "seeds": list(SEEDS), "full_distance_soft_minus_control": full,
                              "full_distance_mean": mean_full, "disagreement_difference_pp": selective,
                              "disagreement_mean_pp": mean_selective,
                              "retained_conditional_pixel_interval_pp": [lo, hi]})
    rowlabels = [("Swin-T" if b == "swin_t" else "ResNet-18") + " / " +
                 ("Uniform" if c == "uniform" else "Tie-Uniform") + "\n" +
                 ("Selected" if s == "selected" else "Epoch 12") for b, s, c in rows]
    axes[0].set_yticks(range(8), rowlabels)
    axes[1].set_yticks(range(8), [])
    for ax in axes:
        ax.set_ylim(7.55, -.55)
        ax.axvline(0, color=INK, linewidth=.8, linestyle="--", zorder=0)
        ax.grid(axis="x", color="#e1e5e8", linewidth=.55)
        for sep in (1.5, 3.5, 5.5):
            ax.axhline(sep, color="#e1e5e8", linewidth=.6)
        ax.tick_params(axis="y", length=0, pad=6)
        ax.spines[["top", "right", "left"]].set_visible(False)
    axes[0].set_xlim(-.029, .0025)
    axes[0].set_xticks([-.025, -.015, 0], ["−0.025", "−0.015", "0"])
    axes[1].set_xlim(-.93, .6)
    axes[1].set_xticks([-.8, -.4, 0, .4])
    axes[0].set_xlabel("Soft − control\nSquared-distance units", labelpad=8)
    axes[1].set_xlabel("Soft − control\nDisagreement difference\n(percentage points)", labelpad=8)
    axes[0].set_title("(a)  Full-distribution agreement\nAll 3,275 test images", loc="left", pad=10, linespacing=1.35)
    axes[1].set_title("(b)  Selective disagreement\n80% MSP: 2,620 images", loc="left", pad=10, linespacing=1.35)
    handles = [Line2D([], [], marker=m, markerfacecolor="white", markeredgecolor=INK, markersize=5,
                      linestyle="none", label=f"Seed {s}") for s, m in zip(SEEDS, markers)]
    handles.append(Line2D([], [], marker="D", color=INK, markersize=5, linestyle="none", label="Mean"))
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.605, .985), ncol=4,
               frameon=False, handletextpad=.35, columnspacing=1.2, fontsize=9.4)
    fig.text(.52, .038, "Negative favors Soft in each endpoint.\nThe horizontal scales have different units.",
             ha="center", fontsize=9, linespacing=1.35)
    save(fig, "Fig7", {"kind": "retained_endpoint_contrasts", "rows": evidence_rows,
                       "distance_population": 3275, "selective_accepted_n": 2620,
                       "selective_score": "msp", "own_model_rankings": True,
                       "seed_vertical_offsets": list(offsets), "mean_vertical_offset": mean_offset,
                       "intervals_recomputed": False,
                       "input_csv_sha256": {str(p.relative_to(PAPER)).replace("\\", "/"): sha(p)
                                             for p in (distance_path, selective_path, interval_path)}})


def main():
    FIG.mkdir(parents=True, exist_ok=True); CHECKS.mkdir(parents=True, exist_ok=True)
    PREVIEWS.mkdir(parents=True, exist_ok=True)
    configure()
    target_controls(); external_pipeline(); endpoint_comparison()
    REPORT["status"] = "passed"
    REPORT["figure_count"] = 3
    REPORT["no_text_overlaps"] = all(not r["layout"]["text_overlaps"] for r in REPORT["figures"].values())
    REPORT["no_text_clipping"] = all(not r["layout"]["out_of_canvas"] for r in REPORT["figures"].values())
    (CHECKS / "visual_figures.json").write_text(json.dumps(REPORT, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": REPORT["status"], "figures": list(REPORT["figures"]),
                      "no_text_overlaps": REPORT["no_text_overlaps"], "no_text_clipping": REPORT["no_text_clipping"]}, indent=2))


if __name__ == "__main__":
    main()
