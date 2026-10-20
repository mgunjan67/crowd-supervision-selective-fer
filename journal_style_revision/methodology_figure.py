"""Deprecated infographic builder, retained as design history only.

The current manuscript uses scientific_diagrams.py instead. Do not regenerate
submission artwork with this superseded module.
"""
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Circle

INK = '#25313D'
BLUE = '#235B83'
GREEN = '#386B53'
ORANGE = '#A34227'
GRAY = '#6C7680'


def draw_methodology(out):
    out = Path(out)
    fig = plt.figure(figsize=(6.5, 8.35))
    ax = fig.add_axes([.018, .018, .964, .964])
    ax.set(xlim=(0, 100), ylim=(0, 124))
    ax.axis('off')
    text_checks = []
    cards = []

    def label(x, y, s, size=10.5, color=INK, bold=False, ha='center'):
        obj = ax.text(x, y, s, fontsize=size, color=color, ha=ha, va='center',
                      fontweight='bold' if bold else 'normal', linespacing=1.15)
        text_checks.append(obj)
        return obj

    def card(x, y, w, h, color=BLUE, fill='white', dashed=False):
        obj = FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.12,rounding_size=.7',
                             edgecolor=color, facecolor=fill, linewidth=.85,
                             linestyle='--' if dashed else '-')
        ax.add_patch(obj)
        cards.append((x, y, w, h, obj))
        return obj

    def route(points, color=INK, dashed=False):
        for a, b in zip(points[:-2], points[1:-1]):
            ax.plot([a[0], b[0]], [a[1], b[1]], color=color, linewidth=.85,
                    linestyle='--' if dashed else '-')
        ax.add_patch(FancyArrowPatch(points[-2], points[-1], arrowstyle='-|>',
                                     mutation_scale=9, linewidth=.85, color=color,
                                     linestyle='--' if dashed else '-'))

    def header(y, part, text, color=BLUE):
        ax.plot([1, 99], [y-2.7, y-2.7], color=color, linewidth=.6)
        label(1, y, part, 11.5, color, True, 'left')
        label(5, y, text, 11.5, color, True, 'left')

    def pixel_stack(x, y, w=9, h=5):
        # Geometry only: no photograph, sample or synthesized face.
        for offset in (1.2, .6, 0):
            ax.add_patch(Rectangle((x+offset, y+offset), w, h,
                                   facecolor='white', edgecolor=GRAY, linewidth=.6))
        for row in range(4):
            for col in range(6):
                tone = ['#D9E0E5', '#ADBCC7', '#F0F3F5'][(row+2*col) % 3]
                ax.add_patch(Rectangle((x+col*w/6, y+row*h/4), w/6, h/4,
                                       facecolor=tone, edgecolor='white', linewidth=.3))

    def strip(x, y, values, width, height=1.8, omitted=False):
        cursor = x
        for k, value in enumerate(values):
            if not value:
                continue
            w = value * width
            ax.add_patch(Rectangle((cursor, y), w, height,
                                   facecolor=('#DBE3E9' if k >= 7 else
                                              ['#235B83', '#577E9B', '#8AA7BC', '#C3D2DD'][k % 4]),
                                   edgecolor='white', linewidth=.5,
                                   hatch='///' if omitted and k >= 7 else None))
            cursor += w
        ax.add_patch(Rectangle((x, y), width, height, facecolor='none', edgecolor=GRAY, linewidth=.55))

    def feature_maps(x, y, color, widths=(6, 5, 4)):
        for j, w in enumerate(widths):
            for offset in (.7, 0):
                ax.add_patch(Rectangle((x+j*7+offset, y+(6-w)/2+offset), w, w,
                                       facecolor='#F0F4F7', edgecolor=color, linewidth=.65))
            for a in range(1, 3):
                ax.plot([x+j*7+w*a/3]*2, [y+(6-w)/2, y+(6-w)/2+w], color=color, lw=.35)
                ax.plot([x+j*7, x+j*7+w], [y+(6-w)/2+w*a/3]*2, color=color, lw=.35)

    def vector(x, y, color):
        for j in range(5):
            ax.add_patch(Rectangle((x, y+j*1.05), 1.5, .86,
                                   facecolor=color if j % 2 == 0 else '#D9E3EB',
                                   edgecolor=color, linewidth=.45))

    header(122, 'a', 'Images, crowd votes and matched supervision targets')
    card(1, 103, 22, 15)
    label(12, 115.2, 'FER2013', bold=True)
    pixel_stack(7, 108, 9, 4)
    label(12, 104.7, '48 × 48 pixels')
    card(28, 103, 19, 15)
    label(37.5, 115.2, 'FER+ votes', bold=True)
    q = np.array([.4, .4, .1, 0, 0, 0, 0, .1, 0, 0])
    # Small vote histogram: category position rather than a measured output.
    for k, value in enumerate(q):
        ax.add_patch(Rectangle((30+k*1.45, 107), 1.15, value*12,
                               facecolor=BLUE if k < 7 else '#D9E3EB',
                               edgecolor=GRAY, linewidth=.5, hatch='///' if k >= 7 else None))
    label(37.5, 104.7, '10 categories')
    route([(23.3, 110.5), (27.7, 110.5)])
    route([(47.3, 110.5), (50, 110.5)])
    # Weighted targets follow the manuscript equations for one illustrative q.
    w = q[:7].sum(); m = q[:7].max()/w
    hard = np.zeros(7); hard[0] = w
    tied = np.zeros(7); tied[:2] = w/2
    uniform = np.full(7, w*(1-m)/6); uniform[0] = w*m
    tie_uniform = np.full(7, w*(1-2*m)/5); tie_uniform[:2] = w*m
    targets = [hard, tied, uniform, tie_uniform, q[:7]]
    for i, (name, values) in enumerate(zip(['Hard', 'Tie-hard', 'Uniform', 'Tie-Uniform', 'Soft'], targets)):
        y = 116.5-i*3.05
        label(51, y, name, ha='left')
        strip(73, y-.9, values/w, 25)
    label(50, 99.8, 'Schematic profiles; same images and supported vote mass', size=10.5)

    header(95, 'b', 'Matched training and isolated operational roles')
    card(1, 57, 22, 33)
    label(12, 86.5, 'Training', bold=True)
    label(12, 83.1, '28,709 images')
    pixel_stack(7, 75.5)
    label(12, 70.2, 'Grayscale × 3\n224 × 224\nNormalize')
    label(12, 62.7, '12 epochs\nSeeds 17, 42, 89')

    for y, title, color, dim, runs in [(75, 'Swin-T', BLUE, 768, '5 targets × 3 seeds = 15 runs'),
                                     (57, 'ResNet-18', GREEN, 512, '3 targets × 3 seeds = 9 runs')]:
        card(28, y, 36, 15, color)
        label(46, y+12.2, f'{title}  |  ImageNet-1K', color=color, bold=True)
        feature_maps(30, y+4.2, color)
        route([(50.7, y+7), (53, y+7)], color)
        vector(54, y+4.2, color)
        ax.add_patch(Circle((59.5, y+6.6), 1.6, facecolor='white', edgecolor=color, linewidth=.8))
        label(59.5, y+6.6, 'g', color=color)
        route([(55.6, y+6.6), (57.7, y+6.6)], color)
        label(46, y+1.7, f'{dim} features → gate → 7', size=10.5)
        label(46, y-1.8, runs, color=color)
    route([(23.2, 81), (27.7, 81)], BLUE)
    route([(23.2, 64), (27.7, 64)], GREEN)

    card(70, 79.5, 29, 10.5)
    label(84.5, 85.7, 'Selection: 1,667', bold=True)
    label(84.5, 82.2, 'Choose checkpoint')
    card(70, 65.5, 29, 10.5)
    label(84.5, 73.4, 'Retain both states', bold=True)
    label(84.5, 68.5, '24 selected states\n24 epoch-12 states')
    route([(64.3, 82), (67, 82), (69.7, 84.7)])
    route([(64.3, 64), (67, 64), (67, 84.7), (69.7, 84.7)])
    route([(84.5, 79.2), (84.5, 76.3)])
    card(70, 52, 29, 10.5)
    label(84.5, 58.3, 'Calibration: 1,642', bold=True)
    label(84.5, 54.8, 'Fit offline thresholds')
    route([(84.5, 65.2), (84.5, 62.8)])
    label(1, 51.5, 'Swin: all five targets; ResNet: Soft / Uniform / Tie-Uniform', ha='left')

    header(46.5, 'c', 'Primary test: 3,275 images; frozen states and thresholds')
    route([(84.5, 51.7), (84.5, 48.8), (97.5, 48.8), (97.5, 44.3)])
    for x, title, lines, color in [
        (1, 'Own-ranking', 'Risk vs coverage\nVote-mass accounting\nPaired seed contrasts', BLUE),
        (35, 'Fixed-image', 'Same accepted images\nCompare predictors\nCompare references', BLUE),
        (69, 'Threshold transfer', 'Calibration → test\nAccepted-set shifts\nFinite-sample checks', GREEN)]:
        card(x, 26.5, 30, 16.8, color)
        label(x+15, 40.5, title, color=color, bold=True)
        label(x+15, 32.8, lines)
    label(50, 23.4, 'One reused image source; four adaptive phases, not external confirmation')

    header(18.5, 'd', 'Separate local prototype; not the evaluation setting', ORANGE)
    # Sensor and composition glyphs, not camera data or a participant interface.
    card(1, 2, 17, 12, ORANGE)
    ax.add_patch(Rectangle((7, 9), 5, 2.7, facecolor='white', edgecolor=ORANGE, linewidth=.7))
    ax.add_patch(Circle((9.5, 10.35), .75, facecolor='white', edgecolor=ORANGE, linewidth=.7))
    label(9.5, 5.7, 'MediaPipe\nOne face', color=ORANGE)
    card(24, 2, 24, 12, ORANGE)
    label(36, 10.2, 'Preprocess', bold=True)
    label(36, 5.7, 'Swin + pooled gate\nCategorical cue')
    card(54, 2, 20, 12, ORANGE)
    label(64, 10.2, 'Sender control', bold=True)
    label(64, 5.7, 'Fresh + opt-in\nOtherwise omit')
    card(80, 2, 19, 12, ORANGE)
    ax.add_patch(Rectangle((83, 8.5), 13, 3, facecolor='white', edgecolor=ORANGE, linewidth=.65))
    for y, length in [(10.6, 9), (9.4, 6)]:
        ax.plot([85, 85+length], [y]*2, color=ORANGE, linewidth=.7)
    label(89.5, 5, 'Message + cue\nLocal preview')
    for a, b in [(18.3, 23.7), (48.3, 53.7), (74.3, 79.7)]:
        route([(a, 8), (b, 8)], ORANGE)
    label(50, -.4, 'Sender-machine backend; no recipient transport or tested communication benefit')

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    intersections = []
    overflows = []
    for i, a in enumerate(text_checks):
        for b in text_checks[i+1:]:
            if a.get_window_extent(renderer).overlaps(b.get_window_extent(renderer)):
                intersections.append((a.get_text(), b.get_text()))
        x, y = a.get_position()
        bounds = a.get_window_extent(renderer)
        for cx, cy, cw, ch, patch in cards:
            if cx < x < cx+cw and cy < y < cy+ch:
                card_bounds = patch.get_window_extent(renderer)
                if not (card_bounds.contains(bounds.x0-2, bounds.y0-2) and card_bounds.contains(bounds.x1+2, bounds.y1+2)):
                    overflows.append(a.get_text())
    assert not intersections and not overflows, {'intersections': intersections, 'overflows': overflows}
    for ext in ('pdf', 'eps', 'svg'):
        fig.savefig(out/'figures'/f'Fig1.{ext}', bbox_inches='tight', pad_inches=.07)
    fig.savefig(out/'figures'/'Methodology-preview.png', dpi=220, bbox_inches='tight', pad_inches=.07)
    plt.close(fig)
    (out/'methodology_diagram_audit.json').write_text(json.dumps({
        'status': 'passed', 'text_overlap_pairs': intersections,
        'minimum_source_font_pt': 10.5, 'illustrative_glyphs_not_results': True,
        'uses_photographs': False, 'new_training_runs': 0,
        'single_pooled_gate': True, 'offline_runtime_separated': True,
        'target_profile_supported_mass': float(w),
        'profile_mass_equal': bool(all(np.isclose(t.sum(), w) for t in targets)),
        'supplied_formats': ['pdf', 'eps', 'svg', 'png']}, indent=2))


if __name__ == '__main__':
    draw_methodology(Path(__file__).resolve().parent)
