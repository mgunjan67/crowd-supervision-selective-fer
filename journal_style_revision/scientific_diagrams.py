"""Scientific schematics with explicit data and tensor contracts.

No photographs, illustrative vote profiles, fake feature maps or measured results.
Coordinates are inches; all figure text is 11 pt before manuscript scaling.
"""
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch, Circle

INK = '#222222'


class Drawing:
    def __init__(self, height):
        self.fig = plt.figure(figsize=(6.5, height))
        self.ax = self.fig.add_axes([0, 0, 1, 1])
        self.ax.set(xlim=(0, 6.5), ylim=(0, height))
        self.ax.axis('off')
        self.texts = []
        self.containers = []

    def text(self, x, y, s, ha='center', bold=False):
        t = self.ax.text(x, y, s, fontsize=11, ha=ha, va='center',
                         fontweight='bold' if bold else 'normal', color=INK,
                         linespacing=1.25)
        self.texts.append(t)
        return t

    def node(self, x, y, w, h, s):
        patch = Rectangle((x, y), w, h, ec=INK, fc='white', lw=.8)
        self.ax.add_patch(patch)
        t = self.text(x+w/2, y+h/2, s)
        self.containers.append((t, patch))

    def arrow(self, points):
        for a, b in zip(points[:-2], points[1:-1]):
            self.ax.plot([a[0], b[0]], [a[1], b[1]], color=INK, lw=.8)
        self.ax.add_patch(FancyArrowPatch(points[-2], points[-1],
                          arrowstyle='-|>', mutation_scale=9, color=INK, lw=.8))

    def panel(self, y, letter, name):
        self.text(.12, y, letter, ha='left', bold=True)
        self.text(.37, y, name, ha='left')

    def export(self, out, number, preview):
        self.fig.canvas.draw()
        renderer = self.fig.canvas.get_renderer()
        overlaps = []
        for i, a in enumerate(self.texts):
            for b in self.texts[i+1:]:
                if a.get_window_extent(renderer).overlaps(b.get_window_extent(renderer)):
                    overlaps.append([a.get_text(), b.get_text()])
        overflows = []
        for text, patch in self.containers:
            t = text.get_window_extent(renderer)
            p = patch.get_window_extent(renderer)
            if not (p.contains(t.x0-2, t.y0-2) and p.contains(t.x1+2, t.y1+2)):
                overflows.append(text.get_text())
        assert not overlaps and not overflows, dict(overlaps=overlaps, overflows=overflows)
        for ext in ('pdf', 'eps', 'svg'):
            self.fig.savefig(out/'figures'/f'Fig{number}.{ext}',
                             bbox_inches='tight', pad_inches=.055)
        self.fig.savefig(out/'figures'/preview, dpi=240,
                         bbox_inches='tight', pad_inches=.055)
        plt.close(self.fig)
        return dict(text_overlap_pairs=overlaps, container_overflows=overflows,
                    minimum_source_font_pt=11)


def draw_methodology(out):
    out = Path(out)
    d = Drawing(5.85)
    d.panel(5.62, 'a', 'Matched optimization and checkpoint selection')
    d.node(.12, 4.59, 1.62, .70, 'Training images\nand FER+ votes')
    d.node(2.04, 4.51, 2.18, .86,
           r'$\ell_i=-w_i\sum_{c=1}^{7}t_{ic}\log p_{ic}$'+'\nAdamW; 12 epochs')
    d.node(4.62, 4.59, 1.72, .70, r'$\theta_1,\ldots,\theta_{12}$'+'\nCandidate states')
    d.arrow([(1.74,4.94),(2.04,4.94)])
    d.arrow([(4.22,4.94),(4.62,4.94)])
    d.text(.14,4.33,'Within each seed: same images,\ninitialization, order and flips',ha='left')
    d.node(.12,3.37,1.62,.70,'Selection images\nand FER+ votes')
    d.node(2.04,3.29,2.18,.86,
           r'$e^*=\arg\min_e J_e$'+'\nCommon selection score')
    d.node(4.62,3.37,1.72,.70,r'$\theta_{e^*},\;\theta_{12}$'+'\nFrozen weights')
    d.arrow([(1.74,3.72),(2.04,3.72)])
    d.arrow([(4.22,3.72),(4.62,3.72)])
    d.arrow([(5.48,4.59),(5.48,4.42),(4.37,4.42),(4.37,3.91),(4.22,3.91)])
    d.panel(2.96,'b','Test ranking and fixed-image comparisons')
    d.node(.12,1.95,1.62,.66,'Test images\nFrozen weights')
    d.node(2.04,1.95,1.58,.66,r'$p_i=f_\theta(x_i)$'+'\n'+r'$s_i=\max_c p_{ic}$')
    d.node(4.12,2.28,2.22,.68,'Own-ranking sets\nEqual accepted count')
    d.node(4.12,1.42,2.22,.68,'Fixed-image sets\nCompare predictors')
    d.arrow([(1.74,2.28),(2.04,2.28)])
    d.arrow([(3.62,2.43),(3.83,2.43),(3.83,2.62),(4.12,2.62)])
    d.arrow([(3.62,2.13),(3.83,2.13),(3.83,1.76),(4.12,1.76)])
    d.text(.14,1.57,r'Test votes: $d_i=1-q_{i\hat c_i}$',ha='left')
    d.text(.14,1.29,'Annotation-based scoring only',ha='left')
    d.panel(.96,'c','Calibration-fitted threshold transfer (original four Swin targets)')
    d.node(.12,.14,1.62,.58,'Calibration images\nVotes; frozen weights')
    d.node(2.04,.14,1.58,.58,r'Fit $\tau$ offline'+'\nFreeze threshold')
    d.node(4.12,.14,2.22,.58,r'Test: accept if $s_i\geq\tau$'+'\nReport risk and coverage')
    d.arrow([(1.74,.43),(2.04,.43)])
    d.arrow([(3.62,.43),(4.12,.43)])
    audit=d.export(out,1,'Methodology-preview.png')
    audit.update(status='passed',new_training_runs=0,uses_photographs=False,
                 calibration_not_used_for_own_ranking=True,
                 selection_does_not_use_test=True,
                 offline_runtime_separated=True,
                 supplied_formats=['pdf','eps','svg','png'])
    (out/'methodology_diagram_audit.json').write_text(json.dumps(audit,indent=2))


def draw_model(out):
    out=Path(out)
    d=Drawing(4.85)
    d.panel(4.62,'a','Swin-T encoder: spatial resolution, channels and block count')
    d.text(.18,4.20,r'$x\in\mathbb{R}^{B\times3\times224\times224}$',ha='left')
    d.text(2.58,4.20,'Patch projection: 4 × 4, stride 4; LayerNorm',ha='left')
    dims=['56 × 56 × 96','28 × 28 × 192','14 × 14 × 384','7 × 7 × 768']
    for j,(shape,depth) in enumerate(zip(dims,(2,2,6,2))):
        x=.12+j*1.59
        d.node(x,3.31,1.45,.64,f'Stage {j+1}: {depth} blocks\n{shape}')
        if j<3:
            d.arrow([(x+1.45,3.63),(x+1.59,3.63)])
    d.text(3.25,3.10,'Arrows between stages: 2 × 2 patch merging')
    d.text(.16,2.87,'Final LayerNorm → permute → global average pooling → flatten',ha='left')
    d.panel(2.60,'b','One post-pooling gate and a seven-output classifier')
    d.text(.14,1.61,r'$z\in\mathbb{R}^{B\times d}$',ha='left')
    d.node(1.48,1.98,1.10,.50,r'$W_1$; ReLU')
    d.node(2.94,1.98,1.10,.50,r'$W_2$; sigmoid')
    d.arrow([(1.02,1.61),(1.23,1.61),(1.23,2.23),(1.48,2.23)])
    d.arrow([(2.58,2.23),(2.94,2.23)])
    d.text(2.03,1.78,r'$d\rightarrow d/16$')
    d.text(3.49,1.78,r'$d/16\rightarrow d$')
    d.ax.add_patch(Circle((4.45,1.61),.15,ec=INK,fc='white',lw=.8))
    d.text(4.45,1.61,r'$\odot$')
    d.arrow([(4.04,2.23),(4.45,2.23),(4.45,1.76)])
    d.text(4.96,2.23,r'$a(z)$')
    d.arrow([(1.02,1.61),(1.23,1.61),(1.23,1.20),(4.16,1.20),(4.16,1.61),(4.30,1.61)])
    d.node(4.90,1.36,1.43,.50,r'Linear $d\rightarrow7$')
    d.arrow([(4.60,1.61),(4.90,1.61)])
    d.arrow([(5.615,1.36),(5.615,.94)])
    d.text(5.615,.76,r'$p=\mathrm{softmax}(\mathrm{logits})$')
    d.text(.15,.90,r'$\tilde z=z\odot a(z)$; gate layers have no biases',ha='left')
    d.text(.15,.60,'Classifier includes a bias; no intermediate-stage gates',ha='left')
    d.text(.15,.26,'Swin-T: d = 768, hidden = 48  |  ResNet-18: d = 512, hidden = 32',ha='left')
    audit=d.export(out,2,'Model-preview.png')
    audit.update(status='passed',single_pooled_gate=True,
                 swin_stage_depths=[2,2,6,2],swin_stage_shapes=dims,
                 head_dimensions={'swin_t':[768,48,768,7],'resnet18':[512,32,512,7]},
                 architecture_claim='Existing implementation, not a proposed new architecture',
                 supplied_formats=['pdf','eps','svg','png'])
    (out/'model_diagram_audit.json').write_text(json.dumps(audit,indent=2))


if __name__=='__main__':
    output=Path(__file__).resolve().parent
    plt.rcParams.update({'font.family':'Arial','pdf.fonttype':42,'ps.fonttype':42})
    draw_methodology(output)
    draw_model(output)
