"""Layout-only redraws of supplementary figures from the frozen ledger."""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

OUT = Path(__file__).resolve().parent
INK, BLUE, ORANGE = '#25313D', '#235B83', '#A34227'
SEEDS = (17, 42, 89)
SEED_COLORS = (BLUE, '#707070', '#386B53')
SEED_STYLES = ('-', '--', ':')

def save(fig, name):
    fig.canvas.draw()
    renderer=fig.canvas.get_renderer()
    for legend in fig.legends:
        bounds=legend.get_window_extent(renderer)
        for ax in fig.axes:
            assert not bounds.overlaps(ax.get_window_extent(renderer)),f'{name}: legend intersects plot'
            if ax.get_title():assert not bounds.overlaps(ax.title.get_window_extent(renderer)),f'{name}: legend intersects title'
    for ext in ('pdf', 'eps'):
        fig.savefig(OUT/'figures'/f'{name}.{ext}', bbox_inches='tight', pad_inches=.12)
    plt.close(fig)

def box(ax, x, y, w, h, label, size=11):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.006,rounding_size=.01',
                              facecolor='#F1F4F6',edgecolor=INK,lw=.9,clip_on=False))
    if label:
        ax.text(x+w/2,y+h/2,label,ha='center',va='center',fontsize=size)

def arrow(ax, a, b):
    ax.annotate('',xy=b,xytext=a,arrowprops={'arrowstyle':'->','lw':1,'color':INK})

def improve_supplement(rows):
    def get(kind, **criteria):
        return [r for r in rows if r['record_type']==kind and all(r[k]==str(v) for k,v in criteria.items())]
    fig,ax=plt.subplots(figsize=(6.5,3.15))
    ax.set(xlim=(0,1),ylim=(0,1));ax.axis('off')
    box(ax,.01,.60,.20,.29,'Input\n3 × 224 × 224')
    box(ax,.28,.60,.27,.29,'Swin-T backbone\nPool + flatten\n768 features')
    box(ax,.81,.60,.18,.29,'Linear\n768 → 7\nwith bias')
    box(ax,.29,.15,.36,.26,'Bias-free gate\n768 → 48 → 768\nReLU; sigmoid')
    ax.text(.72,.745,r'$\odot$',ha='center',va='center',fontsize=20)
    arrow(ax,(.22,.745),(.27,.745));arrow(ax,(.56,.745),(.685,.745));arrow(ax,(.75,.745),(.80,.745))
    arrow(ax,(.44,.59),(.44,.42))
    ax.plot([.66,.72],[.28,.28],color=INK,lw=1)
    arrow(ax,(.72,.28),(.72,.68))
    ax.text(.5,.025,'One gate after pooling; 73,728 additional parameters',ha='center',fontsize=10.5)
    save(fig,'FigS1')

    fig,ax=plt.subplots(figsize=(6.5,3.65))
    ax.set(xlim=(0,1),ylim=(0,1));ax.axis('off')
    ax.text(.02,.97,'Local expression-cue prototype: illustrative controls',weight='bold',va='top',fontsize=11)
    ax.text(.02,.89,'Synthetic text and score; not a participant record\nor measured prediction',va='top',fontsize=10.5)
    box(ax,.02,.36,.38,.39,'Local camera preview\n\nNo facial image shown\n\nExample: neutral\nScore: 0.62 (uncalibrated)',10)
    box(ax,.45,.36,.53,.39,'')
    ax.text(.48,.69,'Compose a message',weight='bold',fontsize=10.5)
    box(ax,.48,.52,.47,.10,'I will send the draft tomorrow.',10)
    ax.add_patch(plt.Rectangle((.48,.455),.021,.035,fill=False,lw=1))
    ax.text(.52,.472,'Attach current expression cue',va='center',fontsize=10)
    ax.text(.48,.39,'Add to local preview',fontsize=10.5,color=BLUE)
    ax.text(.02,.26,'Default: off; camera off or invalid/stale observation: omit the cue',fontsize=10)
    ax.text(.02,.16,'Opt-in resets after each message. No face does not mean neutral.',fontsize=10)
    ax.text(.02,.06,'Messages remain local; no recipient-side transport is implemented.',fontsize=10)
    save(fig,'FigS2')

    fig,axes=plt.subplots(1,2,figsize=(6.5,3.25))
    comparisons=[('soft','uniform'),('soft','hard'),('tie_hard','hard'),('soft','tie_hard')]
    colors=[BLUE,ORANGE,'#447651','#6A5484']
    for ax,state,label in zip(axes,('selected','epoch12'),('Selected checkpoint','Fixed epoch 12')):
        for i,((left,right),color) in enumerate(zip(comparisons,colors)):
            for seed,offset,marker in zip(SEEDS,(-.18,0,.18),('o','s','^')):
                rr=get('seed_contrast',backbone='swin_t',state=state,left=left,right=right,seed=seed)
                assert len(rr)==1,(state,left,right,seed)
                ax.plot(float(rr[0]['difference_pp']),i+offset,marker,color=color,ms=4.3,ls='none',label=f'Seed {seed}' if i==0 else None)
        ax.set_yticks(range(4),['S - U','S - H','T - H','S - T']);ax.invert_yaxis()
        ax.axvline(0,color='#707070',ls=':',lw=.8);ax.set_xlabel('Paired difference (pp)');ax.set_title(label,fontsize=11)
        ax.margins(x=.08,y=.15)
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.5,1),ncol=3,frameon=False,fontsize=10)
    fig.subplots_adjust(left=.10,right=.98,bottom=.19,top=.77,wspace=.35)
    save(fig,'FigS3')

    conds=('hard','tie_hard','uniform','tie_uniform','soft')
    names=('Hard','Tie-hard','Uniform','Tie-Uniform','Soft')
    fig,axes=plt.subplots(5,2,figsize=(6.5,8.4),sharex=True)
    for i,(condition,label) in enumerate(zip(conds,names)):
        for seed,color,style in zip(SEEDS,SEED_COLORS,SEED_STYLES):
            rr=sorted(get('learning_curve',backbone='swin_t',condition=condition,seed=seed),key=lambda r:int(r['epoch']))
            assert len(rr)==12
            for ax,key in zip(axes[i],('train_loss','selection_distance')):
                ax.plot([int(r['epoch']) for r in rr],[float(r[key]) for r in rr],color=color,ls=style,lw=1,label=f'Seed {seed}')
        axes[i,0].set_ylabel(label,fontsize=10.5)
        for ax in axes[i]:
            ax.set_xticks([1,4,8,12]);ax.tick_params(labelsize=10);ax.margins(x=.04)
    axes[0,0].set_title('Training loss',fontsize=11);axes[0,1].set_title('Selection distance',fontsize=11)
    for ax in axes[-1]:ax.set_xlabel('Epoch',fontsize=11)
    handles,labels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.55,.995),ncol=3,frameon=False,fontsize=10.5)
    fig.subplots_adjust(left=.14,right=.97,bottom=.07,top=.91,hspace=.30,wspace=.38)
    save(fig,'LearningCurves')

    fig,axes=plt.subplots(3,3,figsize=(6.5,6.0),sharex=True)
    for i,(condition,label) in enumerate(zip(('soft','uniform','tie_uniform'),('Soft','Uniform','Tie-Uniform'))):
        for j,seed in enumerate(SEEDS):
            rr=sorted(get('learning_curve',backbone='resnet18',condition=condition,seed=seed),key=lambda r:int(r['epoch']))
            assert len(rr)==12
            ax=axes[i,j];ax.plot([int(r['epoch']) for r in rr],[float(r['selection_distance']) for r in rr],color=BLUE,lw=1)
            ax.set_title(f'{label}\nSeed {seed}',fontsize=10.5,pad=7)
            if j==0:ax.set_ylabel('Selection distance',fontsize=10.5)
            if i==2:ax.set_xlabel('Epoch',fontsize=10.5)
            ax.set_xticks([1,6,12]);ax.tick_params(labelsize=10);ax.grid(color='#E2E2E2',lw=.5)
    fig.subplots_adjust(left=.13,right=.97,bottom=.09,top=.91,hspace=.75,wspace=.46)
    save(fig,'ResNetLearning')

    fig,axes=plt.subplots(2,1,figsize=(6.5,4.8),sharex=True)
    for ax,right,label in zip(axes,('uniform','tie_uniform'),('Soft - Uniform (pp)','Soft - Tie-Uniform (pp)')):
        values=[]
        for seed,color,style in zip(SEEDS,SEED_COLORS,SEED_STYLES):
            rr=sorted(get('risk_difference_curve',backbone='resnet18',comparison=right,seed=seed),key=lambda r:int(r['accepted_n']))
            x=np.array([float(r['coverage']) for r in rr]);y=np.array([float(r['difference_pp']) for r in rr]);keep=x>=.2
            assert len(rr)==3275
            values.append(y);ax.plot(x[keep],y[keep],color=color,ls=style,lw=.65,label=f'Seed {seed}')
        ax.plot(x[keep],np.mean(values,axis=0)[keep],color=ORANGE,lw=1.6,label='Mean')
        ax.axhline(0,color='#707070',lw=.7);ax.axvline(.8,color='#707070',ls=':',lw=.8)
        ax.set(xlim=(.2,1),ylabel=label);ax.grid(color='#E2E2E2',lw=.5)
    axes[-1].set_xlabel('Coverage')
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.55,.995),ncol=4,frameon=False,fontsize=10)
    fig.subplots_adjust(left=.14,right=.97,bottom=.12,top=.89,hspace=.22)
    save(fig,'ResNetDifferences')
