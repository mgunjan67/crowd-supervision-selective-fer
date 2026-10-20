"""Exporter contract tests; any generated layout illustration is SYNTHETIC."""
import argparse
import csv
import importlib.util
import json
from pathlib import Path
import numpy as np
import pytest

HERE=Path(__file__).resolve().parent
PATH=HERE.parent/'training/export_followup.py'
spec=importlib.util.spec_from_file_location('publication_export',PATH)
export=importlib.util.module_from_spec(spec);spec.loader.exec_module(export)


def write_csv(path,rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',newline='',encoding='utf-8') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def make_input_bundle(root):
    training=root/'training';results=training/'results';results.mkdir(parents=True,exist_ok=True)
    source=root/'source'
    new={(7,'permutation',seed,state) for seed in export.SEEDS for state in ('selected','epoch12')}
    new.update((8,condition,seed,state) for condition in ('soft','uniform','tie_uniform')
               for seed in export.SEEDS for state in ('selected','epoch12','risk_selected'))
    old={(7,condition,seed,state) for condition in ('soft','uniform','tie_uniform')
         for seed in export.SEEDS for state in ('selected','epoch12')}
    states=[dict(k=k,condition=condition,seed=seed,state=state,n=3275,accepted_n=2620,
                 risk_pct=40.+i/100,distance=.1+i/10000)
            for i,(k,condition,seed,state) in enumerate(sorted(new|old))]
    contrasts=[]
    for k,control,state in [(7,'permutation',s) for s in ('selected','epoch12')]+[
            (8,control,s) for s in ('selected','epoch12','risk_selected') for control in ('uniform','tie_uniform')]:
        i=len(contrasts);center=-.30+i*.08
        values=[center-.20,center,center+.20]
        contrasts.append(dict(k=k,left='soft',right=control,state=state,
                              mean_pp=float(np.mean(values)),seed_differences_pp=values,
                              seed_sd_pp=float(np.std(values,ddof=1)),
                              lower_pp=center-.35,upper_pp=center+.35))
    (results/'summary.json').write_text(json.dumps(dict(training_runs=12,states_new=33,
         primary_test_images=3275,pixel_clusters=3268,contrasts=contrasts,
         synthetic_layout_fixture=True)),encoding='utf-8')
    write_csv(results/'all_states.csv',states)
    fixed=[]
    for contrast in contrasts:
        for seed in export.SEEDS:
            for selector in ('soft',contrast['right']):
                for predictor in ('soft',contrast['right']):
                    fixed.append(dict(k=contrast['k'],state=contrast['state'],seed=seed,control=contrast['right'],
                                      selector=selector,predictor=predictor,accepted_n=2620,risk_pct=40.))
    write_csv(results/'fixed_sets.csv',fixed)
    strata=[]
    for condition in ('soft','uniform','tie_uniform'):
        for seed in export.SEEDS:
            for name,count in [('full',1760),('partial',1500),('zero',15)]:
                for scope in ('all','global80','within80'):
                    accepted=count if scope=='all' else int(np.ceil(.8*count)) if scope=='within80' else (
                        0 if name=='zero' else 1500 if name=='full' else 1120)
                    strata.append(dict(k=8,condition=condition,state='selected',seed=seed,stratum=name,
                                       scope=scope,stratum_n=count,accepted_n=accepted,
                                       risk_pct=40. if accepted else '',unsupported_pp=10. if accepted else '',
                                       excess_pp=15. if accepted else ''))
    write_csv(results/'unsupported_mass_strata.csv',strata)
    for k,condition in [(7,'permutation'),(8,'soft'),(8,'uniform'),(8,'tie_uniform')]:
        for seed in export.SEEDS:
            directory=training/'runs'/f'k{k}_{condition}_seed{seed}';directory.mkdir(parents=True,exist_ok=True)
            for epoch in range(1,13):
                (directory/f'epoch-{epoch:02d}.json').write_text(json.dumps(dict(
                    epoch=epoch,distance=.5+abs(epoch-4)*.01,risk80=.3+abs(epoch-6)*.01)),encoding='utf-8')
    return training,results,source,states,contrasts


def test_empty_and_partially_undefined_means_are_explicit():
    assert export.format_mean([],'risk_pct')=='--'
    assert export.format_mean([dict(risk_pct=1),dict(risk_pct='')],'risk_pct')=='--'
    assert export.format_mean([dict(risk_pct=1),dict(risk_pct=3)],'risk_pct')=='2.00'


def test_exact_state_and_contrast_groups_reject_duplicates(tmp_path):
    _,_,_,states,contrasts=make_input_bundle(tmp_path)
    assert len(export.validate_state_rows(states))==33
    export.validate_contrasts(contrasts)
    bad=[dict(row) for row in states];bad[-1]=dict(bad[-2])
    with pytest.raises(AssertionError):export.validate_state_rows(bad)
    contrasts[-1]=dict(contrasts[-2])
    with pytest.raises(AssertionError):export.validate_contrasts(contrasts)


def test_all_33_states_and_eight_contrasts_export_with_correct_units(tmp_path,monkeypatch):
    training,results,source,_,_=make_input_bundle(tmp_path)
    monkeypatch.setattr(export,'HERE',training);monkeypatch.setattr(export,'RESULTS',results)
    monkeypatch.setattr(export,'SOURCE',source)
    monkeypatch.setattr(export,'figure',lambda rows:None)
    monkeypatch.setattr(export,'sha',lambda path:'fixture-only-no-empirical-hash')
    export.main()
    text=(source/'generated/new_followup_supplement.tex').read_text()
    first=text.split(r'\label{tab:new-all-states}',1)[1].split(r'\bottomrule',1)[0]
    assert sum(' & ' in line and ' & Risk & Distance' not in line for line in first.splitlines())==33
    assert '33 nominal inference states' in text
    assert 'Risk is a percentage; components are percentage-point contributions' in text
    assert 'zero & global80 & 15 & 0.0 & -- & -- & --' in text
    assert 'causal decomposition' in text


def test_figure_preserves_native_pdf_and_editable_svg(tmp_path,monkeypatch):
    _,results,source,_,contrasts=make_input_bundle(tmp_path)
    (source/'figures').mkdir(parents=True)
    monkeypatch.setattr(export,'RESULTS',results)
    monkeypatch.setattr(export,'SOURCE',source)
    export.figure(contrasts)
    for extension in ('pdf','svg'):
        assert (source/f'figures/FollowupContrasts.{extension}').read_bytes()==(
            results/f'FollowupContrasts.{extension}').read_bytes()


def make_layout_fixture(directory):
    checks=HERE.parent/'checks'
    assert directory.resolve().is_relative_to(checks.resolve()),'Layout fixtures must stay below checks/'
    training,results,source,_,_=make_input_bundle(directory)
    export.HERE=training;export.RESULTS=results;export.SOURCE=source
    original=export.plt.Figure.savefig
    def labelled_save(figure,*args,**kwargs):
        if not getattr(figure,'_synthetic_label',False):
            figure.text(.5,.70,'SYNTHETIC LAYOUT FIXTURE - NOT EMPIRICAL',ha='center',va='center',fontsize=8,color='#8a1818')
            figure._synthetic_label=True
        return original(figure,*args,**kwargs)
    export.plt.Figure.savefig=labelled_save
    export.main()
    (directory/'SYNTHETIC_LAYOUT_ONLY.json').write_text(json.dumps(dict(
        synthetic=True,empirical_results=False,purpose='Test typography, grouping and table layout before outcomes exist',
        source_code_sha256=export.sha(PATH)),indent=2),encoding='utf-8')
    return results/'FollowupContrasts.pdf'


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--layout-fixture',type=Path,required=True)
    args=parser.parse_args();print(make_layout_fixture(args.layout_fixture))
