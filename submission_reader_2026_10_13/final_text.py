"""Data-linked narrative for the single bounded final reporting revision."""
import csv
import json
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
PAPER = HERE.parent
GEN = HERE / 'source/generated'


def main():
    claims = json.loads((HERE / 'checks/generated_numeric_claims.json').read_text())
    distance = json.loads((HERE / 'diagnostics/distribution_audit.json').read_text())
    with (PAPER / 'ravdess_transfer_2026_10_11/results/paired_seed_contrasts.csv').open(newline='') as stream:
        contrasts = list(csv.DictReader(stream))
    counter = sorted([r for r in contrasts if r['backbone'] == 'swin_t' and r['state'] == 'epoch12' and r['reference'] == 'neutral_calm_merged' and r['control'] == 'uniform' and r['score'] == 'msp' and float(r['requested_coverage']) == .8], key=lambda r: int(r['seed']))
    assert len(counter) == 3
    counter_values = [float(r['difference_pp']) for r in counter]
    assert all(v < 0 for v in counter_values)
    raw = claims['external_primary']['raw9']
    merged = claims['external_primary']['neutral_calm_merged']
    ds = distance['means']
    def write(name, content):
        (GEN / name).write_text(content + '\n', encoding='utf-8')
    abstract = (
        r'\textbf{Purpose:} To test whether training with crowd votes improves agreement with the full vote distribution and category predictions when a model may withhold uncertain results.\n'
        r'\textbf{Methods:} Twenty-four matched training runs compare five Swin-T and three ResNet-18 targets across three seeds on FER2013/FER+. All 48 validation-selected/epoch-12 states are tested against full votes. Without retraining, 36 states are evaluated on 1,440 RAVDESS speech clips with ten judgments each, using original categories and a reference combining calm and neutral.\n'
        f'\\textbf{{Results:}} Soft has lower held-out squared distribution distance than both matched controls in every seed, backbone and checkpoint state on FER+. Selected Swin mean distance is {ds["swin_t|selected|soft"]:.5f} for Soft versus {ds["swin_t|selected|uniform"]:.5f} for Uniform. At 80\\% coverage, Soft-minus-Uniform category disagreement is {claims["internal_primary_pp"]["swin_t"]:.2f} percentage points for Swin and {claims["internal_primary_pp"]["resnet18"]:+.2f} for ResNet. For external selected Swin models, the raw mean contrast is {float(raw["mean_difference_pp"]):+.2f} points (actor interval [{float(raw["lower95_pp"]):+.2f}, {float(raw["upper95_pp"]):+.2f}]), with two negative seed contrasts and one positive. Epoch-12 merged contrasts instead favor Soft in all three seeds. Intervals describe the fitted pairs and observed votes, not future training seeds or raters. All 36 empirical policy settings with retained predictions exceed calibration targets.\n'
        r'\textbf{Conclusion:} Better agreement with vote distributions does not ensure lower disagreement on retained predictions. Matched controls, fixed-image comparisons, explicit vocabularies and achieved-risk reporting reveal this distinction. External conclusions depend on checkpoint, reference and pipeline. Neither calibrated deployment probabilities nor communication benefit is established.'
    ).replace('\\n', '\n')
    write('abstract.tex', abstract)
    sentence = ('The ordering also changes with the saved model state and annotation reference. At epoch 12 under the merged reference, all three Swin MSP contrasts favor Soft (' + ', '.join(f'{v:+.3f}' for v in counter_values) + ' pp). Uniform therefore does not consistently perform better on the external dataset. This comparison tests the frozen pipelines as a whole. It does not separate the effect of supervision from differences in image capture, population or frame averaging.')
    external_path = GEN / 'external_results.tex'
    content = external_path.read_text(encoding='utf-8')
    anchor = 'The within-source advantage is not established reliably in this transfer setting.'
    assert anchor in content
    external_path.write_text(content.replace(anchor, sentence, 1), encoding='utf-8')
    write('external_discussion.tex',
          'For the validation-selected Swin models, the external mean under the raw reference combines two negative contrasts with one larger positive contrast. The actor interval shows sensitivity to the sampled actors for these fitted models. It does not measure uncertainty over future training seeds. '+sentence+'\n\n'
          'The raw and calm/neutral-merged references assign different scores to the same predictions. Overall disagreement remains high, as does disagreement above the minimum possible with the available categories. These results limit deployment claims regardless of which supervision target performs better. The descriptive matrices and frame summaries do not establish why transfer fails. Category mismatch and frame averaging are possible influences, not demonstrated causes.')
    write('discussion.tex', r'''\subsection{What the findings change in evaluation practice}
The answer to RQ1 depends on what is measured. On the source dataset, Soft consistently improves agreement with the full vote distribution over the controls. Its advantage in category disagreement among retained predictions, however, depends on the configuration. For RQ2, the comparison changes with which images are retained and which annotation reference is used. For RQ3, checks with another backbone and other confidence scores limit the scope of the internal advantage. On the external dataset, the ordering changes with the saved model state and reference. These findings support a reporting workflow, not a new training or rejection method. Four comparisons are important to interpreting the results.
\begin{enumerate}
\item \textbf{Compare meaningful target controls and more than one outcome.} The Swin Hard--Soft difference in selective disagreement is 2.61 pp, but the remaining Soft--Uniform advantage is 0.38 pp (Table~\ref{tab:primary}). Full-distribution distance still favors Soft across both model families and saved states (Table~\ref{tab:distance}). A small selective advantage therefore does not mean that crowd votes have little value for learning the vote distribution. The distance advantage also has a limit: Uniform and Tie-Uniform do not match entropy or overall class proportions. They cannot isolate the effect of knowing which minority categories received votes.
\item \textbf{Compare each model's chosen images and a shared set of images.} Soft-minus-Uniform is +0.085 pp on Uniform-selected images and -0.352 pp on Soft-selected images (Fig.~\ref{fig:cross}). A model that has lower risk on its own retained images does not necessarily make better category predictions on the same images as another model. Prediction and image selection interact. The matrix describes this interaction; it does not uniquely separate their causal effects.
\item \textbf{Specify the full annotation vocabulary and scoring reference.} The same images and predictions yield different F1 under original and crowd-majority labels (Table~\ref{tab:samef1}). Unsupported mass, supported ambiguity and excess describe different parts of disagreement with the full votes. Contempt, unknown and not-a-face also have different meanings. The zero-mass retention counts (Table~\ref{tab:retention}) show information that overall risk alone misses.
\item \textbf{Report measured risk beside the intended target.} All 36 nonempty empirical operating points exceed their targets (Table~\ref{tab:policies}). These observations overlap and are not independent trials. Methods cannot be described as having equal risk when their measured risks differ. The conservative comparator shows how much coverage its more cautious rule costs. It does not establish that its assumptions hold in deployment.
\end{enumerate}

The selective effect is also small in absolute terms. A difference of 0.38 pp across 2,620 retained images corresponds to about 9.9 fewer expected disagreements with a vote drawn from the observed annotations. It does not mean that 9.9 images were corrected: the retained images differ, and each loss can be fractional. Three fitted seeds do not establish which method will generally perform better after retraining. The adaptive ResNet check also does not reproduce Swin's selective advantage. By contrast, the full-distribution advantage holds across the tested source-dataset configurations. These findings concern different measures and are not contradictory.''')
    write('conclusion.tex',
          f'Training with the full crowd votes lowers held-out squared distribution distance relative to both matched softening controls in every tested seed, backbone and saved model state on the source dataset. This does not imply a general advantage in category disagreement among retained predictions. For the validation-selected models, Soft-minus-Uniform disagreement is {claims["internal_primary_pp"]["swin_t"]:.3f} pp for Swin and {claims["internal_primary_pp"]["resnet18"]:+.3f} pp for ResNet. Comparisons on identical images also depend on which model selected those images. On the external dataset, validation-selected Swin differences under the raw reference have mixed directions across seeds. Epoch-12 differences under the merged reference instead favor Soft in all three seeds. Conclusions about transfer therefore depend on the saved model state, annotation reference and frozen pipeline. The practical contribution is a way to report these results together: matched training targets, full-distribution and selective measures, fixed-image comparisons, explicit annotation vocabularies and measured risk. The evidence supports this evaluation practice for the tested configurations. It does not establish calibrated deployment probabilities, a universally better supervision target or a communication benefit.')
    claims['distribution_distance'] = distance
    claims['external_epoch12_merged_swin_seed_differences_pp'] = counter_values
    (HERE / 'checks/generated_numeric_claims.json').write_text(json.dumps(claims, indent=2)+'\n', encoding='utf-8')


if __name__ == '__main__':
    main()
