"""Build the standalone, fully inspectable reproduction notebook."""
import ast
import hashlib
import re
from pathlib import Path
import nbformat as nbf

source=Path('reproduction.py').read_text(encoding='utf8')
digest=hashlib.sha256(Path('reproduction.py').read_bytes()).hexdigest()
embedded=source.split("if __name__ == '__main__':")[0]
# The notebook carries the same implementation fingerprint as the accompanying module.
embedded=embedded.replace("sha256(__file__)\n                  if '__file__' in globals() else 'embedded-notebook'",
                          'EMBEDDED_IMPLEMENTATION_SHA256')
cells=[]
def md(s):cells.append(nbf.v4.new_markdown_cell(s.strip()))
def code(s):cells.append(nbf.v4.new_code_cell(s.strip()))

md(r'''
# Latent Feature Extraction for Musical Genres from Raw Audio
## Source-grounded reproduction, measured results, and discrepancy audit

**Paper:** Arjun Sawhney, Vrinda Vasavada, Woody Wang (2018), supplied `mini_project.pdf`.
**Reference implementation:** [Genre-Extraction](https://github.com/arjunsawknee/Genre-Extraction/tree/00b1802e24d1ab6fb3135b24a44976a51033ed24), pinned commit `00b1802e24d1ab6fb3135b24a44976a51033ed24`.

This notebook contains the complete implementation and follows the paper's experimental progression:
raw-input PCA → two-layer classifier → unsupervised deep autoencoder → joint autoencoder/classifier
→ accuracy, precision/recall/F1 and confusion matrix → 64-D and 4-D embedding PCA.

**Read the comparison labels carefully.** The PDF and released code disagree on loss scaling,
dropout and training duration. It is impossible to implement both conflicting definitions as one
"exact" experiment. We therefore run a released-code version, a paper architecture/schedule version
with the repository loss, and a literal PDF-equation version. All measured scores come from saved
predictions; the paper's reference scores are never used as model outputs or training targets.

The paper's original seed, file enumeration order and final 1,000/1,000 development/test assignment
are unavailable. Our fixed seed is 42, with the segment membership saved to CSV. Consequently,
numerical equality with the paper is not guaranteed, even when the architecture agrees.
''')
md(r'''
### What the supplied materials establish

The first notebook's saved test accuracies are **47.5% baseline** and **42.3% joint model**.
It resamples to 20 kHz, chooses random segments from whole recordings, uses different nonlinearities,
and applies a summed reconstruction error. Those are substantial differences from the linked code.

The initial ablation file has no saved outputs. The subsequently supplied **`abalation.ipynb`**
contains partial results through its deep learning-rate/epoch grid, despite cleared execution-count
fields. Its cell source matches the initial ablation file. Its measurements are incorporated below;
they are development-set results, not completed held-out test evaluations.

| Item | PDF | Released code | This notebook |
|---|---|---|---|
| Genres | classical, jazz, pop, metal | label order classical, jazz, metal, pop | repository label order everywhere |
| Dataset | 8,000 balanced examples | first 20 chunks per song | 100 songs × 20 chunks × 4 genres |
| Segment size | calls 20,000 samples one second at 22.05 kHz | 20,000 native samples | 0.9070 s; no 20 kHz resampling |
| Pooling | mean, width 40 → 500-D | same | same, with exact source comparisons |
| Split | 6,000 / 1,000 / 1,000 | 6,000 / 2,000 | PDF sizes; seeded random segment split |
| Baseline | 500 → 128(tanh) → 4 | Adam 1e-4, 200 epochs, batch 512 | repository settings |
| Encoder | 500 → 256 → 192 → 128 → 64 | sigmoid, tanh, ReLU, ReLU | exact released activations |
| Decoder | 64 → 128 → 192 → 256 → 500 | sigmoid, sigmoid, ReLU, ReLU | exact released activations |
| Head | 64 → 32 → 16 → 4 | tanh, tanh, linear logits | same |
| Dropout | keep 0.9 | keep 0.8 after encoder 192 and decoder 192 | both, in separate runs |
| Joint loss | 0.9 × sum-squared-error + 0.1 × CE | mean-squared-error + 0.1 × CE | both, in separate runs |
| Duration | Figure 3 shows 500 epochs | joint/baseline 200, vanilla 400 | all durations explicitly recorded |
| Initialization | unspecified | Xavier uniform, zero biases | same distribution; different framework RNG |
| Shuffle | random split | training order shuffled once | seeded split + one train-only shuffle |

With 500 inputs, the PDF's loss gives reconstruction **4,500 times the relative weight** used by the
repository. No pretraining, input normalization, MFCCs, spectrograms or pretrained audio network is added.
The vanilla autoencoder is trained on the training subset only; its released script instead fits all 8,000 examples.
''')
md('''
### What the partial ablation run actually shows

The added `abalation.ipynb` supplies development results, with later cells still unfinished:

| Completed check | Observed result | Interpretation |
|---|---|---|
| Preprocessing/selection screening | best baseline dev 52.2% | Changing segment selection alone did not close the gap |
| Baseline LR/epoch/batch screening | final dev 54.7%; best observed epoch 55.6% | Stronger baseline uses LR 1e-3, batch 16, 100 epochs; differs from repository schedule |
| Unsupervised AE + trained linear probe | best probe dev 60.5% | ReLU/tanh bottleneck, LR 3e-4, dropout 0.05, 100 epochs; not the joint classifier |
| Joint loss at gamma=0.9 | sum: 38.8%; mean: 52.3% dev | Reconstruction scaling materially affects this tanh architecture |
| Joint classification-only gamma=0 | 55.4% dev | Turning reconstruction off helps this implementation, but changes the proposed method |
| Partially completed joint LR/epoch grid | largest printed dev 59.7% | LR 3e-4, 200 epochs; no completed final test evaluation is saved |

The source's mixed sigmoid/tanh/ReLU architecture is absent from that sweep. This motivates the
source-grounded models below, rather than selecting another generic tanh variant. The partial
ablation's best-reconstruction selector compares raw `sum` and `mean` losses across rows; those
values differ by 500 and should not be ranked together. The original outputs remain untouched;
the accompanying extractor saves the explicitly printed measurements as CSV files.
''')
md('''
### Run instructions

Run all cells in this notebook using the project `.venv` kernel, or upload this notebook alone to
Google Colab and select a GPU runtime. Dependencies are installed only when absent. The first run
downloads a 1.23 GB GTZAN archive; later runs reuse it. Results persist under `results/`.
Set `WORK_DIR` to a Google Drive folder before running if you need persistence across Colab sessions.

`SMOKE=True` is only a two-epoch software check and is never presented as a paper reproduction.
By default the notebook runs all five main experiments, two additional initialization seeds and
a separate song-disjoint experiment. Checkpoints are reused only when configuration, implementation,
data, split and framework fingerprints agree. Delete a run's `completed.json` to force retraining.
''')
code('''
import importlib.util
import os
import subprocess
import sys
from pathlib import Path

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
os.environ.setdefault('MPLBACKEND', 'Agg')
dependencies = {'numpy':'numpy', 'pandas':'pandas', 'scipy':'scipy',
                'sklearn':'scikit-learn', 'matplotlib':'matplotlib', 'soundfile':'soundfile',
                'requests':'requests', 'torch':'torch'}
missing = [package for module,package in dependencies.items() if importlib.util.find_spec(module) is None]
if missing:
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', *missing])

WORK_DIR = Path.cwd()  # In Colab optionally set this to your mounted Drive directory.
WORK_DIR.mkdir(parents=True, exist_ok=True)
os.chdir(WORK_DIR)
RESULT_DIR = WORK_DIR / 'results' / 'source_grounded'
SMOKE = False
RUN_SEED_ROBUSTNESS = True
RUN_SONG_DISJOINT = True
''')
code(f"EMBEDDED_IMPLEMENTATION_SHA256 = {digest!r}")
titles={'imports':'Imports and reproducibility helpers', 'data':'Dataset download, pooling and split manifests',
        'architecture':'Original model architecture and the two objectives', 'training':'Training and development-only checkpoint selection',
        'reporting':'Metrics, PCA and plotting helpers'}
for chunk in embedded.split('# %% ')[1:]:
    label,body=chunk.split('\n',1)
    md('### '+titles[label.strip()])
    body=body.replace('from __future__ import annotations\n','')
    code(body)
md(r'''
### Build and inspect the 8,000-example dataset

The archive is pinned by SHA-256. Each normal WAV is decoded at its native sampling rate.
The released loader's first 600,000 samples are mean pooled in groups of 40 and its first
10,000 pooled samples become twenty 500-D inputs.

**One explicit data caveat:** both the Marsyas mirror and the earlier Kaggle download contain the
same headerless `jazz.00054.wav`. Only that file's exact SHA-256 permits a RAW PCM16 little-endian,
22.05 kHz mono fallback. This recovers existing amplitudes without padding or duplicating rows,
but cannot prove equivalence with the authors' converted recording. The decoding decision is
recorded per file. The reference CSV audit finds 1,996/2,000 examples matching the source data.
''')
code('''
RESULT_DIR.mkdir(parents=True, exist_ok=True)
audio_root = download_data(WORK_DIR / 'data')
X, y, manifest = build_dataset(audio_root, RESULT_DIR)
split = make_split(y, manifest, RESULT_DIR, seed=42)
environment = save_environment(RESULT_DIR)
print('X:', X.shape, 'class counts:', np.bincount(y))
print('Actual chunk duration:', 20000 / 22050, 'seconds')
display(pd.read_csv(RESULT_DIR / 'audio_files.csv').groupby('decoder').size().rename('recordings'))
display(pd.crosstab(pd.read_csv(RESULT_DIR / 'paper_segment_split.csv')['split'],
                    pd.read_csv(RESULT_DIR / 'paper_segment_split.csv')['genre']))
''')
md('''
### Figure 2: model architecture

The classifier consumes the same 64-D bottleneck that feeds the decoder. Dropout is enabled
only during training, immediately after the encoder's 192-D tanh and decoder's 192-D sigmoid.
''')
code('''
from matplotlib.patches import FancyBboxPatch
fig, ax = plt.subplots(figsize=(14, 4))
ax.set_xlim(-.5, 8.8); ax.set_ylim(-2, 1); ax.axis('off')
encoder_labels = ['500\\ninput', '256\\nsigmoid', '192\\ntanh + dropout', '128\\nReLU', '64\\nReLU',
                  '128\\nsigmoid', '192\\nsigmoid + dropout', '256\\nReLU', '500\\nReLU']
for i, label in enumerate(encoder_labels):
    ax.add_patch(FancyBboxPatch((i-.39,-.2),.78,.8,boxstyle='round,pad=0.03',
                               facecolor='#fce6af' if i==4 else '#dce8f6'))
    ax.text(i,.2,label,ha='center',va='center',fontsize=9)
    if i<8: ax.annotate('',xy=(i+.59,.2),xytext=(i+.4,.2),arrowprops={'arrowstyle':'->'})
for xloc,label in [(4,'64\\nbottleneck'),(5,'32\\ntanh'),(6,'16\\ntanh'),(7,'4\\nlogits')]:
    ax.add_patch(FancyBboxPatch((xloc-.35,-1.5),.7,.65,boxstyle='round,pad=0.03',facecolor='#dcefd6'))
    ax.text(xloc,-1.175,label,ha='center',va='center',fontsize=9)
    if xloc<7: ax.annotate('',xy=(xloc+.62,-1.175),xytext=(xloc+.37,-1.175),arrowprops={'arrowstyle':'->'})
ax.annotate('',xy=(4,-.8),xytext=(4,-.25),arrowprops={'arrowstyle':'->'})
ax.set_title('500-D mean-pooled waveform → shared 64-D encoding → reconstruction and genre logits')
fig.tight_layout()
(RESULT_DIR/'figures').mkdir(exist_ok=True)
fig.savefig(RESULT_DIR/'figures'/'figure2_architecture.png',dpi=160)
display(fig); plt.close(fig)
''')
md('''
### Train the complete experiment set

All schedules and objectives below are fixed before test evaluation. The primary named model is
`paper_architecture_repository_loss`: 500 epochs, keep probability 0.9, repository objective.
The two other joint runs are discrepancy diagnostics, not candidates chosen using test results.
Each run saves its final epoch and its best development checkpoint. We report both.
''')
code('''
experiments = default_experiments(seed=42)
if SMOKE:
    for config in experiments: config.epochs = 2
display(pd.DataFrame([asdict(c) for c in experiments]))
models, histories, provenance = {}, {}, {}
for config in experiments:
    models[config.name], histories[config.name], provenance[config.name] = train_experiment(
        config, X, y, split, RESULT_DIR)
display(pd.DataFrame(provenance).T[['best_epoch','best_dev_score','elapsed_seconds','test_used_for_selection']])
''')
md('''
### Evaluate frozen models and reproduce Table 1

This is the first full-run use of the test set. There is no retraining or threshold adjustment
in response to its accuracy. `best` means selected by development accuracy (by development MSE for
the unsupervised autoencoder); `last` means the predeclared final epoch.
''')
code('''
all_rows, results = [], {}
for config in experiments:
    for checkpoint in ['best','last']:
        table, output = score_model(models[config.name], config, X, y, split, RESULT_DIR, checkpoint)
        all_rows.append(table)
        if checkpoint == 'best': results[config.name] = output
metrics = pd.concat(all_rows, ignore_index=True)
metrics.to_csv(RESULT_DIR/'all_metrics.csv', index=False)
display(metrics)

paper = pd.DataFrame({'model':['baseline','paper_architecture_repository_loss'],
                      'paper_train':[.520,.949], 'paper_dev':[.381,.641], 'paper_test':[.364,.653]})
ours = metrics[metrics.checkpoint.eq('best')].pivot(index='experiment',columns='split',values='accuracy')
comparison = paper.merge(ours, left_on='model', right_index=True)
comparison['test_difference_percentage_points'] = 100*(comparison['test']-comparison.paper_test)
comparison.to_csv(RESULT_DIR/'paper_comparison.csv',index=False)
display(comparison.style.format({c:'{:.3f}' for c in comparison.columns if c != 'model'}))
primary_test = float(ours.loc['paper_architecture_repository_loss','test'])
print(f'Measured primary test accuracy: {primary_test:.1%}; paper: 65.3%.')
print('Meets/exceeds the numerical paper reference:', primary_test >= .653)
if SMOKE: print('SOFTWARE CHECK ONLY: these two-epoch scores are not reproduction results.')
''')
md('### Table 2: held-out precision, recall and F1, compared by genre name')
code('''
primary = 'paper_architecture_repository_loss'
test_truth = y[split['test']]
test_predictions = results[primary]['test']['prediction']
report = classification_report(test_truth, test_predictions, labels=range(4), target_names=GENRES,
                               output_dict=True, zero_division=0)
per_class = pd.DataFrame(report).T.loc[GENRES].reset_index(names='genre')
paper_classes = pd.DataFrame({'genre':GENRES, 'paper_precision':[.783,.606,.515,.670],
                             'paper_recall':[.775,.627,.554,.608], 'paper_f1':[.779,.616,.5337,.638]})
per_class = per_class.merge(paper_classes,on='genre')
per_class.to_csv(RESULT_DIR/'per_class_comparison.csv',index=False)
display(per_class.round(4))
print('Macro F1:', report['macro avg']['f1-score'])
''')
md('''
### Figures 1, 3, 4, 5 and 6

PCA is fitted on training embeddings, then applied to held-out test embeddings. Genre labels
only color the points. A visually separated projection is not itself proof of style transfer
or of reconstruction quality. The plots below use the declared primary model.
''')
code('''
make_figures(X,y,split,experiments,results,histories,RESULT_DIR)
from IPython.display import Image, display
for name in ['figure1_raw_pca.png','figure3_learning_curves.png','figure4_confusion.png',
             'figure5_bottleneck_pca.png','figure6_logits_pca.png']:
    display(Image(filename=str(RESULT_DIR/'figures'/name)))
''')
md('''
### Reconstruction sanity check

The released decoder ends in ReLU despite signed raw waveforms. Check whether it collapses
to all-zero predictions: a useful classifier alone does not establish a useful autoencoder.
The original released joint checkpoint also reconstructs zero on its saved development inputs.
This limitation is retained and reported in the faithful implementation.
''')
code('''
config = next(c for c in experiments if c.name == primary)
model = models[primary]
model.load_state_dict(torch.load(RESULT_DIR/primary/'best.pt',map_location=DEVICE,weights_only=True))
model.eval()
with torch.no_grad():
    out = model(torch.as_tensor(X[split['test'][:16]],device=DEVICE))
recon = out['reconstruction'].cpu().numpy()
reconstruction_audit = {'sampled_test_examples':16,
    'nonzero_reconstructed_values':int(np.count_nonzero(recon)),
    'test_reconstruction_mse':float(results[primary]['test']['metrics']['mse']),
    'zero_predictor_test_mse':float(np.mean(X[split['test']]**2))}
(RESULT_DIR/'reconstruction_audit.json').write_text(json.dumps(reconstruction_audit,indent=2))
print(json.dumps(reconstruction_audit,indent=2))
fig,ax=plt.subplots(figsize=(10,3))
ax.plot(X[split['test'][0]],label='Input pooled waveform')
ax.plot(recon[0],label='Reconstruction')
ax.set(xlabel='Pooled sample',ylabel='Amplitude',title='Reconstruction sanity check'); ax.legend()
fig.tight_layout();fig.savefig(RESULT_DIR/'figures'/'reconstruction_check.png',dpi=160)
display(fig);plt.close(fig)
''')
md('''
### Initialization robustness: report every predeclared seed

Seeds 42, 43 and 44 use the **same** train/dev/test membership. Each uses the same 500-epoch
schedule and selects its checkpoint using development data only. These runs measure initialization
sensitivity; the highest test seed is never substituted for the primary seed-42 result.
''')
code('''
seed_dir = WORK_DIR/'results'/'seed_robustness'
seed_dir.mkdir(parents=True,exist_ok=True)
robustness_rows = []
if RUN_SEED_ROBUSTNESS and not SMOKE:
    seed_models = []
    for init_seed in [43,44]:
        config = Experiment(f'joint_seed_{init_seed}','joint',500,seed=init_seed)
        model, _, _ = train_experiment(config,X,y,split,seed_dir)
        seed_models.append((config,model))
    for config,model in seed_models:
        for checkpoint in ['best','last']:
            table,_ = score_model(model,config,X,y,split,seed_dir,checkpoint)
            robustness_rows.append(table)
    extra_seed_metrics = pd.concat(robustness_rows,ignore_index=True)
    extra_seed_metrics.to_csv(seed_dir/'all_metrics.csv',index=False)
    primary_seed = metrics[metrics.experiment.eq(primary)].copy()
    primary_seed['experiment'] = 'joint_seed_42'
    seed_metrics = pd.concat([primary_seed,extra_seed_metrics],ignore_index=True)
    seed_tests = seed_metrics[seed_metrics.checkpoint.eq('best') & seed_metrics.split.eq('test')]
    display(seed_tests[['experiment','accuracy']])
    print(f'Test accuracy mean: {seed_tests.accuracy.mean():.2%}; sample SD: {seed_tests.accuracy.std(ddof=1):.2%}')
    seed_metrics.to_csv(RESULT_DIR/'three_seed_metrics.csv',index=False)
''')
md('''
### Separate improvement: equal-probability ensemble

This extension averages the softmax probabilities of the three fixed seeds above, with weights
1/3 each. It adds no new training data or feature engineering. Each member checkpoint was selected
by development accuracy. It is **not the paper's single-model experiment**.

The ensemble was added after inspecting individual-run results, so it is an exploratory improvement
on this existing held-out split, not a new independent confirmation test. No ensemble weights,
temperature or thresholds are fitted to test data. Its result is reported separately from the
primary single-model comparison, with member checkpoint hashes and all test predictions saved.
''')
ensemble_source=Path('evaluate_ensemble.py').read_text(encoding='utf8')
ensemble_function=next(node for node in ast.parse(ensemble_source).body
                       if isinstance(node,ast.FunctionDef) and node.name=='evaluate_ensemble')
ensemble_code=re.sub(r'\br\.', '', ast.get_source_segment(ensemble_source,ensemble_function))
code(ensemble_code)
code('''
if RUN_SEED_ROBUSTNESS and not SMOKE:
    ensemble_metrics=evaluate_ensemble(X,y,split,WORK_DIR)
    display(ensemble_metrics)
    ensemble_test=float(ensemble_metrics.loc[ensemble_metrics.split.eq('test'),'accuracy'].iloc[0])
    print(f'Ensemble test accuracy: {ensemble_test:.1%}; paper single model: 65.3%.')
    print('This ensemble meets/exceeds the numerical reference:',ensemble_test >= .653)
''')
md('''
### Separate song-disjoint generalization audit

The paper's segment split lets portions of the same recording appear on both sides of the split.
The earlier manifest quantifies this overlap. Here we hold entire songs apart and keep the
6,000/1,000/1,000 example counts. The exact same baseline and joint schedules are used.
This is a distinct task, so its scores are not substituted into the paper comparison.
Song-disjoint splitting does not remove GTZAN's cross-file duplicates or artist overlap.
''')
code('''
if RUN_SONG_DISJOINT and not SMOKE:
    disjoint_dir=WORK_DIR/'results'/'song_disjoint'
    disjoint_dir.mkdir(parents=True,exist_ok=True)
    disjoint_split=make_split(y,manifest,disjoint_dir,seed=42,song_disjoint=True)
    disjoint_configs=[c for c in default_experiments() if c.name in ['baseline',primary]]
    disjoint_models=[]
    for config in disjoint_configs:
        model,_,_=train_experiment(config,X,y,disjoint_split,disjoint_dir)
        disjoint_models.append((config,model))
    tables=[]
    for config,model in disjoint_models:
        for checkpoint in ['best','last']:
            table,_=score_model(model,config,X,y,disjoint_split,disjoint_dir,checkpoint)
            tables.append(table)
    disjoint_metrics=pd.concat(tables,ignore_index=True)
    disjoint_metrics.to_csv(disjoint_dir/'all_metrics.csv',index=False)
    display(disjoint_metrics)
''')
md('''
### Source validation artifacts and conclusions

The accompanying `validate_reproduction.py` uses the original repository's actual TensorFlow
function bodies to compare pooled inputs, forward activations, objective values, gradients and
ten Adam updates against this port. TensorFlow is optional for training this notebook; rerunning
that audit requires the repository and TensorFlow 2.20's compatibility API.

Interpret the outcome in two parts: whether genre classification approaches 65.3%, and whether
the claimed reconstruction/embedding behavior survives inspection. Preserve weaker runs and
the literal-equation result. Do not call a changed objective or a different held-out split an
exact numerical replication. A higher score on this random segment split is not evidence of
better generalization to unseen songs or of musical style transfer.
''')
code('''
for path in [WORK_DIR/'analysis'/'numerical_validation.json',
             WORK_DIR/'analysis'/'reference_checkpoint_audit.json']:
    if path.exists():
        print(path.name)
        print(path.read_text())
print('All metrics:',RESULT_DIR/'all_metrics.csv')
print('Per-class comparison:',RESULT_DIR/'per_class_comparison.csv')
print('Data and split provenance:',RESULT_DIR/'segments.csv')
print('Figures:',RESULT_DIR/'figures')
print('Final statement:', f"{primary_test:.1%} measured test accuracy versus the paper's 65.3%.")
''')
nb=nbf.v4.new_notebook(cells=cells,metadata={
    'kernelspec':{'display_name':'Python 3 (ML paper reproduction)','language':'python','name':'python3'},
    'language_info':{'name':'python','version':'3.11.9'}})
nbf.validate(nb)
path=Path('ML_Mini_Project_Source_Grounded_Reproduction.ipynb')
nbf.write(nb,path)
print(f'Created {path}: {len(cells)} cells; embedded implementation {digest}')
