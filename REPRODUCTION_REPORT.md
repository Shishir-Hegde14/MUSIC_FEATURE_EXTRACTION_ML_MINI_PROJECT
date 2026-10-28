# Reproduction results and audit

The new, fully executed notebook is **`ML_Mini_Project_Source_Grounded_Reproduction.ipynb`**.
It reproduces the paper's full experimental progression and records the differences between the PDF
and its released implementation. All training and evaluation completed locally on the RTX 4050.

The main single model achieved **63.6% test accuracy**, compared with the paper's **65.3%** and
the earlier notebook's **42.3%**. A separately labelled, equal-probability ensemble of three fixed
training seeds achieved **65.8%**. The ensemble clears the numerical reference by five
examples out of 1,000. Its approximate 95% interval (62.9%–68.7%) contains 65.3%, so this is a
tie rather than better performance, and it is not the paper's single-model experiment.

The classification numbers replicate closely, but the paper's central mechanism does not: the
decoder outputs all zeros in every run, so the reconstruction objective contributes nothing (see
"Reconstruction is a material limitation" below).

## Measured results

These are inference-mode accuracies. `Best dev` selects the epoch using development accuracy only;
the vanilla autoencoder instead selects development reconstruction MSE. The primary experiment
was designated before inspecting its test result. No test accuracy was used to choose its seed,
epoch, objective, thresholds or training duration.

| Model / experiment | Train | Dev | Test | Interpretation |
|---|---:|---:|---:|---|
| Paper: two-layer classifier | 52.00% | 38.10% | 36.40% | Published reference |
| Paper: joint model | 94.90% | 64.10% | 65.30% | Published reference |
| Earlier supplied notebook: classifier | 75.42% | 50.50% | 47.50% | Saved user output |
| Earlier supplied notebook: joint model | 48.17% | 42.80% | 42.30% | Saved user output |
| New classifier, best dev | 52.35% | 41.50% | 41.50% | Source optimizer/schedule |
| New released-code settings, best dev | 80.55% | 62.40% | 60.80% | 200 epochs, keep 0.8 |
| **New primary joint model, best dev** | **95.02%** | **63.90%** | **63.60%** | 500 epochs, keep 0.9, repository loss |
| New literal PDF-equation joint model, best dev | 93.80% | 62.90% | 60.80% | 0.9 × summed squared error + 0.1 × CE |
| **Separate three-seed ensemble** | **94.08%** | **64.80%** | **65.80%** | Average probabilities, weights 1/3 each |

The primary single model improves the earlier joint-model test result by **21.3 percentage points**.
Its final epoch, without development checkpoint selection, gives **97.08% / 62.10% / 62.20%**
train/dev/test. Both final and selected checkpoint results are preserved in `all_metrics.csv`.

The ensemble combines the development-selected checkpoints for training seeds 42, 43 and 44.
Each seed sets both the weight initialization and the one-time training-order shuffle, so the
spread reflects both sources of variation. Their individual test accuracies are **63.6%, 62.0% and 62.4%**, respectively, with mean **62.67%**
and sample standard deviation **0.83 percentage points**. No highest-test seed replaces the primary
result. The ensemble was added after inspecting these individual runs; it is an exploratory
extension on the existing held-out split, not a fresh independent confirmation test.

## What changed after reading the linked code

Reference: [arjunsawknee/Genre-Extraction](https://github.com/arjunsawknee/Genre-Extraction/tree/00b1802e24d1ab6fb3135b24a44976a51033ed24),
commit `00b1802e24d1ab6fb3135b24a44976a51033ed24`.

1. **The actual segmentation is not 20 kHz resampling.** The source loads at 22,050 Hz, mean-pools
   the first 600,000 amplitudes by 40 and retains the first 10,000 pooled values as twenty 500-D
   chunks. Each chunk spans 20,000 native samples, about **0.907 seconds**. This determines the
   paper's 8,000-example selection: 20 consecutive chunks from each of 400 songs.
2. **The encoder is not a uniform tanh network.** Its nonlinearities are sigmoid, tanh, ReLU, ReLU.
   The decoder uses sigmoid, sigmoid, ReLU, ReLU, including the reconstruction output. The head is
   tanh, tanh, linear logits. The new notebook preserves these exact functions and layer widths.
3. **Dropout has specific locations.** It follows the 192-D encoder tanh and the 192-D decoder
   sigmoid. The released training loop uses keep 0.8; the PDF says final keep 0.9. Both are run.
4. **The PDF's loss differs from the released objective.** The source uses `MSE + 0.1 * CE`.
   The PDF's literal equation is `0.9 * SSE + 0.1 * CE`. Since `SSE = 500 * MSE`, the relative
   reconstruction/classification weight differs by **4,500**. These are separately named runs.
5. **The source resolves baseline settings.** Adam LR 1e-4, batch size 512, 200 epochs differs from
   the earlier notebook's LR 1e-3, batch size 32, 50 epochs. Thus the baseline scores are not
   evidence of a regression under an identical training protocol.
6. **Duration and splits also conflict.** The released joint model trains 200 epochs and has a
   6,000/2,000 split; the PDF's Figure 3 shows 500 epochs and states 6,000/1,000/1,000. The new
   primary experiment uses the PDF's split sizes, architecture, keep probability and duration,
   together with the released loss and activations. The vanilla AE trains 400 epochs on training
   data only; its released script instead trains on all 8,000 examples.

The earlier implementation differs on several factors simultaneously. The results do not isolate
one change as the sole cause of its poor accuracy.

## Findings from the newly supplied partial ablation

`abalation.ipynb` has the same source cells as the initial exhaustive ablation notebook, but adds
**28 saved outputs**. All execution-count fields were cleared; the outputs themselves contain the
measurements. The last saved progress is partway through the joint LR/epoch grid. Later activation,
pretraining, leakage and final test cells have no saved results. Its TensorFlow runtime reports no GPU.

The explicitly printed measurements were extracted into `analysis/partial_ablation_results/`:
9 preprocessing runs, 25 baseline LR/epoch runs, 7 baseline batch runs, 72 AE screening runs,
7 AE epoch runs, 20 joint loss/gamma runs and 27 entries from the incomplete joint LR/epoch grid.

- Best preprocessing/selection baseline development accuracy: **52.2%**.
- Best baseline batch sweep endpoint: **54.7%** development accuracy, with LR 1e-3, batch 16,
  100 epochs. Its best observed epoch reaches **55.6%**, but that checkpoint was not restored.
- Best reported unsupervised AE linear-probe development accuracy: **60.5%**, using ReLU hidden
  layers, tanh bottleneck, LR 3e-4, dropout 0.05, 100 epochs and summed reconstruction loss.
- In the tanh joint model at gamma 0.9, changing reconstruction reduction from sum to mean changes
  development accuracy from **38.8% to 52.3%**. Gamma 0, which removes the reconstruction objective,
  reaches **55.4%**.
- The largest development result printed in the incomplete joint LR/epoch grid is **59.7%**,
  at LR 3e-4 and 200 epochs. This is not a held-out test score.

These results support investigating loss scale and nonlinearities rather than attributing the gap
to segment sampling alone. They do not evaluate the source's mixed sigmoid/tanh/ReLU architecture.
Another issue is that the ablation ranks raw reconstruction losses across `sum` and `mean` rows;
those losses differ by a factor of 500 and are not comparable without conversion.

The partial user's results were inspected after the main new experiments completed. They informed
this audit and documentation; they did not retroactively become new training results.

## Per-genre held-out results

| Genre | Primary precision | Primary recall | Primary F1 | Ensemble F1 | Paper F1 |
|---|---:|---:|---:|---:|---:|
| Classical | 0.7671 | 0.7375 | 0.7520 | 0.7737 | 0.7790 |
| Jazz | 0.5376 | 0.5652 | 0.5511 | 0.6015 | 0.6160 |
| Metal | 0.5927 | 0.5976 | 0.5951 | 0.6004 | 0.5337 |
| Pop | 0.6540 | 0.6405 | 0.6472 | 0.6527 | 0.6380 |

The ensemble's macro F1 is **0.6571**. Test class supports are 259 classical, 253 jazz, 246 metal
and 242 pop. The complete dataset is balanced; the source-style shuffled split is not stratified.
All comparisons use genre names, avoiding the earlier notebook's pop/metal label-order difference.

**The paper's Table 2 may not be a test-set table.** Its four recalls average to **0.641**. With
near-balanced classes, mean recall is close to accuracy, and 0.641 is exactly the paper's
*development* accuracy rather than its 65.3% test accuracy. The per-genre comparison above is
therefore approximate.

The primary model's confusion matrix (rows true, columns predicted; classical, jazz, metal, pop):

| | classical | jazz | metal | pop |
|---|---:|---:|---:|---:|
| classical | 191 | 59 | 9 | 0 |
| jazz | 48 | 143 | 38 | 24 |
| metal | 4 | 37 | 147 | 58 |
| pop | 6 | 27 | 54 | 155 |

As in the paper, metal and pop are mistaken for each other (58 and 54). Jazz, however, is
confused in both directions with classical (48/59) and with metal (38/37), more than the paper's
discussion suggests.

## Reconstruction is a material limitation

The primary model's reconstructed values are all zero in the inspected examples. Its complete test
reconstruction MSE is **0.01125987**, equal to the MSE of the all-zero predictor. The released
authors' joint checkpoint exhibits the same behavior on its saved development inputs: reconstruction
MSE **0.01143465**, also equal to the zero predictor.

The vanilla autoencoder shows the same zero-MSE on every split, and its development-selected
epoch is 6. Its 64-D codes are tiny (about ±0.1 in PCA) and show no genre structure.

Because the reconstruction term is constant once the decoder outputs zero, it sends no gradient
to the encoder. The joint model is then effectively a deep 500→…→4 classifier. Its gain over the
two-layer baseline comes from depth and training, not from combining reconstruction with
classification. The 64-D bottleneck plot (Figure 5) also does not show the clean separation the
paper describes: pop and metal spread along one direction, while classical and jazz stay packed
together. The 4-D logits (Figure 6) do separate classical from pop/metal.

Thus classification accuracy and separable embedding plots do not establish successful waveform
reconstruction. The final ReLU output is retained for source fidelity, with this collapse exposed
in the notebook. Fixing that output or redesigning the reconstruction task would be a new method,
not an exact implementation of the released network.

## Data fidelity and generalization limits

The [Marsyas GTZAN archive mirror](https://huggingface.co/datasets/marsyas/gtzan/tree/main/data) is
pinned by SHA-256 `b28cc067ff6199bd826f5d1a6931458586d64acae7f44b2e882b7a97af057531`.
Its `jazz.00054.wav` is headerless, as in the supplied Kaggle source. A fallback decodes that exact
known file as mono PCM16 little-endian at 22.05 kHz. It uses existing samples and adds no padding,
zero examples or duplicated recordings. This decoding assumption is explicitly recorded in
`audio_files.csv`; it does not prove bitwise equivalence with the authors' converted recording.

Comparing our pooled examples to the repository's 2,000 saved development inputs gives
**1,996 matches**, agreeing in labels for every matched row. Four saved jazz examples remain
unmatched. The CSV does not recover the paper's exact final development/test partition or seed.

The source-style segment split has **371 songs shared between train and test**. In a separate
experiment with no song shared across subsets, the same development-selected primary architecture
achieves **59.2% test accuracy**. This distinct protocol should not be substituted for the paper's
random-segment comparison. Song separation alone also does not remove GTZAN's cross-file
duplicates and artist overlap.

For context only, restoring the authors' actual released checkpoints on their historical 2,000-row
development CSV gives **45.85% baseline** and **60.8% joint-model** accuracy. These are historical
development diagnostics, not independently held-out replication scores.

## Validation and reproducible artifacts

`validate_reproduction.py` reads the actual TensorFlow function bodies from the cloned repository,
substituting only the removed Xavier initializer API. With identical trained weights and inputs,
the new PyTorch port passes forward, loss, gradient and ten-update Adam comparisons:

| Check | Maximum absolute error |
|---|---:|
| 64-D encoding | 8.49e-7 |
| Reconstruction | 0 |
| Logits | 5.33e-6 |
| Joint loss | 0 |
| Gradients | 4.47e-8 |
| Weights after ten Adam updates | 3.58e-7 |

Pooling matches `librosa.load(sr=22050)` plus the original reshape/mean operations exactly on the
checked recording from each genre. Adam's epsilon placement is adjusted to reproduce TensorFlow 1's
update equation rather than silently using PyTorch's different convention. Random initial weights
and dropout masks still differ across frameworks; no bitwise training-trajectory claim is made.

The notebook is fully executed with saved outputs and all code cells compile. The two-epoch smoke
run is saved separately and never mixed into the full results. Full experiment checkpoints include
configuration, implementation, data and split fingerprints, the chosen development epoch and a
`test_used_for_selection: false` field. Every classified example has a saved prediction CSV.

Key outputs. The published repository contains only this report and the notebook. Running the
notebook recreates every file under `results/`. The `analysis/` files, the helper scripts
(`validate_reproduction.py`, `audit_reference.py`, `extract_ablation_results.py`) and the earlier
supplied notebooks are kept in the author's local project and are not published here. Their key
numbers are quoted in this report.

- `results/source_grounded/all_metrics.csv`: every main model, split and checkpoint. A loss term
  the model does not have (MSE for the baseline, cross-entropy for the vanilla autoencoder) is blank.
- `results/source_grounded/paper_comparison.csv` and `per_class_comparison.csv`: paper comparisons.
- `results/source_grounded/figures/`: raw PCA, architecture, learning curves, confusion matrices,
  bottleneck PCA, logit PCA and reconstruction sanity plot.
- `results/ensemble_extension/`: probabilities, metrics and hashed ensemble membership.
- `results/song_disjoint/`: separate song-disjoint results and manifests.
- `analysis/numerical_validation.json` and `reference_checkpoint_audit.json`: numerical evidence.
- `analysis/partial_ablation_results/`: measurements recovered from the newly supplied notebook.

All supplied notebooks and the PDF have been preserved. The honest outcome is a substantial
single-model improvement close to the paper, plus a separately identified ensemble that exceeds
its numerical reference. The conflicting released specifications and unavailable original split
prevent a claim of exact numerical replication of the paper.
