# Music genre paper reproduction

Open the **[main notebook](ML_Mini_Project_Source_Grounded_Reproduction.ipynb)** for the complete
implementation, executed outputs, comparisons, and figures. It is standalone and can also run
in [Google Colab](https://colab.research.google.com/github/Shishir-Hegde14/MUSIC_FEATURE_EXTRACTION_ML_MINI_PROJECT/blob/main/ML_Mini_Project_Source_Grounded_Reproduction.ipynb).
Read the **[reproduction report](REPRODUCTION_REPORT.md)** for the findings and qualifications.

The measured primary single-model test accuracy is **63.6%**. A separately labelled equal-weight
three-seed probability ensemble achieves **65.8%**, compared with the paper's **65.3%**.
The ensemble is an exploratory extension, not an exact single-model replication.

## Run locally

The experiments were run in an isolated Python 3.11 environment with CUDA-enabled PyTorch on an
RTX 4050. The notebook includes dependency setup. For the command-line scripts on a fresh checkout,
create an environment first:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-reproduction.txt
```

Select `.venv/Scripts/python.exe` as the notebook interpreter. The first run downloads the pinned
GTZAN archive. Saved metrics, figures, predictions and trained checkpoints are included in this
repository; the local environment, audio downloads, pooled audio caches and smoke runs are excluded.

```powershell
.\.venv\Scripts\python.exe reproduction.py
.\.venv\Scripts\python.exe run_robustness.py
.\.venv\Scripts\python.exe evaluate_ensemble.py
.\.venv\Scripts\python.exe reproduction.py --song-disjoint --output results/song_disjoint
```

The notebook runs these stages itself; the CLI scripts are alternatives. Checkpoints and
configuration/data/implementation fingerprints prevent repeating completed training unnecessarily.
`reproduction.py --smoke --output results/smoke` performs a two-epoch software check only.

To create a fresh environment, use Python 3.11, install `requirements-reproduction.txt`, and install
a CUDA-enabled PyTorch build if GPU execution is needed. The exact measured environment is saved
in `results/source_grounded/environment-freeze.txt`.

For the optional numerical comparison against the original TensorFlow code, clone the pinned
reference separately after preparing the dataset:

```powershell
git clone https://github.com/arjunsawknee/Genre-Extraction.git reference/Genre-Extraction
git -C reference/Genre-Extraction checkout --detach 00b1802e24d1ab6fb3135b24a44976a51033ed24
.\.venv\Scripts\python.exe -m pip install tensorflow==2.20.0 librosa==0.11.0
.\.venv\Scripts\python.exe validate_reproduction.py
```

To execute the notebook using `execute_notebook.py`, first install its local kernel:

```powershell
.\.venv\Scripts\python.exe -m ipykernel install --prefix .venv --name ml-paper-reproduction --display-name "Python (ML paper reproduction)"
```

## Files

- `reproduction.py`: preprocessing, models, TensorFlow-compatible Adam math, training, metrics, PCA.
- `reference/Genre-Extraction/`: optional local clone of the authors' repository, excluded from Git.
- `validate_reproduction.py`: numerical comparison with original TensorFlow function bodies. Requires
  optional `tensorflow==2.20.0` and `librosa==0.11.0`.
- `audit_reference.py`: authors' checkpoint and saved development-input checks.
- `extract_ablation_results.py`: extracts printed measurements from the added partial `abalation.ipynb`.
- `analysis/`: PDF text, input audits, numerical validation, extracted partial ablation measurements.
- `results/source_grounded/`: five complete main experiments, checkpoints, split manifests, predictions,
  metrics, per-class comparisons and all paper-style figures.
- `results/seed_robustness/`: both additional initialization seeds.
- `results/ensemble_extension/`: ensemble probabilities, predictions, metrics and protocol.
- `results/song_disjoint/`: separate unseen-song evaluation.

`build_notebook.py` regenerates the notebook and clears saved outputs; use it only when changing its
source. `execute_notebook.py` executes it with the project's isolated kernel and saves outputs.

The supplied PDF and original notebooks have been preserved. The added `abalation.ipynb` was also
read and its partial results are discussed in the new notebook and report.
