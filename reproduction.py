"""Source-grounded reproduction of Sawhney, Vasavada & Wang (2018).

Ported from arjunsawknee/Genre-Extraction, commit
00b1802e24d1ab6fb3135b24a44976a51033ed24. See REPRODUCTION_REPORT.md.
The notebook embeds this file so it is also usable on its own in Colab.
"""
# %% imports
from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import platform
import random
import subprocess
import sys
import tarfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import soundfile as sf
import torch
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from torch import nn
from torch.nn import functional as F

GENRES = ['classical', 'jazz', 'metal', 'pop']  # Original repository order.
SOURCE_COMMIT = '00b1802e24d1ab6fb3135b24a44976a51033ed24'
DATA_URL = 'https://huggingface.co/datasets/marsyas/gtzan/resolve/main/data/genres.tar.gz'
DATA_SHA256 = 'b28cc067ff6199bd826f5d1a6931458586d64acae7f44b2e882b7a97af057531'
HEADERLESS_JAZZ_SHA256 = 'eb8be6097ee1f32965fb9545cb1621b4685e4b0d44c6584cf0ab97d244ad535f'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_num_threads(min(4, os.cpu_count() or 1))


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # No TF32 or mixed precision: these tiny networks do not need it.
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)


# %% data
def download_data(root=Path('data')):
    """Use the pinned Marsyas archive mirror; audit its damaged jazz WAV explicitly."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    archive = root / 'genres.tar.gz'
    if not archive.exists():
        temporary = archive.with_suffix('.part')
        with requests.get(DATA_URL, stream=True, timeout=(30, 120)) as response:
            response.raise_for_status()
            total = 0
            with temporary.open('wb') as f:
                for block in response.iter_content(4 * 1024 * 1024):
                    f.write(block)
                    total += len(block)
                    if total // (200 * 1024 * 1024) != (total-len(block)) // (200 * 1024 * 1024):
                        print(f'Downloaded {total / 1e6:.0f} MB', flush=True)
        temporary.replace(archive)
    if sha256(archive) != DATA_SHA256:
        raise ValueError('The GTZAN archive checksum does not match the pinned source.')
    target = (root / 'original').resolve()
    with tarfile.open(archive) as tar:
        for member in tar.getmembers():
            if (not member.isfile() or not member.name.endswith(('.au', '.wav'))
                    or Path(member.name).name.startswith('._')):
                continue
            if not any('/' + genre + '/' in member.name for genre in GENRES):
                continue
            dest = (target / member.name).resolve()
            if not dest.is_relative_to(target):
                raise ValueError(f'Unsafe archive path: {member.name}')
            dest.parent.mkdir(parents=True, exist_ok=True)
            if not dest.exists():
                with tar.extractfile(member) as src, dest.open('wb') as dst:
                    dst.write(src.read())
    candidates = [p for p in target.rglob('genres') if all((p / g).is_dir() for g in GENRES)]
    if not candidates:
        raise FileNotFoundError('Expected genres/{classical,jazz,metal,pop} in GTZAN archive.')
    return candidates[0]


def read_audio(path):
    """Decode normal audio; explicitly recover the known headerless PCM file.

    This repair is an assumption about the published mirror, NOT a claim of
    bit-for-bit equivalence with the authors' private converted WAV collection.
    Only the exact pinned file is allowed this fallback. No zero padding.
    """
    try:
        audio, sr = sf.read(path, dtype='float32')
        return audio, sr, 'soundfile'
    except sf.LibsndfileError:
        if Path(path).name != 'jazz.00054.wav' or sha256(path) != HEADERLESS_JAZZ_SHA256:
            raise
        audio, sr = sf.read(path, dtype='float32', format='RAW', subtype='PCM_16',
                            endian='LITTLE', channels=1, samplerate=22050)
        return audio, sr, 'assumed_headerless_little_endian_pcm16_22050_mono'


def pool_song(audio, sample_rate):
    """Exactly the used part of load_data(): 20 successive 20,000-sample chunks.

    Original code pools 600,000 samples but only uses the first 400,000.
    No resampling, normalization, absolute value, or engineered features.
    """
    if sample_rate != 22050:
        raise ValueError(f'Expected native 22050 Hz, received {sample_rate}')
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim != 1 or len(audio) < 600000:
        raise ValueError(f'Expected mono audio with >=600000 samples, got {audio.shape}')
    return audio[:600000].reshape(15000, 40).mean(axis=1)[:10000].reshape(20, 500)


def build_dataset(audio_root, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    xs, ys, manifest, files = [], [], [], []
    for label, genre in enumerate(GENRES):
        paths = sorted(p for p in (list((Path(audio_root)/genre).glob('*.au')) or
                       list((Path(audio_root)/genre).glob('*.wav'))) if not p.name.startswith('._'))
        if len(paths) != 100:
            raise ValueError(f'{genre}: expected 100 recordings, got {len(paths)}')
        for path in paths:
            audio, sr, decoder = read_audio(path)
            x = pool_song(audio, sr)
            xs.append(x)
            ys.extend([label] * 20)
            files.append({'song': path.stem, 'sha256': sha256(path), 'sample_rate': sr,
                          'samples': len(audio), 'genre': genre, 'decoder': decoder})
            for segment in range(20):
                manifest.append({'row': len(manifest), 'song': path.stem, 'genre': genre,
                                 'label': label, 'segment': segment,
                                 'start_sample': segment * 20000,
                                 'end_sample': (segment+1) * 20000,
                                 'duration_seconds': 20000/22050})
    X, y = np.concatenate(xs).astype(np.float32), np.asarray(ys, dtype=np.int64)
    assert X.shape == (8000, 500) and np.isfinite(X).all()
    assert np.array_equal(np.bincount(y), [2000]*4)
    frame = pd.DataFrame(manifest)
    frame.to_csv(output_dir/'segments.csv', index=False)
    pd.DataFrame(files).to_csv(output_dir/'audio_files.csv', index=False)
    np.savez_compressed(output_dir/'pooled_audio.npz', X=X, y=y)
    return X, y, frame


def make_split(y, manifest, output_dir, seed=42, song_disjoint=False):
    """Paper's random segment split; optional separate song-disjoint audit."""
    if song_disjoint:
        songs = manifest.drop_duplicates('song')
        tr_songs, rest = train_test_split(songs.song, train_size=300, random_state=seed,
                                          stratify=songs.label)
        remaining = songs.set_index('song').loc[rest]
        dv_songs, te_songs = train_test_split(rest, train_size=50, random_state=seed,
                                             stratify=remaining.label)
        parts = [np.flatnonzero(manifest.song.isin(s)) for s in [tr_songs, dv_songs, te_songs]]
    else:
        # The source used Python random.shuffle followed by slicing, without stratification.
        order = list(range(len(y)))
        random.Random(seed).shuffle(order)
        parts = [np.asarray(order[:6000]), np.asarray(order[6000:7000]), np.asarray(order[7000:])]
    split = dict(zip(['train', 'dev', 'test'], parts))
    assert [len(split[k]) for k in split] == [6000, 1000, 1000]
    assert len(set(np.concatenate(parts))) == 8000
    tag = 'song_disjoint' if song_disjoint else 'paper_segment'
    annotated = manifest.copy()
    annotated['split'] = ''
    for name, indices in split.items():
        annotated.loc[indices, 'split'] = name
    annotated.to_csv(Path(output_dir)/f'{tag}_split.csv', index=False)
    song_sets = {k: set(manifest.iloc[v].song) for k, v in split.items()}
    overlap = {a+'_'+b: len(song_sets[a] & song_sets[b])
               for a,b in [('train','dev'), ('train','test'), ('dev','test')]}
    if song_disjoint:
        assert all(v == 0 for v in overlap.values())
    info = {'seed': seed, 'protocol': tag, 'overlapping_songs': overlap,
            'class_counts': {k: np.bincount(y[v], minlength=4).tolist() for k,v in split.items()}}
    (Path(output_dir)/f'{tag}_split.json').write_text(json.dumps(info, indent=2))
    print(json.dumps(info, indent=2))
    return split


# %% architecture
class Baseline(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(500, 128), nn.Linear(128, 4)])
        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.layers:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, x):
        hidden = torch.tanh(self.layers[0](x))
        return {'encoding': hidden, 'logits': self.layers[1](hidden)}


class Autoencoder(nn.Module):
    def __init__(self, joint=False, keep_probability=0.9):
        super().__init__()
        self.joint = joint
        self.dropout = nn.Dropout(1-keep_probability)
        self.encoder = nn.ModuleList([nn.Linear(a,b) for a,b in
                                     zip([500,256,192,128], [256,192,128,64])])
        self.decoder = nn.ModuleList([nn.Linear(a,b) for a,b in
                                     zip([64,128,192,256], [128,192,256,500])])
        self.head = nn.ModuleList([nn.Linear(a,b) for a,b in
                                  zip([64,32,16], [32,16,4])]) if joint else nn.ModuleList()
        for layer in list(self.encoder) + list(self.decoder) + list(self.head):
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, x):
        e = torch.sigmoid(self.encoder[0](x))
        e = self.dropout(torch.tanh(self.encoder[1](e)))
        e = F.relu(self.encoder[2](e))
        z = F.relu(self.encoder[3](e))
        d = torch.sigmoid(self.decoder[0](z))
        d = self.dropout(torch.sigmoid(self.decoder[1](d)))
        d = F.relu(self.decoder[2](d))
        reconstructed = F.relu(self.decoder[3](d))  # Source has ReLU even at output.
        result = {'encoding': z, 'reconstruction': reconstructed}
        if self.joint:
            c = torch.tanh(self.head[0](z))
            c = torch.tanh(self.head[1](c))
            result['logits'] = self.head[2](c)
        return result


def objective(outputs, x, y, mode='repository'):
    mse = F.mse_loss(outputs['reconstruction'], x) if 'reconstruction' in outputs else x.new_zeros(())
    ce = F.cross_entropy(outputs['logits'], y) if 'logits' in outputs else x.new_zeros(())
    if 'reconstruction' not in outputs:
        return ce, mse, ce
    if 'logits' not in outputs:
        return mse, mse, ce
    if mode == 'repository':
        loss = mse + 0.1 * ce
    elif mode == 'paper_literal':
        loss = 0.9 * x.shape[1] * mse + 0.1 * ce
    else:
        raise ValueError(mode)
    return loss, mse, ce


# %% training
@dataclass
class Experiment:
    name: str
    kind: str
    epochs: int
    keep_probability: float = 0.9
    objective: str = 'repository'
    seed: int = 42
    learning_rate: float = 1e-4
    batch_size: int = 512
    shuffle_each_epoch: bool = False  # Source shuffles once before training.


def create_model(config):
    return (Baseline() if config.kind == 'baseline' else
            Autoencoder(joint=config.kind == 'joint', keep_probability=config.keep_probability)).to(DEVICE)


@torch.no_grad()
def evaluate(model, x, y, objective_mode='repository', batch_size=1024):
    model.eval()
    totals = np.zeros(3, dtype=np.float64)
    logits, encodings, predictions = [], [], []
    for start in range(0, len(x), batch_size):
        xb, yb = x[start:start+batch_size], y[start:start+batch_size]
        out = model(xb)
        values = objective(out, xb, yb, objective_mode)
        totals += np.array([v.item() for v in values]) * len(xb)
        encodings.append(out['encoding'].cpu().numpy())
        if 'logits' in out:
            logits.append(out['logits'].cpu().numpy())
            predictions.append(out['logits'].argmax(1).cpu().numpy())
    metrics = dict(zip(['loss', 'mse', 'cross_entropy'], (totals / len(x)).tolist()))
    result = {'metrics': metrics, 'encoding': np.concatenate(encodings)}
    if logits:
        result['logits'] = np.concatenate(logits)
        result['prediction'] = np.concatenate(predictions)
        metrics['accuracy'] = accuracy_score(y.cpu().numpy(), result['prediction'])
    return result


def train_experiment(config, X, y, split, output_dir, resume=True):
    """Train on train only; select checkpoints on dev only. Test is never read."""
    seed_everything(config.seed)
    folder = Path(output_dir)/config.name
    folder.mkdir(parents=True, exist_ok=True)
    data_hash = hashlib.sha256(X.tobytes()+y.tobytes()).hexdigest()
    split_hash = hashlib.sha256(b''.join(np.asarray(split[k],dtype=np.int64).tobytes()
                                        for k in ['train','dev','test'])).hexdigest()
    provenance = {'config': asdict(config), 'data_sha256': data_hash, 'split_sha256': split_hash,
                  'source_commit': SOURCE_COMMIT, 'implementation_sha256': sha256(__file__)
                  if '__file__' in globals() else 'embedded-notebook', 'torch': torch.__version__}
    model = create_model(config)
    completed = folder/'completed.json'
    if resume and completed.exists():
        saved = json.loads(completed.read_text())
        if all(saved.get(k) == provenance[k] for k in ['config','data_sha256','split_sha256','source_commit','torch','implementation_sha256']):
            model.load_state_dict(torch.load(folder/'best.pt', map_location=DEVICE, weights_only=True))
            print(f'{config.name}: loaded completed training; test remains unused.', flush=True)
            return model, pd.read_csv(folder/'history.csv'), saved
    train_idx = split['train'].copy()
    # Reproduce the second, training-only shuffle in the released source.
    local_order = list(range(len(train_idx)))
    random.Random(config.seed+1).shuffle(local_order)
    train_idx = train_idx[local_order]
    xtr = torch.as_tensor(X[train_idx], device=DEVICE)
    ytr = torch.as_tensor(y[train_idx], device=DEVICE)
    xdv = torch.as_tensor(X[split['dev']], device=DEVICE)
    ydv = torch.as_tensor(y[split['dev']], device=DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate,
                                 betas=(0.9,0.999), eps=1e-8, foreach=True)
    history, best_score, best_epoch, step = [], -math.inf, 0, 0
    started = time.perf_counter()
    # Check deterministic cuBLAS configuration before the first CUDA multiplication.
    for epoch in range(1, config.epochs+1):
        model.train()
        if config.shuffle_each_epoch:
            order = torch.randperm(len(xtr), device=DEVICE)
        else:
            order = torch.arange(len(xtr), device=DEVICE)
        total, correct = 0., 0
        for start in range(0, len(xtr), config.batch_size):
            indices = order[start:start+config.batch_size]
            xb, yb = xtr[indices], ytr[indices]
            optimizer.zero_grad(set_to_none=True)
            outputs = model(xb)
            loss, _, _ = objective(outputs, xb, yb, config.objective)
            if not torch.isfinite(loss):
                raise FloatingPointError(f'Non-finite loss in {config.name}, epoch {epoch}')
            loss.backward()
            step += 1
            # TF1 Adam adds epsilon to sqrt(UNCORRECTED v), unlike PyTorch.
            # This step-dependent epsilon makes the two update equations agree.
            optimizer.param_groups[0]['eps'] = 1e-8/math.sqrt(1-0.999**step)
            optimizer.step()
            total += loss.item() * len(xb)
            if 'logits' in outputs:
                correct += (outputs['logits'].argmax(1) == yb).sum().item()
        dev = evaluate(model, xdv, ydv, config.objective)['metrics']
        row = {'epoch': epoch, 'train_online_loss': total/len(xtr),
               'train_online_accuracy': correct/len(xtr) if config.kind != 'autoencoder' else np.nan,
               **{'dev_'+k: v for k,v in dev.items()}}
        history.append(row)
        score = dev.get('accuracy', -dev['mse'])
        if score > best_score:
            best_score, best_epoch = score, epoch
            torch.save(model.state_dict(), folder/'best.pt')
        if epoch == 1 or epoch % 25 == 0 or epoch == config.epochs:
            print(f'{config.name}: epoch {epoch}/{config.epochs}, dev {score:.4f}, '
                  f'{time.perf_counter()-started:.1f}s', flush=True)
            pd.DataFrame(history).to_csv(folder/'history.csv', index=False)
    torch.save(model.state_dict(), folder/'last.pt')
    provenance.update(best_epoch=best_epoch, best_dev_score=best_score,
                      elapsed_seconds=time.perf_counter()-started, test_used_for_selection=False)
    completed.write_text(json.dumps(provenance, indent=2))
    pd.DataFrame(history).to_csv(folder/'history.csv', index=False)
    model.load_state_dict(torch.load(folder/'best.pt', map_location=DEVICE, weights_only=True))
    return model, pd.DataFrame(history), provenance


# %% reporting
def score_model(model, config, X, y, split, output_dir, checkpoint='best'):
    folder = Path(output_dir)/config.name
    model.load_state_dict(torch.load(folder/f'{checkpoint}.pt', map_location=DEVICE, weights_only=True))
    rows, results = [], {}
    for name, indices in split.items():
        result = evaluate(model, torch.as_tensor(X[indices],device=DEVICE),
                          torch.as_tensor(y[indices],device=DEVICE), config.objective)
        results[name] = result
        rows.append({'experiment':config.name, 'checkpoint':checkpoint, 'split':name,
                     'n':len(indices), **result['metrics']})
        if 'prediction' in result:
            pd.DataFrame({'row':indices, 'truth':y[indices], 'prediction':result['prediction']}).to_csv(
                folder/f'{checkpoint}_{name}_predictions.csv', index=False)
            report = classification_report(y[indices], result['prediction'], labels=range(4),
                                           target_names=GENRES, output_dict=True, zero_division=0)
            (folder/f'{checkpoint}_{name}_classification.json').write_text(json.dumps(report,indent=2))
    pd.DataFrame(rows).to_csv(folder/f'{checkpoint}_metrics.csv',index=False)
    return pd.DataFrame(rows), results


def pca_panel(train_embeddings, test_embeddings, test_labels, ax, title):
    # Fit PCA on train, transform held-out examples. Labels only color points.
    pca = PCA(n_components=2, svd_solver='full').fit(train_embeddings)
    points = pca.transform(test_embeddings)
    for label, genre in enumerate(GENRES):
        chosen = test_labels == label
        ax.scatter(points[chosen,0],points[chosen,1],s=10,alpha=.55,label=genre)
    ax.set_title(title)
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.legend(fontsize=8)
    return pca


def make_figures(X, y, split, experiments, results, histories, output_dir):
    output_dir = Path(output_dir)
    figure_dir = output_dir/'figures'
    figure_dir.mkdir(exist_ok=True)
    tr, te = split['train'], split['test']
    fig, ax = plt.subplots(figsize=(7,5))
    pca_panel(X[tr],X[te],y[te],ax,'Figure 1: raw 500-D pooled waveforms')
    fig.tight_layout();fig.savefig(figure_dir/'figure1_raw_pca.png',dpi=160);plt.close(fig)
    final_name = 'paper_architecture_repository_loss'
    ae_name = 'vanilla_autoencoder'
    base_name = 'baseline'
    if final_name not in results:
        return
    fig, axes = plt.subplots(1,2,figsize=(12,4))
    h=histories[final_name]
    axes[0].plot(h.epoch,h.train_online_accuracy,label='Train (online, dropout enabled)')
    axes[0].plot(h.epoch,h.dev_accuracy,label='Dev (dropout disabled)')
    axes[0].set(xlabel='Epoch',ylabel='Accuracy',title='Figure 3: genre classification')
    axes[0].legend()
    axes[1].plot(h.epoch,h.train_online_loss,label='Train joint loss')
    axes[1].plot(h.epoch,h.dev_loss,label='Dev joint loss')
    axes[1].set(xlabel='Epoch',ylabel='MSE + 0.1 CE',title='Joint objective');axes[1].legend()
    fig.tight_layout();fig.savefig(figure_dir/'figure3_learning_curves.png',dpi=160);plt.close(fig)
    pred=results[final_name]['test']['prediction']
    cm=confusion_matrix(y[te],pred,labels=range(4))
    pd.DataFrame(cm,index=GENRES,columns=GENRES).to_csv(figure_dir/'confusion_counts.csv')
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for ax, matrix,title in zip(axes,[cm,cm/cm.sum(axis=1,keepdims=True)],['Counts','Row normalized']):
        ax.imshow(matrix,cmap='Blues')
        ax.set(xticks=range(4),yticks=range(4),xticklabels=GENRES,yticklabels=GENRES,
               xlabel='Predicted genre',ylabel='True genre',title='Figure 4: '+title)
        for i in range(4):
            for j in range(4):
                ax.text(j,i,f'{matrix[i,j]:.2f}' if title != 'Counts' else str(matrix[i,j]),
                        ha='center',va='center',color='white' if matrix[i,j] > matrix.max()/2 else 'black')
    fig.tight_layout();fig.savefig(figure_dir/'figure4_confusion.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    for name,ax,title in [(ae_name,axes[0],'Vanilla autoencoder: 64-D'),
                          (final_name,axes[1],'Joint autoencoder: 64-D')]:
        if name in results:
            r=results[name]
            pca_panel(r['train']['encoding'],r['test']['encoding'],y[te],ax,title)
    fig.tight_layout();fig.savefig(figure_dir/'figure5_bottleneck_pca.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    for name,ax,title in [(base_name,axes[0],'Baseline: 4-D logits'),
                          (final_name,axes[1],'Joint autoencoder: 4-D logits')]:
        if name in results:
            r=results[name]
            pca_panel(r['train']['logits'],r['test']['logits'],y[te],ax,title)
    fig.tight_layout();fig.savefig(figure_dir/'figure6_logits_pca.png',dpi=160);plt.close(fig)


def default_experiments(seed=42):
    return [
        Experiment('baseline','baseline',200,seed=seed),
        Experiment('vanilla_autoencoder','autoencoder',400,keep_probability=.8,seed=seed),
        Experiment('released_code_200_epochs','joint',200,keep_probability=.8,seed=seed),
        Experiment('paper_architecture_repository_loss','joint',500,keep_probability=.9,seed=seed),
        Experiment('paper_literal_equation','joint',500,keep_probability=.9,
                   objective='paper_literal',seed=seed),
    ]


def save_environment(output_dir):
    output_dir=Path(output_dir)
    env={'python':sys.version,'platform':platform.platform(),'torch':torch.__version__,
         'numpy':np.__version__,'pandas':pd.__version__,'device':str(DEVICE),
         'gpu':torch.cuda.get_device_name() if torch.cuda.is_available() else None,
         'source_commit':SOURCE_COMMIT,'archive_sha256':DATA_SHA256}
    (output_dir/'environment.json').write_text(json.dumps(env,indent=2))
    frozen=subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True)
    (output_dir/'environment-freeze.txt').write_text(frozen)
    print(json.dumps(env,indent=2))
    return env


def run_suite(output_dir='results/source_grounded', smoke=False, song_disjoint=False):
    output_dir=Path(output_dir)
    output_dir.mkdir(parents=True,exist_ok=True)
    root=download_data()
    X,y,manifest=build_dataset(root,output_dir)
    split=make_split(y,manifest,output_dir,song_disjoint=song_disjoint)
    save_environment(output_dir)
    experiments=default_experiments()
    if song_disjoint:
        experiments=[c for c in experiments if c.name in ['baseline','paper_architecture_repository_loss']]
    if smoke:
        for config in experiments:
            config.epochs=2
    models,histories,provenance={},{},{}
    # Finish every training/selection decision before inspecting the test set.
    for config in experiments:
        models[config.name],histories[config.name],provenance[config.name]=train_experiment(
            config,X,y,split,output_dir)
    all_rows,results=[],{}
    for config in experiments:
        for checkpoint in ['best','last']:
            table,r=score_model(models[config.name],config,X,y,split,output_dir,checkpoint)
            all_rows.append(table)
            if checkpoint=='best':
                results[config.name]=r
    metrics=pd.concat(all_rows,ignore_index=True)
    metrics.to_csv(output_dir/'all_metrics.csv',index=False)
    make_figures(X,y,split,experiments,results,histories,output_dir)
    print(metrics.to_string(index=False))
    return X,y,manifest,split,models,histories,provenance,metrics,results


if __name__ == '__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',default='results/source_grounded')
    parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--song-disjoint',action='store_true')
    args=parser.parse_args()
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    run_suite(args.output,args.smoke,args.song_disjoint)
