"""Audit reference CSV provenance and evaluate the authors' released checkpoint.

These are historical development data, not a new held-out test result.
"""
import json
import os
from pathlib import Path
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL','3')
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
import tensorflow as tf
import torch
import reproduction as r

repo=Path('reference/Genre-Extraction')
saved_x=pd.read_csv(repo/'songs_dev.csv').to_numpy(dtype=np.float32)
saved_y=pd.read_csv(repo/'labels_dev.csv').to_numpy(dtype=np.float32).argmax(1)
data=np.load('results/source_grounded/pooled_audio.npz')
dist,idx=cKDTree(data['X']).query(saved_x)
matched=dist<1e-5
pd.DataFrame({'saved_csv_row':np.arange(len(saved_x)),'nearest_current_row':idx,
              'l2_distance':dist,'matches':matched,'saved_label':saved_y}).to_csv(
                  'analysis/reference_csv_matches.csv',index=False)
report={'csv_rows':len(saved_x),'matching_pooled_inputs':int(matched.sum()),
        'matching_label_agreement':float(np.mean(data['y'][idx[matched]]==saved_y[matched])),
        'unmatched_rows':np.flatnonzero(~matched).tolist(),
        'interpretation':'Historical development-set diagnostic; not independent test performance.'}
for name,model,base in [
    ('baseline',r.Baseline(),'modelWeights/tfsoftmax/tfsoftmax-200'),
    ('joint',r.Autoencoder(joint=True),'modelWeights/softmaxautoencoder-200')]:
    reader=tf.train.load_checkpoint(str(repo/base))
    groups=[(model.layers,'softmax')] if name=='baseline' else [
        (model.encoder,'encoder'),(model.decoder,'decoder'),(model.head,'softmax')]
    with torch.no_grad():
        for layers,group in groups:
            for i,layer in enumerate(layers,1):
                layer.weight.copy_(torch.from_numpy(reader.get_tensor(f'W{i}_{group}').T))
                layer.bias.copy_(torch.from_numpy(reader.get_tensor(f'b{i}_{group}').reshape(-1)))
    result=r.evaluate(model,torch.from_numpy(saved_x),torch.from_numpy(saved_y))
    report[name+'_released_checkpoint_saved_dev_accuracy']=result['metrics']['accuracy']
    if name=='joint':
        report['joint_reconstruction_mse']=result['metrics']['mse']
        report['zero_reconstruction_mse']=float(np.mean(saved_x**2))
Path('analysis/reference_checkpoint_audit.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2),flush=True)
