"""Fixed equal-probability ensemble of the three predeclared initialization seeds.

This is an explicitly separate extension, not a single-model paper reproduction.
"""
import json
import os
from pathlib import Path
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score,classification_report
import reproduction as r


def evaluate_ensemble(X,y,split,work_dir=Path('.')):
    work_dir=Path(work_dir)
    members=[work_dir/'results'/'source_grounded'/'paper_architecture_repository_loss'/'best.pt',
             work_dir/'results'/'seed_robustness'/'joint_seed_43'/'best.pt',
             work_dir/'results'/'seed_robustness'/'joint_seed_44'/'best.pt']
    result_dir=work_dir/'results'/'ensemble_extension'
    result_dir.mkdir(parents=True,exist_ok=True)
    accumulated={k:np.zeros((len(indices),4),dtype=np.float64) for k,indices in split.items()}
    for path in members:
        model=r.Autoencoder(joint=True,keep_probability=.9).to(r.DEVICE)
        model.load_state_dict(torch.load(path,map_location=r.DEVICE,weights_only=True))
        for name,indices in split.items():
            result=r.evaluate(model,torch.as_tensor(X[indices],device=r.DEVICE),
                              torch.as_tensor(y[indices],device=r.DEVICE))
            # Use softmax probabilities, not votes or a test-fitted mixing rule.
            scores=result['logits'].astype(np.float64)
            scores-=scores.max(axis=1,keepdims=True)
            probabilities=np.exp(scores)
            probabilities/=probabilities.sum(axis=1,keepdims=True)
            accumulated[name]+=probabilities/len(members)
    rows=[]
    for name,indices in split.items():
        probabilities=accumulated[name]
        predictions=probabilities.argmax(axis=1)
        rows.append({'experiment':'equal_probability_ensemble_3_seeds','split':name,
                     'n':len(indices),'accuracy':accuracy_score(y[indices],predictions)})
        table=pd.DataFrame({'row':indices,'truth':y[indices],'prediction':predictions})
        for j,genre in enumerate(r.GENRES):table['probability_'+genre]=probabilities[:,j]
        table.to_csv(result_dir/f'{name}_predictions.csv',index=False)
        report=classification_report(y[indices],predictions,labels=range(4),target_names=r.GENRES,
                                     output_dict=True,zero_division=0)
        (result_dir/f'{name}_classification.json').write_text(json.dumps(report,indent=2))
    metrics=pd.DataFrame(rows)
    metrics.to_csv(result_dir/'metrics.csv',index=False)
    (result_dir/'protocol.json').write_text(json.dumps({
        'kind':'separate exploratory extension, not exact single-model reproduction',
        'combination':'arithmetic mean of softmax probabilities','weights':[1/3]*3,
        'seeds':[42,43,44],'members':[{'path':str(p),'sha256':r.sha256(p)} for p in members],
        'selection':'each member selected on development accuracy only',
        'test_fitted_weights':False,'note':'Added after inspecting the individual-run results; no new independent test set.'},indent=2))
    return metrics


if __name__=='__main__':
    data=np.load('results/source_grounded/pooled_audio.npz')
    manifest=pd.read_csv('results/source_grounded/segments.csv')
    split=r.make_split(data['y'],manifest,Path('results/source_grounded'))
    print(evaluate_ensemble(data['X'],data['y'],split).to_string(index=False),flush=True)
