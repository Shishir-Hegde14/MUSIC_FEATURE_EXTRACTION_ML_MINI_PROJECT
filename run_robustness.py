"""Fixed, declared follow-up checks; test labels never drive model selection."""
import json
import os
from pathlib import Path
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
os.environ.setdefault('MPLBACKEND', 'Agg')
import numpy as np
import pandas as pd
import reproduction as r

output=Path('results/seed_robustness')
output.mkdir(parents=True,exist_ok=True)
data=np.load('results/source_grounded/pooled_audio.npz')
X,y=data['X'],data['y']
manifest=pd.read_csv('results/source_grounded/segments.csv')
split=r.make_split(y,manifest,output,seed=42)
configs=[r.Experiment(f'joint_seed_{seed}','joint',500,seed=seed) for seed in [43,44]]
(output/'declared_protocol.json').write_text(json.dumps({'seeds':[42,43,44],
    'split_seed':42,'epochs':500,'keep_probability':.9,'objective':'repository',
    'purpose':'Report every seed, never choose a seed using test accuracy.'},indent=2))
trained=[]
for config in configs:
    model,history,provenance=r.train_experiment(config,X,y,split,output)
    trained.append((config,model))
tables=[]
for config,model in trained:
    for checkpoint in ['best','last']:
        table,_=r.score_model(model,config,X,y,split,output,checkpoint)
        tables.append(table)
table=pd.concat(tables,ignore_index=True)
table.to_csv(output/'all_metrics.csv',index=False)
print(table.to_string(index=False),flush=True)
