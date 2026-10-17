"""Recover only explicitly printed measurements from the user's partial run."""
import json
import re
from pathlib import Path
import pandas as pd

nb=json.loads(Path('abalation.ipynb').read_text(encoding='utf8'))
def streams(index):
    return '\n'.join(''.join(o.get('text',[])) for o in nb['cells'][index].get('outputs',[]))

tables={}
tables['data_preprocessing'] = pd.DataFrame([
    {'preprocessing':a,'selection':b,'dev_accuracy':float(c)}
    for a,b,c in re.findall(r'^(resample20k|native_crop20k|native_adaptive500) (\w+) val= ([\d.]+)',streams(26),re.M)])
tables['baseline_lr_epochs']=pd.DataFrame([
    {'learning_rate':float(a),'epochs':int(b),'dev_accuracy':float(c)}
    for a,b,c in re.findall(r'^([\d.e+-]+) (\d+) val= ([\d.]+)',streams(30),re.M)])
tables['baseline_batch']=pd.DataFrame([
    {'batch_size':int(a),'dev_accuracy':float(b)}
    for a,b in re.findall(r'^batch (\d+) val= ([\d.]+)',streams(31),re.M)])
tables['autoencoder_screen']=pd.DataFrame([
    {'activation':a,'latent_activation':b,'loss_reduction':c,'learning_rate':float(d),
     'probe_dev_accuracy':float(e),'dev_reconstruction_loss':float(f)}
    for a,b,c,d,e,f in re.findall(r'^(\w+) (\w+) (sum|mean) ([\d.e+-]+) probe= ([\d.]+) recon= ([\d.]+)',streams(35),re.M)])
tables['autoencoder_epochs']=pd.DataFrame([
    {'epochs':int(a),'probe_dev_accuracy':float(b),'dev_reconstruction_loss':float(c)}
    for a,b,c in re.findall(r'^(\d+) probe= ([\d.]+) recon= ([\d.]+)',streams(36),re.M)])
tables['joint_gamma']=pd.DataFrame([
    {'loss_reduction':a,'gamma':float(b),'dev_accuracy':float(c),'reconstruction_to_ce_ratio':float(d)}
    for a,b,c,d in re.findall(r'^(sum|mean) gamma ([\d.]+) val ([\d.]+) ratio ([\d.]+)',streams(43),re.M)])
tables['joint_lr_epochs_partial']=pd.DataFrame([
    {'learning_rate':float(a),'epochs':int(b),'dev_accuracy':float(c)}
    for a,b,c in re.findall(r'^([\d.e+-]+) (\d+) val ([\d.]+)',streams(45),re.M)])
out=Path('analysis/partial_ablation_results');out.mkdir(exist_ok=True)
for name,table in tables.items():
    table['source_notebook']='abalation.ipynb'
    table.to_csv(out/(name+'.csv'),index=False)
    print(name,len(table),'printed measurements')
assert len(tables['joint_gamma'])==20
assert len(tables['autoencoder_screen'])==72
