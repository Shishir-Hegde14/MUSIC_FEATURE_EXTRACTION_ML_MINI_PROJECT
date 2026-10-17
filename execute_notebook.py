"""Execute the complete notebook with the project's isolated Python kernel."""
import os
from pathlib import Path

root=Path(__file__).resolve().parent
os.environ['JUPYTER_PATH']=str(root/'.venv'/'share'/'jupyter')
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import nbformat
from nbclient import NotebookClient

path=root/'ML_Mini_Project_Source_Grounded_Reproduction.ipynb'
nb=nbformat.read(path,as_version=4)
nbformat.validate(nb)
for i,cell in enumerate(nb.cells):
    if cell.cell_type=='code':compile(cell.source,f'{path.name}:cell{i}','exec')

def progress(cell,cell_index,**kwargs):
    print(f'Executed cell {cell_index+1}/{len(nb.cells)}',flush=True)
    nbformat.write(nb,path)

client=NotebookClient(nb,timeout=1800,kernel_name='ml-paper-reproduction',
                      resources={'metadata':{'path':str(root)}},on_cell_executed=progress)
try:
    client.execute()
finally:
    nbformat.write(nb,path)
print('Notebook execution completed.',flush=True)
