"""Render the frozen A/B comparison from saved predictions, never from a model."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def render(runs,inventory,output):
    output.mkdir(parents=True,exist_ok=False)
    expected=json.loads(inventory.read_text())['runs']['files'];sources={};rows=[]
    for seed in (17,42,101):
        pair={};alignment=None
        for variant in ('A','B'):
            name=f'final-test/{variant}-{seed}.npz';path=runs/name
            digest=hashlib.sha256(path.read_bytes()).hexdigest()
            if digest!=expected[name]['sha256']:raise ValueError('Persisted prediction hash changed')
            sources[name]=digest
            with np.load(path,allow_pickle=False) as data:
                current=(data['ids'].copy(),data['z_true'].copy())
                if alignment is not None and not all(np.array_equal(a,b) for a,b in zip(alignment,current)):
                    raise ValueError('Pairs are not aligned')
                alignment=current;z=data['z_true'].astype(float)
                error=np.abs(data['prediction'].astype(float)-z)/(1+z)
                pair[variant]={'n':len(z),'outliers':int((error>.15).sum()),'eta15':float((error>.15).mean()),'mae_normalized':float(error.mean())}
        rows.append({'seed':seed,**pair,'delta_eta_pp':100*(pair['A']['eta15']-pair['B']['eta15']),
                     'delta_mae_normalized':pair['A']['mae_normalized']-pair['B']['mae_normalized']})
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,2,figsize=(10,4.8),layout='constrained')
    for ax,key,title,label in [(axes[0],'delta_eta_pp','Outliers: A − B','Difference (percentage points)'),
                               (axes[1],'delta_mae_normalized','Normalized absolute error: A − B','Difference in mean |Δz|/(1+z)')]:
        values=[r[key] for r in rows]
        ax.bar([str(r['seed']) for r in rows],values,color=['#1e7285' if v<0 else '#b95638' for v in values],width=.55)
        ax.axhline(0,color='#444',lw=.8);ax.set_title(title);ax.set_xlabel('Paired seed');ax.set_ylabel(label)
        ax.margins(y=.3)
        for i,v in enumerate(values):ax.annotate(f'{v:+.4f}',(i,v),xytext=(0,-16 if v<0 else 5),textcoords='offset points',ha='center',fontsize=9)
    fig.suptitle('Frozen checkpoint pairs · saved predictions only',fontsize=13)
    fig.savefig(output/'ab-persisted.png',dpi=160);fig.savefig(output/'ab-persisted.svg');plt.close(fig)
    result={'schema':'batch5c-ab-persisted-v1','sources':sources,'rows':rows,'inference_calls':0,
            'interpretation':'Conditional on three previously selected pairs. Negative favors A; MAE worsens for two seeds. No new selection, physical independence or CNN/v2.1 superiority claim.'}
    (output/'ab-persisted.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'rows':rows,'inference_calls':0,'output':str(output)}))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('runs','inventory','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();render(a.runs,a.inventory,a.output)
