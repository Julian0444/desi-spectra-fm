"""Persisted historical diagnostic metrics and visible-only reconstruction controls."""
import csv
import hashlib
import json
from pathlib import Path
import numpy as np

FORMULAS = {'normalized_error':'r=(z_pred-z_true)/(1+z_true)', 'mae_norm':'mean(abs(r))',
            'eta15':'mean(abs(r)>0.15)', 'nmad':'1.4826*median(abs(r-median(r)))',
            'bias':'mean(r)', 'strict_accuracy':'mean(abs(r)<0.15)',
            'rmse':'sqrt(sum(hidden_valid_squared_error)/hidden_valid_pixels); null when denominator=0'}


def metrics(z, prediction):
    z,prediction=np.asarray(z,dtype=np.float64),np.asarray(prediction,dtype=np.float64)
    if z.shape!=prediction.shape or not np.isfinite(z).all() or not np.isfinite(prediction).all():raise ValueError('Invalid paired metric arrays')
    if not len(z):return {'n':0,**{k:None for k in ('mae_norm','eta15','nmad','bias','strict_accuracy')}}
    r=(prediction-z)/(1+z)
    return {'n':len(z),'mae_norm':float(np.mean(abs(r))),'eta15':float(np.mean(abs(r)>.15)),
            'nmad':float(1.4826*np.median(abs(r-np.median(r)))),'bias':float(np.mean(r)),
            'strict_accuracy':float(np.mean(abs(r)<.15)),'outliers':int((abs(r)>.15).sum())}


def fixed_masks(ids,n_tokens,seed=20261009,ratio=.5):
    masks=[]
    for oid in ids:
        digest=hashlib.sha256(f'desi-reconstruction-v1:{seed}:{oid}'.encode()).digest()
        rng=np.random.Generator(np.random.PCG64(int.from_bytes(digest[:8],'little')))
        masks.append(rng.random(n_tokens)<ratio)
    return np.asarray(masks,dtype=bool)


def visible_baselines(target,valid,hidden,width=301):
    target,valid,hidden=np.asarray(target),np.asarray(valid,dtype=bool),np.asarray(hidden,dtype=bool)
    visible=valid & ~hidden
    if visible.sum()<1:raise ValueError('No visible pixels for baseline')
    positions=np.flatnonzero(visible)
    values=target[visible]
    mean=float(values.mean(dtype=np.float64))
    # Copy only observed values. No hidden target values enter any baseline.
    observed=np.zeros(len(target),dtype=np.float64);observed[visible]=values
    kernel=np.ones(width)
    start=width//2
    numerator=np.convolve(observed,kernel,'full')[start:start+len(target)]
    denominator=np.convolve(visible.astype(float),kernel,'full')[start:start+len(target)]
    smooth=np.full(len(target),mean)
    np.divide(numerator,denominator,out=smooth,where=denominator>0)
    return {'mean':np.full(len(target),mean,dtype=np.float32),
            'linear':np.interp(np.arange(len(target)),positions,values).astype(np.float32),
            'moving301':smooth.astype(np.float32)}


def masked_rmse(target,prediction,scored):
    scored=np.asarray(scored,dtype=bool);n=int(scored.sum())
    sse=float(np.square(np.asarray(prediction,dtype=np.float64)[scored]-np.asarray(target,dtype=np.float64)[scored]).sum())
    return {'masked_valid_pixels':n,'sse':sse,'rmse':float(np.sqrt(sse/n)) if n else None}


def write_csv(path,rows,fields=None):
    with Path(path).open('x',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields or list(rows[0]))
        writer.writeheader();writer.writerows(rows)


def summarize(directory,make_figures=True):
    directory=Path(directory)
    with (directory/'predictions.csv').open() as f:rows=list(csv.DictReader(f))
    ids=[r['object_id'] for r in rows]
    if len(set(ids))!=len(ids):raise ValueError('Duplicate persisted predictions')
    z=np.array([float(r['z_true']) for r in rows]);quality=np.array([int(r['zwarn']) for r in rows]);coverage=np.array([float(r['coverage']) for r in rows])
    predictions={k:np.array([float(r[k]) for r in rows]) for k in ('v1_scalar','v21_map','v21_secondary')}
    summary={'scope':'historical diagnostic, not independent generalization evidence','formulas':FORMULAS,
             'redshift':{name:metrics(z,pred) for name,pred in predictions.items()},'full_spectrum_reconstruction_rmse':None,'strata':{},
             'quality_interpretation':'ZWARN=False represents an original warning, not necessarily a wrong label'}
    for axis,groups in [('z',[(f'[{lo},{hi})',(z>=lo)&(z<hi)) for lo,hi in zip([0,.5,1,1.5,2,2.5],[.5,1,1.5,2,2.5,np.inf])]),
                        ('quality',[('original_warning',quality==0),('no_original_warning',quality==1)]),
                        ('coverage',[(f'[{lo},{hi})',(coverage>=lo)&(coverage<hi)) for lo,hi in zip([0,.5,.9,.99],[.5,.9,.99,1.000001])])]:
        summary['strata'][axis]={label:{name:metrics(z[keep],pred[keep]) for name,pred in predictions.items()} for label,keep in groups}
    out1=abs((predictions['v1_scalar']-z)/(1+z))>.15;out2=abs((predictions['v21_map']-z)/(1+z))>.15
    summary['paired']={'common_ids':len(ids),'both_success':int((~out1 & ~out2).sum()),'both_outlier':int((out1 & out2).sum()),
                       'success_to_outlier':int((~out1 & out2).sum()),'outlier_to_success':int((out1 & ~out2).sum())}
    score=np.array([float(r['v21_confidence']) for r in rows]);order=np.lexsort((np.array(ids),-score))
    risk=[{'retained':i+1,'coverage':(i+1)/len(ids),'eta15':float(out2[order[:i+1]].mean()),'minimum_score':float(score[order[i]])} for i in range(len(ids))]
    write_csv(directory/'risk-coverage.csv',risk,fields=['retained','coverage','eta15','minimum_score'])
    with np.load(directory/'reconstruction.npz',allow_pickle=False) as rec:
        rec_ids=rec['ids'].tolist();scored=rec['valid'].astype(bool)&rec['hidden'].astype(bool)
        summary['reconstruction']={'n':len(rec_ids),'methods':{name:masked_rmse(rec['target'],rec[name],scored) for name in ('v1','v21','mean','linear','moving301')},
                                   'normalization':'support v2; statistics from visible valid pixels only, then mask tokens; targets clipped arcsinh',
                                   'mask_seed':20261009,'ratio':.5,'moving_width_fixed_before_evaluation':301}
    manifest=json.loads((directory/'manifest.json').read_text());exclusions=json.loads((directory/'exclusions.json').read_text())
    requested=manifest['requested_ids']
    for stage,evaluated in [('redshift',ids),('reconstruction',rec_ids)]:
        rejected=[r['object_id'] for r in exclusions[stage]]
        if len(rejected)!=len(set(rejected)) or set(rejected)&set(evaluated) or len(rejected)+len(evaluated)!=len(requested):
            raise ValueError('Invalid accounting of evaluated/rejected objects')
    if set(ids)|{r['object_id'] for r in exclusions['redshift']}!=set(requested):raise ValueError('Unexplained missing redshift objects')
    if set(rec_ids)|{r['object_id'] for r in exclusions['reconstruction']}!=set(requested):raise ValueError('Unexplained missing reconstruction objects')
    if [oid for oid in requested if oid in set(ids)]!=ids:raise ValueError('Persisted prediction order differs')
    if [oid for oid in requested if oid in set(rec_ids)]!=rec_ids:raise ValueError('Persisted reconstruction order differs')
    summary['accounting']={'requested':len(requested),'redshift_evaluated':len(ids),'redshift_rejected':len(exclusions['redshift']),
                           'reconstruction_evaluated':len(rec_ids),'reconstruction_rejected':len(exclusions['reconstruction'])}
    with np.load(directory/'predictions.npz',allow_pickle=False) as saved:
        assert saved['ids'].tolist()==ids
        for name,pred in predictions.items():np.testing.assert_array_equal(saved[name],pred)
    (directory/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    if make_figures:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,axes=plt.subplots(1,2,figsize=(10,4))
        for ax,name in zip(axes,('v1_scalar','v21_map')):
            ax.scatter(z,predictions[name],s=5,alpha=.4);ax.plot([0,4],[0,4],color='black',lw=1)
            ax.set(xlabel='Catalog redshift',ylabel='Predicted redshift',title=f'{name}: historical, full spectrum')
        fig.tight_layout();fig.savefig(directory/'redshift.png',dpi=150);plt.close(fig)
        fig,ax=plt.subplots(figsize=(6,4));ax.plot([r['coverage'] for r in risk],[r['eta15'] for r in risk])
        ax.set(xlabel='Retained coverage',ylabel='Outlier fraction (>0.15)',title='Historical concentration ranking; no validated threshold')
        fig.tight_layout();fig.savefig(directory/'risk-coverage.png',dpi=150);plt.close(fig)
        fig,ax=plt.subplots(figsize=(7,4));methods=summary['reconstruction']['methods']
        ax.bar(list(methods),[methods[k]['rmse'] if methods[k]['rmse'] is not None else np.nan for k in methods]);ax.set(ylabel='RMSE on hidden valid pixels',title='Shared fixed masks; visible-only normalization')
        fig.tight_layout();fig.savefig(directory/'reconstruction.png',dpi=150);plt.close(fig)
    return summary
