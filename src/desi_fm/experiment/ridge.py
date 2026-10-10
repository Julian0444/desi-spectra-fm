"""Fixed 128-band ridge control; exactly three regularization fits."""
from __future__ import annotations

from pathlib import Path
import time
import numpy as np

from .artifacts import atomic_write, digest, file_record, source_identity, environment, write_json
from .metrics import save_predictions

ALPHAS = (.1,1.,10.)


def features(flux, valid):
    if flux.shape != valid.shape or flux.ndim != 2 or flux.shape[1] < 128:
        raise ValueError('Ridge requires >=128 grid pixels')
    edges = np.linspace(0,flux.shape[1],129,dtype=int)
    result = np.zeros((len(flux),256),dtype=np.float64)
    for i,(lo,hi) in enumerate(zip(edges[:-1],edges[1:])):
        count = valid[:,lo:hi].sum(axis=1)
        np.divide((flux[:,lo:hi]*valid[:,lo:hi]).sum(axis=1,dtype=np.float64),count,
                  out=result[:,i],where=count>0)
        result[:,128+i] = count/(hi-lo)
    return result


def feature_matrix(data, indices=None, batch_size=256):
    indices = np.arange(len(data)) if indices is None else np.asarray(indices)
    result = np.empty((len(indices),256),dtype=np.float64)
    for first in range(0,len(indices),batch_size):
        batch = data.batch(indices[first:first+batch_size],masked=False)
        result[first:first+batch_size] = features(batch['flux'],batch['valid'])
    return result


def fit_three(x, y):
    """Scaler and normal equations depend solely on the supplied train features."""
    if x.ndim!=2 or y.shape!=(len(x),) or not len(x) or not np.isfinite(x).all() or not np.isfinite(y).all() or (y<0).any():
        raise ValueError('Invalid train inputs for ridge')
    mean, scale = x.mean(axis=0), x.std(axis=0)
    scale[scale<1e-12] = 1.
    target = np.log1p(y); intercept = target.mean()
    standardized = (x-mean)/scale
    gram, rhs = standardized.T@standardized, standardized.T@(target-intercept)
    models = []
    for alpha in ALPHAS:
        coef = np.linalg.solve(gram+alpha*np.eye(x.shape[1]),rhs)
        models.append({'alpha': np.array(alpha), 'mean':mean, 'scale':scale, 'coef':coef, 'intercept':np.array(intercept)})
    return models


def ridge_predict(model, x):
    result = np.expm1(((x-model['mean'])/model['scale'])@model['coef']+model['intercept'])
    if not np.isfinite(result).all():
        raise ValueError('Nonfinite ridge prediction')
    return np.maximum(result,0.)


def run_ridge(train, validation, directory):
    if train.role!='train' or validation.role!='validation':
        raise PermissionError('Ridge fits train and selects on validation only')
    directory = Path(directory); directory.mkdir(parents=True,exist_ok=False)
    identity = {'train':train.identity,'validation':validation.identity,'alphas':list(ALPHAS),
                'source':source_identity(),'environment':environment(),'contract':'128 fixed valid means + 128 fractions; train scaler; log1p target'}
    manifest = {'status':'incomplete','identity':identity}
    write_json(directory/'ridge.json',manifest)
    start = time.perf_counter(); x = feature_matrix(train); train_seconds = time.perf_counter()-start
    start = time.perf_counter(); xv = feature_matrix(validation); validation_seconds = time.perf_counter()-start
    start = time.perf_counter(); models = fit_three(x,train.z); fit_seconds = time.perf_counter()-start
    candidates = []
    for model in models:
        alpha = float(model['alpha']); path = directory/f'ridge-{alpha:g}.npz'
        atomic_write(path,lambda stream: np.savez(stream,**model))
        output = ridge_predict(model,xv)
        predictions = directory/f'validation-{alpha:g}.npz'
        summary = save_predictions(predictions,validation,np.arange(len(validation)),{'prediction':output},
                                   {'variant':'ridge','alpha':alpha,'run_identity':digest(identity)})
        candidates.append({'alpha':alpha,'path':path.name,'file':file_record(path),'predictions':predictions.name,
                           'key':[summary['primary']['eta15'],summary['primary']['mae_norm'],alpha]})
    manifest.update(status='complete',fits=3,candidates=candidates,best=min(candidates,key=lambda c:c['key']),
                    timings={'train_features_seconds':train_seconds,'validation_features_seconds':validation_seconds,'three_fits_seconds':fit_seconds})
    write_json(directory/'ridge.json',manifest)
    return manifest
