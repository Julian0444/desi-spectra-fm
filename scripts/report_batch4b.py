"""Reproduce batch-4B validation decisions and reports from persisted artifacts.

This script never fits a model or performs inference. `validation` cannot load
the test partition; `report` requires the completed frozen final predictions.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from desi_fm.experiment.artifacts import (
    file_record, read_json, source_identity, verify_record, write_json,
)
from desi_fm.experiment.metrics import load_predictions, paired_analysis
from desi_fm.historical import metrics

SEEDS = (17, 42, 101)
METHODS = ('A', 'B', 'CNN')
ROOT = Path(__file__).resolve().parents[1]


def validation_review(runs, output):
    if output.exists():
        raise ValueError('Use a new validation-review destination')
    rows, reports, reference = [], {}, None
    source = source_identity()
    for method in METHODS:
        for seed in SEEDS:
            key = f'{method}-{seed}'
            directory = runs/key
            report = read_json(directory/'run.json')
            identity = report['identity']
            assert report['status'] == 'complete' and report['step'] == 10000
            assert identity['config'] == read_json(ROOT/'configs/batch4.json')
            assert identity['source_sha256'] == source['sha256']
            assert identity['device'] == 'mps' and identity['mode'] == 'training'
            assert report['sampler_position'] == 320000
            assert {event['step'] for event in report['validation_events']} == set(range(500, 10001, 500))
            assert len(report['timings']) == 10000
            candidates = []
            for step in range(500, 10001, 500):
                path = directory/f'validation-{step:06d}.npz'
                arrays, provenance = load_predictions(path)
                assert provenance['role'] == 'validation' and not provenance['artificial_mask']
                assert len(arrays['ids']) == 10000
                if reference is None:
                    reference = arrays
                for field in ('ids', 'groups', 'z_true', 'zwarn', 'coverage'):
                    np.testing.assert_array_equal(arrays[field], reference[field])
                score = metrics(arrays['z_true'], arrays['prediction'])
                candidates.append([score['eta15'], score['mae_norm'], step])
            assert min(candidates) == report['best']['key']
            pointer = read_json(directory/'checkpoints.json')['best']
            verify_record(directory/pointer['path'], {k:pointer[k] for k in ('bytes', 'sha256')})
            rows.append(dict(method=method, seed=seed, key=report['best']['key'],
                             checkpoint=str((directory/pointer['path']).resolve()),
                             checkpoint_file={k:pointer[k] for k in ('bytes', 'sha256')},
                             validations=candidates, trace=report['trace'],
                             class_weights_sha256=identity['class_weights_sha256']))
            reports[key] = report
    for seed in SEEDS:
        a, b = reports[f'A-{seed}'], reports[f'B-{seed}']
        assert a['trace'] == b['trace'] and a['class_weights'] == b['class_weights']
    ridge = read_json(runs/'ridge/ridge.json')
    assert ridge['status'] == 'complete' and ridge['fits'] == 3
    assert ridge['identity']['alphas'] == [.1, 1., 10.]
    assert [c['alpha'] for c in ridge['candidates']] == [.1, 1., 10.]
    scalers = []
    for candidate in ridge['candidates']:
        path = runs/'ridge'/candidate['path']
        verify_record(path, candidate['file'])
        with np.load(path, allow_pickle=False) as saved:
            assert float(saved['alpha']) == candidate['alpha']
            scalers.append((saved['mean'].copy(), saved['scale'].copy()))
        arrays, provenance = load_predictions(runs/'ridge'/candidate['predictions'])
        assert provenance['role'] == 'validation'
        for field in ('ids', 'groups', 'z_true', 'zwarn', 'coverage'):
            np.testing.assert_array_equal(arrays[field], reference[field])
        score = metrics(arrays['z_true'], arrays['prediction'])
        assert candidate['key'] == [score['eta15'], score['mae_norm'], candidate['alpha']]
    for scaler in scalers[1:]:
        for a, b in zip(scalers[0], scaler):
            np.testing.assert_array_equal(a, b)
    assert ridge['best'] == min(ridge['candidates'], key=lambda c:c['key'])
    families = {method: np.mean([r['key'][:2] for r in rows if r['method'] == method], axis=0).tolist()
                for method in METHODS}
    families['ridge'] = ridge['best']['key'][:2]
    winner = min(families, key=lambda method:(*families[method], method))
    if winner == 'ridge':
        candidate = dict(method='ridge', alpha=ridge['best']['alpha'],
                         checkpoint=str((runs/'ridge'/ridge['best']['path']).resolve()),
                         checkpoint_file=ridge['best']['file'])
    else:
        candidate = sorted([r for r in rows if r['method'] == winner],
                           key=lambda r:(*r['key'], r['seed']))[1]
        candidate = {k:v for k,v in candidate.items() if k != 'validations'}
    report = dict(status='passed',source=source,neural=rows,ridge=ridge['best'],
                  family_validation_scores=families,future_use_candidate=candidate,
                  selection_plan=file_record(ROOT/'docs/BATCH_4B_ANALYSIS.md'),
                  report_generator=file_record(__file__),
                  decision_scope='Validation only, before test. All seeds retained. No served-model change.',
                  caveat='Selected validation scores are optimistic; method comparisons include input/architecture differences.')
    write_json(output, report)
    return report


def plot_reports(runs, output, rows, summaries, paired, validation):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,
                         'figure.dpi':140,'savefig.bbox':'tight'})
    colors = {'A':'#177e89','B':'#c04e36','CNN':'#7255a5','ridge':'#6b7280'}
    fig, ax = plt.subplots(figsize=(9,4.5))
    for i,row in enumerate(rows):
        lo,hi = row['eta_interval95']
        ax.errorbar(100*row['metrics']['eta15'],i,
                    xerr=[[100*(row['metrics']['eta15']-lo)],[100*(hi-row['metrics']['eta15'])]],
                    fmt='o',color=colors[row['method']],capsize=3)
    ax.set(yticks=range(len(rows)),yticklabels=[r['name'] for r in rows],
           xlabel='Outlier rate eta15 (%)',title='Final test: every checkpoint, fixed-seed object/group 95% intervals')
    ax.invert_yaxis();ax.grid(axis='x',alpha=.2)
    fig.savefig(output/'test-eta15.png');plt.close(fig)
    fig, axes = plt.subplots(1,3,figsize=(12,3.6),sharey=True)
    for ax,method in zip(axes,METHODS):
        for row in validation['neural']:
            if row['method'] == method:
                values=np.array(row['validations'])
                ax.plot(values[:,2],100*values[:,0],label=str(row['seed']))
        ax.set(title=method,xlabel='Optimizer update');ax.grid(alpha=.2);ax.legend(title='Seed',fontsize=8)
    axes[0].set_ylabel('Validation eta15 (%)')
    fig.suptitle('All scheduled validations; checkpoint selection uses eta15, MAE, then step')
    fig.tight_layout();fig.savefig(output/'validation-trajectories.png');plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,3.6))
    delta=100*np.array(paired['delta_eta_per_seed'])
    ax.scatter(delta,[0,1,2],color=colors['A'],s=45)
    mean=100*paired['mean_delta_eta'];lo,hi=100*np.array(paired['paired_95_interval'])
    ax.errorbar(mean,3,xerr=[[mean-lo],[hi-mean]],fmt='s',color='#1f2937',capsize=5)
    ax.axvline(0,color='#9ca3af',linestyle='--')
    ax.set(yticks=range(4),yticklabels=['Seed 17','Seed 42','Seed 101','Paired mean, 95% CI'],
           xlabel='eta_A − eta_B (percentage points)',title='Reconstruction effect under the fixed recipe')
    ax.invert_yaxis();ax.grid(axis='x',alpha=.2)
    fig.savefig(output/'paired-effect.png');plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(12,3.6),sharex=True,sharey=True)
    for ax,method in zip(axes,METHODS):
        for seed,style in zip(SEEDS,('-','--',':')):
            curve=summaries[f'{method}-{seed}']['risk_coverage']
            assert curve is not None
            ax.plot(100*np.array(curve['coverage']),100*np.array(curve['eta15']),
                    linestyle=style,color=colors[method],label=str(seed))
        ax.set(title=method,xlabel='Retained coverage (%)',xlim=(0,100))
        ax.grid(alpha=.2);ax.legend(title='Seed',fontsize=8)
    axes[0].set_ylabel('Outlier rate among retained (%)')
    fig.suptitle('Fixed confidence scores; no threshold selected on test; ridge has no score')
    fig.tight_layout();fig.savefig(output/'risk-coverage.png');plt.close(fig)


def final_report(runs, validation_path, output):
    output.mkdir(parents=True, exist_ok=False)
    frozen=read_json(runs/'frozen.json');final=read_json(runs/'final-test/final.json')
    validation=read_json(validation_path)
    assert final['status']=='complete'
    assert validation['source']['sha256']==frozen['source']['sha256']==source_identity()['sha256']
    assert validation['selection_plan']==file_record(ROOT/'docs/BATCH_4B_ANALYSIS.md')
    assert validation['report_generator']==file_record(__file__)
    rows,summaries,reference=[],{},None
    for method,seed in [(m,s) for m in METHODS for s in SEEDS]+[('ridge',None)]:
        name=method if seed is None else f'{method}-{seed}'
        path=runs/'final-test'/(name+'.npz')
        arrays,provenance=load_predictions(path)
        assert provenance['role']=='test' and len(arrays['ids'])==10000
        if reference is None:reference=arrays
        for field in ('ids','groups','z_true','zwarn','coverage'):
            np.testing.assert_array_equal(arrays[field],reference[field])
        summary=read_json(path.with_suffix('.summary.json'))
        assert summary['primary']==metrics(arrays['z_true'],arrays['prediction'])
        summaries[name]=summary
        row=dict(name=name,method=method,seed=seed,metrics=summary['primary'],
                 secondary=summary['secondary'],strata=summary['strata'],
                 eta_interval95=final['intervals'][name]['metrics']['all']['interval95'],
                 quality_intervals=final['intervals'][name]['metrics'],
                 prediction=str(path.resolve()),prediction_file=file_record(path))
        if seed is not None:
            run=read_json(runs/name/'run.json')
            selected=next(r for r in frozen['neural'] if r['variant']==method and r['seed']==seed)
            parts={k:float(sum(t[k] for t in run['timings'])) for k in
                   ('read_preprocess_seconds','forward_seconds','backward_seconds','optimizer_seconds')}
            row.update(selected_step=selected['step'],validation_key=selected['validation_key'],
                       checkpoint=selected['checkpoint'],checkpoint_file=selected['file'],
                       parameters=run['parameters'],updates=run['step'],cost={
                           **parts,'update_total_seconds':sum(parts.values()),
                           'validation_seconds':sum(t['seconds'] for t in run['validation_events']),
                           'checkpoint_seconds':sum(t['write_seconds'] for t in run['checkpoint_events']),
                           'peak_host_rss_bytes':run['peak_host_rss_bytes'],
                           'max_sampled_mps_driver_bytes':max(t['mps_driver_allocated_bytes'] for t in run['timings']),
                           'final_inference_and_persist_seconds':read_json(runs/'final-test'/(name+'.cost.json'))['seconds']},
                       recovery_seconds=run['recovery_seconds'],recovered_from=run['recovered_from'],
                       normalization_fallbacks=run['fallbacks'])
        else:
            ridge=read_json(runs/'ridge/ridge.json')
            row.update(alpha=ridge['best']['alpha'],validation_key=ridge['best']['key'],
                       checkpoint=frozen['ridge']['path'],checkpoint_file=frozen['ridge']['file'],
                       fits=ridge['fits'],cost=ridge['timings'])
        rows.append(row)
    paired=paired_analysis([runs/'final-test'/f'A-{s}.npz' for s in SEEDS],
                           [runs/'final-test'/f'B-{s}.npz' for s in SEEDS])
    assert paired==final['paired']==read_json(runs/'paired-analysis.json')
    family_means={method:{key:float(np.mean([r['metrics'][key] for r in rows if r['method']==method]))
                         for key in ('eta15','mae_norm','nmad','bias')} for method in (*METHODS,'ridge')}
    result=dict(schema='batch4b-report-v1',rows=rows,family_means=family_means,paired=paired,
                sensitivity=frozen['sensitivity'],future_use_candidate=validation['future_use_candidate'],
                source=frozen['source'],frozen_file=file_record(runs/'frozen.json'),
                validation_review_file=file_record(validation_path),analysis_plan=file_record(ROOT/'docs/BATCH_4B_ANALYSIS.md'),
                report_generator=file_record(__file__),
                family_mean_scope='Descriptive means over fixed seeds, not 30000 independent objects',
                memory_scope='RSS maximum and sampled MPS driver allocations on unified memory; not additive')
    write_json(output/'results.json',result)
    plot_reports(runs,output,rows,summaries,paired,validation)
    header='| Candidate | Selected step / alpha | Validation eta15 | Test outliers / N | Test eta15 (95% CI) | MAE norm | NMAD | Bias |'
    lines=[header,'|---|---:|---:|---:|---:|---:|---:|---:|']
    for row in rows:
        m=row['metrics'];lo,hi=row['eta_interval95']
        lines.append(f"| {row['name']} | {row.get('selected_step',row.get('alpha'))} | {100*row['validation_key'][0]:.3f}% | {m['outliers']}/{m['n']} | {100*m['eta15']:.3f}% [{100*lo:.3f}, {100*hi:.3f}] | {m['mae_norm']:.6f} | {m['nmad']:.6f} | {m['bias']:.6f} |")
    (output/'results-table.md').write_text('\n'.join(lines)+'\n')
    inventory={str(p.relative_to(runs)):file_record(p) for p in sorted(runs.rglob('*'))
               if p.is_file() and '.lock' not in p.name and output not in p.parents}
    write_json(output/'artifact-inventory.json',dict(root=str(runs.resolve()),files=inventory,
               total_bytes=sum(r['bytes'] for r in inventory.values())))
    print({'output':str(output),'criterion_met':paired['criterion_met'],'family_means':family_means})


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    for command in ('validation','report'):
        p=sub.add_parser(command)
        p.add_argument('--runs',type=Path,required=True)
        p.add_argument('--output',type=Path,required=True)
        if command=='report':p.add_argument('--validation-review',type=Path,required=True)
    args=parser.parse_args()
    if args.command=='validation':
        result=validation_review(args.runs,args.output)
        print({'status':result['status'],'family_validation_scores':result['family_validation_scores'],
               'future_use_candidate':result['future_use_candidate']})
    else:
        final_report(args.runs,args.validation_review,args.output)
