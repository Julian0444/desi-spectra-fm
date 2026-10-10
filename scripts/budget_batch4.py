"""Cost extrapolation from bounded train-only profiles; no fitting or inference."""
from __future__ import annotations

import argparse
import math
from pathlib import Path
import numpy as np

from desi_fm.experiment.artifacts import file_record, read_json, write_json


def budget(root,output):
    root=Path(root);output=Path(output)
    if output.exists():raise ValueError('Use a new budget report')
    directories={'A':'profile-A','B':'profile-B','CNN':'profile-CNN-fixed'}
    profiles={key:read_json(root/d/'profile.json') for key,d in directories.items()}
    measured={}
    components=('read_preprocess_seconds','forward_seconds','backward_seconds','optimizer_seconds')
    for key,r in profiles.items():
        rows=r['timings'];steady=rows[4:]
        totals=[sum(row[k] for k in components) for row in steady]
        measured[key]={'updates':r['step'],'parameters':r['parameters'],
                       'effective_batch':r['identity']['config']['micro_batch']*r['identity']['config']['accumulation'],
                       'steady_updates':len(steady),'median_update_seconds':float(np.median(totals)),
                       'p95_update_seconds':float(np.quantile(totals,.95)),
                       'component_mean_seconds':{c:float(np.mean([row[c] for row in steady])) for c in components},
                       'all_updates_seconds':float(sum(sum(row[c] for c in components) for row in rows)),
                       'checkpoint_write_seconds':r['checkpoint_events'][-1]['write_seconds'],
                       'checkpoint_bytes':r['checkpoint_events'][-1]['bytes'],
                       'recovery_seconds':r['profile_recovery_seconds'],
                       'evaluation_seconds_per_object':r['evaluation_profile']['seconds']/r['evaluation_profile']['objects'],
                       'prediction_persist_seconds_per_object':r['evaluation_profile']['persist_seconds']/r['evaluation_profile']['objects'],
                       'peak_host_rss_bytes':max([r['peak_host_rss_bytes'],r['evaluation_profile']['peak_host_rss_bytes']]+[row['peak_host_rss_bytes'] for row in rows]),
                       'maximum_sampled_mps_driver_bytes':max(row.get('mps_driver_allocated_bytes',0) for row in rows),
                       'normalization_fallbacks':r['fallbacks'],'profile_manifest':file_record(root/directories[key]/'profile.json'),
                       'source':r['source']}
    parity=read_json(root/'data-parity.json');ridge=parity['ridge_profile']
    ridge_train=ridge['features_seconds']*64759/ridge['n']
    ridge_fits=ridge['three_fits_seconds']*64759/ridge['n']
    ridge_validation=ridge['features_seconds']*10000/ridge['n']
    ridge_final=ridge_validation
    estimates={}
    for updates in (10000,30000):
        validation_events=math.ceil(updates/500)
        transformer=3*updates*(measured['A']['median_update_seconds']+measured['B']['median_update_seconds'])
        cnn=3*updates*measured['CNN']['median_update_seconds']
        val=3*validation_events*10000*sum(measured[v]['evaluation_seconds_per_object'] for v in measured)
        checkpoint=3*(validation_events+2)*sum(measured[v]['checkpoint_write_seconds'] for v in measured)
        final=3*10000*sum(measured[v]['evaluation_seconds_per_object'] for v in measured)+ridge_final
        persisted=3*(validation_events+1)*10000*sum(measured[v]['prediction_persist_seconds_per_object'] for v in measured)
        parts={'six_transformer_training_seconds':transformer,'three_cnn_training_seconds':cnn,
               'ridge_train_features_and_three_fits_seconds':ridge_train+ridge_fits,'ridge_validation_seconds':ridge_validation,
               'neural_validations_seconds':val,'checkpoints_seconds':checkpoint,'final_evaluation_seconds':final,
               'persisted_predictions_seconds':persisted,'startup_hashes_and_final_analysis_allowance_seconds':300.}
        total=sum(parts.values())
        estimates[str(updates)]={'updates_per_neural_run':updates,'examples_per_run':updates*32,
                                'effective_train_epochs':updates*32/64759,'validation_events_per_run':validation_events,
                                'components':parts,'total_estimated_seconds':total,'total_estimated_hours':total/3600,
                                'planning_ceiling_with_50pct_margin_hours':total*1.5/3600}
    result={'scope':'4A measurement and prospective 4B cost; no final training/test results',
            'measured':measured,'estimates':estimates,'recommended_updates':10000,
            'recommendation':'Common 10000 updates for all nine runs, same validation cadence and all three seeds; fixed before 4B. This bounds cost, not a convergence guarantee.',
            'ridge_profile':ridge,'ridge_estimation':'Feature cost scales by rows; three-fit subset cost scaled by train rows conservatively includes fixed 256x256 solve cost.',
            'disk':{'estimated_retained_checkpoint_bytes':3*3*sum(m['checkpoint_bytes'] for m in measured.values()),
                    'suggested_additional_4b_reserve_bytes':12_000_000_000},
            'limits':['32 updates per variant; first 4 excluded from steady estimate, all measured values retained',
                      'MPS synchronization brackets forward/backward/optimizer/inference; allocation is sampled, not a true device peak',
                      'Warm local cache and short sessions; thermal throttling, other load and long-run memory behavior unmeasured',
                      'Validation/test costs extrapolated from 128 train objects only; no validation or test inference in 4A',
                      'Checkpoint cost uses final full optimizer state; last+previous+best retained',
                      'A/B were measured at 9d6adb6; CNN at 52e8d0d after equivalent pooling MPS fix. Transformer code/parameters unchanged',
                      'Initial CNN attempt failed before its first optimizer update; its incomplete destination is preserved']}
    write_json(output,result)
    print({k:{'hours':v['total_estimated_hours'],'ceiling_hours':v['planning_ceiling_with_50pct_margin_hours'],'components':v['components']} for k,v in estimates.items()})


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--profiles',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();budget(a.profiles,a.output)
