"""Frozen historical evaluation from verified local IDs; exclusive destinations."""
import argparse
import csv
from dataclasses import asdict
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import resource
import socket
import subprocess
import sys
import time
import numpy as np
import torch
from desi_fm.assets import sha256, verify_file
from desi_fm.data import SpectrumPreprocessConfig, extract_mmu_desi_example, preprocess_spectrum
from desi_fm.historical import fixed_masks, visible_baselines, masked_rmse, write_csv, summarize
from desi_fm.inputs import SpectrumInputError
from desi_fm.local_data import LocalDESIStore
from desi_fm.predict import load_model_from_checkpoint, model_identity

ROOT=Path(__file__).resolve().parents[1]


def json_write(path,data):
    temporary=path.with_suffix('.writing')
    temporary.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n');temporary.replace(path)


def synchronize(device):
    if device.type=='mps':torch.mps.synchronize()
    elif device.type=='cuda':torch.cuda.synchronize()


def run(args):
    start=time.perf_counter();timings={k:0. for k in ('verification','load_weights','reading','preprocessing','forward_full','forward_reconstruction','baselines','writing','summary_figures')}
    network_attempts=[]
    if args.offline:
        os.environ['HF_HUB_OFFLINE']='1';os.environ['HF_DATASETS_OFFLINE']='1'
        def deny(*a,**k):
            network_attempts.append('blocked');raise RuntimeError('Network disabled for offline repeat')
        socket.socket.connect=deny;socket.socket.connect_ex=deny;socket.create_connection=deny
    torch.set_num_threads(4);torch.manual_seed(17)
    output=Path(args.output);output.mkdir(parents=True,exist_ok=False)
    json_write(output/'manifest.json',{'status':'incomplete','command':sys.argv})
    selection=json.loads(Path(args.ids).read_text());requested=selection['ids'][:args.limit] if args.limit else selection['ids']
    if not requested or len(requested)!=len(set(requested)):raise ValueError('Empty or duplicate selection')
    tracked=list((ROOT/'src/desi_fm').glob('*.py'))+[Path(__file__),ROOT/'scripts/recover_historical_ids.py']
    manifest={'status':'incomplete','scope':'bounded profile' if args.limit else 'historical diagnostic, not an independent test',
              'source_head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
              'code_sha256':{str(p.relative_to(ROOT)):sha256(p) for p in tracked},'command':sys.argv,
              'selection_sha256':sha256(args.ids),'requested_ids':requested,'full_historical_selection_n':selection['requested'],
              'input_contract':'spectra-input-v2','mask_seed':20261009,'reconstruction_normalization':'visible-only; fixed before evaluation',
              'tolerances':{'arrays_masks_ids':'exact','prediction_atol':1e-6,'prediction_rtol':1e-6,'metric_atol':1e-8},
              'device':args.device,'batch_size':args.batch_size,'num_workers':0,'torch_cpu_threads':4,'offline_socket_block':args.offline,
              'environment':{name:importlib.metadata.version(name) for name in ('numpy','torch','pyarrow','datasets','matplotlib','huggingface_hub')},
              'python':sys.version,'hardware':{'machine':platform.machine(),'macos':platform.mac_ver()[0],
                         'chip':subprocess.check_output(['sysctl','-n','machdep.cpu.brand_string'],text=True).strip(),
                         'ram_bytes':int(subprocess.check_output(['sysctl','-n','hw.memsize'],text=True))},
              'historical_identity_limit':'Recipe reconstruction from fixed snapshot and audit metadata; 150 direct exported ID/raw anchors; original full CSV lacks IDs'}
    json_write(output/'manifest.json',manifest)
    try:
        t=time.perf_counter();store=LocalDESIStore(args.root,args.index)
        if selection['snapshot_sha256']!=sha256(Path(args.root)/'snapshot.json') or selection['index_sha256']!=sha256(args.index):raise ValueError('Selection dataset identity mismatch')
        locations=store.rows_for_ids(requested);timings['verification']=time.perf_counter()-t
        manifest['dataset']={'root':str(Path(args.root).resolve()),'snapshot_sha256':selection['snapshot_sha256'],'revision':store.manifest['revision'],
                             'index':str(Path(args.index).resolve()),'index_sha256':sha256(args.index)}
        lock=json.loads(Path(args.checkpoints).read_text());device=torch.device(args.device)
        models={};t=time.perf_counter()
        for name in ('v1','v21'):
            row=lock[name];verify_file(row['path'],row['bytes'],row['sha256'])
            models[name]=load_model_from_checkpoint(row['path'],device)
        synchronize(device);timings['load_weights']=time.perf_counter()-t
        manifest['checkpoint_lock_sha256']=sha256(args.checkpoints)
        manifest['checkpoints']={name:{'path':lock[name]['path'],'selection':lock[name]['selection'],**model_identity(model)} for name,model in models.items()}
        config=models['v21'].config
        for field in ('n_pixels','n_tokens','lambda_min','lambda_max','wavelength_grid'):
            if getattr(models['v1'].config,field)!=getattr(config,field):raise ValueError('Models cannot share this input/mask grid')
        pre=SpectrumPreprocessConfig(n_pixels=config.n_pixels,lambda_min=config.lambda_min,lambda_max=config.lambda_max,wavelength_grid=config.wavelength_grid)
        manifest['preprocess']=asdict(pre);manifest['baseline_parameters']={'mean':'global visible mean','linear':'visible model-grid pixel coordinates, nearest endpoint extrapolation','moving301':{'width':301,'fallback':'global visible mean'}}
        all_masks=fixed_masks(requested,config.n_tokens);full=[];rec=[];exclusions={'redshift':[],'reconstruction':[]}
        for i,location in enumerate(locations):
            oid=location['object_id'];t=time.perf_counter();example=store.raw(oid);arrays=extract_mmu_desi_example(example);timings['reading']+=time.perf_counter()-t
            f,v,w,m,z=arrays;metadata={'object_id':oid,'z_true':z,'zwarn':location['zwarn'],'requested_position':i}
            if not location['label_eligible']:
                for stage in exclusions:exclusions[stage].append({'object_id':oid,'cause':'invalid label'})
                continue
            hidden=np.repeat(all_masks[i],config.patch_size)[:config.n_pixels]
            for stage,normalization_mask,target in [('redshift',None,full),('reconstruction',hidden,rec)]:
                t=time.perf_counter()
                try:
                    processed=preprocess_spectrum(f,v,w,m,pre,normalization_mask=normalization_mask)
                    target.append({**metadata,**processed,'token_mask':all_masks[i],'hidden':hidden})
                except SpectrumInputError as exc:exclusions[stage].append({'object_id':oid,'cause':str(exc)})
                timings['preprocessing']+=time.perf_counter()-t
        json_write(output/'exclusions.json',exclusions)
        if not full or not rec:raise ValueError('No eligible objects; exclusions preserved, run incomplete')
        flux=np.stack([x['flux'] for x in full]);valid=np.stack([x['valid'] for x in full]);target=np.stack([x['flux'] for x in rec]);recvalid=np.stack([x['valid'] for x in rec])
        masks=np.stack([x['token_mask'] for x in rec]);hidden=np.stack([x['hidden'] for x in rec]);full_ids=np.array([x['object_id'] for x in full]);rec_ids=np.array([x['object_id'] for x in rec])
        z=np.array([x['z_true'] for x in full],dtype=np.float64);quality=np.array([x['zwarn'] for x in full]);coverage=np.array([x['coverage']['model_valid_fraction'] for x in full])
        t=time.perf_counter()
        np.savez_compressed(output/'inputs.npz',ids=full_ids,flux=flux,valid=valid,z_true=z,zwarn=quality,coverage=coverage,
                            centers=np.array([x['center'] for x in full]),scales=np.array([x['scale'] for x in full]),
                            reconstruction_ids=rec_ids,reconstruction_flux=target,reconstruction_valid=recvalid,
                            reconstruction_centers=np.array([x['center'] for x in rec]),reconstruction_scales=np.array([x['scale'] for x in rec]),
                            requested_ids=np.array(requested),requested_token_masks=all_masks)
        timings['writing']+=time.perf_counter()-t
        predictions={};reconstructions={}
        for name,model in models.items():
            outputs={};t=time.perf_counter()
            with torch.inference_mode():
                for start_batch in range(0,len(full),args.batch_size):
                    if start_batch and start_batch % 400 == 0:print(json.dumps({'model':name,'stage':'full','processed':start_batch,'elapsed_s':time.perf_counter()-t}),flush=True)
                    end=start_batch+args.batch_size
                    batchflux=torch.from_numpy(flux[start_batch:end]).to(device);batchvalid=torch.from_numpy(valid[start_batch:end]).to(device)
                    zero=torch.zeros((len(batchflux),config.n_tokens),dtype=torch.bool,device=device)
                    out=model(batchflux,batchvalid,spectrum_mask=zero)
                    for key in ('z_pred','z_pred_map','z_confidence'):
                        if key in out:outputs.setdefault(key,[]).append(out[key].cpu().numpy())
            synchronize(device);timings['forward_full']+=time.perf_counter()-t
            predictions[name]={key:np.concatenate(values) for key,values in outputs.items()}
            reconstructed=[];t=time.perf_counter()
            with torch.inference_mode():
                for start_batch in range(0,len(rec),args.batch_size):
                    if start_batch and start_batch % 400 == 0:print(json.dumps({'model':name,'stage':'reconstruction','processed':start_batch,'elapsed_s':time.perf_counter()-t}),flush=True)
                    end=start_batch+args.batch_size
                    out=model(torch.from_numpy(target[start_batch:end]).to(device),torch.from_numpy(recvalid[start_batch:end]).to(device),
                              spectrum_mask=torch.from_numpy(masks[start_batch:end]).to(device))
                    reconstructed.append(out['patch_pred'].flatten(1)[:,:config.n_pixels].cpu().numpy())
            synchronize(device);timings['forward_reconstruction']+=time.perf_counter()-t
            reconstructions[name]=np.concatenate(reconstructed)
            print(json.dumps({'model':name,'full_n':len(full),'reconstruction_n':len(rec),'timings_s':timings}),flush=True)
        t=time.perf_counter();baselines={key:[] for key in ('mean','linear','moving301')}
        for i in range(len(rec)):
            values=visible_baselines(target[i],recvalid[i],hidden[i])
            for name in baselines:baselines[name].append(values[name])
        baselines={key:np.stack(values) for key,values in baselines.items()};timings['baselines']=time.perf_counter()-t
        saved={'v1_scalar':predictions['v1']['z_pred'],'v21_map':predictions['v21']['z_pred_map'],
               'v21_secondary':predictions['v21']['z_pred'],'v21_confidence':predictions['v21']['z_confidence']}
        rows=[{'object_id':oid,'z_true':float(z[i]),'zwarn':int(quality[i]),'coverage':float(coverage[i]),**{k:float(v[i]) for k,v in saved.items()}} for i,oid in enumerate(full_ids)]
        recrows=[]
        for i,oid in enumerate(rec_ids):
            row={'object_id':oid}
            for name,pred in {**reconstructions,**baselines}.items():
                value=masked_rmse(target[i],pred[i],recvalid[i].astype(bool)&hidden[i]);row.update({f'{name}_{k}':v for k,v in value.items()})
            recrows.append(row)
        t=time.perf_counter();write_csv(output/'predictions.csv',rows);write_csv(output/'reconstruction.csv',recrows)
        np.savez_compressed(output/'predictions.npz',ids=full_ids,z_true=z,zwarn=quality,coverage=coverage,**saved)
        np.savez_compressed(output/'reconstruction.npz',ids=rec_ids,target=target,valid=recvalid,hidden=hidden,token_masks=masks,**reconstructions,**baselines)
        timings['writing']+=time.perf_counter()-t
        t=time.perf_counter();summary=summarize(output);timings['summary_figures']=time.perf_counter()-t
        manifest.update(status='complete',accounting=summary['accounting'],timings_s=timings,total_s=time.perf_counter()-start,
                        peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                        mps_driver_allocated_bytes=torch.mps.driver_allocated_memory() if device.type=='mps' else None,
                        blocked_network_attempts=len(network_attempts),
                        artifacts={p.name:{'bytes':p.stat().st_size,'sha256':sha256(p)} for p in output.iterdir() if p.is_file() and p.name!='manifest.json'})
        json_write(output/'manifest.json',manifest)
        print(json.dumps({'output':str(output),'total_s':manifest['total_s'],'accounting':summary['accounting'],'redshift':summary['redshift'],'reconstruction':summary['reconstruction'],'timings_s':timings},indent=2))
    except BaseException as exc:
        manifest.update(status='incomplete',failure=f'{type(exc).__name__}: {exc}',timings_s=timings,total_s=time.perf_counter()-start)
        json_write(output/'manifest.json',manifest)
        raise

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('root','index','ids','output'):p.add_argument('--'+name,required=True)
    p.add_argument('--checkpoints',default=str(ROOT/'docs/evidence/batch3-checkpoints.json'))
    p.add_argument('--device',choices=['cpu','mps','cuda'],default='cpu');p.add_argument('--batch-size',type=int,default=8)
    p.add_argument('--limit',type=int);p.add_argument('--offline',action='store_true')
    a=p.parse_args()
    if a.batch_size<1 or (a.limit is not None and a.limit<1):p.error('positive batch-size/limit required')
    run(a)
