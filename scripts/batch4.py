"""Explicit entry points for batch 4. Profiles can only use train."""
from __future__ import annotations

import argparse
from pathlib import Path
import socket
import time
import numpy as np

from desi_fm.experiment.artifacts import read_json, write_json
from desi_fm.experiment.data import ExperimentData
from desi_fm.experiment.training import Trainer, predict, synchronize, memory
from desi_fm.experiment.metrics import save_predictions, paired_analysis


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command',required=True)
    for name in ('profile','train'):
        p = sub.add_parser(name)
        p.add_argument('--data',required=True); p.add_argument('--config',required=True)
        p.add_argument('--variant',required=True,choices=['A','B','CNN']); p.add_argument('--seed',required=True,type=int,choices=[17,42,101])
        p.add_argument('--output',required=True); p.add_argument('--device',default='mps',choices=['mps','cpu'])
        p.add_argument('--resume',action='store_true')
        if name=='profile':
            p.add_argument('--updates',type=int,default=32)
            p.add_argument('--evaluation-objects',type=int,default=128)
        else:
            p.add_argument('--authorize-series',required=True,choices=['BATCH4B'])
    p = sub.add_parser('ridge')
    p.add_argument('--data',required=True); p.add_argument('--output',required=True)
    p.add_argument('--authorize-series',required=True,choices=['BATCH4B'])
    p = sub.add_parser('freeze')
    p.add_argument('--data',required=True); p.add_argument('--runs',required=True)
    p.add_argument('--ridge',required=True); p.add_argument('--output',required=True)
    p = sub.add_parser('final-test')
    p.add_argument('--frozen',required=True); p.add_argument('--output',required=True)
    p.add_argument('--open-test',required=True); p.add_argument('--resume',action='store_true')
    p.add_argument('--device',default='mps',choices=['mps','cpu'])
    p = sub.add_parser('analyze')
    p.add_argument('--predictions',required=True); p.add_argument('--output',required=True)
    args = parser.parse_args()
    # The experiment uses no network, not even an accidental library fallback.
    def offline(*a,**k):
        raise RuntimeError('Network is disabled for the local batch-4 experiment')
    socket.socket.connect = offline
    socket.create_connection = offline
    if args.command in ('profile','train'):
        profile = args.command=='profile'
        if profile and not 1<=args.updates<=200:
            parser.error('Profile budget must be between 1 and 200 updates')
        config = read_json(args.config)
        train = ExperimentData(args.data,'train')
        val = None if profile else ExperimentData(args.data,'validation')
        trainer = Trainer(train,config,args.variant,args.seed,args.output,args.device,profile=profile,resume=args.resume)
        result = trainer.run(validation=val,until=args.updates if profile else None)
        if result['status'] != 'complete':
            print({'status':result['status'],'step':result['step'],'resume_with':'same arguments plus --resume'},flush=True)
            raise SystemExit(130)
        if profile:
            # Never use validation or test to time inference in 4A.
            indices = np.random.default_rng(20261009).choice(len(train),min(args.evaluation_objects,len(train)),replace=False)
            synchronize(trainer.device); start=time.perf_counter()
            outputs = predict(trainer.model,train,indices,config['micro_batch'],trainer.device)
            synchronize(trainer.device); evaluate_seconds=time.perf_counter()-start
            start=time.perf_counter()
            save_predictions(Path(args.output)/'profile-train.npz',train,indices,outputs,
                             {'seed':args.seed,'variant':args.variant,'step':trainer.step,'scope':'cost only, not scientific result'})
            result['evaluation_profile']={'objects':len(indices),'seconds':evaluate_seconds,'persist_seconds':time.perf_counter()-start,**memory(trainer.device)}
            # Exercise disk recovery without another update.
            del trainer
            recovered = Trainer(train,config,args.variant,args.seed,args.output,args.device,profile=True,resume=True)
            result['profile_recovery_seconds']=recovered.recovery_seconds
            result['profile_class_weights_restored']=True
            write_json(Path(args.output)/'profile.json',result)
        print({'status':result['status'],'step':result['step'],'output':args.output})
    elif args.command=='ridge':
        from desi_fm.experiment.ridge import run_ridge
        run_ridge(ExperimentData(args.data,'train'),ExperimentData(args.data,'validation'),args.output)
    elif args.command=='freeze':
        from desi_fm.experiment.final import freeze
        freeze(args.runs,args.ridge,args.data,args.output)
    elif args.command=='final-test':
        from desi_fm.experiment.final import evaluate_final
        evaluate_final(args.frozen,args.output,args.open_test,resume=args.resume,device=args.device)
    else:
        directory=Path(args.predictions)
        output=Path(args.output)
        if output.exists():
            raise ValueError('Use a new analysis output')
        result=paired_analysis([directory/f'A-{s}.npz' for s in (17,42,101)],[directory/f'B-{s}.npz' for s in (17,42,101)])
        write_json(output,result)


if __name__=='__main__':
    main()
