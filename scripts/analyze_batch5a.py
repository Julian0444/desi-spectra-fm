"""Post-hoc 5A checks of saved predictions and metadata; never runs a model.

Usage: python scripts/analyze_batch5a.py --runs ... --data ... --index ... --output NEW
Inspired by Claude's val_curves.py / 06_decision_robustness.py; recomputes
directly from saved NPZ, not the audit's derived matrices. Requires numpy only.
"""
import argparse
import hashlib
import json
import sqlite3
from pathlib import Path

import numpy as np


def analyze(runs, data, index, output):
    output.mkdir(parents=True, exist_ok=False)
    sources = {}

    def record(path):
        path = Path(path)
        sources[str(path)] = {"bytes": path.stat().st_size,
                              "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}

    def errors(path):
        record(path)
        with np.load(path, allow_pickle=False) as d:
            z = d['z_true'].astype(float)
            return np.abs(d['prediction'].astype(float) - z) / (1 + z)

    seeds = (17, 42, 101)
    steps = range(500, 10001, 500)
    scores = {}
    for method in ('A', 'B', 'CNN'):
        for seed in seeds:
            scores[f'{method}-{seed}'] = []
            for step in steps:
                e = errors(runs/f'{method}-{seed}'/f'validation-{step:06d}.npz')
                scores[f'{method}-{seed}'].append(dict(step=step, n=len(e),
                    outliers=int((e > .15).sum()), eta=float((e > .15).mean()), mae=float(e.mean())))
    deltas = [[100*(scores[f'A-{s}'][j]['eta']-scores[f'B-{s}'][j]['eta'])
               for s in seeds] for j in range(20)]
    ranking = {k: sorted(v, key=lambda r: (r['outliers'], r['mae'], r['step'])) for k,v in scores.items()}
    margins = {k: {'selected': v[0], 'runner_up': v[1],
                   'outlier_margin': v[1]['outliers']-v[0]['outliers']} for k,v in ranking.items()}
    top_k = {str(k): [100*(np.mean([r['eta'] for r in ranking[f'A-{s}'][:k]])-
                            np.mean([r['eta'] for r in ranking[f'B-{s}'][:k]])) for s in seeds]
             for k in (1, 2, 3, 5)}
    sensitivity_101 = []
    a_path = runs/'A-101'/'validation-008000.npz'
    with np.load(a_path, allow_pickle=False) as a_data:
        for step in (2500, 8000):
            b_path = runs/'B-101'/f'validation-{step:06d}.npz'
            with np.load(b_path, allow_pickle=False) as b_data:
                assert np.array_equal(a_data['ids'], b_data['ids'])
                assert np.array_equal(a_data['z_true'], b_data['z_true'])
                z = a_data['z_true'].astype(float)
                ae, be = errors(a_path), errors(b_path)
                for label, mask in [('all', np.ones(len(z), bool)),
                                    ('2<=z<3', (z>=2)&(z<3)), ('z>=3', z>=3)]:
                    sensitivity_101.append(dict(b_step=step, stratum=label, n=int(mask.sum()),
                        a_outliers=int((ae[mask]>.15).sum()), b_outliers=int((be[mask]>.15).sum()),
                        delta_eta_pp=float(100*((ae[mask]>.15).mean()-(be[mask]>.15).mean())),
                        delta_mae=float((ae[mask]-be[mask]).mean())))
    test = {}
    rng = np.random.default_rng(20261016)
    for s in seeds:
        a, b = (errors(runs/'final-test'/f'{v}-{s}.npz') for v in ('A','B'))
        delta = a-b
        # Conditional paired object bootstrap; no new inference or selection.
        draws = [float(delta[rng.integers(len(delta), size=len(delta))].mean()) for _ in range(5000)]
        test[str(s)] = dict(delta_eta_pp=float(100*((a>.15).mean()-(b>.15).mean())),
                           delta_mae=float(delta.mean()), delta_mae_ci=np.quantile(draws,[.025,.975]).tolist())
    splits = {}
    for name in ('train','validation','test'):
        path = data/name/'ids.json'; record(path)
        d = json.loads(path.read_text()); splits[name] = d['ids'] if isinstance(d,dict) else d
    record(index)
    with sqlite3.connect(f'file:{index}?mode=ro', uri=True) as conn:
        ordered = [r[0] for r in conn.execute('SELECT object_id FROM objects WHERE label_eligible=1 AND input_eligible=1 ORDER BY ordinal')]
    tail = set(ordered[82000:])

    def bricks(ids):
        # Only imaging DR9 IDs with no mock/sky/Gaia/reserved bits.
        values = np.array([int(x) for x in ids], dtype=np.uint64)
        release = (values >> np.uint64(42)) & np.uint64(65535)
        brick = (values >> np.uint64(22)) & np.uint64(1048575)
        valid = np.isin(release,[9010,9011]) & ((values >> np.uint64(58)) == 0) & (brick >= 1) & (brick <= 662174)
        return brick[valid]

    train_bricks, test_bricks, all_bricks = (bricks(ids) for ids in (splits['train'],splits['test'],ordered))
    result = dict(scope='post-hoc exploratory; no replacement of prespecified criterion',
                  validation_scores=scores, delta_eta_by_step_pp=deltas, selection_margins=margins,
                  mean_top_k_delta_eta_pp=top_k, test_conditional_mae=test,
                  validation_101_sensitivity=sensitivity_101,
                  negative_steps_per_seed=(np.array(deltas)<0).sum(axis=0).tolist(),
                  all_negative_steps=int((np.array(deltas)<0).all(axis=1).sum()),
                  final_validation_mean_delta_mae=float(np.mean([scores[f'A-{s}'][-1]['mae']-scores[f'B-{s}'][-1]['mae'] for s in seeds])),
                  tail={'n':len(tail), **{k:len(tail.intersection(v)) for k,v in splits.items()}},
                  locality={'eligible_dr9':len(all_bricks),'eligible_n':len(ordered),'eligible_bricks':len(np.unique(all_bricks)),
                            'test_dr9':len(test_bricks),'test_shared_train_brick':int(np.isin(test_bricks,train_bricks).sum())},
                  sources=sources)
    (output/'analysis.json').write_text(json.dumps(result,indent=2)+'\n')
    lines=['# 5A: análisis posteriores de predicciones guardadas','',
           'No cambian la selección, el criterio predefinido ni los artefactos del 4B. Sin inferencia nueva.','',
           '| Paso de validación | A−B 17 (pp) | A−B 42 (pp) | A−B 101 (pp) |',
           '|---:|---:|---:|---:|']
    for step, ds in zip(steps,deltas): lines.append(f'| {step} | '+ ' | '.join(f'{d:+.2f}' for d in ds)+' |')
    lines += ['', f'Las tres diferencias son negativas en {result["all_negative_steps"]}/20 pasos; por semilla: {result["negative_steps_per_seed"]}.',
              f'ΔMAE medio en validación al paso final: {result["final_validation_mean_delta_mae"]:+.6f}.', '',
              '| Candidato | Paso seleccionado | Segundo paso | Margen (objetos/10.000) |','|---|---:|---:|---:|']
    for k,m in margins.items(): lines.append(f'| {k} | {m["selected"]["step"]} | {m["runner_up"]["step"]} | {m["outlier_margin"]} |')
    lines += ['', 'Promedio de los k mejores scores de cada variante (no ensemble ni nueva regla):','',
              '| k | A−B 17 (pp) | A−B 42 (pp) | A−B 101 (pp) |','|---:|---:|---:|---:|']
    for k,ds in top_k.items(): lines.append(f'| {k} | '+' | '.join(f'{d:+.2f}' for d in ds)+' |')
    lines += ['', 'Sensibilidad H-10 en **validación guardada**, A-101 paso 8.000 frente a B-101 pasos 2.500/8.000. No evalúa un checkpoint alternativo en test ni autoriza sustituir el seleccionado.', '',
              '| Paso B | Estrato | N | Outliers A/B | Δη (pp) | ΔMAE |', '|---:|---|---:|---|---:|---:|']
    for r in sensitivity_101:
        lines.append(f'| {r["b_step"]} | {r["stratum"]} | {r["n"]} | {r["a_outliers"]}/{r["b_outliers"]} | {r["delta_eta_pp"]:+.2f} | {r["delta_mae"]:+.6f} |')
    lines += ['', 'ΔMAE de test desde predicciones guardadas; IC95 bootstrap pareado por objeto, 5.000 réplicas, semilla 20261016, condicionado a cada par seleccionado. Independencia física no establecida.','',
              '| Semilla | ΔMAE | IC95 |','|---:|---:|---|']
    for s,t in test.items(): lines.append(f'| {s} | {t["delta_mae"]:+.6f} | [{t["delta_mae_ci"][0]:+.6f}, {t["delta_mae_ci"][1]:+.6f}] |')
    lines += ['', 'Identidades de entradas, tabla completa y conteos de localidad/cola: [analysis.json](batch5a-analysis.json).','']
    (output/'analysis.md').write_text('\n'.join(lines))
    print(json.dumps({k:result[k] for k in ('all_negative_steps','negative_steps_per_seed','tail','locality')},indent=2))


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('runs','data','index','output'): p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args(); analyze(a.runs,a.data,a.index,a.output)
