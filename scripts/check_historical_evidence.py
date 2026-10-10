"""Read-only checks of stored batch-1 evidence; no inference, network or LLM.

python scripts/check_historical_evidence.py --copilot-root /path/to/spectra-copilot
Requires local checkpoints for the optional --check-weights comparison. Prints
JSON to stdout; redirect only to a NEW evidence file, never historical results.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    with path.open(newline='') as f:
        return list(csv.DictReader(f))


def col(rows, name):
    return np.array([float(r[name]) for r in rows])


def error(rows, name):
    z = col(rows, 'z_true')
    return np.abs(col(rows, name) - z) / (1 + z)


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--copilot-root', type=Path, required=True)
    ap.add_argument('--check-weights', action='store_true')
    args = ap.parse_args()
    fm = ROOT / 'runs/desi_80k_classhead_v21'
    v1 = read(ROOT / 'runs/calibration/predictions_v1_heldout_canonical.csv')
    last, best = read(fm / 'predictions_last.csv'), read(fm / 'predictions_best.csv')
    baseline = read(args.copilot_root / 'eval/results_baseline.csv')
    agent = read(args.copilot_root / 'eval/results.csv')
    labels = read(args.copilot_root / 'eval/cases/labels.csv')
    assert len(v1) == len(last) == len(best) == 2000
    assert np.array_equal(col(v1, 'z_true'), col(last, 'z_true'))
    assert [r['case'] for r in baseline] == [r['case'] for r in agent] == [r['case'] for r in labels]
    assert np.array_equal(col(baseline, 'z_true'), col(agent, 'z_true'))
    e1, e2 = error(v1, 'z_pred'), error(last, 'z_pred_map')
    eb, ea = error(baseline, 'z_model'), error(agent, 'z_final')
    good = col(last, 'z_confidence') >= .5
    outlier = e2 > .15
    confidence = col(last, 'z_confidence')
    # Pairwise AUROC: larger concentration should predict non-outliers.
    positive, negative = confidence[~outlier], confidence[outlier]
    auc = ((positive[:, None] > negative).sum() + .5 * (positive[:, None] == negative).sum()) / (positive.size * negative.size)
    idx = np.array([int(r['heldout_index']) for r in labels])
    assert np.array_equal(col(labels, 'z_true'), col(last, 'z_true')[idx])
    bins = [.5, 1, 1.5, 2, 2.5]
    pop_bin = np.digitize(col(last, 'z_true'), bins)
    sample_bin = np.digitize(col(baseline, 'z_true'), bins)
    weights = np.bincount(pop_bin, minlength=6) / len(last)
    direct = sum(weights[h] * (eb[sample_bin == h] > .15).mean() for h in range(6))
    difference = outlier.mean() + sum(weights[h] * (
        (eb[sample_bin == h] > .15).astype(float) - outlier[idx][sample_bin == h]).mean() for h in range(6))
    rec = np.load(fm / 'reconstructions.npz', allow_pickle=False)
    assert sha(fm / 'predictions.csv') == sha(fm / 'predictions_last.csv')
    assert sha(fm / 'reconstructions.npz') == sha(fm / 'reconstructions_best.npz')
    assert np.allclose(rec['z_pred'], col(best, 'z_pred')[:50], rtol=0, atol=1e-7)
    flux, valid = rec['flux'], rec['valid']
    hidden = rec['masked_pixels'].astype(bool)
    visible = valid * ~hidden
    target_weight = valid * hidden
    mean = (flux * visible).sum(1) / np.maximum(visible.sum(1), 1)
    smooth = np.zeros_like(flux)
    for i in range(len(flux)):
        num = np.convolve(flux[i] * visible[i], np.ones(301), mode='same')
        den = np.convolve(visible[i], np.ones(301), mode='same')
        smooth[i] = np.where(den > 0, num / np.maximum(den, 1e-6), mean[i])
    def rmse(pred):
        return float(np.sqrt((((pred - flux)**2) * target_weight).sum() / target_weight.sum()))
    metrics = {
        'scope': 'recalculation of historical artifacts; no new inference or independent test',
        'masked_2000': {'v1_outliers': int((e1 > .15).sum()), 'v21_last_outliers': int(outlier.sum()),
                        'v21_best_outliers': int((error(best, 'z_pred_map') > .15).sum()),
                        'retained_at_confidence_05': int(good.sum()),
                        'retained_outliers': int(outlier[good].sum()), 'confidence_auroc': float(auc)},
        'full_spectrum_150': {'model_correct': int((eb < .15).sum()), 'agent_correct': int((ea < .15).sum()),
                             'broken': int(((eb < .15) & (ea >= .15)).sum()),
                             'recovered': int(((eb >= .15) & (ea < .15)).sum()),
                             'hybrid_correct': int(sum(ea[i] < .15 if r['confidence'] == 'high' else eb[i] < .15 for i,r in enumerate(agent))),
                             'tokens_successful_reports': sum(int(r['tokens']) for r in agent),
                             'stratum_counts': np.bincount(sample_bin, minlength=6).tolist()},
        'estimates_not_measurements_of_2000': {'direct_reweighting_error_rate': float(direct),
                                             'paired_difference_error_rate': float(difference)},
        'reconstruction_50_best_mask': {'model_rmse': rmse(rec['reconstruction']),
                                      'visible_only_301px_rmse': rmse(smooth),
                                      'width_was_explored_on_same_examples': True},
    }
    assert metrics['masked_2000']['v1_outliers'] == 452
    assert metrics['masked_2000']['v21_last_outliers'] == 299
    assert (good.sum(), outlier[good].sum()) == (1175, 9)
    assert ((eb < .15).sum(), (ea < .15).sum()) == (139, 119)
    if args.check_weights:
        import torch
        a = torch.load(fm / 'checkpoint_best.pt', map_location='cpu', weights_only=True)
        b = torch.load(fm / 'checkpoint_last.pt', map_location='cpu', weights_only=True)
        assert a['model'].keys() == b['model'].keys()
        assert all(torch.equal(a['model'][k], b['model'][k]) for k in a['model'])
        assert (b['step'], b['epoch']) == (30000, 2)
        metrics['weights'] = {'step': b['step'], 'epoch': b['epoch'], 'equal_tensors': len(b['model']),
                             'last_sha256': sha(fm / 'checkpoint_last.pt')}
    counts = {}
    for root in (ROOT, args.copilot_root):
        manifest = json.loads((root / 'docs/evidence/historical-assets.json').read_text())
        for r in manifest['files']:
            p = root / r['path']
            assert p.stat().st_size == r['bytes'] and sha(p) == r['sha256'], f'Changed historical artifact: {p}'
        counts[root.name] = len(manifest['files'])
    metrics['unchanged_artifacts'] = counts
    print(json.dumps(metrics, indent=2))


if __name__ == '__main__':
    main()
