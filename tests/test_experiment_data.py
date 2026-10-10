import numpy as np
import pytest

from desi_fm.data import SpectrumPreprocessConfig, normalize_grid, preprocess_spectrum
from desi_fm.experiment.data import PairedSampler, sample_masks
from desi_fm.experiment.splits import make_splits


def metadata(n=120):
    return [{'object_id': str(2**60+i), 'z': (i % 12)/2, 'zwarn': i % 2} for i in range(n)]


def test_splits_cover_disjoint_reproducible_and_protected():
    rows = metadata()
    protected = [r['object_id'] for r in rows[:12]]
    first, report = make_splits(rows, protected, validation_size=20, test_size=20)
    second, again = make_splits(rows[::-1], protected, validation_size=20, test_size=20)
    assert first == second and report == again
    ids = {k: {r['object_id'] for r in v} for k, v in first.items()}
    assert {k: len(v) for k, v in ids.items()} == {'train': 80, 'validation': 20, 'test': 20}
    assert not ids['test'] & set(protected)
    assert not ids['train'] & ids['validation'] and not ids['train'] & ids['test'] and not ids['validation'] & ids['test']
    assert set.union(*ids.values()) == {r['object_id'] for r in rows}
    assert any(r['zwarn'] == 0 for r in first['train'])


def test_groups_mixed_strata_protection_rare_and_ineligible_rejected():
    rows = metadata(35)
    relations = {r['object_id']: 'observed-source-a' for r in rows[:7]}
    splits, report = make_splits(rows, [rows[0]['object_id']], relationships=relations, validation_size=7, test_size=7)
    locations = [role for role, rs in splits.items() if any(r['object_id'] in relations for r in rs)]
    assert len(locations) == 1 and locations[0] != 'test'
    assert report['mixed_stratum_groups'] and report['rare_strata']
    assert sum(report['actual'].values()) == len(rows)
    with pytest.raises(ValueError, match='Protected'):
        make_splits(rows, ['missing'], validation_size=7, test_size=7)
    with pytest.raises(ValueError, match='Ineligible'):
        make_splits([{**r, 'z': -1} for r in rows], [], validation_size=7, test_size=7)


def test_masks_and_sampling_independent_of_global_rng_and_resume():
    a, b = PairedSampler(13, 17), PairedSampler(13, 17)
    ai, ap = a.take(40)
    np.random.seed(999)
    bi, bp = b.take(23)
    resumed = PairedSampler(13, 17, b.position)
    ci, cp = resumed.take(17)
    np.testing.assert_array_equal(ai, np.r_[bi, ci])
    np.testing.assert_array_equal(ap, np.r_[bp, cp])
    assert sorted(ai[:13]) == list(range(13))
    masks = sample_masks(list(map(str, ai)), ap, 17, 273)
    np.random.random(10000)
    np.testing.assert_array_equal(masks, sample_masks(list(map(str, ai)), ap, 17, 273))
    assert .45 < masks.mean() < .55
    assert not np.array_equal(masks, sample_masks(list(map(str, ai)), ap, 42, 273))


def test_hidden_values_cannot_change_statistics_visible_inputs_or_fallback():
    cfg = SpectrumPreprocessConfig(n_pixels=64)
    flux = np.arange(64, dtype=np.float32)
    valid = np.ones(64, dtype=np.float32)
    hidden = np.arange(64) >= 32
    a = normalize_grid(flux, valid, cfg, hidden, insufficient='identity')
    altered = flux.copy(); altered[hidden] = 1e9
    b = normalize_grid(altered, valid, cfg, hidden, insufficient='identity')
    assert a['center'] == b['center'] and a['scale'] == b['scale']
    np.testing.assert_array_equal(a['flux'][~hidden], b['flux'][~hidden])
    hidden[:] = True; hidden[:3] = False
    a = normalize_grid(flux, valid, cfg, hidden, insufficient='identity')
    b = normalize_grid(altered, valid, cfg, hidden, insufficient='identity')
    assert a['normalization_fallback'] and a['center'] == b['center'] == 0 and a['scale'] == b['scale'] == 1
    with pytest.raises(ValueError, match='visible'):
        normalize_grid(flux, valid, cfg, hidden)


def test_grid_contract_constant_invalid_edges_and_gaps():
    from desi_fm.data import resample_spectrum
    cfg = SpectrumPreprocessConfig(n_pixels=64)
    wave = cfg.target_wavelength
    flux = np.ones(64, dtype=np.float32)
    mask = np.zeros(64, dtype=bool); mask[20:30] = True
    grid = resample_spectrum(flux, np.ones(64), wave, mask, cfg)
    assert not grid['valid'][20:30].any()
    normalized = normalize_grid(grid['flux'], grid['valid'], cfg)
    original = preprocess_spectrum(flux, np.ones(64), wave, mask, cfg)
    np.testing.assert_array_equal(normalized['flux'], original['flux'])
    assert normalized['scale'] == np.float32(1e-6)
