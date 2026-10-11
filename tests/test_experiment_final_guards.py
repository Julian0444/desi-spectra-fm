"""R4A-01: synthetic candidates must fail before any test access or output."""
from copy import deepcopy

import pytest
import torch

from desi_fm.experiment import final, training
from desi_fm.experiment.artifacts import file_record, read_json, write_json
from desi_fm.experiment.data import ExperimentData, TEST_PHRASE
from desi_fm.experiment.ridge import run_ridge
from test_experiment_training import experiment_root, small_config


@pytest.fixture
def collection(experiment_root, small_config, tmp_path):
    def build(*, mixed_device=False, mixed_environment=False):
        train = ExperimentData(experiment_root, 'train')
        validation = ExperimentData(experiment_root, 'validation')
        cfg = {**small_config, 'updates': 2, 'warmup_updates': 0,
               'validation_every': 1, 'checkpoint_every': 1}
        runs, ridge = tmp_path/'runs', tmp_path/'ridge'
        for variant in ('A', 'B', 'CNN'):
            for seed in (17, 42, 101):
                exceptional = variant == 'B' and seed == 17
                with pytest.MonkeyPatch.context() as patch:
                    if mixed_environment and exceptional:
                        different = deepcopy(training.environment())
                        different['packages']['numpy'] = 'incompatible'
                        patch.setattr(training, 'environment', lambda: different)
                    device = 'mps' if mixed_device and exceptional else 'cpu'
                    training.Trainer(train, cfg, variant, seed,
                                     runs/f'{variant}-{seed}', device).run(
                                         validation=validation, install_signals=False)
        run_ridge(train, validation, ridge)
        return runs, ridge, experiment_root, tmp_path/'frozen.json'
    return build


@pytest.mark.parametrize('kind', ['device', 'environment'])
def test_freeze_rejects_mixed_collection(collection, kind):
    if kind == 'device' and not torch.backends.mps.is_available():
        pytest.skip('Mixed CPU/MPS reproduction requires MPS')
    args = collection(mixed_device=kind == 'device', mixed_environment=kind == 'environment')
    with pytest.raises(ValueError, match='device/environment'):
        final.freeze(*args)
    assert not args[3].exists()
    assert not (args[2]/'test-opening.json').exists()


def test_freeze_rejects_ridge_environment(collection):
    args = collection()
    path = args[1]/'ridge.json'
    report = read_json(path)
    report['identity']['environment']['packages']['numpy'] = 'incompatible'
    write_json(path, report)
    with pytest.raises(ValueError, match='Ridge environment'):
        final.freeze(*args)
    assert not args[3].exists()


@pytest.mark.parametrize('fault', ['requested_device', 'candidate_device',
                                  'candidate_environment', 'current_environment',
                                  'ridge_environment', 'checkpoint_identity'])
def test_preflight_rejects_before_opening_or_predictions(collection, tmp_path, monkeypatch, fault):
    args = collection()
    spec = final.freeze(*args)
    device = 'cpu'
    if fault == 'requested_device':
        device = 'mps'
    elif fault == 'candidate_device':
        spec['neural'][3]['run_identity']['device'] = 'mps'
    elif fault == 'candidate_environment':
        spec['neural'][3]['run_identity']['environment']['python'] = 'incompatible'
    elif fault == 'current_environment':
        changed = deepcopy(training.environment())
        changed['packages']['torch'] = 'incompatible'
        monkeypatch.setattr(final, 'environment', lambda: changed, raising=False)
    elif fault == 'ridge_environment':
        path = args[1]/'ridge.json'
        report = read_json(path)
        report['identity']['environment']['packages']['numpy'] = 'incompatible'
        write_json(path, report)
        spec['ridge']['manifest_file'] = file_record(path)
    else:
        row = spec['neural'][3]
        state = torch.load(row['checkpoint'], map_location='cpu', weights_only=True)
        state['identity']['device'] = 'mps'
        torch.save(state, row['checkpoint'])
        row['file'] = file_record(row['checkpoint'])
    write_json(args[3], spec)
    def forbidden_test_access(*a, **kw):
        pytest.fail('Preflight accessed test before rejecting incompatible candidates')
    monkeypatch.setattr(final, 'ExperimentData', forbidden_test_access)
    output = tmp_path/'final-output'
    with pytest.raises(ValueError):
        final.evaluate_final(args[3], output, TEST_PHRASE, device=device)
    assert not output.exists()
    assert not (args[2]/'test-opening.json').exists()


def test_homogeneous_interrupted_evaluation_resumes_verified_predictions(collection, tmp_path, monkeypatch):
    args = collection()
    final.freeze(*args)
    output = tmp_path/'final-output'
    original = final.predict
    calls = []
    def interrupted(*a, **kw):
        calls.append(1)
        if len(calls) == 4:
            raise RuntimeError('synthetic interruption')
        return original(*a, **kw)
    monkeypatch.setattr(final, 'predict', interrupted)
    with pytest.raises(RuntimeError, match='synthetic interruption'):
        final.evaluate_final(args[3], output, TEST_PHRASE, device='cpu')
    preserved = {p.name: file_record(p) for p in output.glob('*.npz')}
    assert len(preserved) == 3
    registration = file_record(args[2]/'test-opening.json')
    monkeypatch.setattr(final, 'predict', original)
    result = final.evaluate_final(args[3], output, TEST_PHRASE, device='cpu', resume=True)
    assert result['status'] == 'complete' and len(result['files']) == 10
    assert all(file_record(output/name) == record for name, record in preserved.items())
    assert file_record(args[2]/'test-opening.json') == registration
