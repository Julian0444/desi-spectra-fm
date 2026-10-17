"""5A isolated guards, adapted from inspected Claude probes; synthetic only."""
from copy import deepcopy

import pytest
import torch

from desi_fm.experiment import final, training
from desi_fm.experiment import ridge as ridge_module
from desi_fm.experiment.artifacts import file_record, read_json, write_json
from desi_fm.experiment.data import ExperimentData, TEST_PHRASE
from test_experiment_training import experiment_root, small_config  # fixtures del repo


def build(experiment_root, small_config, tmp_path, env=None):
    train = ExperimentData(experiment_root, 'train')
    validation = ExperimentData(experiment_root, 'validation')
    cfg = {**small_config, 'updates': 2, 'warmup_updates': 0, 'validation_every': 1, 'checkpoint_every': 1}
    runs, ridge = tmp_path/'runs', tmp_path/'ridge'
    with pytest.MonkeyPatch.context() as patch:
        if env is not None:
            patch.setattr(training, 'environment', lambda: env)
            patch.setattr(ridge_module, 'environment', lambda: env)
        for variant in ('A', 'B', 'CNN'):
            for seed in (17, 42, 101):
                training.Trainer(train, cfg, variant, seed, runs/f'{variant}-{seed}', 'cpu').run(
                    validation=validation, install_signals=False)
        ridge_module.run_ridge(train, validation, ridge)
    return runs, ridge, experiment_root, tmp_path/'frozen.json'


def forbid_test(monkeypatch):
    def forbidden(*a, **kw):
        pytest.fail('Se abrio el test antes de rechazar candidatos incompatibles')
    monkeypatch.setattr(final, 'ExperimentData', forbidden)


def test_probe_b8_freeze_rejects_homogeneous_foreign_environment(experiment_root, small_config, tmp_path):
    """Las nueve corridas y ridge comparten un entorno X distinto del actual."""
    other = deepcopy(training.environment()); other['packages']['numpy'] = 'otra-version'
    args = build(experiment_root, small_config, tmp_path, env=other)
    with pytest.raises(ValueError, match='Current environment'):
        final.freeze(*args)
    assert not args[3].exists()


def test_probe_b2_final_rejects_neural_environment_even_if_ridge_matches(experiment_root, small_config, tmp_path, monkeypatch):
    """Entorno actual != entorno neural, pero ridge.json coincide con el actual."""
    args = build(experiment_root, small_config, tmp_path)
    spec = final.freeze(*args)
    changed = deepcopy(training.environment()); changed['packages']['torch'] = 'otra-version'
    monkeypatch.setattr(final, 'environment', lambda: changed)
    path = args[1]/'ridge.json'
    report = read_json(path); report['identity']['environment'] = changed; write_json(path, report)
    spec['ridge']['manifest_file'] = file_record(path)
    write_json(args[3], spec)
    forbid_test(monkeypatch)
    output = tmp_path/'final-output'
    with pytest.raises(ValueError, match='Current environment differs'):
        final.evaluate_final(args[3], output, TEST_PHRASE, device='cpu')
    assert not output.exists() and not (args[2]/'test-opening.json').exists()


def test_probe_b3_final_rejects_candidate_with_different_recipe(experiment_root, small_config, tmp_path, monkeypatch):
    """Un candidato con otra receta, coherente entre frozen.json y su checkpoint."""
    args = build(experiment_root, small_config, tmp_path)
    spec = final.freeze(*args)
    row = spec['neural'][3]
    state = torch.load(row['checkpoint'], map_location='cpu', weights_only=True)
    state['identity']['config']['learning_rate'] = .123
    torch.save(state, row['checkpoint'])
    row['run_identity']['config']['learning_rate'] = .123
    row['file'] = file_record(row['checkpoint'])
    write_json(args[3], spec)
    forbid_test(monkeypatch)
    output = tmp_path/'final-output'
    with pytest.raises(ValueError, match='differ in recipe'):
        final.evaluate_final(args[3], output, TEST_PHRASE, device='cpu')
    assert not output.exists() and not (args[2]/'test-opening.json').exists()
