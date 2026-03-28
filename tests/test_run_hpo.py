"""
Tests for scripts/run_hpo.py study inspection mode.

Run with:
  uv run pytest tests/test_run_hpo.py -v
"""

import importlib.util
import sys
from pathlib import Path

import optuna
import pytest
from optuna.distributions import CategoricalDistribution, FloatDistribution
from optuna.trial import TrialState, create_trial

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def _load_module_directly(name: str, file_path: str):
    """Load a Python module directly from file, bypassing package __init__.py."""
    spec = importlib.util.spec_from_file_location(name, file_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


run_hpo = _load_module_directly(
    'scripts.run_hpo',
    str(PROJECT_ROOT / 'scripts' / 'run_hpo.py'),
)


@pytest.fixture
def sample_study(tmp_path):
    """Create a small SQLite-backed study with mixed trial states."""
    storage = f"sqlite:///{tmp_path / 'test_hpo.db'}"
    study = optuna.create_study(
        study_name='eegnet_within_subject_binary',
        storage=storage,
        direction='maximize',
    )
    dists = {
        'learning_rate': FloatDistribution(1e-4, 1e-2, log=True),
        'batch_size': CategoricalDistribution([32, 64, 128]),
    }

    study.add_trial(create_trial(
        state=TrialState.COMPLETE,
        value=0.81,
        params={'learning_rate': 1e-3, 'batch_size': 64},
        distributions=dists,
        intermediate_values={0: 0.52, 1: 0.81},
    ))
    study.add_trial(create_trial(
        state=TrialState.COMPLETE,
        value=0.74,
        params={'learning_rate': 2e-3, 'batch_size': 32},
        distributions=dists,
        intermediate_values={0: 0.48, 1: 0.74},
    ))
    study.add_trial(create_trial(
        state=TrialState.COMPLETE,
        value=0.78,
        params={'learning_rate': 5e-4, 'batch_size': 128},
        distributions=dists,
        intermediate_values={0: 0.50, 1: 0.78},
    ))
    study.add_trial(create_trial(
        state=TrialState.PRUNED,
        value=0.56,
        params={'learning_rate': 3e-3, 'batch_size': 64},
        distributions=dists,
        intermediate_values={0: 0.56},
    ))
    study.add_trial(create_trial(
        state=TrialState.FAIL,
        params={'learning_rate': 8e-4, 'batch_size': 32},
        distributions=dists,
    ))

    running = study.ask()
    running.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
    running.suggest_categorical('batch_size', [32, 64, 128])
    study.enqueue_trial({'learning_rate': 4e-4, 'batch_size': 64})

    return study, storage


def test_collect_study_report_counts_and_top_trials(sample_study):
    study, _ = sample_study
    report = run_hpo.collect_study_report(study, top_k=3)

    assert report['study_name'] == 'eegnet_within_subject_binary'
    assert report['n_trials'] == 7
    assert report['aggregate_counts'] == {
        'complete': 3,
        'incomplete': 2,
        'pruned': 1,
        'aborted': 1,
    }
    assert report['raw_counts'] == {
        'COMPLETE': 3,
        'RUNNING': 1,
        'WAITING': 1,
        'PRUNED': 1,
        'FAIL': 1,
    }
    assert report['category']['subject_count_estimate'] == 2
    assert report['speed_stats']['per_trial_duration_seconds']['count'] == 5
    assert report['speed_stats']['estimated_seconds_per_epoch']['count'] == 4
    assert [trial['final_value'] for trial in report['top_trials']] == [0.81, 0.78, 0.74]
    assert report['top_trials'][0]['params']['batch_size'] == 64


def test_render_study_report_contains_requested_sections(sample_study):
    study, storage = sample_study
    report = run_hpo.collect_study_report(study, top_k=3)
    rendered = run_hpo.render_study_report(
        report,
        model='eegnet',
        paradigm='within_subject',
        task='binary',
        storage_url=storage,
        use_color=False,
    )

    assert 'HPO Study Dashboard | eegnet_within_subject_binary' in rendered
    assert 'State Mix' in rendered
    assert 'summary    complete=3  incomplete=2  pruned=1  failed=1' in rendered
    assert 'Speed' in rendered
    assert 'per-trial duration' in rendered
    assert 'per-epoch speed (estimated)' in rendered
    assert 'Top Trials' in rendered
    assert '#1 trial #0  score 0.8100' in rendered
    assert 'per_epoch=' in rendered
    assert 'Trial Ledger' in rendered
    assert '#005     [RUNNING ]' in rendered
    assert '#006     [WAITING ]' in rendered


def test_main_inspect_study_does_not_start_optimization(
    sample_study,
    monkeypatch,
    capsys,
):
    _, storage = sample_study

    def _unexpected_create_study(*args, **kwargs):
        raise AssertionError('inspect mode should not create or resume a study')

    monkeypatch.setattr(run_hpo.optuna, 'create_study', _unexpected_create_study)
    monkeypatch.setattr(
        sys,
        'argv',
        [
            'run_hpo.py',
            '--paradigm', 'within_subject',
            '--model', 'eegnet',
            '--task', 'binary',
            '--storage', storage,
            '--inspect-study',
        ],
    )

    run_hpo.main()
    output = capsys.readouterr().out

    assert 'HPO Study Dashboard | eegnet_within_subject_binary' in output
    assert 'Top Trials' in output
