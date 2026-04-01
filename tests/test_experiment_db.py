"""
Tests for ExperimentDB — SQLite experiment registry.

Run with: uv run pytest tests/test_experiment_db.py -v
"""

import importlib.util
import os
import sys
import tempfile
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def _load_module_directly(name: str, file_path: str):
    """Load a Python module directly from file, bypassing package __init__.py."""
    spec = importlib.util.spec_from_file_location(name, file_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# Load dataclasses first (no dependencies), then experiment_db
_dc = _load_module_directly(
    'src.results.dataclasses',
    str(PROJECT_ROOT / 'src' / 'results' / 'dataclasses.py'),
)
TrainingResult = _dc.TrainingResult
ComparisonResult = _dc.ComparisonResult

_db = _load_module_directly(
    'src.results.experiment_db',
    str(PROJECT_ROOT / 'src' / 'results' / 'experiment_db.py'),
)
ExperimentDB = _db.ExperimentDB


@pytest.fixture
def db(tmp_path):
    """Create a fresh in-memory-style DB in a temp directory."""
    db_path = str(tmp_path / "test_experiments.db")
    database = ExperimentDB(db_path=db_path)
    yield database
    database.close()


def _make_result(subject_id: str, model_type: str, acc: float) -> TrainingResult:
    """Helper to create a TrainingResult."""
    return TrainingResult(
        subject_id=subject_id,
        task_type='binary',
        model_type=model_type,
        best_val_acc=acc - 0.05,
        test_acc=acc,
        test_acc_majority=acc,
        epochs_trained=20,
        training_time=60.0,
    )


class TestCreateRun:
    def test_basic_create(self, db):
        run_id = db.create_run('20260221_1319', 'within_subject', 'imagery', 'binary')
        assert run_id is not None
        assert '20260221_1319' in run_id
        assert 'within_subject' in run_id

    def test_create_with_channels(self, db):
        run_id = db.create_run(
            '20260221_1319', 'cross_subject', 'imagery', 'binary',
            n_channels=32, channel_config='fdr',
        )
        assert '32ch' in run_id
        assert 'fdr' in run_id

    def test_duplicate_run_id_raises(self, db):
        db.create_run('20260221_1319', 'within_subject', 'imagery', 'binary')
        with pytest.raises(Exception):
            db.create_run('20260221_1319', 'within_subject', 'imagery', 'binary')

    def test_create_with_all_fields(self, db):
        run_id = db.create_run(
            '20260221_1319', 'transfer', 'movement', 'ternary',
            n_channels=8, channel_config='motor_cortex',
            n_subjects=21, wandb_group='transfer_20260221',
            notes='Testing freeze backbone',
        )
        run = db.get_run(run_id)
        assert run is not None
        assert run['paradigm'] == 'movement'
        assert run['task'] == 'ternary'
        assert run['n_channels'] == 8
        assert run['n_subjects'] == 21
        assert run['wandb_group'] == 'transfer_20260221'
        assert run['notes'] == 'Testing freeze backbone'
        assert run['is_complete'] == 0


class TestSubjectResults:
    def test_save_and_get(self, db):
        run_id = db.create_run('20260221_1319', 'within_subject', 'imagery', 'binary')
        result = _make_result('S01', 'eegnet', 0.85)
        db.save_subject_result(run_id, result, wandb_run_id='abc123')

        results = db.get_results(run_id)
        assert len(results) == 1
        assert results[0].subject_id == 'S01'
        assert results[0].model_type == 'eegnet'
        assert results[0].test_acc_majority == 0.85

    def test_upsert(self, db):
        run_id = db.create_run('20260221_1319', 'within_subject', 'imagery', 'binary')

        result_v1 = _make_result('S01', 'eegnet', 0.80)
        db.save_subject_result(run_id, result_v1)

        result_v2 = _make_result('S01', 'eegnet', 0.85)
        db.save_subject_result(run_id, result_v2)

        results = db.get_results(run_id)
        assert len(results) == 1
        assert results[0].test_acc_majority == 0.85

    def test_batch_save(self, db):
        run_id = db.create_run('20260221_1319', 'within_subject', 'imagery', 'binary')
        results = [
            _make_result('S01', 'eegnet', 0.85),
            _make_result('S02', 'eegnet', 0.90),
            _make_result('S01', 'cbramod', 0.82),
            _make_result('S02', 'cbramod', 0.88),
        ]
        db.save_subject_results_batch(run_id, results)

        all_results = db.get_results(run_id)
        assert len(all_results) == 4

    def test_filter_by_model(self, db):
        run_id = db.create_run('20260221_1319', 'within_subject', 'imagery', 'binary')
        db.save_subject_result(run_id, _make_result('S01', 'eegnet', 0.85))
        db.save_subject_result(run_id, _make_result('S01', 'cbramod', 0.82))

        eegnet_results = db.get_results(run_id, model_type='eegnet')
        assert len(eegnet_results) == 1
        assert eegnet_results[0].model_type == 'eegnet'

    def test_get_results_by_model(self, db):
        run_id = db.create_run('20260221_1319', 'within_subject', 'imagery', 'binary')
        db.save_subject_results_batch(run_id, [
            _make_result('S01', 'eegnet', 0.85),
            _make_result('S02', 'eegnet', 0.90),
            _make_result('S01', 'cbramod', 0.82),
        ])

        grouped = db.get_results_by_model(run_id)
        assert 'eegnet' in grouped
        assert 'cbramod' in grouped
        assert len(grouped['eegnet']) == 2
        assert len(grouped['cbramod']) == 1


class TestSummaryAndComparison:
    def test_save_summary(self, db):
        run_id = db.create_run('20260221_1319', 'within_subject', 'imagery', 'binary')
        stats = {'mean': 0.85, 'std': 0.05, 'median': 0.86, 'min': 0.78, 'max': 0.95, 'n_subjects': 21}
        db.save_summary(run_id, 'eegnet', stats)

        summaries = db.get_summary(run_id, 'eegnet')
        assert len(summaries) == 1
        assert summaries[0]['mean_acc'] == 0.85
        assert summaries[0]['n_subjects'] == 21

    def test_save_comparison(self, db):
        run_id = db.create_run('20260221_1319', 'within_subject', 'imagery', 'binary')
        db.save_summary(run_id, 'eegnet', {'mean': 0.82, 'std': 0.05, 'median': 0.83, 'min': 0.7, 'max': 0.95, 'n_subjects': 21})
        db.save_summary(run_id, 'cbramod', {'mean': 0.85, 'std': 0.04, 'median': 0.86, 'min': 0.75, 'max': 0.96, 'n_subjects': 21})

        comparison = ComparisonResult(
            n_subjects=21,
            eegnet_mean=0.82, eegnet_std=0.05, eegnet_median=0.83,
            cbramod_mean=0.85, cbramod_std=0.04, cbramod_median=0.86,
            difference_mean=0.03, difference_std=0.02,
            paired_ttest_t=2.5, paired_ttest_p=0.021,
            wilcoxon_stat=50.0, wilcoxon_p=0.018,
            better_model='cbramod', significant=True,
        )
        db.save_comparison(run_id, comparison)

        loaded = db.get_comparison(run_id)
        assert loaded is not None
        assert loaded.better_model == 'cbramod'
        assert loaded.significant is True
        assert loaded.paired_ttest_p == pytest.approx(0.021)


class TestTransferConfig:
    def test_save_transfer_config(self, db):
        run_id = db.create_run('20260221_1319', 'transfer', 'imagery', 'binary')
        db.save_transfer_config(
            run_id,
            freeze_strategy='backbone',
            pretrained_eegnet='checkpoints/eegnet/best.pt',
            pretrained_cbramod='checkpoints/cbramod/best.pt',
            classifier_type='two_layer',
        )
        run = db.get_run(run_id)
        assert run is not None


class TestQueryOperations:
    def _populate(self, db):
        """Populate DB with multiple runs for querying."""
        # Within-subject 128ch
        r1 = db.create_run('20260220_1000', 'within_subject', 'imagery', 'binary')
        db.save_subject_results_batch(r1, [
            _make_result('S01', 'eegnet', 0.80),
            _make_result('S01', 'cbramod', 0.82),
        ])
        db.save_summary(r1, 'eegnet', {'mean': 0.80, 'std': 0.05, 'median': 0.80, 'min': 0.80, 'max': 0.80, 'n_subjects': 1})
        db.save_summary(r1, 'cbramod', {'mean': 0.82, 'std': 0.04, 'median': 0.82, 'min': 0.82, 'max': 0.82, 'n_subjects': 1})
        db.mark_complete(r1)

        # Within-subject 128ch (newer, better)
        r2 = db.create_run('20260221_1000', 'within_subject', 'imagery', 'binary')
        db.save_subject_results_batch(r2, [
            _make_result('S01', 'eegnet', 0.85),
            _make_result('S01', 'cbramod', 0.88),
        ])
        db.save_summary(r2, 'eegnet', {'mean': 0.85, 'std': 0.03, 'median': 0.85, 'min': 0.85, 'max': 0.85, 'n_subjects': 1})
        db.save_summary(r2, 'cbramod', {'mean': 0.88, 'std': 0.02, 'median': 0.88, 'min': 0.88, 'max': 0.88, 'n_subjects': 1})
        db.mark_complete(r2)

        # Cross-subject 32ch FDR
        r3 = db.create_run('20260222_1000', 'cross_subject', 'imagery', 'binary',
                           n_channels=32, channel_config='fdr')
        db.save_subject_result(r3, _make_result('S01', 'cbramod', 0.75))
        db.save_summary(r3, 'cbramod', {'mean': 0.75, 'std': 0.10, 'median': 0.75, 'min': 0.75, 'max': 0.75, 'n_subjects': 1})
        db.mark_complete(r3)

        # Transfer (incomplete)
        r4 = db.create_run('20260223_1000', 'transfer', 'imagery', 'binary')
        db.save_subject_result(r4, _make_result('S01', 'eegnet', 0.78))
        # Not marked complete

        return r1, r2, r3, r4

    def test_find_runs_all(self, db):
        self._populate(db)
        runs = db.find_runs()
        assert len(runs) == 4

    def test_find_runs_by_experiment_type(self, db):
        self._populate(db)
        runs = db.find_runs(experiment_type='within_subject')
        assert len(runs) == 2

    def test_find_runs_by_channels(self, db):
        self._populate(db)
        runs = db.find_runs(n_channels=32)
        assert len(runs) == 1
        assert runs[0]['channel_config'] == 'fdr'

    def test_find_runs_complete_only(self, db):
        self._populate(db)
        runs = db.find_runs(is_complete=True)
        assert len(runs) == 3

    def test_find_latest_run(self, db):
        self._populate(db)
        latest = db.find_latest_run('imagery', 'binary', 'within_subject')
        assert latest is not None
        assert latest['run_tag'] == '20260221_1000'

    def test_find_run_by_tag(self, db):
        self._populate(db)
        run = db.find_run_by_tag('20260222')
        assert run is not None
        assert run['experiment_type'] == 'cross_subject'

    def test_get_best_run(self, db):
        self._populate(db)
        best = db.get_best_run('imagery', 'binary', 'cbramod', 'within_subject')
        assert best is not None
        assert best['run_tag'] == '20260221_1000'
        assert best['best_mean_acc'] == pytest.approx(0.88)

    def test_find_runs_with_limit(self, db):
        self._populate(db)
        runs = db.find_runs(limit=2)
        assert len(runs) == 2


class TestResume:
    def test_get_incomplete_run(self, db):
        db.create_run('20260221_1319', 'within_subject', 'imagery', 'binary')
        incomplete = db.get_incomplete_run('imagery', 'binary', 'within_subject')
        assert incomplete is not None
        assert incomplete['run_tag'] == '20260221_1319'

    def test_no_incomplete_after_complete(self, db):
        run_id = db.create_run('20260221_1319', 'within_subject', 'imagery', 'binary')
        db.mark_complete(run_id)
        incomplete = db.get_incomplete_run('imagery', 'binary', 'within_subject')
        assert incomplete is None

    def test_get_completed_subjects(self, db):
        run_id = db.create_run('20260221_1319', 'within_subject', 'imagery', 'binary')
        db.save_subject_result(run_id, _make_result('S01', 'eegnet', 0.85))
        db.save_subject_result(run_id, _make_result('S02', 'eegnet', 0.90))

        completed = db.get_completed_subjects(run_id, 'eegnet')
        assert set(completed) == {'S01', 'S02'}

    def test_completed_subjects_filter_model(self, db):
        run_id = db.create_run('20260221_1319', 'within_subject', 'imagery', 'binary')
        db.save_subject_result(run_id, _make_result('S01', 'eegnet', 0.85))
        db.save_subject_result(run_id, _make_result('S01', 'cbramod', 0.82))
        db.save_subject_result(run_id, _make_result('S02', 'eegnet', 0.90))

        eegnet_done = db.get_completed_subjects(run_id, 'eegnet')
        cbramod_done = db.get_completed_subjects(run_id, 'cbramod')
        assert set(eegnet_done) == {'S01', 'S02'}
        assert set(cbramod_done) == {'S01'}


class TestCrossExperimentQueries:
    def test_subject_history(self, db):
        r1 = db.create_run('20260220_1000', 'within_subject', 'imagery', 'binary')
        db.save_subject_result(r1, _make_result('S01', 'eegnet', 0.80))
        db.mark_complete(r1)

        r2 = db.create_run('20260221_1000', 'cross_subject', 'imagery', 'binary')
        db.save_subject_result(r2, _make_result('S01', 'eegnet', 0.75))
        db.mark_complete(r2)

        history = db.get_subject_history('S01')
        assert len(history) == 2
        # Newest first
        assert history[0]['experiment_type'] == 'cross_subject'
        assert history[1]['experiment_type'] == 'within_subject'

    def test_subject_history_filter(self, db):
        r1 = db.create_run('20260220_1000', 'within_subject', 'imagery', 'binary')
        db.save_subject_result(r1, _make_result('S01', 'eegnet', 0.80))
        db.save_subject_result(r1, _make_result('S01', 'cbramod', 0.82))
        db.mark_complete(r1)

        history = db.get_subject_history('S01', model_type='eegnet')
        assert len(history) == 1
        assert history[0]['model_type'] == 'eegnet'


class TestHighLevelQueries:
    """Tests for the high-level query helpers (Phase 3A)."""

    def _setup_within_subject_runs(self, db):
        """Create two completed within-subject runs with both models."""
        # Run 1: lower accuracy
        r1 = db.create_run('20260210_1000', 'within_subject', 'imagery', 'binary')
        for sid in ['S01', 'S02', 'S03']:
            db.save_subject_result(r1, _make_result(sid, 'eegnet', 0.70))
            db.save_subject_result(r1, _make_result(sid, 'cbramod', 0.72))
        db.save_summary(r1, 'eegnet', {'mean': 0.70, 'std': 0.01, 'median': 0.70, 'min': 0.69, 'max': 0.71, 'n_subjects': 3})
        db.save_summary(r1, 'cbramod', {'mean': 0.72, 'std': 0.01, 'median': 0.72, 'min': 0.71, 'max': 0.73, 'n_subjects': 3})
        db.mark_complete(r1)

        # Run 2: higher accuracy
        r2 = db.create_run('20260220_1000', 'within_subject', 'imagery', 'binary')
        for sid in ['S01', 'S02', 'S03']:
            db.save_subject_result(r2, _make_result(sid, 'eegnet', 0.85))
            db.save_subject_result(r2, _make_result(sid, 'cbramod', 0.88))
        db.save_summary(r2, 'eegnet', {'mean': 0.85, 'std': 0.01, 'median': 0.85, 'min': 0.84, 'max': 0.86, 'n_subjects': 3})
        db.save_summary(r2, 'cbramod', {'mean': 0.88, 'std': 0.01, 'median': 0.88, 'min': 0.87, 'max': 0.89, 'n_subjects': 3})
        db.mark_complete(r2)
        return r1, r2

    def test_find_best_within_subject_results(self, db):
        r1, r2 = self._setup_within_subject_runs(db)
        results = db.find_best_within_subject_results('imagery', 'binary', 'eegnet')
        assert results is not None
        assert len(results) == 3
        # Should pick run 2 (higher mean_acc)
        assert all(r.test_acc_majority == 0.85 for r in results)

    def test_find_baseline_within_subject_results_prefers_explicit_baseline(self, db):
        r1, r2 = self._setup_within_subject_runs(db)
        db.set_baseline(r1, 'eegnet', True)

        baseline_results = db.find_baseline_within_subject_results('imagery', 'binary', 'eegnet')
        assert baseline_results is not None
        assert all(r.test_acc_majority == 0.70 for r in baseline_results)

        best_results = db.find_best_within_subject_results('imagery', 'binary', 'eegnet')
        assert best_results is not None
        assert all(r.test_acc_majority == 0.85 for r in best_results)

    def test_find_best_within_subject_results_with_subjects(self, db):
        r1, r2 = self._setup_within_subject_runs(db)
        # Filter to S01 and S02 only
        results = db.find_best_within_subject_results(
            'imagery', 'binary', 'cbramod', subjects={'S01', 'S02'}
        )
        assert results is not None
        assert len(results) == 2
        assert {r.subject_id for r in results} == {'S01', 'S02'}

    def test_find_best_within_subject_results_none(self, db):
        # No data → None
        results = db.find_best_within_subject_results('imagery', 'binary', 'eegnet')
        assert results is None

    def test_find_best_within_subject_results_exclude_run(self, db):
        r1, r2 = self._setup_within_subject_runs(db)
        # Exclude the better run → should return run 1
        results = db.find_best_within_subject_results(
            'imagery', 'binary', 'eegnet', exclude_run_id=r2
        )
        assert results is not None
        assert all(r.test_acc_majority == 0.70 for r in results)

    def test_find_historical_comparison(self, db):
        r1, r2 = self._setup_within_subject_runs(db)
        grouped = db.find_historical_comparison('imagery', 'binary')
        assert grouped is not None
        assert 'eegnet' in grouped
        assert 'cbramod' in grouped
        # Should pick the run with highest combined accuracy (run 2)
        assert all(r.test_acc_majority == 0.85 for r in grouped['eegnet'])
        assert all(r.test_acc_majority == 0.88 for r in grouped['cbramod'])

    def test_find_historical_comparison_with_subject_filter(self, db):
        r1, r2 = self._setup_within_subject_runs(db)
        grouped = db.find_historical_comparison(
            'imagery', 'binary', subjects={'S01', 'S03'}
        )
        assert grouped is not None
        assert len(grouped['eegnet']) == 2
        assert {r.subject_id for r in grouped['eegnet']} == {'S01', 'S03'}

    def test_find_historical_comparison_exclude_run(self, db):
        r1, r2 = self._setup_within_subject_runs(db)
        grouped = db.find_historical_comparison(
            'imagery', 'binary', exclude_run_id=r2
        )
        assert grouped is not None
        # Should fall back to run 1
        assert all(r.test_acc_majority == 0.70 for r in grouped['eegnet'])

    def test_find_historical_comparison_none_single_model(self, db):
        # Only eegnet, no cbramod → should return None
        r = db.create_run('20260220_1000', 'within_subject', 'imagery', 'binary')
        db.save_subject_result(r, _make_result('S01', 'eegnet', 0.80))
        db.save_summary(r, 'eegnet', {'mean': 0.80, 'std': 0, 'median': 0.80, 'min': 0.80, 'max': 0.80, 'n_subjects': 1})
        db.mark_complete(r)
        assert db.find_historical_comparison('imagery', 'binary') is None

    def test_find_best_cross_subject_results(self, db):
        r = db.create_run('20260215_1000', 'cross_subject', 'imagery', 'binary')
        for sid in ['S01', 'S02', 'S03']:
            db.save_subject_result(r, _make_result(sid, 'cbramod', 0.75))
        db.save_summary(r, 'cbramod', {'mean': 0.75, 'std': 0.01, 'median': 0.75, 'min': 0.74, 'max': 0.76, 'n_subjects': 3})
        db.mark_complete(r)

        results = db.find_best_cross_subject_results('imagery', 'binary', 'cbramod')
        assert results is not None
        assert len(results) == 3
        assert all(r.model_type == 'cbramod' for r in results)

    def test_find_baseline_cross_subject_results_prefers_explicit_baseline(self, db):
        r1 = db.create_run('20260215_1000', 'cross_subject', 'imagery', 'binary')
        for sid in ['S01', 'S02', 'S03']:
            db.save_subject_result(r1, _make_result(sid, 'cbramod', 0.75))
        db.save_summary(r1, 'cbramod', {'mean': 0.75, 'std': 0.01, 'median': 0.75, 'min': 0.74, 'max': 0.76, 'n_subjects': 3})
        db.mark_complete(r1)

        r2 = db.create_run('20260216_1000', 'cross_subject', 'imagery', 'binary')
        for sid in ['S01', 'S02', 'S03']:
            db.save_subject_result(r2, _make_result(sid, 'cbramod', 0.82))
        db.save_summary(r2, 'cbramod', {'mean': 0.82, 'std': 0.01, 'median': 0.82, 'min': 0.81, 'max': 0.83, 'n_subjects': 3})
        db.mark_complete(r2)

        db.set_baseline(r1, 'cbramod', True)

        baseline_results = db.find_baseline_cross_subject_results('imagery', 'binary', 'cbramod')
        assert baseline_results is not None
        assert all(r.test_acc_majority == 0.75 for r in baseline_results)

        best_results = db.find_best_cross_subject_results('imagery', 'binary', 'cbramod')
        assert best_results is not None
        assert all(r.test_acc_majority == 0.82 for r in best_results)

    def test_find_best_cross_subject_results_subject_filter(self, db):
        r = db.create_run('20260215_1000', 'cross_subject', 'imagery', 'binary')
        for sid in ['S01', 'S02', 'S03']:
            db.save_subject_result(r, _make_result(sid, 'cbramod', 0.75))
        db.save_summary(r, 'cbramod', {'mean': 0.75, 'std': 0.01, 'median': 0.75, 'min': 0.74, 'max': 0.76, 'n_subjects': 3})
        db.mark_complete(r)

        # Request S04 which doesn't exist → None
        results = db.find_best_cross_subject_results(
            'imagery', 'binary', 'cbramod', subjects={'S01', 'S04'}
        )
        assert results is None

    def test_find_best_cross_subject_results_none(self, db):
        assert db.find_best_cross_subject_results('imagery', 'binary', 'eegnet') is None


class TestHousekeeping:
    def test_delete_run(self, db):
        run_id = db.create_run('20260221_1319', 'within_subject', 'imagery', 'binary')
        db.save_subject_result(run_id, _make_result('S01', 'eegnet', 0.85))
        db.delete_run(run_id)

        assert db.get_run(run_id) is None
        assert db.get_results(run_id) == []

    def test_run_exists(self, db):
        run_id = db.create_run('20260221_1319', 'within_subject', 'imagery', 'binary')
        assert db.run_exists(run_id) is True
        assert db.run_exists('nonexistent') is False

    def test_repr(self, db):
        db.create_run('20260221_1319', 'within_subject', 'imagery', 'binary')
        rep = repr(db)
        assert 'runs=1' in rep
        assert 'results=0' in rep

    def test_count_runs(self, db):
        db.create_run('20260220_1000', 'within_subject', 'imagery', 'binary')
        db.create_run('20260221_1000', 'cross_subject', 'imagery', 'binary')
        assert db.count_runs() == 2
        assert db.count_runs(experiment_type='within_subject') == 1
